from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import httpx
from pydantic import BaseModel, ValidationError


DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MAX_OUTPUT_TOKENS = 6000

ProviderName = Literal["openai", "anthropic"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
MessageRole = Literal["system", "developer", "user", "assistant"]


@dataclass(frozen=True)
class ModelMessage:
    role: MessageRole
    content: str


ModelInput = Union[str, Sequence[Union[ModelMessage, Mapping[str, Any]]]]
StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class ModelGatewayError(RuntimeError):
    """Base error exposed by every provider implementation."""


class ModelTechnicalError(ModelGatewayError):
    """A transient/provider failure for which a provider fallback is allowed."""


class ModelTransportError(ModelTechnicalError):
    """The provider could not be reached or timed out."""


class ModelRateLimitError(ModelTechnicalError):
    """The provider rejected a call due to a rate limit."""


class ModelUnavailableError(ModelTechnicalError):
    """The provider was reachable but temporarily unavailable."""


class ModelRequestError(ModelGatewayError):
    """A non-retriable request/configuration/provider response error."""


class ModelConfigurationError(ModelRequestError):
    """The selected gateway cannot be constructed from current configuration."""


class ModelSchemaError(ModelGatewayError):
    """Structured model output did not validate against the requested schema."""


@runtime_checkable
class ModelGateway(Protocol):
    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        ...

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        ...


@dataclass(frozen=True)
class ModelGatewayConfig:
    provider: ProviderName = DEFAULT_LLM_PROVIDER
    model: str = DEFAULT_OPENAI_MODEL
    fallback_provider: Optional[ProviderName] = "anthropic"
    fallback_model: Optional[str] = DEFAULT_ANTHROPIC_MODEL
    openai_api_key: Optional[str] = field(default=None, repr=False)
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    anthropic_api_url: str = DEFAULT_ANTHROPIC_API_URL
    legal_planner_model: str = DEFAULT_OPENAI_MODEL
    authority_extractor_model: str = DEFAULT_OPENAI_MODEL
    legal_synthesis_model: str = DEFAULT_OPENAI_MODEL
    answer_writer_model: str = DEFAULT_OPENAI_MODEL
    available_models: Tuple[Tuple[ProviderName, str], ...] = ()
    timeout_seconds: float = 110.0
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    transport_retries: int = 2
    schema_retries: int = 1
    retry_base_delay_seconds: float = 0.25

    def api_key_for(self, provider: ProviderName) -> Optional[str]:
        if provider == "openai":
            return self.openai_api_key
        return self.anthropic_api_key

    def model_for_stage(self, stage: str) -> str:
        normalized = stage.strip().lower().replace("-", "_")
        stage_models = {
            "planner": self.legal_planner_model,
            "legal_planner": self.legal_planner_model,
            "authority_extractor": self.authority_extractor_model,
            "authority_extraction": self.authority_extractor_model,
            "legal_synthesis": self.legal_synthesis_model,
            "synthesis": self.legal_synthesis_model,
            "answer_writer": self.answer_writer_model,
            "answer": self.answer_writer_model,
        }
        return stage_models.get(normalized, self.model)

    @property
    def evidence_analyst_model(self) -> str:
        """Public Model→RAG→Model name for the authority/evidence stage."""
        return self.authority_extractor_model


@dataclass(frozen=True)
class ConfiguredModel:
    id: str
    provider: ProviderName
    roles: Tuple[str, ...]
    is_default: bool = False


def _provider(value: Optional[str], *, default: Optional[ProviderName] = None) -> Optional[ProviderName]:
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if normalized in {"none", "off", "disabled"}:
        return None
    if normalized not in {"openai", "anthropic"}:
        raise ModelConfigurationError(f"Unsupported LLM provider: {value!r}")
    return normalized  # type: ignore[return-value]


def _default_model(provider: ProviderName) -> str:
    return DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_ANTHROPIC_MODEL


def _bounded_int(value: Optional[str], default: int, *, minimum: int, maximum: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ModelConfigurationError(f"Expected an integer, got {value!r}") from exc
    return min(maximum, max(minimum, parsed))


def _bounded_float(value: Optional[str], default: float, *, minimum: float, maximum: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ModelConfigurationError(f"Expected a number, got {value!r}") from exc
    return min(maximum, max(minimum, parsed))


def _split_model_entry(entry: str, default_provider: ProviderName) -> Tuple[ProviderName, str]:
    candidate = entry.strip()
    if not candidate:
        raise ModelConfigurationError("Configured model identifiers cannot be empty")
    prefix, separator, suffix = candidate.partition(":")
    if separator and prefix.strip().lower() in {"openai", "anthropic"}:
        provider = _provider(prefix)
        assert provider is not None
        model = suffix.strip()
        if not model:
            raise ModelConfigurationError(f"Missing model identifier in {entry!r}")
        return provider, model
    if candidate.lower().startswith("claude-"):
        return "anthropic", candidate
    if candidate.lower().startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", candidate
    return default_provider, candidate


def get_model_gateway_config(env: Optional[Mapping[str, str]] = None) -> ModelGatewayConfig:
    source: Mapping[str, str] = os.environ if env is None else env
    provider = _provider(source.get("LLM_PROVIDER"), default=DEFAULT_LLM_PROVIDER)
    assert provider is not None
    model = (source.get("LLM_MODEL") or _default_model(provider)).strip()
    if not model:
        model = _default_model(provider)

    fallback_provider = _provider(source.get("LLM_FALLBACK_PROVIDER"), default="anthropic")
    fallback_model_value = source.get("LLM_FALLBACK_MODEL")
    fallback_model = (
        (fallback_model_value or _default_model(fallback_provider)).strip()
        if fallback_provider is not None
        else None
    )

    configured_entries = []
    raw_models = source.get("LLM_MODELS", "")
    for entry in raw_models.split(","):
        if entry.strip():
            configured_entries.append(_split_model_entry(entry, provider))

    stage_default = model
    return ModelGatewayConfig(
        provider=provider,
        model=model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        openai_api_key=(source.get("OPENAI_API_KEY") or "").strip() or None,
        anthropic_api_key=(source.get("ANTHROPIC_API_KEY") or "").strip() or None,
        anthropic_api_url=(
            source.get("ANTHROPIC_API_URL") or DEFAULT_ANTHROPIC_API_URL
        ).strip(),
        legal_planner_model=(source.get("LEGAL_PLANNER_MODEL") or stage_default).strip(),
        authority_extractor_model=(
            source.get("EVIDENCE_ANALYST_MODEL")
            or source.get("AUTHORITY_EXTRACTOR_MODEL")
            or stage_default
        ).strip(),
        legal_synthesis_model=(source.get("LEGAL_SYNTHESIS_MODEL") or stage_default).strip(),
        answer_writer_model=(source.get("ANSWER_WRITER_MODEL") or stage_default).strip(),
        available_models=tuple(configured_entries),
        timeout_seconds=_bounded_float(
            source.get("LLM_REQUEST_TIMEOUT_SECONDS")
            or source.get("LLM_TIMEOUT_SECONDS"),
            110.0,
            minimum=1.0,
            maximum=600.0,
        ),
        max_output_tokens=_bounded_int(
            source.get("LLM_MAX_OUTPUT_TOKENS"),
            DEFAULT_MAX_OUTPUT_TOKENS,
            minimum=1,
            maximum=128_000,
        ),
        transport_retries=_bounded_int(
            source.get("LLM_MAX_RETRIES") or source.get("LLM_TRANSPORT_RETRIES"),
            2,
            minimum=0,
            maximum=8,
        ),
        schema_retries=_bounded_int(
            source.get("LLM_SCHEMA_RETRIES"), 1, minimum=0, maximum=4
        ),
        retry_base_delay_seconds=_bounded_float(
            source.get("LLM_RETRY_BASE_DELAY_SECONDS"),
            0.25,
            minimum=0.0,
            maximum=10.0,
        ),
    )


def provider_for_model(model: str, config: Optional[ModelGatewayConfig] = None) -> ProviderName:
    selected = config or get_model_gateway_config()
    raw = model.strip()
    for provider, configured_model in selected.available_models:
        if raw == configured_model or raw == f"{provider}:{configured_model}":
            return provider
    if raw == selected.model:
        return selected.provider
    if selected.fallback_provider and raw == selected.fallback_model:
        return selected.fallback_provider
    return _split_model_entry(raw, selected.provider)[0]


def _plain_model_id(model: str) -> str:
    prefix, separator, suffix = model.strip().partition(":")
    if separator and prefix.lower() in {"openai", "anthropic"}:
        return suffix.strip()
    return model.strip()


def is_model_gateway_configured(
    config: Optional[ModelGatewayConfig] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    selected = config or get_model_gateway_config(env)
    if selected.api_key_for(selected.provider):
        return True
    return bool(
        selected.fallback_provider
        and selected.fallback_model
        and selected.api_key_for(selected.fallback_provider)
    )


def configured_models(
    config: Optional[ModelGatewayConfig] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[ConfiguredModel, ...]:
    selected = config or get_model_gateway_config(env)
    ordered: list[Tuple[ProviderName, str, str]] = [
        (selected.provider, selected.model, "default"),
        (provider_for_model(selected.legal_planner_model, selected), selected.legal_planner_model, "planner"),
        (
            provider_for_model(selected.authority_extractor_model, selected),
            selected.authority_extractor_model,
            "authority_extractor",
        ),
        (
            provider_for_model(selected.legal_synthesis_model, selected),
            selected.legal_synthesis_model,
            "legal_synthesis",
        ),
        (
            provider_for_model(selected.answer_writer_model, selected),
            selected.answer_writer_model,
            "answer_writer",
        ),
    ]
    ordered.extend((provider, model, "available") for provider, model in selected.available_models)
    if selected.fallback_provider and selected.fallback_model:
        ordered.append((selected.fallback_provider, selected.fallback_model, "fallback"))

    merged: dict[Tuple[ProviderName, str], list[str]] = {}
    for provider, raw_model, role in ordered:
        model = _plain_model_id(raw_model)
        if not selected.api_key_for(provider):
            continue
        roles = merged.setdefault((provider, model), [])
        if role not in roles:
            roles.append(role)
    return tuple(
        ConfiguredModel(
            id=model,
            provider=provider,
            roles=tuple(roles),
            is_default=(provider == selected.provider and model == _plain_model_id(selected.model)),
        )
        for (provider, model), roles in merged.items()
    )


def configured_model_ids(
    config: Optional[ModelGatewayConfig] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    return tuple(item.id for item in configured_models(config, env=env))


# Naming aliases keep callers provider-neutral while making their purpose explicit.
get_configured_models = configured_model_ids
is_llm_configured = is_model_gateway_configured


def _normalize_input(input_value: ModelInput) -> Union[str, list[dict[str, Any]]]:
    if isinstance(input_value, str):
        if not input_value.strip():
            raise ModelRequestError("Model input cannot be empty")
        return input_value
    normalized: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, ModelMessage):
            normalized.append({"role": item.role, "content": item.content})
        elif isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            raise ModelRequestError(f"Unsupported model message: {type(item).__name__}")
    if not normalized:
        raise ModelRequestError("Model input cannot be empty")
    return normalized


def _validate_response_model(response_model: Type[StructuredModelT]) -> None:
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise ModelRequestError("response_model must be a Pydantic BaseModel subclass")


def _model_validate(
    response_model: Type[StructuredModelT], value: Any
) -> StructuredModelT:
    if isinstance(value, response_model):
        return value
    validator = getattr(response_model, "model_validate", None)
    if validator is not None:
        return validator(value)
    return response_model.parse_obj(value)  # type: ignore[attr-defined,no-any-return]


def _model_validate_json(
    response_model: Type[StructuredModelT], value: str
) -> StructuredModelT:
    validator = getattr(response_model, "model_validate_json", None)
    if validator is not None:
        return validator(value)
    return response_model.parse_raw(value)  # type: ignore[attr-defined,no-any-return]


def _model_json_schema(response_model: Type[BaseModel]) -> dict[str, Any]:
    schema_builder = getattr(response_model, "model_json_schema", None)
    if schema_builder is not None:
        return schema_builder()
    return response_model.schema()  # type: ignore[attr-defined,no-any-return]


def _status_code_from_exception(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _classify_provider_exception(provider: str, exc: BaseException) -> ModelGatewayError:
    if isinstance(exc, ModelGatewayError):
        return exc
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return ModelSchemaError(f"{provider} returned invalid structured output")
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
        return ModelTransportError(f"{provider} request timed out")
    if isinstance(exc, (httpx.TransportError, OSError)):
        return ModelTransportError(f"Could not connect to {provider}")

    class_name = type(exc).__name__.lower()
    if "ratelimit" in class_name:
        return ModelRateLimitError(f"{provider} rate limit exceeded")
    if "timeout" in class_name or "connection" in class_name:
        return ModelTransportError(f"Could not connect to {provider}")
    if "lengthfinishreason" in class_name:
        return ModelSchemaError(f"{provider} truncated structured output")
    if "contentfilter" in class_name:
        return ModelRequestError(f"{provider} refused the request")

    status = _status_code_from_exception(exc)
    if status == 429:
        return ModelRateLimitError(f"{provider} rate limit exceeded")
    if status in {408, 409, 425}:
        return ModelTransportError(f"{provider} temporary request failure ({status})")
    if status is not None and status >= 500:
        return ModelUnavailableError(f"{provider} unavailable ({status})")
    if status is not None:
        return ModelRequestError(f"{provider} rejected the request ({status})")
    return ModelRequestError(f"{provider} request failed: {type(exc).__name__}")


class _RetryingProviderGateway:
    provider_name: str

    def __init__(
        self,
        *,
        max_transport_retries: int,
        max_schema_retries: int,
        retry_base_delay_seconds: float,
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._max_transport_retries = max(0, max_transport_retries)
        self._max_schema_retries = max(0, max_schema_retries)
        self._retry_base_delay_seconds = max(0.0, retry_base_delay_seconds)
        self._sleep = sleep

    async def _call_with_technical_retries(self, call: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(self._max_transport_retries + 1):
            try:
                return await call()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                error = _classify_provider_exception(self.provider_name, exc)
                if not isinstance(error, ModelTechnicalError):
                    raise error from exc
                if attempt >= self._max_transport_retries:
                    raise error from exc
                await self._sleep(self._retry_base_delay_seconds * (2**attempt))
        raise AssertionError("unreachable")

    async def _schema_attempts(
        self,
        call: Callable[[], Awaitable[StructuredModelT]],
    ) -> StructuredModelT:
        last_error: Optional[ModelSchemaError] = None
        for attempt in range(self._max_schema_retries + 1):
            try:
                return await call()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                error = _classify_provider_exception(self.provider_name, exc)
                if not isinstance(error, ModelSchemaError):
                    raise error from exc
                last_error = error
                if attempt < self._max_schema_retries:
                    await self._sleep(self._retry_base_delay_seconds * (2**attempt))
        assert last_error is not None
        raise last_error


class OpenAIModelGateway(_RetryingProviderGateway):
    provider_name = "OpenAI"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        timeout_seconds: float = 110.0,
        default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_transport_retries: int = 2,
        max_schema_retries: int = 1,
        retry_base_delay_seconds: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            max_transport_retries=max_transport_retries,
            max_schema_retries=max_schema_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            sleep=sleep,
        )
        self.model = model
        self._default_max_output_tokens = default_max_output_tokens
        if client is None:
            if not api_key:
                raise ModelConfigurationError("OPENAI_API_KEY is required for the OpenAI gateway")
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ModelConfigurationError(
                    "The official openai package is required for the OpenAI gateway"
                ) from exc
            # Gateway retries are the single retry policy, so SDK retries are disabled.
            client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._client = client

    def _request_kwargs(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str],
        model: Optional[str],
        reasoning_effort: Optional[ReasoningEffort],
        max_output_tokens: Optional[int],
        temperature: Optional[float],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": _plain_model_id(model or self.model),
            "input": _normalize_input(input),
            "max_output_tokens": max_output_tokens or self._default_max_output_tokens,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = self._request_kwargs(
            input=input,
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        async def call() -> Any:
            return await self._client.responses.create(**kwargs)

        response = await self._call_with_technical_retries(call)
        try:
            text = response.output_text
        except BaseException as exc:
            raise _classify_provider_exception(self.provider_name, exc) from exc
        if not isinstance(text, str) or not text.strip():
            raise ModelUnavailableError("OpenAI returned an empty text response")
        return text.strip()

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        _validate_response_model(response_model)
        kwargs = self._request_kwargs(
            input=input,
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        kwargs["text_format"] = response_model

        async def parsed_call() -> StructuredModelT:
            async def call() -> Any:
                return await self._client.responses.parse(**kwargs)

            response = await self._call_with_technical_retries(call)
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ModelSchemaError("OpenAI returned no parsed structured output")
            try:
                return _model_validate(response_model, parsed)
            except (ValidationError, TypeError, ValueError) as exc:
                raise ModelSchemaError("OpenAI structured output failed schema validation") from exc

        return await self._schema_attempts(parsed_call)


def _anthropic_effort(effort: ReasoningEffort) -> str:
    return {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    }[effort]


class AnthropicModelGateway(_RetryingProviderGateway):
    provider_name = "Anthropic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,
        client: Any = None,
        api_url: str = DEFAULT_ANTHROPIC_API_URL,
        timeout_seconds: float = 110.0,
        default_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_transport_retries: int = 2,
        max_schema_retries: int = 1,
        retry_base_delay_seconds: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            max_transport_retries=max_transport_retries,
            max_schema_retries=max_schema_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            sleep=sleep,
        )
        if client is None and not api_key:
            raise ModelConfigurationError("ANTHROPIC_API_KEY is required for the Anthropic gateway")
        self.model = model
        self._api_key = api_key or ""
        self._client = client
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds
        self._default_max_output_tokens = default_max_output_tokens

    def _payload(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str],
        model: Optional[str],
        reasoning_effort: Optional[ReasoningEffort],
        max_output_tokens: Optional[int],
        temperature: Optional[float],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> dict[str, Any]:
        normalized = _normalize_input(input)
        if isinstance(normalized, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": normalized}]
        else:
            messages = []
            system_parts = [system_prompt] if system_prompt else []
            for message in normalized:
                role = str(message.get("role") or "user")
                if role in {"system", "developer"}:
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        system_parts.append(content)
                    continue
                messages.append({"role": role, "content": message.get("content", "")})
            system_prompt = "\n\n".join(part for part in system_parts if part)
        payload: dict[str, Any] = {
            "model": _plain_model_id(model or self.model),
            "max_tokens": max_output_tokens or self._default_max_output_tokens,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        output_config: dict[str, Any] = {}
        if reasoning_effort:
            output_config["effort"] = _anthropic_effort(reasoning_effort)
        if response_model is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _model_json_schema(response_model),
            }
        if output_config:
            payload["output_config"] = output_config
        return payload

    async def _post(self, payload: Mapping[str, Any]) -> Any:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self._client is not None:
            return await self._client.post(
                self._api_url,
                headers=headers,
                json=dict(payload),
                timeout=self._timeout_seconds,
            )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            return await client.post(self._api_url, headers=headers, json=dict(payload))

    async def _response_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        async def call() -> Any:
            response = await self._post(payload)
            status = getattr(response, "status_code", 200)
            if status >= 400:
                error = httpx.HTTPStatusError(
                    f"Anthropic returned HTTP {status}",
                    request=getattr(response, "request", httpx.Request("POST", self._api_url)),
                    response=response,
                )
                raise error
            try:
                body = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise ModelUnavailableError("Anthropic returned malformed JSON") from exc
            if not isinstance(body, Mapping):
                raise ModelUnavailableError("Anthropic returned an invalid response envelope")
            return body

        return await self._call_with_technical_retries(call)

    @staticmethod
    def _extract_text(body: Mapping[str, Any]) -> str:
        parts = []
        for item in body.get("content") or []:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload = self._payload(
            input=input,
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        body = await self._response_json(payload)
        text = self._extract_text(body)
        if not text:
            raise ModelUnavailableError("Anthropic returned an empty text response")
        return text

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        _validate_response_model(response_model)
        payload = self._payload(
            input=input,
            system_prompt=system_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_model=response_model,
        )

        async def parsed_call() -> StructuredModelT:
            body = await self._response_json(payload)
            text = self._extract_text(body)
            if not text:
                raise ModelSchemaError("Anthropic returned no structured output")
            try:
                return _model_validate_json(response_model, text)
            except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModelSchemaError("Anthropic structured output failed schema validation") from exc

        return await self._schema_attempts(parsed_call)


class FallbackModelGateway:
    """Uses the fallback strictly after a typed technical provider failure."""

    def __init__(self, primary: ModelGateway, fallback: ModelGateway) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = {
            "input": input,
            "system_prompt": system_prompt,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        try:
            return await self.primary.generate_text(**kwargs)
        except ModelTechnicalError:
            # A primary-provider model id must never leak into a different provider.
            kwargs["model"] = None
            return await self.fallback.generate_text(**kwargs)

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        kwargs = {
            "response_model": response_model,
            "input": input,
            "system_prompt": system_prompt,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        try:
            return await self.primary.generate_structured(**kwargs)
        except ModelTechnicalError:
            kwargs["model"] = None
            return await self.fallback.generate_structured(**kwargs)


class RoutingModelGateway:
    """Routes selectable model ids to the matching provider gateway."""

    def __init__(
        self,
        *,
        default_gateway: ModelGateway,
        config: ModelGatewayConfig,
        provider_gateways: Mapping[ProviderName, ModelGateway],
        model_gateways: Optional[Mapping[str, ModelGateway]] = None,
    ) -> None:
        self.default_gateway = default_gateway
        self.config = config
        self.provider_gateways = dict(provider_gateways)
        self.model_gateways = dict(model_gateways or {})

    def _route(self, model: Optional[str]) -> Tuple[ModelGateway, Optional[str]]:
        if not model:
            return self.default_gateway, None
        plain_model = _plain_model_id(model)
        if plain_model in self.model_gateways:
            return self.model_gateways[plain_model], plain_model
        provider = provider_for_model(model, self.config)
        gateway = self.provider_gateways.get(provider)
        if gateway is None:
            raise ModelConfigurationError(
                f"Model {plain_model!r} requires unconfigured provider {provider!r}"
            )
        return gateway, plain_model

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        gateway, routed_model = self._route(model)
        return await gateway.generate_text(
            input=input,
            system_prompt=system_prompt,
            model=routed_model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        gateway, routed_model = self._route(model)
        return await gateway.generate_structured(
            response_model=response_model,
            input=input,
            system_prompt=system_prompt,
            model=routed_model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )


class _UnavailableModelGateway:
    """Typed technical primary used when only the configured fallback has credentials."""

    def __init__(self, provider: ProviderName) -> None:
        self.provider = provider

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        raise ModelUnavailableError(
            f"The configured primary provider {self.provider!r} has no API credentials"
        )

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        raise ModelUnavailableError(
            f"The configured primary provider {self.provider!r} has no API credentials"
        )


class _BoundModelGateway:
    """Binds a factory-selected model while preserving router/fallback behavior."""

    def __init__(self, gateway: ModelGateway, model: str) -> None:
        self.gateway = gateway
        self.model = _plain_model_id(model)

    async def generate_text(
        self,
        *,
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        return await self.gateway.generate_text(
            input=input,
            system_prompt=system_prompt,
            model=model or self.model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    async def generate_structured(
        self,
        *,
        response_model: Type[StructuredModelT],
        input: ModelInput,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> StructuredModelT:
        return await self.gateway.generate_structured(
            response_model=response_model,
            input=input,
            system_prompt=system_prompt,
            model=model or self.model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )


def _provider_gateway(
    provider: ProviderName,
    *,
    model: str,
    config: ModelGatewayConfig,
    openai_client: Any,
    anthropic_client: Any,
    sleep: Callable[[float], Awaitable[None]],
) -> ModelGateway:
    common = {
        "model": _plain_model_id(model),
        "timeout_seconds": config.timeout_seconds,
        "default_max_output_tokens": config.max_output_tokens,
        "max_transport_retries": config.transport_retries,
        "max_schema_retries": config.schema_retries,
        "retry_base_delay_seconds": config.retry_base_delay_seconds,
        "sleep": sleep,
    }
    if provider == "openai":
        return OpenAIModelGateway(
            api_key=config.openai_api_key,
            client=openai_client,
            **common,
        )
    return AnthropicModelGateway(
        api_key=config.anthropic_api_key,
        client=anthropic_client,
        api_url=config.anthropic_api_url,
        **common,
    )


def create_model_gateway(
    config: Optional[ModelGatewayConfig] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    openai_client: Any = None,
    anthropic_client: Any = None,
    http_client: Any = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ModelGateway:
    selected = config or get_model_gateway_config(env)
    if anthropic_client is not None and http_client is not None:
        raise ModelConfigurationError("Pass only one of anthropic_client or http_client")
    anthropic_http_client = anthropic_client if anthropic_client is not None else http_client
    supplied_clients = {"openai": openai_client, "anthropic": anthropic_http_client}
    primary_available = bool(
        selected.api_key_for(selected.provider) or supplied_clients[selected.provider] is not None
    )
    fallback_provider = selected.fallback_provider
    fallback_model = selected.fallback_model
    fallback_available = bool(
        fallback_provider
        and fallback_model
        and (
            selected.api_key_for(fallback_provider)
            or supplied_clients[fallback_provider] is not None
        )
    )
    if not primary_available and not fallback_available:
        raise ModelConfigurationError(
            "No credentials are configured for either the primary or fallback LLM provider"
        )

    primary: ModelGateway
    if primary_available:
        primary = _provider_gateway(
            selected.provider,
            model=selected.model,
            config=selected,
            openai_client=openai_client,
            anthropic_client=anthropic_http_client,
            sleep=sleep,
        )
    else:
        primary = _UnavailableModelGateway(selected.provider)
    default_gateway: ModelGateway = primary
    provider_gateways: dict[ProviderName, ModelGateway] = {selected.provider: primary}
    model_gateways: dict[str, ModelGateway] = {_plain_model_id(selected.model): primary}

    if fallback_available:
        assert fallback_provider is not None
        assert fallback_model is not None
        fallback = _provider_gateway(
            fallback_provider,
            model=fallback_model,
            config=selected,
            openai_client=openai_client,
            anthropic_client=anthropic_http_client,
            sleep=sleep,
        )
        default_gateway = FallbackModelGateway(primary, fallback)
        provider_gateways[fallback_provider] = fallback
        model_gateways[_plain_model_id(fallback_model)] = fallback
        provider_gateways[selected.provider] = default_gateway
        model_gateways[_plain_model_id(selected.model)] = default_gateway

    return RoutingModelGateway(
        default_gateway=default_gateway,
        config=selected,
        provider_gateways=provider_gateways,
        model_gateways=model_gateways,
    )


def create_model_gateway_for_model(
    model: str,
    config: Optional[ModelGatewayConfig] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    openai_client: Any = None,
    anthropic_client: Any = None,
    http_client: Any = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ModelGateway:
    selected = config or get_model_gateway_config(env)
    gateway = create_model_gateway(
        selected,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        http_client=http_client,
        sleep=sleep,
    )
    # Keeping the router here retains cross-provider selection and the primary
    # gateway's technical fallback while making no-model calls use this id.
    return _BoundModelGateway(gateway, model)


__all__ = [
    "AnthropicModelGateway",
    "ConfiguredModel",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_OPENAI_MODEL",
    "FallbackModelGateway",
    "ModelConfigurationError",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayError",
    "ModelMessage",
    "ModelRateLimitError",
    "ModelRequestError",
    "ModelSchemaError",
    "ModelTechnicalError",
    "ModelTransportError",
    "ModelUnavailableError",
    "OpenAIModelGateway",
    "ReasoningEffort",
    "RoutingModelGateway",
    "configured_model_ids",
    "configured_models",
    "create_model_gateway",
    "create_model_gateway_for_model",
    "get_configured_models",
    "get_model_gateway_config",
    "is_llm_configured",
    "is_model_gateway_configured",
    "provider_for_model",
]
