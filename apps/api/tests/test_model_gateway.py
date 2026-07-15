from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel

from app.model_gateway import (
    AnthropicModelGateway,
    DEFAULT_OPENAI_MODEL,
    FallbackModelGateway,
    ModelConfigurationError,
    ModelFallbackError,
    ModelGateway,
    ModelGatewayConfig,
    ModelProviderRequestError,
    ModelRequestError,
    ModelSchemaError,
    ModelTransportError,
    OpenAIModelGateway,
    RoutingModelGateway,
    StructuredCompatibilityGateway,
    configured_model_ids,
    create_model_gateway,
    create_model_gateway_for_model,
    get_model_gateway_config,
    is_model_gateway_configured,
    provider_for_model,
)


class PlannedAnswer(BaseModel):
    title: str
    confidence: float


class FakeOpenAIResponse:
    def __init__(self, *, output_text: str = "", output_parsed: Any = None) -> None:
        self.output_text = output_text
        self.output_parsed = output_parsed


class FakeOpenAIResponses:
    def __init__(
        self,
        *,
        create_results: Optional[list[Any]] = None,
        parse_results: Optional[list[Any]] = None,
    ) -> None:
        self.create_results = list(create_results or [])
        self.parse_results = list(parse_results or [])
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    @staticmethod
    def _next(results: list[Any]) -> Any:
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._next(self.create_results)

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self._next(self.parse_results)


class FakeOpenAIClient:
    def __init__(self, responses: FakeOpenAIResponses) -> None:
        self.responses = responses


class FakeAnthropicResponse:
    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://api.anthropic.test/v1/messages")

    def json(self) -> Any:
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body


class FakeAnthropicClient:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeStatusError(Exception):
    def __init__(self, status_code: int, body: Any = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


async def no_sleep(_: float) -> None:
    return None


class StubGateway:
    def __init__(
        self,
        *,
        text: str = "ok",
        structured: Optional[BaseModel] = None,
        text_error: Optional[BaseException] = None,
        structured_error: Optional[BaseException] = None,
    ) -> None:
        self.text = text
        self.structured = structured
        self.text_error = text_error
        self.structured_error = structured_error
        self.text_calls: list[dict[str, Any]] = []
        self.structured_calls: list[dict[str, Any]] = []

    async def generate_text(self, **kwargs: Any) -> str:
        self.text_calls.append(kwargs)
        if self.text_error:
            raise self.text_error
        return self.text

    async def generate_structured(
        self, *, response_model: Type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        self.structured_calls.append({"response_model": response_model, **kwargs})
        if self.structured_error:
            raise self.structured_error
        assert self.structured is not None
        return self.structured


class OpenAIModelGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_uses_async_responses_create_and_per_call_reasoning(self) -> None:
        responses = FakeOpenAIResponses(
            create_results=[FakeOpenAIResponse(output_text="  odpowiedź  ")]
        )
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=0,
            sleep=no_sleep,
        )

        result = await gateway.generate_text(
            input="Pytanie",
            system_prompt="Zasady globalne",
            reasoning_effort="high",
            max_output_tokens=321,
            temperature=0.2,
        )

        self.assertEqual(result, "odpowiedź")
        self.assertEqual(len(responses.create_calls), 1)
        call = responses.create_calls[0]
        self.assertEqual(call["model"], DEFAULT_OPENAI_MODEL)
        self.assertEqual(call["input"], "Pytanie")
        self.assertEqual(call["instructions"], "Zasady globalne")
        self.assertEqual(call["reasoning"], {"effort": "high"})
        self.assertEqual(call["max_output_tokens"], 321)
        self.assertEqual(call["temperature"], 0.2)

    async def test_structured_uses_responses_parse_with_pydantic_model(self) -> None:
        parsed = PlannedAnswer(title="teza", confidence=0.8)
        responses = FakeOpenAIResponses(
            parse_results=[FakeOpenAIResponse(output_parsed=parsed)]
        )
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=0,
            sleep=no_sleep,
        )

        result = await gateway.generate_structured(
            response_model=PlannedAnswer,
            input="Zaplanuj analizę",
            reasoning_effort="low",
        )

        self.assertEqual(result, parsed)
        self.assertIs(responses.parse_calls[0]["text_format"], PlannedAnswer)
        self.assertEqual(responses.parse_calls[0]["reasoning"], {"effort": "low"})

    async def test_invalid_schema_is_retried_separately(self) -> None:
        responses = FakeOpenAIResponses(
            parse_results=[
                FakeOpenAIResponse(output_parsed={"title": "brak confidence"}),
                FakeOpenAIResponse(output_parsed={"title": "ok", "confidence": 0.9}),
            ]
        )
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=0,
            max_schema_retries=1,
            retry_base_delay_seconds=0,
            sleep=no_sleep,
        )

        result = await gateway.generate_structured(
            response_model=PlannedAnswer, input="Zaplanuj"
        )

        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(len(responses.parse_calls), 2)

    async def test_schema_error_after_budget_is_not_technical(self) -> None:
        responses = FakeOpenAIResponses(
            parse_results=[FakeOpenAIResponse(output_parsed={"title": "bad"})]
        )
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=0,
            max_schema_retries=0,
            sleep=no_sleep,
        )

        with self.assertRaises(ModelSchemaError):
            await gateway.generate_structured(
                response_model=PlannedAnswer, input="Zaplanuj"
            )

    async def test_transport_and_rate_errors_are_the_only_api_retries(self) -> None:
        request = httpx.Request("POST", "https://api.openai.test/v1/responses")
        responses = FakeOpenAIResponses(
            create_results=[
                httpx.ReadTimeout("timeout", request=request),
                FakeOpenAIResponse(output_text="ok"),
            ]
        )
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=1,
            retry_base_delay_seconds=0,
            sleep=no_sleep,
        )

        self.assertEqual(await gateway.generate_text(input="q"), "ok")
        self.assertEqual(len(responses.create_calls), 2)

    async def test_non_transient_400_is_not_retried(self) -> None:
        responses = FakeOpenAIResponses(create_results=[FakeStatusError(400)])
        gateway = OpenAIModelGateway(
            client=FakeOpenAIClient(responses),
            max_transport_retries=3,
            sleep=no_sleep,
        )

        with self.assertRaises(ModelProviderRequestError) as raised:
            await gateway.generate_text(input="q")
        self.assertEqual(raised.exception.category, "invalid_request")
        self.assertEqual(len(responses.create_calls), 1)


class AnthropicModelGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_uses_httpx_compatible_messages_endpoint(self) -> None:
        client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    {
                        "content": [
                            {"type": "text", "text": "pierwsza"},
                            {"type": "text", "text": "druga"},
                        ]
                    }
                )
            ]
        )
        gateway = AnthropicModelGateway(
            api_key="test-key",
            client=client,
            max_transport_retries=0,
            sleep=no_sleep,
        )

        result = await gateway.generate_text(
            input="Pytanie",
            system_prompt="System",
            reasoning_effort="medium",
            max_output_tokens=200,
        )

        self.assertEqual(result, "pierwsza\ndruga")
        payload = client.calls[0]["json"]
        self.assertEqual(payload["model"], "claude-sonnet-4-6")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Pytanie"}])
        self.assertEqual(payload["system"], "System")
        self.assertEqual(payload["max_tokens"], 200)
        self.assertEqual(payload["output_config"]["effort"], "medium")

    async def test_structured_output_sends_full_schema_and_retries_invalid_json(self) -> None:
        client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    {"content": [{"type": "text", "text": '{"title":"bad"}'}]}
                ),
                FakeAnthropicResponse(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"title":"ok","confidence":0.7}',
                            }
                        ]
                    }
                ),
            ]
        )
        gateway = AnthropicModelGateway(
            api_key="test-key",
            client=client,
            max_transport_retries=0,
            max_schema_retries=1,
            retry_base_delay_seconds=0,
            sleep=no_sleep,
        )

        result = await gateway.generate_structured(
            response_model=PlannedAnswer,
            input="Zaplanuj",
            reasoning_effort="low",
        )

        self.assertEqual(result.title, "ok")
        self.assertEqual(len(client.calls), 2)
        output_format = client.calls[0]["json"]["output_config"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertEqual(output_format["schema"]["required"], ["title", "confidence"])


class FallbackModelGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_is_used_for_technical_failure_and_drops_primary_model(self) -> None:
        primary = StubGateway(text_error=ModelTransportError("timeout"))
        fallback = StubGateway(text="fallback")
        gateway = FallbackModelGateway(primary, fallback)

        result = await gateway.generate_text(input="q", model="gpt-5.6-terra")

        self.assertEqual(result, "fallback")
        self.assertEqual(fallback.text_calls[0]["model"], None)

    async def test_fallback_is_not_used_for_schema_or_request_error(self) -> None:
        expected = PlannedAnswer(title="unused", confidence=0.1)
        for error in (ModelSchemaError("bad schema"), ModelRequestError("bad request")):
            with self.subTest(error=type(error).__name__):
                primary = StubGateway(structured_error=error)
                fallback = StubGateway(structured=expected)
                gateway = FallbackModelGateway(primary, fallback)

                with self.assertRaises(type(error)):
                    await gateway.generate_structured(
                        response_model=PlannedAnswer, input="q"
                    )
                self.assertEqual(fallback.structured_calls, [])

    async def test_fallback_is_used_for_provider_specific_request_rejection(self) -> None:
        primary = StubGateway(
            structured_error=ModelProviderRequestError(
                "OpenAI",
                status_code=404,
                category="model_unavailable",
                error_code="model_not_found",
            )
        )
        expected = PlannedAnswer(title="fallback", confidence=0.8)
        fallback = StubGateway(structured=expected)
        gateway = FallbackModelGateway(primary, fallback)

        result = await gateway.generate_structured(
            response_model=PlannedAnswer,
            input="q",
            model="gpt-5.6-terra",
        )

        self.assertEqual(result, expected)
        self.assertEqual(fallback.structured_calls[0]["model"], None)

    async def test_both_provider_failures_keep_safe_diagnostics(self) -> None:
        primary = StubGateway(
            structured_error=ModelProviderRequestError(
                "OpenAI",
                status_code=404,
                category="model_unavailable",
                error_code="model_not_found",
            )
        )
        fallback = StubGateway(
            structured_error=ModelProviderRequestError(
                "Anthropic",
                status_code=400,
                category="billing",
                error_code="invalid_request_error",
            )
        )
        gateway = FallbackModelGateway(primary, fallback)

        with self.assertRaises(ModelFallbackError) as raised:
            await gateway.generate_structured(
                response_model=PlannedAnswer,
                input="sekret123",
            )

        diagnostic = str(raised.exception)
        self.assertIn("OpenAI provider rejection", diagnostic)
        self.assertIn("Anthropic provider rejection", diagnostic)
        self.assertNotIn("sekret123", diagnostic)


class StructuredCompatibilityGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_format_rejection_falls_back_to_validated_json_text(self) -> None:
        expected = PlannedAnswer(title="json fallback", confidence=0.9)
        provider = StubGateway(
            text='Oto wynik:\n```json\n{"title":"json fallback","confidence":0.9}\n```',
            structured_error=ModelProviderRequestError(
                "OpenAI",
                status_code=400,
                category="request_format",
                error_code="invalid_json_schema",
            ),
        )
        gateway = StructuredCompatibilityGateway(provider)

        result = await gateway.generate_structured(
            response_model=PlannedAnswer,
            input="q",
            system_prompt="system",
        )

        self.assertEqual(result, expected)
        self.assertEqual(len(provider.text_calls), 1)
        self.assertIn("JSON Schema", provider.text_calls[0]["system_prompt"])

    async def test_billing_rejection_does_not_retry_as_text(self) -> None:
        provider = StubGateway(
            structured_error=ModelProviderRequestError(
                "Anthropic",
                status_code=400,
                category="billing",
                error_code="invalid_request_error",
            )
        )
        gateway = StructuredCompatibilityGateway(provider)

        with self.assertRaises(ModelProviderRequestError):
            await gateway.generate_structured(
                response_model=PlannedAnswer,
                input="q",
            )

        self.assertEqual(provider.text_calls, [])

    async def test_factory_recovers_from_openai_native_schema_rejection(self) -> None:
        responses = FakeOpenAIResponses(
            parse_results=[
                FakeStatusError(
                    400,
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_json_schema",
                            "message": "Invalid schema for response_format",
                        }
                    },
                )
            ],
            create_results=[
                FakeOpenAIResponse(
                    output_text='{"title":"recovered","confidence":0.75}'
                )
            ],
        )
        config = get_model_gateway_config(
            {
                "OPENAI_API_KEY": "openai-test",
                "LLM_TRANSPORT_RETRIES": "0",
            }
        )
        gateway = create_model_gateway(
            config,
            openai_client=FakeOpenAIClient(responses),
            sleep=no_sleep,
        )

        result = await gateway.generate_structured(
            response_model=PlannedAnswer,
            input="q",
            model=DEFAULT_OPENAI_MODEL,
        )

        self.assertEqual(result.title, "recovered")
        self.assertEqual(len(responses.parse_calls), 1)
        self.assertEqual(len(responses.create_calls), 1)


class ModelGatewayConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_neutral_defaults_and_configured_model_list(self) -> None:
        config = get_model_gateway_config(
            {
                "OPENAI_API_KEY": "openai-test",
                "ANTHROPIC_API_KEY": "anthropic-test",
                "LLM_MODELS": "gpt-5.6-terra,anthropic:claude-opus-4-8",
            }
        )

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, DEFAULT_OPENAI_MODEL)
        self.assertTrue(is_model_gateway_configured(config))
        self.assertEqual(provider_for_model("claude-opus-4-8", config), "anthropic")
        self.assertIn("gpt-5.6-terra", configured_model_ids(config))
        self.assertIn("claude-opus-4-8", configured_model_ids(config))
        self.assertIn("claude-sonnet-4-6", configured_model_ids(config))
        self.assertNotIn("openai-test", repr(config))

    def test_unconfigured_fallback_model_is_not_advertised(self) -> None:
        config = get_model_gateway_config({"OPENAI_API_KEY": "openai-test"})

        self.assertEqual(configured_model_ids(config), (DEFAULT_OPENAI_MODEL,))

    def test_current_retry_and_timeout_environment_names_are_supported(self) -> None:
        config = get_model_gateway_config(
            {
                "OPENAI_API_KEY": "openai-test",
                "LLM_REQUEST_TIMEOUT_SECONDS": "42",
                "LLM_MAX_RETRIES": "4",
            }
        )

        self.assertEqual(config.timeout_seconds, 42.0)
        self.assertEqual(config.transport_retries, 4)

    def test_factory_requires_only_primary_credentials(self) -> None:
        config = get_model_gateway_config({})
        with self.assertRaises(ModelConfigurationError):
            create_model_gateway(config)

    async def test_factory_routes_explicit_claude_model_directly_to_anthropic(self) -> None:
        openai_responses = FakeOpenAIResponses(
            create_results=[FakeOpenAIResponse(output_text="wrong provider")]
        )
        anthropic_client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    {"content": [{"type": "text", "text": "anthropic"}]}
                )
            ]
        )
        config = get_model_gateway_config(
            {
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "gpt-5.6-terra",
                "OPENAI_API_KEY": "openai-test",
                "ANTHROPIC_API_KEY": "anthropic-test",
                "LLM_MODELS": "gpt-5.6-terra,claude-opus-4-8",
                "LLM_TRANSPORT_RETRIES": "0",
            }
        )
        gateway = create_model_gateway(
            config,
            openai_client=FakeOpenAIClient(openai_responses),
            anthropic_client=anthropic_client,
            sleep=no_sleep,
        )

        self.assertIsInstance(gateway, RoutingModelGateway)
        self.assertIsInstance(gateway, ModelGateway)
        result = await gateway.generate_text(input="q", model="claude-opus-4-8")

        self.assertEqual(result, "anthropic")
        self.assertEqual(openai_responses.create_calls, [])
        self.assertEqual(anthropic_client.calls[0]["json"]["model"], "claude-opus-4-8")

    async def test_model_bound_factory_uses_selected_model_when_call_omits_it(self) -> None:
        responses = FakeOpenAIResponses(
            create_results=[FakeOpenAIResponse(output_text="selected")]
        )
        config = get_model_gateway_config({"OPENAI_API_KEY": "openai-test"})
        gateway = create_model_gateway_for_model(
            "gpt-selected",
            config,
            openai_client=FakeOpenAIClient(responses),
            sleep=no_sleep,
        )

        self.assertEqual(await gateway.generate_text(input="q"), "selected")
        self.assertEqual(responses.create_calls[0]["model"], "gpt-selected")

    async def test_factory_default_path_falls_back_to_anthropic_on_openai_transport(self) -> None:
        request = httpx.Request("POST", "https://api.openai.test/v1/responses")
        openai_responses = FakeOpenAIResponses(
            create_results=[httpx.ReadTimeout("timeout", request=request)]
        )
        anthropic_client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    {"content": [{"type": "text", "text": "fallback"}]}
                )
            ]
        )
        config = replace(
            get_model_gateway_config(
                {
                    "OPENAI_API_KEY": "openai-test",
                    "ANTHROPIC_API_KEY": "anthropic-test",
                }
            ),
            transport_retries=0,
        )
        gateway = create_model_gateway(
            config,
            openai_client=FakeOpenAIClient(openai_responses),
            anthropic_client=anthropic_client,
            sleep=no_sleep,
        )

        self.assertEqual(await gateway.generate_text(input="q"), "fallback")
        self.assertEqual(anthropic_client.calls[0]["json"]["model"], "claude-sonnet-4-6")

    async def test_anthropic_only_environment_keeps_terra_default_but_remains_operational(self) -> None:
        anthropic_client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    {"content": [{"type": "text", "text": "anthropic-only"}]}
                )
            ]
        )
        config = get_model_gateway_config(
            {
                "ANTHROPIC_API_KEY": "anthropic-test",
                "LLM_MAX_RETRIES": "0",
            }
        )

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, DEFAULT_OPENAI_MODEL)
        self.assertTrue(is_model_gateway_configured(config))
        gateway = create_model_gateway(
            config,
            anthropic_client=anthropic_client,
            sleep=no_sleep,
        )

        self.assertEqual(await gateway.generate_text(input="q"), "anthropic-only")
        self.assertEqual(anthropic_client.calls[0]["json"]["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
