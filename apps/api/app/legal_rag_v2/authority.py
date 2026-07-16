from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from pydantic import ValidationError

from app.model_gateway import ModelGateway

from .retrieval import RetrievalCandidate
from .schemas import (
    AuthorityCard,
    AuthoritySourceSpans,
    DocumentSourceSpan,
)


AUTHORITY_CARD_PROMPT_VERSION = "authority_card_extractor_v2_1"
AUTHORITY_CARD_SCHEMA_VERSION = "legal_rag_v2_authority_card_v1"
DEFAULT_AUTHORITY_EXTRACTOR_MODEL = "gpt-5.6-terra"


AUTHORITY_EXTRACTOR_SYSTEM_PROMPT = """\
You extract an evidence card from one Polish legal authority document. Return
only the AuthorityCard required by the structured output schema.

Keep the taxpayer's position separate from the authority's holding and from a
court holding. Facts and questions are not holdings. Distinguish a judgment
that quashes an interpretation from one that dismisses the complaint. Do not
infer missing values: use null or an empty list.

Every material extracted value must be supported by one or more half-open
source spans in the supplied source_text. Span offsets are character offsets
in that exact string. Copy the exact substring into span.quote, and copy the
provided document_id (and chunk_id when present) into each span. Metadata can
identify the document, but it is not evidence for a legal holding.
"""


class AuthorityExtractionError(RuntimeError):
    pass


class SourceSpanValidationError(AuthorityExtractionError):
    pass


@dataclass(frozen=True)
class AuthorityDocument:
    document_id: str
    text: str
    document_type: str
    chunk_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorityExtractionResult:
    card: AuthorityCard
    trace: Mapping[str, Any]

    @property
    def fallback_used(self) -> bool:
        return bool(self.trace.get("fallback_used"))


@runtime_checkable
class AuthorityExtractor(Protocol):
    async def extract(
        self, document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
    ) -> AuthorityExtractionResult:
        ...


class ModelAuthorityExtractor:
    """Structured-output authority extraction through the shared gateway."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        model: Optional[str] = None,
        reasoning_effort: str = "low",
        max_output_tokens: int = 5000,
        heuristic_fallback: Optional["HeuristicAuthorityExtractor"] = None,
    ) -> None:
        self.gateway = gateway
        self.model = model or os.getenv(
            "AUTHORITY_EXTRACTOR_MODEL", DEFAULT_AUTHORITY_EXTRACTOR_MODEL
        )
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        # Supplying this object is the explicit opt-in to heuristic fallback.
        self.heuristic_fallback = heuristic_fallback

    async def extract(
        self, document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
    ) -> AuthorityExtractionResult:
        source = _coerce_document(document)
        try:
            payload = {
                "document_id": source.document_id,
                "chunk_id": source.chunk_id,
                "document_type": source.document_type,
                "metadata": _json_safe_metadata(source.metadata),
                "source_text": source.text,
            }
            raw = await self.gateway.generate_structured(
                response_model=AuthorityCard,
                input=json.dumps(payload, ensure_ascii=False, default=str),
                system_prompt=AUTHORITY_EXTRACTOR_SYSTEM_PROMPT,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
            )
            card = raw if isinstance(raw, AuthorityCard) else AuthorityCard.model_validate(raw)
            if card.document_id != source.document_id:
                raise SourceSpanValidationError(
                    "Extracted AuthorityCard document_id does not match its source"
                )
            validate_authority_source_spans(
                card,
                source.text,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
            )
            return AuthorityExtractionResult(
                card=card,
                trace={
                    "extractor": "model_structured_output",
                    "model": self.model,
                    "prompt_version": AUTHORITY_CARD_PROMPT_VERSION,
                    "schema_version": AUTHORITY_CARD_SCHEMA_VERSION,
                    "source_spans_validated": True,
                    "fallback_used": False,
                },
            )
        except Exception as exc:
            if self.heuristic_fallback is None:
                raise
            fallback = await self.heuristic_fallback.extract(source)
            return AuthorityExtractionResult(
                card=fallback.card,
                trace={
                    **dict(fallback.trace),
                    "fallback_used": True,
                    "fallback_reason": type(exc).__name__,
                    "primary_extractor": "model_structured_output",
                    "primary_model": self.model,
                },
            )

    async def extract_card(
        self, document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
    ) -> AuthorityCard:
        return (await self.extract(document)).card

    async def extract_many(
        self,
        documents: Sequence[
            AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
        ],
    ) -> list[AuthorityExtractionResult]:
        results: list[AuthorityExtractionResult] = []
        for document in documents:
            results.append(await self.extract(document))
        return results


class HeuristicAuthorityExtractor:
    """Conservative and explicitly traceable emergency/offline fallback."""

    trace_marker = "heuristic_authority_extractor_fallback"

    async def extract(
        self, document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
    ) -> AuthorityExtractionResult:
        source = _coerce_document(document)
        facts, fact_spans = _extract_section(
            source,
            (r"opis stanu faktycznego", r"stan faktyczny", r"opis zdarzenia przyszłego"),
        )
        issues, issue_spans = _extract_section(
            source,
            (r"pytani[ea]", r"zagadnienie prawne"),
        )
        taxpayer, taxpayer_spans = _extract_section(
            source,
            (r"stanowisko wnioskodawcy", r"państwa stanowisko", r"stanowisko podatnika"),
        )
        authority_holding, authority_spans = _extract_section(
            source,
            (r"ocena stanowiska", r"stanowisko organu"),
        )
        court_holding, court_spans = _extract_section(
            source,
            (r"rozstrzygnięcie sądu", r"sąd zważył"),
        )
        outcome, outcome_spans = _extract_outcome(source)
        reasoning, reasoning_spans = _extract_reasoning(source)
        provisions, provision_spans = _extract_provisions(source)

        metadata = dict(source.metadata)
        source_spans = AuthoritySourceSpans(
            facts=fact_spans,
            issues=issue_spans,
            cited_provisions=provision_spans,
            taxpayer_position=taxpayer_spans,
            authority_holding=authority_spans,
            court_holding=court_spans,
            outcome=outcome_spans,
            reasoning=reasoning_spans,
        )
        card = AuthorityCard(
            document_id=source.document_id,
            signature=str(metadata.get("signature") or ""),
            document_type=source.document_type,
            authority=str(metadata.get("authority") or ""),
            court=str(metadata.get("court") or ""),
            date=str(metadata.get("date") or metadata.get("published_date") or ""),
            legal_state_date=_optional_string(metadata.get("legal_state_date")),
            tax_domains=[str(value) for value in _as_list(metadata.get("tax_domains"))],
            facts=[facts] if facts else [],
            issues=[issues] if issues else [],
            cited_provisions=provisions,
            taxpayer_position=taxpayer,
            authority_holding=authority_holding,
            court_holding=court_holding,
            outcome=outcome,
            result_for_taxpayer=None,
            reasoning=reasoning,
            distinguishing_facts=[],
            source_spans=source_spans,
            extraction_confidence=0.35,
        )
        validate_authority_source_spans(
            card,
            source.text,
            document_id=source.document_id,
            chunk_id=source.chunk_id,
        )
        return AuthorityExtractionResult(
            card=card,
            trace={
                "extractor": self.trace_marker,
                "model": None,
                "prompt_version": "heuristic_v1_conservative",
                "schema_version": AUTHORITY_CARD_SCHEMA_VERSION,
                "source_spans_validated": True,
                "fallback_used": True,
                "fallback_reason": "explicit_heuristic_extractor",
            },
        )

    async def extract_card(
        self, document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any
    ) -> AuthorityCard:
        return (await self.extract(document)).card


# Explicit wrapper name for dependency-injection code.
FallbackAuthorityExtractor = HeuristicAuthorityExtractor


def validate_authority_source_spans(
    card: AuthorityCard,
    source_text: str,
    *,
    document_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> None:
    """Validate provenance without requiring summaries to be verbatim quotes."""

    expected_document = document_id or card.document_id
    for field_name, spans in card.source_spans:
        for span in spans:
            if span.document_id != expected_document:
                raise SourceSpanValidationError(
                    f"{field_name} span points to a different document"
                )
            if span.start < 0 or span.end > len(source_text) or span.end <= span.start:
                raise SourceSpanValidationError(
                    f"{field_name} span [{span.start}, {span.end}) is outside source text"
                )
            exact = source_text[span.start : span.end]
            if not exact.strip():
                raise SourceSpanValidationError(f"{field_name} span points to empty text")
            if span.quote is not None and span.quote != exact:
                raise SourceSpanValidationError(
                    f"{field_name} span quote does not match exact source text"
                )
            if chunk_id and span.chunk_id and span.chunk_id != chunk_id:
                raise SourceSpanValidationError(
                    f"{field_name} span points to a different source chunk"
                )


def authority_cache_key(
    document: AuthorityDocument | RetrievalCandidate | Mapping[str, Any] | Any,
    *,
    model: str,
    prompt_version: str = AUTHORITY_CARD_PROMPT_VERSION,
    schema_version: str = AUTHORITY_CARD_SCHEMA_VERSION,
) -> str:
    source = _coerce_document(document)
    document_hash = hashlib.sha256(source.text.encode("utf-8")).hexdigest()
    material = "\0".join((document_hash, model, prompt_version, schema_version))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _coerce_document(raw: Any) -> AuthorityDocument:
    if isinstance(raw, AuthorityDocument):
        source = raw
    elif isinstance(raw, RetrievalCandidate):
        source = AuthorityDocument(
            document_id=raw.document_id,
            chunk_id=raw.chunk_id or None,
            text=raw.text,
            document_type=raw.source_type,
            metadata=raw.metadata,
        )
    else:
        getter = raw.get if isinstance(raw, Mapping) else lambda key, default=None: getattr(raw, key, default)
        metadata = dict(getter("metadata", {}) or {})
        document_id = str(getter("document_id", "") or metadata.get("document_id") or "")
        text = str(getter("text", "") or getter("chunk_text", "") or "")
        document_type = str(
            getter("document_type", "")
            or getter("source_type", "")
            or metadata.get("source_type")
            or "authority_document"
        )
        source = AuthorityDocument(
            document_id=document_id,
            text=text,
            document_type=document_type,
            chunk_id=_optional_string(getter("chunk_id", None)),
            metadata=metadata,
        )
    if not source.document_id.strip():
        raise ValueError("Authority source requires a document_id")
    if not source.text.strip():
        raise ValueError("Authority source text cannot be empty")
    if not source.document_type.strip():
        raise ValueError("Authority source requires a document_type")
    return source


def _extract_section(
    source: AuthorityDocument, headings: Sequence[str]
) -> tuple[Optional[str], list[DocumentSourceSpan]]:
    heading = "|".join(f"(?:{item})" for item in headings)
    match = re.search(
        rf"(?im)^\s*(?:{heading})\s*:?[ \t]*\r?\n?",
        source.text,
    )
    if not match:
        return None, []
    following = source.text[match.end() :]
    next_heading = re.search(
        r"(?im)^\s*(?:opis stanu faktycznego|stan faktyczny|opis zdarzenia przyszłego|"
        r"pytani[ea]|zagadnienie prawne|stanowisko wnioskodawcy|państwa stanowisko|"
        r"stanowisko podatnika|ocena stanowiska|stanowisko organu|rozstrzygnięcie sądu|"
        r"sąd zważył|uzasadnienie|pouczenie)\s*:?[ \t]*$",
        following,
    )
    end = match.end() + (next_heading.start() if next_heading else len(following))
    start = match.end()
    while start < end and source.text[start].isspace():
        start += 1
    while end > start and source.text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None, []
    value = source.text[start:end]
    return value, [_span(source, start, end)]


def _extract_outcome(
    source: AuthorityDocument,
) -> tuple[Optional[str], list[DocumentSourceSpan]]:
    patterns = (
        r"[^.!?\n]*(?:skargę oddala|oddala skargę)[^.!?\n]*[.!?]?",
        r"[^.!?\n]*(?:uchyla zaskarżoną interpretację|interpretację uchyla)[^.!?\n]*[.!?]?",
        r"[^.!?\n]*stanowisko[^.!?\n]*(?:jest prawidłowe|jest nieprawidłowe)[^.!?\n]*[.!?]?",
    )
    for pattern in patterns:
        match = re.search(pattern, source.text, re.IGNORECASE)
        if match and match.group(0).strip():
            start, end = match.span()
            while start < end and source.text[start].isspace():
                start += 1
            return source.text[start:end], [_span(source, start, end)]
    return None, []


def _extract_reasoning(
    source: AuthorityDocument,
) -> tuple[Optional[str], list[DocumentSourceSpan]]:
    """Extract only an explicit conclusion-bearing source sentence.

    Retrieval chunks frequently start inside an authority's analysis and no
    longer contain the section heading used by ``_extract_section``.  A short
    sentence carrying an unmistakable conclusion is still useful evidence,
    but is deliberately labelled as reasoning rather than a holding.
    """

    patterns = (
        r"[^.!?\n]*(?:w konsekwencji|zatem|tym samym)[^.!?\n]*(?:[.!?]|$)",
        r"[^.!?\n]*(?:należy uznać|może stanowić koszt|mogą stanowić koszty|"
        r"nie może zostać zalicz\w*|nie mogą zostać zalicz\w*)[^.!?\n]*(?:[.!?]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, source.text, re.IGNORECASE)
        if not match or not match.group(0).strip():
            continue
        start, end = match.span()
        while start < end and source.text[start].isspace():
            start += 1
        value = source.text[start:end]
        return value, [_span(source, start, end)]
    return None, []


def _extract_provisions(
    source: AuthorityDocument,
) -> tuple[list[str], list[DocumentSourceSpan]]:
    pattern = re.compile(
        r"\bart\.\s*\d+[a-z]*(?:\s+ust\.\s*\d+[a-z]*)?"
        r"(?:\s+pkt\s*\d+[a-z]*)?(?:\s+lit\.\s*[a-z])?",
        re.IGNORECASE,
    )
    values: list[str] = []
    spans: list[DocumentSourceSpan] = []
    seen: set[str] = set()
    for match in pattern.finditer(source.text):
        value = " ".join(match.group(0).split())
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
        spans.append(_span(source, match.start(), match.end()))
    return values, spans


def _span(source: AuthorityDocument, start: int, end: int) -> DocumentSourceSpan:
    return DocumentSourceSpan(
        start=start,
        end=end,
        quote=source.text[start:end],
        source_id="authority_document",
        document_id=source.document_id,
        chunk_id=source.chunk_id,
    )


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(metadata), ensure_ascii=False, default=str))


def _optional_string(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "") else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]
