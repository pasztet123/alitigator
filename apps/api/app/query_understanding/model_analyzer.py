"""One bounded, tool-free model call for terminology expansion."""
from __future__ import annotations

from typing import Any

from pydantic import Field

from app.legal_rag_v2.schemas import V2Schema
from app.model_gateway import ModelGateway, ModelGatewayError
from .models import ModelQueryExpansion, ProvisionHint, QuestionCard

QUERY_ANALYZER_VERSION = "query_analyzer_v1"
PROMPT = """You expand terminology for a Polish tax-law retriever. Do not answer the legal question, decide tax liability, create signatures, remove locked concepts or create verified provisions. Use only supplied taxonomy IDs where possible. Return only the requested JSON."""


class ProvisionSuggestion(V2Schema):
    citation: str = Field(min_length=1, max_length=160)


class ModelQueryExpansionResponse(V2Schema):
    primary_issue: str = ""
    secondary_issues: list[str] = Field(default_factory=list, max_length=8)
    legal_concepts: list[str] = Field(default_factory=list, max_length=10)
    statutory_language: list[str] = Field(default_factory=list, max_length=10)
    factual_synonyms: list[str] = Field(default_factory=list, max_length=10)
    likely_document_wording: list[str] = Field(default_factory=list, max_length=8)
    material_distinctions: list[str] = Field(default_factory=list, max_length=8)
    negative_concepts: list[str] = Field(default_factory=list, max_length=8)
    unverified_provision_suggestions: list[ProvisionSuggestion] = Field(default_factory=list, max_length=8)
    uncertainties: list[str] = Field(default_factory=list, max_length=8)


async def analyze_with_model(gateway: ModelGateway, question_card: QuestionCard, matches: object, taxonomy: dict[str, Any], *, model: str) -> tuple[ModelQueryExpansion, dict[str, Any]]:
    try:
        raw = await gateway.generate_structured(
            response_model=ModelQueryExpansionResponse,
            input={"question_card": question_card.to_dict(), "dictionary_matches": [getattr(item, "__dict__", {}) for item in getattr(matches, "matches", ())], "allowed_taxonomy": taxonomy},
            system_prompt=PROMPT, model=model, reasoning_effort="low", max_output_tokens=900,
        )
        response = raw if isinstance(raw, ModelQueryExpansionResponse) else ModelQueryExpansionResponse.model_validate(raw)
    # This is enrichment. Any provider/fake/transport failure must preserve
    # the deterministic plan rather than fail the legal research request.
    except Exception:
        return ModelQueryExpansion(), {"status": "unavailable_or_invalid"}
    payload = response.model_dump(mode="json")
    return ModelQueryExpansion(
        primary_issue=response.primary_issue, secondary_issues=response.secondary_issues, legal_concepts=response.legal_concepts,
        statutory_language=response.statutory_language, factual_synonyms=response.factual_synonyms, likely_document_wording=response.likely_document_wording,
        material_distinctions=response.material_distinctions, negative_concepts=response.negative_concepts,
        unverified_provision_suggestions=[ProvisionHint(item.citation, "model", False) for item in response.unverified_provision_suggestions], uncertainties=response.uncertainties,
    ), payload
