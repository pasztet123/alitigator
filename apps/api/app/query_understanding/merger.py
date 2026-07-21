"""Precedence-preserving merger of deterministic and model query understanding."""
from __future__ import annotations

from app.legal_concepts import load_default_dictionary
from .models import ModelQueryExpansion, QueryPlan, QuestionCard


def _unique(values: list[str], maximum: int) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))[:maximum]


def merge_query_plan(deterministic: QuestionCard | QueryPlan, matches: object | dict, model_output: ModelQueryExpansion | dict | None = None) -> QueryPlan:
    if isinstance(deterministic, QueryPlan):
        card = deterministic.question_card
        model_output = matches if isinstance(matches, dict) else model_output
        matches = None
    else:
        card = deterministic
    model = model_output if isinstance(model_output, ModelQueryExpansion) else ModelQueryExpansion(**(model_output or {}))
    dictionary = load_default_dictionary()
    locked = list(dict.fromkeys(card.locked_institutions))
    deterministic_primary = locked[0] if locked else ""
    conflicts: list[dict[str, str]] = []
    if deterministic_primary and model.primary_issue and model.primary_issue != deterministic_primary:
        conflicts.append({"field": "primary_issue", "deterministic": deterministic_primary, "model": model.primary_issue, "resolution": "keep_deterministic"})
    primary = deterministic_primary or model.primary_issue
    concepts = [
        *[dictionary.by_id[item].canonical_name for item in card.detected_concepts if item in dictionary.by_id],
        *model.legal_concepts,
    ]
    statutory = [
        term for item in card.detected_concepts if item in dictionary.by_id
        for term in dictionary.by_id[item].statutory_terms
    ]
    legal = [
        term for item in card.detected_concepts if item in dictionary.by_id
        for term in dictionary.by_id[item].legal_terms
    ]
    return QueryPlan(
        question_card=card, primary_issue=primary, secondary_issues=_unique(model.secondary_issues, 8),
        locked_institutions=locked, legal_concepts=_unique([*concepts, *legal], 10),
        statutory_language=_unique([*statutory, *model.statutory_language], 10),
        factual_synonyms=_unique(model.factual_synonyms, 10),
        likely_document_wording=_unique(model.likely_document_wording, 8),
        verified_provision_hints=card.verified_provision_hints,
        soft_provision_hints=model.unverified_provision_suggestions,
        material_distinctions=_unique(model.material_distinctions, 8),
        negative_concepts=_unique([*card.negative_concepts, *model.negative_concepts], 8),
        uncertainties=_unique(model.uncertainties, 8), conflicts=conflicts,
    )
