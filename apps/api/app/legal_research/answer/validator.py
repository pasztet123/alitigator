from __future__ import annotations

import re
from typing import Mapping

from app.legal_research.models import AnswerDraft, CalculationRecord, LegalClaim


INTERNAL_ID = re.compile(r"\b(?:claim|fact|calculation|provision|issue)_[A-Za-z0-9_-]+\b", re.I)


def validate_final_answer(
    answer: str, *, draft: AnswerDraft, claims: list[LegalClaim],
    source_registry: Mapping[str, object], calculations: list[CalculationRecord],
) -> list[str]:
    errors: list[str] = []
    approved = {item.claim_id: item for item in claims if item.status in {"approved", "conditional_missing_fact"}}
    if not set(draft.claim_ids_used).issubset(approved):
        errors.append("writer_used_unapproved_claim")
    known_sources = set(source_registry)
    if not set(draft.provision_ids_used + draft.authority_ids_used).issubset(known_sources):
        errors.append("writer_added_unknown_source")
    if INTERNAL_ID.search(answer):
        errors.append("internal_ids_in_final_answer")
    positions = [answer.find(f"{heading}\n") for heading in ("Teza", "Analiza", "Źródła", "Ryzyka i luki")]
    if any(value < 0 for value in positions) or positions != sorted(positions):
        errors.append("invalid_section_structure")
    if approved and not draft.provision_ids_used:
        errors.append("approved_claims_without_source_section")
    valid_calculations = {item.calculation_id for item in calculations if item.validation_status == "valid"}
    if not set(draft.calculation_ids_used).issubset(valid_calculations):
        errors.append("writer_used_invalid_calculation")
    return list(dict.fromkeys(errors))


__all__ = ["validate_final_answer"]
