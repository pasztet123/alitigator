from __future__ import annotations

from app.legal_research.models import CalculationRecord, EvidenceBinding, LegalClaim


def validate_claims(
    claims: list[LegalClaim], *, provision_ids: set[str], authority_ids: set[str],
    calculations: list[CalculationRecord], bindings: list[EvidenceBinding],
) -> list[str]:
    errors: list[str] = []
    calculation_ids = {item.calculation_id for item in calculations if item.validation_status == "valid"}
    bound_pairs = {
        (item.source_id, item.target_id)
        for item in bindings
        if item.target_type == "claim" and item.relation in {"supports", "contradicts", "context_only"}
    }
    for claim in claims:
        approved = claim.status in {"approved", "conditional_missing_fact"}
        if approved and claim.claim_type in {"legal_rule", "application", "deadline", "practical_conclusion"}:
            if not claim.controlling_provision_ids:
                errors.append(f"{claim.claim_id}:material_claim_without_primary_law")
        if not set(claim.controlling_provision_ids).issubset(provision_ids):
            errors.append(f"{claim.claim_id}:unknown_provision")
        referenced_authorities = set(claim.supporting_authority_ids) | set(claim.contrary_authority_ids)
        if not referenced_authorities.issubset(authority_ids):
            errors.append(f"{claim.claim_id}:unknown_authority")
        if any((source_id, claim.claim_id) not in bound_pairs for source_id in referenced_authorities):
            errors.append(f"{claim.claim_id}:unbound_authority")
        if approved and claim.claim_type == "authority_summary" and not claim.supporting_authority_ids:
            errors.append(f"{claim.claim_id}:authority_claim_without_authority")
        if approved and claim.claim_type in {"calculation", "deadline"}:
            if not claim.calculation_ids or not set(claim.calculation_ids).issubset(calculation_ids):
                errors.append(f"{claim.claim_id}:claim_without_valid_calculation")
    return list(dict.fromkeys(errors))


__all__ = ["validate_claims"]
