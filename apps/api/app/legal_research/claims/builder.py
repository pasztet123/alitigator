from app.legal_research.models import LegalClaim


def approved_claims(claims: list[LegalClaim]) -> list[LegalClaim]:
    return [item for item in claims if item.status in {"approved", "conditional_missing_fact"}]


__all__ = ["LegalClaim", "approved_claims"]
