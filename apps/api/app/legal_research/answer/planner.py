from __future__ import annotations

from pydantic import Field

from app.legal_research.models import LegalResearchModel


class AnswerPlan(LegalResearchModel):
    thesis_claim_ids: list[str] = Field(default_factory=list)
    section_claim_ids: dict[str, list[str]] = Field(default_factory=dict)
    allowed_claim_ids: list[str] = Field(default_factory=list)


__all__ = ["AnswerPlan"]
