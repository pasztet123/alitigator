from __future__ import annotations

from typing import Any

from app.model_gateway import ModelGateway

from .schemas import AuthorityCard, EvidenceBinding, LegalRuleEvidence, MissingEvidenceRequest


class EvidenceAnalyst:
    """Structured evidence model boundary; it never writes the final answer."""

    def __init__(self, gateway: ModelGateway, *, model: str) -> None:
        self.gateway = gateway
        self.model = model

    async def analyze(self, *, payload: str, schema: type[Any]) -> Any:
        return await self.gateway.generate_structured(
            response_model=schema,
            input=payload,
            system_prompt=(
                "Classify only supplied legal evidence. Separate taxpayer position, facts, "
                "authority/court holding and outcome. Reject wrong-neighbor events, bind each "
                "source to one issue, preserve exact spans, and request missing evidence. "
                "Do not answer the user's legal question."
            ),
            model=self.model,
            reasoning_effort="medium",
            max_output_tokens=12000,
        )


__all__ = ["AuthorityCard", "EvidenceAnalyst", "EvidenceBinding", "LegalRuleEvidence", "MissingEvidenceRequest"]
