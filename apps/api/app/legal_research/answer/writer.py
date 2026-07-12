from __future__ import annotations

from app.model_gateway import ModelGateway
from app.legal_research.models import AnswerDraft


class AnswerWriter:
    def __init__(self, gateway: ModelGateway, *, model: str) -> None:
        self.gateway = gateway
        self.model = model

    async def write(self, payload: str) -> AnswerDraft:
        return await self.gateway.generate_structured(
            response_model=AnswerDraft, input=payload,
            system_prompt=(
                "Write only from approved claims, explicit facts and calculation records. "
                "Do not add facts, sources, signatures, provision numbers, dates or amounts. "
                "Do not write a sources section; the deterministic renderer owns sources."
            ),
            model=self.model, reasoning_effort="medium", max_output_tokens=10000,
        )


__all__ = ["AnswerWriter"]
