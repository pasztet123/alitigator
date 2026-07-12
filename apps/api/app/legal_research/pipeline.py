from __future__ import annotations

from typing import Any

from app.legal_rag_v2.pipeline import (
    LegalRagV2Pipeline,
    create_default_pipeline as _create_v2_pipeline,
)


class ModelRagModelPipeline:
    """Stable façade for the additive Model → RAG → Model implementation."""

    def __init__(self, inner: LegalRagV2Pipeline) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def run(self, question: str, *, mode: str = "model_rag_model", **kwargs: Any) -> Any:
        if mode not in {"model_rag_model", "legal_rag_v2", "shadow"}:
            raise ValueError(f"unsupported legal research mode: {mode}")
        canonical_mode = "model_rag_model" if mode == "legal_rag_v2" else mode
        return await self.inner.run(question, mode=canonical_mode, **kwargs)


def create_default_pipeline(**kwargs: Any) -> ModelRagModelPipeline:
    return ModelRagModelPipeline(_create_v2_pipeline(**kwargs))


__all__ = ["ModelRagModelPipeline", "create_default_pipeline"]
