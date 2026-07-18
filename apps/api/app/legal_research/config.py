from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LegalResearchConfig:
    mode: str = "legacy"
    artifact_root: Path = Path("artifacts/model_rag_model")
    planner_model: str = "gpt-5.6-terra"
    evidence_analyst_model: str = "gpt-5.6-terra"
    answer_writer_model: str = "gpt-5.6-terra"
    planner_confidence_threshold: float = 0.55
    max_retrieval_iterations: int = 2
    allow_legacy_fallback: bool = True
    min_topic_similarity: float = 0.25
    min_issue_similarity: float = 0.35
    min_material_fact_similarity: float = 0.30
    min_holding_relevance: float = 0.35
    min_claim_entailment: float = 0.55

    @classmethod
    def from_env(cls) -> "LegalResearchConfig":
        return cls(
            mode=(os.getenv("LEGAL_RAG_MODE") or "model_rag_model").strip().lower(),
            artifact_root=Path(os.getenv("MODEL_RAG_MODEL_ARTIFACT_ROOT", "artifacts/model_rag_model")),
            planner_model=os.getenv("LEGAL_PLANNER_MODEL", "gpt-5.6-terra"),
            evidence_analyst_model=os.getenv("EVIDENCE_ANALYST_MODEL", "gpt-5.6-terra"),
            answer_writer_model=os.getenv("ANSWER_WRITER_MODEL", "gpt-5.6-terra"),
            planner_confidence_threshold=float(os.getenv("LEGAL_PLANNER_CONFIDENCE_THRESHOLD", "0.55")),
            max_retrieval_iterations=min(2, max(1, int(os.getenv("LEGAL_MAX_RETRIEVAL_ITERATIONS", "2")))),
            allow_legacy_fallback=_bool("LEGAL_ALLOW_LEGACY_FALLBACK", True),
            min_topic_similarity=float(os.getenv("LEGAL_MIN_TOPIC_SIMILARITY", "0.25")),
            min_issue_similarity=float(os.getenv("LEGAL_MIN_ISSUE_SIMILARITY", "0.35")),
            min_material_fact_similarity=float(os.getenv("LEGAL_MIN_MATERIAL_FACT_SIMILARITY", "0.30")),
            min_holding_relevance=float(os.getenv("LEGAL_MIN_HOLDING_RELEVANCE", "0.35")),
            min_claim_entailment=float(os.getenv("LEGAL_MIN_CLAIM_ENTAILMENT", "0.55")),
        )
