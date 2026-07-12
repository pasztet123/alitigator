from app.legal_research.models import EvidenceBinding


def selected_bindings(bindings: list[EvidenceBinding], *, minimum_score: float) -> list[EvidenceBinding]:
    return [
        item for item in bindings
        if item.relation in {"supports", "contradicts", "context_only"} and item.score >= minimum_score
    ]


def authority_abstention_reasons(
    *, topic_similarity: float, issue_similarity: float,
    material_fact_similarity: float, holding_relevance: float,
    min_topic_similarity: float, min_issue_similarity: float,
    min_material_fact_similarity: float, min_holding_relevance: float,
) -> list[str]:
    checks = {
        "below_min_topic_similarity": (topic_similarity, min_topic_similarity),
        "below_min_issue_similarity": (issue_similarity, min_issue_similarity),
        "below_min_material_fact_similarity": (material_fact_similarity, min_material_fact_similarity),
        "below_min_holding_relevance": (holding_relevance, min_holding_relevance),
    }
    return [reason for reason, (actual, threshold) in checks.items() if actual < threshold]


__all__ = ["EvidenceBinding", "authority_abstention_reasons", "selected_bindings"]
