from __future__ import annotations

from .backend import RetrievalCandidate
from app.legal_research.models import ResearchIssue, RetrievalQuery


def build_issue_queries(issue: ResearchIssue) -> list[RetrievalQuery]:
    """Build generic per-issue query families without nominating outcomes or IDs."""
    bases = [
        ("natural_language", issue.label),
        ("legal_concept", " ".join([issue.legal_mechanism, *issue.possible_legal_concepts])),
        ("fact_signature", " ".join([*issue.transactions, *issue.payments, *issue.taxpayer_roles])),
        ("fact_contrast", " ".join([issue.label, *issue.negative_constraints])),
    ]
    for hint in issue.possible_provision_hints:
        bases.append(("explicit_provision", hint))
    queries: list[RetrievalQuery] = []
    for lane in ("primary", "authority"):
        for index, (family, text) in enumerate(bases, start=1):
            query = " ".join(text.split())
            if not query:
                continue
            queries.append(RetrievalQuery(
                query_id=f"{issue.issue_id}:{lane}:{index}", issue_id=issue.issue_id,
                lane=lane, family=family, query=query,
                positive_constraints=issue.positive_constraints,
                negative_constraints=issue.negative_constraints,
                expected_source_types=list(issue.requested_source_types), generated_by="planner",
            ))
    return queries


__all__ = ["build_issue_queries", "RetrievalCandidate"]
