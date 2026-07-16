"""Issue-scoped primary-law dependencies for transfer-pricing documentation."""

from __future__ import annotations

import re

from .family_foundation import _dedupe, _with_targets
from .schemas import LegalIssue, LegalResearchPlan, QueryFamily


TRANSFER_PRICING_TARGETS = (
    ("CIT", "art. 11a ust. 1 pkt 4"),
    ("CIT", "art. 11c ust. 1"),
    ("CIT", "art. 11k ust. 1"),
    ("CIT", "art. 11k ust. 2"),
    ("CIT", "art. 11k ust. 3"),
    ("CIT", "art. 11l ust. 1"),
    ("CIT", "art. 11n pkt 1"),
    ("CIT", "art. 11t ust. 1"),
)


def question_targets_transfer_pricing(question: str) -> bool:
    return bool(
        re.search(
            r"\b(cen\w*\s+transferow\w*|dokumentacj\w*\s+cen\w*\s+transferow\w*|"
            r"lokaln\w*\s+dokumentacj\w*|local\s+file|transakcj\w*\s+kontrolowan\w*|"
            r"zwolnieni\w*\s+dokumentacyjn\w*|art\.\s*11[klnt]\b)",
            question,
            re.IGNORECASE,
        )
    )


def _is_transfer_pricing_issue(issue: LegalIssue) -> bool:
    haystack = " ".join(
        (
            issue.issue_id,
            issue.label,
            issue.legal_mechanism,
            *issue.possible_provision_concepts,
            *issue.possible_legal_concepts,
            *issue.possible_provision_hints,
        )
    )
    return question_targets_transfer_pricing(haystack)


def enrich_transfer_pricing_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Ensure TP claims have their own issue and exact statutory bundle."""

    if not question_targets_transfer_pricing(question):
        return plan

    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        if _is_transfer_pricing_issue(issue):
            issues.append(_with_targets(issue, TRANSFER_PRICING_TARGETS))
            found = True
        else:
            issues.append(issue)
    if not found:
        issue = LegalIssue(
            issue_id="transfer_pricing_documentation",
            label="Ceny transferowe: obowiązek i zwolnienie dokumentacyjne",
            tax_domains=["CIT"],
            legal_mechanism="transfer_pricing_documentation",
            possible_provision_concepts=_dedupe(
                [f"{domain} {citation}" for domain, citation in TRANSFER_PRICING_TARGETS]
            ),
            requested_source_types=["statute", "interpretation", "judgment"],
            query_families=[
                QueryFamily(
                    family="natural_language",
                    query=question,
                    lane="both",
                    origin="fallback",
                )
            ],
            priority="high",
        )
        issues.append(_with_targets(issue, TRANSFER_PRICING_TARGETS))
    return plan.model_copy(update={"issues": issues})


__all__ = [
    "TRANSFER_PRICING_TARGETS",
    "enrich_transfer_pricing_plan",
    "question_targets_transfer_pricing",
]
