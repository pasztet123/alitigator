"""Evidence dependencies for family-foundation research issues.

This module adds retrieval targets only.  It never supplies a legal result or
benchmark-specific conclusion; every target must still be retrieved, parsed
and bound to an approved claim by the normal v2 pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from .schemas import LegalIssue, LegalResearchPlan, QueryFamily


FAMILY_FOUNDATION_ISSUE_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "family_foundation_allowed_activity_catalog": (
        ("UFR", "art. 5"),
    ),
    "family_foundation_cit_hidden_profit": (
        ("CIT", "art. 24q ust. 1"),
        ("CIT", "art. 24q ust. 1a pkt 2"),
        ("CIT", "art. 24q ust. 1a pkt 3"),
        ("CIT", "art. 24q ust. 1a pkt 6"),
        ("CIT", "art. 24q ust. 2"),
        ("CIT", "art. 24q ust. 5"),
        ("CIT", "art. 24q ust. 6"),
        ("CIT", "art. 24q ust. 7"),
        ("CIT", "art. 12 ust. 5"),
        ("CIT", "art. 12 ust. 5a"),
        ("CIT", "art. 12 ust. 6"),
        ("CIT", "art. 12 ust. 6a"),
        ("CIT", "art. 11a ust. 1 pkt 4"),
        ("UFR", "art. 2 ust. 2"),
    ),
    "family_foundation_disallowed_income_25_percent": (
        ("CIT", "art. 24r ust. 1"),
        ("CIT", "art. 24r ust. 2"),
        ("CIT", "art. 15 ust. 2"),
        ("UFR", "art. 5"),
    ),
    "family_foundation_beneficiary_pit": (
        ("PIT", "art. 20 ust. 1g"),
        ("PIT", "art. 21 ust. 1 pkt 157"),
        ("PIT", "art. 30 ust. 1 pkt 17"),
        ("UFR", "art. 2 ust. 2"),
        ("UFR", "art. 27 ust. 4"),
        ("UFR", "art. 28 ust. 1"),
        ("UFR", "art. 29 ust. 1"),
    ),
    "family_foundation_vat_related_party": (
        ("VAT", "art. 15 ust. 1"),
        ("VAT", "art. 15 ust. 2"),
        ("VAT", "art. 29a ust. 1"),
        ("VAT", "art. 32 ust. 1"),
        ("VAT", "art. 32 ust. 2"),
        ("VAT", "art. 43 ust. 1 pkt 36"),
        ("VAT", "art. 86 ust. 1"),
    ),
}

_LEGACY_WHT_ISSUE_IDS = frozenset(
    {
        "wht_interest",
        "wht_management_services",
        "pay_and_refund",
        "interest_royalties_exemption",
        "beneficial_owner",
    }
)


def _question_requests_wht(question: str) -> bool:
    return bool(
        re.search(
            r"\b(wht|podatek\s+u\s+[źz]r[óo]d[łl]a|withholding|beneficial\s+owner|"
            r"certyfikat\w*\s+rezydencji|nierezydent\w*|upo\b|"
            r"umow\w*\s+o\s+unikaniu\s+podw[óo]jnego\s+opodatkowania)\b",
            question,
            re.IGNORECASE,
        )
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split())
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        result.append(cleaned)
    return result


def _with_targets(issue: LegalIssue, targets: tuple[tuple[str, str], ...]) -> LegalIssue:
    queries = list(issue.query_families)
    existing = {
        (item.lane, " ".join(item.query.casefold().split()))
        for item in queries
    }
    for domain, citation in targets:
        query = f"{domain} {citation}"
        key = ("primary_law", " ".join(query.casefold().split()))
        if key in existing:
            continue
        queries.append(
            QueryFamily(
                family="explicit_provision_reference",
                query=query,
                lane="primary_law",
                origin="fallback",
            )
        )
        existing.add(key)

    payload = issue.model_dump(mode="python")
    payload.update(
        tax_domains=_dedupe([*issue.tax_domains, *(domain for domain, _ in targets)]),
        possible_provision_concepts=_dedupe(
            [*issue.possible_provision_concepts, *(f"{domain} {citation}" for domain, citation in targets)]
        ),
        requested_source_types=_dedupe([*issue.requested_source_types, "statute"]),
        query_families=[item.model_dump(mode="python") for item in queries],
    )
    return LegalIssue.model_validate(payload)


def family_foundation_issue_kind(issue: LegalIssue) -> str | None:
    """Classify model-chosen issue IDs without changing their identity."""

    if issue.issue_id in FAMILY_FOUNDATION_ISSUE_TARGETS:
        return issue.issue_id
    haystack = " ".join(
        [
            issue.issue_id,
            issue.label,
            issue.legal_mechanism,
            *issue.possible_provision_concepts,
            *issue.possible_legal_concepts,
            *issue.possible_provision_hints,
        ]
    ).casefold()
    if "fundacj" not in haystack and "ufr" not in haystack:
        return None
    domains = {domain.upper() for domain in issue.tax_domains}
    if "24r" in haystack or "25%" in haystack or "niedozwol" in haystack:
        return "family_foundation_disallowed_income_25_percent"
    if "24q" in haystack or "ukryt" in haystack:
        return "family_foundation_cit_hidden_profit"
    if "VAT" in domains or "vat" in haystack or "art. 32" in haystack or "wartość rynk" in haystack:
        return "family_foundation_vat_related_party"
    if "PIT" in domains or "pit" in haystack or "beneficjent" in haystack or "grup" in haystack:
        return "family_foundation_beneficiary_pit"
    if "art. 5" in haystack or "katalog" in haystack or "dozwolon" in haystack:
        return "family_foundation_allowed_activity_catalog"
    return None


def enrich_family_foundation_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Attach exact statutory dependencies to recognized family issues."""

    normalized = question.casefold()
    if "fundacj" not in normalized and "ufr" not in normalized:
        return plan

    changed = False
    issues: list[LegalIssue] = []
    for issue in plan.issues:
        # The bounded legacy planner used to confuse domestic loans, interest,
        # management services and a generic "limit" with WHT.  Keep genuine
        # WHT axes when the question says so, but do not expose five unrelated
        # empty sections in a domestic family-foundation analysis.
        if issue.issue_id in _LEGACY_WHT_ISSUE_IDS and not _question_requests_wht(question):
            changed = True
            continue
        kind = family_foundation_issue_kind(issue)
        targets = FAMILY_FOUNDATION_ISSUE_TARGETS.get(kind or "")
        if not targets:
            issues.append(issue)
            continue
        issues.append(_with_targets(issue, targets))
        changed = True
    if not changed:
        return plan
    return plan.model_copy(update={"issues": issues})


__all__ = [
    "FAMILY_FOUNDATION_ISSUE_TARGETS",
    "enrich_family_foundation_plan",
    "family_foundation_issue_kind",
]
