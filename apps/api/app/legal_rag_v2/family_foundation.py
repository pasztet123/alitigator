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
    "family_foundation_investment_income": (
        ("UFR", "art. 5 ust. 1 pkt 3"),
        ("UFR", "art. 5 ust. 1 pkt 4"),
        ("CIT", "art. 6 ust. 1 pkt 25"),
        ("CIT", "art. 6 ust. 6"),
        ("CIT", "art. 6 ust. 7"),
    ),
    "family_foundation_related_party_rent": (
        ("UFR", "art. 5 ust. 1 pkt 2"),
        ("CIT", "art. 6 ust. 1 pkt 25"),
        ("CIT", "art. 6 ust. 8"),
        ("CIT", "art. 11a ust. 1 pkt 4"),
        ("CIT", "art. 19 ust. 1"),
        ("CIT", "art. 24q ust. 8"),
        ("CIT", "art. 24q ust. 9"),
    ),
    "family_foundation_related_party_services": (
        ("CIT", "art. 24q ust. 1"),
        ("CIT", "art. 24q ust. 1a pkt 3"),
        ("CIT", "art. 24q ust. 2"),
        ("CIT", "art. 24q ust. 6"),
        ("CIT", "art. 24q ust. 7"),
    ),
    "family_foundation_borrowing_from_related_party": (
        ("CIT", "art. 24q ust. 1"),
        ("CIT", "art. 24q ust. 1a pkt 1"),
        ("CIT", "art. 24q ust. 2"),
        ("CIT", "art. 24q ust. 6"),
        ("CIT", "art. 24q ust. 7"),
    ),
    "family_foundation_beneficiary_loan": (
        ("UFR", "art. 5 ust. 1 pkt 5 lit. c"),
        ("CIT", "art. 24q ust. 1"),
        ("CIT", "art. 24q ust. 1a pkt 2"),
        ("CIT", "art. 24q ust. 1a pkt 5"),
        ("CIT", "art. 24q ust. 1a pkt 6"),
        ("CIT", "art. 24q ust. 2"),
        ("CIT", "art. 24q ust. 5"),
        ("CIT", "art. 24q ust. 6"),
        ("CIT", "art. 12 ust. 5"),
        ("CIT", "art. 12 ust. 6"),
        ("CIT", "art. 12 ust. 6a"),
    ),
    "family_foundation_beneficiary_benefit": (
        ("UFR", "art. 2 ust. 2"),
        ("CIT", "art. 24q ust. 1"),
        ("CIT", "art. 24q ust. 2"),
        ("CIT", "art. 24q ust. 6"),
        ("PIT", "art. 20 ust. 1g"),
        ("PIT", "art. 21 ust. 1 pkt 157"),
        ("PIT", "art. 21 ust. 49"),
        ("PIT", "art. 30 ust. 1 pkt 17"),
        ("UFR", "art. 27 ust. 4"),
        ("UFR", "art. 28 ust. 1"),
        ("UFR", "art. 29 ust. 1"),
    ),
    "family_foundation_real_estate_activity": (
        ("UFR", "art. 5 ust. 1 pkt 1"),
        ("UFR", "art. 5 ust. 3"),
        ("CIT", "art. 6 ust. 7"),
        ("CIT", "art. 24r ust. 1"),
        ("CIT", "art. 24r ust. 2"),
        ("CIT", "art. 15 ust. 2"),
    ),
    "family_foundation_common_costs": (
        ("CIT", "art. 15 ust. 2"),
        ("CIT", "art. 15 ust. 2a"),
        ("CIT", "art. 24r ust. 2"),
    ),
    "family_foundation_tax_credit_and_reporting": (
        ("CIT", "art. 24q ust. 6"),
        ("CIT", "art. 24q ust. 8"),
        ("CIT", "art. 24q ust. 9"),
        ("CIT", "art. 24s ust. 1"),
        ("CIT", "art. 27 ust. 1"),
    ),
    "family_foundation_vat_transactions": (
        ("VAT", "art. 5 ust. 1"),
        ("VAT", "art. 7 ust. 1"),
        ("VAT", "art. 8 ust. 1"),
        ("VAT", "art. 15 ust. 1"),
        ("VAT", "art. 15 ust. 2"),
        ("VAT", "art. 29a ust. 1"),
        ("VAT", "art. 32 ust. 1"),
        ("VAT", "art. 32 ust. 2"),
        ("VAT", "art. 43 ust. 1 pkt 10"),
        ("VAT", "art. 43 ust. 1 pkt 10a"),
        ("VAT", "art. 43 ust. 1 pkt 36"),
        ("VAT", "art. 86 ust. 1"),
        ("VAT", "art. 90 ust. 1"),
        ("VAT", "art. 90 ust. 2"),
        ("VAT", "art. 99 ust. 1"),
    ),
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
        ("PIT", "art. 21 ust. 49"),
        ("PIT", "art. 30 ust. 1 pkt 17"),
        ("UFR", "art. 2 ust. 2"),
        ("UFR", "art. 27 ust. 4"),
        ("UFR", "art. 28 ust. 1"),
        ("UFR", "art. 29 ust. 1"),
    ),
    "family_foundation_vat_related_party": (
        ("VAT", "art. 5 ust. 1"),
        ("VAT", "art. 7 ust. 1"),
        ("VAT", "art. 8 ust. 1"),
        ("VAT", "art. 15 ust. 1"),
        ("VAT", "art. 15 ust. 2"),
        ("VAT", "art. 29a ust. 1"),
        ("VAT", "art. 32 ust. 1"),
        ("VAT", "art. 32 ust. 2"),
        ("VAT", "art. 43 ust. 1 pkt 36"),
        ("VAT", "art. 86 ust. 1"),
        ("VAT", "art. 90 ust. 1"),
        ("VAT", "art. 90 ust. 2"),
    ),
}


_GENERIC_FAMILY_ISSUE_IDS = frozenset(
    {
        "family_foundation_allowed_activity_catalog",
        "family_foundation_cit_hidden_profit",
        "family_foundation_disallowed_income_25_percent",
        "family_foundation_beneficiary_pit",
        "family_foundation_vat_related_party",
    }
)


# These are transaction/mechanism routers, not legal conclusions.  Their only
# effect is to preserve distinct questions and retrieve the statute bundle for
# each one.  A claim still has to be produced by the model and pass the normal
# primary-law gate.
_TRANSACTION_ISSUE_SPECS: tuple[
    tuple[str, str, tuple[str, ...], str, str], ...
] = (
    (
        "family_foundation_investment_income",
        "Fundacja rodzinna: przychody z inwestycji kapitałowych",
        ("CIT", "UFR"),
        "investment_income_and_family_foundation_exemption",
        r"dywidend|obligacj|papier\w*\s+warto[śs]ciow|inwestycj\w*\s+kapita[łl]",
    ),
    (
        "family_foundation_related_party_rent",
        "Fundacja rodzinna: najem podmiotowi powiązanemu",
        ("CIT", "UFR"),
        "related_party_rent_and_family_foundation_exemption",
        r"(?:najem|wynajm|dzier[żz]aw).{0,160}(?:powi[ąa]zan|fundator|beneficjent|sp[óo][łl]k)|(?:powi[ąa]zan|fundator|beneficjent|sp[óo][łl]k).{0,160}(?:najem|wynajm|dzier[żz]aw)",
    ),
    (
        "family_foundation_related_party_services",
        "Fundacja rodzinna: usługi podmiotu fundatora lub podmiotu powiązanego",
        ("CIT",),
        "related_party_services_hidden_profit",
        r"us[łl]ug.{0,180}(?:fundator|beneficjent|powi[ąa]zan|nale[żz][ąa]c)|(?:fundator|beneficjent|powi[ąa]zan|nale[żz][ąa]c).{0,180}us[łl]ug",
    ),
    (
        "family_foundation_borrowing_from_related_party",
        "Fundacja rodzinna: pożyczka udzielona fundacji i jej koszty",
        ("CIT",),
        "related_party_loan_to_foundation_hidden_profit",
        r"(?:udziel\w*\s+fundacj\w*\s+po[żz]ycz|po[żz]ycz\w*\s+(?:udzielon\w*\s+)?fundacj|po[żz]ycz\w*\s+od\s+(?:fundator|beneficjent)|odsetk\w*.{0,100}(?:fundator|beneficjent))",
    ),
    (
        "family_foundation_beneficiary_loan",
        "Fundacja rodzinna: pożyczka udzielona beneficjentowi",
        ("CIT", "UFR"),
        "beneficiary_loan_hidden_profit",
        r"(?:fundacj\w*.{0,100}udziel\w*.{0,60}po[żz]ycz|po[żz]ycz\w*.{0,100}beneficjent|po[żz]ycz\w*.{0,100}(?:c[óo]rk|syn|ma[łl][żz]on))",
    ),
    (
        "family_foundation_beneficiary_benefit",
        "Fundacja rodzinna: świadczenie oraz PIT beneficjenta",
        ("CIT", "PIT", "UFR"),
        "beneficiary_benefit_cit_and_pit",
        r"(?:[śs]wiadczeni\w*|wyp[łl]at\w*.{0,100}beneficjent|pit\s+beneficjent)",
    ),
    (
        "family_foundation_real_estate_activity",
        "Fundacja rodzinna: obrót nieruchomościami i działalność niedozwolona",
        ("CIT", "UFR"),
        "real_estate_resale_and_disallowed_activity",
        r"(?:mieszka[nń]|lokal|nieruchomo[śs]c).{0,180}(?:sprzeda|zby|odsprzeda)|(?:sprzeda|zby|odsprzeda).{0,180}(?:mieszka[nń]|lokal|nieruchomo[śs]c)",
    ),
    (
        "family_foundation_common_costs",
        "Fundacja rodzinna: koszty wspólne działalności zwolnionej i opodatkowanej",
        ("CIT",),
        "common_cost_allocation",
        r"koszt\w*\s+wsp[óo]ln|przypisa\w*.{0,100}koszt|alokacj\w*.{0,80}koszt",
    ),
    (
        "family_foundation_tax_credit_and_reporting",
        "Fundacja rodzinna: odliczenie podatku, deklaracje i terminy",
        ("CIT",),
        "family_foundation_tax_credit_and_reporting",
        r"(?:odlicz\w*.{0,120}podat|podatek.{0,120}odlicz|deklaracj|termin\w*\s+(?:p[łl]atno[śs]ci|zap[łl]aty))",
    ),
    (
        "family_foundation_vat_transactions",
        "Fundacja rodzinna: VAT od najmu, sprzedaży i prawo do odliczenia",
        ("VAT",),
        "family_foundation_vat_transactions",
        r"\bvat\b|podat\w*\s+od\s+towar\w*\s+i\s+us[łl]ug|podatek\s+naliczon",
    ),
)

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


def _transaction_issue_specs(question: str) -> list[tuple[str, str, tuple[str, ...], str]]:
    """Return independently detected family-foundation mechanisms.

    Complex questions often ask for a transaction-by-transaction answer.  A
    provider or the bounded legacy planner may nevertheless collapse all of
    them into one generic ``24q`` issue.  Detecting the transaction nouns here
    preserves research scope; it deliberately does not classify their legal
    result.
    """

    normalized = " ".join(question.casefold().split())
    return [
        (issue_id, label, domains, mechanism)
        for issue_id, label, domains, mechanism, pattern in _TRANSACTION_ISSUE_SPECS
        if re.search(pattern, normalized, re.IGNORECASE)
    ]


def _new_transaction_issue(
    *,
    issue_id: str,
    label: str,
    domains: tuple[str, ...],
    mechanism: str,
) -> LegalIssue:
    targets = FAMILY_FOUNDATION_ISSUE_TARGETS[issue_id]
    issue = LegalIssue(
        issue_id=issue_id,
        label=label,
        tax_domains=list(domains),
        legal_mechanism=mechanism,
        possible_provision_concepts=[
            f"{domain} {citation}" for domain, citation in targets
        ],
        requested_source_types=["statute", "interpretation", "judgment"],
        query_families=[
            QueryFamily(
                family="natural_language",
                query=label,
                lane="both",
                origin="fallback",
            )
        ],
        priority="high",
    )
    return _with_targets(issue, targets)


def enrich_family_foundation_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Attach exact statutory dependencies to recognized family issues."""

    normalized = question.casefold()
    if "fundacj" not in normalized and "ufr" not in normalized:
        return plan

    transaction_specs = _transaction_issue_specs(question)
    split_transaction_scope = len(transaction_specs) >= 2
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
        # The old five-axis family plan is intentionally replaced only when
        # the question contains several independently detectable mechanisms.
        # Short questions retain the original model-selected issue unchanged.
        if split_transaction_scope and issue.issue_id in _GENERIC_FAMILY_ISSUE_IDS:
            changed = True
            continue
        kind = family_foundation_issue_kind(issue)
        targets = FAMILY_FOUNDATION_ISSUE_TARGETS.get(kind or "")
        if not targets:
            issues.append(issue)
            continue
        issues.append(_with_targets(issue, targets))
        changed = True
    existing_ids = {issue.issue_id for issue in issues}
    for issue_id, label, domains, mechanism in transaction_specs:
        if issue_id in existing_ids:
            continue
        issues.append(
            _new_transaction_issue(
                issue_id=issue_id,
                label=label,
                domains=domains,
                mechanism=mechanism,
            )
        )
        existing_ids.add(issue_id)
        changed = True
    if not changed:
        return plan
    return plan.model_copy(update={"issues": issues})


__all__ = [
    "FAMILY_FOUNDATION_ISSUE_TARGETS",
    "enrich_family_foundation_plan",
    "family_foundation_issue_kind",
]
