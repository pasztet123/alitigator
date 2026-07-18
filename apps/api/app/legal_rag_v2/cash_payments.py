"""Research routing for cash payments that can exclude an income-tax cost.

This is a mechanism classifier, not a case-answer template.  It recognises the
statutory link between the payment-channel obligation and the income-tax
adjustment, then requires each member of that bundle from the corpus.
"""

from __future__ import annotations

import re

from .cit_costs import cost_tax_domain, question_targets_cit_cost_deductibility
from .family_foundation import _dedupe, _with_targets
from .schemas import LegalIssue, LegalResearchPlan, QueryFamily


BUSINESS_PAYMENT_TARGETS = (("PP", "art. 19"),)
PIT_CASH_PAYMENT_COST_TARGETS = (
    *BUSINESS_PAYMENT_TARGETS,
    ("PIT", "art. 22p ust. 1"),
    ("PIT", "art. 22p ust. 2"),
    ("PIT", "art. 22p ust. 3"),
)
CIT_CASH_PAYMENT_COST_TARGETS = (
    *BUSINESS_PAYMENT_TARGETS,
    ("CIT", "art. 15d ust. 1"),
    ("CIT", "art. 15d ust. 2"),
    ("CIT", "art. 15d ust. 3"),
)


def question_targets_cash_payment_cost_exclusion(question: str) -> bool:
    """Identify the payment-channel exclusion without deciding its result."""

    has_payment_channel = bool(
        re.search(
            r"got[óo]wk\w*|bez\s+po[śs]rednictwem\s+rachunku|"
            r"rachunk\w*\s+p[łl]atnicz\w*|ponown\w*.{0,80}przelew|"
            r"przelew\w*.{0,80}(?:got[óo]wk|zwrot)",
            question,
            re.IGNORECASE,
        )
    )
    has_transaction_threshold = bool(
        re.search(
            r"transakcj\w*|rat\w*|15\s*000|limit\w*\s+p[łl]atno[śs]ci|"
            r"jednorazow\w*\s+warto[śs]ci",
            question,
            re.IGNORECASE,
        )
    )
    return has_payment_channel and (
        has_transaction_threshold or question_targets_cit_cost_deductibility(question)
    )


def _is_cash_payment_issue(issue: LegalIssue) -> bool:
    text = " ".join(
        (
            issue.issue_id,
            issue.label,
            issue.legal_mechanism,
            *issue.possible_provision_concepts,
            *issue.possible_legal_concepts,
            *issue.possible_provision_hints,
            *issue.transactions,
            *issue.payments,
        )
    )
    return question_targets_cash_payment_cost_exclusion(text)


def _is_replaceable_general_issue(issue: LegalIssue, tax_domain: str) -> bool:
    """Recognize the planner's unscoped income-tax placeholder.

    A fallback plan may only know that the question concerns PIT or CIT.  Once
    the payment-channel mechanism is identified, retaining that placeholder as
    a second issue creates a broad competing retrieval lane and dilutes the
    controlling evidence bundle.
    """

    issue_domains = {value.upper() for value in issue.tax_domains}
    if issue_domains and tax_domain not in issue_domains:
        return False
    marker = " ".join(
        (issue.issue_id, issue.label, issue.legal_mechanism)
    ).casefold()
    return issue.legal_mechanism.casefold() == "general_tax_analysis" or any(
        value in marker
        for value in ("general_tax", "general tax", "ogólna analiza podatkowa")
    )


def _authority_queries(tax_domain: str) -> list[QueryFamily]:
    tax_article = "art. 22p PIT" if tax_domain == "PIT" else "art. 15d CIT"
    return [
        QueryFamily(
            family="fact_signature",
            query=(
                f"{tax_article} art. 19 Prawo przedsiębiorców płatność gotówką "
                "jednorazowa wartość transakcji raty rachunek płatniczy"
            ),
            lane="authority",
            origin="fallback",
        ),
        QueryFamily(
            family="legal_concept",
            query=(
                f"{tax_article} zwrot gotówki ponowne uregulowanie zobowiązania "
                "przelewem korekta kosztów"
            ),
            lane="authority",
            origin="fallback",
        ),
        QueryFamily(
            family="fact_contrast",
            query=(
                f"{tax_article} dodatkowy przelew bez zwrotu gotówki "
                "nadpłata nie anuluje pierwotnego uregulowania zobowiązania"
            ),
            lane="authority",
            origin="fallback",
        ),
        QueryFamily(
            family="quoted_holding_language",
            query=(
                f"{tax_article} faktyczny zwrot środków anulowanie płatności "
                "ponowne uregulowanie należności za pośrednictwem rachunku"
            ),
            lane="authority",
            origin="fallback",
        ),
    ]


def enrich_cash_payment_cost_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Replace a generic income-cost issue with the payment-channel bundle."""

    if not question_targets_cash_payment_cost_exclusion(question):
        return plan

    tax_domain = cost_tax_domain(question)
    targets = (
        PIT_CASH_PAYMENT_COST_TARGETS
        if tax_domain == "PIT"
        else CIT_CASH_PAYMENT_COST_TARGETS
    )
    issue_id = f"{tax_domain.casefold()}_cash_payment_cost_exclusion"
    label = (
        f"{tax_domain}: płatność poza rachunkiem — wartość transakcji, "
        "wyłączenie kosztu i korekta"
    )
    mechanism = "cash_payment_cost_exclusion"
    authority_queries = _authority_queries(tax_domain)

    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        issue_text = " ".join(
            (issue.issue_id, issue.label, issue.legal_mechanism)
        )
        if (
            issue.issue_id == issue_id
            or _is_cash_payment_issue(issue)
            or _is_replaceable_general_issue(issue, tax_domain)
            or question_targets_cit_cost_deductibility(issue_text)
        ):
            corrected = issue.model_copy(
                update={
                    "issue_id": issue_id,
                    "label": label,
                    "tax_domains": [tax_domain, "PP"],
                    "legal_mechanism": mechanism,
                    "possible_provision_concepts": _dedupe(
                        [f"{domain} {citation}" for domain, citation in targets]
                    ),
                    "transactions": _dedupe(
                        [*issue.transactions, "jednorazowa wartość transakcji", "umowa", "rata"]
                    ),
                    "payments": _dedupe(
                        [*issue.payments, "płatność gotówką", "rachunek płatniczy", "przelew"]
                    ),
                    "positive_fact_constraints": _dedupe(
                        [
                            *issue.positive_fact_constraints,
                            "zwrot gotówki",
                            "ponowne uregulowanie",
                            "anulowanie płatności",
                        ]
                    ),
                    "query_families": [
                        query
                        for query in issue.query_families
                        if query.lane == "authority"
                    ]
                    + authority_queries,
                }
            )
            issues.append(_with_targets(corrected, targets))
            found = True
        else:
            issues.append(issue)

    if not found:
        issue = LegalIssue(
            issue_id=issue_id,
            label=label,
            tax_domains=[tax_domain, "PP"],
            legal_mechanism=mechanism,
            possible_provision_concepts=[
                f"{domain} {citation}" for domain, citation in targets
            ],
            transactions=["jednorazowa wartość transakcji", "umowa", "rata"],
            payments=["płatność gotówką", "rachunek płatniczy", "przelew"],
            positive_fact_constraints=[
                "zwrot gotówki",
                "ponowne uregulowanie",
                "anulowanie płatności",
            ],
            requested_source_types=["statute", "interpretation", "judgment"],
            query_families=authority_queries,
            priority="high",
        )
        issues.append(_with_targets(issue, targets))

    return plan.model_copy(update={"issues": issues})


__all__ = [
    "BUSINESS_PAYMENT_TARGETS",
    "CIT_CASH_PAYMENT_COST_TARGETS",
    "PIT_CASH_PAYMENT_COST_TARGETS",
    "enrich_cash_payment_cost_plan",
    "question_targets_cash_payment_cost_exclusion",
]
