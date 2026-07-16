"""Research-plan safeguards for income-tax cost-deductibility mechanisms.

The model remains responsible for the legal conclusion.  This module only
prevents a concrete cost question from degrading to an unscoped ``CIT`` query
and binds the general cost rule together with any explicitly detected
statutory exclusion.
"""

from __future__ import annotations

import re

from .family_foundation import _dedupe, _with_targets
from .schemas import LegalIssue, LegalResearchPlan, QueryFamily


CIT_COST_BASE_TARGETS = (
    ("CIT", "art. 15 ust. 1"),
    ("CIT", "art. 16 ust. 1"),
)

CIT_CONTRACTUAL_PENALTY_TARGETS = (
    *CIT_COST_BASE_TARGETS,
    ("CIT", "art. 16 ust. 1 pkt 22"),
)

PIT_COST_BASE_TARGETS = (
    ("PIT", "art. 22 ust. 1"),
    ("PIT", "art. 23 ust. 1"),
)


def question_targets_cit_cost_deductibility(question: str) -> bool:
    return bool(
        re.search(
            r"(?:koszt\w*\s+uzyskani\w*\s+przychod|koszt\w*\s+podatkow|"
            r"zalicz\w*.{0,100}\s+do\s+koszt|potr[ąa]calno[śs][ćc]\s+koszt)",
            question,
            re.IGNORECASE,
        )
    )


def question_targets_contractual_penalty_cost(question: str) -> bool:
    return question_targets_cit_cost_deductibility(question) and bool(
        re.search(r"kar\w*\s+umown|odszkodowan", question, re.IGNORECASE)
    )


def cost_tax_domain(question: str) -> str:
    """Select the income-tax act from the taxpayer described by the user.

    Explicit tax names win.  Otherwise a natural person/JDG is PIT and a
    company is CIT.  CIT remains the conservative default for old callers
    that pass only an abstract cost label.
    """

    if re.search(r"\bPIT\b|podat\w*\s+dochodow\w*\s+od\s+os[óo]b\s+fizycz", question, re.I):
        return "PIT"
    if re.search(r"\bCIT\b|podat\w*\s+dochodow\w*\s+od\s+os[óo]b\s+prawn", question, re.I):
        return "CIT"
    if re.search(
        r"jednoosobow\w*\s+działalno\w*|\bJDG\b|osob\w*\s+fizycz\w*|"
        r"prowadz[ęe]\s+(?:własn\w*\s+)?działalno\w*|moj\w*\s+działalno\w*",
        question,
        re.I,
    ):
        return "PIT"
    return "CIT"


def _is_generic_income_tax_issue(issue: LegalIssue) -> bool:
    text = " ".join((issue.issue_id, issue.label, issue.legal_mechanism)).casefold()
    return ("general_tax" in text or "general tax" in text) and bool(
        {item.upper() for item in issue.tax_domains} & {"CIT", "PIT"}
    )


def _is_cit_cost_issue(issue: LegalIssue) -> bool:
    text = " ".join(
        (
            issue.issue_id,
            issue.label,
            issue.legal_mechanism,
            *issue.possible_provision_concepts,
            *issue.possible_legal_concepts,
            *issue.possible_provision_hints,
        )
    )
    return question_targets_cit_cost_deductibility(text)


def enrich_cit_cost_plan(plan: LegalResearchPlan, question: str) -> LegalResearchPlan:
    """Ensure concrete income-tax expense questions have a scoped law lane."""

    if not question_targets_cit_cost_deductibility(question):
        return plan

    tax_domain = cost_tax_domain(question)
    penalty = tax_domain == "CIT" and question_targets_contractual_penalty_cost(question)
    targets = (
        CIT_CONTRACTUAL_PENALTY_TARGETS
        if penalty
        else PIT_COST_BASE_TARGETS
        if tax_domain == "PIT"
        else CIT_COST_BASE_TARGETS
    )
    issue_id = (
        "cit_contractual_penalty_cost"
        if penalty
        else f"{tax_domain.casefold()}_cost_deductibility"
    )
    label = (
        "CIT: kara umowna — koszt podatkowy i ustawowe wyłączenie"
        if penalty
        else f"{tax_domain}: koszt uzyskania przychodów i ustawowe wyłączenia"
    )
    mechanism = "contractual_penalty_cost" if penalty else issue_id

    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        if _is_generic_income_tax_issue(issue):
            continue
        if issue.issue_id == issue_id or _is_cit_cost_issue(issue):
            other_domain = "CIT" if tax_domain == "PIT" else "PIT"
            corrected_issue = issue.model_copy(
                update={
                    "issue_id": issue_id,
                    "label": label,
                    "tax_domains": [tax_domain],
                    "legal_mechanism": mechanism,
                    "possible_provision_concepts": [
                        concept
                        for concept in issue.possible_provision_concepts
                        if not concept.upper().startswith(f"{other_domain} ")
                    ],
                    "query_families": [
                        query
                        for query in issue.query_families
                        if not (
                            query.lane == "primary_law"
                            and query.query.upper().startswith(f"{other_domain} ")
                        )
                    ],
                }
            )
            issues.append(_with_targets(corrected_issue, targets))
            found = True
        else:
            issues.append(issue)

    if not found:
        # Secondary-law retrieval must retain the concrete subject (glasses,
        # insurance, a penalty, etc.); querying only the abstract cost rule
        # retrieves generic and often irrelevant authorities.
        authority_query = f"{label}. {question.strip()}"
        if penalty:
            authority_query += (
                " opóźnienie dostawy wady towarów zwłoka w usunięciu wad "
                "należyta staranność związek z przychodem"
            )
        issue = LegalIssue(
            issue_id=issue_id,
            label=label,
            tax_domains=[tax_domain],
            legal_mechanism=mechanism,
            possible_provision_concepts=_dedupe(
                [f"{domain} {citation}" for domain, citation in targets]
            ),
            requested_source_types=["statute", "interpretation", "judgment"],
            query_families=[
                QueryFamily(
                    family="statutory_concept",
                    query=authority_query,
                    lane="both",
                    origin="fallback",
                )
            ],
            priority="high",
        )
        issues.append(_with_targets(issue, targets))

    return plan.model_copy(update={"issues": issues})


__all__ = [
    "CIT_CONTRACTUAL_PENALTY_TARGETS",
    "CIT_COST_BASE_TARGETS",
    "PIT_COST_BASE_TARGETS",
    "cost_tax_domain",
    "enrich_cit_cost_plan",
    "question_targets_cit_cost_deductibility",
    "question_targets_contractual_penalty_cost",
]
