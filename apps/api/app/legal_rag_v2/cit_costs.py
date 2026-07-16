"""Research-plan safeguards for CIT cost-deductibility mechanisms.

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


def _is_generic_cit_issue(issue: LegalIssue) -> bool:
    text = " ".join((issue.issue_id, issue.label, issue.legal_mechanism)).casefold()
    return "general_tax" in text or "general tax" in text


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
    """Ensure concrete CIT expense questions have a scoped primary-law lane."""

    if not question_targets_cit_cost_deductibility(question):
        return plan

    penalty = question_targets_contractual_penalty_cost(question)
    targets = CIT_CONTRACTUAL_PENALTY_TARGETS if penalty else CIT_COST_BASE_TARGETS
    issue_id = "cit_contractual_penalty_cost" if penalty else "cit_cost_deductibility"
    label = (
        "CIT: kara umowna — koszt podatkowy i ustawowe wyłączenie"
        if penalty
        else "CIT: koszt uzyskania przychodów i ustawowe wyłączenia"
    )
    mechanism = "contractual_penalty_cost" if penalty else "cit_cost_deductibility"

    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        if _is_generic_cit_issue(issue):
            continue
        if issue.issue_id == issue_id or _is_cit_cost_issue(issue):
            issues.append(_with_targets(issue, targets))
            found = True
        else:
            issues.append(issue)

    if not found:
        authority_query = label
        if penalty:
            authority_query += (
                " opóźnienie dostawy wady towarów zwłoka w usunięciu wad "
                "należyta staranność związek z przychodem"
            )
        issue = LegalIssue(
            issue_id=issue_id,
            label=label,
            tax_domains=["CIT"],
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
    "enrich_cit_cost_plan",
    "question_targets_cit_cost_deductibility",
    "question_targets_contractual_penalty_cost",
]
