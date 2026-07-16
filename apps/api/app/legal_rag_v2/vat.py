"""Deterministic research routing for ordinary input-VAT timing questions.

This module selects the statutory bundle only.  It does not decide whether
the taxpayer may deduct VAT in the concrete case; claim synthesis or the
best-effort writer still performs that application.
"""

from __future__ import annotations

import re

from .family_foundation import _dedupe, _with_targets
from .schemas import LegalIssue, LegalResearchPlan, QueryFamily


VAT_INPUT_DEDUCTION_TIMING_TARGETS = (
    ("VAT", "art. 86 ust. 1"),
    ("VAT", "art. 86 ust. 2 pkt 1"),
    ("VAT", "art. 86 ust. 10"),
    ("VAT", "art. 86 ust. 10b pkt 1"),
    ("VAT", "art. 86 ust. 10e"),
    ("VAT", "art. 86 ust. 11"),
    ("VAT", "art. 86 ust. 13"),
    ("VAT", "art. 19a ust. 1"),
    ("VAT", "art. 106na ust. 3"),
    ("VAT", "art. 106na ust. 4"),
    ("VAT", "art. 106nda ust. 11"),
)


def question_targets_input_vat_deduction_timing(question: str) -> bool:
    has_vat_deduction = bool(
        re.search(
            r"(?:odlicz\w*.{0,40}\bVAT\b|\bVAT\b.{0,40}odlicz\w*|"
            r"podat\w*\s+naliczon\w*|prawo\s+do\s+odliczeni\w*)",
            question,
            re.I,
        )
    )
    has_timing = bool(
        re.search(
            r"(?:w\s+kt[óo]r\w*\s+okres|moment\w*\s+odliczeni|kiedy\s+.*odlicz|"
            r"data\s+otrzymani\w*\s+faktur|otrzyma\w*.{0,30}faktur|"
            r"deklaracj\w*\s+za|kolejn\w*\s+okres|miesi[ąa]c\w*)",
            question,
            re.I,
        )
    )
    return has_vat_deduction and has_timing


def _is_generic_vat_issue(issue: LegalIssue) -> bool:
    text = " ".join((issue.issue_id, issue.label, issue.legal_mechanism)).casefold()
    return ("general_tax" in text or "general tax" in text) and "VAT" in {
        domain.upper() for domain in issue.tax_domains
    }


def _is_input_vat_timing_issue(issue: LegalIssue) -> bool:
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
    return question_targets_input_vat_deduction_timing(text) or bool(
        re.search(r"input_vat_deduction|vat_deduction_timing", text, re.I)
    )


def enrich_input_vat_deduction_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    if not question_targets_input_vat_deduction_timing(question):
        return plan

    issue_id = "vat_input_deduction_timing"
    label = "VAT: moment i kolejne okresy odliczenia podatku naliczonego"
    mechanism = "input_vat_deduction_timing"
    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        if _is_generic_vat_issue(issue):
            continue
        if issue.issue_id == issue_id or _is_input_vat_timing_issue(issue):
            corrected = issue.model_copy(
                update={
                    "issue_id": issue_id,
                    "label": label,
                    "tax_domains": ["VAT"],
                    "legal_mechanism": mechanism,
                }
            )
            issues.append(_with_targets(corrected, VAT_INPUT_DEDUCTION_TIMING_TARGETS))
            found = True
        else:
            issues.append(issue)

    if not found:
        issue = LegalIssue(
            issue_id=issue_id,
            label=label,
            tax_domains=["VAT"],
            legal_mechanism=mechanism,
            possible_provision_concepts=_dedupe(
                [f"{domain} {citation}" for domain, citation in VAT_INPUT_DEDUCTION_TIMING_TARGETS]
            ),
            requested_source_types=["statute", "interpretation", "judgment"],
            query_families=[
                QueryFamily(
                    family="statutory_concept",
                    query=f"{label}. {question.strip()}",
                    lane="both",
                    origin="fallback",
                )
            ],
            priority="high",
        )
        issues.append(_with_targets(issue, VAT_INPUT_DEDUCTION_TIMING_TARGETS))

    return plan.model_copy(update={"issues": issues})


__all__ = [
    "VAT_INPUT_DEDUCTION_TIMING_TARGETS",
    "enrich_input_vat_deduction_plan",
    "question_targets_input_vat_deduction_timing",
]
