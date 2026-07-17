"""Deterministic research routing for ordinary input-VAT timing questions.

This module selects the statutory bundle only.  It does not decide whether
the taxpayer may deduct VAT in the concrete case; claim synthesis or the
best-effort writer still performs that application.
"""

from __future__ import annotations

import re

from .family_foundation import _dedupe, _with_targets
from .schemas import LegalIssue, LegalResearchPlan, MissingFact, QueryFamily


VAT_INPUT_DEDUCTION_TIMING_TARGETS = (
    ("VAT", "art. 86 ust. 1"),
    ("VAT", "art. 86 ust. 2 pkt 1"),
    ("VAT", "art. 86 ust. 10"),
    ("VAT", "art. 86 ust. 10b pkt 1"),
    ("VAT", "art. 86 ust. 10e"),
    ("VAT", "art. 86 ust. 11"),
    ("VAT", "art. 86 ust. 13"),
    ("VAT", "art. 19a ust. 1"),
)

VAT_2026_INVOICE_CHANNEL_TARGETS = (
    ("VAT", "art. 106ga ust. 1"),
    ("VAT", "art. 106ga ust. 2 pkt 1"),
    ("VAT", "art. 106ga ust. 2 pkt 2"),
    ("VAT", "art. 106ga ust. 2 pkt 3"),
    ("VAT", "art. 106ga ust. 2 pkt 4"),
    ("VAT", "art. 106ga ust. 2 pkt 5"),
    ("VAT", "art. 106ga ust. 2 pkt 6"),
    ("VAT", "art. 145m ust. 1"),
    ("VAT", "art. 145m ust. 2"),
    ("VAT", "art. 106na ust. 3"),
    ("VAT", "art. 106na ust. 4"),
    ("VAT", "art. 106nda ust. 11"),
    ("VAT", "art. 106nf ust. 10"),
    ("VAT", "art. 106nh ust. 4"),
    ("VAT", "art. 106ng"),
)

VAT_MIXED_USE_VEHICLE_TARGETS = (
    ("VAT", "art. 86 ust. 1"),
    ("VAT", "art. 86a ust. 1"),
    ("VAT", "art. 86a ust. 2 pkt 3"),
    ("VAT", "art. 86a ust. 3 pkt 1 lit. a"),
    ("VAT", "art. 86a ust. 4 pkt 1"),
    ("VAT", "art. 86a ust. 6"),
    ("VAT", "art. 86a ust. 12"),
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


def question_targets_mixed_use_vehicle_vat(question: str) -> bool:
    has_vehicle = bool(
        re.search(r"samoch[oó]d\w*|pojazd\w*|motocykl\w*|auto\b", question, re.I)
    )
    has_deduction = bool(
        re.search(
            r"odlicz\w*.{0,50}\bVAT\b|\bVAT\b.{0,50}odlicz\w*|"
            r"podat\w*\s+naliczon\w*",
            question,
            re.I,
        )
    )
    has_vehicle_expense = bool(
        re.search(
            r"paliw\w*|wydatk\w*\s+eksploatacyjn\w*|użytk\w*\s+mieszan\w*|"
            r"prywatn\w*.{0,50}(?:samoch[oó]d|pojazd|auto)|"
            r"(?:samoch[oó]d|pojazd|auto).{0,50}prywatn\w*|VAT-?26|"
            r"ewidencj\w*\s+przebieg\w*",
            question,
            re.I,
        )
    )
    return has_vehicle and has_deduction and has_vehicle_expense


def _is_mixed_use_vehicle_issue(issue: LegalIssue) -> bool:
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
    return bool(
        re.search(r"mixed_use_vehicle_vat|vehicle_vat_deduction", text, re.I)
        or question_targets_mixed_use_vehicle_vat(text)
    )


def enrich_mixed_use_vehicle_vat_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Route vehicle expenses by the art. 86a mechanism, not ownership form."""

    if not question_targets_mixed_use_vehicle_vat(question):
        return plan

    issue_id = "mixed_use_vehicle_vat"
    label = "VAT: pojazd używany mieszanie — 50%, paliwo i warunki pełnego odliczenia"
    mechanism = "mixed_use_vehicle_vat_deduction"
    authority_queries = [
        QueryFamily(
            family="fact_signature",
            query=(
                "prawo odliczyć 50% podatku VAT od faktur leasing samochodu "
                "osobowego używanego przy prowadzeniu działalności gospodarczej "
                "oraz prywatnie"
            ),
            lane="authority",
            origin="fallback",
        ),
        QueryFamily(
            family="fact_contrast",
            query=(
                "prawo do odliczenia podatku VAT z tytułu nabytego motocykla "
                "50% wydatki eksploatacyjne paliwo art. 86a"
            ),
            lane="authority",
            origin="fallback",
        ),
        QueryFamily(
            family="factual_contrast",
            query=(
                "100% VAT pojazd wyłącznie działalność gospodarcza wykluczenie "
                "użytku prywatnego ewidencja przebiegu VAT-26 art. 86a"
            ),
            lane="authority",
            origin="fallback",
        ),
    ]
    issue_update = {
        "issue_id": issue_id,
        "label": label,
        "tax_domains": ["VAT"],
        "legal_mechanism": mechanism,
        "transactions": [
            "pojazd samochodowy",
            "samochód osobowy",
            "wydatki eksploatacyjne",
            "paliwo",
        ],
        "positive_fact_constraints": [
            "użytkowanie mieszane",
            "cele prywatne",
            "50% podatku naliczonego",
            "bez VAT-26",
            "bez ewidencji przebiegu",
        ],
        "possible_legal_concepts": [
            "mixed_use_vehicle",
            "fifty_percent_input_vat",
            "fuel_as_vehicle_expense",
            "exclusive_business_use",
            "actual_private_use",
            "vat_26_and_mileage_records",
        ],
        "requested_source_types": ["statute", "interpretation", "judgment"],
        "query_families": authority_queries,
        "priority": "high",
    }

    issues: list[LegalIssue] = []
    found = False
    for issue in plan.issues:
        if _is_generic_vat_issue(issue):
            continue
        if issue.issue_id == issue_id or _is_mixed_use_vehicle_issue(issue):
            corrected = issue.model_copy(update=issue_update)
            issues.append(_with_targets(corrected, VAT_MIXED_USE_VEHICLE_TARGETS))
            found = True
        else:
            issues.append(issue)

    if not found:
        issue = LegalIssue(**issue_update)
        issues.append(_with_targets(issue, VAT_MIXED_USE_VEHICLE_TARGETS))
    return plan.model_copy(update={"issues": issues})


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


def _needs_2026_ksef_lane(plan: LegalResearchPlan, question: str) -> bool:
    if re.search(r"\b202[0-5]\b", question):
        return False
    if re.search(r"\b20(?:2[6-9]|[3-9]\d)\b", question):
        return True
    return bool(plan.target_date and plan.target_date >= "2026-02-01")


def _with_missing_fact(
    missing_facts: list[MissingFact],
    *,
    fact_id: str,
    question: str,
    materiality: str = "outcome_determinative",
) -> list[MissingFact]:
    if any(item.fact_id == fact_id for item in missing_facts):
        return missing_facts
    return [
        *missing_facts,
        MissingFact(fact_id=fact_id, question=question, materiality=materiality),
    ]


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

    missing_facts = list(plan.missing_facts)
    missing_facts = _with_missing_fact(
        missing_facts,
        fact_id="vat_cash_method_status",
        question=(
            "Czy nabywca rozlicza VAT metodą kasową albo faktura dokumentuje "
            "transakcję objętą metodą kasową?"
        ),
        materiality="retrieval_relevant",
    )

    if _needs_2026_ksef_lane(plan, question):
        ksef_issue_id = "vat_invoice_channel_2026"
        if not any(issue.issue_id == ksef_issue_id for issue in issues):
            ksef_issue = LegalIssue(
                issue_id=ksef_issue_id,
                label="VAT/KSeF 2026: kanał wystawienia i data otrzymania faktury",
                tax_domains=["VAT"],
                legal_mechanism="invoice_delivery_channel_classification",
                possible_provision_concepts=_dedupe(
                    [
                        f"{domain} {citation}"
                        for domain, citation in VAT_2026_INVOICE_CHANNEL_TARGETS
                    ]
                ),
                possible_legal_concepts=[
                    "ksef_online",
                    "ksef_offline24",
                    "ksef_unavailability",
                    "ksef_failure",
                    "legally_outside_ksef",
                    "invoice_channel_unknown",
                ],
                requested_source_types=["statute", "guidance", "interpretation", "judgment"],
                query_families=[
                    QueryFamily(
                        family="fact_contrast",
                        query=(
                            "KSeF 2026 data otrzymania faktury online offline24 "
                            "niedostępność awaria legalnie poza KSeF limit 10 000 zł. "
                            + question.strip()
                        ),
                        lane="both",
                        origin="fallback",
                    )
                ],
                priority="high",
            )
            issues.insert(0, _with_targets(ksef_issue, VAT_2026_INVOICE_CHANNEL_TARGETS))
        missing_facts = _with_missing_fact(
            missing_facts,
            fact_id="invoice_delivery_channel",
            question=(
                "Czy faktura była wystawiona online w KSeF, w trybie offline24, "
                "podczas niedostępności/awarii, czy legalnie poza KSeF?"
            ),
        )
        missing_facts = _with_missing_fact(
            missing_facts,
            fact_id="ksef_number_assignment_date",
            question="Kiedy KSeF przydzielił fakturze numer identyfikujący?",
        )
        missing_facts = _with_missing_fact(
            missing_facts,
            fact_id="seller_ksef_exception_status",
            question=(
                "Czy wystawca lub transakcja korzystali z ustawowego wyłączenia "
                "albo przejściowego limitu 10 000 zł brutto miesięcznie?"
            ),
        )

    return plan.model_copy(update={"issues": issues, "missing_facts": missing_facts})


__all__ = [
    "VAT_INPUT_DEDUCTION_TIMING_TARGETS",
    "VAT_2026_INVOICE_CHANNEL_TARGETS",
    "VAT_MIXED_USE_VEHICLE_TARGETS",
    "enrich_mixed_use_vehicle_vat_plan",
    "enrich_input_vat_deduction_plan",
    "question_targets_input_vat_deduction_timing",
    "question_targets_mixed_use_vehicle_vat",
]
