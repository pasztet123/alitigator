"""Deterministic issue enrichment and calculations for cross-border WHT cases.

The planner may describe a multi-payment WHT case too broadly.  These helpers
only add retrieval and calculation structure; they never decide treaty
eligibility, beneficial-owner status, or a final preferential rate.
"""

from __future__ import annotations

import re
from typing import Iterable

from .schemas import (
    CalculationRecord,
    LegalIssue,
    LegalResearchPlan,
    ProvisionReference,
    QueryFamily,
)


# Accept inflected Polish wording used in factual questions (for example
# "podlegają podatkowi u źródła"), not only the dictionary form "podatek".
_WHT_RE = re.compile(
    r"\b(wht|podatk\w*\s+u\s+źr[óo]dła|withholding|pay and refund|"
    r"art\.?(?:\s*)2[16]\s+ust\.?(?:\s*)[12])\b",
    re.I,
)
_GERMANY_RE = re.compile(r"\b(niemc(?:y|zech|ami)|niemieck\w*|pl[- ]?de|polsko[- ]?niemieck\w*)\b", re.I)
_INTEREST_RE = re.compile(r"\b(odsetk\w*|interest\w*)\b", re.I)
_ROYALTY_RE = re.compile(r"\b(licencj\w*|należno\w* licencyjn\w*|royalt(?:y|ies))\b", re.I)
_SERVICES_RE = re.compile(r"\b(usług\w*|uslug\w*|zarządz\w*|zarzadz\w*|doradcz\w*|management services?)\b", re.I)
_PAY_REFUND_RE = re.compile(r"\b(pay and refund|2\s*(?:mln|000\s*000)|próg\w*|prog\w*)\b", re.I)
_AMOUNT_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[\s,.]\d{3})+|\d+(?:[,.]\d+)?\s*(?:mln|milion(?:y|ów)?|tys\.?|tysi(?:ąc|ace|ęcy)))\s*(?:zł|pln)?\b",
    re.I,
)


def is_poland_germany_wht_case(question: str) -> bool:
    return bool(_WHT_RE.search(question or "")) and bool(_GERMANY_RE.search(question or ""))


def enrich_crossborder_wht_plan(
    plan: LegalResearchPlan,
    question: str,
) -> LegalResearchPlan:
    """Add issue-scoped primary/authority queries for a PL-DE WHT case."""
    if not is_poland_germany_wht_case(question):
        return plan

    # The rule fallback emits broad WHT issues before this deterministic
    # classifier runs. Keeping them alongside the issue-per-payment bundles
    # doubles every primary/authority lane and can exhaust the request budget.
    # The scoped bundles below fully replace those generic cross-border WHT
    # questions, while unrelated issues remain intact.
    replaced_generic_issue_ids = {
        "wht_interest",
        "wht_management_services",
        "pay_and_refund",
        "interest_royalties_exemption",
        "beneficial_owner",
        "poland_germany_treaty",
    }
    issues = [
        issue for issue in plan.issues
        if issue.issue_id not in replaced_generic_issue_ids
    ]
    existing = {issue.issue_id for issue in issues}

    def add(
        issue_id: str,
        label: str,
        tax_domains: list[str],
        mechanism: str,
        source_types: list[str],
        queries: Iterable[tuple[str, str, str]],
    ) -> None:
        if issue_id in existing:
            return
        families = [
            QueryFamily(family=family, query=query, lane=lane, origin="fallback")
            for family, query, lane in queries
        ]
        issues.append(
            LegalIssue(
                issue_id=issue_id,
                label=label,
                tax_domains=tax_domains,
                legal_mechanism=mechanism,
                payments=[label],
                jurisdictions=["Polska", "Niemcy"],
                possible_provision_concepts=[item.query for item in families],
                requested_source_types=source_types,
                query_families=families,
                priority="high",
            )
        )
        existing.add(issue_id)

    if _INTEREST_RE.search(question):
        add(
            "wht_interest_pl_de_treaty",
            "WHT / UPO PL-DE: odsetki",
            ["CIT"],
            "withholding_interest_treaty_rate",
            ["statute", "tax_treaty", "interpretation", "guidance", "judgment"],
            [
                ("explicit_provision_reference", "UPO Polska Niemcy art. 11 odsetki stawka beneficial owner", "primary_law"),
                ("statutory_concept", "CIT art. 21 ust. 1 pkt 1 odsetki WHT", "primary_law"),
                ("authority_backreference", "MF beneficial owner odsetki WHT objaśnienia podatkowe", "authority"),
            ],
        )
        add(
            "vat_interest_financial_service",
            "VAT: odsetki / usługa finansowa transgraniczna",
            ["VAT"],
            "crossborder_financial_service_vat",
            ["statute", "interpretation", "judgment"],
            [
                ("explicit_provision_reference", "VAT art. 28b ust. 1 miejsce świadczenia odsetki", "primary_law"),
                ("explicit_provision_reference", "VAT art. 17 ust. 1 pkt 4 import usług", "primary_law"),
                ("explicit_provision_reference", "VAT art. 43 ust. 1 pkt 38 zwolnienie usługi finansowe", "primary_law"),
            ],
        )
    if _ROYALTY_RE.search(question):
        add(
            "wht_royalties_pl_de_treaty",
            "WHT / UPO PL-DE: należności licencyjne",
            ["CIT"],
            "withholding_royalty_treaty_rate",
            ["statute", "tax_treaty", "interpretation", "guidance", "judgment"],
            [
                ("explicit_provision_reference", "UPO Polska Niemcy art. 12 należności licencyjne stawka", "primary_law"),
                ("statutory_concept", "CIT art. 21 ust. 1 pkt 1 należności licencyjne WHT", "primary_law"),
                ("authority_backreference", "WHT licencje software beneficial owner interpretacja wyrok", "authority"),
            ],
        )
        add(
            "vat_royalty_crossborder_service",
            "VAT: licencja transgraniczna",
            ["VAT"],
            "crossborder_royalty_service_vat",
            ["statute", "interpretation", "judgment"],
            [
                ("explicit_provision_reference", "VAT art. 28b ust. 1 miejsce świadczenia licencji", "primary_law"),
                ("explicit_provision_reference", "VAT art. 17 ust. 1 pkt 4 import usług licencja", "primary_law"),
            ],
        )
    if _SERVICES_RE.search(question):
        add(
            "wht_services_pl_de_business_profits",
            "WHT / UPO PL-DE: usługi zarządzania",
            ["CIT"],
            "withholding_management_services_business_profits",
            ["statute", "tax_treaty", "interpretation", "guidance", "judgment"],
            [
                ("explicit_provision_reference", "UPO Polska Niemcy art. 7 zyski przedsiębiorstw zakład usługi zarządzania", "primary_law"),
                ("statutory_concept", "CIT art. 21 ust. 1 pkt 2a usługi zarządzania WHT", "primary_law"),
                ("authority_backreference", "cash pool usługi zarządzania WHT interpretacja wyrok", "authority"),
            ],
        )
        add(
            "vat_management_crossborder_service",
            "VAT: usługi zarządzania transgraniczne",
            ["VAT"],
            "crossborder_management_service_vat",
            ["statute", "interpretation", "judgment"],
            [
                ("explicit_provision_reference", "VAT art. 28b ust. 1 miejsce świadczenia usługi zarządzania", "primary_law"),
                ("explicit_provision_reference", "VAT art. 17 ust. 1 pkt 4 import usług zarządzania", "primary_law"),
            ],
        )
    if _PAY_REFUND_RE.search(question):
        add(
            "wht_pay_and_refund_procedure",
            "WHT: pay and refund / procedura płatnika",
            ["CIT"],
            "pay_and_refund_procedure",
            ["statute", "interpretation", "guidance", "judgment"],
            [
                ("explicit_provision_reference", "CIT art. 26 ust. 2e próg 2 000 000", "primary_law"),
                ("explicit_provision_reference", "CIT art. 26 ust. 2g oświadczenie płatnika", "primary_law"),
                ("explicit_provision_reference", "CIT art. 26 ust. 7a 7b 7c oświadczenie termin", "primary_law"),
                ("explicit_provision_reference", "CIT art. 26b opinia o stosowaniu preferencji", "primary_law"),
                ("explicit_provision_reference", "CIT art. 28b zwrot podatku", "primary_law"),
                ("authority_backreference", "pay and refund beneficial owner cash pool objaśnienia MF", "authority"),
            ],
        )

    payload = plan.model_dump(mode="python")
    payload["issues"] = issues
    payload["intent"]["needs_calculations"] = True
    return LegalResearchPlan.model_validate(payload)


def _parse_amount(raw: str) -> int | None:
    value = raw.strip().lower().replace("zł", "").replace("pln", "").strip()
    multiplier = 1
    if re.search(r"mln|milion", value):
        multiplier = 1_000_000
        value = re.sub(r"\s*(?:mln|milion(?:y|ów)?)\b", "", value).strip()
    elif re.search(r"tys", value):
        multiplier = 1_000
        value = re.sub(r"\s*(?:tys\.?|tysi(?:ąc|ace|ęcy))\b", "", value).strip()
    if multiplier == 1:
        value = re.sub(r"[\s,.]", "", value)
    else:
        value = value.replace(" ", "").replace(",", ".")
    try:
        return int(round(float(value) * multiplier))
    except ValueError:
        return None


def _aggregate_payment_amount(question: str) -> int | None:
    normalized = " ".join(question.split())
    explicit_total = re.search(
        r"(?:łącznie|lacznie|suma|razem|w łącznej kwocie|w lacznej kwocie).{0,45}?"
        r"(?P<amount>\d{1,3}(?:[\s,.]\d{3})+|\d+(?:[,.]\d+)?\s*(?:mln|milion(?:y|ów)?|tys\.?|tysi(?:ąc|ace|ęcy)))",
        normalized,
        re.I,
    )
    if explicit_total:
        return _parse_amount(explicit_total.group("amount"))
    amounts: list[int] = []
    for match in _AMOUNT_RE.finditer(normalized):
        amount = _parse_amount(match.group(1))
        context = normalized[max(0, match.start() - 35) : match.end() + 35].lower()
        if amount and not re.search(r"próg|prog|limit", context):
            amounts.append(amount)
    if not amounts:
        return None
    # A stated aggregate is normally the largest monetary value; summing it
    # with its components would double-count the tax base.
    return max(amounts) if max(amounts) >= 1_000_000 else sum(amounts)


class WhtPayAndRefundCalculationEngine:
    threshold = 2_000_000
    domestic_rate = 0.20

    def calculate(
        self,
        plan: LegalResearchPlan,
        bundles: list,
    ) -> list[CalculationRecord]:
        if not any(item.issue_id == "wht_pay_and_refund_procedure" for item in plan.issues):
            return []
        total = _aggregate_payment_amount(plan.user_query)
        if total is None:
            return []
        procedure_bundle = next(
            (item for item in bundles if item.issue_id == "wht_pay_and_refund_procedure"),
            None,
        )
        if procedure_bundle is None:
            return []
        provisions = [
            item
            for item in (
                *procedure_bundle.controlling_provisions,
                *procedure_bundle.dependency_provisions,
                *procedure_bundle.exception_provisions,
            )
            if re.search(r"art\.\s*26\s+ust\.\s*2e", item.citation, re.I)
        ]
        if not provisions:
            return []
        excess = max(0, total - self.threshold)
        domestic_wht = int(round(excess * self.domestic_rate))
        return [
            CalculationRecord(
                calculation_id="wht_pay_and_refund_domestic_wht",
                inputs={
                    "aggregate_payments": total,
                    "threshold_base": self.threshold,
                    "excess": excess,
                    "domestic_rate": self.domestic_rate,
                    "domestic_wht": domestic_wht,
                },
                units={
                    "aggregate_payments": "PLN",
                    "threshold_base": "PLN",
                    "excess": "PLN",
                    "domestic_rate": "fraction",
                    "domestic_wht": "PLN",
                },
                operation="pay_and_refund_domestic_wht_on_excess",
                formula="max(0, aggregate_payments - 2_000_000) × 20%",
                result=domestic_wht,
                rounding="nearest PLN",
                legal_basis=provisions,
                dependencies=["wht_pay_and_refund_procedure"],
            )
        ]
