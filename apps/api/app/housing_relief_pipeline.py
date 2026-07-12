from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from app.controlled_legal_pipeline import (
    LegalPipelineResult,
    bind_authority_cards_to_claims,
    build_renderer_payload,
    render_answer,
    validate_rendered_answer,
)
from app.legal_pipeline import (
    CalculationRecord,
    FactRecord,
    LegalClaim,
    ProvisionRecord,
    ProvisionRegistry,
    validate_claim,
)


HOUSING_RELIEF_BENCHMARK_QUERY = """
Ulga mieszkaniowa po sprzedaży mieszkania w 2025 r. Przychód wynosi 900 000 zł,
dochód wynosi 300 000 zł. Podatnik spłacił 300 000 zł kredytu zaciągniętego
na sprzedane mieszkanie i wpłacił 300 000 zł deweloperowi na nowe mieszkanie.
Przeniesienie własności nowego lokalu ma nastąpić w 2029 r. Oceń art. 21
ust. 1 pkt 131, art. 21 ust. 25 pkt 2, art. 21 ust. 25a, art. 21 ust. 30,
art. 21 ust. 30a oraz podatek z art. 30e.
"""

MONEY_PATTERN = (
    r"(?P<amount>\d+(?:[ .]\d{3})*(?:[,.]\d+)?)\s*"
    r"(?P<scale>tys\.?|tysi(?:ą|a)c\w*|tysi[eę]cy|mln|milion\w*)?\s*zł"
)
MONEY_RE = re.compile(MONEY_PATTERN, re.IGNORECASE)
HOUSING_EXPENSE_PERIOD_YEARS = 3


def is_housing_relief_query(query: str) -> bool:
    normalized = query.lower()
    has_housing_relief = bool(
        re.search(r"ulg\w*\s+mieszkaniow\w*|art\.\s*21\s*ust\.\s*1\s*pkt\s*131", normalized)
    )
    has_real_estate_sale = bool(
        re.search(r"(?:sprzeda\w*|zby\w*).{0,80}(?:mieszka\w*|lokal\w*|nieruchomo\w*)", normalized, re.DOTALL)
        or re.search(r"(?:mieszka\w*|lokal\w*|nieruchomo\w*).{0,80}(?:sprzeda\w*|zby\w*)", normalized, re.DOTALL)
    )
    has_amount_context = bool(
        re.search(r"przych[oó]d\w*|doch[oó]d\w*|art\.\s*30e", normalized)
        or (MONEY_RE.search(query) and re.search(r"sprzeda\w*|zby\w*|naby\w*|kupi\w*", normalized))
    )
    has_housing_expense = bool(
        re.search(r"deweloper\w*|przeniesieni\w*\s+własnoś\w*|przeniesieni\w*\s+wlasnos\w*|kredyt\w*", normalized)
    )
    return (
        (has_housing_relief or has_real_estate_sale)
        and has_amount_context
        and has_housing_expense
    )


def _parse_money_match(match: re.Match[str]) -> int:
    amount = match.group("amount").replace(" ", "")
    scale = (match.group("scale") or "").lower()
    if "," in amount:
        amount = amount.replace(".", "").replace(",", ".")
    elif "." in amount and not scale:
        amount = amount.replace(".", "")
    value = Decimal(amount)
    if scale.startswith(("tys", "tysi")):
        value *= Decimal(1000)
    elif scale.startswith(("mln", "milion")):
        value *= Decimal(1_000_000)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _optional_money_after(label: str, query: str, *, max_chars: int = 80) -> int | None:
    match = re.search(
        rf"{label}.{{0,{max_chars}}}?{MONEY_PATTERN}",
        query,
        re.IGNORECASE | re.DOTALL,
    )
    return _parse_money_match(match) if match else None


def _optional_money_before(label: str, query: str, *, max_chars: int = 80) -> int | None:
    match = re.search(
        rf"{MONEY_PATTERN}.{{0,{max_chars}}}?{label}",
        query,
        re.IGNORECASE | re.DOTALL,
    )
    return _parse_money_match(match) if match else None


def _optional_money_in_context(
    query: str,
    *,
    before_pattern: str = "",
    after_pattern: str = "",
    before_chars: int = 100,
    after_chars: int = 100,
) -> int | None:
    for match in MONEY_RE.finditer(query):
        before = query[max(0, match.start() - before_chars) : match.start()].lower()
        after = query[match.end() : match.end() + after_chars].lower()
        if before_pattern and not re.search(before_pattern, before, re.IGNORECASE):
            continue
        if after_pattern and not re.search(after_pattern, after, re.IGNORECASE):
            continue
        return _parse_money_match(match)
    return None


def _money_after(label: str, query: str, *, max_chars: int = 80) -> int:
    value = _optional_money_after(label, query, max_chars=max_chars)
    if value is None:
        raise ValueError(f"Missing value for {label}")
    return value


def _format_money(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _extract_year(query: str, pattern: str) -> int:
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing year for pattern {pattern}")
    return int(match.group(1))


def housing_expense_deadline_for_sale_year(sale_year: int) -> str:
    """Article 21(1)(131): three years counted from the end of sale year."""
    return date(sale_year + HOUSING_EXPENSE_PERIOD_YEARS, 12, 31).isoformat()


def _extract_transfer_date(query: str) -> str:
    """Extract the ownership-transfer date while retaining day-level boundaries."""
    trigger = (
        r"(?:przeniesieni\w*\s+w(?:ł|l)asnoś\w*|"
        r"akt\w*\s+(?:notarialn\w*|przenosz\w*\s+w(?:ł|l)asnoś\w*))"
    )
    iso_match = re.search(rf"{trigger}.{{0,80}}?\b(20\d{{2}}-\d{{2}}-\d{{2}})\b", query, re.I | re.S)
    if iso_match:
        return date.fromisoformat(iso_match.group(1)).isoformat()
    year_match = re.search(rf"{trigger}.{{0,80}}?\b(20\d{{2}})\b", query, re.I | re.S)
    if not year_match:
        raise ValueError("Missing date for ownership transfer")
    return date(int(year_match.group(1)), 12, 31).isoformat()


@dataclass(frozen=True)
class HousingReliefFacts:
    records: dict[str, FactRecord]
    sale_year: int
    sale_year_end: str
    purchase_year: int | None
    revenue: int
    acquisition_cost: int | None
    income: int
    credit_repayment: int
    developer_payment: int
    declared_housing_expenses: int
    qualified_expenses: int
    disqualified_developer_expense: int
    planned_transfer_year: int
    planned_transfer_date: str
    deadline: str


def parse_housing_relief_facts(query: str) -> HousingReliefFacts:
    sale_year = _extract_year(query, r"(?:sprzeda\w*|zby\w*).*?\b(20\d{2})\b")
    purchase_year_match = re.search(r"(?:naby\w*|kupi\w*|zakup\w*).*?\b(20\d{2})\b", query, re.IGNORECASE)
    purchase_year = int(purchase_year_match.group(1)) if purchase_year_match else None
    revenue = (
        _optional_money_after(r"przych[oó]d\w*", query)
        or _optional_money_after(r"(?:sprzeda\w*|zby\w*)", query, max_chars=100)
        or _optional_money_in_context(query, before_pattern=r"(?:sprzeda\w*|zby\w*|sprzedał|sprzedal)", before_chars=80)
    )
    if revenue is None:
        raise ValueError("Missing sale revenue")
    acquisition_cost = (
        _optional_money_after(r"(?:cena\s+nabycia|koszt\w*\s+nabycia|cena\s+zakupu)", query)
        or _optional_money_after(r"(?:naby\w*|kupi\w*)", query, max_chars=60)
        or _optional_money_before(r"(?:cena\s+nabycia|koszt\w*\s+nabycia|cena\s+zakupu)", query)
    )
    explicit_income = _optional_money_after(r"doch[oó]d\w*", query)
    if explicit_income is not None:
        income = explicit_income
    elif acquisition_cost is not None:
        income = revenue - acquisition_cost
    else:
        raise ValueError("Missing income or acquisition cost")
    credit_repayment = (
        _optional_money_after(r"(?:spłat\w*|spłaci\w*|splaci\w*).{0,50}kredyt\w*", query, max_chars=100)
        or _optional_money_after(r"(?:spłat\w*|spłaci\w*|splaci\w*)", query, max_chars=80)
        or _optional_money_in_context(
            query,
            before_pattern=r"(?:spłat\w*|spłaci\w*|splaci\w*)",
            after_pattern=r"kredyt\w*",
        )
        or 0
    )
    developer_payment = (
        _optional_money_after(r"(?:wpłat\w*|wpłaci\w*|wplaci\w*).{0,60}deweloper\w*", query, max_chars=100)
        or _optional_money_after(r"deweloper\w*", query, max_chars=100)
        or _optional_money_in_context(
            query,
            before_pattern=r"(?:wpłat\w*|wpłaci\w*|wplaci\w*)",
            after_pattern=r"deweloper\w*",
        )
        or _optional_money_in_context(query, after_pattern=r"deweloper\w*")
        or 0
    )
    declared_housing_expenses = (
        _optional_money_after(
            r"(?:wydatk\w*\s+mieszkaniow\w*|kwalifikowan\w*\s+wydatk\w*)",
            query,
        )
        or credit_repayment + developer_payment
    )
    planned_transfer_date = _extract_transfer_date(query)
    planned_transfer_year = int(planned_transfer_date[:4])
    sale_year_end = date(sale_year, 12, 31).isoformat()
    deadline = housing_expense_deadline_for_sale_year(sale_year)
    developer_expense_qualifies = planned_transfer_date <= deadline
    disqualified_developer_expense = 0 if developer_expense_qualifies else developer_payment
    # Credit repayment is not a settled input until its statutory conditions
    # are evidenced.  The baseline calculation must therefore exclude it and
    # a separate conditional scenario may add it back.
    qualified_expenses = developer_payment if developer_expense_qualifies else 0
    records = {
        "sale_year": FactRecord("sale_year", "year", sale_year, subject_role="transaction"),
        "sale_year_end": FactRecord(
            "sale_year_end", "date", sale_year_end, date=sale_year_end, subject_role="transaction"
        ),
        "purchase_year": FactRecord("purchase_year", "year", purchase_year, subject_role="transaction") if purchase_year is not None else FactRecord("purchase_year", "year", None, status="missing", subject_role="transaction"),
        "revenue": FactRecord("revenue", "money", revenue, subject_role="transaction"),
        "acquisition_cost": FactRecord("acquisition_cost", "money", acquisition_cost, subject_role="transaction") if acquisition_cost is not None else FactRecord("acquisition_cost", "money", None, status="missing", subject_role="transaction"),
        "income": FactRecord("income", "money", income, subject_role="transaction"),
        "credit_repayment": FactRecord(
            "credit_repayment",
            "money",
            credit_repayment,
            subject_role="transaction",
        ),
        "developer_payment": FactRecord(
            "developer_payment",
            "money",
            developer_payment,
            subject_role="transaction",
        ),
        "declared_housing_expenses": FactRecord(
            "declared_housing_expenses",
            "money",
            declared_housing_expenses,
            subject_role="transaction",
        ),
        "qualified_housing_expenses": FactRecord(
            "qualified_housing_expenses",
            "money",
            qualified_expenses,
            subject_role="transaction",
        ),
        "disqualified_developer_expense": FactRecord(
            "disqualified_developer_expense",
            "money",
            disqualified_developer_expense,
            subject_role="transaction",
        ),
        "planned_transfer_year": FactRecord(
            "planned_transfer_year",
            "year",
            planned_transfer_year,
            subject_role="transaction",
        ),
        "planned_transfer_date": FactRecord(
            "planned_transfer_date",
            "date",
            planned_transfer_date,
            date=planned_transfer_date,
            subject_role="transaction",
        ),
        "housing_expense_deadline": FactRecord(
            "housing_expense_deadline",
            "date",
            deadline,
            date=deadline,
            subject_role="transaction",
        ),
        "credit_not_previously_tax_preferenced": FactRecord(
            "credit_not_previously_tax_preferenced",
            "bool",
            None,
            status="missing",
            subject_role="transaction",
        ),
        "credit_taken_before_sale": FactRecord(
            "credit_taken_before_sale",
            "bool",
            None,
            status="missing",
            subject_role="transaction",
        ),
    }
    return HousingReliefFacts(
        records=records,
        sale_year=sale_year,
        sale_year_end=sale_year_end,
        purchase_year=purchase_year,
        revenue=revenue,
        acquisition_cost=acquisition_cost,
        income=income,
        credit_repayment=credit_repayment,
        developer_payment=developer_payment,
        declared_housing_expenses=declared_housing_expenses,
        qualified_expenses=qualified_expenses,
        disqualified_developer_expense=disqualified_developer_expense,
        planned_transfer_year=planned_transfer_year,
        planned_transfer_date=planned_transfer_date,
        deadline=deadline,
    )


def can_run_housing_relief_pipeline(query: str) -> bool:
    if not is_housing_relief_query(query):
        return False
    try:
        facts = parse_housing_relief_facts(query)
    except ValueError:
        return False
    return (
        facts.revenue > 0
        and facts.income >= 0
        and facts.credit_repayment > 0
        and facts.qualified_expenses >= 0
        and facts.revenue >= facts.income
    )


def _record(
    provision_id: str,
    citation: str,
    text: str,
    *,
    result_codes: tuple[str, ...],
    legal_mechanism: str = "housing_relief_sale",
    rule_relationship: str = "peer",
    related_provisions: tuple[str, ...] = (),
    special_rule_provisions: tuple[str, ...] = (),
    general_rule_provisions: tuple[str, ...] = (),
) -> ProvisionRecord:
    return ProvisionRecord(
        provision_id=provision_id,
        document_id="pit_act",
        version_id="pit_act_2025-01-01",
        citation=citation,
        article=re.search(r"art\.\s*([0-9a-z]+)", citation, re.I).group(1),
        paragraph=re.search(r"ust\.\s*([0-9a-z]+)", citation, re.I).group(1) if re.search(r"ust\.\s*([0-9a-z]+)", citation, re.I) else None,
        point=re.search(r"pkt\s*([0-9a-z]+)", citation, re.I).group(1) if re.search(r"pkt\s*([0-9a-z]+)", citation, re.I) else None,
        letter=re.search(r"lit\.\s*([a-z])", citation, re.I).group(1) if re.search(r"lit\.\s*([a-z])", citation, re.I) else None,
        text=text,
        effective_from="2025-01-01",
        effective_to=None,
        status="active",
        source_document_id="pit_act",
        source_chunk_ids=(provision_id,),
        source_span_end=len(text),
        display_reference=citation,
        tax_domain="PIT",
        taxpayer_role="taxpayer",
        legal_mechanism=legal_mechanism,
        entailed_result_codes=result_codes,
        rule_relationship=rule_relationship,  # type: ignore[arg-type]
        related_provisions=related_provisions,
        special_rule_provisions=special_rule_provisions,
        general_rule_provisions=general_rule_provisions,
    )


def build_housing_relief_registry() -> ProvisionRegistry:
    records = [
        _record(
            "pit_art_10_ust_1_pkt_8",
            "art. 10 ust. 1 pkt 8 ustawy PIT",
            "Odpłatne zbycie nieruchomości przed upływem właściwego terminu stanowi źródło przychodu.",
            result_codes=("sale_tax_regime",),
        ),
        _record(
            "pit_art_21_ust_1_pkt_131",
            "art. 21 ust. 1 pkt 131 ustawy PIT",
            "Wolne od podatku są dochody w wysokości odpowiadającej iloczynowi dochodu i udziału wydatków mieszkaniowych w przychodzie.",
            result_codes=(
                "housing_relief_formula",
                "housing_relief_exempt_income",
                "housing_relief_developer_deadline",
            ),
        ),
        _record(
            "pit_art_21_ust_25_pkt_2",
            "art. 21 ust. 25 pkt 2 ustawy PIT",
            "Za wydatki mieszkaniowe uważa się spłatę kredytu zaciągniętego na cele mieszkaniowe.",
            result_codes=("credit_repayment_qualified", "credit_on_sold_property_qualified"),
            legal_mechanism="housing_relief_credit_repayment",
            special_rule_provisions=("pit_art_21_ust_30a",),
        ),
        _record(
            "pit_art_21_ust_25a",
            "art. 21 ust. 25a ustawy PIT",
            "Wydatki na nabycie od dewelopera wymagają nabycia własności w ustawowym terminie.",
            result_codes=("housing_relief_developer_deadline",),
            legal_mechanism="",
        ),
        _record(
            "pit_art_21_ust_30",
            "art. 21 ust. 30 ustawy PIT",
            "Ogólna reguła ogranicza ponowne uwzględnianie wydatków już rozliczonych przy ulgach podatkowych.",
            result_codes=("credit_on_sold_property_disqualified",),
            legal_mechanism="housing_relief_credit_repayment",
            rule_relationship="general_rule",
            special_rule_provisions=("pit_art_21_ust_30a",),
        ),
        _record(
            "pit_art_21_ust_30a",
            "art. 21 ust. 30a ustawy PIT",
            "Przepis szczególny dla spłaty kredytu dotyczącego zbywanej nieruchomości.",
            result_codes=("credit_on_sold_property_qualified",),
            legal_mechanism="housing_relief_credit_repayment",
            rule_relationship="special_extension",
            general_rule_provisions=("pit_art_21_ust_30",),
        ),
        _record(
            "pit_art_30e_ust_1",
            "art. 30e ust. 1 ustawy PIT",
            "Podatek od dochodu z odpłatnego zbycia nieruchomości wynosi 19% podstawy obliczenia podatku.",
            result_codes=("housing_relief_tax",),
        ),
    ]
    return ProvisionRegistry(provisions=records)


def calculate_housing_relief(
    facts: HousingReliefFacts,
) -> dict[str, CalculationRecord]:
    exempt_income = int(
        (
            Decimal(facts.income)
            * Decimal(facts.qualified_expenses)
            / Decimal(facts.revenue)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    taxable_income = facts.income - exempt_income
    tax = int(
        (Decimal(taxable_income) * Decimal("0.19")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    qualified_expenses_if_credit_qualifies = facts.qualified_expenses + facts.credit_repayment
    exempt_income_if_credit_qualifies = int(
        (
            Decimal(facts.income)
            * Decimal(qualified_expenses_if_credit_qualifies)
            / Decimal(facts.revenue)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    taxable_income_if_credit_qualifies = facts.income - exempt_income_if_credit_qualifies
    tax_if_credit_qualifies = int(
        (Decimal(taxable_income_if_credit_qualifies) * Decimal("0.19")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    developer_expense_qualifies = facts.planned_transfer_date <= facts.deadline
    return {
        "calc_housing_relief_revenue": CalculationRecord(
            "calc_housing_relief_revenue",
            "identity",
            {"revenue": facts.revenue},
            facts.revenue,
        ),
        "calc_housing_relief_income": CalculationRecord(
            "calc_housing_relief_income",
            "identity",
            {"income": facts.income},
            facts.income,
        ),
        "calc_housing_relief_qualified_expenses": CalculationRecord(
            "calc_housing_relief_qualified_expenses",
            "identity",
            {"qualified_expenses": facts.qualified_expenses},
            facts.qualified_expenses,
        ),
        "calc_housing_relief_credit_repayment": CalculationRecord(
            "calc_housing_relief_credit_repayment",
            "identity",
            {"credit_repayment": facts.credit_repayment},
            facts.credit_repayment,
        ),
        "calc_housing_relief_developer_payment": CalculationRecord(
            "calc_housing_relief_developer_payment",
            "identity",
            {"developer_payment": facts.developer_payment},
            facts.developer_payment,
        ),
        "calc_housing_relief_disqualified_developer_expense": CalculationRecord(
            "calc_housing_relief_disqualified_developer_expense",
            "identity",
            {"disqualified_developer_expense": facts.disqualified_developer_expense},
            facts.disqualified_developer_expense,
        ),
        "calc_housing_relief_exempt_income": CalculationRecord(
            "calc_housing_relief_exempt_income",
            "housing_relief_formula",
            {
                "income": facts.income,
                "qualified_expenses": facts.qualified_expenses,
                "revenue": facts.revenue,
                "formula": "D × W / P",
            },
            exempt_income,
        ),
        "calc_housing_relief_taxable_income": CalculationRecord(
            "calc_housing_relief_taxable_income",
            "subtract",
            {"income": facts.income, "exempt_income": exempt_income},
            taxable_income,
        ),
        "calc_housing_relief_tax": CalculationRecord(
            "calc_housing_relief_tax",
            "multiply",
            {"taxable_income": taxable_income, "rate": Decimal("0.19")},
            tax,
        ),
        "calc_housing_relief_qualified_expenses_credit_scenario": CalculationRecord(
            "calc_housing_relief_qualified_expenses_credit_scenario",
            "conditional_add",
            {
                "baseline_qualified_expenses": facts.qualified_expenses,
                "credit_repayment": facts.credit_repayment,
                "condition": "credit_repayment_statutory_conditions_confirmed",
            },
            qualified_expenses_if_credit_qualifies,
        ),
        "calc_housing_relief_exempt_income_credit_scenario": CalculationRecord(
            "calc_housing_relief_exempt_income_credit_scenario",
            "housing_relief_formula",
            {
                "income": facts.income,
                "qualified_expenses": qualified_expenses_if_credit_qualifies,
                "revenue": facts.revenue,
                "formula": "D × W / P",
                "scenario": "credit_repayment_qualifies",
            },
            exempt_income_if_credit_qualifies,
        ),
        "calc_housing_relief_taxable_income_credit_scenario": CalculationRecord(
            "calc_housing_relief_taxable_income_credit_scenario",
            "subtract",
            {"income": facts.income, "exempt_income": exempt_income_if_credit_qualifies},
            taxable_income_if_credit_qualifies,
        ),
        "calc_housing_relief_tax_credit_scenario": CalculationRecord(
            "calc_housing_relief_tax_credit_scenario",
            "multiply",
            {"taxable_income": taxable_income_if_credit_qualifies, "rate": Decimal("0.19")},
            tax_if_credit_qualifies,
        ),
        "calc_housing_relief_deadline": CalculationRecord(
            "calc_housing_relief_deadline",
            "end_of_third_year_following_sale",
            {
                "sale_year": facts.sale_year,
                "sale_year_end": facts.sale_year_end,
                "statutory_period_years": HOUSING_EXPENSE_PERIOD_YEARS,
                "deadline_period_source": "pit_art_21_ust_1_pkt_131",
                "ownership_condition_source": "pit_art_21_ust_25a",
            },
            facts.deadline,
        ),
        "calc_housing_relief_developer_qualification": CalculationRecord(
            "calc_housing_relief_developer_qualification",
            "compare_transfer_year_to_deadline",
            {
                "planned_transfer_date": facts.planned_transfer_date,
                "deadline": facts.deadline,
                "ownership_condition_source": "pit_art_21_ust_25a",
            },
            developer_expense_qualifies,
        ),
    }


def validate_housing_deadline_invariants(
    facts: HousingReliefFacts,
    calculations: dict[str, CalculationRecord],
) -> tuple[str, ...]:
    """Independently recompute the statutory time limit before claims use it."""
    expected_sale_year_end = date(facts.sale_year, 12, 31).isoformat()
    expected_deadline = housing_expense_deadline_for_sale_year(facts.sale_year)
    deadline = calculations.get("calc_housing_relief_deadline")
    qualification = calculations.get("calc_housing_relief_developer_qualification")
    errors: list[str] = []
    if facts.sale_year_end != expected_sale_year_end:
        errors.append("sale_year_end_invariant_failed")
    if facts.deadline != expected_deadline:
        errors.append("housing_expense_deadline_invariant_failed")
    if deadline is None or deadline.result != expected_deadline:
        errors.append("deadline_calculation_result_invalid")
    elif (
        deadline.inputs.get("sale_year_end") != expected_sale_year_end
        or deadline.inputs.get("statutory_period_years") != HOUSING_EXPENSE_PERIOD_YEARS
        or deadline.inputs.get("deadline_period_source") != "pit_art_21_ust_1_pkt_131"
        or deadline.inputs.get("ownership_condition_source") != "pit_art_21_ust_25a"
    ):
        errors.append("deadline_calculation_provenance_invalid")
    expected_qualification = facts.planned_transfer_date <= expected_deadline
    if qualification is None or qualification.result is not expected_qualification:
        errors.append("developer_qualification_boundary_invalid")
    return tuple(errors)


def _claim(
    claim_id: str,
    text: str,
    result_code: str,
    result: dict[str, object],
    provisions: tuple[str, ...],
    fact_ids: tuple[str, ...],
    *,
    calculation_ids: tuple[str, ...] = (),
    status: str = "approved",
    legal_mechanism: str = "housing_relief_sale",
) -> LegalClaim:
    return LegalClaim(
        claim_id=claim_id,
        axis_id="pit_housing_relief",
        claim_type="calculated_result" if calculation_ids else "legal_conclusion",
        text=text,
        source_provisions=provisions,
        controlling_provisions=provisions,
        fact_dependencies=fact_ids,
        calculation_id=calculation_ids[0] if calculation_ids else None,
        calculation_ids=calculation_ids,
        status=status,  # type: ignore[arg-type]
        result=result,
        result_code=result_code,
        taxpayer_role="taxpayer",
        legal_mechanism=legal_mechanism,
    )


def build_housing_relief_claims(
    facts: HousingReliefFacts,
    calculations: dict[str, CalculationRecord],
) -> dict[str, LegalClaim]:
    exempt_income = int(calculations["calc_housing_relief_exempt_income"].result)
    taxable_income = int(calculations["calc_housing_relief_taxable_income"].result)
    tax = int(calculations["calc_housing_relief_tax"].result)
    exempt_income_if_credit_qualifies = int(calculations["calc_housing_relief_exempt_income_credit_scenario"].result)
    taxable_income_if_credit_qualifies = int(calculations["calc_housing_relief_taxable_income_credit_scenario"].result)
    tax_if_credit_qualifies = int(calculations["calc_housing_relief_tax_credit_scenario"].result)
    developer_expense_qualifies = bool(
        calculations["calc_housing_relief_developer_qualification"].result
    )
    developer_status = (
        "approved_qualifying" if developer_expense_qualifies else "approved_not_qualifying"
    )
    developer_conclusion = (
        "wskazany wydatek deweloperski kwalifikuje się do ulgi."
        if developer_expense_qualifies
        else "wskazany wydatek deweloperski nie kwalifikuje się; to wynik negatywny, a nie ryzyko interpretacyjne."
    )
    claims = [
        _claim(
            "claim_sale_tax_regime",
            "Źródłem opodatkowania jest odpłatne zbycie z art. 10 ust. 1 pkt 8 ustawy PIT, a stawka podatku wynika z art. 30e ust. 1 ustawy PIT.",
            "sale_tax_regime",
            {"income": facts.income, "revenue": facts.revenue, "tax_rate": 0.19},
            ("pit_art_10_ust_1_pkt_8", "pit_art_30e_ust_1"),
            ("income", "revenue"),
            calculation_ids=(
                "calc_housing_relief_income",
                "calc_housing_relief_revenue",
            ),
        ),
        _claim(
            "claim_formula",
            (
                f"Dochód zwolniony trzeba policzyć wyłącznie wzorem D × W / P. "
                f"Dla scenariusza ostrożnego D = {facts.income:,} zł, W = {facts.qualified_expenses:,} zł i P = {facts.revenue:,} zł "
                f"dochód zwolniony wynosi {exempt_income:,} zł. Spłata kredytu nie jest tu jeszcze doliczona, "
                "dopóki nie zostaną potwierdzone jej ustawowe warunki."
            ).replace(",", " "),
            "housing_relief_formula",
            {
                "income": facts.income,
                "revenue": facts.revenue,
                "qualified_housing_expenses": facts.qualified_expenses,
                "declared_housing_expenses": facts.declared_housing_expenses,
                "exempt_income": exempt_income,
                "direct_expense_income_offset_used": False,
            },
            ("pit_art_21_ust_1_pkt_131",),
            ("income", "revenue", "qualified_housing_expenses"),
            calculation_ids=(
                "calc_housing_relief_income",
                "calc_housing_relief_qualified_expenses",
                "calc_housing_relief_revenue",
                "calc_housing_relief_exempt_income",
            ),
        ),
        _claim(
            "claim_expense_not_income",
            (
                f"Kwota {facts.qualified_expenses:,} zł oznacza kwalifikowane wydatki mieszkaniowe w scenariuszu ostrożnym, a nie dochód zwolniony; "
                f"dochód zwolniony po odrębnym obliczeniu proporcji wynosi {exempt_income:,} zł."
            ).replace(",", " "),
            "housing_relief_exempt_income",
            {
                "declared_housing_expenses": facts.declared_housing_expenses,
                "qualified_housing_expenses": facts.qualified_expenses,
                "exempt_income": exempt_income,
                "values_treated_as_identical": False,
            },
            ("pit_art_21_ust_1_pkt_131",),
            ("qualified_housing_expenses", "income", "revenue"),
            calculation_ids=(
                "calc_housing_relief_qualified_expenses",
                "calc_housing_relief_exempt_income",
            ),
        ),
        _claim(
            "claim_tax_result",
            (
                f"Bez potwierdzenia warunków spłaty kredytu pozostały dochód do opodatkowania wynosi {taxable_income:,} zł, "
                f"a podatek wynosi {tax:,} zł."
            ).replace(",", " "),
            "housing_relief_tax",
            {
                "exempt_income": exempt_income,
                "taxable_income": taxable_income,
                "tax": tax,
            },
            ("pit_art_21_ust_1_pkt_131", "pit_art_30e_ust_1"),
            ("income", "qualified_housing_expenses", "revenue"),
            calculation_ids=(
                "calc_housing_relief_exempt_income",
                "calc_housing_relief_taxable_income",
                "calc_housing_relief_tax",
            ),
        ),
        _claim(
            "claim_developer_deadline",
            (
                f"Trzyletni termin wynika z art. 21 ust. 1 pkt 131 ustawy PIT. Upływa {facts.deadline}. "
                "Warunek nabycia własności wynika z art. 21 ust. 25a ustawy PIT. "
                "Własność trzeba nabyć przed upływem tego terminu. "
                f"Skoro przeniesienie własności ma nastąpić {facts.planned_transfer_date}, "
                + developer_conclusion
            ),
            "housing_relief_developer_deadline",
            {
                "housing_expense_deadline": facts.deadline,
                "planned_transfer_year": facts.planned_transfer_year,
                "planned_transfer_date": facts.planned_transfer_date,
                "developer_payment": facts.developer_payment,
                "disqualified_developer_expense": facts.disqualified_developer_expense,
                "developer_expense_qualifies": developer_expense_qualifies,
                "status": developer_status,
                "interpretive_risk_status_used": False,
            },
            ("pit_art_21_ust_1_pkt_131", "pit_art_21_ust_25a"),
            ("planned_transfer_date", "housing_expense_deadline"),
            calculation_ids=("calc_housing_relief_deadline",),
        ),
        _claim(
            "claim_credit_scope",
            (
                f"Spłata {_format_money(facts.credit_repayment)} zł kredytu zaciągniętego na zbywane mieszkanie "
                "może być wydatkiem na własne cele mieszkaniowe. Wymaga łącznego zastosowania "
                "art. 21 ust. 25 pkt 2 ustawy PIT, art. 21 ust. 30 ustawy PIT, oraz przepisu szczególnego "
                "z art. 21 ust. 30a ustawy PIT. Nie wolno jej dyskwalifikować samą regułą ogólną z art. 21 ust. 30."
            ),
            "",
            {
                "special_credit_rule_present": True,
                "credit_repayment": facts.credit_repayment,
                "credit_repayment_qualifies": None,
                "direct_expense_income_offset_used": False,
            },
            (
                "pit_art_21_ust_25_pkt_2",
                "pit_art_21_ust_30",
                "pit_art_21_ust_30a",
            ),
            ("credit_repayment", "credit_taken_before_sale", "credit_not_previously_tax_preferenced"),
            calculation_ids=(),
            status="conditional_missing_fact",
            legal_mechanism="housing_relief_credit_repayment",
        ),
        _claim(
            "claim_credit_qualifying_tax_scenario",
            (
                f"Jeżeli potwierdzisz termin zaciągnięcia kredytu i brak wcześniejszego rozliczenia go w innej uldze, "
                f"spłata {facts.credit_repayment:,} zł zwiększy W do {facts.qualified_expenses + facts.credit_repayment:,} zł; "
                f"dochód zwolniony wyniesie {exempt_income_if_credit_qualifies:,} zł, dochód opodatkowany "
                f"{taxable_income_if_credit_qualifies:,} zł, a PIT {tax_if_credit_qualifies:,} zł."
            ).replace(",", " "),
            "",
            {"status": "conditional_missing_fact", "tax": tax_if_credit_qualifies},
            ("pit_art_21_ust_25_pkt_2", "pit_art_21_ust_30", "pit_art_21_ust_30a"),
            ("credit_taken_before_sale", "credit_not_previously_tax_preferenced"),
            status="conditional_missing_fact",
            legal_mechanism="housing_relief_credit_repayment",
            calculation_ids=(
                "calc_housing_relief_qualified_expenses_credit_scenario",
                "calc_housing_relief_exempt_income_credit_scenario",
                "calc_housing_relief_taxable_income_credit_scenario",
                "calc_housing_relief_tax_credit_scenario",
            ),
        ),
    ]
    return {item.claim_id: item for item in claims}


def run_housing_relief_pipeline(
    query: str,
    *,
    target_date: str = "2026-06-30",
    authority_cards: Iterable[dict[str, str]] = (),
    judgment_lane_outcome: dict[str, object] | None = None,
) -> LegalPipelineResult:
    if not can_run_housing_relief_pipeline(query):
        raise ValueError("Query is not a supported housing-relief controlled case.")
    registry = build_housing_relief_registry()
    facts = parse_housing_relief_facts(query)
    calculations = calculate_housing_relief(facts)
    deadline_errors = validate_housing_deadline_invariants(facts, calculations)
    if deadline_errors:
        raise ValueError("Deadline calculation invariant failed: " + ", ".join(deadline_errors))
    claims = build_housing_relief_claims(facts, calculations)
    for claim in claims.values():
        validation = validate_claim(
            claim,
            registry,
            target_date=target_date,
            facts=facts.records,
            calculations=calculations,
        )
        conditional_missing_fact = (
            claim.status == "conditional_missing_fact"
            and set(validation.errors) == {"missing_fact_dependency"}
        )
        if not validation.claim_supported and not conditional_missing_fact:
            raise ValueError(f"Claim {claim.claim_id} failed: {validation.errors}")
    payload = build_renderer_payload(
        claims,
        registry,
        target_date=target_date,
        calculations=calculations,
        authority_cards=bind_authority_cards_to_claims(authority_cards, claims),
        judgment_lane_outcome=judgment_lane_outcome,
    )
    rendered = render_answer(payload)
    validation = validate_rendered_answer(rendered, payload)
    if not validation.passed:
        rendered = render_answer(payload, compact=True)
        validation = validate_rendered_answer(rendered, payload)
    if not validation.passed:
        raise RuntimeError(f"post_render_validation_failed: {validation.errors}")
    return LegalPipelineResult(
        claims=claims,
        facts=facts.records,
        calculations=calculations,
        renderer_payload=payload.to_dict(),
        answer=rendered.removesuffix("<END_OF_ANALYSIS>").rstrip(),
        render_validation=validation,
    )
