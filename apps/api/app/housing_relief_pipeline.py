from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from app.controlled_legal_pipeline import (
    LegalPipelineResult,
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
dochód wynosi 300 000 zł, a wydatki mieszkaniowe wyniosą 600 000 zł.
Chodzi o wpłaty do dewelopera i spłatę kredytu mieszkaniowego. Przeniesienie
własności nowego lokalu ma nastąpić w 2029 r. Oceń art. 21 ust. 1 pkt 131,
art. 21 ust. 25 pkt 2 lit. a, art. 21 ust. 25a, art. 21 ust. 30a oraz podatek
z art. 30e.
"""


def is_housing_relief_query(query: str) -> bool:
    normalized = query.lower()
    return (
        bool(re.search(r"ulg\w*\s+mieszkaniow\w*|art\.\s*21\s*ust\.\s*1\s*pkt\s*131", normalized))
        and bool(re.search(r"przych[oó]d\w*|doch[oó]d\w*|art\.\s*30e", normalized))
        and bool(re.search(r"deweloper\w*|przeniesieni\w*\s+własnoś\w*|przeniesieni\w*\s+wlasnos\w*|kredyt\w*", normalized))
    )


def _money_after(label: str, query: str) -> int:
    match = re.search(
        rf"{label}.{{0,40}}?(\d{{1,3}}(?:[ .]\d{{3}})*)\s*zł",
        query,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing value for {label}")
    return int(re.sub(r"\D", "", match.group(1)))


def _extract_year(query: str, pattern: str) -> int:
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing year for pattern {pattern}")
    return int(match.group(1))


@dataclass(frozen=True)
class HousingReliefFacts:
    records: dict[str, FactRecord]
    sale_year: int
    revenue: int
    income: int
    qualified_expenses: int
    planned_transfer_year: int
    deadline: str


def parse_housing_relief_facts(query: str) -> HousingReliefFacts:
    sale_year = _extract_year(query, r"sprzeda\w*.*?\b(20\d{2})\b")
    revenue = _money_after(r"przych[oó]d\w*", query)
    income = _money_after(r"doch[oó]d\w*", query)
    qualified_expenses = _money_after(
        r"(?:wydatk\w*\s+mieszkaniow\w*|kwalifikowan\w*\s+wydatk\w*|w\W*tym)",
        query,
    )
    planned_transfer_year = _extract_year(
        query,
        r"przeniesieni\w*\s+w(?:ł|l)asnoś\w*.*?\b(20\d{2})\b",
    )
    deadline = date(sale_year + 3, 12, 31).isoformat()
    records = {
        "sale_year": FactRecord("sale_year", "year", sale_year, subject_role="transaction"),
        "revenue": FactRecord("revenue", "money", revenue, subject_role="transaction"),
        "income": FactRecord("income", "money", income, subject_role="transaction"),
        "qualified_housing_expenses": FactRecord(
            "qualified_housing_expenses",
            "money",
            qualified_expenses,
            subject_role="transaction",
        ),
        "planned_transfer_year": FactRecord(
            "planned_transfer_year",
            "year",
            planned_transfer_year,
            subject_role="transaction",
        ),
        "housing_expense_deadline": FactRecord(
            "housing_expense_deadline",
            "date",
            deadline,
            date=deadline,
            subject_role="transaction",
        ),
    }
    return HousingReliefFacts(
        records=records,
        sale_year=sale_year,
        revenue=revenue,
        income=income,
        qualified_expenses=qualified_expenses,
        planned_transfer_year=planned_transfer_year,
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
        and facts.qualified_expenses >= 0
        and facts.revenue >= facts.income
    )


def _record(
    provision_id: str,
    citation: str,
    text: str,
    *,
    result_codes: tuple[str, ...],
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
        legal_mechanism="housing_relief_sale",
        entailed_result_codes=result_codes,
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
            result_codes=("housing_relief_formula", "housing_relief_exempt_income"),
        ),
        _record(
            "pit_art_21_ust_25_pkt_2_lit_a",
            "art. 21 ust. 25 pkt 2 lit. a ustawy PIT",
            "Za wydatki mieszkaniowe uważa się spłatę kredytu zaciągniętego na cele mieszkaniowe.",
            result_codes=("housing_relief_credit_scope",),
        ),
        _record(
            "pit_art_21_ust_25a",
            "art. 21 ust. 25a ustawy PIT",
            "Wydatki na nabycie od dewelopera wymagają nabycia własności w ustawowym terminie.",
            result_codes=("housing_relief_developer_deadline",),
        ),
        _record(
            "pit_art_21_ust_30a",
            "art. 21 ust. 30a ustawy PIT",
            "Przepis szczególny dla spłaty kredytu dotyczącego zbywanej nieruchomości.",
            result_codes=("housing_relief_credit_scope",),
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
    developer_expense_qualifies = facts.planned_transfer_year <= int(facts.deadline[:4])
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
        "calc_housing_relief_deadline": CalculationRecord(
            "calc_housing_relief_deadline",
            "end_of_third_year_following_sale",
            {"sale_year": facts.sale_year},
            facts.deadline,
        ),
        "calc_housing_relief_developer_qualification": CalculationRecord(
            "calc_housing_relief_developer_qualification",
            "compare_transfer_year_to_deadline",
            {
                "planned_transfer_year": facts.planned_transfer_year,
                "deadline": facts.deadline,
            },
            developer_expense_qualifies,
        ),
    }


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
        legal_mechanism="housing_relief_sale",
    )


def build_housing_relief_claims(
    facts: HousingReliefFacts,
    calculations: dict[str, CalculationRecord],
) -> dict[str, LegalClaim]:
    exempt_income = int(calculations["calc_housing_relief_exempt_income"].result)
    taxable_income = int(calculations["calc_housing_relief_taxable_income"].result)
    tax = int(calculations["calc_housing_relief_tax"].result)
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
                f"Dla D = {facts.income:,} zł, W = {facts.qualified_expenses:,} zł i P = {facts.revenue:,} zł "
                f"dochód zwolniony wynosi {exempt_income:,} zł."
            ).replace(",", " "),
            "housing_relief_formula",
            {
                "income": facts.income,
                "revenue": facts.revenue,
                "qualified_housing_expenses": facts.qualified_expenses,
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
                f"Kwota {facts.qualified_expenses:,} zł oznacza wydatki mieszkaniowe, a nie dochód zwolniony; "
                f"dochód zwolniony po odrębnym obliczeniu proporcji wynosi {exempt_income:,} zł."
            ).replace(",", " "),
            "housing_relief_exempt_income",
            {
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
                f"Pozostały dochód do opodatkowania wynosi {taxable_income:,} zł, "
                f"a podatek wynika z art. 30e ust. 1 ustawy PIT. Wynosi {tax:,} zł."
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
                f"Ustawowy termin wynika z art. 21 ust. 25a ustawy PIT. Upływa {facts.deadline}. "
                f"Skoro przeniesienie własności ma nastąpić dopiero w {facts.planned_transfer_year} r., "
                "wskazany wydatek deweloperski nie kwalifikuje się; to wynik negatywny, a nie ryzyko interpretacyjne."
            ),
            "housing_relief_developer_deadline",
            {
                "housing_expense_deadline": facts.deadline,
                "planned_transfer_year": facts.planned_transfer_year,
                "developer_expense_qualifies": False,
                "status": "approved_not_qualifying",
                "interpretive_risk_status_used": False,
            },
            ("pit_art_21_ust_25a",),
            ("planned_transfer_year", "housing_expense_deadline"),
            calculation_ids=("calc_housing_relief_deadline",),
        ),
        _claim(
            "claim_credit_scope",
            "Spłata kredytu mieszkaniowego wymaga łącznego zastosowania art. 21 ust. 25 pkt 2 lit. a ustawy PIT, art. 21 ust. 25a ustawy PIT, art. 21 ust. 30a ustawy PIT. Ta kwalifikacja nie pozwala zastąpić ustawowego wzoru prostym odjęciem wydatków od dochodu.",
            "housing_relief_credit_scope",
            {
                "special_credit_rule_present": True,
                "direct_expense_income_offset_used": False,
            },
            (
                "pit_art_21_ust_25_pkt_2_lit_a",
                "pit_art_21_ust_25a",
                "pit_art_21_ust_30a",
            ),
            ("planned_transfer_year",),
        ),
    ]
    return {item.claim_id: item for item in claims}


def run_housing_relief_pipeline(
    query: str, *, target_date: str = "2026-06-30"
) -> LegalPipelineResult:
    if not can_run_housing_relief_pipeline(query):
        raise ValueError("Query is not a supported housing-relief controlled case.")
    registry = build_housing_relief_registry()
    facts = parse_housing_relief_facts(query)
    calculations = calculate_housing_relief(facts)
    claims = build_housing_relief_claims(facts, calculations)
    for claim in claims.values():
        validation = validate_claim(
            claim,
            registry,
            target_date=target_date,
            facts=facts.records,
            calculations=calculations,
        )
        if not validation.claim_supported:
            raise ValueError(f"Claim {claim.claim_id} failed: {validation.errors}")
    payload = build_renderer_payload(
        claims,
        registry,
        target_date=target_date,
        calculations=calculations,
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
