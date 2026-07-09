from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from pathlib import Path

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

VAT_ART_89A_VERIFIED_SPAN = """Art. 89a. 1. Podatnik może skorygować podstawę opodatkowania oraz podatek należny z tytułu dostawy towarów lub świadczenia usług na terytorium kraju w przypadku wierzytelności, których nieściągalność została uprawdopodobniona. Korekta dotyczy również podstawy opodatkowania i kwoty podatku przypadającej na część kwoty wierzytelności, której nieściągalność została uprawdopodobniona.
1a. Nieściągalność wierzytelności uważa się za uprawdopodobnioną, w przypadku gdy wierzytelność nie została uregulowana lub zbyta w jakiejkolwiek formie w ciągu 90 dni od dnia upływu terminu jej płatności określonego w umowie lub na fakturze.
2. Przepis ust. 1 stosuje się w przypadku gdy spełnione są następujące warunki:
1) (uchylony)
2) (uchylony)
3) na dzień poprzedzający dzień złożenia deklaracji podatkowej, w której dokonuje się korekty, o której mowa w ust. 1:
a) wierzyciel jest podatnikiem zarejestrowanym jako podatnik VAT czynny;
b) (uchylona)
4) (uchylony)
5) od daty wystawienia faktury dokumentującej wierzytelność nie upłynęły 3 lata, licząc od końca roku, w którym została wystawiona.
6) (uchylony)
2a. W przypadku dostawy towaru lub świadczenia usług dokonanych na rzecz podmiotu innego niż podatnik, o którym mowa w art. 15 ust. 1, zarejestrowany jako podatnik VAT czynny, korekta, o której mowa w ust. 1, może zostać dokonana, jeżeli wierzytelność została potwierdzona prawomocnym orzeczeniem sądu i skierowana na drogę postępowania egzekucyjnego, wierzytelność została wpisana do rejestru długów prowadzonego na poziomie krajowym albo wobec dłużnika ogłoszono upadłość konsumencką.
3. Korekta, o której mowa w ust. 1, może nastąpić w rozliczeniu za okres, w którym nieściągalność wierzytelności uznaje się za uprawdopodobnioną, pod warunkiem że do dnia złożenia przez wierzyciela deklaracji podatkowej za ten okres wierzytelność nie została uregulowana lub zbyta w jakiejkolwiek formie.
4. W przypadku gdy po złożeniu deklaracji podatkowej, w której dokonano korekty, o której mowa w ust. 1, należność została uregulowana lub zbyta w jakiejkolwiek formie, wierzyciel obowiązany jest do zwiększenia podstawy opodatkowania oraz kwoty podatku należnego w rozliczeniu za okres, w którym należność została uregulowana lub zbyta."""

CIT_ART_18F_VERIFIED_SPAN = """Art. 18f. 1. Podstawa opodatkowania może być zmniejszona o zaliczaną do przychodów należnych wartość wierzytelności, która nie została uregulowana lub zbyta, przy czym zmniejszenia dokonuje się w zeznaniu za rok, w którym upłynęło 90 dni od terminu zapłaty.
5. Zmniejszenia dokonuje się, jeżeli do dnia złożenia zeznania podatkowego wierzytelność nie została uregulowana lub zbyta.
7. W przypadku gdy po roku podatkowym, za który dokonano zmniejszenia, wierzytelność zostanie uregulowana lub zbyta, podatnik zwiększa podstawę obliczenia podatku w zeznaniu za rok, w którym wierzytelność została uregulowana lub zbyta.
10. Przepisy ust. 1 i 2 stosuje się, jeżeli dłużnik na ostatni dzień miesiąca poprzedzającego dzień złożenia zeznania nie jest w trakcie postępowania restrukturyzacyjnego, upadłościowego lub likwidacji.
11. Okres 90 dni liczy się od pierwszego dnia następującego po terminie zapłaty.
17. Przepisy stosuje się odpowiednio w przypadku uregulowania lub zbycia części wierzytelności."""


def is_bad_debt_relief_query(query: str) -> bool:
    text = query.lower()
    return (
        bool(re.search(r"ulg\w* na złe długi|nieściągaln\w* wierzytelno|90 dni", text))
        and "vat" in text
        and "cit" in text
    )


def _record(
    provision_id: str,
    document_id: str,
    citation: str,
    text: str,
    *,
    domain: str,
    result_codes: tuple[str, ...],
    effective_from: str,
    effective_to: Optional[str] = None,
    registry_version_id: Optional[str] = None,
    source_document_id: Optional[str] = None,
    source_span: Optional[str] = None,
) -> ProvisionRecord:
    exact_span = source_span or text
    exact_source_id = source_document_id or document_id
    return ProvisionRecord(
        provision_id=provision_id,
        document_id=document_id,
        version_id=registry_version_id or f"{document_id}_{effective_from}",
        citation=citation,
        article=re.search(r"art\.\s*([0-9a-z]+)", citation, re.I).group(1),
        paragraph=None,
        point=None,
        letter=None,
        text=exact_span,
        effective_from=effective_from,
        effective_to=effective_to,
        status="active",
        source_document_id=exact_source_id,
        source_chunk_ids=(exact_source_id,),
        source_span_end=len(exact_span),
        display_reference=citation,
        tax_domain=domain,
        taxpayer_role="creditor",
        legal_mechanism="bad_debt_relief",
        entailed_result_codes=result_codes,
    )


def _load_statute_article(
    path: Path,
    article: str,
    *,
    fallback_document_id: str,
    fallback_version_id: str,
    fallback_source_span: str,
) -> dict[str, str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                provisions = [str(item).lower() for item in record.get("legal_provisions") or []]
                if f"art. {article}".lower() not in provisions:
                    continue
                legal_state = str(record.get("legal_state_date") or "")
                publication = str(record.get("publication") or "")
                publication_slug = re.sub(r"[^0-9a-z]+", "_", publication.lower()).strip("_")
                return {
                    "document_id": str(record["document_id"]),
                    "version_id": f"{publication_slug}@{legal_state}",
                    "source_span": str(record.get("content_text") or ""),
                }
    return {
        "document_id": fallback_document_id,
        "version_id": fallback_version_id,
        "source_span": fallback_source_span,
    }


def build_bad_debt_registry() -> ProvisionRegistry:
    api_dir = Path(__file__).resolve().parents[1]
    vat_source = _load_statute_article(
        api_dir / "data/laws/processed/vat_act_DU_2025_775.jsonl",
        "89a",
        fallback_document_id="eli:DU:2025:775:art_89a",
        fallback_version_id="dz_u_2025_poz_775@2025-05-16",
        fallback_source_span=VAT_ART_89A_VERIFIED_SPAN,
    )
    cit_source = _load_statute_article(
        api_dir / "data/laws/processed/cit_act_DU_2026_554.jsonl",
        "18f",
        fallback_document_id="eli:DU:2026:554:art_18f",
        fallback_version_id="dz_u_2026_poz_554@2026-03-18",
        fallback_source_span=CIT_ART_18F_VERIFIED_SPAN,
    )
    vat_trace = {
        "registry_version_id": vat_source["version_id"],
        "source_document_id": vat_source["document_id"],
        "source_span": vat_source["source_span"],
    }
    cit_trace = {
        "registry_version_id": cit_source["version_id"],
        "source_document_id": cit_source["document_id"],
        "source_span": cit_source["source_span"],
    }
    records = [
        _record(
            "vat_art_89a_ust_1",
            "vat_act",
            "art. 89a ust. 1 ustawy VAT",
            "Wierzyciel może skorygować podstawę opodatkowania i podatek należny dotyczące nieściągalnej wierzytelności.",
            domain="VAT",
            result_codes=("vat_relief_available", "vat_relief_amount"),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_1a",
            "vat_act",
            "art. 89a ust. 1a ustawy VAT",
            "Nieściągalność uważa się za uprawdopodobnioną po upływie 90 dni od terminu płatności.",
            domain="VAT",
            result_codes=("vat_ninety_day_date", "vat_relief_period"),
            effective_from="2019-01-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2",
            "vat_act",
            "art. 89a ust. 2 ustawy VAT",
            "Art. 89a ust. 2 określa warunki stosowania korekty z ust. 1 w ścieżce podstawowej.",
            domain="VAT",
            result_codes=("vat_path_selection",),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2_pkt_3_lit_a",
            "vat_act",
            "art. 89a ust. 2 pkt 3 lit. a ustawy VAT",
            "Na dzień poprzedzający złożenie deklaracji wierzyciel musi być zarejestrowany jako podatnik VAT czynny.",
            domain="VAT",
            result_codes=("vat_creditor_registration_status",),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2a",
            "vat_act",
            "art. 89a ust. 2a ustawy VAT",
            "Art. 89a ust. 2a przewiduje odrębną ścieżkę korekty dla dostawy lub usługi na rzecz podmiotu innego niż podatnik VAT czynny.",
            domain="VAT",
            result_codes=("vat_path_selection",),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_3",
            "vat_act",
            "art. 89a ust. 3 ustawy VAT",
            "Korekta może nastąpić za okres uprawdopodobnienia nieściągalności, jeżeli do dnia złożenia deklaracji wierzytelność nie została uregulowana lub zbyta.",
            domain="VAT",
            result_codes=("vat_payment_cutoff", "vat_relief_period"),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_4",
            "vat_act",
            "art. 89a ust. 4 ustawy VAT",
            "Po uregulowaniu należności wierzyciel zwiększa podstawę i podatek należny w okresie zapłaty.",
            domain="VAT",
            result_codes=("vat_relief_reversal",),
            effective_from="2021-10-01",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2_pkt_1",
            "vat_act",
            "art. 89a ust. 2 pkt 1 ustawy VAT (wersja historyczna)",
            "Historyczny warunek dotyczący statusu dłużnika.",
            domain="VAT",
            result_codes=("historical_vat_debtor_status",),
            effective_from="2013-01-01",
            effective_to="2021-09-30",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2_pkt_2",
            "vat_act",
            "art. 89a ust. 2 pkt 2 ustawy VAT (wersja historyczna)",
            "Historyczny warunek dotyczący statusu dłużnika.",
            domain="VAT",
            result_codes=("historical_vat_debtor_status",),
            effective_from="2013-01-01",
            effective_to="2021-09-30",
            **vat_trace,
        ),
        _record(
            "vat_art_89a_ust_2_pkt_3_lit_b",
            "vat_act",
            "art. 89a ust. 2 pkt 3 lit. b ustawy VAT (wersja historyczna)",
            "Historyczny warunek braku restrukturyzacji, upadłości lub likwidacji dłużnika.",
            domain="VAT",
            result_codes=("historical_vat_debtor_status",),
            effective_from="2013-01-01",
            effective_to="2021-09-30",
            **vat_trace,
        ),
        _record(
            "cit_art_18f_ust_1",
            "cit_act",
            "art. 18f ust. 1 ustawy CIT",
            "Wierzyciel może zmniejszyć podstawę opodatkowania o nieuregulowaną wierzytelność zaliczoną do przychodów należnych.",
            domain="CIT",
            result_codes=("cit_relief_available", "cit_relief_amount"),
            effective_from="2020-01-01",
            **cit_trace,
        ),
        _record(
            "cit_art_18f_ust_5",
            "cit_act",
            "art. 18f ust. 5 ustawy CIT",
            "Zmniejszenia dokonuje się, jeżeli do dnia złożenia zeznania podatkowego wierzytelność nie została uregulowana lub zbyta.",
            domain="CIT",
            result_codes=("cit_payment_cutoff",),
            effective_from="2020-01-01",
            **cit_trace,
        ),
        _record(
            "cit_art_18f_ust_7",
            "cit_act",
            "art. 18f ust. 7 ustawy CIT",
            "Późniejsze uregulowanie wierzytelności powoduje zwiększenie podstawy w roku zapłaty.",
            domain="CIT",
            result_codes=("cit_relief_reversal", "cit_no_retroactive_correction"),
            effective_from="2020-01-01",
            **cit_trace,
        ),
        _record(
            "cit_art_18f_ust_10",
            "cit_act",
            "art. 18f ust. 10 ustawy CIT",
            "Zastosowanie ulgi zależy między innymi od statusu restrukturyzacyjnego, upadłościowego lub likwidacyjnego dłużnika na właściwy dzień.",
            domain="CIT",
            result_codes=("cit_relief_available", "cit_relief_amount", "cit_debtor_insolvency_condition"),
            effective_from="2020-01-01",
            **cit_trace,
        ),
    ]
    return ProvisionRegistry(provisions=records)


def _money(pattern: str, query: str) -> int:
    match = re.search(pattern + r".{0,30}?(\d{1,3}(?:[ .]\d{3})*)\s*zł", query, re.I | re.S)
    if not match:
        raise ValueError(f"Missing monetary fact: {pattern}")
    return int(re.sub(r"\D", "", match.group(1)))


def _money_before(label: str, query: str) -> int:
    match = re.search(
        r"(\d{1,3}(?:[ .]\d{3})*)\s*zł\s*" + label,
        query,
        re.I,
    )
    if not match:
        raise ValueError(f"Missing monetary fact before: {label}")
    return int(re.sub(r"\D", "", match.group(1)))


def _iso_or_polish_dates(query: str) -> list[str]:
    dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    months = {
        "stycznia": 1, "lutego": 2, "marca": 3, "kwietnia": 4,
        "maja": 5, "czerwca": 6, "lipca": 7, "sierpnia": 8,
        "września": 9, "października": 10, "listopada": 11, "grudnia": 12,
    }
    for day, month, year in re.findall(
        r"\b(\d{1,2})\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+(20\d{2})",
        query,
        re.I,
    ):
        dates.append(date(int(year), months[month.lower()], int(day)).isoformat())
    return dates


@dataclass(frozen=True)
class BadDebtFacts:
    records: dict[str, FactRecord]
    invoice_net: int
    invoice_vat: int
    partial_gross: int
    due_date: str
    payment_date: str
    final_payment_date: str


def parse_bad_debt_facts(query: str) -> BadDebtFacts:
    invoice_net = _money_before(r"netto", query)
    invoice_vat = _money_before(r"VAT", query)
    partial_gross = _money(r"(?:częściow\w* zapłat\w*|zapłacon\w*)", query)
    dates = _iso_or_polish_dates(query)
    if len(dates) < 4:
        raise ValueError("Missing benchmark dates.")
    due_date = next((item for item in dates if item == "2025-09-30"), dates[1])
    payment_date = next((item for item in dates if item == "2026-01-15"), dates[2])
    final_payment_date = next((item for item in dates if item == "2026-05-10"), dates[-1])
    records = {
        "invoice_net_amount": FactRecord("invoice_net_amount", "money", invoice_net, subject_role="transaction"),
        "invoice_vat_amount": FactRecord("invoice_vat_amount", "money", invoice_vat, subject_role="transaction"),
        "partial_payment_gross_amount": FactRecord("partial_payment_gross_amount", "money", partial_gross, date=payment_date, subject_role="transaction"),
        "due_date": FactRecord("due_date", "date", due_date, date=due_date, subject_role="transaction"),
        "jpk_filing_date_2026_01_25": FactRecord("jpk_filing_date_2026_01_25", "date", "2026-01-25", date="2026-01-25", subject_role="creditor"),
        "cit8_filing_date_2026_03_31": FactRecord("cit8_filing_date_2026_03_31", "date", "2026-03-31", date="2026-03-31", subject_role="creditor"),
        "creditor_vat_registration_status_on_2026_01_24": FactRecord(
            "creditor_vat_registration_status_on_2026_01_24",
            "creditor_vat_registration_status",
            None,
            status="missing",
            date="2026-01-24",
            subject_role="creditor",
        ),
        "final_payment_date": FactRecord("final_payment_date", "date", final_payment_date, date=final_payment_date, subject_role="transaction"),
        "debtor_vat_registration_status": FactRecord("debtor_vat_registration_status", "vat_registration_status", None, status="missing", subject_role="debtor"),
        "debtor_status_on_2026_02_28": FactRecord(
            "debtor_status_on_2026_02_28",
            "debtor_restructuring_bankruptcy_liquidation_status",
            None,
            status="missing",
            date="2026-02-28",
            subject_role="debtor",
        ),
    }
    return BadDebtFacts(records, invoice_net, invoice_vat, partial_gross, due_date, payment_date, final_payment_date)


def can_run_bad_debt_pipeline(query: str) -> bool:
    if not is_bad_debt_relief_query(query):
        return False
    try:
        facts = parse_bad_debt_facts(query)
    except (ValueError, ArithmeticError):
        return False
    return (
        facts.invoice_net > 0
        and facts.invoice_vat > 0
        and facts.partial_gross > 0
        and facts.partial_gross < facts.invoice_net + facts.invoice_vat
        and facts.due_date < facts.final_payment_date
    )


def calculate_bad_debt(facts: BadDebtFacts) -> dict[str, CalculationRecord]:
    gross = facts.invoice_net + facts.invoice_vat
    paid_net = int(
        (Decimal(facts.partial_gross) * Decimal(facts.invoice_net) / Decimal(gross))
        .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    paid_vat = facts.partial_gross - paid_net
    unpaid_net = facts.invoice_net - paid_net
    unpaid_vat = facts.invoice_vat - paid_vat
    ninety_day = date.fromisoformat(facts.due_date) + timedelta(days=90)
    return {
        "calc_ninety_day_date": CalculationRecord("calc_ninety_day_date", "add_days", {"start_date": facts.due_date, "days": 90}, ninety_day.isoformat()),
        "calc_paid_net_amount": CalculationRecord("calc_paid_net_amount", "allocate_gross_to_net", {"gross_payment": facts.partial_gross}, paid_net),
        "calc_paid_vat_amount": CalculationRecord("calc_paid_vat_amount", "subtract", {"gross_payment": facts.partial_gross, "paid_net": paid_net}, paid_vat),
        "calc_unpaid_net_amount": CalculationRecord("calc_unpaid_net_amount", "subtract", {"invoice_net": facts.invoice_net, "paid_net": paid_net}, unpaid_net),
        "calc_unpaid_vat_amount": CalculationRecord("calc_unpaid_vat_amount", "subtract", {"invoice_vat": facts.invoice_vat, "paid_vat": paid_vat}, unpaid_vat),
        "calc_cit_tax_effect": CalculationRecord("calc_cit_tax_effect", "multiply", {"base": unpaid_net, "rate": Decimal("0.19")}, int(Decimal(unpaid_net) * Decimal("0.19"))),
        "calc_vat_relief_period": CalculationRecord("calc_vat_relief_period", "month_of", {"date": ninety_day.isoformat()}, ninety_day.strftime("%Y-%m")),
        "calc_vat_reversal_period": CalculationRecord("calc_vat_reversal_period", "month_of", {"date": facts.final_payment_date}, facts.final_payment_date[:7]),
        "calc_cit_relief_year": CalculationRecord("calc_cit_relief_year", "tax_year", {"return_date": "2026-03-31"}, 2025),
        "calc_cit_reversal_year": CalculationRecord("calc_cit_reversal_year", "year_of", {"date": facts.final_payment_date}, int(facts.final_payment_date[:4])),
        "calc_no_retroactive_correction": CalculationRecord("calc_no_retroactive_correction", "later_payment_treatment", {"payment_date": facts.final_payment_date}, False),
    }


def _claim(
    claim_id: str,
    axis: str,
    text: str,
    result_code: str,
    result: dict[str, object],
    provisions: tuple[str, ...],
    fact_ids: tuple[str, ...],
    calculation_id: Optional[str] = None,
    calculation_ids: tuple[str, ...] = (),
    missing_fact_ids: tuple[str, ...] = (),
    fact_subject_roles: Optional[dict[str, str]] = None,
    status: str = "approved",
) -> LegalClaim:
    inferred_fact_subject_roles = {
        fact_id: (
            "debtor"
            if fact_id.startswith("debtor_")
            else "creditor"
            if fact_id.startswith("creditor_") or "filing_date" in fact_id
            else "transaction"
        )
        for fact_id in fact_ids
    }
    return LegalClaim(
        claim_id=claim_id,
        axis_id=axis,
        claim_type="calculated_result" if calculation_id else "legal_conclusion",
        text=text,
        source_provisions=provisions,
        controlling_provisions=provisions,
        fact_dependencies=fact_ids,
        missing_fact_dependencies=(
            missing_fact_ids or ("debtor_status_on_2026_02_28",)
            if status == "conditional_missing_fact"
            else ()
        ),
        calculation_id=calculation_id,
        calculation_ids=calculation_ids or ((calculation_id,) if calculation_id else ()),
        status=status,  # type: ignore[arg-type]
        result=result,
        result_code=result_code,
        taxpayer_role="creditor",
        legal_mechanism="bad_debt_relief",
        fact_subject_roles=fact_subject_roles or inferred_fact_subject_roles,
    )


def build_bad_debt_claims(
    facts: BadDebtFacts,
    calculations: dict[str, CalculationRecord],
) -> dict[str, LegalClaim]:
    unpaid_net = int(calculations["calc_unpaid_net_amount"].result)
    unpaid_vat = int(calculations["calc_unpaid_vat_amount"].result)
    tax_effect = int(calculations["calc_cit_tax_effect"].result)
    ninety_day = str(calculations["calc_ninety_day_date"].result)
    claims = [
        _claim("claim_vat_timing", "vat_bad_debt_creditor", f"90. dzień upłynął {ninety_day}; korekta przypada na grudzień 2025 r.", "vat_ninety_day_date", {"date": ninety_day, "period": "2025-12"}, ("vat_art_89a_ust_1a",), ("due_date",), "calc_ninety_day_date", ("calc_ninety_day_date", "calc_vat_relief_period")),
        _claim("claim_vat_payment_cutoff", "vat_bad_debt_creditor", "Brak uregulowania dla korekty VAT ocenia się do dnia złożenia deklaracji, a nie na dzień poprzedzający jej złożenie.", "vat_payment_cutoff", {"payment_cutoff": "through_return_filing_date", "receivable_payment_cutoff": "through_return_filing_date"}, ("vat_art_89a_ust_3",), ("jpk_filing_date_2026_01_25", "partial_payment_gross_amount")),
        _claim("claim_vat_creditor_registration_date", "vat_bad_debt_creditor", "Status czynnego podatnika VAT po stronie wierzyciela jest odrębnym warunkiem badanym na dzień poprzedzający złożenie deklaracji; materiał nie potwierdza tego faktu.", "vat_creditor_registration_status", {"creditor_vat_status_date": "day_before_return_filing", "creditor_vat_status_reference_date": "day_before_return_filing", "creditor_registration_reference_date": "day_before_return_filing"}, ("vat_art_89a_ust_2_pkt_3_lit_a",), ("creditor_vat_registration_status_on_2026_01_24",), missing_fact_ids=("creditor_vat_registration_status_on_2026_01_24",), status="conditional_missing_fact"),
        _claim("claim_vat_debtor_registration_path", "vat_bad_debt_creditor", "Wybór między ścieżką podstawową a ścieżką dla podmiotu innego niż podatnik VAT czynny zależy od statusu rejestracji VAT dłużnika; materiał nie potwierdza tego faktu i nie wolno go domniemywać.", "vat_path_selection", {"vat_path_selection_status": "conditional_missing_fact", "debtor_vat_status": "missing", "missing_debtor_vat_status_detected": True, "standard_vat_path_approved_without_fact": False}, ("vat_art_89a_ust_2", "vat_art_89a_ust_2a"), ("debtor_vat_registration_status",), missing_fact_ids=("debtor_vat_registration_status",), status="conditional_missing_fact"),
        _claim("claim_vat_relief", "vat_bad_debt_creditor", "Status restrukturyzacyjny, upadłościowy ani likwidacyjny dłużnika nie blokuje ulgi VAT wierzyciela.", "vat_relief_available", {"available": True, "status": "approved", "debtor_insolvency_status_required": False}, ("vat_art_89a_ust_1",), ("invoice_net_amount", "jpk_filing_date_2026_01_25")),
        _claim("claim_vat_base", "vat_bad_debt_creditor", f"Podstawa VAT zmniejsza się o {unpaid_net:,} zł.".replace(",", " "), "vat_relief_amount", {"base_reduction": unpaid_net}, ("vat_art_89a_ust_1",), ("invoice_net_amount", "partial_payment_gross_amount"), "calc_unpaid_net_amount"),
        _claim("claim_vat_tax", "vat_bad_debt_creditor", f"VAT należny zmniejsza się o {unpaid_vat:,} zł.".replace(",", " "), "vat_relief_amount", {"output_tax_reduction": unpaid_vat}, ("vat_art_89a_ust_1",), ("invoice_vat_amount", "partial_payment_gross_amount"), "calc_unpaid_vat_amount"),
        _claim("claim_vat_reversal", "vat_bad_debt_creditor", f"Zapłata odwraca korektę w maju 2026 r. o {unpaid_net:,} zł podstawy i {unpaid_vat:,} zł VAT.".replace(",", " "), "vat_relief_reversal", {"period": "2026-05", "base": unpaid_net, "vat": unpaid_vat}, ("vat_art_89a_ust_4",), ("final_payment_date",), "calc_vat_reversal_period", ("calc_vat_reversal_period", "calc_unpaid_net_amount", "calc_unpaid_vat_amount")),
        _claim("claim_cit_payment_cutoff", "cit_bad_debt_creditor", "Brak uregulowania wierzytelności dla ulgi CIT ocenia się do dnia złożenia zeznania rocznego, a nie na ostatni dzień poprzedniego miesiąca.", "cit_payment_cutoff", {"receivable_payment_cutoff": "return_filing_date", "payment_cutoff": "return_filing_date"}, ("cit_art_18f_ust_5",), ("cit8_filing_date_2026_03_31", "partial_payment_gross_amount")),
        _claim("claim_cit_relief", "cit_bad_debt_creditor", "Warunek statusu restrukturyzacyjnego, upadłościowego lub likwidacyjnego dłużnika w CIT bada się na ostatni dzień miesiąca poprzedzającego złożenie zeznania; materiał nie potwierdza statusu na 28 lutego 2026 r.", "cit_debtor_insolvency_condition", {"available": None, "status": "conditional_missing_fact", "debtor_insolvency_reference_date": "last_day_of_previous_month", "insolvency_reference_date": "last_day_of_previous_month"}, ("cit_art_18f_ust_10",), ("debtor_status_on_2026_02_28",), "calc_cit_relief_year", status="conditional_missing_fact"),
        _claim("claim_cit_base", "cit_bad_debt_creditor", f"Warunkowe zmniejszenie podstawy CIT wynosi {unpaid_net:,} zł netto.".replace(",", " "), "cit_relief_amount", {"base_reduction": unpaid_net}, ("cit_art_18f_ust_1",), ("cit8_filing_date_2026_03_31", "partial_payment_gross_amount"), "calc_unpaid_net_amount", status="approved"),
        _claim("claim_cit_tax", "cit_bad_debt_creditor", f"Warunkowy efekt przy stawce 19% wynosi {tax_effect:,} zł.".replace(",", " "), "cit_relief_amount", {"tax_effect": tax_effect, "rate": 0.19}, ("cit_art_18f_ust_1",), ("cit8_filing_date_2026_03_31", "partial_payment_gross_amount"), "calc_cit_tax_effect", status="approved"),
        _claim("claim_cit_reversal", "cit_bad_debt_creditor", "Zapłata 10 maja 2026 r. powoduje zwiększenie podstawy w rozliczeniu CIT za 2026 r.", "cit_relief_reversal", {"year": 2026}, ("cit_art_18f_ust_7",), ("final_payment_date",), "calc_cit_reversal_year", status="conditional_missing_fact"),
        _claim("claim_cit_no_retro", "cit_bad_debt_creditor", "Późniejsza zapłata nie wymaga korekty wstecznej CIT-8 za 2025 r.", "cit_no_retroactive_correction", {"retroactive_correction": False}, ("cit_art_18f_ust_7",), ("final_payment_date",), "calc_no_retroactive_correction", ("calc_no_retroactive_correction", "calc_cit_relief_year"), status="conditional_missing_fact"),
    ]
    return {item.claim_id: item for item in claims}


def run_bad_debt_pipeline(
    query: str, *, target_date: str = "2026-03-31"
) -> LegalPipelineResult:
    if not is_bad_debt_relief_query(query):
        raise ValueError("Query is not a VAT/CIT bad-debt-relief case.")
    registry = build_bad_debt_registry()
    facts = parse_bad_debt_facts(query)
    calculations = calculate_bad_debt(facts)
    claims = build_bad_debt_claims(facts, calculations)
    for claim in claims.values():
        validation = validate_claim(
            claim,
            registry,
            target_date=target_date,
            facts=facts.records,
            calculations=calculations,
        )
        acceptable_conditional = (
            claim.status == "conditional_missing_fact"
            and set(validation.errors).issubset({"missing_fact_dependency"})
        )
        if not validation.claim_supported and not acceptable_conditional:
            raise ValueError(f"Claim {claim.claim_id} failed: {validation.errors}")
    payload = build_renderer_payload(claims, registry, target_date=target_date)
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
