from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Iterable, Optional

from app.legal_pipeline import (
    AnswerPlan,
    AnswerSection,
    CalculationRecord,
    FactRecord,
    LegalClaim,
    ProvisionRecord,
    ProvisionRegistry,
    validate_claim,
)


END_MARKER = "<END_OF_ANALYSIS>"
PLACEHOLDER_RE = re.compile(
    r"zweryfikowany przepis wskazany w źródłach|primary law|\bten przepis\b|TODO|TBD",
    re.IGNORECASE,
)
INTERNAL_RENDER_MARKER_RE = re.compile(
    r"\[(?:claim_id|provision_id|version_id|fact_id|calculation_id):", re.IGNORECASE
)


@dataclass(frozen=True)
class RendererPayload:
    approved_claims: tuple[LegalClaim, ...]
    conditional_claims: tuple[LegalClaim, ...]
    answer_plan: AnswerPlan
    provisions: tuple[dict[str, str], ...]
    calculations: tuple[dict[str, object], ...] = ()
    authority_cards: tuple[dict[str, object], ...] = ()
    interpretation_lane_outcome: dict[str, object] = field(default_factory=dict)
    judgment_lane_outcome: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "approved_claims": [asdict(item) for item in self.approved_claims],
            "conditional_claims": [asdict(item) for item in self.conditional_claims],
            "answer_plan": asdict(self.answer_plan),
            "provisions": [dict(item) for item in self.provisions],
            "calculations": [dict(item) for item in self.calculations],
        }
        if self.authority_cards:
            payload["authority_cards"] = [dict(item) for item in self.authority_cards]
        if self.interpretation_lane_outcome:
            payload["interpretation_lane_outcome"] = dict(self.interpretation_lane_outcome)
        if self.judgment_lane_outcome:
            payload["judgment_lane_outcome"] = dict(self.judgment_lane_outcome)
        return payload


@dataclass(frozen=True)
class RenderValidation:
    passed: bool
    end_marker_present: bool
    missing_claim_ids: tuple[str, ...]
    placeholder_count: int
    unknown_provision_ids: tuple[str, ...]
    thesis_contradictions: tuple[str, ...]
    truncated: bool
    errors: tuple[str, ...]
    missing_required_sections: tuple[str, ...] = ()
    tables_closed: bool = True
    thesis_analysis_duplicate_ratio: float = 0.0


@dataclass(frozen=True)
class LegalPipelineResult:
    claims: dict[str, LegalClaim]
    facts: dict[str, FactRecord]
    calculations: dict[str, CalculationRecord]
    renderer_payload: dict[str, object]
    answer: str
    render_validation: RenderValidation


def _provision(
    provision_id: str,
    document_id: str,
    citation: str,
    text: str,
    effective_from: str,
    effective_to: Optional[str] = None,
) -> ProvisionRecord:
    version_id = f"{document_id}_{effective_from}"
    return ProvisionRecord(
        provision_id=provision_id,
        document_id=document_id,
        version_id=version_id,
        citation=citation,
        article=re.search(r"art\.\s*([0-9a-z]+)", citation, re.I).group(1),
        paragraph=None,
        point=None,
        letter=None,
        text=text,
        effective_from=effective_from,
        effective_to=effective_to,
        status="active",
        source_document_id=document_id,
        source_chunk_ids=(f"{provision_id}:source",),
        source_span_end=len(text),
        display_reference=citation,
    )


def build_mixed_invoice_registry() -> ProvisionRegistry:
    provisions = [
        _provision("vat_art_108a_ust_1", "vat_act", "art. 108a ust. 1 ustawy VAT", "Dobrowolne zastosowanie MPP.", "2018-07-01"),
        _provision("vat_art_108a_ust_1a", "vat_act", "art. 108a ust. 1a ustawy VAT", "Obowiązkowy MPP obejmuje płatność za towary lub usługi z załącznika nr 15, gdy faktura przekracza próg.", "2019-11-01"),
        _provision("cit_art_15d_ust_1_pkt_3", "cit_act", "art. 15d ust. 1 pkt 3 ustawy CIT", "Wyłączenie kosztu przy płatności z pominięciem obowiązkowego MPP.", "2020-01-01"),
        _provision("ord_art_117ba_par_1", "ordynacja", "art. 117ba § 1 Ordynacji podatkowej", "Odpowiedzialność solidarna za płatność na rachunek spoza wykazu.", "2020-01-01"),
        _provision("ord_art_117ba_par_3", "ordynacja", "art. 117ba § 3 Ordynacji podatkowej", "Wyłączenia odpowiedzialności solidarnej.", "2020-01-01"),
        _provision("ord_art_117ba_par_4_historical", "ordynacja", "art. 117ba § 4 Ordynacji podatkowej (wersja historyczna)", "ZAW-NR do organu właściwego dla wystawcy faktury.", "2020-01-01", "2021-12-31"),
        _provision("ord_art_117ba_par_4", "ordynacja", "art. 117ba § 4 Ordynacji podatkowej", "ZAW-NR do naczelnika urzędu skarbowego właściwego dla podatnika dokonującego zapłaty.", "2022-01-01"),
        _provision("vat_art_96b", "vat_act", "art. 96b ustawy VAT", "Wykaz podatników VAT i rachunków rozliczeniowych.", "2019-09-01"),
    ]
    return ProvisionRegistry(provisions=provisions)


def is_mixed_invoice_query(query: str) -> bool:
    text = query.lower()
    return (
        bool(re.search(r"\b(mpp|podzielon\w* płatno\w*|split payment)\b", text))
        and bool(re.search(r"\b(płatnoś[ćc]\s*a|payment\s*a)\b", text))
        and bool(re.search(r"\b(płatnoś[ćc]\s*b|payment\s*b)\b", text))
    )


def _claim(
    claim_id: str,
    conclusion: str,
    result: dict[str, object],
    controlling: tuple[str, ...],
    facts: tuple[str, ...],
    *,
    dependencies: tuple[str, ...] = (),
    calculations: tuple[str, ...] = (),
    status: str = "approved",
) -> LegalClaim:
    return LegalClaim(
        claim_id=claim_id,
        axis_id="mixed_invoice",
        claim_type="calculated_result" if calculations else "legal_conclusion",
        text=conclusion,
        source_provisions=controlling,
        controlling_provisions=controlling,
        dependency_provisions=dependencies,
        fact_dependencies=facts,
        calculation_id=calculations[0] if calculations else None,
        calculation_ids=calculations,
        status=status,  # type: ignore[arg-type]
        result=result,
    )


def _money_after(label: str, query: str, default: int) -> int:
    match = re.search(
        rf"{label}.{{0,40}}?(\d{{1,3}}(?:[ .]\d{{3}})*)\s*zł",
        query,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return default
    return int(re.sub(r"\D", "", match.group(1)))


def build_mixed_invoice_claims(query: str = "") -> tuple[
    dict[str, LegalClaim], dict[str, FactRecord], dict[str, CalculationRecord]
]:
    invoice_total = _money_after(r"faktur\w*", query, 30_000)
    payment_a_net = _money_after(r"(?:płatnoś[ćc]|payment)\s*a", query, 20_000)
    payment_b_net = _money_after(r"(?:płatnoś[ćc]|payment)\s*b", query, 10_000)
    facts = {
        "fact_invoice_total": FactRecord("fact_invoice_total", "money", invoice_total),
        "fact_payment_a_annex15": FactRecord("fact_payment_a_annex15", "bool", True),
        "fact_payment_b_transport_only": FactRecord("fact_payment_b_transport_only", "bool", True),
        "fact_payment_a_net": FactRecord("fact_payment_a_net", "money", payment_a_net),
        "fact_payment_b_net": FactRecord("fact_payment_b_net", "money", payment_b_net),
        "fact_payment_a_white_list": FactRecord("fact_payment_a_white_list", "bool", None, status="missing"),
        "fact_target_date": FactRecord("fact_target_date", "date", "2026-06-30", date="2026-06-30"),
    }
    calculations = {
        "calc_cost_payment_a": CalculationRecord("calc_cost_payment_a", "mpp_cost_gate", {"net": payment_a_net, "mpp_used": False}, 0),
        "calc_cost_payment_b": CalculationRecord("calc_cost_payment_b", "mpp_cost_gate", {"net": payment_b_net, "mpp_required": False}, payment_b_net),
    }
    claims = [
        _claim("claim_mpp_payment_a", "Płatność A podlega obowiązkowemu MPP.", {"mpp_mandatory": True}, ("vat_art_108a_ust_1a",), ("fact_invoice_total", "fact_payment_a_annex15")),
        _claim("claim_mpp_payment_b", "Płatność B nie podlega obowiązkowemu MPP.", {"mpp_mandatory": False}, ("vat_art_108a_ust_1a",), ("fact_invoice_total", "fact_payment_b_transport_only")),
        _claim("claim_mpp_payment_b_voluntary", "Płatność B może zostać dobrowolnie wykonana w MPP.", {"voluntary_mpp_allowed": True}, ("vat_art_108a_ust_1",), ("fact_payment_b_transport_only",)),
        _claim("claim_cost_payment_a", f"Do kosztów nie można zaliczyć {payment_a_net:,} zł netto z Płatności A, jeżeli pominięto obowiązkowy MPP; wynik kosztowy wynosi 0 zł.".replace(",", " "), {"net_cost": 0}, ("cit_art_15d_ust_1_pkt_3",), ("fact_payment_a_net",), calculations=("calc_cost_payment_a",)),
        _claim("claim_cost_payment_b", f"Płatność B pozostaje kosztem w kwocie {payment_b_net:,} zł netto w zakresie reguły MPP.".replace(",", " "), {"net_cost": payment_b_net}, ("cit_art_15d_ust_1_pkt_3",), ("fact_payment_b_net",), calculations=("calc_cost_payment_b",)),
        _claim("claim_joint_liability_payment_a", "Odpowiedzialność solidarna dla Płatności A zależy od rachunku z wykazu i ewentualnego wyłączenia.", {"joint_liability": "conditional_missing_fact"}, ("ord_art_117ba_par_1", "ord_art_117ba_par_3"), ("fact_payment_a_white_list",), dependencies=("vat_art_96b",), status="conditional_missing_fact"),
        _claim("claim_joint_liability_payment_b", "Płatność B nie powoduje odpowiedzialności solidarnej na podstawie analizowanego mechanizmu.", {"joint_liability": False}, ("ord_art_117ba_par_1", "ord_art_117ba_par_3"), ("fact_payment_b_transport_only",), dependencies=("vat_art_96b",)),
        _claim("claim_zaw_nr_authority", "W 2026 r. ZAW-NR składa się do naczelnika urzędu skarbowego właściwego dla płatnika.", {"competent_for": "payer"}, ("ord_art_117ba_par_4",), ("fact_payment_a_white_list",), status="conditional_missing_fact"),
        _claim("claim_zaw_nr_historical_excluded", "Historyczna właściwość organu wystawcy faktury nie ma zastosowania w 2026 r.", {"historical_rule_applies": False}, ("ord_art_117ba_par_4",), ("fact_target_date",)),
        _claim("claim_invoice_scope", "Próg bada się na poziomie całej faktury, lecz obowiązek MPP obejmuje płatność za pozycje z załącznika nr 15.", {"threshold_scope": "invoice", "payment_scope": "annex_15_items"}, ("vat_art_108a_ust_1a",), ("fact_invoice_total",)),
    ]
    return {claim.claim_id: claim for claim in claims}, facts, calculations


def build_renderer_payload(
    claims: dict[str, LegalClaim],
    registry: ProvisionRegistry,
    *,
    target_date: str,
    calculations: Optional[dict[str, CalculationRecord]] = None,
    authority_cards: Iterable[dict[str, object]] = (),
    interpretation_lane_outcome: Optional[dict[str, object]] = None,
    judgment_lane_outcome: Optional[dict[str, object]] = None,
) -> RendererPayload:
    approved: list[LegalClaim] = []
    conditional: list[LegalClaim] = []
    used_ids: set[str] = set()
    for claim in claims.values():
        controlling = claim.controlling_provisions or claim.source_provisions
        resolved = [registry.get(item, target_date) for item in controlling]
        if not resolved or any(item is None or not item.display_reference for item in resolved):
            continue
        provenance = tuple(
            {
                "provision_id": item.provision_id,
                "display_reference": item.display_reference,
                "version_id": item.version_id,
                "source_span": item.text,
                "status": item.status,
            }
            for item in resolved
            if item is not None
        )
        enriched_claim = replace(
            claim,
            provenance=provenance,
            version_id=(
                str(provenance[0]["version_id"]) if len(provenance) == 1 else "multiple"
            ),
        )
        if claim.status == "conditional_missing_fact":
            conditional.append(enriched_claim)
        elif claim.status in {"approved", "supported"}:
            approved.append(enriched_claim)
        used_ids.update(controlling)
        used_ids.update(claim.dependency_provisions)
    provisions = tuple(
        {
            "provision_id": item.provision_id,
            "display_reference": item.display_reference,
            "version_id": item.version_id,
            "source_span": item.text,
        }
        for provision_id in sorted(used_ids)
        for item in [registry.get(provision_id, target_date)]
        if item is not None and item.display_reference
    )
    ordered_ids = tuple(claims)
    axis_sections: list[AnswerSection] = []
    for prefix, title in (("vat_", "VAT"), ("cit_", "CIT")):
        axis_claim_ids = tuple(
            claim_id
            for claim_id, claim in claims.items()
            if claim.axis_id.lower().startswith(prefix)
        )
        if axis_claim_ids:
            axis_sections.append(AnswerSection(prefix.rstrip("_"), title, axis_claim_ids))
    return RendererPayload(
        approved_claims=tuple(approved),
        conditional_claims=tuple(conditional),
        answer_plan=AnswerPlan(
            sections=(
                AnswerSection("thesis", "Teza", ordered_ids),
                AnswerSection("analysis", "Analiza", ordered_ids),
                *axis_sections,
                AnswerSection("sources", "Źródła", ()),
                AnswerSection("risks", "Ryzyka i luki", ()),
            ),
            allowed_claim_ids=ordered_ids,
        ),
        provisions=provisions,
        calculations=tuple(
            {
                "calculation_id": item.calculation_id,
                "operation": item.operation,
                "inputs": item.inputs,
                "result": item.result,
            }
            for item in (calculations or {}).values()
        ),
        authority_cards=tuple(dict(item) for item in authority_cards),
        interpretation_lane_outcome=dict(interpretation_lane_outcome or {}),
        judgment_lane_outcome=dict(judgment_lane_outcome or {}),
    )


def bind_authority_cards_to_claims(
    authority_cards: Iterable[dict[str, object]],
    claims: dict[str, LegalClaim],
) -> tuple[dict[str, object], ...]:
    """Preserve only pairwise authority-to-claim bindings in the trace payload."""
    claim_ids_by_issue: dict[str, list[str]] = {}
    for claim in claims.values():
        claim_ids_by_issue.setdefault(claim.axis_id, []).append(claim.claim_id)
    default_issue = next(iter(claim_ids_by_issue), "general")
    bound: list[dict[str, object]] = []
    for raw_card in authority_cards:
        card = dict(raw_card)
        issue_id = str(card.get("issue_id") or default_issue)
        raw_bindings = card.get("claim_bindings")
        bindings: list[dict[str, object]] = []
        if isinstance(raw_bindings, (list, tuple)):
            for item in raw_bindings:
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("claim_id") or "")
                try:
                    score = float(item.get("score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                reason = str(item.get("reason") or "").strip()
                if claim_id in claims and score > 0 and reason:
                    bindings.append({"claim_id": claim_id, "score": score, "reason": reason})
        if not bindings:
            # Compatibility path for manually supplied cards.  Deliberately bind
            # one claim only: never recreate the old all-claims authority pool.
            fallback_claims = claim_ids_by_issue.get(issue_id) or [next(iter(claims), "")]
            if fallback_claims and fallback_claims[0]:
                bindings.append(
                    {
                        "claim_id": fallback_claims[0],
                        "score": 0.5,
                        "reason": "Ręcznie przekazane źródło przypisano do jednego kontrolowanego wniosku.",
                    }
                )
        claim_ids = [str(item["claim_id"]) for item in bindings]
        card["issue_id"] = issue_id
        card["claim_ids"] = ", ".join(claim_ids)
        card["claim_bindings"] = bindings
        card["binding_score"] = max((float(item["score"]) for item in bindings), default=0.0)
        card["binding_reason"] = "; ".join(str(item["reason"]) for item in bindings)
        card.setdefault("holding", "Brak wystarczającego fragmentu do odtworzenia tezy rozstrzygnięcia.")
        card.setdefault("similarity_reason", "Zwrócono ją dla tego samego zagadnienia podatkowego.")
        card.setdefault("distinguishing_facts", "Porównaj fakty wskazane w pytaniu ze stanem faktycznym źródła.")
        bound.append(card)
    return tuple(bound)


def _references_for_claim(claim: LegalClaim, provision_map: dict[str, str]) -> str:
    return ", ".join(
        provision_map[item]
        for item in (*claim.controlling_provisions, *claim.dependency_provisions)
        if item in provision_map
    )


def _thesis_claims(payload: RendererPayload) -> list[LegalClaim]:
    approved = list(payload.approved_claims)
    preferred_codes = (
        "housing_relief_tax",
        "housing_relief_developer_deadline",
        "credit_on_sold_property_qualified",
        "sale_tax_regime",
    )
    selected = [claim for code in preferred_codes for claim in approved if claim.result_code == code]
    selected.extend(claim for claim in approved if claim not in selected)
    return selected[:3]


def _render_calculation_result(claim: LegalClaim, calculation_map: dict[str, dict[str, object]]) -> str:
    calculation_ids = tuple(dict.fromkeys((*claim.calculation_ids, *((claim.calculation_id,) if claim.calculation_id else ()))))
    details: list[str] = []

    def money(value: object) -> str:
        return f"{int(value):,} zł".replace(",", " ") if isinstance(value, int) else str(value)

    for calculation_id in calculation_ids:
        calculation = calculation_map.get(calculation_id)
        if calculation is None:
            continue
        operation = str(calculation.get("operation") or "")
        inputs = calculation.get("inputs") or {}
        result = calculation.get("result")
        if not isinstance(inputs, dict):
            continue
        if operation == "housing_relief_formula":
            details.append(
                "Wzór D × W / P: "
                f"D = {money(inputs.get('income'))}, W = {money(inputs.get('qualified_expenses'))}, "
                f"P = {money(inputs.get('revenue'))}; wynik = {money(result)}."
            )
        elif operation == "subtract":
            details.append(
                f"Dochód opodatkowany: {money(inputs.get('income'))} − {money(inputs.get('exempt_income'))} = {money(result)}."
            )
        elif operation == "multiply":
            rate = inputs.get("rate")
            rate_text = f"{Decimal(str(rate)) * 100:g}%" if rate is not None else "stawka ustawowa"
            details.append(
                f"PIT: {money(inputs.get('taxable_income'))} × {rate_text} = {money(result)}."
            )
        elif operation == "conditional_add":
            details.append(
                f"Scenariusz z kredytem: W = {money(inputs.get('baseline_qualified_expenses'))} + "
                f"{money(inputs.get('credit_repayment'))} = {money(result)}."
            )
        elif isinstance(result, (int, float, Decimal)) and not isinstance(result, bool):
            details.append(f"Wynik obliczenia: {money(result)}.")
    return (" " + " ".join(details)) if details else ""


def _analysis_explanation(claim: LegalClaim) -> str:
    housing_explanations = {
        "sale_tax_regime": "Sprzedaż nastąpiła przed upływem pięciu pełnych lat liczonych od końca roku nabycia, więc wchodzi do reżimu PIT dla odpłatnego zbycia.",
        "housing_relief_formula": "Zwolnienie jest obliczane proporcją dochodu, wydatków kwalifikowanych i przychodu; wydatku nie wolno utożsamiać z dochodem zwolnionym.",
        "housing_relief_exempt_income": "Najpierw ustala się wydatki kwalifikowane, a dopiero potem część dochodu zwolnioną według ustawowego wzoru.",
        "housing_relief_tax": "Po odjęciu dochodu zwolnionego od dochodu ze sprzedaży pozostała część podlega stawce 19%.",
        "housing_relief_developer_deadline": "Termin biegnie od końca roku sprzedaży; dla wydatku deweloperskiego konieczne jest uzyskanie własności przed jego upływem.",
        "credit_on_sold_property_qualified": "Przepis szczególny dotyczący kredytu ze zbywanej nieruchomości działa wraz z regułą ogólną i nie pozwala pominąć tego wyjątku.",
    }
    return housing_explanations.get(
        claim.result_code,
        "Zastosowanie przepisu wymaga odniesienia jego warunków do wskazanych w pytaniu faktów.",
    )


def _authority_summary(item: dict[str, object]) -> str:
    def words(value: object, limit: int) -> str:
        items = str(value or "").split()
        return " ".join(items[:limit]) + ("…" if len(items) > limit else "")

    raw_source_type = str(item.get("source_type") or "authority")
    source_type = {
        "interpretation": "interpretacja",
        "judgment": "orzeczenie",
    }.get(raw_source_type, raw_source_type)
    label = str(item.get("label") or "brak oznaczenia")
    date = str(item.get("date") or "")
    issue = str(item.get("issue_label") or item.get("issue_id") or "brak")
    holding = words(item.get("holding"), 42)
    outcome = str(item.get("outcome") or "brak jednoznacznego wyniku")
    similarity = words(item.get("similarity_reason"), 14)
    distinction = words(item.get("distinguishing_facts"), 10)
    authority_status = str(item.get("authority_status") or "current_authority")
    status_note = (
        " Status: źródło historyczne — nie jest przedstawiane jako wykładnia obecnego przepisu."
        if authority_status == "historical_authority"
        else ""
    )
    summary = (
        f"- [{source_type}] {label}"
        + (f" ({date})" if date else "")
        + f"; zagadnienie: {issue}. Holding: {holding}. Wynik: {outcome}. "
        + f"Podobieństwo: {similarity}. Różnica: {distinction}.{status_note}"
    )
    source_url = str(item.get("source_url") or "")
    return summary + (f" Link: {source_url}" if source_url else "")


def render_answer(payload: RendererPayload, *, compact: bool = False) -> str:
    provision_map = {item["provision_id"]: item["display_reference"] for item in payload.provisions}
    calculation_map = {
        str(item.get("calculation_id") or ""): item
        for item in payload.calculations
        if str(item.get("calculation_id") or "")
    }
    thesis = [f"- {claim.text}" for claim in _thesis_claims(payload)]
    analysis_by_axis: dict[str, list[str]] = {}
    for claim in payload.approved_claims:
        references = _references_for_claim(claim, provision_map)
        source_note = f" Zastosowane przepisy: {references}." if references else ""
        axis_title = (
            "VAT" if claim.axis_id.lower().startswith("vat_")
            else "CIT" if claim.axis_id.lower().startswith("cit_")
            else ""
        )
        analysis_by_axis.setdefault(axis_title, []).append(
            "- " + _analysis_explanation(claim)
            + source_note
            + _render_calculation_result(claim, calculation_map)
        )
    analysis_parts: list[str] = []
    for axis_title, lines in analysis_by_axis.items():
        if axis_title:
            analysis_parts.append(f"### {axis_title}\n" + "\n".join(lines))
        else:
            analysis_parts.extend(lines)
    sources = [f"- {item['display_reference']}." for item in payload.provisions]
    sources.extend(_authority_summary(item) for item in payload.authority_cards)
    if payload.interpretation_lane_outcome:
        selected_count = int(payload.interpretation_lane_outcome.get("selected_count") or 0)
        status = str(payload.interpretation_lane_outcome.get("status") or "completed")
        if selected_count == 0 and status in {"deadline_exceeded", "error", "completed_with_errors"}:
            sources.append("- Interpretacje: wyszukiwanie nie zostało ukończone; brak wyniku nie oznacza braku relewantnych źródeł.")
        elif selected_count == 0:
            sources.append("- Interpretacje: nie znaleziono dostatecznie relewantnej interpretacji dla tego zagadnienia.")
    if payload.judgment_lane_outcome:
        selected_count = int(payload.judgment_lane_outcome.get("selected_count") or 0)
        status = str(payload.judgment_lane_outcome.get("status") or "completed")
        empty_reason = str(payload.judgment_lane_outcome.get("empty_result_reason") or "")
        if selected_count == 0 and (status in {"deadline_exceeded", "error", "completed_with_errors"} or empty_reason == "retrieval_error"):
            sources.append("- Orzeczenia: wyszukiwanie nie zostało ukończone; brak wyniku nie oznacza braku relewantnych źródeł.")
        elif selected_count == 0:
            sources.append("- Orzeczenia: nie znaleziono dostatecznie relewantnego orzeczenia dla tego zagadnienia.")
    risks = [f"- {claim.text}" for claim in payload.conditional_claims] or [
        "- Brak warunkowych twierdzeń wymagających dodatkowego faktu."
    ]
    return (
        "Teza\n" + "\n".join(thesis) +
        "\n\nAnaliza\n" + "\n".join(analysis_parts) +
        "\n\nŹródła\n" + "\n".join(sources) +
        "\n\nRyzyka i luki\n" + "\n".join(risks) +
        f"\n\n{END_MARKER}"
    )


def validate_rendered_answer(
    answer: str,
    payload: RendererPayload,
) -> RenderValidation:
    thesis_text = answer.partition("\n\nAnaliza\n")[0]
    analysis_text = answer.partition("\n\nAnaliza\n")[2].partition("\n\nŹródła\n")[0]
    missing: tuple[str, ...] = ()
    placeholders = len(PLACEHOLDER_RE.findall(answer))
    sources_text = answer.partition("\n\nŹródła\n")[2].partition("\n\nRyzyka i luki\n")[0]
    known_provisions = {item["display_reference"] for item in payload.provisions}
    # A holding quoted in the Sources section can refer to a neighbouring
    # provision from the authority's own reasoning.  That is evidence, not an
    # application claim.  Primary-law validation must therefore cover the
    # thesis, analysis and risks, while authority quotations remain subject to
    # their separate binding/provenance checks below.
    source_start = answer.find("\n\nŹródła\n")
    source_end = answer.find("\n\nRyzyka i luki\n", source_start) if source_start >= 0 else -1
    answer_for_primary_law_validation = (
        answer[:source_start] + answer[source_end:]
        if source_start >= 0 and source_end >= 0
        else answer
    )
    referenced = set(
        re.findall(
            r"art\.\s*[0-9a-z]+(?:\s+(?:ust\.|pkt|§)\s*[0-9a-z]+)*[^,\n.)]*",
            answer_for_primary_law_validation,
            re.I,
        )
    )
    unknown = tuple(sorted(
        item.strip()
        for item in referenced
        if not any(
            item.split(" (", 1)[0].split(" [", 1)[0].strip() in known
            for known in known_provisions
        )
    ))
    duplicate_ratio = SequenceMatcher(
        None,
        re.sub(r"\s+", " ", thesis_text.lower()),
        re.sub(r"\s+", " ", analysis_text.lower()),
    ).ratio()
    contradictions = tuple(
        claim.claim_id
        for claim in _thesis_claims(payload)
        if claim.text not in thesis_text
    )
    material_claims_without_provenance = tuple(
        claim.claim_id
        for claim in [*payload.approved_claims, *payload.conditional_claims]
        if claim.is_material
        and (
            not claim.claim_id
            or not claim.controlling_provisions
            or not claim.provenance
            or any(
                not source.get("provision_id")
                or not source.get("display_reference")
                or not source.get("version_id")
                or not source.get("source_span")
                for source in claim.provenance
            )
        )
    )
    numeric_claims_without_calculation = tuple(
        claim.claim_id
        for claim in [*payload.approved_claims, *payload.conditional_claims]
        if claim.is_material
        and claim.claim_type == "calculated_result"
        and re.search(r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*zł)?\b", claim.text)
        and not claim.calculation_ids
        and not claim.calculation_id
    )
    calculation_map = {
        str(item.get("calculation_id") or ""): item
        for item in payload.calculations
        if str(item.get("calculation_id") or "")
    }
    calculation_result_mismatches: list[str] = []
    for claim in [*payload.approved_claims, *payload.conditional_claims]:
        linked_ids = tuple(
            item
            for item in (*claim.calculation_ids, *((claim.calculation_id,) if claim.calculation_id else ()))
            if item
        )
        for calculation_id in linked_ids:
            calculation = calculation_map.get(calculation_id)
            if calculation is None:
                continue
            result = calculation.get("result")
            if not _rendered_text_contains_result(answer, result):
                calculation_result_mismatches.append(calculation_id)
    marker = answer.rstrip().endswith(END_MARKER)
    missing_sections_list: list[str] = []
    for section in payload.answer_plan.sections:
        pattern = (
            rf"(?:^|\n)###\s+{re.escape(section.title)}\n"
            if section.title in {"VAT", "CIT"}
            else rf"(?:^|\n){re.escape(section.title)}\n"
        )
        if not re.search(pattern, answer):
            missing_sections_list.append(section.title)
    missing_sections = tuple(missing_sections_list)
    empty_required_sections_list: list[str] = []
    section_titles = [section.title for section in payload.answer_plan.sections]
    for index, title in enumerate(section_titles):
        if title in missing_sections:
            continue
        start_pattern = (
            rf"(?m)^\s*###\s+{re.escape(title)}\s*$"
            if title in {"VAT", "CIT"}
            else rf"(?m)^\s*{re.escape(title)}\s*$"
        )
        start_match = re.search(start_pattern, answer)
        if not start_match:
            continue
        next_titles = section_titles[index + 1 :]
        if title == "Analiza":
            next_titles = [
                next_title
                for next_title in next_titles
                if next_title not in {"VAT", "CIT"}
            ]
        next_patterns = []
        for next_title in next_titles:
            next_patterns.append(
                rf"^\s*###\s+{re.escape(next_title)}\s*$"
                if next_title in {"VAT", "CIT"}
                else rf"^\s*{re.escape(next_title)}\s*$"
            )
        if next_patterns:
            next_match = re.search(
                "(?m)" + "|".join(f"(?:{pattern})" for pattern in next_patterns),
                answer[start_match.end() :],
            )
            section_body = answer[start_match.end() : start_match.end() + next_match.start()] if next_match else answer[start_match.end() :]
        else:
            section_body = answer[start_match.end() :]
        cleaned_body = section_body.replace(END_MARKER, "").strip()
        if not cleaned_body:
            empty_required_sections_list.append(title)
    empty_required_sections = tuple(empty_required_sections_list)
    table_lines = [
        line.strip() for line in answer.splitlines() if line.strip().startswith("|")
    ]
    tables_closed = all(line.endswith("|") and line.count("|") >= 3 for line in table_lines)
    sources_have_exact_markers = bool(payload.provisions and sources_text.strip())
    sources_missing_exact_pairs = tuple(
        item["display_reference"]
        for item in payload.provisions
        if item["display_reference"] not in sources_text
    )
    stripped = answer.removesuffix(END_MARKER).rstrip()
    truncated = not marker or (bool(stripped) and stripped[-1] not in ".!?)]")
    allowed_claim_ids = set(payload.answer_plan.allowed_claim_ids)
    unbound_authorities: list[str] = []
    for authority in payload.authority_cards:
        claim_ids = [
            item.strip()
            for item in str(authority.get("claim_ids") or "").split(",")
            if item.strip()
        ]
        bindings = authority.get("claim_bindings")
        bindings_are_pairwise = isinstance(bindings, list) and all(
            isinstance(item, dict)
            and str(item.get("claim_id") or "") in allowed_claim_ids
            and float(item.get("score") or 0) > 0
            and bool(str(item.get("reason") or "").strip())
            for item in bindings
        )
        strict_card = "authority_score" in authority
        holding_span = authority.get("holding_source_span")
        if (
            not authority.get("issue_id")
            or not claim_ids
            or any(claim_id not in allowed_claim_ids for claim_id in claim_ids)
            or not bindings_are_pairwise
            or not authority.get("holding")
            or not authority.get("similarity_reason")
            or not authority.get("distinguishing_facts")
            or (
                strict_card
                and (
                    authority.get("holding_complete_sentence") is not True
                    or not authority.get("holding_section")
                    or not isinstance(holding_span, dict)
                    or not holding_span.get("chunk_id")
                    or not isinstance(holding_span.get("start"), int)
                    or not isinstance(holding_span.get("end"), int)
                )
            )
        ):
            unbound_authorities.append(str(authority.get("label") or "unknown"))
    exposed_internal_ids = tuple(
        identifier
        for identifier in (
            *payload.answer_plan.allowed_claim_ids,
            *(str(item.get("calculation_id") or "") for item in payload.calculations),
        )
        if identifier and identifier in answer
    )
    errors: list[str] = []
    if not marker:
        errors.append("end_marker_missing")
    if missing:
        errors.append("required_claims_missing")
    if placeholders:
        errors.append("generic_placeholder")
    if unknown:
        errors.append("unknown_provision_reference")
    if contradictions:
        errors.append("thesis_claim_contradiction")
    if duplicate_ratio >= 0.35:
        errors.append("thesis_analysis_excessive_duplication")
    if material_claims_without_provenance:
        errors.append("material_claim_without_complete_provenance")
    if numeric_claims_without_calculation:
        errors.append("numeric_claim_without_calculation_id")
    if calculation_result_mismatches:
        errors.append("formula_result_text_mismatch")
    if truncated:
        errors.append("truncated_output")
    if missing_sections:
        errors.append("required_sections_missing")
    if empty_required_sections:
        errors.append("required_sections_empty")
    if not sources_have_exact_markers or sources_missing_exact_pairs:
        errors.append("sources_missing_exact_references")
    if not tables_closed:
        errors.append("tables_not_closed")
    if INTERNAL_RENDER_MARKER_RE.search(answer) or exposed_internal_ids:
        errors.append("internal_metadata_exposed")
    if unbound_authorities:
        errors.append("authority_card_unbound_or_incomplete")
    return RenderValidation(
        passed=not errors,
        end_marker_present=marker,
        missing_claim_ids=missing,
        placeholder_count=placeholders,
        unknown_provision_ids=tuple(sorted(unknown)),
        thesis_contradictions=contradictions,
        truncated=truncated,
        errors=tuple(errors),
        missing_required_sections=missing_sections,
        tables_closed=tables_closed,
        thesis_analysis_duplicate_ratio=duplicate_ratio,
    )


def _rendered_text_contains_result(answer: str, result: object) -> bool:
    if isinstance(result, bool) or result is None:
        return True
    if isinstance(result, int):
        normalized = f"{result:,}".replace(",", r"[ ,]")
        return bool(re.search(rf"\b(?:{normalized}|{result})\b", answer))
    if isinstance(result, float):
        decimal_value = Decimal(str(result)).normalize()
        normalized = str(decimal_value).replace(".", r"[.,]")
        return bool(re.search(rf"\b{normalized}\b", answer))
    if isinstance(result, Decimal):
        normalized = format(result.normalize(), "f").replace(".", r"[.,]")
        return bool(re.search(rf"\b{normalized}\b", answer))
    return True


def run_legal_pipeline(
    query: str,
    *,
    target_date: str = "2026-06-30",
    authority_cards: Iterable[dict[str, object]] = (),
    interpretation_lane_outcome: Optional[dict[str, object]] = None,
    judgment_lane_outcome: Optional[dict[str, object]] = None,
) -> LegalPipelineResult:
    if not is_mixed_invoice_query(query):
        raise ValueError("No controlled pipeline is registered for this query.")
    registry = build_mixed_invoice_registry()
    claims, facts, calculations = build_mixed_invoice_claims(query)
    for claim in claims.values():
        validation = validate_claim(
            claim,
            registry,
            target_date=target_date,
            facts=facts,
            calculations=calculations,
        )
        if claim.status == "approved" and not validation.claim_supported:
            raise ValueError(f"Claim {claim.claim_id} failed validation: {validation.errors}")
    payload = build_renderer_payload(
        claims,
        registry,
        target_date=target_date,
        calculations=calculations,
        authority_cards=bind_authority_cards_to_claims(authority_cards, claims),
        interpretation_lane_outcome=interpretation_lane_outcome,
        judgment_lane_outcome=judgment_lane_outcome,
    )
    answer = render_answer(payload)
    validation = validate_rendered_answer(answer, payload)
    if not validation.passed:
        answer = render_answer(payload, compact=True)
        validation = validate_rendered_answer(answer, payload)
    if not validation.passed:
        raise RuntimeError(f"Fail-closed render validation: {validation.errors}")
    return LegalPipelineResult(
        claims=claims,
        facts=facts,
        calculations=calculations,
        renderer_payload=payload.to_dict(),
        answer=answer.removesuffix(END_MARKER).rstrip(),
        render_validation=validation,
    )
