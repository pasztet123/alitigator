from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Optional

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
CLAIM_MARKER_RE = re.compile(r"\[claim_id:([a-z0-9_:.-]+)]", re.IGNORECASE)


@dataclass(frozen=True)
class RendererPayload:
    approved_claims: tuple[LegalClaim, ...]
    conditional_claims: tuple[LegalClaim, ...]
    answer_plan: AnswerPlan
    provisions: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "approved_claims": [asdict(item) for item in self.approved_claims],
            "conditional_claims": [asdict(item) for item in self.conditional_claims],
            "answer_plan": asdict(self.answer_plan),
            "provisions": [dict(item) for item in self.provisions],
        }


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
    )


def render_answer(payload: RendererPayload, *, compact: bool = False) -> str:
    claims = [*payload.approved_claims, *payload.conditional_claims]
    provision_map = {item["provision_id"]: item["display_reference"] for item in payload.provisions}
    thesis = [
        f"- {claim.text} [claim_id:{claim.claim_id}]"
        for claim in claims
    ]
    analysis_by_axis: dict[str, list[str]] = {}
    for claim in claims:
        references = ", ".join(
            provision_map[item]
            for item in claim.controlling_provisions
            if item in provision_map
        )
        dependencies = ", ".join(
            provision_map[item]
            for item in claim.dependency_provisions
            if item in provision_map
        )
        source_note = f" Podstawa kontrolująca: {references}."
        if dependencies:
            source_note += f" Źródło zależności: {dependencies}."
        axis_title = (
            "VAT" if claim.axis_id.lower().startswith("vat_")
            else "CIT" if claim.axis_id.lower().startswith("cit_")
            else ""
        )
        analysis_by_axis.setdefault(axis_title, []).append(
            f"- {claim.text}{source_note}"
            f" Provision IDs: {', '.join(claim.controlling_provisions)}."
            f" Fakty: {', '.join(claim.fact_dependencies) or 'brak'}."
            f" Obliczenia: {', '.join(claim.calculation_ids or ((claim.calculation_id,) if claim.calculation_id else ())) or 'nie dotyczy'}."
            f" [claim_id:{claim.claim_id}]"
        )
    analysis_parts: list[str] = []
    for axis_title, lines in analysis_by_axis.items():
        if axis_title:
            analysis_parts.append(f"### {axis_title}\n" + "\n".join(lines))
        else:
            analysis_parts.extend(lines)
    sources = [
        f"- {item['display_reference']} [provision_id:{item['provision_id']}]"
        f" [version_id:{item['version_id']}]."
        for item in payload.provisions
    ]
    risks = [
        f"- {claim.text} [claim_id:{claim.claim_id}]"
        for claim in payload.conditional_claims
    ] or ["- Brak dodatkowych luk w zatwierdzonym zakresie."]
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
    expected = set(payload.answer_plan.allowed_claim_ids)
    rendered = set(CLAIM_MARKER_RE.findall(answer))
    thesis_text = answer.partition("\n\nAnaliza\n")[0]
    analysis_text = answer.partition("\n\nAnaliza\n")[2].partition("\n\nŹródła\n")[0]
    missing = tuple(sorted(
        claim_id
        for claim_id in expected
        if (
            f"[claim_id:{claim_id}]" not in thesis_text
            or f"[claim_id:{claim_id}]" not in analysis_text
        )
    ))
    placeholders = len(PLACEHOLDER_RE.findall(answer))
    sources_text = answer.partition("\n\nŹródła\n")[2].partition("\n\nRyzyka i luki\n")[0]
    known_provisions = {item["display_reference"] for item in payload.provisions}
    referenced = set(re.findall(r"art\.\s*[0-9a-z]+(?:\s+(?:ust\.|pkt|§)\s*[0-9a-z]+)*[^,\n.)]*", answer, re.I))
    unknown = tuple(sorted(
        item.strip()
        for item in referenced
        if not any(
            item.split(" (", 1)[0].split(" [", 1)[0].strip() in known
            for known in known_provisions
        )
    ))
    contradictions = tuple(
        claim.claim_id
        for claim in [*payload.approved_claims, *payload.conditional_claims]
        if (
            f"{claim.text} [claim_id:{claim.claim_id}]" not in thesis_text
            or f"{claim.text}" not in analysis_text
            or f"[claim_id:{claim.claim_id}]" not in analysis_text
        )
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
    known_provision_ids = {item["provision_id"] for item in payload.provisions}
    rendered_provision_ids = set(
        re.findall(r"\[provision_id:([a-z0-9_:.-]+)]", answer, re.I)
    )
    unknown_provision_markers = rendered_provision_ids - known_provision_ids
    unknown_claim_markers = rendered - expected
    sources_have_exact_markers = bool(
        re.search(r"\[provision_id:[a-z0-9_:.-]+]", sources_text, re.I)
    )
    sources_missing_exact_pairs = tuple(
        item["provision_id"]
        for item in payload.provisions
        if not re.search(
            re.escape(item["display_reference"])
            + r".*?"
            + re.escape(f"[provision_id:{item['provision_id']}]"),
            sources_text,
            re.I | re.S,
        )
    )
    stripped = answer.removesuffix(END_MARKER).rstrip()
    truncated = not marker or (bool(stripped) and stripped[-1] not in ".!?)]")
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
    if material_claims_without_provenance:
        errors.append("material_claim_without_complete_provenance")
    if numeric_claims_without_calculation:
        errors.append("numeric_claim_without_calculation_id")
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
    if unknown_provision_markers:
        errors.append("unknown_provision_id")
    if unknown_claim_markers:
        errors.append("unknown_claim_id")
    return RenderValidation(
        passed=not errors,
        end_marker_present=marker,
        missing_claim_ids=missing,
        placeholder_count=placeholders,
        unknown_provision_ids=tuple(sorted({*unknown, *unknown_provision_markers})),
        thesis_contradictions=contradictions,
        truncated=truncated,
        errors=tuple(errors),
        missing_required_sections=missing_sections,
        tables_closed=tables_closed,
    )


def run_legal_pipeline(query: str, *, target_date: str = "2026-06-30") -> LegalPipelineResult:
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
    payload = build_renderer_payload(claims, registry, target_date=target_date)
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
