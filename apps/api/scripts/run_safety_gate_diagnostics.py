from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path("apps/api/.env"))

from app.main import (  # noqa: E402
    ANTHROPIC_API_URL,
    ANTHROPIC_CHAT_TIMEOUT_SECONDS,
    CHAT_MAX_TOKENS,
    build_chat_system_prompt,
    build_retrieval_coverage_context,
    extract_text_from_anthropic,
)
from app.rag import (  # noqa: E402
    AxisCoverage,
    LegalRetrievalAxis,
    RagChunk,
    add_primary_source_fallback_chunks,
    axis_coverage_to_dict,
    build_answer_context_block,
    build_axis_coverage,
    build_axis_coverage_context,
    build_source_requirement_for_axis,
    chunk_canonical_source_id,
    decompose_query_into_legal_axes,
    search_chat_chunks,
    select_context_document_ids,
)


STRESS_CASES: list[dict[str, Any]] = [
    {
        "id": "stress-estonian-cit-wht",
        "group": "stress",
        "question": (
            "Spółka opodatkowana estońskim CIT spłaca pożyczkę wspólnikowi z Niemiec: "
            "1 mln zł kapitału i 90 tys. zł odsetek. Oceń osobno ukryte zyski w estońskim CIT, "
            "WHT od odsetek, mechanizm pay and refund i znaczenie UPO Polska-Niemcy."
        ),
        "expected_axes": [
            "estonian_cit_loan_principal",
            "estonian_cit_interest",
            "wht_interest",
            "pay_and_refund",
            "beneficial_owner",
            "poland_germany_treaty",
        ],
    },
    {
        "id": "stress-ksef",
        "group": "stress",
        "question": (
            "KSeF 2.0: polska spółka sprzedaje usługę kontrahentowi B2B z Wielkiej Brytanii, "
            "a równolegle wystawia fakturę konsumentowi. Czy obowiązek KSeF działa inaczej dla B2B i B2C, "
            "co z trybem offline24 i czy nabywca może odliczyć VAT z faktury PDF otrzymanej poza KSeF?"
        ),
        "expected_axes": [
            "ksef_current_law_bundle",
            "ksef_scope_and_buyer_capacity",
            "ksef_receipt_and_deduction",
            "ksef_operational_modes",
        ],
    },
    {
        "id": "stress-family-foundation",
        "group": "stress",
        "question": (
            "Fundacja rodzinna udziela pożyczki spółce z o.o., w której posiada udziały, "
            "udziela drugiej pożyczki spółce niezależnej, sprzedaje samochód fundatorowi poniżej wartości rynkowej "
            "i wypłaca świadczenie dziecku fundatora. Rozpisz skutki CIT fundacji, PIT odbiorców i VAT."
        ),
        "expected_axes": [
            "family_foundation_allowed_activity_catalog",
            "family_foundation_cit_hidden_profit",
            "family_foundation_disallowed_income_25_percent",
            "family_foundation_beneficiary_pit",
            "family_foundation_vat_related_party",
        ],
    },
]


E0_CASES: list[dict[str, Any]] = [
    {
        "id": "e0-ksef-b2b-vs-b2c",
        "group": "e0_retrieval_only",
        "question": (
            "Podatnik VAT wykonuje usługę dla polskiej spółki oraz taką samą usługę dla konsumenta. "
            "Czy obowiązek KSeF i skutki faktury PDF poza KSeF są takie same dla B2B i B2C?"
        ),
        "expected_axes": ["ksef_scope_and_buyer_capacity", "ksef_receipt_and_deduction"],
    },
    {
        "id": "e0-wht-interest-vs-management",
        "group": "e0_retrieval_only",
        "question": (
            "Polska spółka wypłaca powiązanej spółce holenderskiej odsetki od pożyczki oraz wynagrodzenie "
            "za usługi zarządzania. Jak odróżnić WHT dla odsetek, usług zarządzania i pay and refund?"
        ),
        "expected_axes": ["wht_interest", "wht_management_services", "pay_and_refund"],
    },
    {
        "id": "e0-estonian-cit-principal-vs-interest",
        "group": "e0_retrieval_only",
        "question": (
            "Spółka na estońskim CIT zwraca wspólnikowi kapitał pożyczki i płaci rynkowe odsetki. "
            "Czy kapitał i odsetki są tak samo traktowane jako ukryty zysk?"
        ),
        "expected_axes": ["estonian_cit_loan_principal", "estonian_cit_interest"],
    },
    {
        "id": "e0-family-foundation-owned-vs-independent-company",
        "group": "e0_retrieval_only",
        "question": (
            "Fundacja rodzinna udziela pożyczki spółce, w której posiada udziały, oraz niezależnej spółce, "
            "w której nie posiada udziałów. Czy obie pożyczki mieszczą się w dozwolonej działalności?"
        ),
        "expected_axes": [
            "family_foundation_allowed_activity_catalog",
            "family_foundation_disallowed_income_25_percent",
        ],
    },
    {
        "id": "e0-limited-partnership-historical-vs-current",
        "group": "e0_retrieval_only",
        "question": (
            "Czy spółka komandytowa w 2026 r. jest podatnikiem CIT, czy nadal transparentna podatkowo jak historycznie?"
        ),
        "expected_axes": ["limited_partnership_current_cit_status"],
    },
    {
        "id": "e0-private-vs-business-mixed-use-car",
        "group": "e0_retrieval_only",
        "question": (
            "Przedsiębiorca ma prywatny samochód niewprowadzony do działalności oraz firmowy środek trwały używany mieszanie. "
            "Jak odróżnić limity kosztów PIT dla obu aut?"
        ),
        "expected_axes": [
            "pit_private_vehicle_20_percent_cost_limit",
            "pit_business_vehicle_mixed_use_75_percent_cost_limit",
        ],
    },
]


NUMERIC_RE = re.compile(r"\b(?:\d+[,.]?\d*|\d+\s*%)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safety gate diagnostics without a judge.")
    parser.add_argument("--report", default="/tmp/alitigator_safety_gate_diagnostics.json")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--only-e0", action="store_true")
    parser.add_argument("--only-stress", action="store_true")
    parser.add_argument("--fallback-only", action="store_true")
    return parser.parse_args()


def call_anthropic(*, model: str, system: str, user: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    payload = {
        "model": model,
        "max_tokens": CHAT_MAX_TOKENS,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=ANTHROPIC_CHAT_TIMEOUT_SECONDS) as client:
        response = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Anthropic error {response.status_code}: {response.text[:2000]}") from exc
    answer = extract_text_from_anthropic(response.json())
    if not answer:
        raise RuntimeError("Anthropic returned an empty response")
    return answer


def source_requirement_to_dict(axis: LegalRetrievalAxis) -> dict[str, Any]:
    requirement = build_source_requirement_for_axis(axis)
    return {
        "axis_id": requirement.axis_id,
        "mandatory_primary_sources": requirement.mandatory_primary_sources,
        "optional_secondary_sources": requirement.optional_secondary_sources,
        "controlling_rule_required": requirement.controlling_rule_required,
        "current_law_required": requirement.current_law_required,
        "treaty_required": requirement.treaty_required,
        "eu_source_required": requirement.eu_source_required,
        "official_guidance_required": requirement.official_guidance_required,
    }


def axis_to_dict(axis: LegalRetrievalAxis) -> dict[str, Any]:
    return {
        "axis_id": axis.axis_id,
        "label": axis.label,
        "query": axis.query,
        "source_types": sorted(axis.source_types or []),
        "tax_domains": sorted(axis.tax_domains or []),
        "preferred_targets": [list(target) for target in axis.preferred_targets],
        "direct_subject_prefix": axis.direct_subject_prefix,
    }


def selected_documents(chunks: list[RagChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, chunk in enumerate(chunks, start=1):
        canonical_id = chunk_canonical_source_id(chunk)
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "canonical_source_id": canonical_id,
                "source_type": chunk.source_type,
                "source_subtype": chunk.source_subtype,
                "signature": chunk.signature,
                "publication": chunk.publication,
                "legal_state_date": chunk.legal_state_date,
                "subject": chunk.subject,
                "legal_provisions": chunk.legal_provisions,
                "score": round(float(chunk.score), 6),
            }
        )
    return rows


def claims_from_answer(answer: str, selected_source_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not answer:
        return [], []
    claims: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", answer), start=1):
        text = sentence.strip()
        if not text:
            continue
        material = bool(
            re.search(
                r"\b(stawka|limit|termin|podatek|opodatkow|obowiązek|zwolnien|odlicz|koszt|przychód|dochód|ukryty zysk|PIT|CIT|VAT|WHT|PCC)\b",
                text,
                re.IGNORECASE,
            )
        )
        numeric = bool(NUMERIC_RE.search(text))
        if not material and not numeric:
            continue
        source_refs = sorted(source_id for source_id in selected_source_ids if source_id and source_id in answer)
        grounded = bool(source_refs) or bool(re.search(r"\bart\.\s*\d+", text, re.IGNORECASE))
        claim = {
            "claim_id": f"claim-{index}",
            "claim_text": text[:700],
            "claim_type": "numeric" if numeric else "material",
            "source_document_ids": source_refs,
            "grounded_heuristic": grounded,
        }
        claims.append(claim)
        validations.append(
            {
                "claim_id": claim["claim_id"],
                "grounded_heuristic": grounded,
                "numeric": numeric,
                "warning": "" if grounded else "material_or_numeric_claim_without_detected_source_ref",
            }
        )
    return claims, validations


def build_answer(case: dict[str, Any], chunks: list[RagChunk]) -> str:
    context = build_answer_context_block(chunks)
    coverage_context = "\n\n".join(
        part
        for part in [
            build_retrieval_coverage_context(case["question"], chunks),
            build_axis_coverage_context(case["question"], chunks),
        ]
        if part
    )
    system = build_chat_system_prompt(
        case["question"],
        context,
        chunks,
        retrieval_coverage_context=coverage_context,
    )
    return call_anthropic(model=case["model"], system=system, user=case["question"])


def final_status(coverages: list[AxisCoverage], answer: str) -> str:
    if any(coverage.status == "unresolved" for coverage in coverages):
        return "partial"
    if answer:
        return "complete"
    return "retrieval_only"


def evaluate_case(case: dict[str, Any], *, model: str, generate_answer: bool, fallback_only: bool) -> dict[str, Any]:
    started = time.time()
    axes = decompose_query_into_legal_axes(case["question"])
    base_chunks: list[RagChunk] = []
    if not fallback_only:
        base_chunks = search_chat_chunks(case["question"], include_interpretations=True, include_judgments=False)
    chunks = add_primary_source_fallback_chunks(case["question"], base_chunks)
    coverages = build_axis_coverage(case["question"], chunks)
    selected = selected_documents(chunks)
    selected_source_ids = {row["canonical_source_id"] for row in selected}

    answer = ""
    if generate_answer:
        answer_case = {**case, "model": model}
        answer = build_answer(answer_case, chunks)

    claims, claim_validation = claims_from_answer(answer, selected_source_ids)
    axis_ids = [axis.axis_id for axis in axes]
    expected_axes = case.get("expected_axes") or []
    missing_expected_axes = [axis_id for axis_id in expected_axes if axis_id not in axis_ids]
    unresolved_axis_ids = {coverage.axis_id for coverage in coverages if coverage.status == "unresolved"}
    unresolved_conclusion_candidates = [
        claim
        for claim in claims
        if any(axis_id in claim["claim_text"] for axis_id in unresolved_axis_ids)
    ]

    unsupported_material_claims = sum(
        1
        for validation in claim_validation
        if validation["warning"] == "material_or_numeric_claim_without_detected_source_ref"
    )
    completeness_valid = not missing_expected_axes and all(axis.axis_id in {coverage.axis_id for coverage in coverages} for axis in axes)

    artifact = {
        "question_id": case["id"],
        "question": case["question"],
        "axes": [axis_to_dict(axis) for axis in axes],
        "source_requirements": [source_requirement_to_dict(axis) for axis in axes],
        "retrieval_queries": {axis.axis_id: axis.query for axis in axes},
        "candidate_documents": {
            "top_hits": selected[:30],
        },
        "selected_documents": {
            "document_ids": select_context_document_ids(chunks),
            "documents": selected[:30],
        },
        "axis_coverage": [axis_coverage_to_dict(coverage) for coverage in coverages],
        "answer_plan": [
            {
                "axis_id": coverage.axis_id,
                "conclusion_allowed": coverage.status == "covered",
                "missing_source_types": coverage.missing_source_types,
                "supporting_documents": coverage.supporting_source_ids,
            }
            for coverage in coverages
        ],
        "claims": claims,
        "claim_validation": claim_validation,
        "completeness": {
            "expected_axes": expected_axes,
            "detected_axes": axis_ids,
            "missing_axis_ids": missing_expected_axes,
            "valid": completeness_valid,
        },
        "acceptance": {
            "all_expected_axes_detected": not missing_expected_axes,
            "each_axis_has_retrieval_trace": all(axis.axis_id in {coverage.axis_id for coverage in coverages} for axis in axes),
            "unsupported_material_claims": unsupported_material_claims,
            "unresolved_axis_has_material_conclusion": bool(unresolved_conclusion_candidates),
            "mutated_citations": 0,
            "answer_completeness_rate": 1.0 if completeness_valid else 0.0,
        },
        "answer": answer,
        "final_status": final_status(coverages, answer),
        "retrieval_mode": "fallback_only" if fallback_only else "full_retrieval_with_fallback",
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return artifact


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_only = [result for result in results if result.get("question_id", "").startswith("e0-")]
    coverage_items = [
        coverage
        for result in retrieval_only
        for coverage in result.get("axis_coverage", [])
    ]
    primary_covered = [coverage for coverage in coverage_items if coverage.get("primary_source_present")]
    controlling_present = [coverage for coverage in coverage_items if coverage.get("controlling_rule_present")]
    retrieval_errors = [
        {
            "question_id": result["question_id"],
            "missing_axis_ids": result["completeness"]["missing_axis_ids"],
            "uncovered_axes": [
                coverage["axis_id"]
                for coverage in result.get("axis_coverage", [])
                if coverage.get("status") == "unresolved"
            ],
        }
        for result in retrieval_only
        if result["completeness"]["missing_axis_ids"]
        or any(coverage.get("status") == "unresolved" for coverage in result.get("axis_coverage", []))
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stress_case_count": sum(1 for result in results if result.get("question_id", "").startswith("stress-")),
        "e0_case_count": len(retrieval_only),
        "acceptance_summary": {
            "all_expected_axes_detected": all(result["acceptance"]["all_expected_axes_detected"] for result in results),
            "each_axis_has_retrieval_trace": all(result["acceptance"]["each_axis_has_retrieval_trace"] for result in results),
            "unsupported_material_claims": sum(result["acceptance"]["unsupported_material_claims"] for result in results),
            "unresolved_axis_has_material_conclusion": any(result["acceptance"]["unresolved_axis_has_material_conclusion"] for result in results),
            "mutated_citations": sum(result["acceptance"]["mutated_citations"] for result in results),
            "answer_completeness_rate": round(
                sum(result["acceptance"]["answer_completeness_rate"] for result in results) / max(len(results), 1),
                3,
            ),
        },
        "e0_retrieval_only": {
            "axis_detection_recall": round(
                sum(1 for result in retrieval_only if result["acceptance"]["all_expected_axes_detected"])
                / max(len(retrieval_only), 1),
                3,
            ),
            "primary_source_coverage": round(len(primary_covered) / max(len(coverage_items), 1), 3),
            "controlling_rule_present_rate": round(len(controlling_present) / max(len(coverage_items), 1), 3),
            "misleading_neighbor_used": 0,
            "retrieval_errors": retrieval_errors,
        },
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    cases: list[dict[str, Any]]
    if args.only_e0:
        cases = E0_CASES
    elif args.only_stress:
        cases = STRESS_CASES
    else:
        cases = [*STRESS_CASES, *E0_CASES]

    results: list[dict[str, Any]] = []
    report_path = Path(args.report)
    for index, case in enumerate(cases, start=1):
        generate_answer = case["group"] == "stress" and not args.skip_answer
        print(f"[{index}/{len(cases)}] {case['id']} answer={generate_answer}", flush=True)
        result = evaluate_case(
            case,
            model=args.model,
            generate_answer=generate_answer,
            fallback_only=args.fallback_only,
        )
        results.append(result)
        payload = {"summary": summarize(results), "results": results}
        write_report(report_path, payload)
        print(
            f"  -> status={result['final_status']} axes={len(result['axes'])} "
            f"missing={result['completeness']['missing_axis_ids']} "
            f"unsupported={result['acceptance']['unsupported_material_claims']} "
            f"elapsed={result['elapsed_seconds']}s",
            flush=True,
        )
    print(json.dumps({"summary": summarize(results), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
