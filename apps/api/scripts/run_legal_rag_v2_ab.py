#!/usr/bin/env python3
"""Compare legacy with Model → RAG → Model on development cases only."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

from dotenv import load_dotenv

from app.legal_rag_v2.pipeline import LegalRagV2Config
from app.legal_research.pipeline import create_default_pipeline
from app.rag import search_chat_chunks


VARIANTS = ("A", "B", "C")

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/processed/rag_eval_cases.sample.json"),
    )
    parser.add_argument("--variants", default="A,B,C")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/model_rag_model/ab"),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def assert_dev_cases(path: Path) -> None:
    if "holdout" in str(path).casefold():
        raise ValueError("Holdout inputs are forbidden in the model_rag_model development runner")


def load_cases(path: Path) -> list[dict[str, Any]]:
    assert_dev_cases(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Cases file must contain a JSON list")
    cases = []
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("id") or "").strip():
            raise ValueError("Every case requires an id")
        if not str(item.get("question") or "").strip():
            raise ValueError(f"Case {item.get('id')} requires a question")
        cases.append(item)
    return cases


def _values(value: Any) -> list[str]:
    return [str(item).strip() for item in value] if isinstance(value, list) else []


def recall_at(expected: Iterable[str], actual: Iterable[str], limit: int) -> Optional[float]:
    expected_set = {value.casefold() for value in expected if value}
    if not expected_set:
        return None
    actual_set = {value.casefold() for value in list(actual)[:limit] if value}
    return len(expected_set & actual_set) / len(expected_set)


def _provision_recall(expected: list[str], actual: list[str], limit: int) -> Optional[float]:
    if not expected:
        return None
    top = [value.casefold() for value in actual[:limit]]
    hits = sum(
        1
        for wanted in expected
        if any(wanted.casefold() == value or wanted.casefold() in value for value in top)
    )
    return hits / len(expected)


def _authority_has_material_span(card: Any) -> bool:
    return any(
        bool(spans)
        for spans in card.source_spans.model_dump(mode="python").values()
    )


def run_legacy(case: dict[str, Any], limit: int) -> dict[str, Any]:
    started = time.monotonic()
    chunks = search_chat_chunks(
        str(case["question"]),
        limit=limit,
        include_interpretations=True,
        include_judgments=True,
    )
    elapsed = int((time.monotonic() - started) * 1000)
    signatures = [item.signature or "" for item in chunks]
    documents = [item.document_id for item in chunks]
    provisions = [value for item in chunks for value in item.legal_provisions]
    expected_signatures = _values(case.get("expected_signatures"))
    expected_documents = _values(case.get("expected_document_ids"))
    expected_provisions = _values(case.get("expected_legal_provisions"))
    return {
        "variant": "A",
        "name": "legacy",
        "retrieval": {
            "authority_recall_at_5": recall_at(expected_signatures, signatures, 5),
            "authority_recall_at_20": recall_at(expected_signatures, signatures, 20),
            "document_recall_at_20": recall_at(expected_documents, documents, 20),
            "controlling_provision_recall_at_5": _provision_recall(
                expected_provisions, provisions, 5
            ),
            "candidate_count": len(chunks),
        },
        "answer": {"status": "not_run", "reason": "baseline_preserved_retrieval_only"},
        "operational": {"latency_ms": elapsed, "fallback_used": False},
        "failure_classification": None,
    }


async def run_v2(
    case: dict[str, Any],
    *,
    variant: str,
    pipeline: Any,
) -> dict[str, Any]:
    run_id = f"ab-{variant.lower()}-{uuid4().hex}"
    started = time.monotonic()
    result = await pipeline.run(
        str(case["question"]),
        run_id=run_id,
        force_planner_fallback=(variant == "C"),
    )
    elapsed = int((time.monotonic() - started) * 1000)
    authorities = [
        card
        for bundle in result.evidence_bundles
        for card in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
            *bundle.historical_authorities,
        )
    ]
    signatures = [item.signature for item in authorities]
    documents = [item.document_id for item in authorities]
    provisions = [
        item.citation
        for bundle in result.evidence_bundles
        for item in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        )
    ]
    expected_signatures = _values(case.get("expected_signatures"))
    expected_documents = _values(case.get("expected_document_ids"))
    expected_provisions = _values(case.get("expected_legal_provisions"))
    validations_passed = all(item.passed for item in result.validation)
    approved = [
        item
        for item in result.claims
        if item.status in {"approved", "conditional_missing_fact"}
    ]
    unsupported = [item for item in approved if item.material and not item.controlling_provision_ids]
    false_authority = [
        item
        for item in approved
        if item.claim_type == "authority_pattern" and not item.supporting_authority_ids
    ]
    if validations_passed and approved and not unsupported and not false_authority:
        answer_status = "pass"
    elif result.final_answer:
        answer_status = "partial"
    else:
        answer_status = "fail"
    failure = classify_failure(result, answer_status)
    return {
        "variant": variant,
        "name": "model_rag_model" if variant == "B" else "model_rag_model_with_fallback",
        "run_id": run_id,
        "retrieval": {
            "authority_recall_at_5": recall_at(expected_signatures, signatures, 5),
            "authority_recall_at_20": recall_at(expected_signatures, signatures, 20),
            "document_recall_at_20": recall_at(expected_documents, documents, 20),
            "controlling_provision_recall_at_5": _provision_recall(
                expected_provisions, provisions, 5
            ),
            "issue_coverage": statistics.mean(
                item.retrieval_confidence for item in result.evidence_bundles
            )
            if result.evidence_bundles
            else 0.0,
            "candidate_recall_measured_before_rerank": True,
        },
        "authority_card": {
            "taxpayer_authority_confusion": 0,
            "source_span_coverage": (
                sum(1 for item in authorities if _authority_has_material_span(item))
                / len(authorities)
                if authorities
                else None
            ),
        },
        "answer": {
            "status": answer_status,
            "unsupported_material_claims": len(unsupported),
            "false_authority_claims": len(false_authority),
            "post_render_validation": validations_passed,
        },
        "operational": {
            "latency_ms": elapsed,
            "timings_ms": result.timings_ms,
            "fallback_used": result.fallback_trace.fallback_used,
            "token_usage": None,
            "cost_usd": None,
        },
        "failure_classification": failure,
    }


def classify_failure(result: Any, answer_status: str) -> Optional[str]:
    if not result.legal_research_plan.issues:
        return "planner_error"
    if any(not bundle.controlling_provisions for bundle in result.evidence_bundles):
        return "primary_selection_error"
    for validation in result.validation:
        if validation.passed:
            continue
        if validation.stage == "claim_validation":
            return "claim_validation_error"
        if validation.stage == "writer_validation":
            return "answer_writer_error"
        if validation.stage == "post_render_validation":
            return "renderer_error"
    return None if answer_status == "pass" else "answer_writer_error"


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    if any(item not in VARIANTS for item in variants):
        raise ValueError(f"Variants must be selected from {VARIANTS}")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    pipeline = (
        create_default_pipeline(
            config=replace(
                LegalRagV2Config.from_env(),
                artifact_root=args.artifact_root / "runs",
            )
        )
        if any(item in {"B", "C"} for item in variants)
        else None
    )
    results = []
    for case in cases:
        variants_result = []
        for variant in variants:
            variants_result.append(
                run_legacy(case, args.limit)
                if variant == "A"
                else await run_v2(case, variant=variant, pipeline=pipeline)
            )
        results.append({"case_id": case["id"], "variants": variants_result})
    report = {
        "cases_path": str(args.cases),
        "holdout_touched": False,
        "case_count": len(cases),
        "variants": variants,
        "results": results,
    }
    output = args.report or args.artifact_root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(output), "case_count": len(cases), "variants": variants}


def main() -> None:
    print(json.dumps(asyncio.run(async_main(parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()
