from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.hybrid_authority_rag import (
    baseline_retrieval_for_comparison,
    build_structured_claim_inputs,
    run_hybrid_authority_retrieval,
    to_jsonable,
    write_hybrid_trace_artifacts,
)
from app.rag import RagChunk, build_axis_coverage, list_citations


VARIANTS = (
    ("A", "baseline", False),
    ("B", "hybrid_authority", False),
    ("C", "hybrid_authority_clarifier", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/C hybrid authority RAG comparison on dev cases only.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/processed/rag_eval_cases.sample.json"),
        help="Dev/seed benchmark cases. Holdout paths are rejected.",
    )
    parser.add_argument(
        "--exclude-cases",
        type=Path,
        help="Optional dev exclusion file. Use this for holdout exclusions; do not pass holdout as --cases.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/hybrid_rag_experiment"),
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Safety override for local diagnostics only. Do not use for the experiment.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    cases: list[dict[str, Any]] = []
    for position, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case {position} in {path} must be an object")
        case_id = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        if not case_id or not question:
            raise ValueError(f"Case {position} in {path} needs id and question")
        cases.append(item)
    return cases


def normalize_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clarification_fixture(case: dict[str, Any]) -> dict[str, str]:
    raw = (
        case.get("clarifier_answers")
        or case.get("clarification_answers")
        or case.get("clarifications")
        or {}
    )
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if str(value).strip()}
    if isinstance(raw, list):
        fixture: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                key = str(item.get("id") or item.get("dimension") or "").strip()
                value = str(item.get("answer") or item.get("value") or "").strip()
                if key and value:
                    fixture[key] = value
        return fixture
    return {}


def first_match_rank(expected: list[str], actual: list[str]) -> Optional[int]:
    expected_set = {item for item in expected if item}
    if not expected_set:
        return None
    for rank, value in enumerate(actual, start=1):
        if value in expected_set:
            return rank
    return None


def recall_at(expected: list[str], actual: list[str], limit: int) -> Optional[float]:
    if not expected:
        return None
    top = set(actual[:limit])
    return len([value for value in expected if value in top]) / len(expected)


def provision_recall_at(expected: list[str], chunks: list[RagChunk], limit: int) -> Optional[float]:
    if not expected:
        return None
    expected_lower = [value.lower() for value in expected]
    actual = [
        provision.lower()
        for chunk in chunks[:limit]
        for provision in chunk.legal_provisions
    ]
    hits = 0
    for wanted in expected_lower:
        if any(wanted == value or wanted in value for value in actual):
            hits += 1
    return hits / len(expected_lower)


def authority_type_coverage(chunks: list[RagChunk]) -> float:
    present = {chunk.source_type for chunk in chunks if chunk.source_type in {"interpretation", "judgment"}}
    return len(present) / 2


def temporal_mismatch_rate(chunks: list[RagChunk]) -> float:
    checked = 0
    mismatches = 0
    for chunk in chunks:
        raw_date = (chunk.legal_state_date or chunk.published_date or "")[:10]
        if not raw_date:
            continue
        checked += 1
        try:
            year = int(raw_date[:4])
        except ValueError:
            continue
        if year < 2017:
            mismatches += 1
    return mismatches / checked if checked else 0.0


def baseline_metrics(case: dict[str, Any], chunks: list[RagChunk], elapsed_ms: int) -> dict[str, Any]:
    expected_signatures = normalize_values(case.get("expected_signatures"))
    expected_documents = normalize_values(case.get("expected_document_ids"))
    expected_provisions = normalize_values(case.get("expected_legal_provisions"))
    signatures = [chunk.signature or "" for chunk in chunks]
    documents = [chunk.document_id for chunk in chunks]
    axis_coverage = build_axis_coverage(str(case["question"]), chunks)
    issue_coverage = (
        statistics.mean(coverage.coverage_score for coverage in axis_coverage)
        if axis_coverage
        else (1.0 if chunks else 0.0)
    )
    first_signature_rank = first_match_rank(expected_signatures, signatures)
    first_document_rank = first_match_rank(expected_documents, documents)
    return {
        "expected_signature_recall_at_5": recall_at(expected_signatures, signatures, 5),
        "expected_signature_recall_at_10": recall_at(expected_signatures, signatures, 10),
        "expected_signature_recall_at_20": recall_at(expected_signatures, signatures, 20),
        "mandatory_authority_recall_at_5": recall_at(expected_signatures, signatures, 5),
        "mandatory_authority_recall_at_20": recall_at(expected_signatures, signatures, 20),
        "controlling_provision_recall_at_5": provision_recall_at(expected_provisions, chunks, 5),
        "controlling_provision_recall_at_10": provision_recall_at(expected_provisions, chunks, 10),
        "wrong_neighbor_rate_at_5": 0.0,
        "temporal_mismatch_rate": temporal_mismatch_rate(chunks),
        "issue_coverage": issue_coverage,
        "authority_type_coverage": authority_type_coverage(chunks),
        "first_signature_rank": first_signature_rank,
        "first_document_rank": first_document_rank,
        "latency_ms": elapsed_ms,
        "retrieval_latency_ms": elapsed_ms,
        "reranking_latency_ms": 0,
        "authority_card_latency_ms": 0,
        "answer_latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "context_characters": sum(len(chunk.chunk_text) for chunk in chunks),
        "documents_considered": len(chunks),
        "documents_sent_to_answer_model": len(chunks),
        "answer_pass": 0,
        "answer_partial": 0,
        "answer_fail": 0,
        "answer_not_run": True,
    }


def hybrid_metrics(case: dict[str, Any], result: Any) -> dict[str, Any]:
    chunks = list(result.selected_chunks)
    expected_signatures = normalize_values(case.get("expected_signatures"))
    expected_documents = normalize_values(case.get("expected_document_ids"))
    expected_provisions = normalize_values(case.get("expected_legal_provisions"))
    signatures = [chunk.signature or "" for chunk in chunks]
    documents = [chunk.document_id for chunk in chunks]
    wrong_neighbors = [
        item
        for item in result.reranked_documents[:5]
        if "wrong neighbor" in " ".join(item.get("negative_reasons") or [])
    ]
    card_metrics = {
        "taxpayer_position_authority_holding_confusion": 0,
        "authority_holding_extraction_accuracy": None,
        "court_outcome_extraction_accuracy": None,
        "cited_provision_extraction_accuracy": None,
        "temporal_status_accuracy": None,
        "distinguishing_fact_accuracy": None,
    }
    clarifier_metrics = {
        "clarification_questions_count": len(result.clarifier.questions),
        "questions_materially_relevant": None,
        "questions_answerable_from_original_query": 0,
        "retrieval_gain_after_clarification": None,
        "answer_gain_after_clarification": None,
    }
    return {
        "expected_signature_recall_at_5": recall_at(expected_signatures, signatures, 5),
        "expected_signature_recall_at_10": recall_at(expected_signatures, signatures, 10),
        "expected_signature_recall_at_20": recall_at(expected_signatures, signatures, 20),
        "mandatory_authority_recall_at_5": recall_at(expected_signatures, signatures, 5),
        "mandatory_authority_recall_at_20": recall_at(expected_signatures, signatures, 20),
        "controlling_provision_recall_at_5": provision_recall_at(expected_provisions, chunks, 5),
        "controlling_provision_recall_at_10": provision_recall_at(expected_provisions, chunks, 10),
        "wrong_neighbor_rate_at_5": len(wrong_neighbors) / 5,
        "temporal_mismatch_rate": temporal_mismatch_rate(chunks),
        "issue_coverage": statistics.mean(bundle.retrieval_confidence for bundle in result.evidence_bundles) if result.evidence_bundles else 0.0,
        "authority_type_coverage": authority_type_coverage(chunks),
        "first_signature_rank": first_match_rank(expected_signatures, signatures),
        "first_document_rank": first_match_rank(expected_documents, documents),
        "latency_ms": result.timings.get("total_ms", 0),
        "retrieval_latency_ms": result.timings.get("primary_retrieval_ms", 0) + result.timings.get("authority_retrieval_ms", 0),
        "reranking_latency_ms": result.timings.get("reranking_ms", 0),
        "authority_card_latency_ms": result.timings.get("authority_retrieval_ms", 0),
        "answer_latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "context_characters": sum(len(chunk.chunk_text) for chunk in chunks),
        "documents_considered": len(result.candidate_documents),
        "documents_sent_to_answer_model": len(chunks),
        "answer_pass": 0,
        "answer_partial": 0,
        "answer_fail": 0,
        "answer_not_run": True,
        "authority_card_metrics": card_metrics,
        "clarifier_metrics": clarifier_metrics,
    }


def evaluate_variant(case: dict[str, Any], variant_id: str, name: str, clarifier: bool, limit: int) -> dict[str, Any]:
    question = str(case["question"])
    started = time.perf_counter()
    if variant_id == "A":
        chunks = list(
            baseline_retrieval_for_comparison(
                question,
                limit=limit,
                include_interpretations=True,
                include_judgments=True,
            )
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        structured_inputs = build_structured_claim_inputs(question, chunks)
        return {
            "variant": variant_id,
            "name": name,
            "retrieval_mode": "baseline",
            "metrics": baseline_metrics(case, chunks, elapsed_ms),
            "citations": list_citations(chunks),
            "top_hits": [chunk_hit(chunk, rank) for rank, chunk in enumerate(chunks, start=1)],
            "structured_inputs": structured_inputs,
            "failure_category": infer_failure_category(case, chunks, None),
        }

    result = run_hybrid_authority_retrieval(
        question,
        include_interpretations=True,
        include_judgments=True,
        clarifier_enabled=clarifier,
        clarification_fixture=clarification_fixture(case) if clarifier else None,
    )
    write_hybrid_trace_artifacts(
        result,
        renderer_payload={
            "benchmark_variant": variant_id,
            "case_id": case.get("id"),
            "answer_not_run": True,
        },
        validation={"benchmark": "retrieval_only"},
    )
    chunks = list(result.selected_chunks)
    return {
        "variant": variant_id,
        "name": name,
        "retrieval_mode": result.retrieval_mode,
        "run_id": result.run_id,
        "metrics": hybrid_metrics(case, result),
        "citations": list_citations(chunks),
        "top_hits": [chunk_hit(chunk, rank) for rank, chunk in enumerate(chunks, start=1)],
        "hybrid_trace_summary": {
            "intent_profile": to_jsonable(result.intent_profile),
            "clarifier": to_jsonable(result.clarifier),
            "issue_graph": to_jsonable(result.issue_graph),
            "evidence_bundles": to_jsonable(result.evidence_bundles),
        },
        "failure_category": infer_failure_category(case, chunks, result),
    }


def chunk_hit(chunk: RagChunk, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "signature": chunk.signature,
        "source_type": chunk.source_type,
        "source_subtype": chunk.source_subtype,
        "subject": chunk.subject,
        "published_date": chunk.published_date,
        "legal_state_date": chunk.legal_state_date,
        "legal_provisions": list(chunk.legal_provisions),
    }


def infer_failure_category(case: dict[str, Any], chunks: list[RagChunk], hybrid_result: Any) -> Optional[str]:
    expected_signatures = normalize_values(case.get("expected_signatures"))
    expected_provisions = normalize_values(case.get("expected_legal_provisions"))
    if expected_signatures and recall_at(expected_signatures, [chunk.signature or "" for chunk in chunks], 20) == 0:
        if hybrid_result and not hybrid_result.candidate_documents:
            return "authority_retrieval_error"
        return "document_selection_error"
    if expected_provisions and provision_recall_at(expected_provisions, chunks, 10) == 0:
        return "primary_retrieval_error"
    return None


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for case_result in results:
        for variant_result in case_result["variants"]:
            by_variant.setdefault(variant_result["variant"], []).append(variant_result)
    summary: dict[str, Any] = {}
    for variant, items in by_variant.items():
        metric_names = sorted({key for item in items for key in item["metrics"] if isinstance(item["metrics"].get(key), (int, float))})
        summary[variant] = {
            "case_count": len(items),
            "metrics": {
                name: round(statistics.mean(float(item["metrics"][name]) for item in items if item["metrics"].get(name) is not None), 4)
                for name in metric_names
                if any(item["metrics"].get(name) is not None for item in items)
            },
            "failures_by_category": count_by(
                item.get("failure_category") or "none"
                for item in items
            ),
        }
    return summary


def count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def write_report(artifact_root: Path, payload: dict[str, Any]) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    results_path = artifact_root / "comparison_results.json"
    report_path = artifact_root / "comparison_report.md"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Authority RAG Comparison",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Cases: {payload['case_count']}",
        "",
        "## Aggregate Results",
        "",
    ]
    for variant, summary in payload["summary"].items():
        lines.append(f"### Variant {variant}")
        for key in (
            "mandatory_authority_recall_at_20",
            "expected_signature_recall_at_20",
            "controlling_provision_recall_at_10",
            "wrong_neighbor_rate_at_5",
            "issue_coverage",
            "latency_ms",
        ):
            value = summary["metrics"].get(key)
            if value is not None:
                lines.append(f"- {key}: {value}")
        lines.append(f"- failures_by_category: {summary['failures_by_category']}")
        lines.append("")
    lines.extend(["## Per Case", ""])
    for case in payload["results"]:
        lines.append(f"### {case['id']}")
        lines.append(case["question"])
        for variant in case["variants"]:
            metrics = variant["metrics"]
            lines.append(
                f"- {variant['variant']} {variant['name']}: "
                f"authority@20={metrics.get('mandatory_authority_recall_at_20')} "
                f"primary@10={metrics.get('controlling_provision_recall_at_10')} "
                f"latency_ms={metrics.get('latency_ms')} "
                f"failure={variant.get('failure_category') or 'none'}"
            )
        lines.append("")
    lines.extend(render_delta_sections(payload["results"]))
    lines.append("## Recommendation")
    lines.append("")
    lines.append(recommendation(payload["summary"]))
    lines.append("")
    return "\n".join(lines)


def render_delta_sections(results: list[dict[str, Any]]) -> list[str]:
    improved: list[str] = []
    regressed: list[str] = []
    clarifier_helped: list[str] = []
    clarifier_redundant: list[str] = []
    for case in results:
        variants = {item["variant"]: item for item in case["variants"]}
        a = variants["A"]["metrics"].get("mandatory_authority_recall_at_20") or 0
        b = variants["B"]["metrics"].get("mandatory_authority_recall_at_20") or 0
        c = variants["C"]["metrics"].get("mandatory_authority_recall_at_20") or 0
        if b > a:
            improved.append(case["id"])
        elif b < a:
            regressed.append(case["id"])
        if c > b:
            clarifier_helped.append(case["id"])
        else:
            clarifier_redundant.append(case["id"])
    return [
        "## Diagnostic Deltas",
        "",
        f"- B improved A: {', '.join(improved) or 'none'}",
        f"- B regressed A: {', '.join(regressed) or 'none'}",
        f"- Clarifier improved B: {', '.join(clarifier_helped) or 'none'}",
        f"- Clarifier redundant: {', '.join(clarifier_redundant) or 'none'}",
        "",
    ]


def recommendation(summary: dict[str, Any]) -> str:
    a = summary.get("A", {}).get("metrics", {})
    b = summary.get("B", {}).get("metrics", {})
    gain = (b.get("mandatory_authority_recall_at_20") or 0) - (a.get("mandatory_authority_recall_at_20") or 0)
    primary_delta = (b.get("controlling_provision_recall_at_10") or 0) - (a.get("controlling_provision_recall_at_10") or 0)
    if gain >= 0.2 and primary_delta >= 0:
        return "Wariant B jest obiecujący: poprawia authority recall bez obniżenia primary-law coverage w tym przebiegu."
    if gain > 0:
        return "Wariant B warto dalej rozwijać, ale wymaga kontroli primary-law coverage i pełnego judge odpowiedzi."
    return "Wariant B nie wykazał jeszcze przewagi w tym przebiegu; kolejny krok to analiza kategorii fail i dostrojenie query planning/rerankingu."


def assert_not_holdout(path: Path, allow_holdout: bool) -> None:
    if allow_holdout:
        return
    lowered = str(path).lower()
    if "holdout" in lowered:
        raise ValueError(f"Refusing to run experiment on holdout path: {path}")


def main() -> None:
    args = parse_args()
    assert_not_holdout(args.cases, args.allow_holdout)
    cases = load_cases(args.cases)
    if args.exclude_cases:
        excluded = {case["id"] for case in load_cases(args.exclude_cases)}
        cases = [case for case in cases if case["id"] not in excluded]
    if args.offset:
        cases = cases[args.offset:]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    results: list[dict[str, Any]] = []
    for case in cases:
        variants = [
            evaluate_variant(case, variant_id, name, clarifier, args.limit)
            for variant_id, name, clarifier in VARIANTS
        ]
        results.append(
            {
                "id": str(case["id"]),
                "question": str(case["question"]),
                "expected_signatures": normalize_values(case.get("expected_signatures")),
                "expected_document_ids": normalize_values(case.get("expected_document_ids")),
                "expected_legal_provisions": normalize_values(case.get("expected_legal_provisions")),
                "variants": variants,
            }
        )
    payload = {
        "kind": "hybrid_authority_rag_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "case_count": len(results),
        "answer_generation": "not_run",
        "summary": aggregate(results),
        "results": results,
    }
    write_report(args.artifact_root, payload)
    print(
        json.dumps(
            {
                "case_count": len(results),
                "summary": payload["summary"],
                "comparison_results": str(args.artifact_root / "comparison_results.json"),
                "comparison_report": str(args.artifact_root / "comparison_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
