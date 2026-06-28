from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.rag import (
    build_context_block,
    get_rag_config,
    inspect_search,
    list_citations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval against a JSON case file.")
    parser.add_argument(
        "--cases",
        default=None,
        help="Path to a JSON file with retrieval evaluation cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override retrieval limit for all cases.",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Exit with status 1 if any case misses expected documents or signatures.",
    )
    parser.add_argument(
        "--raw-source",
        default=None,
        help="Canonical EUREKA raw JSONL source used to validate expected signatures.",
    )
    parser.add_argument(
        "--skip-canonical-validation",
        action="store_true",
        help="Allow cases whose expected source is absent from the canonical raw JSONL.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write a durable JSON report after every evaluated case.",
    )
    parser.add_argument(
        "--tax-domain",
        default=None,
        help="Restrict retrieval to a single tax domain such as PIT, CIT, or VAT.",
    )
    parser.add_argument(
        "--enforce-query-domain",
        action="store_true",
        help="Require retrieval to stay inside the requested tax domain.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        choices=["interpretation", "statute", "judgment", "commentary"],
        default=None,
        help="Restrict retrieval to one or more source types. May be repeated.",
    )
    return parser.parse_args()


def get_default_cases_path() -> Path:
    config = get_rag_config()
    return config.processed_path.parent / "rag_eval_cases.sample.json"


def get_default_raw_source_path() -> Path:
    return get_rag_config().processed_path.parent.parent / "raw" / "eureka_interpretations.raw.jsonl"


def load_canonical_signatures(path: Path) -> set[str]:
    signatures: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            signature = str((record.get("summary") or {}).get("SYG") or "").strip()
            if signature:
                signatures.add(signature)
    return signatures


def validate_cases_against_canonical_source(cases: list[dict[str, Any]], raw_path: Path) -> None:
    canonical_signatures = load_canonical_signatures(raw_path)
    missing = [
        str(case["id"])
        for case in cases
        if not set(normalize_values(case.get("expected_signatures"))) & canonical_signatures
    ]
    if missing:
        raise ValueError(
            "Expected signatures absent from canonical raw source "
            f"{raw_path}: {', '.join(missing)}"
        )


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Evaluation cases JSON must be a list of objects")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case at position {position} must be an object, not {type(case).__name__}")
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        expected_documents = normalize_values(case.get("expected_document_ids"))
        expected_signatures = normalize_values(case.get("expected_signatures"))
        if not case_id:
            raise ValueError(f"Evaluation case at position {position} is missing id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        if not question:
            raise ValueError(f"Evaluation case {case_id} is missing a non-empty question")
        if not expected_documents and not expected_signatures:
            raise ValueError(f"Evaluation case {case_id} needs expected_document_ids or expected_signatures")
        seen_ids.add(case_id)
        cases.append(case)
    return cases


def normalize_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def first_match_rank(expected_values: list[str], actual_values: list[str]) -> Optional[int]:
    expected_set = {value for value in expected_values if value}
    if not expected_set:
        return None
    for rank, value in enumerate(actual_values, start=1):
        if value in expected_set:
            return rank
    return None


def hit_within(rank: Optional[int], limit: int) -> bool:
    return rank is not None and rank <= limit


def filter_expected_hits(
    hits: list[dict[str, Any]], *, expected_document_ids: list[str], expected_signatures: list[str]
) -> list[dict[str, Any]]:
    expected_documents = {value for value in expected_document_ids if value}
    expected_signature_values = {value for value in expected_signatures if value}
    return [
        hit
        for hit in hits
        if str(hit.get("document_id") or "") in expected_documents
        or str(hit.get("signature") or "") in expected_signature_values
    ]


def expected_source_present(
    *, expected_document_ids: list[str], expected_signatures: list[str]
) -> bool:
    """Whether at least one expected source is available in the active local index."""
    config = get_rag_config()
    if not config.db_path.exists():
        return False
    clauses: list[str] = []
    values: list[str] = []
    if expected_document_ids:
        clauses.append("document_id IN (" + ", ".join("?" for _ in expected_document_ids) + ")")
        values.extend(expected_document_ids)
    if expected_signatures:
        clauses.append("signature IN (" + ", ".join("?" for _ in expected_signatures) + ")")
        values.extend(expected_signatures)
    with sqlite3.connect(config.db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM documents WHERE " + " OR ".join(clauses) + " LIMIT 1", values
        ).fetchone()
    return row is not None


def evaluate_case(
    case: dict[str, Any],
    *,
    override_limit: Optional[int] = None,
    tax_domain: Optional[str] = None,
    enforce_query_domain: bool = False,
    source_types: Optional[set[str]] = None,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip() or "unnamed-case"
    question = str(case.get("question") or "").strip()
    if not question:
        raise ValueError(f"Case {case_id} is missing a non-empty question")

    limit = override_limit or int(case.get("limit") or 0) or None
    tax_domains = {tax_domain.upper()} if tax_domain else None
    inspection = inspect_search(
        question,
        limit=limit,
        source_types=source_types,
        tax_domains=tax_domains,
        enforce_query_domain=enforce_query_domain,
    )
    chunks = inspection.chunks
    raw_candidate_pool = inspection.raw_candidate_pool
    top_document_ids = [chunk.document_id for chunk in chunks]
    top_signatures = [chunk.signature for chunk in chunks if chunk.signature]
    expected_document_ids = normalize_values(case.get("expected_document_ids"))
    expected_signatures = normalize_values(case.get("expected_signatures"))
    source_present = expected_source_present(
        expected_document_ids=expected_document_ids,
        expected_signatures=expected_signatures,
    )
    raw_document_ids = [hit["document_id"] for hit in raw_candidate_pool]
    raw_signatures = [str(hit["signature"]) for hit in raw_candidate_pool if hit["signature"]]

    matched_documents = [document_id for document_id in expected_document_ids if document_id in top_document_ids]
    matched_signatures = [signature for signature in expected_signatures if signature in top_signatures]
    first_document_rank = first_match_rank(expected_document_ids, top_document_ids)
    first_signature_rank = first_match_rank(expected_signatures, top_signatures)
    raw_first_document_rank = first_match_rank(expected_document_ids, raw_document_ids)
    raw_first_signature_rank = first_match_rank(expected_signatures, raw_signatures)
    raw_first_hit_rank = min(
        [rank for rank in [raw_first_document_rank, raw_first_signature_rank] if rank is not None],
        default=None,
    )
    first_hit_rank = min(
        [rank for rank in [first_document_rank, first_signature_rank] if rank is not None],
        default=None,
    )
    passed = True
    if expected_document_ids and not matched_documents:
        passed = False
    if expected_signatures and not matched_signatures:
        passed = False

    return {
        "id": case_id,
        "question": question,
        "notes": str(case.get("notes") or "").strip() or None,
        "expected_answer": str(case.get("expected_answer") or "").strip() or None,
        "expected_document_ids": expected_document_ids,
        "expected_signatures": expected_signatures,
        "expected_source_present": source_present,
        "passed": passed,
        "matched_documents": matched_documents,
        "matched_signatures": matched_signatures,
        "first_document_rank": first_document_rank,
        "first_signature_rank": first_signature_rank,
        "first_hit_rank": first_hit_rank,
        "raw_candidate_count": len(raw_candidate_pool),
        "raw_first_document_rank": raw_first_document_rank,
        "raw_first_signature_rank": raw_first_signature_rank,
        "raw_first_hit_rank": raw_first_hit_rank,
        "expected_in_raw_candidate_pool": raw_first_hit_rank is not None,
        "lost_in_rerank": raw_first_hit_rank is not None and first_hit_rank is None,
        "hit_top1": hit_within(first_hit_rank, 1),
        "hit_top3": hit_within(first_hit_rank, 3),
        "hit_top6": hit_within(first_hit_rank, 6),
        "retrieved_count": inspection.retrieved_count,
        "selected_count": inspection.selected_count,
        "selected_context_chars": inspection.selected_context_chars,
        "match_query": inspection.match_query,
        "citations": list_citations(chunks),
        "context_block": build_context_block(chunks),
        "top_hits": inspection.hits,
        "expected_final_hits": filter_expected_hits(
            inspection.hits,
            expected_document_ids=expected_document_ids,
            expected_signatures=expected_signatures,
        ),
        "expected_raw_candidates": filter_expected_hits(
            raw_candidate_pool,
            expected_document_ids=expected_document_ids,
            expected_signatures=expected_signatures,
        ),
        "raw_candidate_pool": raw_candidate_pool,
    }


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases) if args.cases else get_default_cases_path()
    cases = load_cases(cases_path)
    if not args.skip_canonical_validation:
        raw_path = Path(args.raw_source) if args.raw_source else get_default_raw_source_path()
        validate_cases_against_canonical_source(cases, raw_path)
    report_path = Path(args.report) if args.report else None
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(
            evaluate_case(
                case,
                override_limit=args.limit,
                tax_domain=args.tax_domain,
                enforce_query_domain=args.enforce_query_domain,
                source_types=set(args.source_type) if args.source_type else None,
            )
        )
        if report_path:
            write_report(report_path, cases_path=cases_path, results=results, complete=False)
    summary = write_report(report_path, cases_path=cases_path, results=results, complete=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_miss and any(not result["passed"] for result in results):
        raise SystemExit(1)


def write_report(
    report_path: Optional[Path], *, cases_path: Path, results: list[dict[str, Any]], complete: bool
) -> dict[str, Any]:
    summary = {
        "cases_path": str(cases_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "case_count": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "top1_hits": sum(1 for result in results if result["hit_top1"]),
        "top3_hits": sum(1 for result in results if result["hit_top3"]),
        "top6_hits": sum(1 for result in results if result["hit_top6"]),
        "source_covered_case_count": sum(1 for result in results if result["expected_source_present"]),
        "source_covered_passed": sum(
            1 for result in results if result["expected_source_present"] and result["passed"]
        ),
        "source_covered_top1_hits": sum(
            1 for result in results if result["expected_source_present"] and result["hit_top1"]
        ),
        "source_covered_top3_hits": sum(
            1 for result in results if result["expected_source_present"] and result["hit_top3"]
        ),
        "source_covered_top6_hits": sum(
            1 for result in results if result["expected_source_present"] and result["hit_top6"]
        ),
        "results": results,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(report_path)
    return summary


if __name__ == "__main__":
    main()
