"""Evaluate whether statute retrieval finds the expected provision.

This is deliberately separate from interpretation evaluation: a correct legal
basis does not prove that the closest factual interpretation was retrieved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag import inspect_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument(
        "--exclude-cases",
        type=Path,
        help="Optional JSON case file whose ids are excluded from this evaluation.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Zero-based case offset; useful for resumable long-running evaluations.",
    )
    parser.add_argument(
        "--max-cases", type=int,
        help="Evaluate at most this many cases after --offset.",
    )
    parser.add_argument(
        "--tax-domain",
        help="Optional explicit legal domain (for example VAT) used to scope statute retrieval.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Law evaluation cases must be a flat list of objects")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for position, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {position} must be an object")
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        expected = [str(value).strip().lower() for value in case.get("expected_legal_provisions") or [] if str(value).strip()]
        if not case_id or not question or not expected:
            raise ValueError(f"Case {position} needs id, question and expected_legal_provisions")
        if case_id in ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        ids.add(case_id)
        cases.append({"id": case_id, "question": question, "expected_legal_provisions": expected, "notes": case.get("notes")})
    return cases


def evaluate_case(case: dict[str, Any], limit: int, tax_domain: str | None = None) -> dict[str, Any]:
    inspection = inspect_search(
        case["question"],
        limit=limit,
        source_types={"statute"},
        enforce_query_domain=True,
        tax_domains={tax_domain} if tax_domain else None,
    )
    actual = [provision.lower() for chunk in inspection.chunks for provision in chunk.legal_provisions]
    expected = case["expected_legal_provisions"]
    first_rank = next((rank for rank, value in enumerate(actual, start=1) if value in expected), None)
    raw_candidate_pool = inspection.raw_candidate_pool
    raw_provisions = [
        provision.lower()
        for hit in raw_candidate_pool
        for provision in hit.get("legal_provisions", [])
    ]
    raw_first_rank = next((rank for rank, value in enumerate(raw_provisions, start=1) if value in expected), None)
    return {
        **case,
        "passed": first_rank is not None,
        "first_hit_rank": first_rank,
        "hit_top1": first_rank == 1,
        "hit_top3": first_rank is not None and first_rank <= 3,
        "hit_top6": first_rank is not None and first_rank <= 6,
        "raw_first_hit_rank": raw_first_rank,
        "expected_in_raw_candidate_pool": raw_first_rank is not None,
        "lost_in_rerank": raw_first_rank is not None and first_rank is None,
        "actual_legal_provisions": actual,
        "raw_candidate_pool": raw_candidate_pool,
        "top_hits": inspection.hits,
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.exclude_cases:
        excluded_ids = {case["id"] for case in load_cases(args.exclude_cases)}
        cases = [case for case in cases if case["id"] not in excluded_ids]
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    cases = cases[args.offset:]
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        cases = cases[:args.max_cases]
    results = [evaluate_case(case, args.limit, args.tax_domain) for case in cases]
    summary = {
        "kind": "legal_provision_retrieval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "complete": True,
        "case_count": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "top1_hits": sum(result["hit_top1"] for result in results),
        "top3_hits": sum(result["hit_top3"] for result in results),
        "top6_hits": sum(result["hit_top6"] for result in results),
        "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
