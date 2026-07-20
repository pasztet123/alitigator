"""Evaluate the isolated July 7 interpretation retrieval against labeled questions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.legacy_interpretations import search_tax_interpretations


DEFAULT_CASES_PATH = Path("data/processed/interpretation_retrieval_diagnostics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six-result tax-interpretation retrieval diagnostic suite."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="Exit unsuccessfully if a labeled target interpretation is absent from the six results.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    cases: list[dict[str, Any]] = []
    for position, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case {position} must be an object")
        case_id = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        if not case_id or not question:
            raise ValueError(f"Case {position} needs id and question")
        for field in (
            "expected_mechanism",
            "expected_provisions",
            "target_interpretations",
            "forbidden_wrong_neighbors",
        ):
            if field not in item:
                raise ValueError(f"Case {case_id} needs {field}")
        cases.append(item)
    return cases


def first_rank(signatures: list[str], wanted: set[str]) -> int | None:
    for rank, signature in enumerate(signatures, start=1):
        if signature in wanted:
            return rank
    return None


def evaluate_case(case: dict[str, Any], *, limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = search_tax_interpretations(str(case["question"]), limit=limit)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    results = [
        {
            "rank": rank,
            "signature": str(chunk.signature or ""),
            "subject": str(chunk.subject or ""),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
    signatures = [result["signature"] for result in results]
    targets = {str(value).strip() for value in case.get("target_interpretations", []) if str(value).strip()}
    forbidden = {
        str(value).strip()
        for value in case.get("forbidden_wrong_neighbors", [])
        if str(value).strip()
    }
    return {
        "id": str(case["id"]),
        "question": str(case["question"]),
        "expected_mechanism": str(case["expected_mechanism"]),
        "target_rank": first_rank(signatures, targets) if targets else None,
        "forbidden_ranks": [
            rank for rank, signature in enumerate(signatures, start=1) if signature in forbidden
        ],
        "latency_ms": elapsed_ms,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    if args.limit != 6:
        raise ValueError("This diagnostic suite is defined for exactly six returned interpretations")
    cases = load_cases(args.cases)
    if args.case_ids:
        wanted_ids = set(args.case_ids)
        cases = [case for case in cases if str(case["id"]) in wanted_ids]
    if not cases:
        raise ValueError("No diagnostic cases selected")

    results = [evaluate_case(case, limit=args.limit) for case in cases]
    labeled_results = [
        result
        for case, result in zip(cases, results)
        if any(str(value).strip() for value in case.get("target_interpretations", []))
    ]
    target_hits = [result for result in labeled_results if result["target_rank"] is not None]
    forbidden_hits = [result for result in results if result["forbidden_ranks"]]
    report = {
        "limit": args.limit,
        "cases": results,
        "summary": {
            "case_count": len(results),
            "labeled_case_count": len(labeled_results),
            "target_hit_count": len(target_hits),
            "target_hit_rate": len(target_hits) / len(labeled_results) if labeled_results else None,
            "forbidden_neighbor_case_count": len(forbidden_hits),
            "average_latency_ms": round(
                sum(result["latency_ms"] for result in results) / len(results)
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if args.fail_on_target_miss and len(target_hits) != len(labeled_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
