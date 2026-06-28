from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge single-case law eval reports.")
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--exclude-cases", type=Path)
    return parser.parse_args()


def load_results(reports_dir: Path, expected_count: int) -> list[dict[str, Any]]:
    results_by_index: dict[int, dict[str, Any]] = {}
    for index in range(expected_count):
        report_path = reports_dir / f"{index}.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing report: {report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_results = payload.get("results") or []
        if len(report_results) != 1:
            raise ValueError(f"Expected exactly one result in {report_path}")
        results_by_index[index] = report_results[0]
    return [results_by_index[index] for index in range(expected_count)]


def main() -> None:
    args = parse_args()
    results = load_results(args.reports_dir, args.expected_count)
    summary = {
        "kind": "legal_provision_retrieval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "exclude_cases_path": str(args.exclude_cases) if args.exclude_cases else None,
        "complete": True,
        "case_count": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "top1_hits": sum(result["hit_top1"] for result in results),
        "top3_hits": sum(result["hit_top3"] for result in results),
        "top6_hits": sum(result["hit_top6"] for result in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
