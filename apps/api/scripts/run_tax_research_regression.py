"""Record repeatable diagnostics for the tax-authority retrieval profiles.

The runner is intentionally retrieval-only: it never invokes the answer writer
or persists anything to the corpus.  It is usable against a local MariaDB
snapshot and produces an auditable JSON report for A--F regression cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
load_dotenv(API_DIR / ".env")

from app import legacy_interpretations


DEFAULT_CASES = API_DIR / "tests" / "fixtures" / "tax_research_regression_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A-F tax research regression diagnostics.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--case", action="append", dest="case_ids")
    return parser.parse_args()


def _rank_for(signatures: list[str], wanted: set[str]) -> int | None:
    return next((rank for rank, signature in enumerate(signatures, 1) if signature in wanted), None)


def _candidate_trace(question: str, *, limit: int) -> dict[str, Any]:
    return legacy_interpretations.search_tax_interpretations_with_trace(
        question, limit=limit
    ).to_trace()


def _corpus_signatures(signatures: set[str]) -> set[str]:
    """Test-only inventory check; production retrieval never receives signatures."""

    if not signatures:
        return set()
    if legacy_interpretations.july7_mysql_rag.is_mysql_rag_configured():
        documents_table, _ = legacy_interpretations.july7_mysql_rag.get_mysql_target()
        with legacy_interpretations.july7_mysql_rag.mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT signature FROM `{documents_table}` WHERE signature IN ({', '.join(['%s'] * len(signatures))})",
                    tuple(sorted(signatures)),
                )
                return {str(row["signature"]) for row in cursor.fetchall() if row.get("signature")}
    config = legacy_interpretations.july7_rag.get_rag_config()
    connection = legacy_interpretations.july7_rag.get_connection(config.db_path)
    try:
        rows = connection.execute(
            f"SELECT signature FROM documents WHERE signature IN ({', '.join(['?'] * len(signatures))})",
            tuple(sorted(signatures)),
        ).fetchall()
        return {str(row[0]) for row in rows if row[0]}
    finally:
        connection.close()


def run_case(case: dict[str, Any], *, limit: int) -> dict[str, Any]:
    question = str(case["question"])
    started = time.perf_counter()
    trace = _candidate_trace(question, limit=limit)
    final = trace["final_results"]
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    results = [
        {"rank": rank, **dict(chunk)} for rank, chunk in enumerate(final, 1)
    ]
    signatures = [item["signature"] for item in results]
    targets = {str(item) for item in case.get("targets", [])}
    forbidden = {str(item) for item in case.get("forbidden", [])}
    target_rank = _rank_for(signatures, targets)
    def relevant(item: dict[str, Any]) -> bool:
        return item.get("relation") in {"direct", "strong_analogy"} and not item.get("reject")

    # The renderer presents different mechanisms in a separately labelled
    # section.  Precision metrics intentionally measure the principal lane
    # (direct/strong analogy), rather than treating a disclosed warning as a
    # top-result failure.
    principal_results = [item for item in results if relevant(item)]

    def wrong_neighbour(item: dict[str, Any]) -> bool:
        return item.get("signature") in forbidden

    candidate_ids = set(str(item) for item in trace.get("candidate_document_ids", []))
    corpus_targets = _corpus_signatures(targets)
    missing_from_corpus = sorted(targets - corpus_targets)
    missing_from_candidates = sorted(corpus_targets - candidate_ids)
    return {
        "id": case["id"],
        "question": question,
        "metrics": {
            "target_rank": target_rank,
            "relevant_in_top_5": len(principal_results[:5]),
            "relevant_in_top_10": len(principal_results[:10]),
            "wrong_neighbors_in_top_5": sum(wrong_neighbour(item) for item in principal_results[:5]),
            "wrong_neighbors_in_top_10": sum(wrong_neighbour(item) for item in principal_results[:10]),
            "candidate_pool_size": int((trace.get("candidate_counts") or {}).get("deduplicated", 0)),
            "target_present_before_rerank": bool(targets and candidate_ids.intersection(targets)),
            "target_present_after_rerank": any(item.get("signature") in targets for item in trace.get("reranker_scores", [])),
        },
        "latency_ms": elapsed_ms,
        "failure_kind": (
            "target_not_in_corpus" if missing_from_corpus
            else "candidate_generation_failure" if missing_from_candidates
            else None
        ),
        "missing_target_signatures": missing_from_corpus or missing_from_candidates,
        "trace": trace,
        "final_results": results,
    }


def main() -> int:
    args = parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case["id"] in wanted]
    report = {
        "profile": "interpretations_july7",
        "limit": args.limit,
        "cases": [run_case(case, limit=args.limit) for case in cases],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
