"""Run retrieval-only A--F diagnostics through the current V2 profile.

The runner stops before authority extraction, synthesis and answer writing.  It
therefore measures only the planner, query enrichment and authority retrieval
that this regression suite is intended to change.
"""

from __future__ import annotations

import argparse
import asyncio
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

from app.legal_rag_v2.pipeline import _enrich_research_plan, create_default_pipeline


DEFAULT_CASES = API_DIR / "tests" / "fixtures" / "tax_research_regression_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A--F retrieval-only V2 diagnostics.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--force-planner-fallback", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    return parser.parse_args()


def _authority_candidates(retrieval: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lane in retrieval.authorities:
        for candidate in lane.candidates:
            signature = str(candidate.metadata.get("signature") or "")
            identity = signature or candidate.document_id or candidate.candidate_id
            if identity in seen:
                continue
            seen.add(identity)
            rows.append({
                "signature": signature,
                "document_id": candidate.document_id,
                "source_type": candidate.source_type,
                "score": candidate.score,
                "subject": str(candidate.metadata.get("subject") or ""),
                "component_scores": dict(candidate.component_scores),
                "positive_reasons": list(candidate.positive_reasons),
                "negative_reasons": list(candidate.negative_reasons),
            })
    return rows


async def run_case(case: dict[str, Any], *, force_planner_fallback: bool) -> dict[str, Any]:
    question = str(case["question"])
    pipeline = create_default_pipeline()
    started = time.perf_counter()
    planner_outcome = await pipeline.planner.plan(
        question,
        force_fallback=force_planner_fallback,
    )
    plan = _enrich_research_plan(planner_outcome.plan, question)
    retrieval = await pipeline.retriever.retrieve(plan)
    authorities = _authority_candidates(retrieval)
    targets = {str(value) for value in case.get("targets", [])}
    target_rank = next(
        (rank for rank, row in enumerate(authorities, 1) if row["signature"] in targets),
        None,
    )
    return {
        "id": case["id"],
        "question": question,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "metrics": {
            "target_rank": target_rank,
            "target_present_before_rerank": any(
                row.get("signature") in targets
                for lane in retrieval.authorities
                for row in lane.trace
                if row.get("event") == "candidate_source"
            ),
            "candidate_pool_size": sum(
                lane.candidate_count_before_rerank for lane in retrieval.authorities
            ),
        },
        "planner_output": plan.model_dump(mode="json"),
        "retrieval_trace": list(retrieval.trace),
        "final_authorities": authorities,
    }


async def async_main(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case["id"] in wanted]
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(await asyncio.wait_for(
                run_case(case, force_planner_fallback=args.force_planner_fallback),
                timeout=args.timeout_seconds,
            ))
        except Exception as exc:
            results.append({
                "id": case["id"],
                "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            })
    report = {
        "profile": "current_legal_rag",
        "force_planner_fallback": args.force_planner_fallback,
        "timeout_seconds": args.timeout_seconds,
        "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
