from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag import build_context_block, list_citations, search_chat_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KSeF chat retrieval guardrails.")
    parser.add_argument(
        "--cases",
        default="data/processed/rag_eval_cases.ksef.json",
        type=Path,
        help="JSON case file with KSeF retrieval guardrails.",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fail-on-miss", action="store_true")
    return parser.parse_args()


def normalize_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("KSeF evaluation cases must be a list of objects")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {position} must be an object")
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        if not case_id or not question:
            raise ValueError(f"Case {position} needs id and question")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(case_id)
        cases.append(case)
    return cases


def evaluate_case(case: dict[str, Any], *, limit: int) -> dict[str, Any]:
    question = str(case["question"]).strip()
    chunks = search_chat_chunks(question, limit=limit)
    context = build_context_block(chunks).lower()
    document_ids = [chunk.document_id for chunk in chunks]
    signatures = [chunk.signature for chunk in chunks if chunk.signature]
    provisions = [provision.lower() for chunk in chunks for provision in chunk.legal_provisions]

    expected_document_ids = normalize_values(case.get("expected_document_ids"))
    expected_signatures = normalize_values(case.get("expected_signatures"))
    expected_legal_provisions = [value.lower() for value in normalize_values(case.get("expected_legal_provisions"))]
    expected_context_terms = [value.lower() for value in normalize_values(case.get("expected_context_terms"))]
    forbidden_context_terms = [value.lower() for value in normalize_values(case.get("forbidden_context_terms"))]

    missing_documents = [value for value in expected_document_ids if value not in document_ids]
    missing_signatures = [value for value in expected_signatures if value not in signatures]
    missing_provisions = [value for value in expected_legal_provisions if value not in provisions]
    missing_terms = [value for value in expected_context_terms if value not in context]
    forbidden_terms_found = [value for value in forbidden_context_terms if value in context]

    passed = not (
        missing_documents
        or missing_signatures
        or missing_provisions
        or missing_terms
        or forbidden_terms_found
    )

    return {
        "id": str(case["id"]),
        "question": question,
        "notes": str(case.get("notes") or "").strip() or None,
        "passed": passed,
        "missing_documents": missing_documents,
        "missing_signatures": missing_signatures,
        "missing_legal_provisions": missing_provisions,
        "missing_context_terms": missing_terms,
        "forbidden_context_terms_found": forbidden_terms_found,
        "document_ids": document_ids,
        "signatures": signatures,
        "legal_provisions": provisions,
        "citations": list_citations(chunks),
        "top_hits": [
            {
                "rank": position,
                "document_id": chunk.document_id,
                "signature": chunk.signature,
                "source_type": chunk.source_type,
                "subject": chunk.subject,
                "chunk_index": chunk.chunk_index,
                "legal_provisions": chunk.legal_provisions,
            }
            for position, chunk in enumerate(chunks, start=1)
        ],
    }


def write_report(report_path: Path | None, *, cases_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "kind": "ksef_chat_retrieval",
        "cases_path": str(cases_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "case_count": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in summary.items() if key != "results"}


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    results = [evaluate_case(case, limit=args.limit) for case in cases]
    summary = write_report(args.report, cases_path=args.cases, results=results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_miss and any(not result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
