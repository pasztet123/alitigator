from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from app.main import (
    ANTHROPIC_API_URL,
    ANTHROPIC_CHAT_TIMEOUT_SECONDS,
    CHAT_MAX_TOKENS,
    build_chat_system_prompt,
    extract_text_from_anthropic,
)
from app.rag import (
    RagChunk,
    build_answer_context_block,
    fetch_document_contexts,
    get_rag_config,
    search_chat_chunks,
    select_context_document_ids,
)


RESOLUTION_RE = re.compile(
    r"(stanowisk[oa].{0,180}?(?:prawidłow|nieprawidłow)|"
    r"(?:organ|dyrektor|sąd|nsa|wsa).{0,220}?(?:uznaje|stwierdza|wskazuje|oddala|uchyla)|"
    r"(?:oddala|uchyla)\s+(?:skarg|zaskarżon))",
    re.IGNORECASE | re.DOTALL,
)
TAXPAYER_POSITION_RE = re.compile(r"stanowisko\s+(?:wnioskodawcy|podatnika|państwa)", re.IGNORECASE)
AUTHORITY_POSITION_RE = re.compile(
    r"(ocena stanowiska|uzasadnienie interpretacji|stanowisk[oa].{0,120}?(?:prawidłow|nieprawidłow)|organ\s+(?:uznaje|stwierdza))",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run variant D: routing + bundles + full document context.")
    parser.add_argument("--baseline", default="/tmp/alitigator_claude_rag_benchmark_v2.json")
    parser.add_argument("--compare-c", default="/tmp/alitigator_C_routing_with_bundles.json")
    parser.add_argument("--report", default="/tmp/alitigator_D_document_context.json")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [result["case"] for result in payload["results"]]


def load_c_results(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(result["case_id"]): result for result in payload["results"]}


def choose_smoke_cases(cases: list[dict[str, Any]], c_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case in cases:
        result = c_results.get(case["id"]) or {}
        if not result.get("full_resolution_found") and result.get("judge_result") in {"partial", "fail"}:
            selected.append(case)
        if len(selected) >= 5:
            break
    return selected


def call_anthropic(*, model: str, system: str, user: str, max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
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
    text = extract_text_from_anthropic(response.json())
    if not text:
        raise RuntimeError("Anthropic returned an empty response")
    return text


def build_answer(case: dict[str, Any], chunks: list[RagChunk], context: str, *, model: str) -> str:
    system = build_chat_system_prompt(case["question"], context, chunks)
    return call_anthropic(model=model, system=system, user=case["question"], max_tokens=CHAT_MAX_TOKENS)


def build_judge_prompt(case: dict[str, Any], retrieval: dict[str, Any], answer: str) -> str:
    return (
        "Jesteś surowym ewaluatorem jakości odpowiedzi RAG dla polskiego prawa podatkowego. "
        "Oceń wyłącznie na podstawie pytania, listy wybranych źródeł, kontekstu oraz odpowiedzi modelu. "
        "Zwróć wyłącznie JSON bez markdown.\n\n"
        "Zachowaj ten schemat i znaczenie pól:\n"
        "{\n"
        '  "retrieval_quality": "pass|partial|fail",\n'
        '  "tax_domain_routing": "correct|mixed|wrong",\n'
        '  "special_rule_found": true,\n'
        '  "cross_domain_contamination": false,\n'
        '  "full_resolution_found": true,\n'
        '  "resolution_used_by_answer": true,\n'
        '  "must_query_user_for_civil_law": false,\n'
        '  "main_issues": ["..."],\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Kryteria:\n"
        "- pass: odpowiedź ma właściwe źródła, właściwą domenę podatkową i używa rozstrzygnięcia organu/sądu, nie stanowiska wnioskodawcy.\n"
        "- partial: część istotnych źródeł lub rozstrzygnięcia jest użyta, ale są braki, szum albo niepełna synteza.\n"
        "- fail: odpowiedź opiera się na złej domenie, złych źródłach, stanowisku niewłaściwej strony albo pomija kluczowe rozstrzygnięcie.\n"
        "- full_resolution_found oznacza, czy w kontekście jest pełna ocena/rozstrzygnięcie organu albo sądu, nie tylko opis faktów lub stanowisko podatnika.\n"
        "- resolution_used_by_answer oznacza, czy odpowiedź faktycznie wykorzystała to rozstrzygnięcie.\n\n"
        f"CASE:\n{json.dumps(case, ensure_ascii=False, indent=2)}\n\n"
        f"RETRIEVAL_AND_CONTEXT_METADATA:\n{json.dumps(retrieval, ensure_ascii=False, indent=2)}\n\n"
        f"ANSWER:\n{answer}\n"
    )


def judge_case(case: dict[str, Any], retrieval: dict[str, Any], answer: str, *, model: str) -> dict[str, Any]:
    prompt = build_judge_prompt(case, retrieval, answer)
    raw = call_anthropic(
        model=model,
        system="Zwracasz wyłącznie poprawny JSON zgodny ze schematem użytkownika.",
        user=prompt,
        max_tokens=1800,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise RuntimeError(f"Judge returned non-JSON: {raw[:500]}")
    return json.loads(match.group(0))


def chunk_index_report(document_ids: list[str]) -> list[dict[str, Any]]:
    config = get_rag_config()
    if not config.db_path.exists():
        return []
    reports: list[dict[str, Any]] = []
    with sqlite3.connect(config.db_path) as connection:
        for document_id in document_ids:
            rows = connection.execute(
                "SELECT chunk_index FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
            indexes = [int(row[0]) for row in rows]
            if indexes:
                expected = set(range(min(indexes), max(indexes) + 1))
                missing = sorted(expected - set(indexes))
            else:
                missing = []
            duplicates = sorted({index for index in indexes if indexes.count(index) > 1})
            reports.append(
                {
                    "document_id": document_id,
                    "chunk_indexes": indexes,
                    "missing_indexes": missing,
                    "duplicate_indexes": duplicates,
                    "reconstruction_complete": bool(indexes) and not missing and not duplicates,
                }
            )
    return reports


def document_rank_report(chunks: list[RagChunk], context: str) -> list[dict[str, Any]]:
    document_ids = select_context_document_ids(chunks)
    docs = fetch_document_contexts(document_ids, seed_chunks=chunks)
    scores = {chunk.document_id: max([c.score for c in chunks if c.document_id == chunk.document_id]) for chunk in chunks}
    rows: list[dict[str, Any]] = []
    for rank, document in enumerate(docs, start=1):
        present_text = context[context.find(f"document_id: {document.document_id}"):] if f"document_id: {document.document_id}" in context else document.text
        rows.append(
            {
                "document_rank": rank,
                "document_id": document.document_id,
                "document_relevance": round(float(scores.get(document.document_id, 0.0)), 6),
                "contains_resolution": bool(RESOLUTION_RE.search(document.text)),
                "contains_resolution_in_context": bool(RESOLUTION_RE.search(present_text)),
                "source_type": document.source_type,
                "signature": document.signature,
                "subject": document.subject,
            }
        )
    return rows


def context_truncated(chunks: list[RagChunk], context: str) -> bool:
    max_chars = get_rag_config().document_context_max_chars
    if len(context) >= max_chars:
        return True
    document_ids = select_context_document_ids(chunks)
    included = re.findall(r"^document_id:\s*(.+)$", context, flags=re.MULTILINE)
    return bool(document_ids and included and len(set(included)) < len(document_ids))


def evaluate_case(case: dict[str, Any], *, model: str, skip_answer: bool, skip_judge: bool) -> dict[str, Any]:
    try:
        chunks = search_chat_chunks(case["question"])
        context = build_answer_context_block(chunks)
        retrieval_backend = os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite")
    except Exception:
        if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() not in {"mysql", "mariadb"}:
            raise
        original_backend = os.environ.get("ALITIGATOR_RAG_BACKEND")
        os.environ["ALITIGATOR_RAG_BACKEND"] = "sqlite"
        try:
            chunks = search_chat_chunks(case["question"])
            context = build_answer_context_block(chunks)
            retrieval_backend = "sqlite_fallback"
        finally:
            if original_backend is None:
                os.environ.pop("ALITIGATOR_RAG_BACKEND", None)
            else:
                os.environ["ALITIGATOR_RAG_BACKEND"] = original_backend
    document_ids = select_context_document_ids(chunks)
    doc_ranks = document_rank_report(chunks, context)
    retrieval_meta = {
        "selected_document_ids": document_ids,
        "selected_document_count": len(document_ids),
        "context_chars": len(context),
        "context_truncated": context_truncated(chunks, context),
        "resolution_present_in_context": bool(RESOLUTION_RE.search(context)),
        "taxpayer_position_present_in_context": bool(TAXPAYER_POSITION_RE.search(context)),
        "authority_position_present_in_context": bool(AUTHORITY_POSITION_RE.search(context)),
        "document_ranks": doc_ranks,
        "document_reconstruction": chunk_index_report(document_ids),
        "top_hits": [
            {
                "rank": index,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "signature": chunk.signature,
                "source_type": chunk.source_type,
                "subject": chunk.subject,
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
        "retrieval_backend": retrieval_backend,
    }
    answer = "" if skip_answer else build_answer(case, chunks, context, model=model)
    judge_raw = {} if skip_judge else judge_case(case, retrieval_meta, answer, model=model)
    judge_result = judge_raw.get("retrieval_quality") or "fail"
    return {
        "case_id": case["id"],
        "case_number": case["number"],
        "case_title": case["title"],
        "question": case["question"],
        **retrieval_meta,
        "resolution_used_by_answer": bool(judge_raw.get("resolution_used_by_answer")),
        "mandatory_sources_found": bool(judge_raw.get("special_rule_found")),
        "domain_contamination": bool(judge_raw.get("cross_domain_contamination")),
        "full_resolution_found": bool(judge_raw.get("full_resolution_found")),
        "special_rule_found": bool(judge_raw.get("special_rule_found")),
        "domain_routing_correct": judge_raw.get("tax_domain_routing") == "correct",
        "taxpayer_routing_correct": judge_raw.get("tax_domain_routing") == "correct",
        "judge_result": judge_result,
        "judge_score": float(judge_raw.get("confidence") or 0.0),
        "judge_raw": judge_raw,
        "answer": answer,
    }


def summarize(results: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    completed_results = [result for result in results if not result.get("error")]
    return {
        "variant": variant,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "completed_case_count": len(completed_results),
        "error_count": sum(1 for result in results if result.get("error")),
        "pass": sum(1 for result in completed_results if result["judge_result"] == "pass"),
        "partial": sum(1 for result in completed_results if result["judge_result"] == "partial"),
        "fail": sum(1 for result in completed_results if result["judge_result"] == "fail"),
        "domain_routing_correct": sum(1 for result in completed_results if result["domain_routing_correct"]),
        "taxpayer_routing_correct": sum(1 for result in completed_results if result["taxpayer_routing_correct"]),
        "mandatory_sources_found": sum(1 for result in completed_results if result["mandatory_sources_found"]),
        "full_resolution_found": sum(1 for result in completed_results if result["full_resolution_found"]),
        "resolution_present_in_context": sum(1 for result in completed_results if result["resolution_present_in_context"]),
        "resolution_used_by_answer": sum(1 for result in completed_results if result["resolution_used_by_answer"]),
        "domain_contamination": sum(1 for result in completed_results if result["domain_contamination"]),
        "special_rule_found": sum(1 for result in completed_results if result["special_rule_found"]),
        "avg_score": round(sum(result["judge_score"] for result in completed_results) / max(len(completed_results), 1), 3),
        "avg_context_chars": round(sum(result["context_chars"] for result in completed_results) / max(len(completed_results), 1), 1),
        "results": results,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    load_dotenv(Path("apps/api/.env"))
    args = parse_args()
    cases = load_cases(Path(args.baseline))
    c_results = load_c_results(Path(args.compare_c))
    if args.smoke:
        cases = choose_smoke_cases(cases, c_results)
        report_path = Path(args.report).with_name(Path(args.report).stem + "_smoke.json")
        variant = "D_document_context_smoke"
    else:
        report_path = Path(args.report)
        variant = "D_document_context"
    if args.limit:
        cases = cases[: args.limit]
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]

    results: list[dict[str, Any]] = []
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        results = previous.get("results") or []
        done = {result.get("case_id") for result in results if result.get("case_id")}
        cases = [case for case in cases if case["id"] not in done]
        print(f"Resuming {report_path}: done={len(done)} remaining={len(cases)}", flush=True)
    for index, case in enumerate(cases, start=1):
        started = time.time()
        print(f"[{index}/{len(cases)}] {case['id']} {case['title']}", flush=True)
        try:
            results.append(evaluate_case(case, model=args.model, skip_answer=args.skip_answer, skip_judge=args.skip_judge))
        except Exception as exc:
            results.append(
                {
                    "case_id": case["id"],
                    "case_number": case["number"],
                    "case_title": case["title"],
                    "question": case["question"],
                    "error": repr(exc),
                    "judge_result": "fail",
                    "judge_score": 0.0,
                    "context_chars": 0,
                    "resolution_present_in_context": False,
                    "resolution_used_by_answer": False,
                    "domain_routing_correct": False,
                    "taxpayer_routing_correct": False,
                    "mandatory_sources_found": False,
                    "full_resolution_found": False,
                    "domain_contamination": False,
                    "special_rule_found": False,
                }
            )
        payload = summarize(results, variant=variant)
        write_report(report_path, payload)
        print(
            f"  -> {results[-1]['judge_result']} context={results[-1]['context_chars']} "
            f"resolution_in_context={results[-1]['resolution_present_in_context']} "
            f"used={results[-1]['resolution_used_by_answer']} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    print(json.dumps(summarize(results, variant=variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
