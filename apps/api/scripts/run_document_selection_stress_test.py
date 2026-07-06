from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


E0_CASES: list[dict[str, Any]] = [
    {
        "id": "docsel-ksef-b2b-vs-b2c",
        "category": "contrastive_pair",
        "question": (
            "Podatnik VAT, który przekroczył przejściowy limit pozwalający w 2026 r. wystawiać faktury poza KSeF, "
            "wykonuje tego samego dnia dwie identyczne usługi. Pierwszą świadczy polskiej spółce, a drugą konsumentowi. "
            "Obu nabywcom wysyła faktury PDF poza KSeF. Czy obie faktury zostały wystawione z naruszeniem obowiązku "
            "KSeF? Czy polska spółka może odliczyć VAT z otrzymanego PDF?"
        ),
        "expected_domains": ["VAT", "KSeF"],
        "must_find": ["B2C", "odlicz", "KSeF"],
        "must_not_prioritize": ["wizualizacja", "sankcj"],
        "fatal_patterns": ["b2c.*obowiązk", "brak prawa do odliczenia", "nie może odliczyć"],
        "current_law_patterns": ["2026", "obowiązkowy ksef", "faktur"],
        "controlling_patterns": ["b2c", "ksef", "odlicz"],
        "resolution_patterns": ["konsument", "b2c", "odlicz"],
    },
    {
        "id": "docsel-wht-interest-vs-management",
        "category": "mixed_payment_routing",
        "question": (
            "Polska spółka wypłaca w jednym roku powiązanej spółce holenderskiej 1,4 mln zł odsetek od pożyczki oraz "
            "1,2 mln zł wynagrodzenia za usługi zarządzania. Czy łączna kwota 2,6 mln zł powoduje obowiązek "
            "zastosowania mechanizmu pay and refund? Jak należy osobno przeanalizować oba rodzaje płatności?"
        ),
        "expected_domains": ["CIT", "WHT", "DTA"],
        "must_find": ["21 ust. 1 pkt 1", "21 ust. 1 pkt 2a", "26 ust. 2e", "Holandi"],
        "must_not_prioritize": ["dywidend"],
        "fatal_patterns": ["2,6 mln.*pay and refund", "sum.*zarządz.*odset"],
        "current_law_patterns": ["26 ust. 2e", "pay and refund"],
        "controlling_patterns": ["21 ust. 1 pkt 1", "21 ust. 1 pkt 2a", "26 ust. 2e"],
        "resolution_patterns": ["odset", "zarządz", "holandi", "umow"],
    },
    {
        "id": "docsel-estonian-cit-principal-vs-interest",
        "category": "contrastive_components",
        "question": (
            "Spółka opodatkowana estońskim CIT zwraca wspólnikowi 1 mln zł kapitału pożyczki i płaci mu 90 tys. zł "
            "rynkowych odsetek. Czy obie kwoty stanowią ukryty zysk?"
        ),
        "expected_domains": ["CIT", "Estonian CIT"],
        "must_find": ["28m ust. 3", "28m ust. 4 pkt 3"],
        "must_not_prioritize": ["pożyczki udzielonej przez spółkę wspólnikowi"],
        "fatal_patterns": ["kapitał.*ukryty zysk", "rynkow.*wyłącz.*odset"],
        "current_law_patterns": ["28m"],
        "controlling_patterns": ["28m ust. 3", "28m ust. 4 pkt 3"],
        "resolution_patterns": ["odset", "kapitał", "ukryt"],
    },
    {
        "id": "docsel-family-foundation-loan-scope",
        "category": "contrastive_pair",
        "question": (
            "Fundacja rodzinna udziela dwóch oprocentowanych pożyczek. Pierwszą spółce z o.o., w której posiada 30% "
            "udziałów. Drugą spółce z o.o., z którą nie jest powiązana i w której nie posiada udziałów. Czy obie "
            "pożyczki mieszczą się w dozwolonym zakresie działalności fundacji rodzinnej?"
        ),
        "expected_domains": ["Family foundation", "CIT"],
        "must_find": ["art. 5 ust. 1 pkt 5", "6 ust. 1 pkt 25", "24r"],
        "must_not_prioritize": ["beneficjent"],
        "fatal_patterns": ["obie.*dozwolon", "każd.*oprocentowan.*dozwolon", "obie.*niedozwolon"],
        "current_law_patterns": ["fundacji rodzinnej", "24r"],
        "controlling_patterns": ["art. 5 ust. 1 pkt 5", "udział", "24r"],
        "resolution_patterns": ["udział", "pożycz", "dozwolon"],
    },
    {
        "id": "docsel-limited-partnership-current-law",
        "category": "temporal_versioning",
        "question": (
            "Polska spółka komandytowa osiąga dochód w 2026 r. Czy podatnikiem podatku dochodowego jest sama spółka, "
            "czy wyłącznie jej wspólnicy?"
        ),
        "expected_domains": ["CIT"],
        "must_find": ["spółka komandytowa", "podatnik", "CIT"],
        "must_not_prioritize": ["transparentna podatkowo", "spółka jawna"],
        "fatal_patterns": ["transparentn", "wyłącznie.*wspóln"],
        "current_law_patterns": ["2026", "podatnik", "spółka komandytowa"],
        "controlling_patterns": ["spółka komandytowa", "podatnik", "cit"],
        "resolution_patterns": ["spółka komandytowa", "podatnik"],
    },
    {
        "id": "docsel-private-vs-business-car",
        "category": "contrastive_pair",
        "question": (
            "Przedsiębiorca ponosi wydatki na dwa samochody. Pierwszy jest jego prywatną własnością i nie został "
            "wprowadzony do działalności. Drugi stanowi środek trwały firmy, ale jest używany również prywatnie. "
            "Jaką część wydatków eksploatacyjnych może zaliczyć do kosztów PIT w odniesieniu do każdego samochodu?"
        ),
        "expected_domains": ["PIT"],
        "must_find": ["23 ust. 1 pkt 46", "23 ust. 1 pkt 46a"],
        "must_not_prioritize": ["leasing", "150 000", "50% VAT"],
        "fatal_patterns": ["75%.*prywat", "20%.*środek trwały", "50%.*vat"],
        "current_law_patterns": ["23 ust. 1 pkt 46"],
        "controlling_patterns": ["23 ust. 1 pkt 46", "23 ust. 1 pkt 46a", "20%", "75%"],
        "resolution_patterns": ["20%", "75%", "samoch"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E0 document selection stress test.")
    parser.add_argument("--report", default="/tmp/alitigator_E0_document_selection.json")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def contains_any(text: str, patterns: list[str]) -> bool:
    haystack = norm(text)
    return any(re.search(pattern.lower(), haystack) for pattern in patterns)


def contains_all(text: str, patterns: list[str]) -> bool:
    haystack = norm(text)
    return all(re.search(pattern.lower(), haystack) for pattern in patterns)


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
        "Oceń diagnostyczny test E0 wyboru dokumentów w RAG dla prawa podatkowego. "
        "Zwróć wyłącznie JSON bez markdown.\n\n"
        "Punktacja maks. 10:\n"
        "- domain_routing: 0 albo 1\n"
        "- controlling_rule_present: 0-2\n"
        "- relevant_resolution_present: 0-2\n"
        "- misleading_neighbor_rejected: 0-2\n"
        "- legal_conclusion_correct: 0-2\n"
        "- no_fatal_error: 0 albo 1\n"
        "Fatal error ogranicza answer_result do maksymalnie partial.\n\n"
        "Schemat:\n"
        "{\n"
        '  "domain_routing": 0,\n'
        '  "controlling_rule_present_score": 0,\n'
        '  "relevant_resolution_present_score": 0,\n'
        '  "misleading_neighbor_rejected_score": 0,\n'
        '  "legal_conclusion_correct_score": 0,\n'
        '  "no_fatal_error": 0,\n'
        '  "total_score": 0,\n'
        '  "answer_result": "pass|partial|fail",\n'
        '  "controlling_rule_present": false,\n'
        '  "relevant_resolution_present": false,\n'
        '  "misleading_neighbor_present": false,\n'
        '  "misleading_neighbor_used": false,\n'
        '  "current_law_source_present": false,\n'
        '  "main_issues": ["..."]\n'
        "}\n\n"
        f"CASE:\n{json.dumps(case, ensure_ascii=False, indent=2)}\n\n"
        f"RETRIEVAL:\n{json.dumps(retrieval, ensure_ascii=False, indent=2)}\n\n"
        f"ANSWER:\n{answer}\n"
    )


def judge_case(case: dict[str, Any], retrieval: dict[str, Any], answer: str, *, model: str) -> dict[str, Any]:
    raw = call_anthropic(
        model=model,
        system="Zwracasz wyłącznie poprawny JSON zgodny ze schematem użytkownika.",
        user=build_judge_prompt(case, retrieval, answer),
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
            expected = set(range(min(indexes), max(indexes) + 1)) if indexes else set()
            reports.append(
                {
                    "document_id": document_id,
                    "chunk_indexes": indexes,
                    "missing_indexes": sorted(expected - set(indexes)),
                    "duplicate_indexes": sorted({index for index in indexes if indexes.count(index) > 1}),
                    "reconstruction_complete": bool(indexes) and len(indexes) == len(set(indexes)) and set(indexes) == expected,
                }
            )
    return reports


def document_rank_report(chunks: list[RagChunk], context: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    document_ids = select_context_document_ids(chunks)
    documents = fetch_document_contexts(document_ids, seed_chunks=chunks)
    scores = {chunk.document_id: max(c.score for c in chunks if c.document_id == chunk.document_id) for chunk in chunks}
    rows: list[dict[str, Any]] = []
    for rank, document in enumerate(documents, start=1):
        source = "\n".join([document.source_type or "", document.signature or "", document.subject or "", document.text])
        rows.append(
            {
                "document_rank": rank,
                "document_id": document.document_id,
                "document_relevance": round(float(scores.get(document.document_id, 0.0)), 6),
                "source_type": document.source_type,
                "signature": document.signature,
                "subject": document.subject,
                "controlling_rule_present": contains_any(source, case["controlling_patterns"]),
                "relevant_resolution_present": contains_any(source, case["resolution_patterns"]),
                "misleading_neighbor_present": contains_any(source, case["must_not_prioritize"]),
                "current_law_source_present": contains_any(source, case["current_law_patterns"]),
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


def heuristic_judge(case: dict[str, Any], retrieval: dict[str, Any], answer: str) -> dict[str, Any]:
    context = retrieval["context_excerpt_for_heuristics"]
    top_doc_text = "\n".join(
        f"{rank['source_type']} {rank['signature']} {rank['subject']}" for rank in retrieval["selected_document_ranks"][:3]
    )
    judged_text = f"{context}\n{answer}"
    domain_routing = int(contains_all(top_doc_text + "\n" + context, case["expected_domains"]))
    controlling_rule_present = contains_any(judged_text, case["controlling_patterns"])
    relevant_resolution_present = contains_any(judged_text, case["resolution_patterns"])
    misleading_neighbor_present = contains_any(context, case["must_not_prioritize"])
    misleading_neighbor_used = bool(answer and contains_any(answer, case["must_not_prioritize"]))
    current_law_source_present = contains_any(judged_text, case["current_law_patterns"])
    fatal_error = bool(answer and contains_any(answer, case["fatal_patterns"]))

    score = 0
    score += domain_routing
    score += 2 if controlling_rule_present else 0
    score += 2 if relevant_resolution_present else 0
    score += 0 if misleading_neighbor_used else (1 if misleading_neighbor_present else 2)
    score += 2 if answer and not fatal_error and relevant_resolution_present and controlling_rule_present else 0
    score += 0 if fatal_error else 1
    if score >= 9:
        result = "pass"
    elif score >= 6:
        result = "partial"
    else:
        result = "fail"
    if fatal_error and result == "pass":
        result = "partial"
    return {
        "domain_routing": domain_routing,
        "controlling_rule_present_score": 2 if controlling_rule_present else 0,
        "relevant_resolution_present_score": 2 if relevant_resolution_present else 0,
        "misleading_neighbor_rejected_score": 0 if misleading_neighbor_used else (1 if misleading_neighbor_present else 2),
        "legal_conclusion_correct_score": 2 if answer and not fatal_error and relevant_resolution_present and controlling_rule_present else 0,
        "no_fatal_error": 0 if fatal_error else 1,
        "total_score": score,
        "answer_result": result,
        "controlling_rule_present": controlling_rule_present,
        "relevant_resolution_present": relevant_resolution_present,
        "misleading_neighbor_present": misleading_neighbor_present,
        "misleading_neighbor_used": misleading_neighbor_used,
        "current_law_source_present": current_law_source_present,
        "main_issues": [],
    }


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
    doc_ranks = document_rank_report(chunks, context, case)
    retrieval = {
        "selected_document_ids": document_ids,
        "selected_document_ranks": doc_ranks,
        "context_chars": len(context),
        "context_truncated": context_truncated(chunks, context),
        "controlling_rule_present": contains_any(context, case["controlling_patterns"]),
        "relevant_resolution_present": contains_any(context, case["resolution_patterns"]),
        "misleading_neighbor_present": contains_any(context, case["must_not_prioritize"]),
        "current_law_source_present": contains_any(context, case["current_law_patterns"]),
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
        "context_excerpt_for_heuristics": context[:30000],
    }
    answer = "" if skip_answer else build_answer(case, chunks, context, model=model)
    judge_raw = heuristic_judge(case, retrieval, answer)
    if not skip_judge and answer:
        judge_raw = judge_case(case, retrieval, answer, model=model)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_domains": case["expected_domains"],
        "selected_document_ids": document_ids,
        "selected_document_ranks": doc_ranks,
        "context_chars": retrieval["context_chars"],
        "context_truncated": retrieval["context_truncated"],
        "controlling_rule_present": bool(judge_raw.get("controlling_rule_present")),
        "relevant_resolution_present": bool(judge_raw.get("relevant_resolution_present")),
        "misleading_neighbor_present": bool(judge_raw.get("misleading_neighbor_present")),
        "misleading_neighbor_used": bool(judge_raw.get("misleading_neighbor_used")),
        "current_law_source_present": bool(judge_raw.get("current_law_source_present")),
        "answer_result": judge_raw.get("answer_result", "fail"),
        "total_score": int(judge_raw.get("total_score") or 0),
        "judge_raw": judge_raw,
        "answer": answer,
        "document_reconstruction": retrieval["document_reconstruction"],
        "top_hits": retrieval["top_hits"],
        "retrieval_backend": retrieval_backend,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if not result.get("error")]
    return {
        "variant": "E0_document_selection_stress_test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "completed_case_count": len(completed),
        "error_count": sum(1 for result in results if result.get("error")),
        "pass": sum(1 for result in completed if result["answer_result"] == "pass"),
        "partial": sum(1 for result in completed if result["answer_result"] == "partial"),
        "fail": sum(1 for result in completed if result["answer_result"] == "fail"),
        "avg_score": round(sum(result["total_score"] for result in completed) / max(len(completed), 1), 2),
        "controlling_rule_present": sum(1 for result in completed if result["controlling_rule_present"]),
        "relevant_resolution_present": sum(1 for result in completed if result["relevant_resolution_present"]),
        "misleading_neighbor_present": sum(1 for result in completed if result["misleading_neighbor_present"]),
        "misleading_neighbor_used": sum(1 for result in completed if result["misleading_neighbor_used"]),
        "current_law_source_present": sum(1 for result in completed if result["current_law_source_present"]),
        "avg_context_chars": round(sum(result["context_chars"] for result in completed) / max(len(completed), 1), 1),
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
    cases = E0_CASES
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["id"] in requested]

    report_path = Path(args.report)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        started = time.time()
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        try:
            result = evaluate_case(case, model=args.model, skip_answer=args.skip_answer, skip_judge=args.skip_judge)
        except Exception as exc:
            result = {
                "case_id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "error": repr(exc),
                "answer_result": "fail",
                "total_score": 0,
                "context_chars": 0,
                "controlling_rule_present": False,
                "relevant_resolution_present": False,
                "misleading_neighbor_present": False,
                "misleading_neighbor_used": False,
                "current_law_source_present": False,
            }
        results.append(result)
        write_report(report_path, summarize(results))
        print(
            f"  -> {result['answer_result']} score={result['total_score']} context={result['context_chars']} "
            f"rule={result['controlling_rule_present']} resolution={result['relevant_resolution_present']} "
            f"neighbor_used={result['misleading_neighbor_used']} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
