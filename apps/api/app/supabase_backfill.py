from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.rag import get_rag_config
from app.supabase_rag import (
    build_supabase_headers,
    get_supabase_target,
    is_supabase_rag_configured,
    load_sync_state,
    reindex_corpus_to_supabase,
)


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fetch_remote_counts() -> dict[str, Any]:
    if not is_supabase_rag_configured():
        return {"configured": False}

    config = get_rag_config()
    schema, documents_table, chunks_table, _ = get_supabase_target()
    supabase_url = Path("/")
    del supabase_url

    import os

    base = os.environ["SUPABASE_URL"].rstrip("/")
    headers = build_supabase_headers(schema=schema)
    response_headers = {**headers, "Prefer": "count=exact"}

    with httpx.Client(timeout=config.supabase_request_timeout) as client:
        documents_response = client.get(
            f"{base}/rest/v1/{documents_table}",
            headers=response_headers,
            params={"select": "document_id", "limit": 1},
        )
        chunks_response = client.get(
            f"{base}/rest/v1/{chunks_table}",
            headers=response_headers,
            params={"select": "chunk_id", "limit": 1},
        )

    documents_response.raise_for_status()
    chunks_response.raise_for_status()

    return {
        "configured": True,
        "documents_content_range": documents_response.headers.get("content-range"),
        "chunks_content_range": chunks_response.headers.get("content-range"),
        "documents_status": documents_response.status_code,
        "chunks_status": chunks_response.status_code,
    }


def build_status_payload() -> dict[str, Any]:
    config = get_rag_config()
    return {
        "processed_path": str(config.processed_path),
        "processed_rows": count_jsonl_rows(config.processed_path),
        "supabase_state_path": str(config.supabase_state_path),
        "supabase_state": load_sync_state(config.supabase_state_path),
        "remote": fetch_remote_counts(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or inspect resumable Supabase RAG backfill.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--no-remote-status", action="store_true")
    parser.add_argument("--compare-remote-hashes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-batch-size", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = get_rag_config()

    if args.reset_state and config.supabase_state_path.exists():
        config.supabase_state_path.unlink()

    if args.status:
        print(json.dumps(build_status_payload(), ensure_ascii=False, indent=2))
        return

    result = reindex_corpus_to_supabase(
        limit=args.limit,
        force=args.force,
        compare_remote_hashes=args.compare_remote_hashes,
        batch_size=args.batch_size,
        chunk_batch_size=args.chunk_batch_size,
        emit_progress=not args.no_progress,
        reverse_order=args.reverse_order,
    )
    payload = {"result": result}
    if not args.no_remote_status:
        payload["status"] = build_status_payload()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()