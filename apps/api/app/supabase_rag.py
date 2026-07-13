from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from app.rag import (
    RagChunk,
    build_document_payload,
    compute_embedding,
    expand_search_query,
    get_rag_config,
    iter_processed_records,
)
from app.rag_runtime import corpus_manifest_hash, iter_configured_corpus_sources


def is_supabase_rag_enabled() -> bool:
    return os.getenv("ALITIGATOR_RAG_USE_SUPABASE", "true").lower() in {"1", "true", "yes"}


def is_supabase_rag_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))


def is_supabase_sync_enabled() -> bool:
    return os.getenv("ALITIGATOR_RAG_SUPABASE_SYNC", "false").lower() in {"1", "true", "yes"}


def get_supabase_target() -> tuple[str, str, str, str]:
    schema = os.getenv("ALITIGATOR_RAG_SUPABASE_SCHEMA", "public")
    documents_table = os.getenv("ALITIGATOR_RAG_SUPABASE_DOCUMENTS_TABLE", "eureka_interpretations")
    chunks_table = os.getenv("ALITIGATOR_RAG_SUPABASE_CHUNKS_TABLE", "eureka_chunks")
    search_function = os.getenv("ALITIGATOR_RAG_SUPABASE_SEARCH_FUNCTION", "search_eureka_chunks")
    return schema, documents_table, chunks_table, search_function


def build_supabase_headers(*, schema: str) -> dict[str, str]:
    secret = os.getenv("SUPABASE_SECRET_KEY", "")
    return {
        "apikey": secret,
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
        "Accept-Profile": schema,
        "Content-Profile": schema,
    }


def quote_in_filter(values: list[str]) -> str:
    safe_values = [value.replace('"', '""') for value in values if value]
    return "in.(" + ",".join(f'"{value}"' for value in safe_values) + ")"


def chunked(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.ReadError,
)


def load_sync_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_sync_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
            return response
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                raise

        if attempt >= max_retries:
            if last_error is not None:
                raise last_error
            break
        time.sleep(min(2**attempt, 8))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Supabase request failed without explicit error: {method} {url}")


def delete_chunks_for_documents(
    client: httpx.Client,
    *,
    supabase_url: str,
    chunks_table: str,
    document_ids: list[str],
    max_retries: int,
    batch_size: int,
) -> None:
    for batch in chunked(document_ids, max(1, batch_size)):
        try:
            response = request_with_retry(
                client,
                "DELETE",
                f"{supabase_url}/rest/v1/{chunks_table}",
                max_retries=max_retries,
                params={"document_id": quote_in_filter(batch)},
            )
            response.raise_for_status()
        except Exception:
            if len(batch) <= 1:
                raise
            midpoint = len(batch) // 2
            delete_chunks_for_documents(
                client,
                supabase_url=supabase_url,
                chunks_table=chunks_table,
                document_ids=batch[:midpoint],
                max_retries=max_retries,
                batch_size=max(1, midpoint),
            )
            delete_chunks_for_documents(
                client,
                supabase_url=supabase_url,
                chunks_table=chunks_table,
                document_ids=batch[midpoint:],
                max_retries=max_retries,
                batch_size=max(1, len(batch) - midpoint),
            )


def post_rows_resilient(
    client: httpx.Client,
    *,
    supabase_url: str,
    table: str,
    on_conflict: str,
    rows: list[dict[str, Any]],
    max_retries: int,
    row_label: str,
) -> None:
    if not rows:
        return

    try:
        response = request_with_retry(
            client,
            "POST",
            f"{supabase_url}/rest/v1/{table}",
            max_retries=max_retries,
            params={"on_conflict": on_conflict},
            json=rows,
        )
        response.raise_for_status()
    except Exception:
        if len(rows) <= 1:
            if rows:
                failed_row = rows[0]
                raise RuntimeError(
                    f"Supabase upsert failed for {row_label}={failed_row.get(row_label, failed_row.get('document_id') or failed_row.get('chunk_id') or 'unknown')}"
                )
            raise
        midpoint = len(rows) // 2
        post_rows_resilient(
            client,
            supabase_url=supabase_url,
            table=table,
            on_conflict=on_conflict,
            rows=rows[:midpoint],
            max_retries=max_retries,
            row_label=row_label,
        )
        post_rows_resilient(
            client,
            supabase_url=supabase_url,
            table=table,
            on_conflict=on_conflict,
            rows=rows[midpoint:],
            max_retries=max_retries,
            row_label=row_label,
        )


def upload_rows_in_batches(
    client: httpx.Client,
    *,
    supabase_url: str,
    table: str,
    on_conflict: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_retries: int,
    row_label: str,
) -> None:
    for batch in chunked(rows, max(1, batch_size)):
        post_rows_resilient(
            client,
            supabase_url=supabase_url,
            table=table,
            on_conflict=on_conflict,
            rows=batch,
            max_retries=max_retries,
            row_label=row_label,
        )


def fetch_remote_hashes(document_ids: list[str]) -> dict[str, str]:
    if not document_ids or not is_supabase_rag_configured():
        return {}

    schema, documents_table, _, _ = get_supabase_target()
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    config = get_rag_config()
    headers = build_supabase_headers(schema=schema)

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=config.supabase_request_timeout, headers=headers) as client:
        for batch in chunked(document_ids, max(1, config.supabase_batch_size)):
            response = request_with_retry(
                client,
                "GET",
                f"{supabase_url}/rest/v1/{documents_table}",
                max_retries=config.supabase_max_retries,
                params={
                    "select": "document_id,content_sha256",
                    "document_id": quote_in_filter(batch),
                },
            )
            response.raise_for_status()
            rows.extend(response.json())
    return {
        str(row.get("document_id") or ""): str(row.get("content_sha256") or "")
        for row in rows
        if row.get("document_id")
    }


def upsert_records_to_supabase(
    records: list[dict[str, Any]],
    *,
    force: bool = False,
    chunk_batch_size: Optional[int] = None,
) -> dict[str, Any]:
    if not is_supabase_rag_configured():
        raise RuntimeError("Supabase RAG wymaga SUPABASE_URL i SUPABASE_SECRET_KEY")

    config = get_rag_config()
    effective_chunk_batch_size = max(1, chunk_batch_size or config.supabase_chunk_batch_size)
    schema, documents_table, chunks_table, _ = get_supabase_target()
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    headers = build_supabase_headers(schema=schema)

    processed = 0
    indexed = 0
    skipped = 0
    documents_payload: list[dict[str, Any]] = []
    chunks_payload: list[dict[str, Any]] = []

    document_ids = [str(record.get("document_id") or "").strip() for record in records if record.get("document_id")]
    remote_hashes = {} if force else fetch_remote_hashes(document_ids)

    for record in records:
        processed += 1
        document_id = str(record.get("document_id") or "").strip()
        if not document_id:
            skipped += 1
            continue

        current_sha = str(record.get("content_sha256") or "")
        if not force and current_sha and remote_hashes.get(document_id) == current_sha:
            skipped += 1
            continue

        document_payload, chunk_payloads = build_document_payload(record, config)
        if not document_payload or not chunk_payloads:
            skipped += 1
            continue

        documents_payload.append(document_payload)
        chunks_payload.extend(chunk_payloads)
        indexed += 1

    if not documents_payload:
        return {
            "processed": processed,
            "indexed": indexed,
            "skipped": skipped,
            "chunk_count": 0,
            "document_count": 0,
            "indexed_document_ids": [],
        }

    indexed_document_ids = [str(row["document_id"]) for row in documents_payload]

    with httpx.Client(timeout=config.supabase_request_timeout, headers=headers) as client:
        delete_chunks_for_documents(
            client,
            supabase_url=supabase_url,
            chunks_table=chunks_table,
            document_ids=indexed_document_ids,
            max_retries=config.supabase_max_retries,
            batch_size=min(config.supabase_batch_size, 50),
        )

        upload_rows_in_batches(
            client,
            supabase_url=supabase_url,
            table=documents_table,
            on_conflict="document_id",
            rows=documents_payload,
            batch_size=config.supabase_batch_size,
            max_retries=config.supabase_max_retries,
            row_label="document_id",
        )

        upload_rows_in_batches(
            client,
            supabase_url=supabase_url,
            table=chunks_table,
            on_conflict="chunk_id",
            rows=chunks_payload,
            batch_size=effective_chunk_batch_size,
            max_retries=config.supabase_max_retries,
            row_label="chunk_id",
        )

    return {
        "processed": processed,
        "indexed": indexed,
        "skipped": skipped,
        "chunk_count": len(chunks_payload),
        "document_count": len(documents_payload),
        "indexed_document_ids": indexed_document_ids,
    }


def sync_records_to_supabase(
    records: list[dict[str, Any]],
    *,
    force: bool = False,
    chunk_batch_size: Optional[int] = None,
) -> dict[str, Any]:
    return upsert_records_to_supabase(records, force=force, chunk_batch_size=chunk_batch_size)

def reindex_corpus_to_supabase(
    *,
    limit: Optional[int] = None,
    force: bool = False,
    compare_remote_hashes: bool = True,
    batch_size: Optional[int] = None,
    chunk_batch_size: Optional[int] = None,
    emit_progress: bool = False,
    reverse_order: bool = False,
) -> dict[str, Any]:
    config = get_rag_config()
    sources = iter_configured_corpus_sources(config)
    missing = [source.path for source in sources if source.required and not source.path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured corpus not found: {missing[0]}")
    state_path = config.supabase_state_path
    persisted_state = {} if force or limit is not None else load_sync_state(state_path)
    # v1 stored one global cursor.  Never let that cursor skip sources added
    # later: it is only migrated to the primary source, all others start at 0.
    source_state = dict(persisted_state.get("sources") or {})
    if not source_state and persisted_state:
        source_state = {sources[0].source_id: {"processed": int(persisted_state.get("processed") or 0), "status": persisted_state.get("status")}} if sources else {}
    manifest_hash = corpus_manifest_hash(sources)
    if persisted_state.get("schema_version") == 2 and persisted_state.get("manifest_hash") != manifest_hash:
        # Source bytes or configured paths changed. Restart cursors; upserts
        # remain idempotent and no remote rows are deleted as part of backfill.
        source_state = {}
    processed = 0
    indexed = 0
    skipped = 0
    chunk_count = 0
    indexed_document_ids: list[str] = []
    effective_batch_size = max(1, batch_size or config.supabase_batch_size)
    effective_chunk_batch_size = max(1, chunk_batch_size or config.supabase_chunk_batch_size)

    for source in sources:
        prior = source_state.get(source.source_id, {})
        resume_from = 0 if force else int(prior.get("processed") or 0)
        batch: list[dict[str, Any]] = []
        source_processed = 0
        for record_index, record in enumerate(iter_processed_records(source.path, reverse=reverse_order)):
            if record_index < resume_from: continue
            if limit is not None and processed >= limit: break
            batch.append(record); processed += 1; source_processed += 1
            if len(batch) < effective_batch_size: continue
            result = upsert_records_to_supabase(batch, force=force or not compare_remote_hashes, chunk_batch_size=effective_chunk_batch_size)
            indexed += int(result["indexed"]); skipped += int(result["skipped"]); chunk_count += int(result["chunk_count"]); indexed_document_ids.extend(result["indexed_document_ids"])
            source_state[source.source_id] = {"path": str(source.path), "processed": resume_from + source_processed, "status": "running", "last_document_id": str(batch[-1].get("document_id") or "")}
            if limit is None: write_sync_state(state_path, {"schema_version": 2, "manifest_hash": manifest_hash, "status": "running", "sources": source_state})
            batch = []
        if batch:
            result = upsert_records_to_supabase(batch, force=force or not compare_remote_hashes, chunk_batch_size=effective_chunk_batch_size)
            indexed += int(result["indexed"]); skipped += int(result["skipped"]); chunk_count += int(result["chunk_count"]); indexed_document_ids.extend(result["indexed_document_ids"])
        source_state[source.source_id] = {"path": str(source.path), "processed": resume_from + source_processed, "status": "completed" if limit is None else "partial", "last_document_id": indexed_document_ids[-1] if indexed_document_ids else None}
        if limit is not None and processed >= limit: break
    if limit is None:
        write_sync_state(
            state_path,
            {
                "schema_version": 2,
                "manifest_hash": manifest_hash,
                "status": "completed",
                "processed": processed,
                "indexed": indexed,
                "skipped": skipped,
                "chunk_count": chunk_count,
                "last_document_id": indexed_document_ids[-1] if indexed_document_ids else None,
                "sources": source_state,
            },
        )

    return {
        "processed": processed,
        "indexed": indexed,
        "skipped": skipped,
        "chunk_count": chunk_count,
        "db_path": "supabase",
        "total_documents": indexed,
        "total_chunks": chunk_count,
        "indexed_document_ids": indexed_document_ids,
    }


def search_chunks_supabase(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    source_subtypes: Optional[set[str]] = None,
    tax_domains: Optional[set[str]] = None,
    document_ids: Optional[set[str]] = None,
    signatures: Optional[set[str]] = None,
    legal_provisions: Optional[set[str]] = None,
) -> list[RagChunk]:
    if not is_supabase_rag_enabled() or not is_supabase_rag_configured():
        return []

    schema, _, _, search_function = get_supabase_target()
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    headers = build_supabase_headers(schema=schema)
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    query_embedding, _ = compute_embedding(expanded_query, dimensions=config.embedding_dimensions)

    with httpx.Client(timeout=config.supabase_request_timeout, headers=headers) as client:
        response = request_with_retry(
            client,
            "POST",
            f"{supabase_url}/rest/v1/rpc/{search_function}",
            max_retries=config.supabase_max_retries,
            json={
                "search_query": expanded_query,
                "match_count": effective_limit,
                "query_embedding": query_embedding,
                "lexical_weight": config.hybrid_lexical_weight,
                "semantic_weight": config.hybrid_semantic_weight,
                # RPC applies these predicates before candidate limits.
                "p_source_types": sorted(source_types) if source_types else None,
                "p_source_subtypes": sorted(source_subtypes) if source_subtypes else None,
                "p_tax_domains": sorted(tax_domains) if tax_domains else None,
                "p_document_ids": sorted(document_ids) if document_ids else None,
                "p_signatures": sorted(signatures) if signatures else None,
                "p_legal_provisions": sorted(legal_provisions) if legal_provisions else None,
            },
        )
        response.raise_for_status()

    rows = response.json()
    return [
        RagChunk(
            chunk_id=str(row.get("chunk_id") or ""),
            document_id=str(row.get("document_id") or ""),
            chunk_index=int(row.get("chunk_index") or 0),
            score=float(row.get("score") or 0.0),
            chunk_text=str(row.get("chunk_text") or ""),
            subject=str(row.get("subject") or "Bez tytułu"),
            signature=str(row.get("signature") or "") or None,
            published_date=str(row.get("published_date") or "") or None,
            source_url=str(row.get("source_url") or "") or None,
            category=str(row.get("category") or "") or None,
            source=str(row.get("source") or ""),
            source_type=str(row.get("source_type") or "interpretation"),
            source_subtype=str(row.get("source_subtype") or "") or None,
            authority=str(row.get("authority") or "") or None,
            publication=str(row.get("publication") or "") or None,
            legal_state_date=str(row.get("legal_state_date") or "") or None,
            source_pages=[int(value) for value in (row.get("source_pages") or []) if str(value).isdigit()],
            legal_provisions=[str(value) for value in (row.get("legal_provisions") or [])],
        )
        for row in rows
        if row.get("document_id") and row.get("chunk_text")
    ]
