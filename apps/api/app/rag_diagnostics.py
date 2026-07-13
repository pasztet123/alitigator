"""Read-only corpus and backend diagnostics shared by CLI and admin health."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from app.rag_runtime import iter_configured_corpus_sources, resolve_rag_runtime


class RagDiagnosticsBackend(Protocol):
    """Read-only diagnostic contract implemented by every RAG storage target."""

    def inventory(self) -> dict[str, Any]:
        """Return counts, type/subtype distribution and sampled stable IDs."""
        ...


def _safe_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("document_id", "source_type", "source_subtype", "signature", "subject")}


def inventory_local_corpus(config: Any) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for source in iter_configured_corpus_sources(config):
        item: dict[str, Any] = {"path": str(source.path), "source_id": source.source_id, "expected_source_types": list(source.expected_source_types), "exists": source.path.exists(), "readable": False}
        if not source.path.exists():
            sources.append(item); continue
        item["file_size"] = source.path.stat().st_size
        types: Counter[str] = Counter(); subtypes: Counter[str] = Counter(); document_ids: set[str] = set()
        lines = valid = invalid = signatures = content = without_content = provisions = 0; samples: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        try:
            with source.path.open("rb") as handle:
                for raw in handle:
                    digest.update(raw); line = raw.strip()
                    if not line: continue
                    lines += 1
                    try: record = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError): invalid += 1; continue
                    valid += 1
                    document_ids.add(str(record.get("document_id") or ""))
                    source_type = str(record.get("source_type") or "interpretation").lower(); types[source_type] += 1
                    subtype = str(record.get("source_subtype") or "").lower();
                    if subtype: subtypes[subtype] += 1
                    if record.get("signature"): signatures += 1
                    if str(record.get("content_text") or record.get("content_text_clean") or record.get("content_html") or "").strip(): content += 1
                    else: without_content += 1
                    if record.get("legal_provisions"): provisions += 1
                    if len(samples) < 3: samples.append(_safe_sample(record))
            item.update({"readable": True, "sha256": digest.hexdigest(), "line_count": lines, "valid_json_count": valid, "invalid_json_count": invalid, "unique_document_count": len(document_ids - {""}), "source_type_distribution": dict(types), "source_subtype_distribution": dict(subtypes), "signature_count": signatures, "records_with_content": content, "records_without_content": without_content, "records_with_legal_provisions": provisions, "samples": samples})
        except OSError as exc:
            item["error"] = type(exc).__name__
        sources.append(item)
    return {"sources": sources, "totals": {"source_count": len(sources), "valid_json_count": sum(int(item.get("valid_json_count", 0)) for item in sources)}}


def _sqlite_inventory(path: Path) -> dict[str, Any]:
    if not path.exists(): return {"backend": "sqlite", "available": False, "reason": "database file missing"}
    with sqlite3.connect("file:%s?mode=ro" % path, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT source_type, source_subtype, COUNT(*) count FROM documents GROUP BY source_type, source_subtype").fetchall()
        counts = {"%s:%s" % (row["source_type"], row["source_subtype"] or ""): int(row["count"]) for row in rows}
        docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]; chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        samples = connection.execute("SELECT document_id, source_type, source_subtype, signature FROM documents ORDER BY document_id LIMIT 12").fetchall()
    return {"backend": "sqlite", "available": True, "documents": docs, "chunks": chunks, "by_type_subtype": counts, "samples": [dict(row) for row in samples]}


def _mysql_inventory() -> dict[str, Any]:
    from app.mysql_rag import is_mysql_rag_configured, mysql_connection, get_mysql_target
    if not is_mysql_rag_configured(): return {"backend": "mysql", "available": False, "reason": "not configured"}
    docs, chunks = get_mysql_target()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT source_type, COALESCE(source_subtype, '') source_subtype, COUNT(*) count FROM `%s` GROUP BY source_type, source_subtype" % docs)
            rows = cursor.fetchall(); cursor.execute("SELECT COUNT(*) count FROM `%s`" % chunks); chunk_count = int((cursor.fetchone() or {}).get("count") or 0)
            cursor.execute("SELECT document_id, source_type, source_subtype, signature FROM `%s` ORDER BY document_id LIMIT 12" % docs)
            samples = cursor.fetchall()
    return {"backend": "mysql", "available": True, "documents": sum(int(row.get("count") or 0) for row in rows), "chunks": chunk_count, "by_type_subtype": {"%s:%s" % (row.get("source_type"), row.get("source_subtype")): int(row.get("count") or 0) for row in rows}, "samples": samples}


def _supabase_inventory() -> dict[str, Any]:
    """REST-only inventory; does not issue mutations or expose credentials."""
    from app.supabase_rag import build_supabase_headers, get_supabase_target, is_supabase_rag_configured
    import httpx
    import os
    if not is_supabase_rag_configured(): return {"backend": "supabase", "available": False, "reason": "not configured"}
    schema, documents, chunks, _ = get_supabase_target(); url = os.getenv("SUPABASE_URL", "").rstrip("/")
    headers = build_supabase_headers(schema=schema); headers["Prefer"] = "count=exact"
    with httpx.Client(timeout=15, headers=headers) as client:
        docs_response = client.get("%s/rest/v1/%s" % (url, documents), params={"select": "document_id,source_type,source_subtype,signature", "limit": "1000"}); docs_response.raise_for_status()
        chunks_response = client.get("%s/rest/v1/%s" % (url, chunks), params={"select": "chunk_id", "limit": "1"}); chunks_response.raise_for_status()
    samples = docs_response.json(); counts: Counter[str] = Counter("%s:%s" % (row.get("source_type") or "", row.get("source_subtype") or "") for row in samples)
    def total(response: Any) -> int:
        content_range = response.headers.get("content-range", "*/0")
        return int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else len(response.json())
    return {"backend": "supabase", "available": True, "documents": total(docs_response), "chunks": total(chunks_response), "by_type_subtype_sample": dict(counts), "samples": samples[:12], "warning": "distribution is sampled to 1000 rows; totals use Content-Range"}


class SQLiteRagDiagnostics:
    def __init__(self, db_path: Path) -> None: self.db_path = db_path
    def inventory(self) -> dict[str, Any]: return _sqlite_inventory(self.db_path)


class MySQLRagDiagnostics:
    def inventory(self) -> dict[str, Any]: return _mysql_inventory()


class SupabaseRagDiagnostics:
    def inventory(self) -> dict[str, Any]: return _supabase_inventory()


def collect_corpus_health(config: Any) -> dict[str, Any]:
    runtime = resolve_rag_runtime()
    diagnostics: RagDiagnosticsBackend = (
        MySQLRagDiagnostics() if runtime.read_backend == "mysql" else
        SupabaseRagDiagnostics() if runtime.read_backend == "supabase" else
        SQLiteRagDiagnostics(Path(config.db_path))
    )
    backend = diagnostics.inventory()
    types = {key.split(":", 1)[0] for key, count in backend.get("by_type_subtype", {}).items() if count}
    missing = sorted({"statute", "interpretation", "judgment"} - types)
    return {"runtime": {"read_backend": runtime.read_backend, "write_backend": runtime.write_backend, "fallback_backend": runtime.fallback_backend}, "backend": backend, "status": "healthy" if not missing else "degraded", "missing_required_source_types": missing}
