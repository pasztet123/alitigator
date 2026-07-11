"""Build or resume the versioned legal-rag-v2 embedding index.

This command reads the configured corpus and writes a separate SQLite vector
index. It never migrates or mutates the production corpus database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .embeddings import (
    EmbeddingInput,
    OfflineHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VersionedEmbeddingIndex,
)
from .provision_graph import ProvisionParser


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _json_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _rows_from_sqlite(limit: int | None) -> Iterable[Mapping[str, Any]]:
    from app.rag import get_rag_config

    path = get_rag_config().db_path
    if not path.exists():
        return []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    sql = """
        SELECT c.chunk_id, c.document_id, c.chunk_text,
               d.source_type, d.source_subtype, d.tax_domain,
               d.legal_state_date, d.published_date,
               d.legal_provisions_json, d.source_pages_json
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        ORDER BY c.document_id, c.chunk_index
    """
    if limit is not None:
        sql += " LIMIT ?"
    try:
        rows = connection.execute(sql, (limit,) if limit is not None else ()).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _rows_from_mysql(limit: int | None) -> Iterable[Mapping[str, Any]]:
    from app.mysql_rag import get_mysql_target, mysql_connection

    documents_table, chunks_table = get_mysql_target()
    if not all(
        _SAFE_IDENTIFIER_RE.fullmatch(value)
        for value in (documents_table, chunks_table)
    ):
        raise ValueError("Unsafe MySQL RAG table identifier")
    sql = f"""
        SELECT c.chunk_id, c.document_id, c.chunk_text,
               d.source_type, d.source_subtype, d.tax_domain,
               d.legal_state_date, d.published_date,
               d.legal_provisions_json, d.source_pages_json
        FROM `{chunks_table}` c
        JOIN `{documents_table}` d ON d.document_id = c.document_id
        ORDER BY c.document_id, c.chunk_index
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]


def iter_embedding_inputs(limit: int | None = None) -> Iterable[EmbeddingInput]:
    backend = os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower()
    rows = (
        _rows_from_mysql(limit)
        if backend in {"mysql", "mariadb"}
        else _rows_from_sqlite(limit)
    )
    parser = ProvisionParser()
    for row in rows:
        text = str(row.get("chunk_text") or "")
        document_id = str(row.get("document_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        metadata = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "source_type": str(row.get("source_type") or ""),
            "source_subtype": str(row.get("source_subtype") or ""),
            "tax_domains": [str(row.get("tax_domain") or "").upper()]
            if row.get("tax_domain")
            else [],
            "legal_state_date": str(row.get("legal_state_date") or ""),
            "published_date": str(row.get("published_date") or ""),
            "legal_provisions": _json_list(row.get("legal_provisions_json")),
            "source_pages": _json_list(row.get("source_pages_json")),
        }
        units = parser.parse(
            text,
            document_id=document_id or chunk_id,
            version_id=str(row.get("legal_state_date") or "current"),
            metadata=metadata,
        )
        if units:
            for unit in units:
                yield EmbeddingInput(
                    item_id=unit.provision_id,
                    text=unit.text,
                    metadata={
                        **metadata,
                        "provision_id": unit.provision_id,
                        "citation": unit.citation,
                        "effective_from": unit.effective_from,
                        "effective_to": unit.effective_to,
                    },
                )
        elif text.strip():
            yield EmbeddingInput(item_id=chunk_id, text=text, metadata=metadata)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dimensions = args.dimensions
    if args.offline_hash:
        if not args.allow_offline_hash:
            raise RuntimeError(
                "Offline hash embeddings require --allow-offline-hash explicit opt-in"
            )
        provider = OfflineHashEmbeddingProvider(dimensions=dimensions)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for real embedding reindex")
        provider = OpenAIEmbeddingProvider(
            model=args.model,
            dimensions=dimensions,
            api_key=api_key,
        )
    index = VersionedEmbeddingIndex(
        args.index_path,
        provider,
        schema_version=args.schema_version,
        chunker_version=args.chunker_version,
    )
    try:
        report = await index.index_records(
            iter_embedding_inputs(args.limit),
            batch_size=args.batch_size,
        )
        return {
            "index_path": str(args.index_path),
            "provider": provider.model,
            "dimensions": provider.dimensions,
            "total": report.total,
            "indexed": report.indexed,
            "skipped": report.skipped,
            "batches_committed": report.batches_committed,
        }
    finally:
        index.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(
            os.getenv(
                "EMBEDDING_INDEX_PATH",
                "artifacts/legal_rag_v2/embedding_index.sqlite3",
            )
        ),
    )
    parser.add_argument(
        "--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    parser.add_argument(
        "--dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
    )
    parser.add_argument(
        "--schema-version", default=os.getenv("EMBEDDING_SCHEMA_VERSION", "v1")
    )
    parser.add_argument(
        "--chunker-version",
        default=os.getenv("EMBEDDING_CHUNKER_VERSION", "provision_units_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline-hash", action="store_true")
    parser.add_argument("--allow-offline-hash", action="store_true")
    return parser


def main() -> None:
    result = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
