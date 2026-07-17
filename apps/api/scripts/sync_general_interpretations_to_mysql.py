"""Synchronize MF general interpretations from the local corpus to MariaDB.

The Eureka source historically stored both individual and general
interpretations under ``source_type=interpretation`` and distinguished them
only with ``source_subtype=general``.  The legal-research pipeline has a
separate authority lane for general interpretations, so this command writes
the canonical remote representation:

``source_type=general_interpretation, source_subtype=general``.

It replaces only the selected document IDs, including their chunks and exact
citations.  The operation is idempotent and deliberately leaves every other
corpus source untouched.  Use ``--normalize-local`` to migrate the local
SQLite index after the remote transaction succeeds.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from backfill_missing_sqlite_to_mysql import CHUNK_COLUMNS, DOCUMENT_COLUMNS, search_text


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GENERAL_SOURCE_TYPE = "general_interpretation"
GENERAL_SOURCE_SUBTYPE = "general"
DEFAULT_AUTHORITY = "Minister Finansów"


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start:start + max(1, size)]


def canonical_general_document(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a document row in the canonical MF-general representation."""
    value = {column: row[column] for column in DOCUMENT_COLUMNS}
    value["source"] = str(value.get("source") or "eureka").strip() or "eureka"
    value["source_type"] = GENERAL_SOURCE_TYPE
    value["source_subtype"] = GENERAL_SOURCE_SUBTYPE
    value["authority"] = str(value.get("authority") or "").strip() or DEFAULT_AUTHORITY
    return value


def general_search_text(row: Mapping[str, Any]) -> str:
    """Keep a stable authority-class label in MariaDB's full-text index."""
    base = search_text(row)  # shared metadata and chunk content envelope
    return "\n".join(
        value
        for value in ("Interpretacja ogólna | Minister Finansów", base)
        if value
    )


def load_general_interpretations(sqlite_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]:
    if not sqlite_path.exists():
        raise SystemExit(f"Local SQLite index is unavailable: {sqlite_path}")
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        documents = [
            canonical_general_document(dict(row))
            for row in connection.execute(
                """
                SELECT *
                FROM documents
                WHERE source = 'eureka'
                  AND source_subtype = 'general'
                  AND source_type IN ('interpretation', 'general_interpretation')
                ORDER BY document_id
                """
            ).fetchall()
        ]
        if not documents:
            return [], [], []
        document_ids = [str(item["document_id"]) for item in documents]
        placeholders = ", ".join("?" for _ in document_ids)
        chunk_rows = connection.execute(
            f"""
            SELECT c.*, d.*
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.document_id IN ({placeholders})
            ORDER BY c.document_id, c.chunk_index, c.chunk_id
            """,
            document_ids,
        ).fetchall()
        chunks = [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "chunk_index": int(row["chunk_index"]),
                "chunk_text": str(row["chunk_text"] or ""),
                "chunk_chars": int(row["chunk_chars"] or 0),
                "provision_id": str(row["provision_id"] or ""),
                "display_reference": str(row["display_reference"] or ""),
                "search_text": general_search_text(row),
                "question_text": str(row["question_text"] or ""),
                "facts_text": str(row["facts_text"] or ""),
                "tax_domain": str(row["tax_domain"] or ""),
            }
            for row in chunk_rows
        ]
        citations = [
            (str(row["chunk_id"]), str(row["citation"]))
            for row in connection.execute(
                f"""
                SELECT cc.chunk_id, cc.citation
                FROM chunk_citations cc
                JOIN chunks c ON c.chunk_id = cc.chunk_id
                WHERE c.document_id IN ({placeholders})
                ORDER BY cc.chunk_id, cc.citation
                """,
                document_ids,
            ).fetchall()
        ]
        return documents, chunks, citations
    finally:
        connection.close()


def normalize_local_documents(sqlite_path: Path, document_ids: list[str]) -> int:
    """Migrate only synchronized general-MF records in the local SQLite DB."""
    if not document_ids:
        return 0
    connection = sqlite3.connect(sqlite_path)
    try:
        changed = 0
        for batch in batched(document_ids, 500):
            placeholders = ", ".join("?" for _ in batch)
            cursor = connection.execute(
                f"""
                UPDATE documents
                SET source_type = ?, source_subtype = ?,
                    authority = CASE WHEN TRIM(authority) = '' THEN ? ELSE authority END
                WHERE document_id IN ({placeholders})
                  AND source_subtype = 'general'
                  AND source_type IN ('interpretation', 'general_interpretation')
                """,
                (GENERAL_SOURCE_TYPE, GENERAL_SOURCE_SUBTYPE, DEFAULT_AUTHORITY, *batch),
            )
            changed += max(0, int(cursor.rowcount))
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def remote_count(cursor: Any, documents_table: str, document_ids: list[str]) -> dict[str, int]:
    cursor.execute(
        f"""
        SELECT source_type, COUNT(*) AS count
        FROM `{documents_table}`
        WHERE source_subtype = %s
        GROUP BY source_type
        """,
        (GENERAL_SOURCE_SUBTYPE,),
    )
    by_type = {str(row["source_type"]): int(row["count"]) for row in cursor.fetchall()}
    selected = 0
    for batch in batched(document_ids, 500):
        cursor.execute(
            f"SELECT COUNT(*) AS count FROM `{documents_table}` "
            f"WHERE document_id IN ({', '.join('%s' for _ in batch)})",
            batch,
        )
        selected += int(cursor.fetchone()["count"])
    return {
        "remote_general_interpretations": by_type.get(GENERAL_SOURCE_TYPE, 0),
        "remote_legacy_general_interpretations": by_type.get("interpretation", 0),
        "remote_selected_documents": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument("--sqlite-path", default="apps/api/data/processed/eureka_rag.sqlite3")
    parser.add_argument("--apply", action="store_true", help="Replace the scoped records in MariaDB.")
    parser.add_argument(
        "--normalize-local",
        action="store_true",
        help="After a successful remote replacement, migrate matching local SQLite document types too.",
    )
    parser.add_argument("--document-batch-size", type=int, default=100)
    parser.add_argument("--chunk-batch-size", type=int, default=500)
    parser.add_argument("--citation-batch-size", type=int, default=1000)
    args = parser.parse_args()

    if args.normalize_local and not args.apply:
        raise SystemExit("--normalize-local requires --apply so the local migration follows a successful remote sync")
    sqlite_path = Path(args.sqlite_path)
    documents, chunks, citations = load_general_interpretations(sqlite_path)
    document_ids = [str(item["document_id"]) for item in documents]
    if not documents:
        print({"local_general_interpretations": 0, "apply": args.apply}, flush=True)
        return 0

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)
    from app.mysql_rag import ensure_schema, get_mysql_target, mysql_connection

    documents_table, chunks_table = get_mysql_target()
    citations_table = f"{chunks_table}_citations"
    if not all(_SAFE_IDENTIFIER_RE.fullmatch(value) for value in (documents_table, chunks_table, citations_table)):
        raise SystemExit("Unsafe MariaDB table configuration")

    with mysql_connection() as remote:
        # A dry run must remain read-only.  Schema preparation belongs to the
        # explicit apply path, before the scoped replacement transaction.
        if args.apply:
            ensure_schema(remote)
        with remote.cursor() as cursor:
            before = remote_count(cursor, documents_table, document_ids)
        summary = {
            "local_general_interpretations": len(documents),
            "local_chunks": len(chunks),
            "local_citations": len(citations),
            **before,
            "apply": args.apply,
            "normalize_local": args.normalize_local,
        }
        print(summary, flush=True)
        if not args.apply:
            return 0

        document_sql = "INSERT INTO `%s` (%s) VALUES (%s)" % (
            documents_table,
            ", ".join("`%s`" % column for column in DOCUMENT_COLUMNS),
            ", ".join(["%s"] * len(DOCUMENT_COLUMNS)),
        )
        chunk_sql = "INSERT INTO `%s` (%s) VALUES (%s)" % (
            chunks_table,
            ", ".join("`%s`" % column for column in CHUNK_COLUMNS),
            ", ".join(["%s"] * len(CHUNK_COLUMNS)),
        )
        try:
            with remote.cursor() as cursor:
                for batch in batched(document_ids, 500):
                    placeholders = ", ".join("%s" for _ in batch)
                    cursor.execute(
                        f"DELETE cc FROM `{citations_table}` cc "
                        f"JOIN `{chunks_table}` c ON c.chunk_id = cc.chunk_id "
                        f"WHERE c.document_id IN ({placeholders})",
                        batch,
                    )
                    cursor.execute(
                        f"DELETE FROM `{chunks_table}` WHERE document_id IN ({placeholders})",
                        batch,
                    )
                    cursor.execute(
                        f"DELETE FROM `{documents_table}` WHERE document_id IN ({placeholders})",
                        batch,
                    )
                for batch in batched(documents, args.document_batch_size):
                    cursor.executemany(document_sql, [tuple(row[column] for column in DOCUMENT_COLUMNS) for row in batch])
                for batch in batched(chunks, args.chunk_batch_size):
                    cursor.executemany(chunk_sql, [tuple(row[column] for column in CHUNK_COLUMNS) for row in batch])
                citation_sql = f"INSERT INTO `{citations_table}` (chunk_id, citation) VALUES (%s, %s)"
                for batch in batched(citations, args.citation_batch_size):
                    cursor.executemany(citation_sql, batch)
            remote.commit()
        except Exception:
            remote.rollback()
            raise

        with remote.cursor() as cursor:
            after = remote_count(cursor, documents_table, document_ids)

    normalized_local = normalize_local_documents(sqlite_path, document_ids) if args.normalize_local else 0
    print(
        {
            "replaced_general_interpretations": len(documents),
            "inserted_general_interpretation_chunks": len(chunks),
            "inserted_general_interpretation_citations": len(citations),
            "local_documents_normalized": normalized_local,
            **after,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
