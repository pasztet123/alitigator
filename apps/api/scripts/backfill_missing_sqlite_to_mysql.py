"""Safely add corpus rows absent from MariaDB using the local SQLite index.

This loader deliberately does not truncate, delete, or alter remote tables. It
only inserts missing document IDs, chunk IDs and citations in batches, keeping
the production FULLTEXT index online during the backfill. It can be run
repeatedly and repairs interrupted migrations where a document exists but only
some of its chunks were copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


DOCUMENT_COLUMNS = (
    "document_id", "content_sha256", "source", "source_type", "source_subtype",
    "authority", "jurisdiction", "act_title", "publication", "legal_state_date",
    "source_pages_json", "subject", "signature", "published_date", "source_url",
    "category", "keywords_json", "legal_provisions_json", "issues_json",
    "law_tags_json", "tax_domain", "signature_family", "question_text", "facts_text",
    "decision_text", "indexed_at",
)
CHUNK_COLUMNS = (
    "chunk_id", "document_id", "chunk_index", "chunk_text", "chunk_chars",
    "provision_id", "display_reference", "search_text", "question_text", "facts_text",
    "tax_domain",
)


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def selected_shard(document_id: str, *, shard_count: int, shard_index: int) -> bool:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count == shard_index


def search_text(row: sqlite3.Row) -> str:
    def list_text(field: str) -> str:
        try:
            return " | ".join(str(item).strip() for item in json.loads(row[field] or "[]") if str(item).strip())
        except json.JSONDecodeError:
            return ""
    return "\n".join(part for part in (
        str(row["subject"] or "").strip(), str(row["signature"] or "").strip(),
        str(row["category"] or "").strip(), list_text("keywords_json"),
        list_text("legal_provisions_json"), list_text("issues_json"),
        list_text("law_tags_json"), str(row["chunk_text"] or "").strip(),
    ) if part)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insert missing local RAG documents, chunks and citations into MariaDB without deleting data."
    )
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument("--sqlite-path", default="apps/api/data/processed/eureka_rag.sqlite3")
    parser.add_argument(
        "--source-type",
        action="append",
        choices=("statute", "interpretation", "general_interpretation", "judgment"),
    )
    parser.add_argument("--document-batch-size", type=int, default=40)
    parser.add_argument("--chunk-batch-size", type=int, default=1500)
    parser.add_argument("--citation-batch-size", type=int, default=2000)
    parser.add_argument(
        "--sync-all-citations",
        action="store_true",
        help=(
            "Idempotently INSERT IGNORE every selected local citation, including citations "
            "for chunks that already exist remotely. This can process millions of rows."
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)
    from app.mysql_rag import get_mysql_target, mysql_connection

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        raise SystemExit("Local SQLite index is unavailable")
    selected_types = set(args.source_type or ("statute", "interpretation", "general_interpretation", "judgment"))
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Invalid shard selection")
    sqlite_connection = sqlite3.connect(sqlite_path)
    sqlite_connection.row_factory = sqlite3.Row
    documents_table, chunks_table = get_mysql_target()

    with mysql_connection() as mysql_connection_handle:
        with mysql_connection_handle.cursor() as cursor:
            # The loader inserts parent documents before their children.  The
            # checks are disabled only for this connection to avoid per-row
            # foreign-key work during a large, already validated import.
            cursor.execute("SET SESSION foreign_key_checks = 0")
            cursor.execute("SET SESSION unique_checks = 0")
            cursor.execute("SELECT document_id FROM `%s`" % documents_table)
            present_ids = {str(row["document_id"]) for row in cursor.fetchall()}
            cursor.execute("SELECT chunk_id FROM `%s`" % chunks_table)
            present_chunk_ids = {str(row["chunk_id"]) for row in cursor.fetchall()}

        placeholders = ", ".join("?" for _ in selected_types)
        local_rows = sqlite_connection.execute(
            "SELECT * FROM documents WHERE source_type IN (%s) ORDER BY document_id" % placeholders,
            tuple(sorted(selected_types)),
        ).fetchall()
        missing = [
            row for row in local_rows
            if str(row["document_id"]) not in present_ids
            and selected_shard(
                str(row["document_id"]),
                shard_count=args.shard_count,
                shard_index=args.shard_index,
            )
        ]
        by_type: dict[str, int] = {}
        for row in missing:
            source_type = str(row["source_type"])
            by_type[source_type] = by_type.get(source_type, 0) + 1
        missing_chunk_ids: list[str] = []
        missing_chunks_by_type: dict[str, int] = {}
        local_chunk_cursor = sqlite_connection.execute(
            """
            SELECT c.chunk_id, c.document_id, d.source_type
            FROM chunks c JOIN documents d ON d.document_id = c.document_id
            WHERE d.source_type IN (%s)
            ORDER BY c.chunk_id
            """ % placeholders,
            tuple(sorted(selected_types)),
        )
        while True:
            rows = local_chunk_cursor.fetchmany(5000)
            if not rows:
                break
            for row in rows:
                chunk_id = str(row["chunk_id"])
                document_id = str(row["document_id"])
                if chunk_id in present_chunk_ids or not selected_shard(
                    document_id,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                ):
                    continue
                missing_chunk_ids.append(chunk_id)
                source_type = str(row["source_type"])
                missing_chunks_by_type[source_type] = missing_chunks_by_type.get(source_type, 0) + 1

        missing_citation_count = 0
        for chunk_id_batch in chunks(missing_chunk_ids, 800):
            row = sqlite_connection.execute(
                "SELECT COUNT(*) AS count FROM chunk_citations WHERE chunk_id IN (%s)"
                % ", ".join("?" for _ in chunk_id_batch),
                chunk_id_batch,
            ).fetchone()
            missing_citation_count += int(row["count"] if row else 0)

        citation_candidates_by_type: dict[str, int] = {}
        if args.sync_all_citations:
            citation_count_rows = sqlite_connection.execute(
                """
                SELECT d.source_type, COUNT(*) AS count
                FROM chunk_citations cc
                JOIN chunks c ON c.chunk_id = cc.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                WHERE d.source_type IN (%s)
                GROUP BY d.source_type
                """ % placeholders,
                tuple(sorted(selected_types)),
            ).fetchall()
            citation_candidates_by_type = {
                str(row["source_type"]): int(row["count"])
                for row in citation_count_rows
            }

        print(
            {
                "missing_documents": by_type,
                "missing_chunks": missing_chunks_by_type,
                "citations_for_missing_chunks": missing_citation_count,
                "all_citation_candidates": citation_candidates_by_type,
                "sync_all_citations": args.sync_all_citations,
                "dry_run": args.dry_run,
            },
            flush=True,
        )
        if args.dry_run:
            return 0

        document_sql = "INSERT IGNORE INTO `%s` (%s) VALUES (%s)" % (
            documents_table,
            ", ".join("`%s`" % column for column in DOCUMENT_COLUMNS),
            ", ".join(["%s"] * len(DOCUMENT_COLUMNS)),
        )
        chunk_sql = "INSERT IGNORE INTO `%s` (%s) VALUES (%s)" % (
            chunks_table,
            ", ".join("`%s`" % column for column in CHUNK_COLUMNS),
            ", ".join(["%s"] * len(CHUNK_COLUMNS)),
        )
        citations_table = "%s_citations" % chunks_table
        copied_documents = copied_chunks = copied_citations = 0
        for batch in chunks(missing, max(1, args.document_batch_size)):
            with mysql_connection_handle.cursor() as cursor:
                cursor.executemany(
                    document_sql,
                    [tuple(row[column] for column in DOCUMENT_COLUMNS) for row in batch],
                )
            mysql_connection_handle.commit()
            copied_documents += len(batch)
            print({"documents": copied_documents}, flush=True)

        effective_chunk_batch_size = min(max(1, args.chunk_batch_size), 800)
        for chunk_id_batch in chunks(missing_chunk_ids, effective_chunk_batch_size):
            rows = sqlite_connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text, c.chunk_chars,
                       c.provision_id, c.display_reference, d.*
                FROM chunks c JOIN documents d ON d.document_id = c.document_id
                WHERE c.chunk_id IN (%s) ORDER BY c.document_id, c.chunk_index
                """ % ", ".join("?" for _ in chunk_id_batch),
                chunk_id_batch,
            ).fetchall()
            values = [
                (
                    row["chunk_id"], row["document_id"], row["chunk_index"], row["chunk_text"],
                    row["chunk_chars"], row["provision_id"], row["display_reference"], search_text(row),
                    row["question_text"], row["facts_text"], row["tax_domain"],
                ) for row in rows
            ]
            if values:
                with mysql_connection_handle.cursor() as cursor:
                    cursor.executemany(chunk_sql, values)
                copied_chunks += len(values)
            if rows:
                citations = []
                inserted_chunk_ids = [str(row["chunk_id"]) for row in rows]
                citations.extend(sqlite_connection.execute(
                    "SELECT chunk_id, citation FROM chunk_citations WHERE chunk_id IN (%s)"
                    % ", ".join("?" for _ in inserted_chunk_ids),
                    inserted_chunk_ids,
                ).fetchall())
                if citations:
                    with mysql_connection_handle.cursor() as cursor:
                        cursor.executemany(
                            "INSERT IGNORE INTO `%s` (chunk_id, citation) VALUES (%%s, %%s)" % citations_table,
                            [(row["chunk_id"], row["citation"]) for row in citations],
                        )
                    copied_citations += len(citations)
            mysql_connection_handle.commit()
            print({"documents": copied_documents, "chunks": copied_chunks, "citations": copied_citations}, flush=True)

        if args.sync_all_citations:
            citation_sql = "INSERT IGNORE INTO `%s` (chunk_id, citation) VALUES (%%s, %%s)" % citations_table
            citation_cursor = sqlite_connection.execute(
                """
                SELECT cc.chunk_id, cc.citation, c.document_id
                FROM chunk_citations cc
                JOIN chunks c ON c.chunk_id = cc.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                WHERE d.source_type IN (%s)
                ORDER BY cc.chunk_id, cc.citation
                """ % placeholders,
                tuple(sorted(selected_types)),
            )
            citation_batch_size = max(1, args.citation_batch_size)
            attempted_citations = 0
            inserted_citations = 0
            batch_number = 0
            while True:
                rows = citation_cursor.fetchmany(citation_batch_size)
                if not rows:
                    break
                values = [
                    (str(row["chunk_id"]), str(row["citation"]))
                    for row in rows
                    if selected_shard(
                        str(row["document_id"]),
                        shard_count=args.shard_count,
                        shard_index=args.shard_index,
                    )
                ]
                if not values:
                    continue
                with mysql_connection_handle.cursor() as cursor:
                    cursor.executemany(citation_sql, values)
                    inserted_citations += max(0, int(cursor.rowcount))
                mysql_connection_handle.commit()
                attempted_citations += len(values)
                batch_number += 1
                if batch_number % 100 == 0:
                    print(
                        {
                            "citation_rows_checked": attempted_citations,
                            "citation_rows_inserted": inserted_citations,
                        },
                        flush=True,
                    )
            copied_citations += inserted_citations
            print(
                {
                    "citation_rows_checked": attempted_citations,
                    "citation_rows_inserted": inserted_citations,
                    "citations_total_inserted_this_run": copied_citations,
                },
                flush=True,
            )

    sqlite_connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
