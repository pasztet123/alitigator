from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv


def build_search_text(document_row: sqlite3.Row, chunk_text: str) -> str:
    def loads_text(field: str) -> list[str]:
        raw = document_row[field] or "[]"
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    parts = [
        str(document_row["subject"] or "").strip(),
        str(document_row["signature"] or "").strip(),
        str(document_row["category"] or "").strip(),
        " | ".join(loads_text("keywords_json")),
        " | ".join(loads_text("legal_provisions_json")),
        " | ".join(loads_text("issues_json")),
        " | ".join(loads_text("law_tags_json")),
        chunk_text.strip(),
    ]
    return "\n".join(part for part in parts if part)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast migration from local SQLite RAG index to MariaDB/MySQL.")
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument("--sqlite-path", default="apps/api/data/processed/eureka_rag.sqlite3")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--document-batch-size", type=int, default=500)
    parser.add_argument("--chunk-batch-size", type=int, default=2000)
    parser.add_argument("--progress-every-documents", type=int, default=100)
    parser.add_argument("--start-after-document-id", default="")
    parser.add_argument("--rebuild-indexes", action="store_true", default=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)

    from app.mysql_rag import ensure_schema, get_mysql_target, mysql_connection

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite index not found: {sqlite_path}")

    sqlite_connection = sqlite3.connect(sqlite_path)
    sqlite_connection.row_factory = sqlite3.Row
    documents_table, chunks_table = get_mysql_target()

    with mysql_connection() as mysql_connection_handle:
        ensure_schema(mysql_connection_handle)
        with mysql_connection_handle.cursor() as mysql_cursor:
            if args.truncate:
                print("Truncating MySQL tables...", flush=True)
                mysql_cursor.execute(f"DELETE FROM `{chunks_table}`")
                mysql_cursor.execute(f"DELETE FROM `{documents_table}`")
                mysql_connection_handle.commit()

            print("Dropping secondary chunk indexes for faster bulk load...", flush=True)
            for statement in (
                f"ALTER TABLE `{chunks_table}` DROP INDEX ft_search",
                f"ALTER TABLE `{chunks_table}` DROP INDEX idx_document_id",
                f"ALTER TABLE `{chunks_table}` DROP INDEX idx_document_chunk",
            ):
                try:
                    mysql_cursor.execute(statement)
                except Exception:
                    pass
            mysql_connection_handle.commit()

            sqlite_documents = sqlite_connection.cursor()
            start_after_document_id = args.start_after_document_id.strip()
            if start_after_document_id:
                sqlite_documents.execute(
                    "SELECT * FROM documents WHERE document_id > ? ORDER BY document_id",
                    (start_after_document_id,),
                )
            else:
                sqlite_documents.execute("SELECT * FROM documents ORDER BY document_id")

            migrated_documents = 0
            migrated_chunks = 0
            progress_documents = 0
            while True:
                document_rows = sqlite_documents.fetchmany(args.document_batch_size)
                if not document_rows:
                    break

                mysql_cursor.executemany(
                    f"""
                    INSERT INTO `{documents_table}` (
                        document_id, content_sha256, source, source_type, source_subtype, authority,
                        jurisdiction, act_title, publication, legal_state_date, source_pages_json,
                        subject, signature, published_date, source_url, category, keywords_json,
                        legal_provisions_json, issues_json, law_tags_json, tax_domain, signature_family,
                        question_text, facts_text, decision_text, indexed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON DUPLICATE KEY UPDATE
                        content_sha256 = VALUES(content_sha256),
                        source = VALUES(source),
                        source_type = VALUES(source_type),
                        source_subtype = VALUES(source_subtype),
                        authority = VALUES(authority),
                        jurisdiction = VALUES(jurisdiction),
                        act_title = VALUES(act_title),
                        publication = VALUES(publication),
                        legal_state_date = VALUES(legal_state_date),
                        source_pages_json = VALUES(source_pages_json),
                        subject = VALUES(subject),
                        signature = VALUES(signature),
                        published_date = VALUES(published_date),
                        source_url = VALUES(source_url),
                        category = VALUES(category),
                        keywords_json = VALUES(keywords_json),
                        legal_provisions_json = VALUES(legal_provisions_json),
                        issues_json = VALUES(issues_json),
                        law_tags_json = VALUES(law_tags_json),
                        tax_domain = VALUES(tax_domain),
                        signature_family = VALUES(signature_family),
                        question_text = VALUES(question_text),
                        facts_text = VALUES(facts_text),
                        decision_text = VALUES(decision_text),
                        indexed_at = VALUES(indexed_at)
                    """,
                    [
                        (
                            row["document_id"],
                            row["content_sha256"],
                            row["source"],
                            row["source_type"],
                            row["source_subtype"],
                            row["authority"],
                            row["jurisdiction"],
                            row["act_title"],
                            row["publication"],
                            row["legal_state_date"],
                            row["source_pages_json"],
                            row["subject"],
                            row["signature"],
                            row["published_date"],
                            row["source_url"],
                            row["category"],
                            row["keywords_json"],
                            row["legal_provisions_json"],
                            row["issues_json"],
                            row["law_tags_json"],
                            row["tax_domain"],
                            row["signature_family"],
                            row["question_text"],
                            row["facts_text"],
                            row["decision_text"],
                            row["indexed_at"],
                        )
                        for row in document_rows
                    ],
                )
                mysql_connection_handle.commit()
                migrated_documents += len(document_rows)

                for document_row in document_rows:
                    sqlite_chunks = sqlite_connection.cursor()
                    sqlite_chunks.execute(
                        """
                        SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text, c.chunk_chars, d.*
                        FROM chunks c
                        JOIN documents d ON d.document_id = c.document_id
                        WHERE c.document_id = ?
                        ORDER BY c.chunk_index
                        """,
                        (document_row["document_id"],),
                    )

                    while True:
                        chunk_rows = sqlite_chunks.fetchmany(args.chunk_batch_size)
                        if not chunk_rows:
                            break
                        mysql_cursor.executemany(
                            f"""
                            INSERT INTO `{chunks_table}` (
                                chunk_id, document_id, chunk_index, chunk_text, chunk_chars,
                                search_text, question_text, facts_text, tax_domain
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                document_id = VALUES(document_id),
                                chunk_index = VALUES(chunk_index),
                                chunk_text = VALUES(chunk_text),
                                chunk_chars = VALUES(chunk_chars),
                                search_text = VALUES(search_text),
                                question_text = VALUES(question_text),
                                facts_text = VALUES(facts_text),
                                tax_domain = VALUES(tax_domain)
                            """,
                            [
                                (
                                    row["chunk_id"],
                                    row["document_id"],
                                    row["chunk_index"],
                                    row["chunk_text"],
                                    row["chunk_chars"],
                                    build_search_text(row, row["chunk_text"]),
                                    row["question_text"],
                                    row["facts_text"],
                                    row["tax_domain"],
                                )
                                for row in chunk_rows
                            ],
                        )
                        mysql_connection_handle.commit()
                        migrated_chunks += len(chunk_rows)

                    progress_documents += 1
                    if (
                        args.progress_every_documents > 0
                        and progress_documents % args.progress_every_documents == 0
                    ):
                        print(
                            {
                                "documents_migrated": migrated_documents,
                                "chunks_migrated": migrated_chunks,
                                "last_document_id": document_row["document_id"],
                            },
                            flush=True,
                        )

            print("Rebuilding chunk indexes...", flush=True)
            for statement in (
                f"ALTER TABLE `{chunks_table}` ADD INDEX idx_document_id (document_id)",
                f"ALTER TABLE `{chunks_table}` ADD INDEX idx_document_chunk (document_id, chunk_index)",
                f"ALTER TABLE `{chunks_table}` ADD FULLTEXT INDEX ft_search (search_text, question_text, facts_text, tax_domain)",
            ):
                try:
                    mysql_cursor.execute(statement)
                except Exception:
                    pass
            mysql_connection_handle.commit()

            mysql_cursor.execute(f"SELECT COUNT(*) AS count FROM `{documents_table}`")
            total_documents = int(mysql_cursor.fetchone()["count"])
            mysql_cursor.execute(f"SELECT COUNT(*) AS count FROM `{chunks_table}`")
            total_chunks = int(mysql_cursor.fetchone()["count"])

    sqlite_connection.close()
    print({"total_documents": total_documents, "total_chunks": total_chunks}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
