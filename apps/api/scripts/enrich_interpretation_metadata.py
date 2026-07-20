"""Backfill locally derivable Eureka metadata without downloading documents.

The command reads existing corpus records only.  It adds canonical provision
aliases and fills blank, unambiguous tax-domain values.  It never changes
content, signatures, source URLs, or source-owned values, and is idempotent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from dotenv import load_dotenv


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.eureka_metadata import enrich_interpretation_metadata


DEFAULT_SQLITE_PATH = API_DIR / "data" / "processed" / "eureka_rag.sqlite3"
SELECT_FIELDS = """
    document_id, subject, keywords_json, legal_provisions_json, issues_json,
    law_tags_json, tax_domain, question_text, facts_text
"""


def json_list(value: object) -> list[object]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def changed_metadata(row: dict[str, Any]) -> tuple[str, str] | None:
    existing_domain = str(row.get("tax_domain") or "").strip().upper()
    existing_provisions = [str(value).strip() for value in json_list(row.get("legal_provisions_json")) if str(value).strip()]
    metadata = enrich_interpretation_metadata(
        tax_domain=existing_domain,
        law_tags=json_list(row.get("law_tags_json")),
        legal_provisions=existing_provisions,
        issues=json_list(row.get("issues_json")),
        subject=str(row.get("subject") or ""),
        question_text=str(row.get("question_text") or ""),
        facts_text=str(row.get("facts_text") or ""),
    )
    provisions_json = json.dumps(list(metadata.legal_provisions), ensure_ascii=False)
    existing_json = json.dumps(existing_provisions, ensure_ascii=False)
    if metadata.tax_domain == existing_domain and provisions_json == existing_json:
        return None
    return metadata.tax_domain, provisions_json


def batches(values: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    for value in values:
        pending.append(value)
        if len(pending) >= size:
            yield pending
            pending = []
    if pending:
        yield pending


def apply_mysql_metadata_batch(
    connection: Any,
    *,
    documents_table: str,
    updates: list[tuple[str, str, str]],
) -> None:
    """Apply one update batch with two server-side statements.

    ``executemany(UPDATE ...)`` sends one round trip for every document.  The
    corpus is on a remote MariaDB instance, so a temporary staging table keeps
    the update bounded even for a gradual backfill.
    """

    if not updates:
        return
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TEMPORARY TABLE IF NOT EXISTS metadata_backfill_updates (
                document_id varchar(191) PRIMARY KEY,
                tax_domain varchar(64) NOT NULL,
                legal_provisions_json LONGTEXT NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        values_sql = ", ".join(["(%s, %s, %s)"] * len(updates))
        values = tuple(value for update in updates for value in (update[2], update[0], update[1]))
        cursor.execute(
            """
            INSERT INTO metadata_backfill_updates
                (document_id, tax_domain, legal_provisions_json)
            VALUES """ + values_sql + """
            ON DUPLICATE KEY UPDATE
                tax_domain = VALUES(tax_domain),
                legal_provisions_json = VALUES(legal_provisions_json)
            """,
            values,
        )
        cursor.execute(
            f"""
            UPDATE `{documents_table}` d
            JOIN metadata_backfill_updates u ON u.document_id = d.document_id
            SET d.tax_domain = u.tax_domain,
                d.legal_provisions_json = u.legal_provisions_json
            """
        )
        cursor.execute("TRUNCATE TABLE metadata_backfill_updates")
    connection.commit()


def update_sqlite(path: Path, *, apply: bool, batch_size: int) -> dict[str, int]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    counts: Counter[str] = Counter()
    try:
        cursor = connection.execute(
            f"SELECT {SELECT_FIELDS} FROM documents WHERE source_type = 'interpretation' ORDER BY document_id"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            updates: list[tuple[str, str, str]] = []
            domain_updates: list[tuple[str, str]] = []
            for source_row in rows:
                row = dict(source_row)
                counts["scanned"] += 1
                result = changed_metadata(row)
                if result is None:
                    continue
                domain, provisions_json = result
                previous_domain = str(row.get("tax_domain") or "").strip().upper()
                previous_provisions = str(row.get("legal_provisions_json") or "[]")
                if domain != previous_domain:
                    counts["tax_domain_changed"] += 1
                    domain_updates.append((domain, str(row["document_id"])))
                if provisions_json != previous_provisions:
                    counts["legal_provisions_changed"] += 1
                updates.append((domain, provisions_json, str(row["document_id"])))
            counts["documents_changed"] += len(updates)
            if apply and updates:
                connection.executemany(
                    "UPDATE documents SET tax_domain = ?, legal_provisions_json = ? WHERE document_id = ?",
                    updates,
                )
                # Keep only the cheap domain column in SQLite FTS in sync.
                # Rewriting every metadata column would rebuild hundreds of
                # thousands of FTS rows and is unnecessary for the direct
                # document-metadata channels used by production.
                connection.executemany(
                    """
                    UPDATE chunks_fts
                    SET tax_domain = ?
                    WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?)
                    """,
                    domain_updates,
                )
                connection.commit()
    finally:
        connection.close()
    return dict(counts)


def update_mysql(*, apply: bool, batch_size: int) -> dict[str, int]:
    from app.legacy_july7.mysql_rag import get_mysql_target, mysql_connection

    documents_table, chunks_table = get_mysql_target()
    counts: Counter[str] = Counter()
    # PyMySQL's default cursor buffers the result set.  Keep reading and
    # writing on distinct connections nevertheless: executing an UPDATE on
    # the reader would discard its unread result rows after the first batch.
    with mysql_connection() as read_connection, mysql_connection() as write_connection:
        with read_connection.cursor() as read_cursor:
            read_cursor.execute(
                f"SELECT {SELECT_FIELDS} FROM `{documents_table}` WHERE source_type = %s ORDER BY document_id",
                ("interpretation",),
            )
            while True:
                rows = read_cursor.fetchmany(batch_size)
                if not rows:
                    break
                updates: list[tuple[str, str, str]] = []
                for row in rows:
                    counts["scanned"] += 1
                    result = changed_metadata(row)
                    if result is None:
                        continue
                    domain, provisions_json = result
                    previous_domain = str(row.get("tax_domain") or "").strip().upper()
                    previous_provisions = str(row.get("legal_provisions_json") or "[]")
                    if domain != previous_domain:
                        counts["tax_domain_changed"] += 1
                    if provisions_json != previous_provisions:
                        counts["legal_provisions_changed"] += 1
                    updates.append((domain, provisions_json, str(row["document_id"])))
                counts["documents_changed"] += len(updates)
                if apply and updates:
                    apply_mysql_metadata_batch(
                        write_connection,
                        documents_table=documents_table,
                        updates=updates,
                    )

        if apply:
            # Only the previously blank domain field is mirrored into chunks;
            # source text and existing full-text fields remain untouched.
            with write_connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE `{chunks_table}` c
                    JOIN `{documents_table}` d ON d.document_id = c.document_id
                    SET c.tax_domain = d.tax_domain
                    WHERE d.source_type = %s
                      AND d.tax_domain <> ''
                      AND c.tax_domain <> d.tax_domain
                    """,
                    ("interpretation",),
                )
                counts["chunk_tax_domain_changed"] = cursor.rowcount
            write_connection.commit()
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill canonical Eureka interpretation metadata from the local corpus.")
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--mysql", action="store_true", help="Apply the same idempotent metadata update to configured MariaDB.")
    parser.add_argument("--env-file", type=Path, default=API_DIR / ".env")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--apply", action="store_true", help="Persist changes; without it the command is a read-only preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not args.sqlite_path.exists():
        raise SystemExit(f"SQLite corpus not found: {args.sqlite_path}")
    report: dict[str, object] = {
        "apply": args.apply,
        "sqlite": update_sqlite(args.sqlite_path, apply=args.apply, batch_size=args.batch_size),
    }
    if args.mysql:
        load_dotenv(args.env_file)
        report["mysql"] = update_mysql(apply=args.apply, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
