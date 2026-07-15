"""Atomically replace only UPO records in MariaDB from the local SQLite index.

Unlike the generic "missing rows" loader, this removes stale treaty chunks
whose content hash changed.  The scope is strictly ``source_subtype=tax_treaty``;
all other legal sources remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

from backfill_missing_sqlite_to_mysql import CHUNK_COLUMNS, DOCUMENT_COLUMNS, search_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument("--sqlite-path", default="apps/api/data/processed/eureka_rag.sqlite3")
    parser.add_argument(
        "--manifest-path",
        default="apps/api/data/laws/processed/tax_treaties_core_manifest.json",
        help="Completeness manifest produced with the local treaty corpus.",
    )
    parser.add_argument("--apply", action="store_true", help="Perform the scoped remote replacement.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise SystemExit("Treaty manifest is unavailable; rebuild the local UPO corpus before remote sync")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    incomplete = [
        f"{item.get('slug')}/{item.get('variant')}"
        for item in manifest
        if item.get("status") != "ready" or item.get("missing_numeric_articles")
    ]
    if incomplete:
        raise SystemExit(
            "Refusing remote treaty replacement because the local UPO corpus is incomplete: "
            + ", ".join(incomplete)
        )

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)
    from app.mysql_rag import get_mysql_target, mysql_connection

    local = sqlite3.connect(args.sqlite_path)
    local.row_factory = sqlite3.Row
    documents = local.execute(
        "SELECT * FROM documents WHERE source_subtype = 'tax_treaty' ORDER BY document_id"
    ).fetchall()
    chunks = local.execute(
        """
        SELECT c.*, d.* FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.source_subtype = 'tax_treaty'
        ORDER BY c.document_id, c.chunk_index
        """
    ).fetchall()
    citations = local.execute(
        """
        SELECT cc.chunk_id, cc.citation FROM chunk_citations cc
        JOIN chunks c ON c.chunk_id = cc.chunk_id
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.source_subtype = 'tax_treaty'
        ORDER BY cc.chunk_id, cc.citation
        """
    ).fetchall()

    documents_table, chunks_table = get_mysql_target()
    citations_table = f"{chunks_table}_citations"
    with mysql_connection() as remote:
        with remote.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS count FROM `{documents_table}` WHERE source_subtype = 'tax_treaty'"
            )
            remote_documents = int(cursor.fetchone()["count"])
        summary = {
            "local_documents": len(documents),
            "local_chunks": len(chunks),
            "local_citations": len(citations),
            "remote_treaty_documents_to_replace": remote_documents,
            "apply": args.apply,
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
                cursor.execute(
                    f"DELETE cc FROM `{citations_table}` cc "
                    f"JOIN `{chunks_table}` c ON c.chunk_id = cc.chunk_id "
                    f"JOIN `{documents_table}` d ON d.document_id = c.document_id "
                    "WHERE d.source_subtype = 'tax_treaty'"
                )
                cursor.execute(
                    f"DELETE c FROM `{chunks_table}` c "
                    f"JOIN `{documents_table}` d ON d.document_id = c.document_id "
                    "WHERE d.source_subtype = 'tax_treaty'"
                )
                cursor.execute(f"DELETE FROM `{documents_table}` WHERE source_subtype = 'tax_treaty'")
                cursor.executemany(
                    document_sql,
                    [tuple(row[column] for column in DOCUMENT_COLUMNS) for row in documents],
                )
                cursor.executemany(
                    chunk_sql,
                    [
                        (
                            row["chunk_id"], row["document_id"], row["chunk_index"], row["chunk_text"],
                            row["chunk_chars"], row["provision_id"], row["display_reference"], search_text(row),
                            row["question_text"], row["facts_text"], row["tax_domain"],
                        )
                        for row in chunks
                    ],
                )
                cursor.executemany(
                    f"INSERT INTO `{citations_table}` (chunk_id, citation) VALUES (%s, %s)",
                    [(row["chunk_id"], row["citation"]) for row in citations],
                )
            remote.commit()
        except Exception:
            remote.rollback()
            raise

    print({"replaced_treaty_documents": len(documents), "inserted_treaty_chunks": len(chunks)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
