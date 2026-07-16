"""Atomically replace the legacy family-foundation summaries with exact UFR law.

The input is the provision-level ELI consolidated-text JSONL committed with
the application.  Dry-run is the default; ``--apply`` performs only the
scoped replacement and leaves every other RAG document untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


ACT_TITLE = "Ustawa z dnia 26 stycznia 2023 r. o fundacji rodzinnej"
DOCUMENT_PREFIX = "pl-ustawa-o-fundacji-rodzinnej-2023-326-art.-"
LEGACY_DOCUMENT_IDS = (
    "family-foundation-primary-ufr-art-5-27-29",
    "family-foundation-primary-cit-24q-24r",
    "family-foundation-primary-pit-beneficiary-rates",
    "family-foundation-primary-vat-related-party-transactions",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument(
        "--jsonl-path",
        default="apps/api/data/laws/processed/family_foundation_primary_bundle.jsonl",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)

    from app.mysql_rag import (
        build_mysql_chunk_rows,
        delete_document_mysql,
        ensure_schema,
        get_mysql_target,
        insert_chunks_mysql,
        mysql_connection,
        upsert_document_mysql,
    )
    from app.rag import get_rag_config

    path = Path(args.jsonl_path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("UFR JSONL is empty")
    invalid = [
        str(record.get("document_id") or "")
        for record in records
        if not str(record.get("document_id") or "").startswith(DOCUMENT_PREFIX)
        or record.get("act_title") != ACT_TITLE
        or record.get("source_type") != "statute"
        or "UFR" not in set(record.get("law_tags") or [])
    ]
    if invalid:
        raise SystemExit(f"Refusing unscoped UFR sync; invalid records: {invalid[:5]}")

    config = get_rag_config()
    prepared = []
    for record in records:
        document_row, chunk_rows = build_mysql_chunk_rows(record, config=config)
        if not document_row or not chunk_rows:
            raise SystemExit(f"UFR record produced no index chunks: {record['document_id']}")
        prepared.append((document_row, chunk_rows))

    documents_table, _ = get_mysql_target()
    with mysql_connection() as connection:
        ensure_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT document_id FROM `{documents_table}` "
                "WHERE act_title = %s OR document_id IN ("
                + ",".join(["%s"] * len(LEGACY_DOCUMENT_IDS))
                + ") ORDER BY document_id",
                (ACT_TITLE, *LEGACY_DOCUMENT_IDS),
            )
            remote_ids = [str(row["document_id"]) for row in cursor.fetchall()]

        summary = {
            "local_documents": len(prepared),
            "local_chunks": sum(len(chunks) for _, chunks in prepared),
            "remote_documents_to_replace": len(remote_ids),
            "legacy_documents_present": sum(
                document_id in LEGACY_DOCUMENT_IDS for document_id in remote_ids
            ),
            "apply": args.apply,
        }
        print(summary, flush=True)
        if not args.apply:
            return 0

        try:
            for document_id in remote_ids:
                delete_document_mysql(connection, document_id)
            for document_row, chunk_rows in prepared:
                upsert_document_mysql(connection, document_row)
                insert_chunks_mysql(
                    connection,
                    chunk_rows,
                    declared_legal_provisions=json.loads(
                        document_row.get("legal_provisions_json") or "[]"
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    print(
        {
            "replaced_remote_documents": len(remote_ids),
            "inserted_ufr_documents": len(prepared),
            "inserted_ufr_chunks": sum(len(chunks) for _, chunks in prepared),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
