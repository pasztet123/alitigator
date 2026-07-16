"""Atomically rebuild the current Polish primary-law corpus in MariaDB.

The source JSONL files are committed, official ELI-derived consolidated texts.
Dry-run is the default.  ``--apply`` replaces only documents whose ``act_title``
matches one of the validated local acts; treaties, interpretations, judgments
and every unrelated document remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_PATHS = (
    "apps/api/data/laws/processed/excise_act_DU_2026_412.jsonl",
    "apps/api/data/laws/processed/vat_act_DU_2025_775_codified_2026-05-05.jsonl",
    "apps/api/data/laws/processed/cit_act_DU_2026_554.jsonl",
    "apps/api/data/laws/processed/pit_act_DU_2026_592.jsonl",
    "apps/api/data/laws/processed/pcc_act_DU_2026_191.jsonl",
    "apps/api/data/laws/processed/inheritance_gift_tax_act_DU_2026_478.jsonl",
    "apps/api/data/laws/processed/tax_ordinance_DU_2026_622.jsonl",
    "apps/api/data/laws/processed/local_taxes_act_DU_2025_707.jsonl",
    "apps/api/data/laws/processed/family_foundation_primary_bundle.jsonl",
)

REQUIRED_REFERENCES = {
    "CIT": {"art. 11n pkt 1", "art. 15 ust. 2", "art. 24q ust. 1", "art. 24r ust. 1"},
    "PIT": {"art. 20 ust. 1g", "art. 21 ust. 1 pkt 157", "art. 30 ust. 1 pkt 17"},
    "VAT": {"art. 15 ust. 1", "art. 29a ust. 1", "art. 32 ust. 2", "art. 86 ust. 1"},
    "UFR": {"art. 2 ust. 2", "art. 5 ust. 1", "art. 27 ust. 4", "art. 29 ust. 1"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="apps/api/.env")
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

    records: list[dict] = []
    act_titles: set[str] = set()
    seen_document_ids: set[str] = set()
    for raw_path in DEFAULT_PATHS:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"Current primary-law source is missing: {path}")
        file_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        file_titles = {str(record.get("act_title") or "").strip() for record in file_records}
        if not file_records or len(file_titles) != 1 or "" in file_titles:
            raise SystemExit(f"Refusing ambiguous primary-law source: {path}")
        for record in file_records:
            document_id = str(record.get("document_id") or "")
            if (
                record.get("source_type") != "statute"
                or record.get("source_subtype") == "tax_treaty"
                or not document_id.startswith("pl-")
                or document_id in seen_document_ids
            ):
                raise SystemExit(f"Refusing invalid or duplicate primary-law record: {document_id}")
            seen_document_ids.add(document_id)
        act_titles.update(file_titles)
        records.extend(file_records)

    config = get_rag_config()
    prepared = []
    references_by_domain: dict[str, set[str]] = {}
    for record in records:
        document_row, chunk_rows = build_mysql_chunk_rows(record, config=config)
        if not document_row or not chunk_rows:
            raise SystemExit(f"Primary-law record produced no chunks: {record['document_id']}")
        prepared.append((document_row, chunk_rows))
        domain = str(document_row.get("tax_domain") or "").upper()
        references_by_domain.setdefault(domain, set()).update(
            str(chunk.get("display_reference") or "")
            for chunk in chunk_rows
            if str(chunk.get("display_reference") or "")
        )

    missing_audit = {
        domain: sorted(required - references_by_domain.get(domain, set()))
        for domain, required in REQUIRED_REFERENCES.items()
        if required - references_by_domain.get(domain, set())
    }
    if missing_audit:
        raise SystemExit(f"Refusing incomplete primary-law sync: {missing_audit}")

    documents_table, _ = get_mysql_target()
    with mysql_connection() as connection:
        ensure_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT document_id FROM `{documents_table}` WHERE act_title IN ("
                + ",".join(["%s"] * len(act_titles))
                + ") ORDER BY document_id",
                tuple(sorted(act_titles)),
            )
            remote_ids = [str(row["document_id"]) for row in cursor.fetchall()]

        print(
            {
                "acts": len(act_titles),
                "local_documents": len(prepared),
                "local_chunks": sum(len(chunks) for _, chunks in prepared),
                "remote_documents_to_replace": len(remote_ids),
                "required_reference_audit": "passed",
                "apply": args.apply,
            },
            flush=True,
        )
        if not args.apply:
            return 0

        try:
            for document_id in remote_ids:
                delete_document_mysql(connection, document_id)
            for index, (document_row, chunk_rows) in enumerate(prepared, start=1):
                upsert_document_mysql(connection, document_row)
                insert_chunks_mysql(
                    connection,
                    chunk_rows,
                    declared_legal_provisions=json.loads(
                        document_row.get("legal_provisions_json") or "[]"
                    ),
                )
                if index % 250 == 0:
                    print(
                        {"progress_documents": index, "total_documents": len(prepared)},
                        flush=True,
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    print(
        {
            "replaced_remote_documents": len(remote_ids),
            "inserted_primary_documents": len(prepared),
            "inserted_primary_chunks": sum(len(chunks) for _, chunks in prepared),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
