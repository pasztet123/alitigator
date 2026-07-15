"""Rebuild every local UPO record and refresh its SQLite retrieval index.

The operation is idempotent. It regenerates the canonical treaty JSONL from
the official PDFs/cached OCR, removes stale treaty records from SQLite, and
then indexes exactly the regenerated article units. It never touches
non-treaty documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _article_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_treaty: dict[str, set[str]] = defaultdict(set)
    for record in records:
        document_id = str(record.get("document_id") or "")
        treaty_id = document_id.split("-art.-", 1)[0]
        provisions = record.get("legal_provisions") or []
        if treaty_id and provisions:
            by_treaty[treaty_id].add(str(provisions[0]).removeprefix("art. "))

    result: list[dict[str, object]] = []
    for treaty_id, articles in sorted(by_treaty.items()):
        numeric = sorted(int(value) for value in articles if value.isdigit())
        amending_protocol = "-protokol_" in treaty_id
        result.append(
            {
                "treaty_id": treaty_id,
                "article_count": len(articles),
                "numeric_article_count": len(numeric),
                "coverage_type": "amending_protocol" if amending_protocol else "full_text",
                "missing_numeric_articles": ([] if amending_protocol else (
                    [number for number in range(1, max(numeric) + 1) if str(number) not in articles]
                    if numeric else []
                )),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apps/api/data/laws/processed/tax_treaties_core.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("apps/api/data/laws/processed/tax_treaties_core_manifest.json"),
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("apps/api/data/processed/eureka_rag.sqlite3"),
    )
    parser.add_argument(
        "--from-existing-jsonl",
        action="store_true",
        help=(
            "Rebuild only SQLite from the committed treaty JSONL instead of "
            "rerunning PDF/OCR extraction. Useful for a production sync host."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path("apps/api").resolve()))
    from app.rag import (
        RagConfig,
        delete_document,
        get_connection,
        get_rag_config,
        index_record,
    )
    if args.from_existing_jsonl:
        if not args.output.exists() or not args.manifest.exists():
            raise SystemExit("Committed treaty JSONL or manifest is unavailable")
        records = [
            json.loads(line)
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        from app.treaty_chunk import CORE_TREATY_SOURCES, build_outputs

        records, manifest = build_outputs(list(CORE_TREATY_SOURCES))
    audit = _article_audit(records)
    print(
        json.dumps(
            {
                "records": len(records),
                "sources_ready": sum(1 for item in manifest if item.get("status") == "ready"),
                "sources_partial": sum(1 for item in manifest if item.get("status") == "partial_text_only"),
                "sources_total": len(manifest),
                "audit": audit,
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return 0

    if not args.from_existing_jsonl:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    configured = get_rag_config()
    config = RagConfig(**{**configured.__dict__, "db_path": args.sqlite_path})
    connection = get_connection(args.sqlite_path)
    try:
        stale_ids = [
            str(row["document_id"])
            for row in connection.execute(
                "SELECT document_id FROM documents WHERE source_subtype = 'tax_treaty'"
            ).fetchall()
        ]
        for document_id in stale_ids:
            delete_document(connection, document_id)
        indexed_chunks = sum(index_record(connection, record, config) for record in records)
        # ``index_record`` maintains the FTS row for each inserted chunk and
        # ``delete_document`` removes the old one.  Rebuilding the FTS table
        # here would rewrite the entire (multi-GB) corpus for a treaty-only
        # change, extending the write lock without improving correctness.
        connection.commit()
    finally:
        connection.close()
    print(json.dumps({"removed_stale_treaty_documents": len(stale_ids), "indexed_chunks": indexed_chunks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
