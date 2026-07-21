"""Populate the versioned DocumentCard cache without blocking request traffic."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from app.legal_rag_v2.document_validation import build_document_card
from app.mysql_rag import get_mysql_target, mysql_connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--after", default="")
    args = parser.parse_args()
    documents_table, _ = get_mysql_target()
    limit_sql = "LIMIT %s" if args.limit else ""
    params: list[object] = [args.after]
    if args.limit: params.append(args.limit)
    query = f"SELECT document_id, source_type, subject, signature, tax_domain, legal_provisions_json, question_text, facts_text, decision_text FROM `{documents_table}` WHERE document_id > %s ORDER BY document_id {limit_sql}"
    completed = 0
    with mysql_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, params)
        for row in cursor.fetchall():
            candidate = SimpleNamespace(
                candidate_id=row["document_id"], document_id=row["document_id"], source_type=row["source_type"],
                text="\n".join(str(row[key] or "") for key in ("question_text", "facts_text", "decision_text")),
                metadata={"subject": row["subject"], "signature": row["signature"], "tax_domains": [row["tax_domain"]] if row["tax_domain"] else [], "legal_provisions": row["legal_provisions_json"]},
            )
            build_document_card(candidate)
            completed += 1
    print({"completed": completed})


if __name__ == "__main__":
    main()
