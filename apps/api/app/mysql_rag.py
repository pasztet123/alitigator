from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import pymysql
from pymysql.cursors import DictCursor

from app.rag import (
    FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS,
    JUDGMENT_INTENT_RE,
    JUDGMENT_ONLY_CONTEXT_RE,
    KSEF_CURRENT_BUNDLE_DOCUMENT_IDS,
    KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS,
    KSEF_FOREIGN_SALE_STATUTE_TARGETS,
    QUERY_TOKEN_RE,
    RagChunk,
    RagDocumentContext,
    RetrievalInspection,
    build_document_context_from_rows,
    build_article_family_match_score,
    build_chunk_payload,
    build_context_block,
    build_match_query,
    build_provision_id,
    annotate_chunk_evidence_role,
    build_ksef_current_law_statute_targets,
    chunk_canonical_source_id,
    build_estonian_cit_hidden_profit_statute_targets,
    build_poland_germany_treaty_statute_targets,
    build_shareholder_company_asset_sale_statute_targets,
    build_wht_pay_and_refund_service_statute_targets,
    build_structured_profile,
    build_subject_phrase_match_score,
    compute_hash_semantic_scores,
    detect_domains,
    detect_procedural_article_targets,
    derive_source_subtype,
    detect_mechanisms,
    diversify_top_document_window,
    expand_search_query,
    extract_judgment_signatures,
    extract_normalized_provision_references,
    extract_primary_article_key,
    extract_statute_target_from_text,
    filter_index_chunks,
    get_query_expansion_terms,
    get_rag_config,
    json_dump,
    index_content_fingerprint,
    join_search_text,
    normalize_source_type,
    normalize_provision_reference,
    query_targets_interpretation_procedure,
    query_targets_ksef_current_law,
    query_targets_ksef_foreign_sale,
    query_targets_estonian_cit_hidden_profit,
    query_targets_family_foundation_mechanism,
    query_targets_poland_germany_treaty,
    query_targets_shareholder_company_asset_sale,
    query_targets_small_taxpayer_foreign_vat,
    query_targets_wht_pay_and_refund_services,
    decompose_query_into_legal_axes,
    order_chunks_by_statute_targets,
    rerank_chunks_within_documents,
    rank_hybrid_local_candidates,
    ranking_terms,
    resolve_statute_tax_domains,
    row_to_rag_chunk,
    select_diverse_chunks,
    split_into_chunks,
    utc_now_iso,
    build_legal_match_score,
    build_mechanism_match_score,
    build_pcc_interpretation_match_score,
    build_ksef_foreign_sale_match_score,
    build_shareholder_company_asset_sale_match_score,
    build_small_taxpayer_foreign_vat_match_score,
    build_local_hybrid_score,
    build_judgment_metadata_match_score,
    build_statute_match_score,
    build_interpretation_section_match_score,
    compute_cross_encoder_scores,
    _merge_axis_search_chunks,
    resolve_cross_blend_weight,
    LegalRetrievalAxis,
    LegalSourcePlan,
    build_legal_source_plan,
    dedupe_chunks_by_canonical_source,
    legal_source_plan_primary_satisfied,
    required_primary_document_ids_for_query,
)


_MYSQL_SEARCH_SCHEMA_READY = False


def is_mysql_rag_enabled() -> bool:
    return os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}


def is_mysql_rag_configured() -> bool:
    return all(
        os.getenv(name)
        for name in (
            "ALITIGATOR_RAG_MYSQL_HOST",
            "ALITIGATOR_RAG_MYSQL_DATABASE",
            "ALITIGATOR_RAG_MYSQL_USER",
            "ALITIGATOR_RAG_MYSQL_PASSWORD",
        )
    )


def get_mysql_target() -> tuple[str, str]:
    return (
        os.getenv("ALITIGATOR_RAG_MYSQL_DOCUMENTS_TABLE", "rag_documents"),
        os.getenv("ALITIGATOR_RAG_MYSQL_CHUNKS_TABLE", "rag_chunks"),
    )


def get_mysql_connection_kwargs() -> dict[str, Any]:
    ssl_disabled = os.getenv("ALITIGATOR_RAG_MYSQL_SSL_DISABLED", "false").lower() in {"1", "true", "yes"}
    kwargs: dict[str, Any] = {
        "host": os.getenv("ALITIGATOR_RAG_MYSQL_HOST", "").strip(),
        "port": int(os.getenv("ALITIGATOR_RAG_MYSQL_PORT", "3306")),
        "user": os.getenv("ALITIGATOR_RAG_MYSQL_USER", "").strip(),
        "password": os.getenv("ALITIGATOR_RAG_MYSQL_PASSWORD", ""),
        "database": os.getenv("ALITIGATOR_RAG_MYSQL_DATABASE", "").strip(),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
        "connect_timeout": int(os.getenv("ALITIGATOR_RAG_MYSQL_CONNECT_TIMEOUT_SECONDS", "10")),
        "read_timeout": int(os.getenv("ALITIGATOR_RAG_MYSQL_READ_TIMEOUT_SECONDS", "90")),
        "write_timeout": int(os.getenv("ALITIGATOR_RAG_MYSQL_WRITE_TIMEOUT_SECONDS", "30")),
    }
    if not ssl_disabled:
        kwargs["ssl"] = {}
    return kwargs


@contextmanager
def mysql_connection() -> Iterable[pymysql.connections.Connection]:
    connection = pymysql.connect(**get_mysql_connection_kwargs())
    try:
        yield connection
    finally:
        connection.close()


def ensure_schema(connection: pymysql.connections.Connection) -> None:
    documents_table, chunks_table = get_mysql_target()
    citations_table = f"{chunks_table}_citations"
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{documents_table}` (
                document_id varchar(191) PRIMARY KEY,
                content_sha256 varchar(64) NULL,
                source varchar(64) NOT NULL DEFAULT 'eureka',
                source_type varchar(32) NOT NULL DEFAULT 'interpretation',
                source_subtype varchar(32) NOT NULL DEFAULT '',
                authority varchar(191) NOT NULL DEFAULT '',
                jurisdiction varchar(16) NOT NULL DEFAULT 'PL',
                act_title varchar(512) NOT NULL DEFAULT '',
                publication varchar(255) NOT NULL DEFAULT '',
                legal_state_date varchar(32) NOT NULL DEFAULT '',
                source_pages_json LONGTEXT NOT NULL,
                subject TEXT NOT NULL,
                signature varchar(255) NULL,
                published_date varchar(64) NULL,
                source_url TEXT NULL,
                category varchar(255) NULL,
                keywords_json LONGTEXT NOT NULL,
                legal_provisions_json LONGTEXT NOT NULL,
                issues_json LONGTEXT NOT NULL,
                law_tags_json LONGTEXT NOT NULL,
                tax_domain varchar(64) NOT NULL DEFAULT '',
                signature_family varchar(64) NOT NULL DEFAULT '',
                question_text MEDIUMTEXT NOT NULL,
                facts_text MEDIUMTEXT NOT NULL,
                decision_text MEDIUMTEXT NOT NULL,
                indexed_at varchar(64) NOT NULL,
                KEY idx_source_type (source_type),
                KEY idx_tax_domain (tax_domain),
                KEY idx_published_date (published_date),
                KEY idx_content_sha256 (content_sha256)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{chunks_table}` (
                chunk_id varchar(191) PRIMARY KEY,
                document_id varchar(191) NOT NULL,
                chunk_index int NOT NULL,
                chunk_text MEDIUMTEXT NOT NULL,
                chunk_chars int NOT NULL,
                provision_id varchar(255) NOT NULL DEFAULT '',
                display_reference varchar(191) NOT NULL DEFAULT '',
                search_text MEDIUMTEXT NOT NULL,
                question_text MEDIUMTEXT NOT NULL,
                facts_text MEDIUMTEXT NOT NULL,
                tax_domain varchar(64) NOT NULL DEFAULT '',
                FULLTEXT KEY ft_search (search_text, question_text, facts_text, tax_domain),
                KEY idx_document_id (document_id),
                KEY idx_document_chunk (document_id, chunk_index),
                KEY idx_display_reference (display_reference),
                CONSTRAINT fk_rag_chunks_document FOREIGN KEY (document_id)
                    REFERENCES `{documents_table}`(document_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        for column in (
            "provision_id varchar(255) NOT NULL DEFAULT ''",
            "display_reference varchar(191) NOT NULL DEFAULT ''",
        ):
            cursor.execute(
                f"ALTER TABLE `{chunks_table}` ADD COLUMN IF NOT EXISTS {column}"
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_display_reference ON `{chunks_table}` (display_reference)"
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{citations_table}` (
                chunk_id varchar(191) NOT NULL,
                citation varchar(191) NOT NULL,
                PRIMARY KEY (chunk_id, citation),
                KEY idx_citation (citation, chunk_id),
                CONSTRAINT fk_rag_chunk_citations_chunk FOREIGN KEY (chunk_id)
                    REFERENCES `{chunks_table}`(chunk_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    ensure_chunk_fulltext_index(connection)
    connection.commit()


def ensure_chunk_fulltext_index(connection: pymysql.connections.Connection) -> None:
    _, chunks_table = get_mysql_target()
    expected_columns = ("search_text", "question_text", "facts_text", "tax_domain")
    fulltext_indexes: dict[str, list[tuple[int, str]]] = {}

    with connection.cursor() as cursor:
        cursor.execute(f"SHOW INDEX FROM `{chunks_table}`")
        for row in cursor.fetchall():
            if str(row.get("Index_type") or "").upper() != "FULLTEXT":
                continue
            key_name = str(row.get("Key_name") or "")
            sequence = int(row.get("Seq_in_index") or 0)
            column_name = str(row.get("Column_name") or "")
            fulltext_indexes.setdefault(key_name, []).append((sequence, column_name))

        for columns in fulltext_indexes.values():
            ordered_columns = tuple(column for _, column in sorted(columns))
            if ordered_columns == expected_columns:
                return

        if "ft_search" in fulltext_indexes:
            cursor.execute(f"ALTER TABLE `{chunks_table}` DROP INDEX `ft_search`")
        cursor.execute(
            f"""
            ALTER TABLE `{chunks_table}`
            ADD FULLTEXT KEY `ft_search` (`search_text`, `question_text`, `facts_text`, `tax_domain`)
            """
        )
    connection.commit()


def ensure_search_schema_ready() -> None:
    global _MYSQL_SEARCH_SCHEMA_READY
    if _MYSQL_SEARCH_SCHEMA_READY:
        return

    with mysql_connection() as connection:
        ensure_schema(connection)
    _MYSQL_SEARCH_SCHEMA_READY = True


def mysql_backend_label() -> str:
    host = os.getenv("ALITIGATOR_RAG_MYSQL_HOST", "").strip()
    database = os.getenv("ALITIGATOR_RAG_MYSQL_DATABASE", "").strip()
    return f"mysql://{host}/{database}" if host and database else "mysql"


def index_exists_mysql() -> bool:
    if not is_mysql_rag_configured():
        return False
    documents_table, chunks_table = get_mysql_target()
    try:
        ensure_search_schema_ready()
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW TABLES LIKE %s", (documents_table,))
                documents_exists = cursor.fetchone() is not None
                cursor.execute(f"SHOW TABLES LIKE %s", (chunks_table,))
                chunks_exists = cursor.fetchone() is not None
                return documents_exists and chunks_exists
    except Exception:
        return False


def local_record_to_mysql_document(record: dict[str, Any]) -> dict[str, Any]:
    subject = (str(record.get("subject") or "Bez tytułu")).strip() or "Bez tytułu"
    signature = (str(record.get("signature") or "")).strip() or None
    keywords = [str(value).strip() for value in record.get("keywords") or [] if str(value).strip()]
    legal_provisions = [str(value).strip() for value in record.get("legal_provisions") or [] if str(value).strip()]
    issues = [str(value).strip() for value in record.get("issues") or [] if str(value).strip()]
    law_tags = [str(value).strip() for value in record.get("law_tags") or [] if str(value).strip()]
    profile = build_structured_profile(record)
    source_pages = [int(page) for page in record.get("source_pages") or [] if str(page).isdigit()]
    return {
        "document_id": str(record.get("document_id") or "").strip(),
        "content_sha256": index_content_fingerprint(record),
        "source": (str(record.get("source") or "eureka")).strip() or "eureka",
        "source_type": normalize_source_type(record),
        "source_subtype": derive_source_subtype(record),
        "authority": str(record.get("authority") or "").strip(),
        "jurisdiction": (str(record.get("jurisdiction") or "PL")).strip() or "PL",
        "act_title": str(record.get("act_title") or "").strip(),
        "publication": str(record.get("publication") or "").strip(),
        "legal_state_date": str(record.get("legal_state_date") or "").strip(),
        "source_pages_json": json_dump(source_pages),
        "subject": subject,
        "signature": signature,
        "published_date": record.get("published_date"),
        "source_url": record.get("source_url"),
        "category": record.get("category"),
        "keywords_json": json_dump(keywords),
        "legal_provisions_json": json_dump(legal_provisions),
        "issues_json": json_dump(issues),
        "law_tags_json": json_dump(law_tags),
        "tax_domain": profile["tax_domain"],
        "signature_family": profile["signature_family"],
        "question_text": profile["question_text"],
        "facts_text": profile["facts_text"],
        "decision_text": profile["decision_text"],
        "indexed_at": utc_now_iso(),
    }


def build_mysql_chunk_rows(
    record: dict[str, Any],
    *,
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from app.rag import build_record_index_chunks, clean_document_text

    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return {}, []

    document_text = clean_document_text(record)
    if not document_text:
        return {}, []

    chunks = build_record_index_chunks(
        record,
        document_text,
        target_chars=config.chunk_target_chars,
        overlap_chars=config.chunk_overlap_chars,
    )
    if not chunks:
        return {}, []

    document_row = local_record_to_mysql_document(record)
    search_keywords = json.loads(document_row["keywords_json"])
    search_legal_provisions = json.loads(document_row["legal_provisions_json"])
    search_issues = json.loads(document_row["issues_json"])
    search_law_tags = json.loads(document_row["law_tags_json"])
    chunk_rows: list[dict[str, Any]] = []
    for chunk_index, chunk_text in enumerate(chunks):
        first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "")
        display_reference = (
            normalize_provision_reference(first_line)
            if re.fullmatch(
                r"art\.\s*\d+[a-z]?(?:\s+(?:ust\.\s*\d+[a-z]?|§\s*\d+[a-z]?))?"
                r"(?:\s+pkt\s*\d+[a-z]?)?(?:\s+lit\.\s*[a-z])?",
                first_line,
                re.IGNORECASE,
            )
            else ""
        )
        article_document_id = str(record.get("article_document_id") or "").strip() or re.sub(
            r"-part-\d+(?:-occurrence-\d+)?$", "", document_id
        )
        provision_id = (
            build_provision_id(article_document_id, display_reference)
            if display_reference
            else ""
        )
        chunk_payload = build_chunk_payload(
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            subject=document_row["subject"],
            signature=document_row["signature"],
            published_date=document_row["published_date"],
            source_url=document_row["source_url"],
            category=document_row["category"],
            keywords=search_keywords,
            legal_provisions=search_legal_provisions,
            issues=search_issues,
            law_tags=search_law_tags,
            embedding_dimensions=config.embedding_dimensions,
        )
        search_text = "\n".join(
            value
            for value in (
                document_row["subject"],
                document_row["signature"] or "",
                document_row["category"] or "",
                join_search_text(search_keywords),
                join_search_text(search_legal_provisions),
                join_search_text(search_issues),
                join_search_text(search_law_tags),
                chunk_text,
            )
            if value
        )
        chunk_rows.append(
            {
                "chunk_id": chunk_payload["chunk_id"],
                "document_id": document_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "chunk_chars": len(chunk_text),
                "provision_id": provision_id,
                "display_reference": display_reference,
                "search_text": search_text,
                "question_text": document_row["question_text"],
                "facts_text": document_row["facts_text"],
                "tax_domain": document_row["tax_domain"],
            }
        )
    return document_row, chunk_rows


def fetch_document_state_mysql(connection: pymysql.connections.Connection, document_id: str) -> Optional[str]:
    documents_table, _ = get_mysql_target()
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT content_sha256 FROM `{documents_table}` WHERE document_id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
    return None if row is None else str(row.get("content_sha256") or "")


def delete_document_mysql(connection: pymysql.connections.Connection, document_id: str) -> None:
    documents_table, chunks_table = get_mysql_target()
    with connection.cursor() as cursor:
        cursor.execute(f"DELETE FROM `{chunks_table}` WHERE document_id = %s", (document_id,))
        cursor.execute(f"DELETE FROM `{documents_table}` WHERE document_id = %s", (document_id,))


def upsert_document_mysql(connection: pymysql.connections.Connection, row: dict[str, Any]) -> None:
    documents_table, _ = get_mysql_target()
    columns = list(row.keys())
    assignments = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in columns if column != "document_id")
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(f"`{column}`" for column in columns)
    values = tuple(row[column] for column in columns)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO `{documents_table}` ({column_sql})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {assignments}
            """,
            values,
        )


def insert_chunks_mysql(connection: pymysql.connections.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _, chunks_table = get_mysql_target()
    citations_table = f"{chunks_table}_citations"
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(f"`{column}`" for column in columns)
    values = [tuple(row[column] for column in columns) for row in rows]
    with connection.cursor() as cursor:
        cursor.executemany(
            f"""
            INSERT INTO `{chunks_table}` ({column_sql})
            VALUES ({placeholders})
            """,
            values,
        )
        citation_rows = [
            (row["chunk_id"], citation)
            for row in rows
            for citation in extract_normalized_provision_references(
                str(row.get("chunk_text") or "")
            )
        ]
        if citation_rows:
            cursor.executemany(
                f"INSERT IGNORE INTO `{citations_table}` (chunk_id, citation) VALUES (%s, %s)",
                citation_rows,
            )


def reindex_corpus_mysql(*, limit: Optional[int] = None, force: bool = False) -> dict[str, Any]:
    from app.rag import iter_processed_records

    config = get_rag_config()
    if not config.processed_path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {config.processed_path}")
    missing_additional_paths = [path for path in config.additional_source_paths if not path.exists()]
    if missing_additional_paths:
        raise FileNotFoundError(f"Additional RAG source not found: {missing_additional_paths[0]}")
    if not is_mysql_rag_configured():
        raise RuntimeError("MySQL RAG backend is enabled, but connection variables are incomplete")

    processed = 0
    indexed = 0
    skipped = 0
    chunk_count = 0
    indexed_document_ids: list[str] = []

    with mysql_connection() as connection:
        ensure_schema(connection)
        source_paths = (config.processed_path, *config.additional_source_paths)
        pending_writes = 0
        for source_path in source_paths:
            for record in iter_processed_records(source_path):
                if limit is not None and processed >= limit:
                    break
                processed += 1
                document_id = str(record.get("document_id") or "").strip()
                if not document_id:
                    skipped += 1
                    continue

                current_sha = index_content_fingerprint(record)
                stored_sha = fetch_document_state_mysql(connection, document_id)
                if not force and stored_sha and stored_sha == current_sha:
                    skipped += 1
                    continue

                document_row, chunk_rows = build_mysql_chunk_rows(record, config=config)
                if not document_row or not chunk_rows:
                    skipped += 1
                    continue

                delete_document_mysql(connection, document_id)
                upsert_document_mysql(connection, document_row)
                insert_chunks_mysql(connection, chunk_rows)

                indexed += 1
                chunk_count += len(chunk_rows)
                indexed_document_ids.append(document_id)
                pending_writes += 1
                if pending_writes >= 100:
                    connection.commit()
                    pending_writes = 0
            if limit is not None and processed >= limit:
                break
        connection.commit()

        documents_table, chunks_table = get_mysql_target()
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) AS count FROM `{documents_table}`")
            total_documents = int((cursor.fetchone() or {}).get("count") or 0)
            cursor.execute(f"SELECT COUNT(*) AS count FROM `{chunks_table}`")
            total_chunks = int((cursor.fetchone() or {}).get("count") or 0)

    return {
        "processed": processed,
        "indexed": indexed,
        "skipped": skipped,
        "chunk_count": chunk_count,
        "db_path": mysql_backend_label(),
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "indexed_document_ids": indexed_document_ids,
    }


def build_mysql_boolean_query(query: str, *, max_tokens: int = 24) -> Optional[str]:
    tokens: list[str] = []
    for match in QUERY_TOKEN_RE.finditer(query):
        token = match.group(0).lower()
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        return None
    return " ".join(f"{token}*" if len(token) >= 4 else token for token in tokens[:max_tokens])


def build_mysql_candidate_queries(query: str) -> list[str]:
    candidates = [
        build_mysql_boolean_query(query),
        build_mysql_boolean_query(" ".join(get_query_expansion_terms(query))),
    ]
    return list(dict.fromkeys(value for value in candidates if value))


def escape_pymysql_query_literals(sql: str) -> str:
    # PyMySQL uses percent-formatting for placeholders, so any literal "%" in
    # dynamically composed SQL must be escaped to avoid placeholder mismatches.
    return re.sub(r"%(?!s)", "%%", sql)


def build_type_and_domain_clause(
    *,
    source_types: Optional[set[str]],
    enforce_query_domain: bool,
    tax_domains: Optional[set[str]],
    detection_query: str,
    config: Any,
) -> tuple[str, list[Any], set[str]]:
    allowed_types = sorted({value.lower() for value in source_types or set() if value})
    clauses: list[str] = []
    values: list[Any] = []
    if allowed_types:
        clauses.append("d.source_type IN (" + ", ".join(["%s"] * len(allowed_types)) + ")")
        values.extend(allowed_types)

    query_domains = {domain.upper() for domain in detect_domains(detection_query)}
    query_domains.update(domain.upper() for domain in tax_domains or set() if domain)
    if (config.domain_filter_enabled or enforce_query_domain) and query_domains:
        sorted_domains = sorted(query_domains)
        domain_checks = ["UPPER(d.tax_domain) IN (" + ", ".join(["%s"] * len(sorted_domains)) + ")"]
        values.extend(sorted_domains)
        for domain in sorted_domains:
            domain_checks.append("d.legal_provisions_json LIKE %s")
            values.append(f"%[{domain}]%")
        clauses.append("(" + " OR ".join(domain_checks) + ")")
    return (" AND " + " AND ".join(clauses)) if clauses else "", values, query_domains


def select_candidate_rows_mysql(
    where_sql: str,
    params: list[Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    documents_table, chunks_table = get_mysql_target()
    sql = escape_pymysql_query_literals(
        f"""
        SELECT
            c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
            d.subject, d.signature, d.published_date, d.source_url, d.category,
            d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
            d.facts_text, d.question_text, d.tax_domain, d.source, d.source_type,
            d.source_subtype, d.authority, d.publication, d.legal_state_date,
            d.source_pages_json, c.chunk_chars, 0.0 AS lexical_score
        FROM `{chunks_table}` c
        JOIN `{documents_table}` d ON d.document_id = c.document_id
        WHERE {where_sql}
        ORDER BY d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
        LIMIT %s
        """
    )
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (*params, limit))
            return list(cursor.fetchall())


def fetch_rows_by_document_ids_mysql(
    document_ids: list[str] | tuple[str, ...],
    *,
    source_type: Optional[str] = None,
    chunk_limit_per_document: Optional[int] = None,
) -> list[dict[str, Any]]:
    clean_ids = [str(document_id).strip() for document_id in document_ids if str(document_id).strip()]
    if not clean_ids:
        return []
    documents_table, chunks_table = get_mysql_target()
    source_clause = " AND d.source_type = %s" if source_type else ""
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    d.subject, d.signature, d.published_date, d.source_url, d.category,
                    d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
                    d.facts_text, d.question_text, d.tax_domain, d.source, d.source_type,
                    d.source_subtype, d.authority, d.publication, d.legal_state_date,
                    d.source_pages_json, 0.0 AS lexical_score
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE c.document_id IN ({", ".join(["%s"] * len(clean_ids))}){source_clause}
                ORDER BY c.document_id ASC, c.chunk_index ASC
                """,
                (*clean_ids, *([source_type] if source_type else [])),
            )
            rows = list(cursor.fetchall())
    if chunk_limit_per_document is None:
        return rows
    limited_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        document_id = str(row["document_id"])
        if counts.get(document_id, 0) >= chunk_limit_per_document:
            continue
        limited_rows.append(row)
        counts[document_id] = counts.get(document_id, 0) + 1
    return limited_rows


def fetch_rows_by_subject_prefix_mysql(
    subject_prefix: str,
    *,
    source_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    prefix = str(subject_prefix).strip()
    if not prefix:
        return []
    documents_table, chunks_table = get_mysql_target()
    source_clause = " AND d.source_type = %s" if source_type else ""
    values: list[Any] = [f"{prefix}%"]
    if source_type:
        values.append(source_type)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    d.subject, d.signature, d.published_date, d.source_url, d.category,
                    d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
                    d.facts_text, d.question_text, d.tax_domain, d.source, d.source_type,
                    d.source_subtype, d.authority, d.publication, d.legal_state_date,
                    d.source_pages_json, 0.0 AS lexical_score
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE d.subject LIKE %s{source_clause}
                ORDER BY d.subject ASC, c.document_id ASC, c.chunk_index ASC
                """,
                tuple(values),
            )
            return list(cursor.fetchall())


def fetch_document_contexts_mysql(document_ids: list[str], *, seed_chunks: list[RagChunk]) -> list[RagDocumentContext]:
    clean_ids = [str(document_id).strip() for document_id in document_ids if str(document_id).strip()]
    if not clean_ids:
        return []
    documents_table, chunks_table = get_mysql_target()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    d.subject, d.signature, d.published_date, d.source_url, d.category,
                    d.legal_provisions_json, d.source, d.source_type, d.source_subtype,
                    d.authority, d.publication, d.legal_state_date, d.source_pages_json
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE c.document_id IN ({", ".join(["%s"] * len(clean_ids))})
                ORDER BY c.document_id ASC, c.chunk_index ASC
                """,
                tuple(clean_ids),
            )
            rows = list(cursor.fetchall())
    return build_document_context_from_rows(rows, ordered_document_ids=clean_ids, seed_chunks=seed_chunks)


def fetch_statute_rows_by_targets_mysql(
    targets: list[tuple[str, str]],
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    if not targets:
        return []
    clauses = ["(UPPER(d.tax_domain) = %s AND d.legal_provisions_json LIKE %s)" for _ in targets]
    values: list[Any] = []
    for domain, article_key in targets:
        values.extend((domain.upper(), f'%\"art. {article_key}\"%'))
    documents_table, chunks_table = get_mysql_target()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    d.subject, d.signature, d.published_date, d.source_url, d.category,
                    d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
                    d.facts_text, d.question_text, d.tax_domain, d.source, d.source_type,
                    d.source_subtype, d.authority, d.publication, d.legal_state_date,
                    d.source_pages_json, 0.0 AS lexical_score
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE d.source_type = 'statute'
                  AND c.chunk_index = 0
                  AND ({' OR '.join(clauses)})
                ORDER BY d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
                """,
                tuple(values),
            )
            rows = list(cursor.fetchall())

    order = {(domain.upper(), article_key): index for index, (domain, article_key) in enumerate(targets)}

    def row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
        article_key = extract_primary_article_key(row)
        domain = str(row["tax_domain"] or "").upper()
        return order.get((domain, article_key), len(order)), str(row["subject"] or "")

    deduped: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for row in sorted(rows, key=row_sort_key):
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        deduped.append(row)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def retrieve_deterministic_statute_chunks_mysql(
    query: str,
    *,
    plan: Optional[LegalSourcePlan] = None,
    limit: Optional[int] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    source_plan = plan or build_legal_source_plan(query)
    target_limit = max(limit or config.retrieval_limit, len(source_plan.statute_targets) or 1)

    rows: list[dict[str, Any]] = []
    if source_plan.statute_targets:
        rows.extend(fetch_statute_rows_by_targets_mysql(list(source_plan.statute_targets), limit=None))
    for axis in source_plan.axes:
        if not axis.direct_subject_prefix:
            continue
        rows.extend(fetch_rows_by_subject_prefix_mysql(axis.direct_subject_prefix, source_type="statute"))
    required_document_ids = required_primary_document_ids_for_query(query)
    if required_document_ids:
        rows.extend(
            fetch_rows_by_document_ids_mysql(
                required_document_ids,
                source_type="statute",
                chunk_limit_per_document=1,
            )
        )

    ranked_chunks = [
        row_to_rag_chunk(row, score=200.0, evidence_role="deterministic_primary_law")
        for row in rows
    ]
    ordered_chunks = order_chunks_by_statute_targets(ranked_chunks, list(source_plan.statute_targets))
    return [
        annotate_chunk_evidence_role(chunk, "deterministic_primary_law")
        for chunk in dedupe_chunks_by_canonical_source(ordered_chunks)
    ][:target_limit]


def fetch_candidate_rows_mysql(
    query: str,
    *,
    effective_limit: int,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
    detection_query: Optional[str] = None,
) -> tuple[str, list[dict[str, Any]]]:
    if not is_mysql_rag_configured():
        return "", []

    config = get_rag_config()
    detection_text = detection_query or query
    query_domains: set[str]
    filter_sql, filter_values, query_domains = build_type_and_domain_clause(
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
        detection_query=detection_text,
        config=config,
    )
    ensure_search_schema_ready()
    documents_table, chunks_table = get_mysql_target()
    candidate_limit = max(config.candidate_pool_limit, effective_limit * 20)
    query_rows: list[list[dict[str, Any]]] = []

    statute_exact_articles: set[str] = set()
    statute_family_prefixes: set[str] = set()
    if source_types == {"statute"}:
        statute_family_prefixes, statute_exact_articles = detect_procedural_article_targets(detection_text)

    if source_types == {"statute"} and (statute_exact_articles or statute_family_prefixes):
        clauses: list[str] = []
        values: list[Any] = []
        statute_filter_sql, statute_filter_values, _ = build_type_and_domain_clause(
            source_types=None,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
            detection_query=detection_text,
            config=config,
        )
        for article in sorted(statute_exact_articles):
            clauses.append("d.legal_provisions_json = %s")
            values.append(json_dump([f"art. {article}"]))
        for prefix in sorted(statute_family_prefixes):
            clauses.append("d.legal_provisions_json LIKE %s")
            values.append(f'%art. {prefix}%')
        if clauses:
            query_rows.append(
                select_candidate_rows_mysql(
                    "d.source_type = 'statute' AND c.chunk_index = 0 AND (" + " OR ".join(clauses) + ")" + statute_filter_sql,
                    [*values, *statute_filter_values],
                    limit=candidate_limit,
                )
            )

    match_queries = build_mysql_candidate_queries(query)
    if not match_queries and not query_rows:
        return "", []
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            for match_query in match_queries:
                cursor.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
                        d.facts_text, d.question_text, d.tax_domain, d.source, d.source_type,
                        d.source_subtype, d.authority, d.publication, d.legal_state_date,
                        d.source_pages_json,
                        MATCH(c.search_text, c.question_text, c.facts_text, c.tax_domain)
                            AGAINST (%s IN BOOLEAN MODE) AS lexical_score
                    FROM `{chunks_table}` c
                    JOIN `{documents_table}` d ON d.document_id = c.document_id
                    WHERE MATCH(c.search_text, c.question_text, c.facts_text, c.tax_domain)
                        AGAINST (%s IN BOOLEAN MODE)
                        {filter_sql}
                    ORDER BY lexical_score DESC, d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
                    LIMIT %s
                    """,
                    (match_query, match_query, *filter_values, candidate_limit),
                )
                query_rows.append(list(cursor.fetchall()))

    rows: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    chunks_per_document: dict[str, int] = {}
    max_chunks_per_document = max(config.retrieval_max_chunks_per_document, 1)
    for rank in range(max((len(group) for group in query_rows), default=0)):
        for group in query_rows:
            if rank >= len(group):
                continue
            row = group[rank]
            chunk_id = str(row["chunk_id"])
            document_id = str(row["document_id"])
            candidate_domains = {str(row["tax_domain"] or "").upper()} if row.get("tax_domain") else set()
            if (config.domain_filter_enabled or enforce_query_domain) and query_domains and candidate_domains and not (candidate_domains & query_domains):
                continue
            if chunk_id in seen_chunks or chunks_per_document.get(document_id, 0) >= max_chunks_per_document:
                continue
            rows.append(row)
            seen_chunks.add(chunk_id)
            chunks_per_document[document_id] = chunks_per_document.get(document_id, 0) + 1
            if len(rows) >= candidate_limit:
                return " || ".join(match_queries), rows
    return " || ".join(match_queries), rows


def _resolve_axis_scope_mysql(
    axis: LegalRetrievalAxis,
    *,
    source_types: Optional[set[str]],
    tax_domains: Optional[set[str]],
) -> Optional[tuple[Optional[set[str]], Optional[set[str]]]]:
    axis_source_types = set(axis.source_types) if axis.source_types else None
    if source_types is not None:
        axis_source_types = set(source_types) if axis_source_types is None else axis_source_types & set(source_types)
        if axis_source_types is not None and not axis_source_types:
            return None

    axis_tax_domains = set(axis.tax_domains) if axis.tax_domains else None
    if tax_domains is not None:
        axis_tax_domains = set(tax_domains) if axis_tax_domains is None else axis_tax_domains & set(tax_domains)
        if axis_tax_domains is not None and not axis_tax_domains:
            return None

    return axis_source_types, axis_tax_domains


def _search_chunks_single_query_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    _, rows = fetch_candidate_rows_mysql(
        expanded_query,
        effective_limit=effective_limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
        detection_query=query,
    )
    return rank_hybrid_local_candidates(rows, query=expanded_query, effective_limit=effective_limit, config=config)


def _search_chunks_by_legal_axes_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> tuple[list[RagChunk], list[LegalRetrievalAxis]]:
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    axes = decompose_query_into_legal_axes(query)
    if len(axes) <= 1:
        return [], axes

    scoped_axis_chunks: list[list[RagChunk]] = []
    active_axes: list[LegalRetrievalAxis] = []
    for axis in axes:
        axis_scope = _resolve_axis_scope_mysql(axis, source_types=source_types, tax_domains=tax_domains)
        if axis_scope is None:
            continue
        axis_source_types, axis_tax_domains = axis_scope
        active_axes.append(axis)
        axis_limit = max(1, math.ceil(effective_limit / max(len(axes), 1)))
        axis_chunks = _search_chunks_single_query_mysql(
                axis.query,
                limit=axis_limit,
                source_types=axis_source_types,
                enforce_query_domain=enforce_query_domain or bool(axis_tax_domains),
                tax_domains=axis_tax_domains,
            )
        if axis.direct_subject_prefix:
            direct_rows = fetch_rows_by_subject_prefix_mysql(
                axis.direct_subject_prefix,
                source_type="statute" if axis_source_types is None or "statute" in axis_source_types else None,
            )
            direct_chunks = (
                rank_hybrid_local_candidates(
                    direct_rows,
                    query=axis.query,
                    effective_limit=max(axis_limit, len(direct_rows)),
                    config=config,
                )
                if direct_rows
                else []
            )
            ordered_direct_chunks = order_chunks_by_statute_targets(direct_chunks, list(axis.preferred_targets))
            direct_limit = max(axis_limit, len(axis.preferred_targets) or axis_limit)
            axis_chunks = [*ordered_direct_chunks[:direct_limit], *axis_chunks]
        scoped_axis_chunks.append(axis_chunks)

    if not scoped_axis_chunks:
        return [], axes

    return _merge_axis_search_chunks(scoped_axis_chunks, effective_limit=effective_limit), active_axes


def inspect_search_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> RetrievalInspection:
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    match_query, candidate_rows = fetch_candidate_rows_mysql(
        expanded_query,
        effective_limit=effective_limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
        detection_query=query,
    )
    chunks = rank_hybrid_local_candidates(
        candidate_rows,
        query=expanded_query,
        effective_limit=effective_limit,
        config=config,
    )
    selected_chunks = select_diverse_chunks(chunks)
    selected_context_chars = sum(len(chunk.chunk_text.strip()) for chunk in selected_chunks)
    return RetrievalInspection(
        query=query,
        match_query=match_query or build_match_query(expanded_query),
        requested_limit=effective_limit,
        retrieved_count=len(chunks),
        selected_count=len(selected_chunks),
        selected_context_chars=selected_context_chars,
        hits=[
            {
                "rank": position,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "subject": chunk.subject,
                "signature": chunk.signature,
                "source_type": chunk.source_type,
                "source_subtype": chunk.source_subtype,
                "published_date": chunk.published_date,
                "source_url": chunk.source_url,
            }
            for position, chunk in enumerate(chunks, start=1)
        ],
        chunks=selected_chunks,
        raw_candidate_pool=[
            {
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "signature": str(row["signature"] or "") or None,
                "subject": str(row["subject"]),
                "source_type": str(row["source_type"]),
                "lexical_score": float(row.get("lexical_score") or 0.0),
            }
            for rank, row in enumerate(candidate_rows, start=1)
        ],
    )


def search_chunks_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    axis_chunks, axes = _search_chunks_by_legal_axes_mysql(
        query,
        limit=limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
    )
    if axis_chunks:
        return axis_chunks
    if axes:
        fallback_query = axes[0].query if len(axes) == 1 else query
        return _search_chunks_single_query_mysql(
            fallback_query,
            limit=limit,
            source_types=source_types,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
        )
    return _search_chunks_single_query_mysql(
        query,
        limit=limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
    )


def search_chat_chunks_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    judgment_requested_by_query = bool(JUDGMENT_INTENT_RE.search(query) or extract_judgment_signatures(query))
    include_judgments = judgment_requested_by_query if include_judgments is None else include_judgments
    source_plan = build_legal_source_plan(
        query,
        include_interpretations=include_interpretations,
        include_judgments=include_judgments,
    )
    deterministic_statutes = retrieve_deterministic_statute_chunks_mysql(
        query,
        plan=source_plan,
        limit=max(effective_limit, len(source_plan.statute_targets) or 1),
    )
    judgment_only_context = bool(JUDGMENT_ONLY_CONTEXT_RE.search(query))
    statute_domains = resolve_statute_tax_domains(query)
    explicit_query_domains = bool(statute_domains)
    if judgment_only_context:
        statute_limit = 0
        interpretation_limit = 0
        judgment_limit = effective_limit
    elif not include_interpretations and not include_judgments:
        statute_limit = effective_limit
        interpretation_limit = 0
        judgment_limit = 0
    elif include_judgments and not include_interpretations:
        statute_limit = max(1, effective_limit - 1)
        interpretation_limit = 0
        judgment_limit = max(1, effective_limit - statute_limit)
    elif include_judgments:
        statute_limit = max(1, effective_limit // 4) if statute_domains else 1
        interpretation_limit = min(max(2, effective_limit // 2), max(effective_limit - statute_limit, 1))
        judgment_limit = max(1, effective_limit - statute_limit - interpretation_limit)
    else:
        judgment_limit = 0
        statute_limit = effective_limit if not include_interpretations else max(1, effective_limit // 2)
        if query_targets_ksef_current_law(query):
            statute_limit = min(effective_limit - 1, max(6, math.ceil(effective_limit * 0.75)))
        interpretation_limit = max(1, effective_limit - statute_limit) if include_interpretations else 0

    if interpretation_limit and query_targets_ksef_foreign_sale(query):
        interpretation_rows = fetch_rows_by_document_ids_mysql(
            KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS,
            source_type="interpretation",
        )
        interpretations = (
            rank_hybrid_local_candidates(
                interpretation_rows,
                query=expanded_query,
                effective_limit=interpretation_limit,
                config=config,
            )
            if interpretation_rows
            else []
        )
    else:
        interpretations = (
            search_chunks_mysql(
                query,
                limit=interpretation_limit,
                source_types={"interpretation"},
                enforce_query_domain=explicit_query_domains,
                tax_domains=statute_domains,
            )
            if interpretation_limit
            else []
        )
    interpretations = rerank_chunks_within_documents(
        interpretations,
        query=expanded_query,
        config=config,
        source_type="interpretation",
        max_chunks_per_document=4,
    )
    judgments = (
        search_chunks_mysql(
            query,
            limit=judgment_limit,
            source_types={"judgment"},
            enforce_query_domain=explicit_query_domains,
            tax_domains=statute_domains,
        )
        if include_judgments
        else []
    )
    judgments = rerank_chunks_within_documents(
        judgments,
        query=expanded_query,
        config=config,
        source_type="judgment",
        max_chunks_per_document=4,
    )
    statutes = (
        search_chunks_mysql(
            query,
            limit=statute_limit,
            source_types={"statute"},
            enforce_query_domain=True,
            tax_domains=statute_domains,
        )
        if statute_limit and not query_targets_ksef_foreign_sale(query)
        else []
    )
    statutes = order_chunks_by_statute_targets(
        dedupe_chunks_by_canonical_source([*deterministic_statutes, *statutes]),
        list(source_plan.statute_targets),
    )
    direct_ksef_bundle_rows = fetch_rows_by_document_ids_mysql(
        KSEF_CURRENT_BUNDLE_DOCUMENT_IDS,
        source_type="statute",
        chunk_limit_per_document=1,
    ) if query_targets_ksef_current_law(query) and statute_limit else []
    direct_family_foundation_bundle_rows = fetch_rows_by_document_ids_mysql(
        FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS,
        source_type="statute",
        chunk_limit_per_document=1,
    ) if query_targets_family_foundation_mechanism(query) and statute_limit else []

    preferred_targets: list[tuple[str, str]] = []
    if query_targets_ksef_foreign_sale(query):
        preferred_targets.extend(KSEF_FOREIGN_SALE_STATUTE_TARGETS)
    if query_targets_ksef_current_law(query):
        preferred_targets.extend(build_ksef_current_law_statute_targets(query))
    if query_targets_wht_pay_and_refund_services(query):
        preferred_targets.extend(build_wht_pay_and_refund_service_statute_targets(query))
    if query_targets_poland_germany_treaty(query):
        preferred_targets.extend(build_poland_germany_treaty_statute_targets(query))
    if query_targets_estonian_cit_hidden_profit(query):
        preferred_targets.extend(build_estonian_cit_hidden_profit_statute_targets(query))
    if query_targets_shareholder_company_asset_sale(query):
        preferred_targets.extend(build_shareholder_company_asset_sale_statute_targets(query))
    if query_targets_small_taxpayer_foreign_vat(query):
        preferred_targets.extend([("CIT", "4a"), ("CIT", "19"), ("CIT", "12")])
    for chunk in interpretations:
        for provision in chunk.legal_provisions:
            target = extract_statute_target_from_text(provision)
            if target and (not statute_domains or target[0] in statute_domains) and target not in preferred_targets:
                preferred_targets.append(target)
    _, procedural_exact_articles = detect_procedural_article_targets(query)
    if statute_limit and procedural_exact_articles:
        hinted_domains = statute_domains or {"VAT", "CIT", "PIT", "PCC", "AKCYZA", "ORDYNACJA", "NIERUCHOMOŚCI"}
        for domain in sorted(hinted_domains):
            for article_key in sorted(procedural_exact_articles):
                target = (domain, article_key)
                if target not in preferred_targets:
                    preferred_targets.append(target)

    hinted_rows = fetch_statute_rows_by_targets_mysql(preferred_targets, limit=None if query_targets_ksef_current_law(query) else statute_limit) if statute_limit else []
    if direct_ksef_bundle_rows or direct_family_foundation_bundle_rows:
        hinted_rows = [*direct_ksef_bundle_rows, *direct_family_foundation_bundle_rows, *hinted_rows]
    hinted_statutes = (
        rank_hybrid_local_candidates(
            hinted_rows,
            query=expanded_query,
            effective_limit=statute_limit,
            config=config,
        )
        if hinted_rows
        else []
    )
    merged_statutes: list[RagChunk] = []
    seen_statute_sources: set[str] = set()
    for chunk in statutes:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_statute_sources:
            continue
        seen_statute_sources.add(canonical_source_id)
        merged_statutes.append(annotate_chunk_evidence_role(chunk, "governing_statute"))
        if len(merged_statutes) >= statute_limit:
            break
    bundle_cap = 2
    if (
        query_targets_poland_germany_treaty(query)
        or query_targets_wht_pay_and_refund_services(query)
        or query_targets_estonian_cit_hidden_profit(query)
        or query_targets_ksef_current_law(query)
        or query_targets_family_foundation_mechanism(query)
    ):
        bundle_cap = 10
    bundle_limit = min(bundle_cap, max(0, len(hinted_statutes)))
    bundle_statutes: list[RagChunk] = []
    for chunk in hinted_statutes:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_statute_sources:
            continue
        seen_statute_sources.add(canonical_source_id)
        bundle_statutes.append(annotate_chunk_evidence_role(chunk, "bundle_source"))
        if len(bundle_statutes) >= bundle_limit:
            break

    primary_chunks = [*merged_statutes, *bundle_statutes]
    if source_plan.primary_required and not legal_source_plan_primary_satisfied(source_plan, primary_chunks):
        return primary_chunks[:effective_limit] if primary_chunks else []

    mixed: list[RagChunk] = [*primary_chunks]
    if include_judgments:
        mixed.extend(judgments)
    mixed.extend(interpretations)
    return mixed[: effective_limit + len(bundle_statutes)]
