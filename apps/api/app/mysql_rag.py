from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import pymysql
from pymysql.cursors import DictCursor

from app.rag import (
    JUDGMENT_INTENT_RE,
    JUDGMENT_ONLY_CONTEXT_RE,
    KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS,
    KSEF_FOREIGN_SALE_STATUTE_TARGETS,
    QUERY_TOKEN_RE,
    RagChunk,
    RetrievalInspection,
    build_article_family_match_score,
    build_chunk_payload,
    build_context_block,
    build_match_query,
    build_shareholder_company_asset_sale_statute_targets,
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
    extract_primary_article_key,
    extract_statute_target_from_text,
    filter_index_chunks,
    get_query_expansion_terms,
    get_rag_config,
    json_dump,
    join_search_text,
    normalize_source_type,
    query_targets_interpretation_procedure,
    query_targets_ksef_foreign_sale,
    query_targets_shareholder_company_asset_sale,
    query_targets_small_taxpayer_foreign_vat,
    rank_hybrid_local_candidates,
    ranking_terms,
    resolve_statute_tax_domains,
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
    resolve_cross_blend_weight,
)


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
                search_text MEDIUMTEXT NOT NULL,
                question_text MEDIUMTEXT NOT NULL,
                facts_text MEDIUMTEXT NOT NULL,
                tax_domain varchar(64) NOT NULL DEFAULT '',
                FULLTEXT KEY ft_search (search_text, question_text, facts_text, tax_domain),
                KEY idx_document_id (document_id),
                KEY idx_document_chunk (document_id, chunk_index),
                CONSTRAINT fk_rag_chunks_document FOREIGN KEY (document_id)
                    REFERENCES `{documents_table}`(document_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    connection.commit()


def mysql_backend_label() -> str:
    host = os.getenv("ALITIGATOR_RAG_MYSQL_HOST", "").strip()
    database = os.getenv("ALITIGATOR_RAG_MYSQL_DATABASE", "").strip()
    return f"mysql://{host}/{database}" if host and database else "mysql"


def index_exists_mysql() -> bool:
    if not is_mysql_rag_configured():
        return False
    documents_table, chunks_table = get_mysql_target()
    try:
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
        "content_sha256": record.get("content_sha256"),
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
    from app.rag import clean_document_text

    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return {}, []

    document_text = clean_document_text(record)
    if not document_text:
        return {}, []

    chunks = [document_text] if record.get("pre_chunked") else split_into_chunks(
        document_text,
        target_chars=config.chunk_target_chars,
        overlap_chars=config.chunk_overlap_chars,
    )
    chunks = filter_index_chunks(record, chunks)
    if not chunks:
        return {}, []

    document_row = local_record_to_mysql_document(record)
    search_keywords = json.loads(document_row["keywords_json"])
    search_legal_provisions = json.loads(document_row["legal_provisions_json"])
    search_issues = json.loads(document_row["issues_json"])
    search_law_tags = json.loads(document_row["law_tags_json"])
    chunk_rows: list[dict[str, Any]] = []
    for chunk_index, chunk_text in enumerate(chunks):
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

                current_sha = str(record.get("content_sha256") or "")
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
                    d.source_pages_json, c.chunk_chars
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE {where_sql}
                ORDER BY d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
                LIMIT %s
                """,
                (*params, limit),
            )
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
        for article in sorted(statute_exact_articles):
            clauses.append("d.legal_provisions_json = %s")
            values.append(json_dump([f"art. {article}"]))
        for prefix in sorted(statute_family_prefixes):
            clauses.append("d.legal_provisions_json LIKE %s")
            values.append(f'%art. {prefix}%')
        if clauses:
            query_rows.append(
                select_candidate_rows_mysql(
                    "d.source_type = 'statute' AND c.chunk_index = 0 AND (" + " OR ".join(clauses) + ")" + filter_sql.replace(" AND d.source_type IN (%s)", ""),
                    [*values, *filter_values],
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


def search_chat_chunks_mysql(
    query: str,
    *,
    limit: Optional[int] = None,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    judgment_requested_by_query = bool(JUDGMENT_INTENT_RE.search(query) or extract_judgment_signatures(query))
    include_judgments = judgment_requested_by_query if include_judgments is None else include_judgments
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
        interpretation_limit = max(1, effective_limit - statute_limit) if include_interpretations else 0

    if interpretation_limit and query_targets_ksef_foreign_sale(query):
        interpretation_rows = fetch_rows_by_document_ids_mysql(
            KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS,
            source_type="interpretation",
        )
        interpretations = (
            rank_hybrid_local_candidates(
                interpretation_rows,
                query=expand_search_query(query),
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

    preferred_targets: list[tuple[str, str]] = []
    if query_targets_ksef_foreign_sale(query):
        preferred_targets.extend(KSEF_FOREIGN_SALE_STATUTE_TARGETS)
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

    hinted_rows = fetch_statute_rows_by_targets_mysql(preferred_targets, limit=statute_limit) if statute_limit else []
    hinted_statutes = (
        rank_hybrid_local_candidates(
            hinted_rows,
            query=expand_search_query(query),
            effective_limit=statute_limit,
            config=config,
        )
        if hinted_rows
        else []
    )
    merged_statutes: list[RagChunk] = []
    seen_statute_chunks: set[str] = set()
    prefer_hinted_statutes = (
        query_targets_ksef_foreign_sale(query)
        or query_targets_shareholder_company_asset_sale(query)
        or query_targets_small_taxpayer_foreign_vat(query)
    )
    statute_candidates = [*hinted_statutes, *statutes] if prefer_hinted_statutes else [*statutes, *hinted_statutes]
    for chunk in statute_candidates:
        if chunk.chunk_id in seen_statute_chunks:
            continue
        seen_statute_chunks.add(chunk.chunk_id)
        merged_statutes.append(chunk)
        if len(merged_statutes) >= statute_limit:
            break

    mixed: list[RagChunk] = []
    for position in range(max(len(judgments), len(merged_statutes), len(interpretations))):
        if include_judgments and position < len(judgments):
            mixed.append(judgments[position])
        if position < len(merged_statutes):
            mixed.append(merged_statutes[position])
        if position < len(interpretations):
            mixed.append(interpretations[position])
    return mixed[:effective_limit]
