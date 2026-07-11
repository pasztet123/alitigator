"""Policy-free FTS candidate adapters for legal RAG v2.

The legacy search module mixes storage with topic routing and benchmark-era
document boosts.  V2 uses this adapter for broad lexical recall only.  Query
families, legal concepts and metadata filters must already be present in the
model-produced :class:`LegalResearchPlan`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from .retrieval import RetrievalCandidate


_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", re.UNICODE)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _query_tokens(query: str, *, limit: int = 24) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0).casefold()
        if len(token) < 2 or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _sqlite_match_query(query: str) -> str:
    # OR keeps candidate generation recall-oriented. Legal precision belongs to
    # the transparent reranker, not to an opaque FTS routing rule.
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"*' for token in _query_tokens(query))


def _mysql_match_query(query: str) -> str:
    return " ".join(f"{token}*" if len(token) >= 4 else token for token in _query_tokens(query))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _normalized_source_type(row: Mapping[str, Any]) -> str:
    source_type = str(row.get("source_type") or "unknown").lower()
    source_subtype = str(row.get("source_subtype") or "").lower()
    if source_type == "statute" and source_subtype == "tax_treaty":
        return "tax_treaty"
    return source_type


def _candidate_from_row(row: Mapping[str, Any], *, backend: str, rank: int) -> RetrievalCandidate:
    chunk_id = str(row.get("chunk_id") or "")
    document_id = str(row.get("document_id") or "")
    lexical_score = float(row.get("lexical_score") or 0.0)
    # SQLite BM25 is lower-is-better and commonly negative; MySQL MATCH is
    # higher-is-better. RRF ultimately uses rank, while this normalized value is
    # retained only as an auditable component.
    normalized_score = 1.0 / max(rank, 1)
    return RetrievalCandidate(
        candidate_id=chunk_id or f"{document_id}:{rank}",
        document_id=document_id,
        chunk_id=chunk_id,
        text=str(row.get("chunk_text") or ""),
        source_type=_normalized_source_type(row),
        score=normalized_score,
        metadata={
            "subject": str(row.get("subject") or ""),
            "signature": str(row.get("signature") or ""),
            "published_date": str(row.get("published_date") or ""),
            "source_url": str(row.get("source_url") or ""),
            "category": str(row.get("category") or ""),
            "source": str(row.get("source") or ""),
            "source_subtype": str(row.get("source_subtype") or ""),
            "authority": str(row.get("authority") or ""),
            "publication": str(row.get("publication") or ""),
            "legal_state_date": str(row.get("legal_state_date") or ""),
            "tax_domains": [str(row.get("tax_domain") or "").upper()]
            if row.get("tax_domain")
            else [],
            "legal_provisions": [str(value) for value in _json_list(row.get("legal_provisions_json"))],
            "source_pages": _json_list(row.get("source_pages_json")),
            "lexical_score_raw": lexical_score,
        },
        backend=backend,
        channel_ranks={"lexical": rank},
        component_scores={"lexical_rank": normalized_score},
        positive_reasons=("generic_full_text_match",),
    )


class CorpusFtsBackend:
    """Generic SQLite/MySQL full-text retrieval with no legal-topic router."""

    trace_marker = "legal_rag_v2_generic_fts"

    def __init__(self, *, backend: str | None = None, sqlite_path: str | Path | None = None) -> None:
        selected = (backend or os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite")).strip().lower()
        if selected == "mariadb":
            selected = "mysql"
        if selected not in {"sqlite", "mysql"}:
            raise ValueError(f"Unsupported v2 FTS backend: {selected!r}")
        self.backend = selected
        self.sqlite_path = Path(sqlite_path).expanduser() if sqlite_path else None

    async def search(
        self,
        query: str,
        *,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> Sequence[RetrievalCandidate]:
        if limit <= 0 or not query.strip():
            return []
        return await asyncio.to_thread(
            self._search_sync,
            query,
            limit,
            source_types,
            metadata_filters,
        )

    def _search_sync(
        self,
        query: str,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[RetrievalCandidate]:
        if self.backend == "mysql":
            rows = self._search_mysql(query, limit, source_types, metadata_filters)
        else:
            rows = self._search_sqlite(query, limit, source_types, metadata_filters)
        return [
            _candidate_from_row(row, backend=self.trace_marker, rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    @staticmethod
    def _storage_source_types(source_types: frozenset[str]) -> list[str]:
        values = {value.lower() for value in source_types}
        if "tax_treaty" in values or "regulation" in values:
            values.add("statute")
        return sorted(values - {"tax_treaty", "regulation"})

    @staticmethod
    def _tax_domains(metadata_filters: Mapping[str, Any]) -> list[str]:
        raw = metadata_filters.get("tax_domains") or []
        values = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
        return sorted({str(value).upper() for value in values if str(value).strip()})

    def _search_sqlite(
        self,
        query: str,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        match_query = _sqlite_match_query(query)
        if not match_query:
            return []
        if self.sqlite_path is None:
            from app.rag import ensure_local_index_ready, get_rag_config

            ensure_local_index_ready()
            db_path = get_rag_config().db_path
        else:
            db_path = self.sqlite_path
        if not db_path.exists():
            return []

        types = self._storage_source_types(source_types)
        domains = self._tax_domains(metadata_filters)
        clauses = ["chunks_fts MATCH ?"]
        values: list[Any] = [match_query]
        if types:
            clauses.append("d.source_type IN (" + ",".join("?" for _ in types) + ")")
            values.extend(types)
        if domains:
            clauses.append("UPPER(d.tax_domain) IN (" + ",".join("?" for _ in domains) + ")")
            values.extend(domains)
        values.append(limit)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                       d.subject, d.signature, d.published_date, d.source_url,
                       d.category, d.tax_domain, d.source, d.source_type,
                       d.source_subtype, d.authority, d.publication,
                       d.legal_state_date, d.legal_provisions_json,
                       d.source_pages_json,
                       bm25(chunks_fts, 1.0, 2.5, 4.0, 1.5, 2.5, 2.5,
                            5.0, 4.0, 3.0) AS lexical_score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                JOIN documents d ON d.document_id = c.document_id
                WHERE """
                + " AND ".join(clauses)
                + " ORDER BY lexical_score, d.published_date DESC, c.chunk_index ASC LIMIT ?",
                tuple(values),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def _search_mysql(
        self,
        query: str,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        match_query = _mysql_match_query(query)
        if not match_query:
            return []
        from app.mysql_rag import get_mysql_target, mysql_connection

        documents_table, chunks_table = get_mysql_target()
        for identifier in (documents_table, chunks_table):
            if not _SAFE_IDENTIFIER_RE.fullmatch(identifier):
                raise ValueError("Unsafe MySQL RAG table identifier")
        types = self._storage_source_types(source_types)
        domains = self._tax_domains(metadata_filters)
        clauses = [
            "MATCH(c.search_text, c.question_text, c.facts_text, c.tax_domain) "
            "AGAINST (%s IN BOOLEAN MODE)"
        ]
        values: list[Any] = [match_query]
        if types:
            clauses.append("d.source_type IN (" + ",".join("%s" for _ in types) + ")")
            values.extend(types)
        if domains:
            clauses.append("UPPER(d.tax_domain) IN (" + ",".join("%s" for _ in domains) + ")")
            values.extend(domains)
        sql = f"""
            SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                   d.subject, d.signature, d.published_date, d.source_url,
                   d.category, d.tax_domain, d.source, d.source_type,
                   d.source_subtype, d.authority, d.publication,
                   d.legal_state_date, d.legal_provisions_json,
                   d.source_pages_json,
                   MATCH(c.search_text, c.question_text, c.facts_text, c.tax_domain)
                       AGAINST (%s IN BOOLEAN MODE) AS lexical_score
            FROM `{chunks_table}` c
            JOIN `{documents_table}` d ON d.document_id = c.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY lexical_score DESC, d.published_date DESC,
                     c.chunk_index ASC, c.chunk_id ASC
            LIMIT %s
        """
        # First placeholder computes score, second applies the MATCH filter.
        params = (match_query, *values, limit)
        with mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]


__all__ = ["CorpusFtsBackend"]
