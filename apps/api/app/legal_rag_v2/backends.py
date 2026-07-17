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
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .retrieval import RetrievalCandidate


_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", re.UNICODE)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPLICIT_REFERENCE_RE = re.compile(
    r"\bart\.\s*\d+[a-z]*"
    r"(?:\s*(?:ust\.\s*\d+[a-z]*|§\s*\d+[a-z]*))?"
    r"(?:\s*pkt\s*\d+[a-z]*)?"
    r"(?:\s*lit\.\s*[a-z])?",
    re.IGNORECASE,
)
_DISPLAY_REFERENCE_RE = re.compile(
    r"art\.\s*\d+[a-z]*"
    r"(?:\s+(?:ust\.\s*\d+[a-z]*|§\s*\d+[a-z]*))?"
    r"(?:\s+pkt\s*\d+[a-z]*)?"
    r"(?:\s+lit\.\s*[a-z])?",
    re.IGNORECASE,
)
_QUERY_STOP_WORDS = frozenset(
    {"ale", "czy", "dla", "jak", "jest", "która", "które", "nie", "oraz", "się", "ten", "tego", "tym", "ust", "art", "pkt"}
)


def _query_tokens(query: str, *, limit: int = 24) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0).casefold()
        if len(token) < 3 or token in _QUERY_STOP_WORDS or token in tokens:
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


def _merge_document_chunks(chunks: Sequence[str], *, max_chars: int = 60_000) -> str:
    merged = ""
    for raw in chunks:
        chunk = str(raw or "").strip()
        if not chunk:
            continue
        if not merged:
            merged = chunk
        else:
            overlap = 0
            for size in range(min(500, len(merged), len(chunk)), 40, -1):
                if merged[-size:] == chunk[:size]:
                    overlap = size
                    break
            merged += chunk[overlap:] if overlap else "\n\n" + chunk
        if len(merged) >= max_chars:
            return merged[:max_chars]
    return merged


def _authority_document_excerpt(text: str, *, max_chars: int = 18_000) -> str:
    """Keep facts, explicit disposition and legal assessment in one compact source."""

    if len(text) <= max_chars:
        return text
    intervals: list[tuple[int, int]] = [(0, min(3_500, len(text)))]
    heading_pattern = re.compile(
        r"(?im)^\s*(?:ocena\s+stanowiska|ocena\s*\n\s*stanowiska|"
        r"uzasadnienie(?:\s+interpretacji\s+indywidualnej)?|"
        r"rozstrzygnięcie\s+sądu|sąd\s+zważył)\s*:?[ \t]*$"
    )
    for match in heading_pattern.finditer(text):
        intervals.append((max(0, match.start() - 1_000), min(len(text), match.start() + 9_000)))
    intervals.append((max(0, len(text) - 3_500), len(text)))

    selected: list[tuple[int, int]] = []
    budget = max_chars
    for start, end in intervals:
        for previous_start, previous_end in selected:
            if previous_start <= start < previous_end:
                start = previous_end
        if start >= end or budget <= 0:
            continue
        end = min(end, start + budget)
        selected.append((start, end))
        budget -= end - start
    selected.sort()
    return "\n\n[... pominięto fragment dokumentu ...]\n\n".join(
        text[start:end] for start, end in selected
    )


def _normalize_reference(value: str) -> str:
    return " ".join(value.casefold().replace("artykuł", "art.").split()).strip(" .;:,")


def _explicit_reference(query: str) -> str:
    match = _EXPLICIT_REFERENCE_RE.search(query)
    return _normalize_reference(match.group(0)) if match else ""


def _display_reference(row: Mapping[str, Any]) -> str:
    """Recover exact citations from legacy multi-letter article chunks."""

    declared = str(row.get("display_reference") or "").strip()
    if declared:
        return _normalize_reference(declared)
    first_line = next(
        (line.strip() for line in str(row.get("chunk_text") or "").splitlines() if line.strip()),
        "",
    )
    if _DISPLAY_REFERENCE_RE.fullmatch(first_line):
        return _normalize_reference(first_line)
    return ""


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
    if source_type == "interpretation" and source_subtype == "general":
        # Backward compatibility for local indexes created before general MF
        # interpretations received their own canonical authority type.
        return "general_interpretation"
    return source_type


def _candidate_from_row(row: Mapping[str, Any], *, backend: str, rank: int) -> RetrievalCandidate:
    chunk_id = str(row.get("chunk_id") or "")
    document_id = str(row.get("document_id") or "")
    lexical_score = float(row.get("lexical_score") or 0.0)
    display_reference = _display_reference(row)
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
            "act_title": str(row.get("act_title") or ""),
            "legal_state_date": str(row.get("legal_state_date") or ""),
            "tax_domains": [str(row.get("tax_domain") or "").upper()]
            if row.get("tax_domain")
            else [],
            "legal_provisions": (
                [display_reference]
                if display_reference
                else [str(value) for value in _json_list(row.get("legal_provisions_json"))]
            ),
            "display_reference": display_reference,
            "provision_id": str(row.get("provision_id") or ""),
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

    async def hydrate_document(
        self,
        candidate: RetrievalCandidate,
    ) -> RetrievalCandidate:
        """Expand a selected authority chunk to its source document."""

        if not candidate.document_id:
            return candidate
        return await asyncio.to_thread(self._hydrate_document_sync, candidate)

    def _hydrate_document_sync(
        self,
        candidate: RetrievalCandidate,
    ) -> RetrievalCandidate:
        if self.backend == "mysql":
            from app.mysql_rag import get_mysql_target, mysql_connection

            _documents_table, chunks_table = get_mysql_target()
            if not _SAFE_IDENTIFIER_RE.fullmatch(chunks_table):
                raise ValueError("Unsafe MySQL RAG table identifier")
            with mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"SELECT chunk_text FROM `{chunks_table}` "
                        "WHERE document_id = %s ORDER BY chunk_index ASC, chunk_id ASC",
                        (candidate.document_id,),
                    )
                    chunks = [str(row.get("chunk_text") or "") for row in cursor.fetchall()]
        else:
            if self.sqlite_path is None:
                from app.rag import ensure_local_index_ready, get_rag_config

                ensure_local_index_ready()
                db_path = get_rag_config().db_path
            else:
                db_path = self.sqlite_path
            if not db_path.exists():
                return candidate
            connection = sqlite3.connect(db_path)
            try:
                chunks = [
                    str(row[0] or "")
                    for row in connection.execute(
                        "SELECT chunk_text FROM chunks WHERE document_id = ? "
                        "ORDER BY chunk_index ASC, chunk_id ASC",
                        (candidate.document_id,),
                    ).fetchall()
                ]
            finally:
                connection.close()
        text = _authority_document_excerpt(_merge_document_chunks(chunks))
        if not text:
            return candidate
        return replace(
            candidate,
            text=text,
            chunk_id=f"document:{candidate.document_id}:full",
            positive_reasons=tuple(
                dict.fromkeys((*candidate.positive_reasons, "authority_document_hydrated"))
            ),
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
        if "general_interpretation" in values:
            # The local SQLite corpus may still contain pre-migration rows
            # marked as interpretation + source_subtype=general.  They are
            # normalized at candidate construction and filtered by the lane.
            values.add("interpretation")
        return sorted(values - {"tax_treaty", "regulation"})

    @staticmethod
    def _tax_domains(metadata_filters: Mapping[str, Any]) -> list[str]:
        raw = metadata_filters.get("tax_domains") or []
        values = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
        return sorted({str(value).upper() for value in values if str(value).strip()})

    @staticmethod
    def _domain_clause(
        domains: list[str],
        source_types: frozenset[str],
        placeholder: str,
    ) -> tuple[str, list[str]]:
        """Keep tax treaties visible inside CIT/PIT issue lanes.

        Treaty records are stored as statutes but carry the generic treaty tax
        domain.  Filtering them out before reranking makes a PL-DE treaty lane
        impossible even when the correct article is in the corpus.
        """
        if not domains:
            return "", []
        domain_filter = "UPPER(d.tax_domain) IN (" + ",".join(placeholder for _ in domains) + ")"
        if "tax_treaty" in source_types:
            return f"({domain_filter} OR LOWER(d.source_subtype) = 'tax_treaty')", domains
        return domain_filter, domains

    def _search_sqlite(
        self,
        query: str,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        explicit_reference = _explicit_reference(query)
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
        domain_clause, domain_values = self._domain_clause(domains, source_types, "?")
        if domain_clause:
            clauses.append(domain_clause)
            values.extend(domain_values)
        values.append(limit)
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            if explicit_reference:
                reference_prefix = f"{explicit_reference}%"
                exact_clauses = ["(c.display_reference LIKE ? OR c.chunk_text LIKE ?)"]
                exact_values: list[Any] = [reference_prefix, f"{explicit_reference}\n%"]
                if types:
                    exact_clauses.append("d.source_type IN (" + ",".join("?" for _ in types) + ")")
                    exact_values.extend(types)
                if domain_clause:
                    exact_clauses.append(domain_clause)
                    exact_values.extend(domain_values)
                exact_rows = connection.execute(
                    """
                    SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                           c.display_reference, c.provision_id,
                           d.subject, d.signature, d.published_date, d.source_url,
                           d.category, d.tax_domain, d.source, d.source_type,
                           d.source_subtype, d.authority, d.act_title, d.publication,
                           d.legal_state_date, d.legal_provisions_json,
                           d.source_pages_json, 1000.0 AS lexical_score
                    FROM chunks c JOIN documents d ON d.document_id = c.document_id
                    WHERE """ + " AND ".join(exact_clauses)
                    + " ORDER BY (LOWER(c.display_reference) = LOWER(?)) DESC, "
                    + "(c.chunk_text LIKE ?) DESC, LENGTH(c.chunk_text) DESC, "
                    + "d.legal_state_date DESC, c.chunk_id ASC LIMIT ?",
                    tuple(
                        [
                            *exact_values,
                            explicit_reference,
                            f"{explicit_reference}\n%",
                            limit,
                        ]
                    ),
                ).fetchall()
                if exact_rows:
                    return [dict(row) for row in exact_rows]
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                       c.display_reference, c.provision_id,
                       d.subject, d.signature, d.published_date, d.source_url,
                       d.category, d.tax_domain, d.source, d.source_type,
                       d.source_subtype, d.authority, d.act_title, d.publication,
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
        explicit_reference = _explicit_reference(query)
        match_query = _mysql_match_query(query)
        if not match_query:
            return []
        from app.mysql_rag import get_mysql_target, mysql_connection
        from app.rag import treaty_direct_subject_prefix

        documents_table, chunks_table = get_mysql_target()
        citations_table = f"{chunks_table}_citations"
        for identifier in (documents_table, chunks_table, citations_table):
            if not _SAFE_IDENTIFIER_RE.fullmatch(identifier):
                raise ValueError("Unsafe MySQL RAG table identifier")
        types = self._storage_source_types(source_types)
        domains = self._tax_domains(metadata_filters)
        treaty_prefix = treaty_direct_subject_prefix(query) if "tax_treaty" in source_types else None
        if treaty_prefix:
            # Treaty article numbers recur in every UPO.  An unscoped exact
            # lookup for art. 11 can therefore choose a treaty for a different
            # country.  The query family already names the jurisdiction, so
            # make that a deterministic predicate and skip broad FULLTEXT.
            treaty_clauses = [
                "d.source_type = 'statute'",
                "LOWER(d.source_subtype) = 'tax_treaty'",
                "LOWER(d.subject) LIKE LOWER(%s)",
            ]
            treaty_values: list[Any] = [f"{treaty_prefix}%"]
            if explicit_reference:
                reference_prefix = f"{explicit_reference}%"
                treaty_clauses.append("c.display_reference LIKE %s")
                treaty_values.append(reference_prefix)
            treaty_sql = f"""
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                       c.display_reference, c.provision_id,
                       d.subject, d.signature, d.published_date, d.source_url,
                       d.category, d.tax_domain, d.source, d.source_type,
                       d.source_subtype, d.authority, d.act_title, d.publication,
                       d.legal_state_date, d.legal_provisions_json,
                       d.source_pages_json, 1000.0 AS lexical_score
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE {' AND '.join(treaty_clauses)}
                ORDER BY (c.display_reference LIKE %s) DESC, c.chunk_index ASC, c.chunk_id ASC
                LIMIT %s
            """
            with mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(treaty_sql, (*treaty_values, reference_prefix if explicit_reference else "", limit))
                    return [dict(row) for row in cursor.fetchall()]
        if explicit_reference:
            reference_prefix = f"{explicit_reference}%"
            exact_clauses = ["(c.display_reference LIKE %s OR c.chunk_text LIKE %s)"]
            exact_values: list[Any] = [reference_prefix, f"{explicit_reference}\n%"]
            if types:
                exact_clauses.append("d.source_type IN (" + ",".join("%s" for _ in types) + ")")
                exact_values.extend(types)
            domain_clause, domain_values = self._domain_clause(domains, source_types, "%s")
            if domain_clause:
                exact_clauses.append(domain_clause)
                exact_values.extend(domain_values)
            exact_sql = f"""
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                       c.display_reference, c.provision_id,
                       d.subject, d.signature, d.published_date, d.source_url,
                       d.category, d.tax_domain, d.source, d.source_type,
                       d.source_subtype, d.authority, d.act_title, d.publication,
                       d.legal_state_date, d.legal_provisions_json,
                       d.source_pages_json, 1000.0 AS lexical_score
                FROM `{chunks_table}` c
                JOIN `{documents_table}` d ON d.document_id = c.document_id
                WHERE {' AND '.join(exact_clauses)}
                ORDER BY (LOWER(c.display_reference) = LOWER(%s)) DESC,
                         (c.chunk_text LIKE %s) DESC, CHAR_LENGTH(c.chunk_text) DESC,
                         d.legal_state_date DESC, c.chunk_id ASC LIMIT %s
            """
            with mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        exact_sql,
                        (
                            *exact_values,
                            explicit_reference,
                            f"{explicit_reference}\n%",
                            limit,
                        ),
                    )
                    rows = [dict(row) for row in cursor.fetchall()]
                    if rows:
                        return rows
        clauses = [
            "MATCH(c.search_text, c.question_text, c.facts_text, c.tax_domain) "
            "AGAINST (%s IN BOOLEAN MODE)"
        ]
        values: list[Any] = [match_query]
        if types:
            clauses.append("d.source_type IN (" + ",".join("%s" for _ in types) + ")")
            values.extend(types)
        domain_clause, domain_values = self._domain_clause(domains, source_types, "%s")
        if domain_clause:
            clauses.append(domain_clause)
            values.extend(domain_values)
        sql = f"""
            SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                   c.display_reference, c.provision_id,
                   d.subject, d.signature, d.published_date, d.source_url,
                   d.category, d.tax_domain, d.source, d.source_type,
                   d.source_subtype, d.authority, d.act_title, d.publication,
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
