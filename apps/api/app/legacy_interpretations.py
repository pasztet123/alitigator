"""Interpretation search pinned to the July 7, 2026 retrieval revision."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import unicodedata
from contextvars import ContextVar
from dataclasses import replace

from app.legacy_july7 import rag as july7_rag
from app.legacy_july7 import mysql_rag as july7_mysql_rag


JULY7_RETRIEVAL_COMMIT = "6a23c08"
JULY7_RETRIEVAL_DATE = "2026-07-07"

_CORPUS_HEALTH_CACHE_SECONDS = 60.0
_corpus_health_cache: tuple[float, dict[str, object]] | None = None


def get_july7_interpretation_corpus_health() -> dict[str, object]:
    """Report whether the isolated interpretation corpus can be read.

    This deliberately checks only aggregate counts.  It lets the public health
    endpoint distinguish a configured backend from a corpus that is actually
    available to the retrieval-only MVP, without exposing database details or
    any document content.
    """
    global _corpus_health_cache
    now = time.monotonic()
    if _corpus_health_cache and now - _corpus_health_cache[0] < _CORPUS_HEALTH_CACHE_SECONDS:
        return dict(_corpus_health_cache[1])

    backend = get_july7_interpretation_backend()
    try:
        if july7_mysql_rag.is_mysql_rag_configured():
            documents_table, chunks_table = july7_mysql_rag.get_mysql_target()
            with july7_mysql_rag.mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"SELECT COUNT(*) AS count FROM `{documents_table}` WHERE source_type = %s",
                        ("interpretation",),
                    )
                    document_count = int((cursor.fetchone() or {}).get("count") or 0)
                    cursor.execute(
                        f"""
                        SELECT COUNT(*) AS count
                        FROM `{chunks_table}` c
                        JOIN `{documents_table}` d ON d.document_id = c.document_id
                        WHERE d.source_type = %s
                        """,
                        ("interpretation",),
                    )
                    chunk_count = int((cursor.fetchone() or {}).get("count") or 0)
        else:
            config = july7_rag.get_rag_config()
            if not config.db_path.exists():
                raise FileNotFoundError(config.db_path)
            connection = july7_rag.get_connection(config.db_path)
            try:
                document_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM documents WHERE source_type = 'interpretation'"
                    ).fetchone()[0]
                )
                chunk_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM chunks c
                        JOIN documents d ON d.document_id = c.document_id
                        WHERE d.source_type = 'interpretation'
                        """
                    ).fetchone()[0]
                )
            finally:
                connection.close()
        report: dict[str, object] = {
            "backend": backend,
            "available": document_count > 0 and chunk_count > 0,
            "interpretation_documents": document_count,
            "interpretation_chunks": chunk_count,
        }
    except Exception:
        report = {
            "backend": backend,
            "available": False,
            "reason": "corpus_unavailable",
        }

    _corpus_health_cache = (now, report)
    return dict(report)


def get_july7_interpretation_backend() -> str:
    """Identify the isolated snapshot path without exposing connection details."""
    return "mysql_july7_snapshot" if july7_mysql_rag.is_mysql_rag_configured() else "sqlite_july7_snapshot"


def _do_not_reindex_live_corpus() -> None:
    """A historical query must not rebuild the active multi-gigabyte index."""


# The July snapshot normally rebuilds its local database when any corpus input
# is newer than the database.  Its database is shared with the live backend, so
# doing so from an interactive historical lookup would mutate production state
# and can take minutes.  Retrieval itself remains the July 7 implementation.
july7_rag.ensure_local_index_ready = _do_not_reindex_live_corpus

_get_july7_rag_config = july7_rag.get_rag_config


def _get_bounded_july7_rag_config():
    """Keep the retrieval-only request bounded and document-diverse."""
    config = _get_july7_rag_config()
    cross_encoder_enabled = os.getenv(
        "JULY7_INTERPRETATIONS_CROSS_ENCODER_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes"}
    return replace(
        config,
        cross_encoder_enabled=cross_encoder_enabled,
        # The MVP returns documents, so a second chunk from the same document
        # cannot improve the result set before the full-document hydration.
        retrieval_max_chunks_per_document=1,
    )


july7_rag.get_rag_config = _get_bounded_july7_rag_config
july7_mysql_rag.get_rag_config = _get_bounded_july7_rag_config

_active_user_query: ContextVar[str | None] = ContextVar("active_july7_user_query", default=None)


def _normalize_for_match(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(character) != "Mn"
    )


_HISTORICAL_QUERY_STOPWORDS = frozenset(
    _normalize_for_match(value)
    for value in {
        "albo", "ale", "bardzo", "będzie", "bez", "być", "całości", "czy", "dla", "działalności",
        "działalność", "gospodarcza", "gospodarczego", "jest", "jako", "jeżeli", "jego",
        "jej", "której", "lub", "nie", "przy", "kiedy", "która", "który", "mają", "może", "oraz",
        "podlega", "podstawę", "przez", "sprzedaż", "tego", "tych", "ustawy", "wtedy",
        "wyniku", "został", "została", "zostało", "zostały", "zostać", "powinien",
        "powinna", "powinno", "powinny", "zaliczyć", "stanowi", "stanowić",
        "podatkowy", "podatkowe",
        "firma", "firmę", "firmy", "spółka", "spółki", "możliwy", "możliwość", "moga",
        "interpretacji", "interpretacje", "interpretację", "dotyczących", "dotyczące",
        "poszukaj", "pokaż", "znajdź", "proszę", "tam", "innym", "innego", "wyłącznie",
        "związku", "realizacji", "realizacją", "klient", "klienta",
    }
)


_CONTEXTUAL_ANCHOR_PREFIXES = (
    "dzialaln", "gospodarcz",
    "podczas", "wiel", "godzin", "pracuj", "uzywan", "prowadz", "ponies",
    "zalicz", "stanow", "moz", "moga", "formaln", "osobn", "calkowic",
    "jednoczes", "charakter", "wynik", "podstaw", "podatkow", "firm", "musi", "polsk",
)

_TAX_INTENT_PREFIXES = ("koszt", "uzysk", "przychod", "podat", "vat", "pit", "cit", "pcc", "wht")
_MAX_HISTORICAL_COOCURRENCE_TERMS = 8
_MAX_HISTORICAL_PAIR_PROBES = 24
_GROUP_PRIMARY_CONTEXTUAL_PREFIXES = (
    "przedsiebiorc", "podatni", "wydatek", "wydatk", "dzialaln", "gospodarcz",
)


def _prefix_stem(value: str) -> str:
    """Return a conservative lexical prefix shared by common Polish inflections."""
    for suffix in (
        "owego", "owych", "owymi", "owej", "owemu", "aniem", "enia", "aniu",
        "nego", "nych", "nymi", "nego", "owej", "owie", "ami", "ach", "ego",
        "emu", "ymi", "owa", "owe", "owy", "ych", "ymi", "cie", "cia", "ciu",
        "nia", "nie", "nym", "nej", "owa", "owe", "owy", "owi", "ami", "ach",
        "em", "ow", "om", "a", "e", "i", "o", "u", "y",
    ):
        # Cutting ``-cie`` off words such as ``mieście`` leaves ``mies``.  In
        # a prefix FTS index that also matches the far more frequent
        # ``miejsce`` / ``miejscu``, which is a particularly destructive false
        # neighbour for questions about rented housing.  Keep this ending
        # unless the remaining stem is still specific enough.
        if suffix == "cie" and len(value) - len(suffix) < 5:
            continue
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[:-len(suffix)]
    return value


def _query_anchor_entries(query: str) -> list[tuple[str, bool]]:
    """Extract lexical concepts and generic identifier-like token signals."""
    entries: list[tuple[str, bool]] = []
    for token in july7_rag.QUERY_TOKEN_RE.findall(query):
        normalized = _normalize_for_match(token)
        identifier_like = sum(character.isupper() for character in token) >= 2 or any(
            character.isdigit() for character in token
        )
        if (len(normalized) < 3 and not identifier_like) or normalized in _HISTORICAL_QUERY_STOPWORDS:
            continue
        stem = _prefix_stem(normalized)
        if stem.startswith(_CONTEXTUAL_ANCHOR_PREFIXES):
            continue
        for index, (known_stem, known_identifier_like) in enumerate(entries):
            if known_stem == stem:
                entries[index] = (known_stem, known_identifier_like or identifier_like)
                break
        else:
            entries.append((stem, identifier_like))
    return entries


def _query_anchor_stems(query: str) -> list[str]:
    """Extract material lexical concepts without domain-specific expansions."""
    return [stem for stem, _ in _query_anchor_entries(query)]


def _is_tax_intent_anchor(anchor: str) -> bool:
    """Identify generic tax-outcome words that occur in most authorities."""
    return anchor.startswith(_TAX_INTENT_PREFIXES)


def _build_generic_probe_pairs(query: str) -> list[tuple[str, str]]:
    """Choose a small, lexical set of co-occurrence probes from the question.

    Each probe is derived only from the words present in the question.  In a
    natural-language list (for example ``zakup, karma i leczenie psa``), the
    leading action has to be compared with *each* listed object, not only with
    the first item.  Adjacent pairs retain local phrases as a second signal.
    This is a generic lexical rule; it does not encode tax topics or document
    types.
    """
    anchor_entries = _query_anchor_entries(_active_user_query.get() or query)
    if not anchor_entries:
        return []

    content_entries = [item for item in anchor_entries if not _is_tax_intent_anchor(item[0])]
    tax_entries = [item for item in anchor_entries if _is_tax_intent_anchor(item[0])]
    adjacent_content_pairs = list(zip(content_entries, content_entries[1:]))
    pairs: list[tuple[str, str]] = []
    if len(content_entries) >= 2:
        first_content = content_entries[0]
        pairs.extend(
            (first_content[0], entry[0])
            for entry in content_entries[1 : 1 + _MAX_HISTORICAL_COOCURRENCE_TERMS]
        )
        # Introductory words such as "przedsiębiorca" or "wydatek" still
        # matter to the ranker, but the first concrete object/action is a
        # better bridge across an enumerated fact list.
        concrete_primary = next(
            (
                entry
                for entry in content_entries
                if not entry[0].startswith(_GROUP_PRIMARY_CONTEXTUAL_PREFIXES)
            ),
            first_content,
        )
        pairs.extend(
            (concrete_primary[0], entry[0])
            for entry in content_entries
            if entry[0] != concrete_primary[0]
        )
        pairs.extend((left[0], right[0]) for left, right in adjacent_content_pairs)

    if content_entries and tax_entries:
        longest_tax = max(tax_entries, key=lambda item: min(len(item[0]), 12))
        first_content = content_entries[0]
        most_specific_content = max(content_entries, key=lambda item: min(len(item[0]), 12))
        pairs.extend(
            [
                (first_content[0], longest_tax[0]),
                (most_specific_content[0], longest_tax[0]),
            ]
        )
        identifier_contents = [item for item in content_entries if item[1]]
        if identifier_contents:
            identifier = max(identifier_contents, key=lambda item: min(len(item[0]), 12))
            pairs.append((identifier[0], longest_tax[0]))

    if not pairs and len(anchor_entries) == 1:
        pairs.append((anchor_entries[0][0], anchor_entries[0][0]))
    return list(dict.fromkeys(pair for pair in pairs if all(pair)))[:_MAX_HISTORICAL_PAIR_PROBES]


def _query_recall_anchors(query: str) -> list[str]:
    """Keep all descriptive query words in one bounded full-text channel."""
    anchors = _query_anchor_stems(_active_user_query.get() or query)
    distinctive = [anchor for anchor in anchors if not _is_tax_intent_anchor(anchor)]
    return distinctive or anchors


def _build_historical_cooccurrence_groups(query: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return bounded generic ``anchor + any listed fact`` lexical groups.

    The first concrete word keeps an enumerated fact pattern together.  A
    second group uses the shortest remaining concrete term: this preserves
    precise short nouns and identifiers (for example ``NIP``, ``KSeF`` or a
    named object) without a query-per-pair fan-out.
    """
    anchors = _query_recall_anchors(query)
    if len(anchors) < 2:
        return []
    concrete_anchors = [
        anchor
        for anchor in anchors
        if not anchor.startswith(_GROUP_PRIMARY_CONTEXTUAL_PREFIXES)
    ]
    primary = concrete_anchors[0] if concrete_anchors else anchors[0]
    primary_alternatives = tuple(
        anchor for anchor in anchors if anchor != primary
    )[:_MAX_HISTORICAL_COOCURRENCE_TERMS]
    groups = [(primary, primary_alternatives)] if primary_alternatives else []

    secondary_candidates = [anchor for anchor in concrete_anchors if anchor != primary]
    if secondary_candidates:
        secondary = min(secondary_candidates, key=lambda anchor: (len(anchor), anchor))
        secondary_alternatives = tuple(
            anchor for anchor in anchors if anchor != secondary
        )[:_MAX_HISTORICAL_COOCURRENCE_TERMS]
        if secondary_alternatives:
            groups.append((secondary, secondary_alternatives))
    return groups


def _query_precision_anchor(query: str) -> str | None:
    """Return the secondary, short concrete anchor when a query has one."""
    groups = _build_historical_cooccurrence_groups(query)
    return groups[1][0] if len(groups) > 1 else None


def _build_bounded_historical_match_queries(query: str, *, config=None) -> list[str]:
    """Keep July's SQLite FTS query selective on today's much larger corpus.

    The descriptive terms are kept in one FTS channel, excluding only broad
    tax-outcome words when the question contains more concrete facts.
    """
    candidate_query = _active_user_query.get() or query
    return _build_bounded_historical_sqlite_queries(candidate_query, config=config)


july7_rag._build_candidate_match_queries = july7_rag.build_candidate_match_queries
july7_rag.build_candidate_match_queries = _build_bounded_historical_match_queries


def _build_bounded_historical_mysql_queries(query: str) -> list[str]:
    """Build generic Boolean probes that retain lexical co-occurrence.

    In MySQL Boolean mode whitespace means OR.  A single long query therefore
    recalled any chunk containing ``miejsce`` for a question mentioning
    ``mieście`` and hit the database limit before relevant documents were even
    candidates.  Pair probes are derived solely from neighbouring and bridge
    words present in the user's question; they are not a taxonomy or a set of
    hard-coded legal cases.
    """
    grouped_queries = [
        f"+{primary}* +({' '.join(f'{anchor}*' for anchor in alternatives)})"
        for primary, alternatives in _build_historical_cooccurrence_groups(query)
    ]
    # Two selective groups are enough to keep both the leading fact and a
    # short precise object in recall.  A broad OR fallback admitted arbitrary
    # one-word neighbours before the ranker could compare the actual facts.
    queries = grouped_queries
    return list(dict.fromkeys(candidate for candidate in queries if candidate)) or july7_mysql_rag._build_mysql_candidate_queries(query)


def _build_bounded_historical_sqlite_queries(query: str, *, config=None) -> list[str]:
    """SQLite equivalent of the generic MySQL co-occurrence probes."""
    grouped_queries = [
        f'"{primary}"* AND (' + " OR ".join(
            f'"{anchor}"*' for anchor in alternatives
        ) + ")"
        for primary, alternatives in _build_historical_cooccurrence_groups(query)
    ]
    queries = grouped_queries
    return list(dict.fromkeys(candidate for candidate in queries if candidate)) or july7_rag._build_candidate_match_queries(query, config=config)


july7_mysql_rag._build_mysql_candidate_queries = july7_mysql_rag.build_mysql_candidate_queries
july7_mysql_rag.build_mysql_candidate_queries = _build_bounded_historical_mysql_queries


def _query_coverage(chunk: july7_rag.RagChunk, query: str) -> tuple[int, int, int, int, int, int]:
    """Count distinct query concepts occurring in a complete interpretation."""
    anchors = _query_anchor_stems(query)
    distinctive_anchors = [anchor for anchor in anchors if not _is_tax_intent_anchor(anchor)]
    tax_anchors = [anchor for anchor in anchors if _is_tax_intent_anchor(anchor)]
    subject = _normalize_for_match(chunk.subject or "")
    document = _normalize_for_match(chunk.chunk_text or "")
    distinctive_document_matches = sum(anchor in document for anchor in distinctive_anchors)
    distinctive_subject_matches = sum(anchor in subject for anchor in distinctive_anchors)
    tax_document_matches = sum(anchor in document for anchor in tax_anchors)
    tax_subject_matches = sum(anchor in subject for anchor in tax_anchors)
    document_matches = sum(anchor in document for anchor in anchors)
    subject_matches = sum(anchor in subject for anchor in anchors)
    return (
        distinctive_document_matches,
        tax_document_matches,
        distinctive_subject_matches,
        tax_subject_matches,
        document_matches,
        subject_matches,
    )


def _query_pair_matches(chunk: july7_rag.RagChunk, query: str) -> int:
    document = _normalize_for_match(" ".join((chunk.subject or "", chunk.chunk_text or "")))
    return sum(
        left in document and right in document
        for left, right in _build_generic_probe_pairs(query)
    )


def _query_subject_pair_matches(chunk: july7_rag.RagChunk, query: str) -> int:
    """Prefer a query co-occurrence in the concise document subject.

    A full interpretation contains boilerplate and historical facts, so two
    ordinary query words can occur far apart for an unrelated reason.  The
    subject is a short editor-supplied description and makes the same lexical
    overlap much stronger evidence without introducing any topic-specific
    rules.
    """
    subject = _normalize_for_match(chunk.subject or "")
    return sum(
        left in subject and right in subject
        for left, right in _build_generic_probe_pairs(query)
    )


def _query_precision_anchor_matches(chunk: july7_rag.RagChunk, query: str, *, subject_only: bool) -> int:
    """Check the secondary short query term in a subject or matching chunk."""
    anchor = _query_precision_anchor(query)
    if not anchor:
        return 0
    text = chunk.subject or "" if subject_only else " ".join((chunk.subject or "", chunk.chunk_text or ""))
    return int(anchor in _normalize_for_match(text))


_ARTICLE_REFERENCE_RE = re.compile(
    r"\bart\.?\s*(?P<article>\d+[a-z]{0,3})(?:\s*(?:ust\.?|§|par\.?|pkt)\s*(?P<unit>\d+[a-z]{0,3}))?",
    re.IGNORECASE,
)


def _row_value(row: object, key: str) -> object:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None


def _row_json_strings(row: object, key: str) -> list[str]:
    raw_value = _row_value(row, key)
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value if str(value).strip()]
    try:
        decoded = json.loads(str(raw_value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(value) for value in decoded] if isinstance(decoded, list) else []


def _query_article_references(query: str) -> set[tuple[str, str | None]]:
    return {
        (match.group("article").lower(), (match.group("unit") or "").lower() or None)
        for match in _ARTICLE_REFERENCE_RE.finditer(_normalize_for_match(query))
    }


def _metadata_match_score(row: object, *, query: str) -> tuple[int, int, int]:
    """Return soft metadata evidence for a candidate interpretation.

    Keywords are independently curated document phrases, so lexical agreement
    with the user's own terms is useful even when the same terms occur only in
    a factual section outside the winning chunk.  A legal-provision score is
    intentionally zero unless the user explicitly cites an article.  Neither
    signal filters candidates; both only affect ordering among candidates
    already recalled from the document text.
    """
    keyword_text = " ".join(_row_json_strings(row, "keywords_json"))
    normalized_keywords = _normalize_for_match(keyword_text)
    keyword_anchors = [
        anchor
        for anchor in _query_recall_anchors(query)
        if len(anchor) >= 4 and not _is_tax_intent_anchor(anchor)
    ]
    keyword_coverage = sum(anchor in normalized_keywords for anchor in keyword_anchors)
    keyword_pairs = sum(
        left in normalized_keywords and right in normalized_keywords
        for left, right in _build_generic_probe_pairs(query)
    )

    requested_articles = _query_article_references(query)
    provision_matches = 0
    if requested_articles:
        provision_text = _normalize_for_match(" ".join(_row_json_strings(row, "legal_provisions_json"))).replace("-", " ")
        available_articles = _query_article_references(provision_text)
        for article, unit in requested_articles:
            if (article, unit) in available_articles:
                provision_matches += 2 if unit else 1
            elif unit is None and any(candidate_article == article for candidate_article, _ in available_articles):
                provision_matches += 1

    return keyword_pairs, keyword_coverage, provision_matches


def _dedupe_and_filter_relevant_chunks(
    chunks: list[july7_rag.RagChunk],
    *,
    query: str,
    limit: int,
) -> list[july7_rag.RagChunk]:
    selected: list[july7_rag.RagChunk] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        if chunk.source_type != "interpretation":
            continue
        if chunk.document_id in seen_documents:
            continue
        seen_documents.add(chunk.document_id)
        selected.append(chunk)
    # Hydration is a presentation step, not another retrieval pass.  A full
    # interpretation contains statutory boilerplate that can mention many
    # unrelated words and otherwise push a strong candidate below a loose
    # neighbour.  Preserve the candidate rank (which used the matching chunk,
    # subject, keywords and explicit provisions), while keeping the full-text
    # overlap only as a late tie-breaker.  We deliberately retain up to six
    # candidates whenever the corpus supplied them.
    return sorted(
        selected,
        key=lambda chunk: (
            _query_precision_anchor_matches(chunk, query, subject_only=True),
            _query_subject_pair_matches(chunk, query),
            _query_precision_anchor_matches(chunk, query, subject_only=False),
            chunk.score,
            _query_pair_matches(chunk, query),
            _query_coverage(chunk, query)[1],
            _query_coverage(chunk, query)[0],
            _query_coverage(chunk, query)[3],
            _query_coverage(chunk, query)[2],
            _query_coverage(chunk, query)[4],
            _query_coverage(chunk, query)[5],
        ),
        reverse=True,
    )[:limit]


def _select_interpretation_documents(
    chunks: list[july7_rag.RagChunk],
    *,
    limit: int,
) -> list[july7_rag.RagChunk]:
    """Keep one ranked seed per document before loading complete texts."""
    selected: list[july7_rag.RagChunk] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        if chunk.source_type != "interpretation" or chunk.document_id in seen_documents:
            continue
        seen_documents.add(chunk.document_id)
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def hydrate_tax_interpretation_documents(
    chunks: list[july7_rag.RagChunk],
) -> list[july7_rag.RagChunk]:
    """Replace selected snippets with their complete, ordered interpretations.

    The July 7 retrieval code is deliberately chunk-based.  That is suitable
    for ranking, but a user selecting the interpretations-only MVP expects a
    document, not an arbitrary 1--2 kB slice that may begin mid-sentence.
    Hydration happens *after* the historical ranker has selected documents, so
    it cannot affect retrieval quality or silently route through the current
    backend.
    """
    document_ids = list(dict.fromkeys(
        str(chunk.document_id).strip() for chunk in chunks if str(chunk.document_id).strip()
    ))
    if not document_ids:
        return []

    if july7_mysql_rag.is_mysql_rag_configured():
        rows = july7_mysql_rag.fetch_rows_by_document_ids_mysql(
            document_ids,
            source_type="interpretation",
        )
    else:
        rows = july7_rag.fetch_rows_by_document_ids(
            document_ids,
            config=july7_rag.get_rag_config(),
            source_type="interpretation",
        )

    documents = july7_rag.build_document_context_from_rows(
        rows,
        ordered_document_ids=document_ids,
        seed_chunks=chunks,
    )
    document_by_id = {document.document_id: document for document in documents if document.text.strip()}

    hydrated: list[july7_rag.RagChunk] = []
    for chunk in chunks:
        document = document_by_id.get(chunk.document_id)
        if document is None:
            continue
        hydrated.append(
            replace(
                chunk,
                chunk_index=0,
                chunk_text=document.text,
                subject=document.subject,
                signature=document.signature,
                published_date=document.published_date,
                source_url=document.source_url,
                category=document.category,
                source=document.source,
                source_type=document.source_type,
                source_subtype=document.source_subtype,
                authority=document.authority,
                publication=document.publication,
                legal_state_date=document.legal_state_date,
                source_pages=document.source_pages,
                legal_provisions=document.legal_provisions,
            )
        )
    return hydrated


def _fetch_historical_sqlite_candidate_rows(
    query: str,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    """Read a bounded, untyped FTS pool from the July 7 SQLite snapshot."""
    config = july7_rag.get_rag_config()
    july7_rag.ensure_local_index_ready()
    if not config.db_path.exists():
        return []
    match_queries = _build_bounded_historical_match_queries(query, config=config)
    if not match_queries:
        return []

    candidate_limit = max(config.candidate_pool_limit, limit * 20)
    connection = july7_rag.get_connection(config.db_path)
    try:
        query_rows = [
            connection.execute(
                """
                SELECT
                    c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    d.subject, d.signature, d.published_date, d.source_url, d.category,
                    d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json,
                    d.facts_text, d.question_text, d.tax_domain,
                    d.source, d.source_type, d.source_subtype, d.authority, d.publication,
                    d.legal_state_date, d.source_pages_json,
                    bm25(chunks_fts, 1.0, 2.5, 4.0, 1.5, 2.5, 2.5, 5.0, 4.0, 3.0) AS lexical_score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                JOIN documents d ON d.document_id = c.document_id
                WHERE chunks_fts MATCH ?
                  AND d.source_type = 'interpretation'
                ORDER BY lexical_score, d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
                LIMIT ?
                """,
                (match_query, candidate_limit),
            ).fetchall()
            for match_query in match_queries
        ]
    finally:
        connection.close()

    rows: list[sqlite3.Row] = []
    seen_chunks: set[str] = set()
    seen_documents: set[str] = set()
    for rank in range(max((len(group) for group in query_rows), default=0)):
        for group in query_rows:
            if rank >= len(group):
                continue
            row = group[rank]
            chunk_id = str(row["chunk_id"])
            document_id = str(row["document_id"])
            if chunk_id in seen_chunks or document_id in seen_documents:
                continue
            seen_chunks.add(chunk_id)
            seen_documents.add(document_id)
            rows.append(row)
            if len(rows) >= candidate_limit:
                return rows
    return rows


def _generic_candidate_score(
    chunk: july7_rag.RagChunk,
    *,
    query: str,
    source_rank: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    """Score a candidate solely from lexical overlap and FTS rank."""
    (
        distinctive_document,
        tax_document,
        distinctive_subject,
        tax_subject,
        document_matches,
        subject_matches,
    ) = _query_coverage(
        chunk,
        query,
    )
    pair_matches = _query_pair_matches(chunk, query)
    subject_pair_matches = _query_subject_pair_matches(chunk, query)
    precision_subject_matches = _query_precision_anchor_matches(chunk, query, subject_only=True)
    precision_document_matches = _query_precision_anchor_matches(chunk, query, subject_only=False)
    return (
        precision_subject_matches,
        subject_pair_matches,
        precision_document_matches,
        pair_matches,
        tax_document,
        distinctive_document,
        tax_subject,
        distinctive_subject,
        document_matches,
        subject_matches,
        -source_rank,
    )


def _rank_historical_candidates(
    rows: list,
    *,
    query: str,
    limit: int,
) -> list[july7_rag.RagChunk]:
    """Rank the July 7 candidate pool without topic or source-specific boosts."""
    if not rows:
        return []
    ranked: list[tuple[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int], july7_rag.RagChunk]] = []
    for source_rank, row in enumerate(rows, start=1):
        chunk = july7_rag.row_to_rag_chunk(row, score=0.0)
        if chunk.source_type != "interpretation":
            continue
        generic_score = _generic_candidate_score(
            chunk,
            query=query,
            source_rank=source_rank,
        )
        keyword_pairs, keyword_coverage, provision_matches = _metadata_match_score(row, query=query)
        # Metadata only breaks ties between lexical neighbours: a document
        # must still have reached this pool through text search, and the score
        # below continues to include all body/subject overlap signals.
        score = (
            generic_score[0],
            generic_score[1],
            generic_score[2],
            provision_matches,
            keyword_pairs,
            keyword_coverage,
            *generic_score[3:],
        )
        numeric_score = (
            generic_score[0] * 1_000_000
            + generic_score[1] * 100_000
            + generic_score[2] * 10_000
            + generic_score[3] * 1_000
            + provision_matches * 100
            + keyword_pairs * 10
            + keyword_coverage
            + generic_score[4] * 0.1
            + generic_score[5] * 0.01
        )
        ranked.append((score, replace(chunk, score=numeric_score)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[july7_rag.RagChunk] = []
    seen_documents: set[str] = set()
    for _, chunk in ranked:
        if chunk.document_id in seen_documents:
            continue
        seen_documents.add(chunk.document_id)
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def _search_historical_sqlite(query: str, *, limit: int) -> list[july7_rag.RagChunk]:
    """Run generic candidate retrieval and ranking against the SQLite snapshot."""
    return _rank_historical_candidates(
        _fetch_historical_sqlite_candidate_rows(query, limit=limit),
        query=query,
        limit=limit,
    )


def _search_historical_mysql(query: str, *, limit: int) -> list[july7_rag.RagChunk]:
    """Run generic candidate retrieval and ranking against the MariaDB corpus."""
    _, rows = july7_mysql_rag.fetch_candidate_rows_mysql(
        query,
        effective_limit=limit,
        source_types={"interpretation"},
        enforce_query_domain=False,
        tax_domains=None,
        detection_query=query,
    )
    return _rank_historical_candidates(rows, query=query, limit=limit)


def search_tax_interpretations(query: str, *, limit: int | None = None) -> list[july7_rag.RagChunk]:
    """Return only individual tax interpretations using the July 7 ranker.

    The snapshot deliberately searches its local SQLite index directly.  This
    Production has no local SQLite corpus, so it uses the separately vendored
    July 7 MySQL retrieval implementation against the read-only corpus.  Local
    development falls back to the separately vendored July 7 SQLite path.  The
    selected chunks are then hydrated to full documents *before* fact filtering:
    a winning chunk can be the legal reasoning section, while the matching
    dental facts and the question appear elsewhere in the same interpretation.
    """
    configured_limit = int(os.getenv("JULY7_INTERPRETATIONS_RETRIEVAL_LIMIT", "6"))
    effective_limit = max(1, min(limit or configured_limit, 20))
    # The ranker sees a broader generic lexical pool; only the final output is
    # constrained to six documents.  This prevents one broad tax phrase from
    # deciding the whole result set before the full-document overlap check.
    # Select a materially wider document window than the six displayed
    # results.  A candidate is only a matching chunk; several documents can
    # share the same broad tax wording, while the decisive factual phrase may
    # live elsewhere in their full text and become visible only after
    # hydration.  This remains bounded to keep the retrieval-only request
    # predictable.
    candidate_limit = min(30, max(effective_limit * 3, 18))
    active_query_token = _active_user_query.set(query)
    try:
        if july7_mysql_rag.is_mysql_rag_configured():
            chunks = _search_historical_mysql(query, limit=candidate_limit)
        else:
            chunks = _search_historical_sqlite(query, limit=candidate_limit)
    finally:
        _active_user_query.reset(active_query_token)
    document_candidates = _select_interpretation_documents(
        chunks,
        limit=candidate_limit,
    )
    hydrated_documents = hydrate_tax_interpretation_documents(document_candidates)
    return _dedupe_and_filter_relevant_chunks(
        hydrated_documents,
        query=query,
        limit=effective_limit,
    )
