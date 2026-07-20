"""Interpretation search pinned to the July 7, 2026 retrieval revision."""

from __future__ import annotations

import os
import re
import unicodedata
from contextvars import ContextVar
from dataclasses import replace

from app.legacy_july7 import rag as july7_rag
from app.legacy_july7 import mysql_rag as july7_mysql_rag


JULY7_RETRIEVAL_COMMIT = "6a23c08"
JULY7_RETRIEVAL_DATE = "2026-07-07"


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
    """Keep the retrieval-only request bounded; opt in to model reranking."""
    config = _get_july7_rag_config()
    cross_encoder_enabled = os.getenv(
        "JULY7_INTERPRETATIONS_CROSS_ENCODER_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes"}
    return replace(config, cross_encoder_enabled=cross_encoder_enabled)


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
        "albo", "bardzo", "będzie", "czy", "dla", "jest", "jako", "jeżeli", "przy",
        "kiedy", "która", "który", "mają", "może", "oraz", "podlega", "przez",
        "sprzedaż", "tego", "tych", "ustawy", "wtedy", "wyniku", "został",
        "interpretacji", "interpretacje", "interpretację", "dotyczących", "dotyczące",
        "poszukaj", "pokaż", "znajdź", "proszę", "kosztem", "uzyskania", "przychodu",
    }
)


def _query_anchor_stems(query: str) -> list[str]:
    """Extract material facts, never the request to search interpretations."""
    stems: list[str] = []
    for token in july7_rag.QUERY_TOKEN_RE.findall(query):
        normalized = _normalize_for_match(token)
        if (
            (len(normalized) < 4 and normalized not in {"vat", "pit", "cit", "pcc"})
            or normalized in _HISTORICAL_QUERY_STOPWORDS
        ):
            continue
        if normalized.startswith("implant"):
            stem = "implant"
        elif normalized.startswith("protez"):
            stem = "protez"
        elif normalized.startswith("zeb"):
            stem = "zęb"
        else:
            stem = normalized
        if stem not in stems:
            stems.append(stem)
    return stems


def _build_bounded_historical_match_queries(query: str, *, config=None) -> list[str]:
    """Keep July's SQLite FTS query selective on today's much larger corpus.

    In July the corpus was small enough for a broad OR query.  Running that
    query against the current 10 GB index produces hundreds of thousands of
    candidates before the historical ranker gets a chance to score them.
    """
    candidate_query = _active_user_query.get() or query
    anchors = _query_anchor_stems(candidate_query)
    if anchors:
        return [" OR ".join(f'"{anchor}"*' for anchor in anchors[:6])]
    return july7_rag._build_candidate_match_queries(query, config=config)


july7_rag._build_candidate_match_queries = july7_rag.build_candidate_match_queries
july7_rag.build_candidate_match_queries = _build_bounded_historical_match_queries


def _build_bounded_historical_mysql_queries(query: str) -> list[str]:
    """Use precise Boolean FULLTEXT probes with the July 7 MySQL ranker."""
    anchors = _query_anchor_stems(_active_user_query.get() or query)
    if not anchors:
        return july7_mysql_rag._build_mysql_candidate_queries(query)

    dental = "zęb" in anchors
    implant_or_prosthesis = [item for item in anchors if item in {"implant", "protez"}]
    if dental and implant_or_prosthesis:
        normalized_query = _normalize_for_match(_active_user_query.get() or query)
        asks_for_tax_cost = all(term in normalized_query for term in ("koszt", "uzyskan", "przychod"))
        cost_terms = "+koszt* +uzyskan* +przychod*" if asks_for_tax_cost else ""

        # A literal "implant + tooth" probe found the exact document, but
        # missed equally useful dental-cost interpretations whose facts use
        # "proteza" rather than "implant".  This expands recall only within
        # the same dental concept and only after requiring the tax-cost issue.
        # The July 7 ranker still orders the combined candidate pool.
        dental_concepts = [*implant_or_prosthesis]
        if "implant" in dental_concepts and "protez" not in dental_concepts:
            dental_concepts.append("protez")
        return list(dict.fromkeys(
            f"+{item}* +zęb* {cost_terms}".strip()
            for item in dental_concepts
        ))
    return [" ".join(f"+{item}*" for item in anchors[:3])]


july7_mysql_rag._build_mysql_candidate_queries = july7_mysql_rag.build_mysql_candidate_queries
july7_mysql_rag.build_mysql_candidate_queries = _build_bounded_historical_mysql_queries


def _relevance_groups(query: str) -> list[tuple[str, ...]]:
    normalized_query = _normalize_for_match(query)
    groups: list[tuple[str, ...]] = []
    if re.search(r"\bimplant\w*|\bprotez\w*", normalized_query):
        groups.append(("implant", "protez"))
    if re.search(r"\b(zeb\w*|dentyst\w*|stomatolog\w*|protety\w*)", normalized_query):
        groups.append(("zeb", "dentyst", "stomatolog", "protety"))
    if groups:
        return groups

    anchors = _query_anchor_stems(query)
    return [(anchor,) for anchor in anchors[:2]]


def _chunk_matches_query_facts(chunk: july7_rag.RagChunk, query: str) -> bool:
    source_text = _normalize_for_match(" ".join((chunk.subject or "", chunk.chunk_text or "")))
    normalized_query = _normalize_for_match(query)
    asks_for_tax_cost = all(term in normalized_query for term in ("koszt", "uzyskan", "przychod"))
    has_tax_cost = bool(re.search(r"koszt\w*.{0,48}uzyskan\w*.{0,48}przychod\w*", source_text))
    has_related_dental_relief = bool(re.search(r"ulg\w*\s+rehabilit", source_text))
    return (
        all(any(stem in source_text for stem in group) for group in _relevance_groups(query))
        # Dental interpretations often concern the same individual expense
        # under the rehabilitation relief rather than business KUP.  Keep
        # those as clearly related authorities, but never admit a merely
        # topical dental/VAT result.
        and (not asks_for_tax_cost or has_tax_cost or has_related_dental_relief)
    )


def _direct_cost_match_rank(chunk: july7_rag.RagChunk) -> int:
    """Put the exact KUP issue ahead of analogous relief interpretations."""
    subject = _normalize_for_match(chunk.subject or "")
    return 0 if re.search(r"koszt\w*.{0,48}uzyskan\w*.{0,48}przychod\w*", subject) else 1


def _dedupe_and_filter_relevant_chunks(
    chunks: list[july7_rag.RagChunk],
    *,
    query: str,
    limit: int,
) -> list[july7_rag.RagChunk]:
    selected: list[july7_rag.RagChunk] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        if chunk.source_type != "interpretation" or not _chunk_matches_query_facts(chunk, query):
            continue
        if chunk.document_id in seen_documents:
            continue
        seen_documents.add(chunk.document_id)
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return sorted(selected, key=_direct_cost_match_rank)


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


def _search_historical_sqlite(query: str, *, limit: int) -> list[july7_rag.RagChunk]:
    """Run the snapshot locally without its public backend router."""
    axis_chunks, axes = july7_rag._search_chunks_by_legal_axes(
        query,
        limit=limit,
        source_types={"interpretation"},
        enforce_query_domain=False,
        tax_domains=None,
    )
    if axis_chunks:
        return axis_chunks
    fallback_query = axes[0].query if len(axes) == 1 else query
    return july7_rag._search_chunks_single_query(
        fallback_query,
        limit=limit,
        source_types={"interpretation"},
        enforce_query_domain=False,
        tax_domains=None,
    )


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
    configured_limit = int(os.getenv("JULY7_INTERPRETATIONS_RETRIEVAL_LIMIT", "5"))
    effective_limit = max(1, min(limit or configured_limit, 20))
    active_query_token = _active_user_query.set(query)
    try:
        candidate_limit = min(48, max(effective_limit * 4, 20))
        if july7_mysql_rag.is_mysql_rag_configured():
            chunks = july7_mysql_rag.search_chunks_mysql(
                query,
                limit=candidate_limit,
                source_types={"interpretation"},
                enforce_query_domain=False,
                tax_domains=None,
            )
        else:
            chunks = _search_historical_sqlite(query, limit=candidate_limit)
    finally:
        _active_user_query.reset(active_query_token)
    document_candidates = _select_interpretation_documents(
        chunks,
        limit=max(effective_limit * 3, 15),
    )
    hydrated_documents = hydrate_tax_interpretation_documents(document_candidates)
    return _dedupe_and_filter_relevant_chunks(
        hydrated_documents,
        query=query,
        limit=effective_limit,
    )
