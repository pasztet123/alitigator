"""Interpretation search pinned to the July 7, 2026 retrieval revision."""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import replace

from app.legacy_july7 import rag as july7_rag


JULY7_RETRIEVAL_COMMIT = "6a23c08"
JULY7_RETRIEVAL_DATE = "2026-07-07"


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

_HISTORICAL_QUERY_STOPWORDS = {
    "albo", "bardzo", "będzie", "czy", "dla", "jest", "jako", "jeżeli",
    "kiedy", "która", "który", "mają", "może", "oraz", "podlega", "przez",
    "sprzedaż", "tego", "tych", "ustawy", "wtedy", "wyniku", "został",
}
_active_user_query: ContextVar[str | None] = ContextVar("active_july7_user_query", default=None)


def _build_bounded_historical_match_queries(query: str, *, config=None) -> list[str]:
    """Keep July's FTS query selective on today's much larger corpus.

    In July the corpus was small enough for a broad OR query.  Running that
    query against the current 10 GB index produces hundreds of thousands of
    candidates before the historical ranker gets a chance to score them.
    """
    candidate_query = _active_user_query.get() or query
    tokens = [
        token.lower()
        for token in july7_rag.QUERY_TOKEN_RE.findall(candidate_query)
        if len(token) >= 4 and token.lower() not in _HISTORICAL_QUERY_STOPWORDS
    ]
    distinctive_tokens = list(dict.fromkeys(tokens))[:6]
    if len(distinctive_tokens) >= 2:
        return [" AND ".join(f'"{token}"*' for token in distinctive_tokens)]
    return july7_rag._build_candidate_match_queries(query, config=config)


july7_rag._build_candidate_match_queries = july7_rag.build_candidate_match_queries
july7_rag.build_candidate_match_queries = _build_bounded_historical_match_queries


def search_tax_interpretations(query: str, *, limit: int | None = None) -> list[july7_rag.RagChunk]:
    """Return only individual tax interpretations using the July 7 ranker.

    The snapshot deliberately searches its local SQLite index directly.  This
    avoids the current backend router and preserves the July 7 query expansion,
    axis decomposition, scoring and within-document reranking.  Its FTS recall
    is narrowed only to keep the historical query practical on today's corpus.
    """
    configured_limit = int(os.getenv("JULY7_INTERPRETATIONS_RETRIEVAL_LIMIT", "8"))
    effective_limit = max(1, min(limit or configured_limit, 20))
    tax_domains = july7_rag.resolve_statute_tax_domains(query)
    active_query_token = _active_user_query.set(query)
    try:
        chunks = july7_rag.search_chunks(
            query,
            limit=effective_limit,
            source_types={"interpretation"},
            enforce_query_domain=bool(tax_domains),
            tax_domains=tax_domains or None,
        )
    finally:
        _active_user_query.reset(active_query_token)
    return [chunk for chunk in chunks if chunk.source_type == "interpretation"]
