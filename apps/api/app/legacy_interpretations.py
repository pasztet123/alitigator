"""Interpretation search pinned to the July 7, 2026 retrieval revision."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.legal_institutions import InstitutionMatcher
from app.legal_institutions.schema import InstitutionDefinition
from app.legacy_july7 import rag as july7_rag
from app.legacy_july7 import mysql_rag as july7_mysql_rag
from app.tax_research import (
    AnchorSet,
    CandidateAssessment,
    ResearchUnderstanding,
    assess_candidate,
    build_anchors as _build_research_anchors,
    candidate_boolean_queries,
    understand_tax_research_question,
)


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
_active_named_institution_query: ContextVar[bool] = ContextVar(
    "active_july7_named_institution_query",
    default=False,
)
_candidate_diagnostics: ContextVar[dict[str, int]] = ContextVar(
    "july7_candidate_diagnostics",
    default={},
)


@dataclass(frozen=True)
class TaxResearchDocument:
    chunk: july7_rag.RagChunk
    assessment: CandidateAssessment

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.chunk.document_id,
            "signature": self.chunk.signature or "",
            "subject": self.chunk.subject,
            "published_date": self.chunk.published_date,
            "source_url": self.chunk.source_url,
            **self.assessment.to_dict(),
        }


@dataclass(frozen=True)
class TaxResearchSearchResult:
    question: str
    understanding: ResearchUnderstanding
    database_queries: tuple[str, ...]
    candidate_counts: Mapping[str, int]
    candidates_before_rerank: tuple[dict[str, object], ...]
    candidate_document_ids: tuple[str, ...]
    reranker_scores: tuple[dict[str, object], ...]
    validation_results: tuple[dict[str, object], ...]
    documents: tuple[TaxResearchDocument, ...]
    institution_matches: tuple[dict[str, object], ...] = ()
    locked_institution_ids: tuple[str, ...] = ()
    institution_filter_rejections: int = 0

    @property
    def chunks(self) -> list[july7_rag.RagChunk]:
        return [item.chunk for item in self.documents]

    def to_trace(self) -> dict[str, object]:
        return {
            "pipeline": "interpretations_july7",
            "original_question": self.question,
            "planner_output": self.understanding.to_dict(),
            "anchors": self.understanding.anchors.to_dict(),
            "database_queries": list(self.database_queries),
            "candidate_counts": dict(self.candidate_counts),
            "candidates_before_rerank": list(self.candidates_before_rerank),
            "candidate_document_ids": list(self.candidate_document_ids),
            "reranker_scores": list(self.reranker_scores),
            "validation_results": list(self.validation_results),
            "candidates_after_rerank": [item.to_dict() for item in self.documents],
            "final_results": [item.to_dict() for item in self.documents],
            "named_institutions": {
                "matches": list(self.institution_matches),
                "locked_institution_ids": list(self.locked_institution_ids),
                "filter_rejections": self.institution_filter_rejections,
            },
            "vector_index_available": False,
        }


def build_anchors(query: str) -> AnchorSet:
    """Public, testable safe-anchor classifier for interpretation search."""

    return _build_research_anchors(query)


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
    """Build selective probes from safe concepts and statutory hypotheses.

    Unlike the previous prefix grouping, an ambiguous grammatical stem cannot
    become the sole required term.  The legal hypothesis is generated from the
    question, never from a target document or a signature.
    """
    # A named-institution channel is already a bounded, editorially verified
    # phrase or statutory hint.  Do not append the generic planner's broad
    # fallbacks (for example ``PIT art. 22``) to it: that would turn a precise
    # expansion-relief lookup back into a high-volume business-cost search.
    if _active_named_institution_query.get():
        return july7_mysql_rag._build_mysql_candidate_queries(query)
    understanding = understand_tax_research_question(_active_user_query.get() or query)
    queries = candidate_boolean_queries(understanding)
    return queries or july7_mysql_rag._build_mysql_candidate_queries(query)


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
    understanding = understand_tax_research_question(query)
    classified = [
        (
            chunk,
            assess_candidate(
                understanding,
                subject=chunk.subject,
                text=chunk.chunk_text,
                provisions=chunk.legal_provisions,
            ),
        )
        for chunk in selected
    ]
    # The compatibility API only returns useful main-lane results.  It never
    # pads the list with an orthogonal tax mechanism merely to reach ``limit``.
    usable = [item for item in classified if not item[1].reject and item[1].relation != "context_only"]
    return [
        replace(chunk, score=assessment.score)
        for chunk, assessment in sorted(
            usable,
            key=lambda item: (item[1].score, item[0].score, item[0].document_id),
            reverse=True,
        )[:limit]
    ]


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


def score_candidate_row(
    row: object,
    *,
    question: str,
    understanding: ResearchUnderstanding | None = None,
) -> CandidateAssessment:
    """Score one chunk with bounded, named, legal-research components."""

    chunk = july7_rag.row_to_rag_chunk(row, score=0.0)
    return assess_candidate(
        understanding or understand_tax_research_question(question),
        subject=chunk.subject,
        text=chunk.chunk_text,
        provisions=_row_json_strings(row, "legal_provisions_json"),
        tax_domain=str(_row_value(row, "tax_domain") or ""),
    )


def _institution_trace_match(match: object) -> dict[str, object]:
    """Serialize only the auditable recognition facts for the July 7 trace."""

    return {
        "institution_id": getattr(match, "institution_id", ""),
        "canonical_name": getattr(match, "canonical_name", ""),
        "match_type": getattr(match, "match_type", ""),
        "confidence": getattr(match, "confidence", 0.0),
        "matched_text": getattr(match, "matched_text", ""),
        "locked": bool(getattr(match, "locked", False)),
    }


def _locked_institution_definitions(
    query: str,
) -> tuple[InstitutionMatcher, tuple[dict[str, object], ...], tuple[InstitutionDefinition, ...]]:
    """Recognise named institutions before the historical generic ranker.

    The July 7 profile intentionally retains its own retrieval implementation.
    Recognition is nevertheless shared with V2: an explicit active institution
    is a hard relevance constraint, never just another lexical boost.
    """

    matcher = InstitutionMatcher()
    match_result = matcher.match(query)
    locked_ids = tuple(item.institution_id for item in match_result.matches if item.locked)
    return (
        matcher,
        tuple(_institution_trace_match(item) for item in match_result.matches),
        matcher.definitions_for(locked_ids),
    )


def _institution_candidate_queries(definitions: Sequence[InstitutionDefinition]) -> tuple[str, ...]:
    """Use one selective, canonical FTS channel per explicit institution.

    The historical MariaDB adapter expands one FTS request into its own bounded
    lexical variants.  Sending aliases and provisions as separate requests
    multiplies the read cost and can exhaust the interactive timeout without
    improving the authority gate.  The full-document marker check below still
    considers aliases and verified provisions when judging a fetched document.
    """

    return tuple(dict.fromkeys(definition.canonical_name for definition in definitions))


def _assess_locked_institution_candidate(
    *,
    matcher: InstitutionMatcher,
    definitions: Sequence[InstitutionDefinition],
    chunk: july7_rag.RagChunk,
) -> CandidateAssessment:
    """Accept only documents carrying an explicit marker of a locked institution."""

    for definition in definitions:
        markers = matcher.document_markers(
            definition,
            text=chunk.chunk_text,
            metadata={
                "subject": chunk.subject,
                "legal_provisions": chunk.legal_provisions,
            },
        )
        if markers:
            # A named phrase in the editorial subject is materially stronger
            # than an incidental mention in the legal reasoning of a long,
            # multi-relief interpretation.  Both remain eligible, but the
            # former must be shown first.
            subject_markers = matcher.document_markers(
                definition,
                text=chunk.subject,
            )
            provision_markers = matcher.document_markers(
                definition,
                text="",
                metadata={"legal_provisions": chunk.legal_provisions},
            )
            if subject_markers:
                score = 100.0
                relation = "direct"
            elif provision_markers:
                score = 85.0
                relation = "direct"
            else:
                score = 70.0
                relation = "strong_analogy"
            return CandidateAssessment(
                relation=relation,
                reject=False,
                reason="Zgodność z rozpoznaną instytucją prawa podatkowego.",
                document_mechanism=definition.institution_id,
                material_differences=(),
                score=score,
                components={
                    "named_institution_marker": float(len(markers)),
                    "named_institution_subject_marker": float(len(subject_markers)),
                    "named_institution_provision_marker": float(len(provision_markers)),
                    "named_institution_gate": 1.0,
                },
            )
    return CandidateAssessment(
        relation="different_mechanism",
        reject=True,
        reason="Brak markera rozpoznanej instytucji prawa podatkowego.",
        document_mechanism="unknown",
        material_differences=("missing_locked_institution_markers",),
        score=0.0,
        components={"named_institution_gate": 0.0},
    )


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
    institution_matcher: InstitutionMatcher | None = None,
    locked_definitions: Sequence[InstitutionDefinition] = (),
) -> list[july7_rag.RagChunk]:
    """Rank a document-diverse candidate pool on a bounded 0--100 scale."""
    if not rows:
        return []
    understanding = understand_tax_research_question(query)
    ranked: list[tuple[tuple[float, int, int], july7_rag.RagChunk]] = []
    for source_rank, row in enumerate(rows, start=1):
        chunk = july7_rag.row_to_rag_chunk(row, score=0.0)
        if chunk.source_type != "interpretation":
            continue
        if institution_matcher and locked_definitions:
            assessment = _assess_locked_institution_candidate(
                matcher=institution_matcher,
                definitions=locked_definitions,
                chunk=chunk,
            )
        else:
            assessment = score_candidate_row(row, question=query, understanding=understanding)
        # Database rank only breaks ties.  A title prefix never receives a
        # separate, unbounded privilege over the legal/factual match.
        ranked.append(
            (
                (assessment.score, -int(assessment.reject), -source_rank),
                replace(chunk, score=assessment.score),
            )
        )

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


def _metadata_candidate_pairs(understanding: ResearchUnderstanding) -> list[tuple[str, str]]:
    """Reuse the bounded Boolean probes for the document-title channel."""

    return [
        match.groups()
        for probe in candidate_boolean_queries(understanding, max_queries=6)
        for match in re.finditer(r"\+([a-z0-9]+)\*\s+\+([a-z0-9]+)\*", probe)
    ]


def _candidate_provision_articles(understanding: ResearchUnderstanding) -> list[str]:
    """Extract article identifiers from the generic planner output."""

    return list(dict.fromkeys(
        match.group(1).lower()
        for provision in understanding.candidate_provisions
        for match in re.finditer(r"\bart\.?\s*(\d+[a-z]{0,3})\b", provision, re.IGNORECASE)
    ))


def _search_historical_sqlite(
    query: str,
    *,
    limit: int,
    supplemental_queries: Sequence[str] = (),
    institution_matcher: InstitutionMatcher | None = None,
    locked_definitions: Sequence[InstitutionDefinition] = (),
) -> list[july7_rag.RagChunk]:
    """Run generic candidate retrieval and ranking against the SQLite snapshot."""
    rows: list[sqlite3.Row] = []
    seen_chunks: set[str] = set()
    retrieval_queries = supplemental_queries if locked_definitions and supplemental_queries else (query,)
    for retrieval_query in dict.fromkeys(retrieval_queries):
        query_token = _active_user_query.set(retrieval_query)
        named_query_token = _active_named_institution_query.set(retrieval_query != query)
        try:
            candidate_rows = _fetch_historical_sqlite_candidate_rows(retrieval_query, limit=limit)
        finally:
            _active_named_institution_query.reset(named_query_token)
            _active_user_query.reset(query_token)
        for row in candidate_rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in seen_chunks:
                rows.append(row)
                seen_chunks.add(chunk_id)
    _candidate_diagnostics.set({"raw": len(rows), "deduplicated": len({str(row["document_id"]) for row in rows})})
    return _rank_historical_candidates(
        rows,
        query=query,
        limit=limit,
        institution_matcher=institution_matcher,
        locked_definitions=locked_definitions,
    )


def _search_historical_mysql(
    query: str,
    *,
    limit: int,
    supplemental_queries: Sequence[str] = (),
    institution_matcher: InstitutionMatcher | None = None,
    locked_definitions: Sequence[InstitutionDefinition] = (),
) -> list[july7_rag.RagChunk]:
    """Run generic candidate retrieval and ranking against the MariaDB corpus."""
    fts_rows: list[object] = []
    # Once an explicit named institution is locked, its deterministic channels
    # are the recall contract.  Running the broad natural-language probe too
    # would reintroduce generic "business expense" neighbours and needlessly
    # scan a much larger FTS result set.
    retrieval_queries = supplemental_queries if locked_definitions and supplemental_queries else (query,)
    for retrieval_query in dict.fromkeys(retrieval_queries):
        # The vendored July adapter reads the ContextVar when it builds its
        # Boolean probes.  Set it per deterministic query so that a canonical
        # name such as "ulga na ekspansję" genuinely reaches MariaDB instead
        # of being silently replaced by the broad natural-language question.
        query_token = _active_user_query.set(retrieval_query)
        try:
            _, rows = july7_mysql_rag.fetch_candidate_rows_mysql(
                retrieval_query,
                # The adapter retains its own bounded recall window.  We merge
                # those rows before document-level validation, never return a
                # supplemental probe directly to the user.
                effective_limit=min(limit, 6),
                source_types={"interpretation"},
                enforce_query_domain=False,
                tax_domains=None,
                detection_query=retrieval_query,
            )
        finally:
            _active_user_query.reset(query_token)
        fts_rows.extend(rows)
    understanding = understand_tax_research_question(query)
    # Metadata is a distinct candidate-generation channel, not a score
    # override.  It makes concise editorial subjects and curated keywords
    # available to the same generic reranker as the FTS snippets.
    if locked_definitions:
        # The canonical named-institution FTS probe is the retrieval contract.
        # Generic title/provision probes would reintroduce unrelated documents
        # and create additional MariaDB scans before the marker gate runs.
        title_rows: list[object] = []
        provision_rows: list[object] = []
    else:
        metadata_pairs = _metadata_candidate_pairs(understanding)
        title_rows = july7_mysql_rag.fetch_subject_candidate_rows_mysql(
            metadata_pairs,
            source_type="interpretation",
            limit_per_pair=80,
        )
        provision_rows = july7_mysql_rag.fetch_provision_candidate_rows_mysql(
            tax_domain=understanding.tax_domain,
            articles=_candidate_provision_articles(understanding),
            source_type="interpretation",
        )
    seen_chunks: set[str] = set()
    merged_rows: list[object] = []
    for row in [*fts_rows, *title_rows, *provision_rows]:
        chunk_id = str(_row_value(row, "chunk_id") or "")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        merged_rows.append(row)
    _candidate_diagnostics.set({
        "raw": len(fts_rows) + len(title_rows) + len(provision_rows),
        "deduplicated": len({str(_row_value(row, "document_id") or "") for row in merged_rows}),
        "fts": len(fts_rows),
        "title_metadata": len(title_rows),
        "provision_metadata": len(provision_rows),
    })
    return _rank_historical_candidates(
        merged_rows,
        query=query,
        limit=limit,
        institution_matcher=institution_matcher,
        locked_definitions=locked_definitions,
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
    return search_tax_interpretations_with_trace(query, limit=limit).chunks


def search_tax_interpretations_with_trace(
    query: str,
    *,
    limit: int | None = None,
) -> TaxResearchSearchResult:
    """Retrieve and classify interpretations with an auditable document trace."""

    configured_limit = int(os.getenv("JULY7_INTERPRETATIONS_RETRIEVAL_LIMIT", "6"))
    effective_limit = max(1, min(limit or configured_limit, 20))
    candidate_limit = max(100, int(os.getenv("TAX_RESEARCH_CANDIDATE_POOL_LIMIT", "120")))
    candidate_limit = min(candidate_limit, 160)
    understanding = understand_tax_research_question(query)
    institution_matcher, institution_matches, locked_definitions = _locked_institution_definitions(query)
    supplemental_queries = _institution_candidate_queries(locked_definitions)
    diagnostics_token = _candidate_diagnostics.set({})
    active_query_token = _active_user_query.set(query)
    try:
        if july7_mysql_rag.is_mysql_rag_configured():
            if locked_definitions:
                chunks = _search_historical_mysql(
                    query,
                    limit=candidate_limit,
                    supplemental_queries=supplemental_queries,
                    institution_matcher=institution_matcher,
                    locked_definitions=locked_definitions,
                )
            else:
                chunks = _search_historical_mysql(query, limit=candidate_limit)
        else:
            if locked_definitions:
                chunks = _search_historical_sqlite(
                    query,
                    limit=candidate_limit,
                    supplemental_queries=supplemental_queries,
                    institution_matcher=institution_matcher,
                    locked_definitions=locked_definitions,
                )
            else:
                chunks = _search_historical_sqlite(query, limit=candidate_limit)
    finally:
        _active_user_query.reset(active_query_token)
    candidate_diagnostics = _candidate_diagnostics.get()
    _candidate_diagnostics.reset(diagnostics_token)
    document_candidates = _select_interpretation_documents(chunks, limit=candidate_limit)
    # The expensive full-document validator sees the top 50 unique documents:
    # this is the audit-sized window required for legal-mechanism validation,
    # while the preceding candidate pool still holds at least 100 documents.
    validation_candidates = document_candidates[:50]
    hydrated_documents = hydrate_tax_interpretation_documents(validation_candidates)
    classified: list[TaxResearchDocument] = []
    for chunk in hydrated_documents:
        assessment = (
            _assess_locked_institution_candidate(
                matcher=institution_matcher,
                definitions=locked_definitions,
                chunk=chunk,
            )
            if locked_definitions
            else assess_candidate(
                understanding,
                subject=chunk.subject,
                text=chunk.chunk_text,
                provisions=chunk.legal_provisions,
            )
        )
        classified.append(TaxResearchDocument(replace(chunk, score=assessment.score), assessment))
    classified.sort(key=lambda item: (item.assessment.score, item.chunk.document_id), reverse=True)
    main = [item for item in classified if not item.assessment.reject and item.assessment.relation in {"direct", "strong_analogy"}]
    context = [item for item in classified if not item.assessment.reject and item.assessment.relation == "context_only"]
    secondary = [item for item in classified if item.assessment.relation == "different_mechanism"]
    documents = tuple(
        main[:effective_limit]
        if locked_definitions
        else [
            *main[:effective_limit],
            *context[: max(0, effective_limit - len(main))],
            *secondary[: max(0, effective_limit - len(main) - len(context))],
        ]
    )
    database_queries = (
        tuple(f"named_institution_canonical:{item}" for item in supplemental_queries)
        if locked_definitions
        else tuple([
            *_build_bounded_historical_mysql_queries(query),
            *(f"title_metadata:{left}+{right}" for left, right in _metadata_candidate_pairs(understanding)),
            *(f"provision_metadata:{understanding.tax_domain}:art.{article}" for article in _candidate_provision_articles(understanding)),
        ])
    )
    return TaxResearchSearchResult(
        question=query,
        understanding=understanding,
        database_queries=database_queries,
        candidate_counts={
            "raw": int(candidate_diagnostics.get("raw", len(chunks))),
            "deduplicated": int(candidate_diagnostics.get("deduplicated", len(document_candidates))),
            "fts": int(candidate_diagnostics.get("fts", 0)),
            "title_metadata": int(candidate_diagnostics.get("title_metadata", 0)),
            "provision_metadata": int(candidate_diagnostics.get("provision_metadata", 0)),
            "validated": len(hydrated_documents),
            "after_rerank": len(classified),
            "named_institution_filtered": sum(item.assessment.reject for item in classified) if locked_definitions else 0,
        },
        candidates_before_rerank=tuple(
            {
                "document_id": chunk.document_id,
                "signature": chunk.signature or "",
                "subject": chunk.subject,
                "score": chunk.score,
            }
            for chunk in document_candidates[:30]
        ),
        candidate_document_ids=tuple(chunk.signature or chunk.document_id for chunk in document_candidates),
        reranker_scores=tuple(
            {
                "document_id": item.chunk.document_id,
                "signature": item.chunk.signature or "",
                "score": item.assessment.score,
                "components": dict(item.assessment.components),
            }
            for item in classified[:50]
        ),
        validation_results=tuple(item.to_dict() for item in classified[:50]),
        documents=documents,
        institution_matches=institution_matches,
        locked_institution_ids=tuple(item.institution_id for item in locked_definitions),
        institution_filter_rejections=sum(item.assessment.reject for item in classified) if locked_definitions else 0,
    )


def get_research_relation(chunk: july7_rag.RagChunk, query: str = "") -> str:
    """Compatibility helper for diagnostics; detailed search keeps full data."""

    if not query:
        return "unclassified"
    return assess_candidate(
        understand_tax_research_question(query),
        subject=chunk.subject,
        text=chunk.chunk_text,
        provisions=chunk.legal_provisions,
    ).relation
