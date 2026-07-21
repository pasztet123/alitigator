from __future__ import annotations

import asyncio
import inspect
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .embeddings import EmbeddingHit, VersionedEmbeddingIndex
from .provision_graph import ProvisionGraph, ProvisionUnit
from .schemas import (
    LegalIssue,
    LegalResearchPlan,
    MissingPrimaryRequest,
    QueryFamily,
    RerankScore,
)
from app.tax_research import (
    TAX_RESEARCH_MECHANISMS,
    assess_candidate,
    research_understanding_from_fields,
)
from app.legal_institutions import InstitutionMatcher
from .document_validation import build_document_card, build_question_card, evaluate_document_relevance


PRIMARY_SOURCE_TYPES = frozenset({"statute", "regulation", "tax_treaty"})
AUTHORITY_SOURCE_TYPES = frozenset(
    {
        "interpretation",
        "general_interpretation",
        "guidance",
        "judgment",
        "resolution",
    }
)
_WORD_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", re.UNICODE)
_PROVISION_REFERENCE_RE = re.compile(
    r"\bart\.\s*\d+[a-zA-Z]*"
    r"(?:\s*(?:ust\.\s*\d+[a-zA-Z]*|§\s*\d+[a-zA-Z]*))?"
    r"(?:\s*pkt\s*\d+[a-zA-Z]*)?"
    r"(?:\s*lit\.\s*[a-zA-Z])?",
    re.IGNORECASE,
)
_RELATIVE_PROVISION_REFERENCE_RE = re.compile(
    r"\bust\.\s*\d+[a-zA-Z]*"
    r"(?:\s*pkt\s*\d+[a-zA-Z]*)?"
    r"(?:\s*lit\.\s*[a-zA-Z])?",
    re.IGNORECASE,
)
_ARTICLE_ONLY_RE = re.compile(r"\bart\.\s*(\d+[a-zA-Z]*)", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    text: str
    source_type: str
    document_id: str = ""
    chunk_id: str = ""
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    backend: str = ""
    query_families: tuple[str, ...] = ()
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    component_scores: Mapping[str, float] = field(default_factory=dict)
    positive_reasons: tuple[str, ...] = ()
    negative_reasons: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_type": self.source_type,
            "score": self.score,
            "metadata": dict(self.metadata),
            "backend": self.backend,
            "query_families": list(self.query_families),
            "channel_ranks": dict(self.channel_ranks),
            "component_scores": dict(self.component_scores),
            "positive_reasons": list(self.positive_reasons),
            "negative_reasons": list(self.negative_reasons),
        }


@runtime_checkable
class RetrievalBackend(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> Sequence[RetrievalCandidate | Mapping[str, Any] | Any]:
        ...


# Short alias used by callers that treat the backend as a candidate source.
CandidateBackend = RetrievalBackend


class LegacySearchBackendAdapter:
    """Narrow adapter around the old chunk search, never its planning rules."""

    trace_marker = "legacy_backend_adapter"

    def __init__(self, search_fn: Any = None) -> None:
        self._search_fn = search_fn

    async def search(
        self,
        query: str,
        *,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[RetrievalCandidate]:
        search_fn = self._search_fn
        if search_fn is None:
            # Kept lazy so importing v2 does not initialize the legacy backend.
            from app.rag import search_chunks

            search_fn = search_chunks
        tax_domains = {
            str(value)
            for value in _as_values(metadata_filters.get("tax_domains"))
            if str(value)
        }
        kwargs = {
            "limit": limit,
            "source_types": set(source_types),
            "enforce_query_domain": bool(tax_domains),
            "tax_domains": tax_domains or None,
        }
        try:
            if inspect.iscoroutinefunction(search_fn):
                raw = search_fn(query, **kwargs)
            else:
                raw = await asyncio.to_thread(search_fn, query, **kwargs)
        except TypeError:
            # Small fakes often expose only the core portable parameters.
            fallback_kwargs = {"limit": limit, "source_types": set(source_types)}
            if inspect.iscoroutinefunction(search_fn):
                raw = search_fn(query, **fallback_kwargs)
            else:
                raw = await asyncio.to_thread(search_fn, query, **fallback_kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
        return [
            _coerce_candidate(
                item,
                backend=self.trace_marker,
                rank=rank,
            )
            for rank, item in enumerate(raw or (), start=1)
        ]

    async def search_chunks(self, *args: Any, **kwargs: Any) -> list[RetrievalCandidate]:
        return await self.search(*args, **kwargs)


# Descriptive alias retained for dependency injection configuration.
LegacyBackendAdapter = LegacySearchBackendAdapter


@dataclass(frozen=True)
class LaneResult:
    issue_id: str
    lane: str
    query_families: tuple[QueryFamily, ...]
    candidates: tuple[RetrievalCandidate, ...]
    candidate_count_before_rerank: int
    trace: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LegalRetrievalResult:
    primary_law: tuple[LaneResult, ...]
    authorities: tuple[LaneResult, ...]
    trace: tuple[dict[str, Any], ...]

    @property
    def primary_results(self) -> tuple[LaneResult, ...]:
        return self.primary_law

    @property
    def authority_results(self) -> tuple[LaneResult, ...]:
        return self.authorities


RetrievalResult = LegalRetrievalResult


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_limit_per_query: int = 30
    vector_limit_per_query: int = 30
    selected_limit_per_issue: int = 10
    rrf_k: int = 60
    require_vector_index: bool = False
    graph_depth: int = 2
    lane_concurrency: int = 2


class TransparentLegalReranker:
    """Auditable metadata/fact reranker; every score has named components."""

    def __init__(self, institution_matcher: InstitutionMatcher | None = None) -> None:
        self.institution_matcher = institution_matcher or InstitutionMatcher()

    def score(
        self,
        issue: LegalIssue,
        candidate: RetrievalCandidate,
        *,
        target_date: Optional[str],
    ) -> RerankScore:
        text = f"{candidate.text} {_metadata_text(candidate.metadata)}".casefold()
        explicit_references = _issue_explicit_references(issue)
        candidate_references = _candidate_references(candidate)
        assessment = None
        if issue.legal_mechanism in TAX_RESEARCH_MECHANISMS:
            research_understanding = research_understanding_from_fields(
                tax_domain=(issue.tax_domains[0] if issue.tax_domains else ""),
                legal_mechanism=issue.legal_mechanism,
                candidate_provisions=issue.possible_provision_concepts,
                material_concepts=[*issue.transactions, *issue.payments, *issue.positive_fact_constraints],
                negative_concepts=issue.negative_fact_constraints,
            )
            raw_provisions = candidate.metadata.get("legal_provisions") or []
            if isinstance(raw_provisions, str):
                raw_provisions = [raw_provisions]
            assessment = assess_candidate(
                research_understanding,
                subject=str(candidate.metadata.get("subject") or ""),
                text=candidate.text,
                provisions=[str(value) for value in raw_provisions],
                tax_domain=(str((candidate.metadata.get("tax_domains") or [""])[0]) if candidate.metadata.get("tax_domains") else ""),
            )
        locked_marker_ids: list[str] = []
        for definition in self.institution_matcher.definitions_for(issue.locked_institution_ids):
            if self.institution_matcher.document_markers(
                definition,
                text=candidate.text,
                metadata=candidate.metadata,
            ):
                locked_marker_ids.append(definition.institution_id)
        components: dict[str, float] = {
            "fusion": max(0.0, candidate.score),
            "source_type": 1.0
            if candidate.source_type in set(issue.requested_source_types)
            else 0.5,
            "tax_domain": _field_overlap(issue.tax_domains, candidate.metadata, "tax_domains", text),
            "taxpayer_role": _text_coverage(issue.taxpayer_roles, text),
            "transaction": _text_coverage(issue.transactions, text),
            "payment": _text_coverage(issue.payments, text),
            "jurisdiction": _text_coverage(issue.jurisdictions, text),
            "positive_constraints": _text_coverage(issue.positive_fact_constraints, text),
            "provision_concepts": _text_coverage(issue.possible_provision_concepts, text),
            "temporal": 1.0 if _effective_on(candidate.metadata, target_date) else 0.0,
            # A declared provision is a hard retrieval target, not just a
            # keyword.  This prevents dozens of fragments of a general
            # article (notably CIT art. 28b) from pushing art. 26(2e), 2g and
            # 7a–7c out of the selected evidence set.
            "explicit_provision": max(
                (
                    _reference_match_specificity(required, found)
                    for required in explicit_references
                    for found in candidate_references
                ),
                default=0.0,
            ),
            "research_mechanism": (assessment.components["mechanism_match"] / 25.0) if assessment else 0.0,
            "research_transaction": (assessment.components["expense_or_transaction_match"] / 18.0) if assessment else 0.0,
            "research_wrong_neighbor": -1.0 if assessment and assessment.reject else 0.0,
            # A named-institution marker is evidence that a document retrieved
            # by a deterministic channel actually concerns that institution.
            # It is a material ranking signal, not a stand-alone conclusion.
            "locked_institution_marker": 1.0 if locked_marker_ids else 0.0,
        }
        negative_hits = [
            value for value in issue.negative_fact_constraints if _phrase_present(value, text)
        ]
        components["negative_constraint_penalty"] = -min(1.0, len(negative_hits) * 0.5)

        weights = {
            "fusion": 5.0,
            "source_type": 0.6,
            "tax_domain": 1.1,
            "taxpayer_role": 0.6,
            "transaction": 0.9,
            "payment": 0.7,
            "jurisdiction": 0.5,
            "positive_constraints": 1.0,
            "provision_concepts": 1.0,
            "temporal": 1.2,
            "explicit_provision": 8.0,
            "negative_constraint_penalty": 1.4,
            "research_mechanism": 3.4,
            "research_transaction": 2.4,
            "research_wrong_neighbor": 8.0,
            "locked_institution_marker": 7.5,
        }
        final_score = sum(components[name] * weights[name] for name in components)
        positive = [
            name
            for name, value in components.items()
            if value > 0 and name not in {"fusion", "source_type"}
        ]
        negative: list[str] = []
        if components["temporal"] == 0.0:
            negative.append("outside_target_legal_period")
        negative.extend(f"negative_constraint:{value}" for value in negative_hits)
        if assessment and assessment.reject:
            negative.append(f"wrong_legal_mechanism:{assessment.document_mechanism}")
        if assessment and assessment.relation in {"direct", "strong_analogy"}:
            positive.append(f"research_relation:{assessment.relation}")
        positive.extend(f"locked_institution_marker:{item}" for item in locked_marker_ids)
        return RerankScore(
            final_score=final_score,
            component_scores=components,
            positive_reasons=positive,
            negative_reasons=negative,
        )

    def rerank(
        self,
        issue: LegalIssue,
        candidates: Sequence[RetrievalCandidate],
        *,
        target_date: Optional[str],
    ) -> list[RetrievalCandidate]:
        reranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            result = self.score(issue, candidate, target_date=target_date)
            reranked.append(
                replace(
                    candidate,
                    score=result.final_score,
                    component_scores=result.component_scores,
                    positive_reasons=tuple(result.positive_reasons),
                    negative_reasons=tuple(result.negative_reasons),
                )
            )
        return sorted(reranked, key=lambda item: (-item.score, item.candidate_id))


LegalReranker = TransparentLegalReranker


_ISSUE_EXPLICIT_REFERENCE_RE = re.compile(
    r"\bart\.\s*\d+[a-z]*"
    r"(?:\s*(?:ust\.\s*\d+[a-z]*|§\s*\d+[a-z]*))?"
    r"(?:\s*pkt\s*\d+[a-z]*)?"
    r"(?:\s*lit\.\s*[a-z])?",
    re.IGNORECASE,
)
_EXPLICIT_QUERY_DOMAIN_RE = re.compile(
    r"^\s*(CIT|PIT|VAT|UFR|PCC|SD|ORDYNACJA|OP|AKCYZA|EXCISE|PP)\b",
    re.IGNORECASE,
)


def _normalise_reference(value: str) -> str:
    return " ".join(value.casefold().replace("artykuł", "art.").split()).strip(" .;:,")


def _issue_explicit_references(issue: LegalIssue) -> tuple[str, ...]:
    """References explicitly requested by a primary-law query family."""
    result: list[str] = []
    for family in issue.query_families:
        if family.lane not in {"primary_law", "both"}:
            continue
        if family.family not in {"explicit_provision_reference", "explicit_provision"}:
            continue
        match = _ISSUE_EXPLICIT_REFERENCE_RE.search(family.query)
        if not match:
            continue
        reference = _normalise_reference(match.group(0))
        if reference and reference not in result:
            result.append(reference)
    return tuple(result)


def _explicit_family_target(family: QueryFamily) -> tuple[str, str] | None:
    """Return the act domain and citation declared by one exact query.

    Article numbers repeat across Polish statutes.  The domain in ``UFR art.
    5`` therefore scopes that individual lookup; using every domain attached
    to the wider issue can silently substitute VAT or CIT art. 5.
    """

    if family.lane not in {"primary_law", "both"}:
        return None
    if family.family not in {"explicit_provision_reference", "explicit_provision"}:
        return None
    reference_match = _ISSUE_EXPLICIT_REFERENCE_RE.search(family.query)
    if not reference_match:
        return None
    domain_match = _EXPLICIT_QUERY_DOMAIN_RE.search(family.query)
    return (
        domain_match.group(1).upper() if domain_match else "",
        _normalise_reference(reference_match.group(0)),
    )


def _candidate_references(candidate: RetrievalCandidate) -> tuple[str, ...]:
    values: list[str] = []
    display = str(candidate.metadata.get("display_reference") or "")
    if display:
        values.append(display)
    raw = candidate.metadata.get("legal_provisions") or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (list, tuple, set, frozenset)):
        values.extend(str(value) for value in raw)
    return tuple(_normalise_reference(value) for value in values if value)


def _authority_provision_overlap(
    issue: LegalIssue,
    candidate: RetrievalCandidate,
) -> bool:
    """Match an authority's article citation despite editorial granularity."""

    expected_values = [
        _normalise_reference(match.group(0))
        for concept in issue.possible_provision_concepts
        for match in _ISSUE_EXPLICIT_REFERENCE_RE.finditer(concept)
    ]
    found_values = [*_candidate_references(candidate)]
    found_values.extend(
        _normalise_reference(match.group(0))
        for match in _ISSUE_EXPLICIT_REFERENCE_RE.finditer(candidate.text)
    )

    def article(value: str) -> str:
        match = re.search(r"\bart\.\s*\d+[a-z]*", value, re.IGNORECASE)
        return _normalise_reference(match.group(0)) if match else ""

    expected_articles = {article(value) for value in expected_values} - {""}
    found_articles = {article(value) for value in found_values} - {""}
    return bool(expected_articles.intersection(found_articles))


def _reference_match_specificity(required: str, found: str) -> float:
    """Score an exact provision above one of its finer editorial units."""
    if found == required:
        return 2.0
    if found.startswith(required + " "):
        return 1.0
    return 0.0


def _candidate_matches_exact_target(
    candidate: RetrievalCandidate,
    *,
    domain: str,
    reference: str,
) -> bool:
    candidate_domains = {
        str(value).upper()
        for value in candidate.metadata.get("tax_domains") or []
        if str(value).strip()
    }
    if domain and domain not in candidate_domains:
        return False
    return any(
        _reference_match_specificity(reference, found) > 0
        for found in _candidate_references(candidate)
    )


def _select_with_exact_targets(
    issue: LegalIssue,
    families: Sequence[QueryFamily],
    candidates: Sequence[RetrievalCandidate],
    *,
    ordinary_limit: int,
) -> tuple[RetrievalCandidate, ...]:
    """Pin one verified candidate per exact act-and-provision target.

    Raising top-k to the number of dependencies is insufficient: many child
    units of one long article can still occupy every slot.  Pinning happens
    only after normal retrieval and reranking and never manufactures evidence.
    """

    targets = [target for family in families if (target := _explicit_family_target(family))]
    pinned: list[RetrievalCandidate] = []
    pinned_ids: set[str] = set()
    for domain, reference in targets:
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id not in pinned_ids
                and _candidate_matches_exact_target(
                    candidate,
                    domain=domain,
                    reference=reference,
                )
            ),
            None,
        )
        if match is None:
            continue
        pinned.append(match)
        pinned_ids.add(match.candidate_id)
    selected_limit = max(ordinary_limit, len(targets), len(pinned))
    remainder = [
        candidate for candidate in candidates if candidate.candidate_id not in pinned_ids
    ]
    return tuple((*pinned, *remainder[: max(0, selected_limit - len(pinned))]))


class _BaseLane:
    lane_name = ""
    source_types: frozenset[str] = frozenset()

    def __init__(
        self,
        backend: RetrievalBackend,
        *,
        embedding_index: Optional[VersionedEmbeddingIndex] = None,
        provision_graph: Optional[ProvisionGraph] = None,
        reranker: Optional[TransparentLegalReranker] = None,
        config: Optional[RetrievalConfig] = None,
        institution_matcher: InstitutionMatcher | None = None,
    ) -> None:
        self.backend = backend
        self.embedding_index = embedding_index
        self.provision_graph = provision_graph
        self.institution_matcher = institution_matcher or InstitutionMatcher()
        self.reranker = reranker or TransparentLegalReranker(self.institution_matcher)
        self.config = config or RetrievalConfig()

    def _source_type_groups(self, issue: LegalIssue) -> tuple[frozenset[str], ...]:
        """Return independent source pools requested for this issue.

        Tax treaties are stored in the statute table and interpretations tend
        to outnumber judgments.  Passing every lane type to one query allowed
        an unrelated treaty into ordinary PIT bundles and let interpretations
        consume the whole authority top-k.  Each requested authority class is
        therefore recalled independently, while the primary lane is narrowed
        to the exact classes declared by the research plan.
        """

        requested = {str(value).lower() for value in issue.requested_source_types}
        if self.lane_name == "authority":
            interpretive = frozenset(
                value
                for value in {"interpretation", "general_interpretation"}
                if value in self.source_types
                and (
                    value in requested
                    or "interpretation" in requested
                    or "general_interpretation" in requested
                )
            )
            guidance = frozenset({"guidance"}) if "guidance" in requested else frozenset()
            judicial = frozenset(
                value
                for value in {"judgment", "resolution"}
                if value in self.source_types
                and (value in requested or "judgment" in requested or "resolution" in requested)
            )
            groups = tuple(group for group in (interpretive, guidance, judicial) if group)
            return groups or (self.source_types,)
        allowed = frozenset(value for value in self.source_types if value in requested)
        return (allowed or self.source_types,)

    async def retrieve(
        self, plan: LegalResearchPlan, issue: LegalIssue
    ) -> LaneResult:
        families = _families_for_lane(plan, issue, self.lane_name)
        metadata_filters: dict[str, Any] = {
            "tax_domains": list(issue.tax_domains),
            "target_date": plan.target_date,
        }
        channel_lists: list[tuple[str, str, Sequence[RetrievalCandidate]]] = []
        trace: list[dict[str, Any]] = []
        source_type_groups = self._source_type_groups(issue)
        vector_source_types = frozenset(
            value for group in source_type_groups for value in group
        )
        named_authority_lookup = bool(issue.locked_institution_ids)
        preloaded_lexical: dict[tuple[int, int], list[RetrievalCandidate]] = {}
        if named_authority_lookup:
            # Deterministic channels are independent (canonical name, aliases,
            # provision hints and statutory wording).  Keep their fan-out
            # bounded. MariaDB FULLTEXT requests use separate connections,
            # but concurrent searches against the same corpus were observed
            # to lose the high-specificity result under load; execute this
            # small lookup serially and deterministically.
            requests = [
                (family_index, group_index, family, source_group)
                for family_index, family in enumerate(families)
                for group_index, source_group in enumerate(source_type_groups)
            ]
            semaphore = asyncio.Semaphore(1)

            async def bounded_lookup(family: QueryFamily, source_group: frozenset[str]) -> list[RetrievalCandidate]:
                async with semaphore:
                    try:
                        return await asyncio.wait_for(
                            _call_backend(
                                self.backend,
                                family.query,
                                limit=min(self.config.lexical_limit_per_query, 8),
                                source_types=source_group,
                                metadata_filters={
                                    **metadata_filters,
                                    **({"tax_domains": [_explicit_family_target(family)[0]]}
                                       if _explicit_family_target(family) and _explicit_family_target(family)[0]
                                       else {}),
                                },
                            ),
                            timeout=9.0,
                        )
                    except asyncio.TimeoutError:
                        # One sparse provision channel must not consume the
                        # whole interactive request.  The bounded lookup is
                        # given enough time for the remote MariaDB FTS round
                        # trip, while a true timeout remains observable as a
                        # zero-result channel.
                        return []

            responses = await asyncio.gather(
                *(bounded_lookup(family, source_group) for _, _, family, source_group in requests)
            )
            preloaded_lexical = {
                (family_index, group_index): result
                for (family_index, group_index, _, _), result in zip(requests, responses)
            }

        for family_index, family in enumerate(families):
            family_filters = dict(metadata_filters)
            explicit_target = _explicit_family_target(family)
            if explicit_target and explicit_target[0]:
                family_filters["tax_domains"] = [explicit_target[0]]
            # Interpretation and judgment pools are independent, so retrieve
            # them concurrently. Combined with the lower lane concurrency,
            # this keeps the effective database fan-out bounded while halving
            # single-issue wall time on remote MySQL.
            lexical_groups = (
                [preloaded_lexical[(family_index, group_index)] for group_index in range(len(source_type_groups))]
                if named_authority_lookup
                else await asyncio.gather(
                    *(
                        _call_backend(
                            self.backend,
                            family.query,
                            limit=self.config.lexical_limit_per_query,
                            source_types=source_group,
                            metadata_filters=family_filters,
                        )
                        for source_group in source_type_groups
                    )
                )
            )
            for source_group, lexical in zip(source_type_groups, lexical_groups):
                lexical = [
                    item
                    for item in lexical
                    if item.source_type in source_group
                    and _effective_on(item.metadata, plan.target_date)
                ]
                group_name = "+".join(sorted(source_group))
                channel_lists.append((f"lexical:{group_name}", family.family, lexical))
                trace.append(
                    {
                        "event": "candidate_source",
                        "lane": self.lane_name,
                        "family": family.family,
                        "channel": "lexical",
                        "source_type_group": sorted(source_group),
                        "count": len(lexical),
                        "backend": getattr(self.backend, "trace_marker", type(self.backend).__name__),
                        "fallback_origin": family.origin == "fallback",
                    }
                )

            if self.embedding_index is not None:
                try:
                    hits = await self.embedding_index.query(
                        family.query,
                        limit=self.config.vector_limit_per_query,
                    )
                except Exception as exc:
                    if self.config.require_vector_index:
                        raise
                    trace.append(
                        {
                            "event": "vector_source_error",
                            "lane": self.lane_name,
                            "family": family.family,
                            "error": type(exc).__name__,
                        }
                    )
                else:
                    vector_candidates = [
                        _candidate_from_embedding_hit(hit)
                        for hit in hits
                        if _embedding_hit_allowed(hit, vector_source_types, plan.target_date)
                    ]
                    channel_lists.append(("vector", family.family, vector_candidates))
                    trace.append(
                        {
                            "event": "candidate_source",
                            "lane": self.lane_name,
                            "family": family.family,
                            "channel": "real_vector",
                            "count": len(vector_candidates),
                            "embedding_model": self.embedding_index.provider.model,
                        }
                    )
            elif self.config.require_vector_index:
                raise RuntimeError("A real embedding index is required by retrieval configuration")
            else:
                trace.append(
                    {
                        "event": "candidate_source_unavailable",
                        "lane": self.lane_name,
                        "family": family.family,
                        "channel": "real_vector",
                    }
                )

        raw_candidates = _unique_candidates(
            item for _, _, items in channel_lists for item in items
        )
        metadata_ranked = sorted(
            raw_candidates,
            key=lambda item: (
                -_candidate_metadata_score(issue, item, plan.target_date),
                item.candidate_id,
            ),
        )
        if metadata_ranked:
            channel_lists.append(("metadata", "issue_metadata", metadata_ranked))
            trace.append(
                {
                    "event": "candidate_source",
                    "lane": self.lane_name,
                    "family": "issue_metadata",
                    "channel": "metadata",
                    "count": len(metadata_ranked),
                }
            )
        reference_candidates = self._reference_candidates(
            raw_candidates, plan.target_date
        )
        if reference_candidates:
            channel_lists.append(
                ("references", "provision_dependencies", reference_candidates)
            )
            trace.append(
                {
                    "event": "candidate_source",
                    "lane": self.lane_name,
                    "family": "provision_dependencies",
                    "channel": "references",
                    "count": len(reference_candidates),
                    "target_date": plan.target_date,
                }
            )
        fused = reciprocal_rank_fusion(channel_lists, rrf_k=self.config.rrf_k)
        before_rerank = len(fused)
        reranked = self.reranker.rerank(
            issue,
            fused,
            target_date=plan.target_date,
        )
        if self.lane_name == "authority" and (
            issue.transactions
            or issue.payments
            or issue.positive_fact_constraints
            or issue.possible_provision_concepts
        ) and not issue.locked_institution_ids:
            unfiltered_count = len(reranked)

            def materially_relevant(candidate: RetrievalCandidate) -> bool:
                if candidate.source_type.casefold() == "guidance":
                    # Curated official guidance commonly uses abstract rather
                    # than case-fact wording and already forms a narrow pool.
                    return True
                if candidate.component_scores.get("research_wrong_neighbor", 0.0) < 0:
                    return False
                fact_match = (
                    candidate.component_scores.get("transaction", 0.0) >= 0.15
                    or candidate.component_scores.get("payment", 0.0) >= 0.15
                    or candidate.component_scores.get("positive_constraints", 0.0)
                    >= 0.15
                )
                provision_match = _authority_provision_overlap(issue, candidate)
                has_fact_scope = bool(
                    issue.transactions
                    or issue.payments
                    or issue.positive_fact_constraints
                )
                if has_fact_scope and issue.possible_provision_concepts:
                    return fact_match and provision_match
                if has_fact_scope:
                    return fact_match
                return provision_match

            reranked = [candidate for candidate in reranked if materially_relevant(candidate)]
            trace.append(
                {
                    "event": "authority_material_relevance_filter",
                    "candidate_count_before": unfiltered_count,
                    "candidate_count_after": len(reranked),
                }
            )
        if self.lane_name == "authority" and issue.locked_institution_ids:
            # The candidate is classified independently from the question.
            # In particular, the question's WHT/SaaS plan must never label a
            # housing or rehabilitation interpretation as a WHT authority.
            question_card = build_question_card(question=plan.user_query, issue=issue)
            hydrate_document = getattr(self.backend, "hydrate_document", None)

            async def source_for_validation(candidate: RetrievalCandidate) -> RetrievalCandidate:
                """Use complete source text for the bounded validation pool.

                Retrieval remains chunk-oriented, but a direct authority must
                not be classified from an arbitrary first chunk when the
                backend can cheaply hydrate its source document.
                """

                if hydrate_document is None:
                    return candidate
                try:
                    hydrated = await asyncio.wait_for(hydrate_document(candidate), timeout=8.0)
                except Exception:
                    return candidate
                if not isinstance(hydrated, RetrievalCandidate):
                    return candidate
                return replace(
                    hydrated,
                    metadata={
                        **dict(hydrated.metadata),
                        "document_validation_hydrated": len(hydrated.text) > len(candidate.text),
                    },
                )

            # This is deliberately bounded: the card cache makes repeat
            # requests cheap and a sparse query cannot trigger a whole-corpus
            # expansion before validation.
            hydration_limit = max(12, self.config.selected_limit_per_issue * 2)
            hydrated_prefix = await asyncio.gather(
                *(source_for_validation(candidate) for candidate in reranked[:hydration_limit])
            )
            reranked = [*hydrated_prefix, *reranked[hydration_limit:]]
            retained: list[RetrievalCandidate] = []
            for candidate in reranked:
                document_card = build_document_card(candidate, matcher=self.institution_matcher)
                validation = evaluate_document_relevance(question_card, document_card)
                if validation.passed:
                    retained.append(replace(
                        candidate,
                        metadata={
                            **dict(candidate.metadata),
                            "document_card": document_card.to_dict(),
                                "document_validation": {
                                "institution_gate_passed": validation.passed,
                                "relation": validation.relation,
                                "rejection_reason": validation.reason,
                                "matched_institutions": list(validation.matched_institutions),
                                    "axes": dict(validation.axes or {}),
                                },
                                "question_locked_institutions": list(question_card.locked_institutions),
                            },
                    ))
                    continue
                trace.append(
                    {
                        "event": "institution_filter_rejection",
                        "lane": self.lane_name,
                        "candidate_signature": {
                            "candidate_id": candidate.candidate_id,
                            "document_id": candidate.document_id,
                            "chunk_id": candidate.chunk_id,
                        },
                        "reason": validation.reason,
                        "institution_ids": list(issue.locked_institution_ids),
                        "document_card": document_card.to_dict(),
                        "relation": validation.relation,
                        "axes": dict(validation.axes or {}),
                    }
                )
            reranked = retained
        if self.lane_name == "authority":
            before_document_diversity = len(reranked)
            reranked = _authority_document_diverse_candidates(reranked)
            trace.append(
                {
                    "event": "authority_document_diversity",
                    "candidate_count_before": before_document_diversity,
                    "document_count_after": len(reranked),
                }
            )
        # A plan may declare more exact primary-law dependencies than the
        # ordinary top-k.  Never truncate below the number of independently
        # requested exact provisions; doing so makes completeness impossible
        # before evidence validation even begins.
        if self.lane_name == "authority":
            selected = _select_authorities_with_query_coverage(
                families,
                reranked,
                ordinary_limit=self.config.selected_limit_per_issue,
            )
        else:
            selected = _select_with_exact_targets(
                issue,
                families,
                reranked,
                ordinary_limit=self.config.selected_limit_per_issue,
            )
        return LaneResult(
            issue_id=issue.issue_id,
            lane=self.lane_name,
            query_families=families,
            candidates=selected,
            candidate_count_before_rerank=before_rerank,
            trace=tuple(trace),
        )

    def _reference_candidates(
        self,
        candidates: Sequence[RetrievalCandidate],
        target_date: Optional[str],
    ) -> list[RetrievalCandidate]:
        return []


class PrimaryLawLane(_BaseLane):
    lane_name = "primary_law"
    source_types = PRIMARY_SOURCE_TYPES

    def _reference_candidates(
        self,
        candidates: Sequence[RetrievalCandidate],
        target_date: Optional[str],
    ) -> list[RetrievalCandidate]:
        if self.provision_graph is None:
            return []
        provision_ids = [
            str(candidate.metadata.get("provision_id") or "") for candidate in candidates
        ]
        dependencies = self.provision_graph.resolve_dependencies(
            (value for value in provision_ids if value),
            target_date=target_date,
            max_depth=self.config.graph_depth,
        )
        existing = {item.candidate_id for item in candidates}
        result: list[RetrievalCandidate] = []
        for provision in dependencies:
            candidate = _candidate_from_provision(provision, self.provision_graph)
            if candidate.candidate_id in existing:
                continue
            existing.add(candidate.candidate_id)
            result.append(candidate)
        return result


class AuthorityLane(_BaseLane):
    lane_name = "authority"
    source_types = AUTHORITY_SOURCE_TYPES


class LegalRetriever:
    """Run primary-law and authority retrieval independently for every issue."""

    def __init__(
        self,
        backend: RetrievalBackend,
        *,
        embedding_index: Optional[VersionedEmbeddingIndex] = None,
        provision_graph: Optional[ProvisionGraph] = None,
        reranker: Optional[TransparentLegalReranker] = None,
        config: Optional[RetrievalConfig] = None,
        primary_enabled: bool = True,
        authority_enabled: bool = True,
        institution_matcher: InstitutionMatcher | None = None,
    ) -> None:
        self.primary_enabled = primary_enabled
        self.authority_enabled = authority_enabled
        common = {
            "embedding_index": embedding_index,
            "provision_graph": provision_graph,
            "reranker": reranker,
            "config": config,
            "institution_matcher": institution_matcher,
        }
        self.primary_lane = PrimaryLawLane(backend, **common)
        self.authority_lane = AuthorityLane(backend, **common)

    async def retrieve(self, plan: LegalResearchPlan) -> LegalRetrievalResult:
        jobs: list[Any] = []
        job_types: list[str] = []
        # Each lane may open a database connection.  Running every issue and
        # both lanes at once overloads the remote MySQL pool on multi-issue
        # questions, which was the source of intermittent unfinished V2 runs.
        semaphore = asyncio.Semaphore(max(1, self.primary_lane.config.lane_concurrency))

        async def run_lane(lane: _BaseLane, issue: LegalIssue) -> LaneResult:
            async with semaphore:
                return await lane.retrieve(plan, issue)

        for issue in plan.issues:
            if self.primary_enabled:
                jobs.append(run_lane(self.primary_lane, issue))
                job_types.append("primary_law")
            # Authority retrieval is deliberately separate, but only issues
            # that ask for an authority source need that lane.  VAT bundles
            # that request primary law only must not spend a full query on an
            # irrelevant interpretation/orzeczenie fallback.
            needs_authority = bool(
                set(issue.requested_source_types).intersection(AUTHORITY_SOURCE_TYPES)
            )
            if self.authority_enabled and needs_authority:
                jobs.append(run_lane(self.authority_lane, issue))
                job_types.append("authority")
        results = await asyncio.gather(*jobs) if jobs else []
        primary = tuple(
            item for kind, item in zip(job_types, results) if kind == "primary_law"
        )
        authorities = tuple(
            item for kind, item in zip(job_types, results) if kind == "authority"
        )
        trace: list[dict[str, Any]] = [
            {
                "event": "dual_lane_retrieval",
                "primary_enabled": self.primary_enabled,
                "authority_enabled": self.authority_enabled,
                "issues": len(plan.issues),
                "primary_lane_results": len(primary),
                "authority_lane_results": len(authorities),
            }
        ]
        for item in (*primary, *authorities):
            trace.extend(item.trace)

        # Both lanes deliberately run even if one lane is empty.  A partial
        # statute result is useful evidence, but it must not suppress
        # interpretations or judgments: authorities often reveal the missing
        # editorial unit.  At most two directed retries are made, avoiding a
        # retrieval loop while preserving the initial candidate pools.
        if self.primary_enabled:
            authority_refs = (
                _cited_provisions_by_issue(authorities)
                if self.authority_enabled
                else {}
            )
            primary_dependency_refs = _cited_provisions_by_issue(primary)
            missing_authority_backrefs = _references_missing_from_lane(
                primary,
                authority_refs,
            )
            missing_primary_dependency_refs = _references_missing_from_lane(
                primary,
                primary_dependency_refs,
            )
            primary_retry_refs = _merge_references_by_issue(
                primary_dependency_refs,
                authority_refs,
            )
            missing_primary_refs = _references_missing_from_lane(
                primary,
                primary_retry_refs,
            )
            if missing_primary_refs:
                retried_primary = await self._retry_lane(
                    plan, primary, missing_primary_refs, lane="primary_law"
                )
                primary = retried_primary
                trace.append(
                    {
                        "event": "authority_backreference_retry",
                        "retrieval_iteration": 1,
                        "executed": True,
                        "discovered_from_authority": missing_authority_backrefs,
                        "discovered_from_primary_dependencies": missing_primary_dependency_refs,
                    }
                )
            else:
                trace.append({"event": "authority_backreference_retry", "retrieval_iteration": 1, "executed": False, "discovered_from_authority": {}})

            if self.authority_enabled:
                primary_refs = _cited_provisions_by_issue(primary)
                missing_authority_refs = _references_missing_from_lane(
                    authorities,
                    primary_refs,
                )
                if missing_authority_refs:
                    authorities = await self._retry_lane(
                        plan, authorities, missing_authority_refs, lane="authority"
                    )
                    trace.append(
                        {
                            "event": "primary_to_authority_retry",
                            "retrieval_iteration": 2,
                            "executed": True,
                            "discovered_from_primary": missing_authority_refs,
                        }
                    )
                else:
                    trace.append({"event": "primary_to_authority_retry", "retrieval_iteration": 2, "executed": False, "discovered_from_primary": {}})
        return LegalRetrievalResult(primary, authorities, tuple(trace))

    async def recover_missing_primary_law(
        self,
        plan: LegalResearchPlan,
        current: LegalRetrievalResult,
        requests: Sequence[MissingPrimaryRequest],
        *,
        max_requests_per_issue: int = 2,
        max_requests_total: int = 8,
    ) -> tuple[LegalRetrievalResult, tuple[dict[str, Any], ...]]:
        """Recover only model-identified, verified gaps in primary law.

        The model contributes a bounded retrieval hypothesis, never evidence.
        Each request is tried through progressively broader queries and a
        candidate is merged only when its source metadata or text confirms the
        exact requested editorial unit.  Existing selected candidates remain
        intact, so an unsuccessful second pass cannot cause a retrieval
        regression or replace an answer with a topical neighbour.
        """
        if not self.primary_enabled or not requests:
            return current, ()

        issues = {issue.issue_id: issue for issue in plan.issues}
        primary_by_issue = {lane.issue_id: lane for lane in current.primary_law}
        events: list[dict[str, Any]] = []
        accepted_by_issue: dict[str, int] = {}
        total_attempted = 0

        for request in requests:
            if total_attempted >= max(1, max_requests_total):
                events.append(
                    {
                        "event": "model_primary_gap_request_skipped",
                        "reason": "total_request_limit",
                        "issue_id": request.issue_id,
                        "reference": request.reference,
                    }
                )
                continue
            issue = issues.get(request.issue_id)
            reference = _requested_primary_reference(request.reference)
            if issue is None or reference is None:
                events.append(
                    {
                        "event": "model_primary_gap_request_rejected",
                        "reason": "unknown_issue_or_invalid_reference",
                        "issue_id": request.issue_id,
                        "reference": request.reference,
                    }
                )
                continue
            if accepted_by_issue.get(issue.issue_id, 0) >= max(1, max_requests_per_issue):
                events.append(
                    {
                        "event": "model_primary_gap_request_skipped",
                        "reason": "per_issue_request_limit",
                        "issue_id": issue.issue_id,
                        "reference": reference,
                    }
                )
                continue

            original = primary_by_issue.get(issue.issue_id)
            existing_candidates = original.candidates if original else ()
            domain = _primary_recovery_domain(issue, request.act)
            if any(
                _candidate_matches_primary_recovery(candidate, domain=domain, reference=reference)
                for candidate in existing_candidates
            ):
                events.append(
                    {
                        "event": "model_primary_gap_request_skipped",
                        "reason": "already_verified_in_initial_retrieval",
                        "issue_id": issue.issue_id,
                        "act": request.act,
                        "reference": reference,
                    }
                )
                continue

            total_attempted += 1
            matched: tuple[RetrievalCandidate, ...] = ()
            selected_strategy = ""
            retry_trace: list[dict[str, Any]] = []
            for strategy, family in _primary_recovery_families(
                act=request.act,
                reference=reference,
                reason=request.reason,
                domain=domain,
            ):
                retry_issue = issue.model_copy(
                    update={
                        "tax_domains": [domain] if domain else list(issue.tax_domains),
                        "requested_source_types": list(
                            dict.fromkeys([*issue.requested_source_types, "statute"])
                        ),
                        "query_families": [family],
                    }
                )
                retry = await self.primary_lane.retrieve(plan, retry_issue)
                retry_trace.extend(retry.trace)
                matched = tuple(
                    candidate
                    for candidate in retry.candidates
                    if _candidate_matches_primary_recovery(
                        candidate,
                        domain=domain,
                        reference=reference,
                    )
                )
                if matched:
                    selected_strategy = strategy
                    break

            if not matched:
                events.append(
                    {
                        "event": "model_primary_gap_recovery",
                        "executed": True,
                        "recovered": False,
                        "issue_id": issue.issue_id,
                        "act": request.act,
                        "reference": reference,
                        "reason": request.reason,
                    }
                )
                continue

            merged = _unique_candidates((*existing_candidates, *matched))
            reranked = self.primary_lane.reranker.rerank(
                issue,
                merged,
                target_date=plan.target_date,
            )
            recovery_family = next(
                family
                for strategy, family in _primary_recovery_families(
                    act=request.act,
                    reference=reference,
                    reason=request.reason,
                    domain=domain,
                )
                if strategy == selected_strategy
            )
            families = tuple([
                *(original.query_families if original else ()),
                recovery_family,
            ])
            primary_by_issue[issue.issue_id] = LaneResult(
                issue_id=issue.issue_id,
                lane="primary_law",
                query_families=families,
                candidates=_select_with_exact_targets(
                    issue,
                    families,
                    reranked,
                    ordinary_limit=max(
                        self.primary_lane.config.selected_limit_per_issue,
                        len(existing_candidates) + len(matched),
                    ),
                ),
                candidate_count_before_rerank=(
                    (original.candidate_count_before_rerank if original else 0) + len(matched)
                ),
                trace=tuple([
                    *(original.trace if original else ()),
                    *retry_trace,
                    {
                        "event": "model_primary_gap_candidates_merged",
                        "reference": reference,
                        "strategy": selected_strategy,
                        "preserved_initial_candidates": len(existing_candidates),
                        "verified_recovered_candidates": len(matched),
                    },
                ]),
            )
            accepted_by_issue[issue.issue_id] = accepted_by_issue.get(issue.issue_id, 0) + 1
            events.append(
                {
                    "event": "model_primary_gap_recovery",
                    "executed": True,
                    "recovered": True,
                    "issue_id": issue.issue_id,
                    "act": request.act,
                    "reference": reference,
                    "reason": request.reason,
                    "strategy": selected_strategy,
                    "recovered_document_ids": sorted({candidate.document_id for candidate in matched}),
                }
            )

        if not any(item.get("recovered") for item in events):
            return current, tuple(events)
        primary = tuple(
            primary_by_issue.get(issue.issue_id)
            for issue in plan.issues
            if primary_by_issue.get(issue.issue_id) is not None
        )
        return (
            LegalRetrievalResult(
                primary_law=primary,
                authorities=current.authorities,
                trace=tuple((*current.trace, *events)),
            ),
            tuple(events),
        )

    async def _retry_lane(
        self,
        plan: LegalResearchPlan,
        existing: tuple[LaneResult, ...],
        references: Mapping[str, list[str]],
        *,
        lane: str,
    ) -> tuple[LaneResult, ...]:
        lane_impl = self.primary_lane if lane == "primary_law" else self.authority_lane
        existing_by_issue = {item.issue_id: item for item in existing}
        retried: list[LaneResult] = []
        for issue in plan.issues:
            cited = references.get(issue.issue_id, [])
            original = existing_by_issue.get(issue.issue_id)
            if not cited:
                if original is not None:
                    retried.append(original)
                continue
            # Initial candidates are merged below, so a retry must execute
            # only newly discovered citations. Re-running every natural query
            # multiplies latency and can let the same broad candidates crowd
            # out the backreference lane again.
            families = list(
                QueryFamily(
                    family="authority_backreference",
                    query=citation,
                    lane=lane,
                    origin="model",
                )
                for citation in cited[:12]
            )
            retry_issue = issue.model_copy(update={"query_families": families})
            retry = await lane_impl.retrieve(plan, retry_issue)
            if original is None:
                retried.append(retry)
                continue
            merged_candidates = _unique_candidates((*original.candidates, *retry.candidates))
            reranked = lane_impl.reranker.rerank(issue, merged_candidates, target_date=plan.target_date)
            merged_families = tuple((*original.query_families, *retry.query_families))
            retried.append(
                LaneResult(
                    issue_id=issue.issue_id,
                    lane=lane,
                    query_families=merged_families,
                    candidates=_select_with_exact_targets(
                        issue,
                        merged_families,
                        reranked,
                        ordinary_limit=lane_impl.config.selected_limit_per_issue,
                    ),
                    candidate_count_before_rerank=original.candidate_count_before_rerank + retry.candidate_count_before_rerank,
                    trace=tuple((*original.trace, *retry.trace, {"event": "backreference_candidates_merged", "lane": lane, "preserved_initial_candidates": len(original.candidates), "references": cited})),
                )
            )
        return tuple(retrieved for retrieved in retried)


def _cited_provisions_by_issue(
    lanes: Sequence[LaneResult],
) -> dict[str, list[str]]:
    """Collect citations from metadata and source text without topic rules."""
    result: dict[str, list[str]] = {}
    for lane in lanes:
        values: list[str] = []
        for candidate in lane.candidates:
            raw = candidate.metadata.get("legal_provisions") or candidate.metadata.get("citation") or []
            if isinstance(raw, str):
                raw = [raw]
            values.extend(str(item).strip() for item in raw if str(item).strip())
            values.extend(match.group(0).strip() for match in _PROVISION_REFERENCE_RE.finditer(candidate.text))
            article_match = _ARTICLE_ONLY_RE.search(candidate.text)
            if article_match:
                article = article_match.group(1)
                values.extend(
                    f"art. {article} {match.group(0).strip()}"
                    for match in _RELATIVE_PROVISION_REFERENCE_RE.finditer(candidate.text)
                )
        # Case-insensitive stable de-duplication makes retries reproducible.
        seen: set[str] = set()
        result[lane.issue_id] = [
            value for value in values
            if not (value.casefold() in seen or seen.add(value.casefold()))
        ]
    return result


def _merge_references_by_issue(
    *groups: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for group in groups:
        for issue_id, references in group.items():
            values = merged.setdefault(issue_id, [])
            known = {value.casefold() for value in values}
            for reference in references:
                if reference.casefold() not in known:
                    values.append(reference)
                    known.add(reference.casefold())
    return merged


def _references_missing_from_lane(
    lanes: Sequence[LaneResult],
    references: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    """Retry exact citations absent from a lane, even when it has neighbours.

    A populated lane is not necessarily a complete lane.  The two-reference
    cap bounds fan-out while allowing a retrieved statute or authority to
    reveal a controlling dependency that broad first-pass candidates missed.
    """
    existing = {item.issue_id: item for item in lanes}
    result: dict[str, list[str]] = {}
    for issue_id, cited in references.items():
        lane = existing.get(issue_id)
        found = (
            _retrieved_provisions_by_issue([lane]).get(issue_id, [])
            if lane is not None
            else []
        )
        missing: list[str] = []
        for raw_reference in cited:
            required = _normalise_reference(raw_reference)
            if not required or any(
                _reference_match_specificity(required, _normalise_reference(value)) > 0
                for value in found
            ):
                continue
            missing.append(raw_reference)
            if len(missing) >= 2:
                break
        if missing:
            result[issue_id] = missing
    return result


def _retrieved_provisions_by_issue(
    lanes: Sequence[LaneResult],
) -> dict[str, list[str]]:
    """Collect provisions actually represented by candidate metadata.

    Textual cross-references are discovery edges, not proof that the target
    editorial unit itself was retrieved. Keeping those concepts separate is
    what lets the second pass follow ``art. 22p → art. 19``.
    """

    result: dict[str, list[str]] = {}
    for lane in lanes:
        values: list[str] = []
        for candidate in lane.candidates:
            values.extend(_candidate_references(candidate))
        seen: set[str] = set()
        result[lane.issue_id] = [
            value
            for value in values
            if value and not (value.casefold() in seen or seen.add(value.casefold()))
        ]
    return result


def _requested_primary_reference(value: str) -> Optional[str]:
    """Normalize one model request and reject anything but an exact article."""
    match = _ISSUE_EXPLICIT_REFERENCE_RE.search(str(value or ""))
    if match is None:
        return None
    reference = _normalise_reference(match.group(0))
    return reference or None


def _primary_recovery_domain(issue: LegalIssue, act: str) -> str:
    """Scope recovery to a domain already declared by the research plan.

    The model's act label improves the query wording, but it must not expand
    the legal domain beyond the independently grounded issue plan.
    """
    domains = [str(value).upper() for value in issue.tax_domains if str(value).strip()]
    act_text = str(act or "").upper()
    return next((domain for domain in domains if re.search(rf"\b{re.escape(domain)}\b", act_text)), domains[0] if len(domains) == 1 else "")


def _article_level_reference(reference: str) -> str:
    match = re.search(r"\bart\.\s*\d+[a-z]*", reference, re.IGNORECASE)
    return _normalise_reference(match.group(0)) if match else reference


def _primary_recovery_families(
    *,
    act: str,
    reference: str,
    reason: str,
    domain: str,
) -> tuple[tuple[str, QueryFamily], ...]:
    """Build the fixed, progressively broader recovery ladder.

    The strategy is data-agnostic: it works for any tax act and exact article
    supplied by the model, without benchmark or subject-matter branches.
    """
    act_label = " ".join(str(act or "").split())[:128]
    article = _article_level_reference(reference)
    reason_text = " ".join(str(reason or "").split())[:320]
    prefix = domain or act_label
    candidates = (
        ("normalized_provision_id", f"{prefix} {reference}".strip()),
        ("exact_textual_reference", reference),
        ("article_level", f"{prefix} {article}".strip()),
        ("act_level_fts", f"{act_label} {reason_text}".strip()),
    )
    seen: set[str] = set()
    result: list[tuple[str, QueryFamily]] = []
    for strategy, query in candidates:
        normalized = " ".join(query.split())
        if not normalized or normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        result.append(
            (
                strategy,
                QueryFamily(
                    family="explicit_provision",
                    query=normalized,
                    lane="primary_law",
                    origin="model",
                ),
            )
        )
    return tuple(result)


def _candidate_matches_primary_recovery(
    candidate: RetrievalCandidate,
    *,
    domain: str,
    reference: str,
) -> bool:
    if candidate.source_type not in PRIMARY_SOURCE_TYPES:
        return False
    if _candidate_matches_exact_target(candidate, domain=domain, reference=reference):
        return True
    candidate_domains = {
        str(value).upper()
        for value in candidate.metadata.get("tax_domains") or []
        if str(value).strip()
    }
    if domain and domain not in candidate_domains:
        return False
    return any(
        _reference_match_specificity(reference, _normalise_reference(match.group(0))) > 0
        for match in _PROVISION_REFERENCE_RE.finditer(candidate.text)
    )


def reciprocal_rank_fusion(
    ranked_lists: Sequence[
        tuple[str, str, Sequence[RetrievalCandidate | Mapping[str, Any] | Any]]
    ],
    *,
    rrf_k: int = 60,
) -> list[RetrievalCandidate]:
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    candidates: dict[str, RetrievalCandidate] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    families: dict[str, set[str]] = {}
    for channel, family, raw_items in ranked_lists:
        # Deterministic institution channels are additional recall, never a
        # replacement for the model/user families.  A bounded RRF multiplier
        # gives the explicit canonical/alias/provision evidence priority over
        # broad lexical neighbours once both have been retrieved.
        channel_weight = 2.5 if family.startswith("named_institution_") else 1.0
        seen_in_list: set[str] = set()
        for rank, raw in enumerate(raw_items, start=1):
            item = _coerce_candidate(raw, backend=channel, rank=rank)
            if item.candidate_id in seen_in_list:
                continue
            seen_in_list.add(item.candidate_id)
            candidates.setdefault(item.candidate_id, item)
            scores[item.candidate_id] = scores.get(item.candidate_id, 0.0) + channel_weight / (
                rrf_k + rank
            )
            channel_key = f"{channel}:{family}"
            ranks.setdefault(item.candidate_id, {})[channel_key] = rank
            families.setdefault(item.candidate_id, set()).add(family)
    fused = [
        replace(
            item,
            score=scores[item_id],
            query_families=tuple(sorted(families[item_id])),
            channel_ranks=dict(ranks[item_id]),
        )
        for item_id, item in candidates.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, item.candidate_id))


def _families_for_lane(
    plan: LegalResearchPlan, issue: LegalIssue, lane: str
) -> tuple[QueryFamily, ...]:
    families = [
        family
        for family in issue.query_families
        if family.lane in {lane, "both"}
    ]
    if issue.locked_institution_ids:
        # An explicit institution has its own bounded recall contract.  The
        # planner's broad natural-language family (often "business expense")
        # must not enter either authority or primary-law retrieval once a
        # deterministic institution has supplied its own query contract.
        model_fact_families = [
            family for family in families
            if family.family == "fact_signature"
        ]
        families = [family for family in families if family.origin == "deterministic"]
        if lane == "authority":
            # Preserve a bounded fact-only channel beside the deterministic
            # institution channels.  It recalls documents whose own text
            # contains the user's distinguishing terminology, while the
            # document card below remains the independent relevance gate.
            families.extend(model_fact_families)
            distinctive_terms = _distinctive_user_terms(plan.user_query)
            if distinctive_terms:
                families.append(QueryFamily(
                    family="user_terminology",
                    query=distinctive_terms,
                    lane="authority",
                    origin="user",
                ))
        if lane == "primary":
            provision_families = [
                family for family in families
                if family.family == "named_institution_provision"
            ]
            families = provision_families or families[:1]
    if not families:
        # This fallback is still sourced solely from immutable plan fields. It
        # performs no domain detection or static legal query expansion.
        query = plan.user_query.strip() or issue.label
        families = [
            QueryFamily(
                family="natural_language",
                query=query,
                lane=lane,
                origin="user" if plan.user_query.strip() else "model",
            )
        ]
    seen: set[tuple[str, str]] = set()
    result: list[QueryFamily] = []
    for family in families:
        key = (family.family, " ".join(family.query.split()).casefold())
        if key not in seen:
            seen.add(key)
            result.append(family)
    return tuple(result)


def _distinctive_user_terms(question: str) -> str:
    """Keep user-supplied acronyms and mixed-case terms as a narrow FTS cue.

    This is lexical preservation, not a topic expansion: no taxonomy or
    answer knowledge is added.  Acronyms such as product, contract or form
    names are often the only tokens separating two documents that share one
    tax institution.
    """

    terms: list[str] = []
    for index, token in enumerate(_WORD_RE.findall(question)):
        if index == 0 or len(token) < 2:
            continue
        if token.isupper() or any(character.isupper() for character in token[1:]):
            if token.casefold() not in {item.casefold() for item in terms}:
                terms.append(token)
        if len(terms) >= 6:
            break
    return " ".join(terms)


async def _call_backend(
    backend: RetrievalBackend,
    query: str,
    *,
    limit: int,
    source_types: frozenset[str],
    metadata_filters: Mapping[str, Any],
) -> list[RetrievalCandidate]:
    method = getattr(backend, "search", None) or getattr(backend, "search_chunks", None)
    if method is None:
        raise TypeError("Retrieval backend must implement async search or search_chunks")
    raw = method(
        query,
        limit=limit,
        source_types=source_types,
        metadata_filters=metadata_filters,
    )
    if inspect.isawaitable(raw):
        raw = await raw
    return [
        _coerce_candidate(item, backend=type(backend).__name__, rank=rank)
        for rank, item in enumerate(raw or (), start=1)
    ]


def _coerce_candidate(raw: Any, *, backend: str, rank: int) -> RetrievalCandidate:
    if isinstance(raw, RetrievalCandidate):
        return raw
    getter = raw.get if isinstance(raw, Mapping) else lambda key, default=None: getattr(raw, key, default)
    chunk_id = str(getter("chunk_id", "") or "")
    document_id = str(getter("document_id", "") or "")
    candidate_id = str(
        getter("candidate_id", "")
        or chunk_id
        or (f"{document_id}:{getter('chunk_index', rank)}" if document_id else f"{backend}:{rank}")
    )
    text = str(getter("text", "") or getter("chunk_text", "") or "")
    metadata = dict(getter("metadata", {}) or {})
    for key in (
        "subject",
        "signature",
        "published_date",
        "source_url",
        "authority",
        "legal_state_date",
        "legal_provisions",
        "tax_domains",
        "provision_id",
        "effective_from",
        "effective_to",
    ):
        value = getter(key, None)
        if value not in (None, "", [], ()):
            metadata.setdefault(key, value)
    return RetrievalCandidate(
        candidate_id=candidate_id,
        document_id=document_id,
        chunk_id=chunk_id,
        text=text,
        source_type=str(getter("source_type", "") or metadata.get("source_type") or "unknown"),
        score=float(getter("score", 0.0) or 0.0),
        metadata=metadata,
        backend=backend,
    )


def _candidate_from_embedding_hit(hit: EmbeddingHit) -> RetrievalCandidate:
    metadata = dict(hit.metadata)
    return RetrievalCandidate(
        candidate_id=hit.item_id,
        document_id=str(metadata.get("document_id") or ""),
        chunk_id=str(metadata.get("chunk_id") or hit.item_id),
        text=hit.text,
        source_type=str(metadata.get("source_type") or "unknown"),
        score=hit.score,
        metadata=metadata,
        backend="versioned_embedding_index",
    )


def _candidate_from_provision(
    provision: ProvisionUnit, graph: ProvisionGraph
) -> RetrievalCandidate:
    relationships = sorted(
        {
            edge.relationship
            for edge in graph.edges
            if edge.source_id == provision.provision_id or edge.target_id == provision.provision_id
        }
    )
    return RetrievalCandidate(
        candidate_id=provision.provision_id,
        document_id=provision.document_id,
        chunk_id=provision.provision_id,
        text=provision.text,
        source_type=str(provision.metadata.get("source_type") or "statute"),
        score=1.0 / 61.0,
        metadata={
            **dict(provision.metadata),
            "provision_id": provision.provision_id,
            "citation": provision.citation,
            "effective_from": provision.effective_from,
            "effective_to": provision.effective_to,
            "graph_relationships": relationships,
        },
        backend="provision_graph",
        channel_ranks={"references": 1},
        positive_reasons=("provision_graph_dependency",),
    )


def _authority_document_diverse_candidates(
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Return one conclusion-bearing representative per authority document."""

    grouped: dict[str, list[RetrievalCandidate]] = {}
    document_order: list[str] = []
    for candidate in candidates:
        document_id = candidate.document_id or candidate.candidate_id
        if document_id not in grouped:
            grouped[document_id] = []
            document_order.append(document_id)
        grouped[document_id].append(candidate)

    markers = (
        r"ocena\s+stanowiska",
        r"stanowisko.{0,80}(?:jest\s+)?(?:prawidłowe|nieprawidłowe)",
        r"sąd\s+zważył",
        r"rozstrzygnięcie\s+sądu",
        r"oddala\s+skargę",
        r"uchyla\s+zaskarżon",
        r"w\s+konsekwencji",
        r"(?:należy|trzeba)\s+uznać",
        r"prawo\s+do\s+odliczeni",
    )

    def material_score(candidate: RetrievalCandidate) -> int:
        text = candidate.text.casefold()
        return sum(bool(re.search(marker, text, re.I)) for marker in markers)

    representatives: list[RetrievalCandidate] = []
    for document_id in document_order:
        document_candidates = grouped[document_id]
        document_score = max(item.score for item in document_candidates)
        representative = max(
            document_candidates,
            key=lambda item: (material_score(item), item.score, -len(item.text)),
        )
        representatives.append(
            replace(
                representative,
                score=document_score,
                positive_reasons=tuple(
                    dict.fromkeys(
                        (*representative.positive_reasons, "authority_document_diversity")
                    )
                ),
            )
        )
    return sorted(representatives, key=lambda item: (-item.score, item.candidate_id))


def _select_authorities_with_query_coverage(
    families: Sequence[QueryFamily],
    candidates: Sequence[RetrievalCandidate],
    *,
    ordinary_limit: int,
) -> tuple[RetrievalCandidate, ...]:
    """Keep both authority classes visible across intentional query families."""

    if ordinary_limit <= 0:
        return ()
    selected: list[RetrievalCandidate] = []
    selected_ids: set[str] = set()

    def source_kind(candidate: RetrievalCandidate) -> str:
        source_type = candidate.source_type.casefold()
        if source_type in {"interpretation", "general_interpretation", "guidance"}:
            return "interpretive"
        if source_type in {"judgment", "resolution"}:
            return "judicial"
        return "other"

    family_names = list(dict.fromkeys(family.family for family in families))
    for required_kind, rounds in (("interpretive", 2), ("judicial", 1)):
        for _round in range(rounds):
            for family_name in family_names:
                match = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id not in selected_ids
                        and source_kind(candidate) == required_kind
                        and family_name in candidate.query_families
                    ),
                    None,
                )
                if match is None:
                    continue
                selected.append(match)
                selected_ids.add(match.candidate_id)
                if len(selected) >= ordinary_limit:
                    return tuple(selected)

    for candidate in candidates:
        if candidate.candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
        if len(selected) >= ordinary_limit:
            break
    return tuple(selected)


def _unique_candidates(
    candidates: Iterable[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        result.append(candidate)
    return result


def _candidate_metadata_score(
    issue: LegalIssue,
    candidate: RetrievalCandidate,
    target_date: Optional[str],
) -> float:
    text = f"{candidate.text} {_metadata_text(candidate.metadata)}".casefold()
    return (
        2.0 * _field_overlap(issue.tax_domains, candidate.metadata, "tax_domains", text)
        + _text_coverage(issue.taxpayer_roles, text)
        + _text_coverage(issue.transactions, text)
        + _text_coverage(issue.payments, text)
        + _text_coverage(issue.jurisdictions, text)
        + _text_coverage(issue.positive_fact_constraints, text)
        + (1.0 if _effective_on(candidate.metadata, target_date) else 0.0)
    )


def _embedding_hit_allowed(
    hit: EmbeddingHit, source_types: frozenset[str], target_date: Optional[str]
) -> bool:
    source_type = str(hit.metadata.get("source_type") or "unknown")
    return source_type in source_types and _effective_on(hit.metadata, target_date)


def _effective_on(metadata: Mapping[str, Any], target_date: Optional[str]) -> bool:
    if not target_date:
        return True
    target = _date(target_date)
    if target is None:
        return False
    start = _date(metadata.get("effective_from"))
    end = _date(metadata.get("effective_to"))
    if start is not None or end is not None:
        return (start is None or start <= target) and (end is None or target <= end)
    # Publication date alone is not an effective-period assertion. Authorities
    # may be newer while describing the requested historical legal state.
    legal_state = _date(metadata.get("legal_state_date"))
    return legal_state is None or legal_state <= target


def _date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, (list, tuple, set, frozenset)):
            values.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            values.append(str(value))
    return " ".join(values)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _WORD_RE.findall(value) if len(token) >= 2}


def _phrase_present(value: str, text: str) -> bool:
    terms = _tokens(value)
    return bool(terms) and terms.issubset(_tokens(text))


def _text_coverage(values: Iterable[str], text: str) -> float:
    material = [value for value in values if str(value).strip()]
    if not material:
        return 0.5
    return sum(1 for value in material if _phrase_present(str(value), text)) / len(material)


def _field_overlap(
    expected: Iterable[str], metadata: Mapping[str, Any], key: str, text: str
) -> float:
    expected_values = {str(value).casefold() for value in expected if str(value)}
    if not expected_values:
        return 0.5
    actual_values = {str(value).casefold() for value in _as_values(metadata.get(key))}
    if expected_values.intersection(actual_values):
        return 1.0
    return _text_coverage(expected_values, text)


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]
