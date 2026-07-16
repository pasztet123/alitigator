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
from .schemas import LegalIssue, LegalResearchPlan, QueryFamily, RerankScore


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
    lane_concurrency: int = 6


class TransparentLegalReranker:
    """Auditable metadata/fact reranker; every score has named components."""

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
    r"^\s*(CIT|PIT|VAT|UFR|PCC|SD|ORDYNACJA|OP|AKCYZA|EXCISE)\b",
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
    ) -> None:
        self.backend = backend
        self.embedding_index = embedding_index
        self.provision_graph = provision_graph
        self.reranker = reranker or TransparentLegalReranker()
        self.config = config or RetrievalConfig()

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

        for family in families:
            family_filters = dict(metadata_filters)
            explicit_target = _explicit_family_target(family)
            if explicit_target and explicit_target[0]:
                family_filters["tax_domains"] = [explicit_target[0]]
            lexical = await _call_backend(
                self.backend,
                family.query,
                limit=self.config.lexical_limit_per_query,
                source_types=self.source_types,
                metadata_filters=family_filters,
            )
            lexical = [
                item
                for item in lexical
                if item.source_type in self.source_types
                and _effective_on(item.metadata, plan.target_date)
            ]
            channel_lists.append(("lexical", family.family, lexical))
            trace.append(
                {
                    "event": "candidate_source",
                    "lane": self.lane_name,
                    "family": family.family,
                    "channel": "lexical",
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
                        if _embedding_hit_allowed(hit, self.source_types, plan.target_date)
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
        # A plan may declare more exact primary-law dependencies than the
        # ordinary top-k.  Never truncate below the number of independently
        # requested exact provisions; doing so makes completeness impossible
        # before evidence validation even begins.
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
    ) -> None:
        self.primary_enabled = primary_enabled
        self.authority_enabled = authority_enabled
        common = {
            "embedding_index": embedding_index,
            "provision_graph": provision_graph,
            "reranker": reranker,
            "config": config,
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
        if self.primary_enabled and self.authority_enabled:
            authority_refs = _cited_provisions_by_issue(authorities)
            missing_primary_refs = _references_for_missing_lane(primary, authority_refs)
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
                        "discovered_from_authority": missing_primary_refs,
                    }
                )
            else:
                trace.append({"event": "authority_backreference_retry", "retrieval_iteration": 1, "executed": False, "discovered_from_authority": {}})

            primary_refs = _cited_provisions_by_issue(primary)
            missing_authority_refs = _references_for_missing_lane(authorities, primary_refs)
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


def _references_for_missing_lane(
    lanes: Sequence[LaneResult],
    references: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    """Keep directed retries targeted to a lane that produced no evidence.

    Retrying an already populated lane with every citation found in the other
    lane fans one WHT question into dozens of remote database queries.  The
    initial lanes remain independent and complete; this is only the recovery
    path for an empty lane.
    """
    existing = {item.issue_id: item for item in lanes}
    return {
        issue_id: cited[:2]
        for issue_id, cited in references.items()
        if cited and not (existing.get(issue_id) and existing[issue_id].candidates)
    }


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
        seen_in_list: set[str] = set()
        for rank, raw in enumerate(raw_items, start=1):
            item = _coerce_candidate(raw, backend=channel, rank=rank)
            if item.candidate_id in seen_in_list:
                continue
            seen_in_list.add(item.candidate_id)
            candidates.setdefault(item.candidate_id, item)
            scores[item.candidate_id] = scores.get(item.candidate_id, 0.0) + 1.0 / (
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
