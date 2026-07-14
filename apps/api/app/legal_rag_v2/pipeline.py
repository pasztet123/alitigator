"""One evidence-gated production flow for legal RAG v2."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol
from uuid import uuid4

from pydantic import Field

from app.model_gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelTechnicalError,
    create_model_gateway,
    get_model_gateway_config,
)

from .backends import CorpusFtsBackend
from .embeddings import (
    OfflineHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VersionedEmbeddingIndex,
)
from .planner import LegalQueryPlanner, PlannerOutcome
from .provision_graph import ProvisionGraph as RuntimeProvisionGraph
from .provision_graph import ProvisionParser, ProvisionUnit
from .retrieval import (
    LegalRetrievalResult,
    LegalRetriever,
    RetrievalCandidate,
    RetrievalConfig,
)
from .schemas import (
    AnswerPlan,
    AnswerSection,
    AuthorityCard,
    CalculationRecord,
    DocumentSourceSpan,
    EvidenceBundle,
    FallbackTrace,
    LegalClaim,
    LegalResearchPlan,
    PipelineResult,
    ProvisionGraph,
    ProvisionGraphEdge,
    ProvisionReference,
    ValidationRecord,
    V2Schema,
    WriterAnalysisSection,
    WriterOutput,
    WriterSource,
)
from .trace import TraceWriter
from .wht import WhtPayAndRefundCalculationEngine, enrich_crossborder_wht_plan


PIPELINE_VERSION = "legal_rag_v2_1"
SYNTHESIS_PROMPT_VERSION = "legal_claim_synthesis_v1"
ANSWER_PROMPT_VERSION = "legal_answer_writer_v1"

GLOBAL_LEGAL_SYSTEM_RULES = """\
You are a component in an evidence-gated Polish tax-law research pipeline.
Primary law controls the normative rule. Interpretations describe tax-authority
practice and judgments describe court reasoning; neither replaces legislation.
Never add facts, provisions, document IDs or signatures that are absent from
the supplied payload. Distinguish the taxpayer's position from the authority's
or court's holding. Respect the target legal-state date, expose conflicting
evidence and leave unsupported conclusions blocked. Use only validated claims.
"""

CLAIM_SYNTHESIS_RULES = GLOBAL_LEGAL_SYSTEM_RULES + """

Produce only the requested structured claim set. Apply the controlling
provisions to the explicitly grounded facts, but do not write the final answer.
Every material approved or conditional claim needs primary-law IDs and source
spans from the payload. An authority-pattern claim needs concrete authority
document IDs. A numeric result needs a calculation ID produced by code. If the
evidence is insufficient, return a blocked status rather than guessing.
"""

ANSWER_WRITER_RULES = GLOBAL_LEGAL_SYSTEM_RULES + """

Write a structured answer plan result, not free-form Markdown. You may
paraphrase only the supplied validated claims. Do not change claim status,
perform a calculation, create a citation or infer a missing fact. Put material
uncertainty in risks_and_gaps and list every claim ID you actually use.
"""


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.getenv("K_REVISION", "unknown")


class ClaimSet(V2Schema):
    claims: list[LegalClaim] = Field(default_factory=list)


class CalculationEngine(Protocol):
    def calculate(
        self,
        plan: LegalResearchPlan,
        bundles: list[EvidenceBundle],
    ) -> list[CalculationRecord]:
        ...


class NoOpCalculationEngine:
    """Default engine: never lets an LLM invent a numeric result."""

    def calculate(
        self,
        plan: LegalResearchPlan,
        bundles: list[EvidenceBundle],
    ) -> list[CalculationRecord]:
        return []


class AuthorityExtractor(Protocol):
    async def extract(self, candidate: RetrievalCandidate) -> Any:
        ...


@dataclass(frozen=True)
class LegalRagV2Config:
    artifact_root: Path = Path("artifacts/model_rag_model")
    planner_model: str = "gpt-5.6-terra"
    authority_extractor_model: str = "gpt-5.6-terra"
    synthesis_model: str = "gpt-5.6-terra"
    answer_writer_model: str = "gpt-5.6-terra"
    planner_confidence_threshold: float = 0.55
    allow_legacy_fallback: bool = True
    primary_candidates_per_issue: int = 8
    authority_candidates_per_issue: int = 8
    require_real_embeddings: bool = False

    @classmethod
    def from_env(cls) -> "LegalRagV2Config":
        model_config = get_model_gateway_config()
        return cls(
            artifact_root=Path(
                os.getenv(
                    "MODEL_RAG_MODEL_ARTIFACT_ROOT",
                    os.getenv("LEGAL_RAG_V2_ARTIFACT_ROOT", "artifacts/model_rag_model"),
                )
            ),
            planner_model=os.getenv("LEGAL_PLANNER_MODEL", model_config.legal_planner_model),
            authority_extractor_model=os.getenv(
                "EVIDENCE_ANALYST_MODEL",
                os.getenv("AUTHORITY_EXTRACTOR_MODEL", model_config.authority_extractor_model),
            ),
            synthesis_model=os.getenv(
                "LEGAL_SYNTHESIS_MODEL", model_config.legal_synthesis_model
            ),
            answer_writer_model=os.getenv(
                "ANSWER_WRITER_MODEL", model_config.answer_writer_model
            ),
            planner_confidence_threshold=float(
                os.getenv("LEGAL_RAG_V2_PLANNER_CONFIDENCE_THRESHOLD", "0.55")
            ),
            allow_legacy_fallback=_env_bool(
                "LEGAL_ALLOW_LEGACY_FALLBACK",
                _env_bool("LEGAL_RAG_V2_ALLOW_LEGACY_FALLBACK", True),
            ),
            primary_candidates_per_issue=max(
                1, int(os.getenv("LEGAL_RAG_V2_PRIMARY_LIMIT_PER_ISSUE", "8"))
            ),
            authority_candidates_per_issue=max(
                1, int(os.getenv("LEGAL_RAG_V2_AUTHORITY_LIMIT_PER_ISSUE", "8"))
            ),
            require_real_embeddings=_env_bool(
                "LEGAL_RAG_V2_REQUIRE_REAL_EMBEDDINGS", False
            ),
        )


@dataclass(frozen=True)
class AuthorityExtraction:
    card: AuthorityCard
    trace: dict[str, Any]


class LegalRagV2Pipeline:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        planner: LegalQueryPlanner,
        retriever: LegalRetriever,
        authority_extractor: Optional[AuthorityExtractor] = None,
        calculation_engine: Optional[CalculationEngine] = None,
        config: Optional[LegalRagV2Config] = None,
        trace_factory: Optional[Callable[[str], TraceWriter]] = None,
    ) -> None:
        self.gateway = gateway
        self.planner = planner
        self.retriever = retriever
        self.authority_extractor = authority_extractor
        self.calculation_engine = calculation_engine or WhtPayAndRefundCalculationEngine()
        self.config = config or LegalRagV2Config.from_env()
        self.trace_factory = trace_factory or (
            lambda run_id: TraceWriter(run_id, root=self.config.artifact_root)
        )

    async def run(
        self,
        question: str,
        *,
        mode: str = "legal_rag_v2",
        request_id: Optional[str] = None,
        run_id: Optional[str] = None,
        target_date: Optional[str] = None,
        force_planner_fallback: bool = False,
    ) -> PipelineResult:
        if mode not in {"model_rag_model", "legal_rag_v2", "shadow"}:
            raise ValueError(f"Unsupported v2 run mode: {mode}")
        if not question.strip():
            raise ValueError("question cannot be empty")

        request_id = request_id or str(uuid4())
        run_id = run_id or uuid4().hex
        trace = self.trace_factory(run_id)
        trace.initialize_required()
        timings: dict[str, int] = {}
        validations: list[ValidationRecord] = []
        started_total = time.monotonic()

        trace.write_json(
            "request.json",
            {
                "request_id": request_id,
                "run_id": run_id,
                "mode": mode,
                "question": question,
                "target_date": target_date,
                "pipeline_version": PIPELINE_VERSION,
            },
        )
        trace.write_json(
            "runtime.json",
            {
                "pipeline_mode": mode,
                "retrieval_mode": "issue_scoped_bidirectional",
                "rag_backend": type(self.retriever.primary_lane.backend).__name__,
                "planner_mode": "model_first",
                "planner_provider": get_model_gateway_config().provider,
                "planner_model": self.config.planner_model,
                "authority_extractor_mode": type(self.authority_extractor).__name__ if self.authority_extractor else "unavailable",
                "evidence_provider": get_model_gateway_config().provider,
                "evidence_model": self.config.authority_extractor_model,
                "answer_provider": type(self.gateway).__name__,
                "answer_model": self.config.answer_writer_model,
                "writer_provider": get_model_gateway_config().provider,
                "writer_model": self.config.answer_writer_model,
                "provider": get_model_gateway_config().provider,
                "model": self.config.answer_writer_model,
                "git_commit": _git_commit(),
                "api_version": os.getenv("ALITIGATOR_API_VERSION", "2.0.0"),
                "controlled_pipeline_used": False,
                "fallbacks_used": [],
            },
        )
        trace.write_json(
            "model_config.json",
            {
                "planner_model": self.config.planner_model,
                "authority_extractor_model": self.config.authority_extractor_model,
                "legal_synthesis_model": self.config.synthesis_model,
                "answer_writer_model": self.config.answer_writer_model,
                "planner_reasoning_effort": self.planner.reasoning_effort,
                "authority_reasoning_effort": "low",
                "synthesis_reasoning_effort": "medium",
                "answer_reasoning_effort": "medium",
                "prompt_versions": {
                    "planner": "legal_query_planner_v2_1",
                    "synthesis": SYNTHESIS_PROMPT_VERSION,
                    "answer": ANSWER_PROMPT_VERSION,
                },
            },
        )

        stage = time.monotonic()
        planner_outcome = await self.planner.plan(
            question,
            target_date=target_date,
            force_fallback=force_planner_fallback,
        )
        timings["planner"] = _elapsed_ms(stage)
        plan = enrich_crossborder_wht_plan(planner_outcome.plan, question)
        trace.write_json("legal_research_plan.json", plan)
        trace.write_json("research_plan.json", plan)
        trace.write_json("clarification.json", plan.clarification)
        trace.write_json("fallback_trace.json", planner_outcome.fallback_trace)
        trace.write_json("planner_fallback.json", planner_outcome.fallback_trace)
        trace.write_json(
            "runtime.json",
            {
                "pipeline_mode": mode,
                "retrieval_mode": "issue_scoped_bidirectional",
                "rag_backend": type(self.retriever.primary_lane.backend).__name__,
                "planner_mode": "model_first",
                "planner_provider": get_model_gateway_config().provider,
                "planner_model": self.config.planner_model,
                "authority_extractor_mode": type(self.authority_extractor).__name__ if self.authority_extractor else "unavailable",
                "evidence_provider": get_model_gateway_config().provider,
                "evidence_model": self.config.authority_extractor_model,
                "answer_provider": type(self.gateway).__name__,
                "answer_model": self.config.answer_writer_model,
                "writer_provider": get_model_gateway_config().provider,
                "writer_model": self.config.answer_writer_model,
                "provider": get_model_gateway_config().provider,
                "model": self.config.answer_writer_model,
                "git_commit": _git_commit(),
                "api_version": os.getenv("ALITIGATOR_API_VERSION", "2.0.0"),
                "controlled_pipeline_used": False,
                "fallbacks_used": ([planner_outcome.fallback_trace.fallback_reason] if planner_outcome.fallback_trace.fallback_used else []),
            },
        )

        stage = time.monotonic()
        retrieval = await self.retriever.retrieve(plan)
        candidate_recall = _candidate_presence_recall(retrieval, plan)
        if candidate_recall < 1.0 and self.config.allow_legacy_fallback:
            augmented = self.planner.fallback_for_insufficient_recall(
                question,
                plan,
                target_date=target_date,
                candidate_recall=candidate_recall,
            )
            if augmented.fallback_trace.fallback_used:
                planner_outcome = augmented
                plan = enrich_crossborder_wht_plan(augmented.plan, question)
                trace.write_json("legal_research_plan.json", plan)
                trace.write_json("fallback_trace.json", augmented.fallback_trace)
                trace.write_json("planner_fallback.json", augmented.fallback_trace)
                retrieval = await self.retriever.retrieve(plan)
                trace.write_json(
                    "runtime.json",
                    {
                        "pipeline_mode": mode,
                        "retrieval_mode": "issue_scoped_bidirectional",
                        "rag_backend": type(self.retriever.primary_lane.backend).__name__,
                        "planner_mode": "model_first",
                        "planner_provider": get_model_gateway_config().provider,
                        "planner_model": self.config.planner_model,
                        "authority_extractor_mode": type(self.authority_extractor).__name__ if self.authority_extractor else "unavailable",
                        "evidence_provider": get_model_gateway_config().provider,
                        "evidence_model": self.config.authority_extractor_model,
                        "answer_provider": type(self.gateway).__name__,
                        "answer_model": self.config.answer_writer_model,
                        "writer_provider": get_model_gateway_config().provider,
                        "writer_model": self.config.answer_writer_model,
                        "provider": get_model_gateway_config().provider,
                        "model": self.config.answer_writer_model,
                        "git_commit": _git_commit(),
                        "api_version": os.getenv("ALITIGATOR_API_VERSION", "2.0.0"),
                        "controlled_pipeline_used": False,
                        "fallbacks_used": [augmented.fallback_trace.fallback_reason],
                    },
                )
        timings["retrieval"] = _elapsed_ms(stage)
        self._write_retrieval_trace(trace, retrieval)
        trace.write_json(
            "backreferences.json",
            [item for item in retrieval.trace if "backreference" in str(item.get("event", "")) or "primary_to_authority" in str(item.get("event", ""))],
        )
        second_pass_events = [
            item for item in retrieval.trace
            if item.get("event") in {"authority_backreference_retry", "primary_to_authority_retry"}
        ]
        trace.write_json("second_pass_queries.json", second_pass_events)
        trace.write_json(
            "second_pass_candidates.json",
            [item for item in second_pass_events if item.get("executed")],
        )
        trace.write_json("missing_evidence_requests.json", [])

        stage = time.monotonic()
        authority_cards, authority_trace = await self._extract_authorities(retrieval)
        timings["authority_extraction"] = _elapsed_ms(stage)
        trace.write_json(
            "authority_cards.json",
            {"by_issue": authority_cards, "extraction_trace": authority_trace},
        )
        trace.write_json("legal_rules.json", [])
        trace.write_json(
            "wrong_neighbor_rejections.json",
            [
                {
                    "issue_id": lane.issue_id,
                    "document_id": candidate.document_id,
                    "reasons": list(candidate.negative_reasons),
                }
                for lane in retrieval.authorities
                for candidate in lane.candidates
                if any("negative_constraint" in reason for reason in candidate.negative_reasons)
            ],
        )

        stage = time.monotonic()
        runtime_graph, graph_schema, provision_refs = _build_provision_graph(
            retrieval,
            target_date=plan.target_date,
        )
        timings["provision_graph"] = _elapsed_ms(stage)
        trace.write_json("provision_graph.json", graph_schema)

        bundles = _build_evidence_bundles(
            plan,
            retrieval,
            authority_cards,
            runtime_graph,
            provision_refs,
        )
        trace.write_json("evidence_bundles.json", bundles)
        evidence_bindings = [
            {
                "source_id": authority.document_id,
                "target_id": bundle.issue_id,
                "target_type": "issue",
                "relation": "supports",
                "score": authority.extraction_confidence,
                "reason": "selected_for_issue_after_legal_reranking",
                "supporting_span_ids": [
                    f"{span.document_id}:{span.start}:{span.end}"
                    for _, spans in authority.source_spans
                    for span in spans
                ],
            }
            for bundle in bundles
            for authority in bundle.supporting_authorities
        ]
        trace.write_json("evidence_bindings.json", evidence_bindings)
        trace.write_json(
            "issue_coverage.json",
            [
                {
                    "issue_id": item.issue_id,
                    "controlling_provision_present": item.controlling_provision_present,
                    "dependency_coverage": item.dependency_coverage,
                    "exception_coverage": item.exception_coverage,
                    "temporal_validation_passed": item.temporal_validation_passed,
                    "authority_candidates_present": item.authority_candidates_present,
                    "supporting_authorities_present": item.supporting_authorities_present,
                    "contrary_authorities_present": item.contrary_authorities_present,
                    "status": item.coverage_status,
                }
                for item in bundles
            ],
        )
        trace.write_json("provision_lineage.json", _provision_lineage(retrieval, bundles))
        trace.write_json("authority_lineage.json", _authority_lineage(retrieval, bundles))

        calculations = self.calculation_engine.calculate(plan, bundles)
        trace.write_json("calculations.json", calculations)

        stage = time.monotonic()
        claims, synthesis_validation = await self._synthesize_and_validate_claims(
            question,
            plan,
            bundles,
            calculations,
        )
        timings["claim_synthesis"] = _elapsed_ms(stage)
        validations.append(synthesis_validation)
        trace.write_json("claims.json", claims)
        claim_bindings = [
            {
                "source_id": authority_id,
                "target_id": claim.claim_id,
                "target_type": "claim",
                "relation": (
                    "contradicts"
                    if authority_id in claim.contrary_authority_ids
                    else "supports"
                ),
                "score": claim.confidence,
                "reason": "authority_selected_by_structured_claim_synthesis_and_validated_within_issue_bundle",
                "supporting_span_ids": [],
            }
            for claim in claims
            for authority_id in (
                *claim.supporting_authority_ids,
                *claim.contrary_authority_ids,
            )
        ]
        trace.write_json("evidence_bindings.json", [*evidence_bindings, *claim_bindings])

        answer_plan = _build_answer_plan(plan, claims, calculations)
        trace.write_json("answer_plan.json", answer_plan)
        writer_payload = {
            "question": question,
            "legal_research_plan": plan,
            "evidence_bundles": bundles,
            "validated_claims": claims,
            "calculations": calculations,
            "answer_plan": answer_plan,
        }
        trace.write_json("writer_payload.json", writer_payload)

        stage = time.monotonic()
        writer_output, writer_validation = await self._write_answer(writer_payload)
        timings["answer_writer"] = _elapsed_ms(stage)
        validations.append(writer_validation)
        trace.write_json("writer_output.json", writer_output)

        final_answer = render_structured_answer(writer_output)
        render_validation = validate_rendered_answer(
            final_answer,
            writer_output=writer_output,
            claims=claims,
            bundles=bundles,
        )
        validations.append(render_validation)
        trace.write_text("final_answer.txt", final_answer)
        trace.write_json(
            "provision_lineage.json",
            _provision_lineage(
                retrieval,
                bundles,
                claims=claims,
                writer_output=writer_output,
                final_answer=final_answer,
            ),
        )
        trace.write_json(
            "authority_lineage.json",
            _authority_lineage(retrieval, bundles, claims=claims, final_answer=final_answer),
        )
        trace.write_json("validation.json", validations)
        timings["total"] = _elapsed_ms(started_total)
        trace.write_json("timings.json", timings)
        trace.write_json(
            "costs.json",
            {
                "status": "usage_not_exposed_by_gateway_contract",
                "total_cost_usd": None,
            },
        )
        trace.write_json(
            "token_usage.json",
            {
                "status": "usage_not_exposed_by_gateway_contract",
                "stages": [],
            },
        )
        trace.write_json(
            "metrics.json",
            {
                "issue_recall": candidate_recall,
                "approved_claims_without_primary_source": sum(
                    1 for claim in claims
                    if claim.status in {"approved", "conditional_missing_fact"}
                    and claim.material and not claim.controlling_provision_ids
                ),
                "blank_legal_references": sum(
                    1 for bundle in bundles
                    for provision in (*bundle.controlling_provisions, *bundle.dependency_provisions, *bundle.exception_provisions)
                    if not provision.citation.strip()
                ),
                "secondary_sources_discarded_when_primary_incomplete": False,
                "authority_backreference_retry_executed": any(item.get("event") == "authority_backreference_retry" and item.get("executed") for item in retrieval.trace),
                "partial_primary_candidates_preserved": True,
                "wrong_neighbor_rate": (
                    sum(
                        1 for lane in retrieval.authorities for candidate in lane.candidates
                        if any("negative_constraint" in reason for reason in candidate.negative_reasons)
                    )
                    / max(1, sum(len(lane.candidates) for lane in retrieval.authorities))
                ),
                "authority_abstention_rate": (
                    1.0
                    - sum(len(cards) for cards in authority_cards.values())
                    / max(1, sum(len(lane.candidates) for lane in retrieval.authorities))
                ),
                "second_retrieval_rate": float(
                    any(
                        item.get("executed")
                        for item in retrieval.trace
                        if item.get("event") in {"authority_backreference_retry", "primary_to_authority_retry"}
                    )
                ),
                "fallback_rate": float(planner_outcome.fallback_trace.fallback_used),
                "latency_ms": timings,
                "cost_per_request_usd": None,
            },
        )

        return PipelineResult(
            request_id=request_id,
            run_id=run_id,
            mode=mode,
            legal_research_plan=plan,
            fallback_trace=planner_outcome.fallback_trace,
            provision_graph=graph_schema,
            evidence_bundles=bundles,
            claims=claims,
            calculations=calculations,
            answer_plan=answer_plan,
            writer_output=writer_output,
            final_answer=final_answer,
            validation=validations,
            timings_ms=timings,
            costs={},
        )

    @staticmethod
    def _write_retrieval_trace(
        trace: TraceWriter,
        retrieval: LegalRetrievalResult,
    ) -> None:
        primary_queries = [
            {
                "issue_id": lane.issue_id,
                "families": [item.model_dump(mode="json") for item in lane.query_families],
            }
            for lane in retrieval.primary_law
        ]
        authority_queries = [
            {
                "issue_id": lane.issue_id,
                "families": [item.model_dump(mode="json") for item in lane.query_families],
            }
            for lane in retrieval.authorities
        ]
        primary_candidates = [
            {"issue_id": lane.issue_id, **candidate.to_dict()}
            for lane in retrieval.primary_law
            for candidate in lane.candidates
        ]
        authority_candidates = [
            {"issue_id": lane.issue_id, **candidate.to_dict()}
            for lane in retrieval.authorities
            for candidate in lane.candidates
        ]
        trace.write_json("primary_queries.json", primary_queries)
        trace.write_json("primary_candidates.json", primary_candidates)
        trace.write_json("authority_queries.json", authority_queries)
        trace.write_json("authority_candidates.json", authority_candidates)
        trace.write_json(
            "reranking.json",
            [
                {
                    "issue_id": lane.issue_id,
                    "lane": lane.lane,
                    "candidate_count_before_rerank": lane.candidate_count_before_rerank,
                    "candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "final_score": candidate.score,
                            "component_scores": dict(candidate.component_scores),
                            "positive_reasons": list(candidate.positive_reasons),
                            "negative_reasons": list(candidate.negative_reasons),
                        }
                        for candidate in lane.candidates
                    ],
                    "trace": list(lane.trace),
                }
                for lane in (*retrieval.primary_law, *retrieval.authorities)
            ],
        )
        trace.write_json(
            "first_pass_reranking.json",
            [
                {
                    "issue_id": lane.issue_id,
                    "lane": lane.lane,
                    "candidate_count_before_rerank": lane.candidate_count_before_rerank,
                    "candidates": [
                        {
                            "document_id": candidate.document_id,
                            "issue_id": lane.issue_id,
                            "final_score": candidate.score,
                            "issue_match": candidate.component_scores.get("tax_domain", 0.0),
                            "material_fact_match": candidate.component_scores.get("positive_constraints", 0.0),
                            "provision_match": candidate.component_scores.get("provision_concepts", 0.0),
                            "role_match": candidate.component_scores.get("taxpayer_role", 0.0),
                            "transaction_match": candidate.component_scores.get("transaction", 0.0),
                            "temporal_match": candidate.component_scores.get("temporal", 0.0),
                            "holding_relevance": candidate.component_scores.get("holding_relevance", 0.0),
                            "wrong_neighbor_penalty": abs(candidate.component_scores.get("negative_constraint_penalty", 0.0)),
                            "positive_reasons": list(candidate.positive_reasons),
                            "negative_reasons": list(candidate.negative_reasons),
                        }
                        for candidate in lane.candidates
                    ],
                }
                for lane in (*retrieval.primary_law, *retrieval.authorities)
            ],
        )

    async def _extract_authorities(
        self,
        retrieval: LegalRetrievalResult,
    ) -> tuple[dict[str, list[AuthorityCard]], list[dict[str, Any]]]:
        cards: dict[str, list[AuthorityCard]] = {}
        traces: list[dict[str, Any]] = []
        if self.authority_extractor is None:
            for lane in retrieval.authorities:
                cards[lane.issue_id] = []
                traces.append(
                    {
                        "issue_id": lane.issue_id,
                        "extractor": "unavailable",
                        "candidate_count": len(lane.candidates),
                    }
                )
            return cards, traces

        for lane in retrieval.authorities:
            issue_cards: list[AuthorityCard] = []
            for candidate in lane.candidates[: self.config.authority_candidates_per_issue]:
                try:
                    extracted = await self.authority_extractor.extract(candidate)
                except Exception as exc:
                    traces.append(
                        {
                            "issue_id": lane.issue_id,
                            "document_id": candidate.document_id,
                            "extractor_error": type(exc).__name__,
                        }
                    )
                    continue
                card = getattr(extracted, "card", extracted)
                if not isinstance(card, AuthorityCard):
                    card = AuthorityCard.model_validate(card)
                _validate_authority_spans(card, candidate)
                issue_cards.append(card)
                trace_payload = getattr(extracted, "trace", {})
                traces.append(
                    {
                        "issue_id": lane.issue_id,
                        "document_id": candidate.document_id,
                        **dict(trace_payload or {}),
                    }
                )
            cards[lane.issue_id] = issue_cards
        return cards, traces

    async def _synthesize_and_validate_claims(
        self,
        question: str,
        plan: LegalResearchPlan,
        bundles: list[EvidenceBundle],
        calculations: list[CalculationRecord],
    ) -> tuple[list[LegalClaim], ValidationRecord]:
        payload = {
            "prompt_version": SYNTHESIS_PROMPT_VERSION,
            "question": question,
            "plan": plan.model_dump(mode="json"),
            "evidence_bundles": [item.model_dump(mode="json") for item in bundles],
            "calculation_records": [item.model_dump(mode="json") for item in calculations],
        }
        try:
            output = await self.gateway.generate_structured(
                response_model=ClaimSet,
                input=json.dumps(payload, ensure_ascii=False),
                system_prompt=CLAIM_SYNTHESIS_RULES,
                model=self.config.synthesis_model,
                reasoning_effort="medium",
                max_output_tokens=12000,
            )
            claims = output.claims
        except ModelGatewayError as exc:
            claims = _blocked_claims(plan, bundles, reason=type(exc).__name__)
            return claims, ValidationRecord(
                stage="claim_validation",
                passed=False,
                errors=["structured_claim_synthesis_failed"],
                warnings=[type(exc).__name__],
            )

        claims, errors, warnings = validate_claims(
            claims,
            plan=plan,
            bundles=bundles,
            calculations=calculations,
        )
        return claims, ValidationRecord(
            stage="claim_validation",
            passed=not errors,
            errors=errors,
            warnings=warnings,
        )

    async def _write_answer(
        self,
        writer_payload: dict[str, Any],
    ) -> tuple[WriterOutput, ValidationRecord]:
        serializable = {
            key: value.model_dump(mode="json")
            if isinstance(value, V2Schema)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, list) and value and isinstance(value[0], V2Schema)
            else value
            for key, value in writer_payload.items()
        }
        try:
            output = await self.gateway.generate_structured(
                response_model=WriterOutput,
                input=json.dumps(serializable, ensure_ascii=False),
                system_prompt=ANSWER_WRITER_RULES,
                model=self.config.answer_writer_model,
                reasoning_effort="medium",
                max_output_tokens=10000,
            )
        except ModelGatewayError as exc:
            output = _deterministic_writer_output(writer_payload)
            return output, ValidationRecord(
                stage="writer_validation",
                passed=False,
                errors=["structured_answer_writer_failed"],
                warnings=[type(exc).__name__, "deterministic_renderer_fallback"],
            )

        errors = validate_writer_output(
            output,
            answer_plan=writer_payload["answer_plan"],
            bundles=writer_payload["evidence_bundles"],
        )
        if errors:
            output = _deterministic_writer_output(writer_payload)
        return output, ValidationRecord(
            stage="writer_validation",
            passed=not errors,
            errors=errors,
            warnings=["deterministic_renderer_fallback"] if errors else [],
        )


def create_default_pipeline(
    *,
    gateway: Optional[ModelGateway] = None,
    config: Optional[LegalRagV2Config] = None,
) -> LegalRagV2Pipeline:
    config = config or LegalRagV2Config.from_env()
    gateway = gateway or create_model_gateway()
    planner = LegalQueryPlanner(
        gateway,
        model=config.planner_model,
        minimum_confidence=config.planner_confidence_threshold,
    )
    embedding_index = _open_embedding_index_if_available()
    retriever = LegalRetriever(
        CorpusFtsBackend(),
        embedding_index=embedding_index,
        config=RetrievalConfig(
            selected_limit_per_issue=max(
                config.primary_candidates_per_issue,
                config.authority_candidates_per_issue,
            ),
            require_vector_index=config.require_real_embeddings,
        ),
    )
    authority_extractor: Optional[AuthorityExtractor] = None
    try:
        from .authority import HeuristicAuthorityExtractor, ModelAuthorityExtractor

        authority_extractor = ModelAuthorityExtractor(
            gateway,
            model=config.authority_extractor_model,
            heuristic_fallback=HeuristicAuthorityExtractor(),
        )
    except ImportError:
        authority_extractor = None
    return LegalRagV2Pipeline(
        gateway=gateway,
        planner=planner,
        retriever=retriever,
        authority_extractor=authority_extractor,
        config=config,
    )


def _open_embedding_index_if_available() -> Optional[VersionedEmbeddingIndex]:
    path = Path(
        os.getenv(
            "EMBEDDING_INDEX_PATH",
            "artifacts/model_rag_model/embedding_index.sqlite3",
        )
    )
    if not path.exists():
        return None
    provider_name = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
    if provider_name == "openai" and os.getenv("OPENAI_API_KEY"):
        provider = OpenAIEmbeddingProvider(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            dimensions=dimensions,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider_name in {"offline", "hash"} and _env_bool(
        "LEGAL_RAG_V2_ALLOW_OFFLINE_HASH_EMBEDDINGS", False
    ):
        provider = OfflineHashEmbeddingProvider(dimensions=dimensions)
    else:
        return None
    return VersionedEmbeddingIndex(
        path,
        provider,
        schema_version=os.getenv("EMBEDDING_SCHEMA_VERSION", "v1"),
        chunker_version=os.getenv("EMBEDDING_CHUNKER_VERSION", "provision_units_v1"),
    )


def _candidate_presence_recall(
    retrieval: LegalRetrievalResult,
    plan: LegalResearchPlan,
) -> float:
    if not plan.issues:
        return 0.0
    primary = {lane.issue_id: bool(lane.candidates) for lane in retrieval.primary_law}
    return sum(1 for issue in plan.issues if primary.get(issue.issue_id, False)) / len(plan.issues)


def _build_provision_graph(
    retrieval: LegalRetrievalResult,
    *,
    target_date: Optional[str] = None,
) -> tuple[RuntimeProvisionGraph, ProvisionGraph, dict[str, ProvisionReference]]:
    parser = ProvisionParser()
    graph = RuntimeProvisionGraph()
    candidate_by_unit: dict[str, RetrievalCandidate] = {}
    for lane in retrieval.primary_law:
        for candidate in lane.candidates:
            version_id = str(candidate.metadata.get("version_id") or "current")
            units = parser.parse(
                candidate.text,
                document_id=_graph_document_id(candidate),
                version_id=version_id,
                effective_from=str(candidate.metadata.get("effective_from") or "") or None,
                effective_to=str(candidate.metadata.get("effective_to") or "") or None,
                metadata={
                    **dict(candidate.metadata),
                    "source_type": candidate.source_type,
                    "chunk_id": candidate.chunk_id,
                },
            )
            if not units:
                units = (_synthetic_provision_unit(candidate, version_id),)
            for unit in units:
                graph.add_provision(unit)
                candidate_by_unit[unit.provision_id] = candidate
    graph.populate_inferred_edges()

    references: dict[str, ProvisionReference] = {}
    for unit in graph.provisions:
        candidate = candidate_by_unit[unit.provision_id]
        references[unit.provision_id] = _provision_reference(
            unit,
            candidate,
            target_date=target_date,
        )
    schema = ProvisionGraph(
        provisions=list(references.values()),
        edges=[
            ProvisionGraphEdge(
                source_provision_id=edge.source_id,
                target_provision_id=edge.target_id,
                relationship=edge.relationship,
                verified=bool(edge.metadata.get("verified", False)),
            )
            for edge in graph.edges
        ],
    )
    return graph, schema, references


def _synthetic_provision_unit(
    candidate: RetrievalCandidate,
    version_id: str,
) -> ProvisionUnit:
    citations = candidate.metadata.get("legal_provisions") or []
    citation = str(citations[0]) if citations else str(
        candidate.metadata.get("subject") or "retrieved provision"
    )
    return ProvisionUnit(
        provision_id=str(candidate.metadata.get("provision_id") or candidate.candidate_id),
        document_id=_graph_document_id(candidate),
        version_id=version_id,
        citation=citation,
        text=candidate.text,
        effective_from=str(candidate.metadata.get("effective_from") or "") or None,
        effective_to=str(candidate.metadata.get("effective_to") or "") or None,
        source_span_start=0,
        source_span_end=len(candidate.text),
        metadata={
            **dict(candidate.metadata),
            "source_type": candidate.source_type,
            "chunk_id": candidate.chunk_id,
        },
    )


def _provision_reference(
    unit: ProvisionUnit,
    candidate: RetrievalCandidate,
    *,
    target_date: Optional[str] = None,
) -> ProvisionReference:
    return ProvisionReference(
        provision_id=unit.provision_id,
        document_id=candidate.document_id or unit.document_id,
        version_id=unit.version_id,
        citation=unit.citation,
        article=unit.article,
        paragraph=unit.section or unit.paragraph,
        point=unit.point,
        letter=unit.letter,
        effective_from=unit.effective_from,
        effective_to=unit.effective_to,
        status="active" if _effective_unit(unit, target_date=target_date) else "historical",
        text=unit.text,
        source_span=DocumentSourceSpan(
            start=max(0, unit.source_span_start),
            end=max(unit.source_span_start + 1, unit.source_span_end),
            quote=(
                candidate.text[
                    max(0, unit.source_span_start) : max(
                        unit.source_span_start + 1, unit.source_span_end
                    )
                ]
                or None
            ),
            source_id=candidate.chunk_id or candidate.candidate_id,
            document_id=candidate.document_id or candidate.candidate_id,
            chunk_id=candidate.chunk_id or None,
        ),
    )


def _graph_document_id(candidate: RetrievalCandidate) -> str:
    """Use one graph identity for editorial units split across source records."""
    citations = candidate.metadata.get("legal_provisions") or []
    if isinstance(citations, str):
        citations = [citations]
    article = next(
        (
            match.group(1).casefold()
            for value in citations
            for match in [re.search(r"\bart\.\s*(\d+[a-z]?)", str(value), re.IGNORECASE)]
            if match
        ),
        "",
    )
    identity = "|".join(
        part
        for part in (
            str(candidate.metadata.get("act_title") or "").strip().casefold(),
            str(candidate.metadata.get("publication") or "").strip().casefold(),
            article,
        )
        if part
    )
    if not identity:
        return candidate.document_id or candidate.candidate_id
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"act-article:{digest}"


def _build_evidence_bundles(
    plan: LegalResearchPlan,
    retrieval: LegalRetrievalResult,
    authority_cards: dict[str, list[AuthorityCard]],
    graph: RuntimeProvisionGraph,
    references: dict[str, ProvisionReference],
) -> list[EvidenceBundle]:
    primary_by_issue = {lane.issue_id: lane for lane in retrieval.primary_law}
    authority_by_issue = {lane.issue_id: lane for lane in retrieval.authorities}
    candidate_units: dict[str, list[ProvisionReference]] = {}
    for reference in references.values():
        chunk_id = reference.source_span.chunk_id if reference.source_span else None
        if chunk_id:
            candidate_units.setdefault(chunk_id, []).append(reference)

    bundles: list[EvidenceBundle] = []
    for issue in plan.issues:
        lane = primary_by_issue.get(issue.issue_id)
        controlling: list[ProvisionReference] = []
        dependencies: list[ProvisionReference] = []
        exceptions: list[ProvisionReference] = []
        issue_provision_ids: set[str] = set()
        if lane:
            for candidate in lane.candidates:
                units = sorted(
                    [
                    unit
                    for unit in candidate_units.get(candidate.chunk_id, [])
                    if unit.status == "active"
                    ],
                    key=_reference_specificity,
                    reverse=True,
                )
                if units:
                    issue_provision_ids.update(unit.provision_id for unit in units)
                    if not controlling:
                        controlling.append(units[0])
                    else:
                        dependencies.append(units[0])
                    dependencies.extend(units[1:])

        # A relation determines legal role.  The most relevant retrieved unit
        # starts as controlling, but a special/overriding unit is promoted and
        # the general referenced rule remains visible as its dependency.
        for edge in graph.edges:
            if edge.source_id not in issue_provision_ids and edge.target_id not in issue_provision_ids:
                continue
            source = references.get(edge.source_id)
            target = references.get(edge.target_id)
            if edge.relationship in {"special_rule_for", "overrides"}:
                if source and source.status == "active":
                    controlling.append(source)
                if target and target.status == "active":
                    dependencies.append(target)
            elif edge.relationship == "exception_to":
                if source and source.status == "active":
                    exceptions.append(source)
                if target and target.status == "active":
                    dependencies.append(target)
            else:
                for related in (source, target):
                    if related and related.status == "active":
                        dependencies.append(related)

        controlling = _dedupe_provisions(controlling)
        controlling_ids = {item.provision_id for item in controlling}
        dependencies = [
            item for item in _dedupe_provisions(dependencies)
            if item.provision_id not in controlling_ids
        ]
        exceptions = [
            item for item in _dedupe_provisions(exceptions)
            if item.provision_id not in controlling_ids
        ]

        cards = authority_cards.get(issue.issue_id, [])
        current_cards: list[AuthorityCard] = []
        historical_cards: list[AuthorityCard] = []
        for card in cards:
            if _card_matches_target_date(card, plan.target_date):
                current_cards.append(card)
            else:
                historical_cards.append(card)
        missing_sources: list[str] = []
        if not controlling:
            missing_sources.append("primary_law")
        authority_lane = authority_by_issue.get(issue.issue_id)
        if not authority_lane or not authority_lane.candidates:
            missing_sources.append("authority")
        required_dependency_patterns = _required_wht_issue_dependency_patterns(issue.issue_id)
        all_issue_provisions = (*controlling, *dependencies, *exceptions)
        missing_dependencies = [
            label
            for label, pattern in required_dependency_patterns
            if not any(re.search(pattern, item.citation, re.IGNORECASE) for item in all_issue_provisions)
        ]
        missing_sources.extend(f"required_primary:{item}" for item in missing_dependencies)
        retrieval_confidence = min(
            1.0,
            (0.65 if controlling else 0.0)
            + (0.25 if current_cards else 0.0)
            + (0.10 if dependencies or exceptions else 0.0),
        )
        dependency_coverage = (
            round(
                (len(required_dependency_patterns) - len(missing_dependencies))
                / len(required_dependency_patterns),
                2,
            )
            if required_dependency_patterns
            else 1.0 if dependencies else (0.5 if controlling else 0.0)
        )
        exception_coverage = 1.0 if exceptions else (0.5 if controlling else 0.0)
        coverage_status = (
            "complete"
            if controlling
            and not missing_dependencies
            and _all_active((*controlling, *dependencies, *exceptions))
            else "partial" if controlling or current_cards else "missing"
        )
        bundles.append(
            EvidenceBundle(
                issue_id=issue.issue_id,
                controlling_provisions=controlling,
                dependency_provisions=dependencies,
                exception_provisions=exceptions,
                supporting_authorities=current_cards,
                contrary_authorities=[],
                historical_authorities=historical_cards,
                missing_sources=missing_sources,
                missing_facts=[item.fact_id for item in plan.missing_facts],
                retrieval_confidence=retrieval_confidence,
                coverage_status=coverage_status,
                controlling_provision_present=bool(controlling),
                dependency_coverage=dependency_coverage,
                exception_coverage=exception_coverage,
                temporal_validation_passed=bool(controlling) and _all_active((*controlling, *dependencies, *exceptions)),
                authority_candidates_present=bool(authority_lane and authority_lane.candidates),
                supporting_authorities_present=bool(current_cards),
                contrary_authorities_present=False,
            )
        )
    return bundles


def _required_wht_issue_dependency_patterns(issue_id: str) -> tuple[tuple[str, str], ...]:
    if issue_id == "wht_pay_and_refund_procedure":
        return (
            ("art_26_2e", r"art\.\s*26\s+ust\.\s*2e"),
            ("art_26_2g", r"art\.\s*26\s+ust\.\s*2g"),
            ("art_26_7a", r"art\.\s*26\s+ust\.\s*7a"),
            ("art_26_7b", r"art\.\s*26\s+ust\.\s*7b"),
            ("art_26_7c", r"art\.\s*26\s+ust\.\s*7c"),
            ("art_26b", r"art\.\s*26b"),
            ("art_28b", r"art\.\s*28b"),
        )
    if issue_id == "vat_interest_financial_service":
        return (
            ("vat_art_28b", r"art\.\s*28b"),
            ("vat_art_17_1_4", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4"),
            ("vat_financial_exemption", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*38"),
        )
    if issue_id in {"vat_royalty_crossborder_service", "vat_management_crossborder_service"}:
        return (
            ("vat_art_28b", r"art\.\s*28b"),
            ("vat_art_17_1_4", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4"),
        )
    return ()


def _reference_specificity(reference: ProvisionReference) -> int:
    return sum(
        bool(value)
        for value in (
            reference.article,
            reference.paragraph,
            reference.point,
            reference.letter,
        )
    )


def _all_active(provisions: Iterable[ProvisionReference]) -> bool:
    return all(item.status == "active" for item in provisions)


def _provision_lineage(
    retrieval: LegalRetrievalResult,
    bundles: list[EvidenceBundle],
    *,
    claims: Optional[list[LegalClaim]] = None,
    writer_output: Optional[WriterOutput] = None,
    final_answer: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Record exact-reference transport without relying on article text parsing."""
    selected_ids = {
        provision.provision_id
        for bundle in bundles
        for provision in (*bundle.controlling_provisions, *bundle.dependency_provisions, *bundle.exception_provisions)
    }
    candidate_ids = {
        str(candidate.metadata.get("provision_id") or candidate.candidate_id)
        for lane in retrieval.primary_law
        for candidate in lane.candidates
    }
    claim_ids = {
        provision_id
        for claim in claims or []
        for provision_id in claim.controlling_provision_ids
    }
    writer_ids = {
        source.source_id for source in (writer_output.sources if writer_output else [])
    }
    return [
        {
            "provision_id": provision_id,
            "candidate_stage": True,
            "selected_stage": provision_id in selected_ids,
            "evidence_bundle_stage": provision_id in selected_ids,
            "claim_stage": provision_id in claim_ids if claims is not None else None,
            "writer_payload_stage": provision_id in claim_ids if claims is not None else None,
            "final_answer_stage": (provision_id in writer_ids and bool(final_answer)) if writer_output is not None else None,
            "drop_reason": (
                None if provision_id in selected_ids and (claims is None or provision_id in claim_ids)
                else "not_selected_after_reranking" if provision_id not in selected_ids
                else "not_used_by_claim_synthesis"
            ),
        }
        for provision_id in sorted(candidate_ids)
    ]


def _authority_lineage(
    retrieval: LegalRetrievalResult,
    bundles: list[EvidenceBundle],
    *,
    claims: Optional[list[LegalClaim]] = None,
    final_answer: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Track the first stage at which an authority leaves the evidence chain."""
    candidate_ids = {
        candidate.document_id
        for lane in retrieval.authorities
        for candidate in lane.candidates
    }
    selected_ids = {
        authority.document_id
        for bundle in bundles
        for authority in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
            *bundle.historical_authorities,
        )
    }
    claim_ids = {
        authority_id
        for claim in claims or []
        for authority_id in (
            *claim.supporting_authority_ids,
            *claim.contrary_authority_ids,
        )
    }
    return [
        {
            "authority_id": authority_id,
            "candidate_stage": True,
            "selected_stage": authority_id in selected_ids,
            "evidence_bundle_stage": authority_id in selected_ids,
            "claim_stage": authority_id in claim_ids if claims is not None else None,
            "writer_payload_stage": authority_id in claim_ids if claims is not None else None,
            "final_answer_stage": (
                authority_id in claim_ids and bool(final_answer)
                if claims is not None
                else None
            ),
            "drop_reason": (
                None
                if authority_id in selected_ids and (claims is None or authority_id in claim_ids)
                else "not_selected_after_reranking"
                if authority_id not in selected_ids
                else "not_bound_to_approved_claim"
            ),
        }
        for authority_id in sorted(candidate_ids)
    ]


def validate_claims(
    claims: list[LegalClaim],
    *,
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
    calculations: list[CalculationRecord],
) -> tuple[list[LegalClaim], list[str], list[str]]:
    issues = {item.issue_id for item in plan.issues}
    facts = {item.fact_id for item in plan.facts} | {
        item.fact_id for item in plan.missing_facts
    }
    missing_facts = {item.fact_id for item in plan.missing_facts}
    calculations_by_id = {item.calculation_id for item in calculations}
    provisions_by_issue = {
        bundle.issue_id: {
            item.provision_id
            for item in (
                *bundle.controlling_provisions,
                *bundle.dependency_provisions,
                *bundle.exception_provisions,
            )
        }
        for bundle in bundles
    }
    authorities_by_issue = {
        bundle.issue_id: {
            item.document_id
            for item in (
                *bundle.supporting_authorities,
                *bundle.contrary_authorities,
                *bundle.historical_authorities,
            )
        }
        for bundle in bundles
    }
    bundle_by_issue = {item.issue_id: item for item in bundles}
    source_documents_by_issue = {
        bundle.issue_id: {
            item.document_id
            for item in (
                *bundle.controlling_provisions,
                *bundle.dependency_provisions,
                *bundle.exception_provisions,
            )
        }
        | authorities_by_issue.get(bundle.issue_id, set())
        for bundle in bundles
    }
    errors: list[str] = []
    warnings: list[str] = []
    validated: list[LegalClaim] = []
    seen: set[str] = set()
    for claim in claims:
        claim_errors: list[str] = []
        if claim.claim_id in seen:
            claim_errors.append("duplicate_claim_id")
        seen.add(claim.claim_id)
        if claim.issue_id not in issues:
            claim_errors.append("unknown_issue_id")
        if not set(claim.controlling_provision_ids).issubset(
            provisions_by_issue.get(claim.issue_id, set())
        ):
            claim_errors.append("unknown_controlling_provision")
        authority_ids = set(claim.supporting_authority_ids) | set(
            claim.contrary_authority_ids
        )
        if not authority_ids.issubset(authorities_by_issue.get(claim.issue_id, set())):
            claim_errors.append("unknown_authority_document")
        if not set(claim.fact_dependencies).issubset(facts):
            claim_errors.append("unknown_fact_dependency")
        if not set(claim.calculation_ids).issubset(calculations_by_id):
            claim_errors.append("unknown_calculation")
        if any(
            span.document_id not in source_documents_by_issue.get(claim.issue_id, set())
            for span in claim.source_spans
        ):
            claim_errors.append("unknown_claim_source_span")
        if claim.status == "approved" and set(claim.fact_dependencies) & missing_facts:
            claim = claim.model_copy(update={"status": "conditional_missing_fact"})
            warnings.append(f"{claim.claim_id}:downgraded_for_missing_fact")
        if claim.claim_type == "calculation" and not claim.calculation_ids:
            claim_errors.append("numeric_claim_without_calculation")
        bundle = bundle_by_issue.get(claim.issue_id)
        if claim.status in {"approved", "conditional_missing_fact"} and bundle is not None:
            missing_required = [
                item for item in bundle.missing_sources
                if item.startswith("required_primary:")
            ]
            if missing_required:
                claim = claim.model_copy(update={"status": "blocked_incomplete_dependency_bundle"})
                warnings.append(f"{claim.claim_id}:blocked_for_incomplete_issue_bundle")
        if claim_errors:
            errors.extend(f"{claim.claim_id}:{item}" for item in claim_errors)
            claim = claim.model_copy(update={"status": "blocked_invalid_provision"})
        validated.append(claim)
    if not validated:
        validated = _blocked_claims(plan, bundles, reason="empty_claim_set")
        errors.append("empty_claim_set")
    return validated, errors, warnings


def _blocked_claims(
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
    *,
    reason: str,
) -> list[LegalClaim]:
    bundle_by_issue = {item.issue_id: item for item in bundles}
    result: list[LegalClaim] = []
    for issue in plan.issues:
        bundle = bundle_by_issue.get(issue.issue_id)
        status = (
            "blocked_missing_primary_law"
            if bundle is None or not bundle.controlling_provisions
            else "blocked_conflicting_evidence"
        )
        result.append(
            LegalClaim(
                claim_id=f"blocked_{issue.issue_id}",
                issue_id=issue.issue_id,
                claim_type="risk",
                text=f"Nie można zatwierdzić materialnej konkluzji dla: {issue.label}.",
                status=status,
                result=f"Wniosek zablokowany ({reason}).",
                controlling_provision_ids=[],
                supporting_authority_ids=[],
                contrary_authority_ids=[],
                fact_dependencies=[],
                calculation_ids=[],
                source_spans=[],
                confidence=0.0,
                material=True,
            )
        )
    return result


def _build_answer_plan(
    plan: LegalResearchPlan,
    claims: list[LegalClaim],
    calculations: list[CalculationRecord],
) -> AnswerPlan:
    allowed = [
        item.claim_id
        for item in claims
        if item.status in {"approved", "conditional_missing_fact"}
    ]
    if not allowed:
        allowed = [item.claim_id for item in claims]
    sections: list[AnswerSection] = []
    for issue in plan.issues:
        issue_claim_ids = [
            claim.claim_id for claim in claims if claim.issue_id == issue.issue_id
        ]
        sections.append(
            AnswerSection(
                section_id=f"analysis_{issue.issue_id}",
                title=issue.label,
                purpose="Apply validated primary law to grounded facts and show authority practice.",
                required_claim_ids=issue_claim_ids,
            )
        )
    return AnswerPlan(
        thesis_claim_ids=allowed,
        sections=sections,
        allowed_claim_ids=allowed,
        calculation_ids=[item.calculation_id for item in calculations],
    )


def validate_writer_output(
    output: WriterOutput,
    *,
    answer_plan: AnswerPlan,
    bundles: list[EvidenceBundle],
) -> list[str]:
    errors: list[str] = []
    allowed_claims = set(answer_plan.allowed_claim_ids)
    used = set(output.claim_ids_used)
    used.update(
        claim_id
        for section in output.analysis_sections
        for claim_id in section.claim_ids_used
    )
    if not used.issubset(allowed_claims):
        errors.append("writer_used_unknown_or_blocked_claim")
    allowed_source_citations = {
        item.provision_id: item.citation
        for bundle in bundles
        for item in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        )
    }
    allowed_source_citations.update({
        item.document_id: (item.signature or item.document_id)
        for bundle in bundles
        for item in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
            *bundle.historical_authorities,
        )
    })
    if any(item.source_id not in allowed_source_citations for item in output.sources):
        errors.append("writer_used_unknown_source")
    if any(
        item.source_id in allowed_source_citations
        and _normalize_citation(item.citation)
        != _normalize_citation(allowed_source_citations[item.source_id])
        for item in output.sources
    ):
        errors.append("writer_changed_source_citation")
    if any(not set(item.claim_ids).issubset(allowed_claims) for item in output.sources):
        errors.append("writer_source_uses_unknown_claim")

    allowed_citations = " ".join(item.citation.casefold() for item in output.sources)
    body = " ".join(
        [output.thesis, *(item.content for item in output.analysis_sections)]
    )
    for reference in re.findall(r"\bart\.\s*\d+[a-z]*(?:\s+ust\.\s*\d+[a-z]*)?", body, re.I):
        if " ".join(reference.casefold().split()) not in " ".join(allowed_citations.split()):
            errors.append("writer_invented_provision_reference")
            break
    known_signatures = {
        item.signature.casefold()
        for bundle in bundles
        for item in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
            *bundle.historical_authorities,
        )
        if item.signature
    }
    for signature in re.findall(r"\b(?:I|II)?\s*(?:FSK|SA/[A-Z]{1,3})\s+\d+/\d+\b", body, re.I):
        if " ".join(signature.casefold().split()) not in {
            " ".join(value.split()) for value in known_signatures
        }:
            errors.append("writer_invented_judgment_signature")
            break
    return list(dict.fromkeys(errors))


def _deterministic_writer_output(payload: dict[str, Any]) -> WriterOutput:
    claims: list[LegalClaim] = payload["validated_claims"]
    plan: LegalResearchPlan = payload["legal_research_plan"]
    bundles: list[EvidenceBundle] = payload["evidence_bundles"]
    answer_plan: AnswerPlan = payload["answer_plan"]
    allowed = set(answer_plan.allowed_claim_ids)
    selected = [item for item in claims if item.claim_id in allowed]
    thesis = " ".join(item.result for item in selected) or (
        "Brak materialnej konkluzji, która przeszła walidację źródeł."
    )
    sections = [
        WriterAnalysisSection(
            section_id=f"analysis_{issue.issue_id}",
            title=issue.label,
            content="\n".join(
                f"- {claim.text} {claim.result}"
                for claim in selected
                if claim.issue_id == issue.issue_id
            )
            or "Brak zatwierdzonego twierdzenia dla tej osi.",
            claim_ids_used=[
                claim.claim_id
                for claim in selected
                if claim.issue_id == issue.issue_id
            ],
        )
        for issue in plan.issues
    ]
    sources: list[WriterSource] = []
    for bundle in bundles:
        for provision in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        ):
            sources.append(
                WriterSource(
                    source_id=provision.provision_id,
                    label="Przepis",
                    citation=provision.citation,
                    claim_ids=[
                        claim.claim_id
                        for claim in selected
                        if provision.provision_id in claim.controlling_provision_ids
                    ],
                )
            )
        for authority in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
        ):
            sources.append(
                WriterSource(
                    source_id=authority.document_id,
                    label=authority.document_type,
                    citation=authority.signature or authority.document_id,
                    claim_ids=[
                        claim.claim_id
                        for claim in selected
                        if authority.document_id
                        in (
                            *claim.supporting_authority_ids,
                            *claim.contrary_authority_ids,
                        )
                    ],
                )
            )
    missing_questions = [item.question for item in plan.missing_facts]
    return WriterOutput(
        thesis=thesis,
        analysis_sections=sections,
        sources=_dedupe_writer_sources(sources),
        risks_and_gaps=missing_questions
        or ["Nie znaleziono dodatkowych luk poza oznaczonymi statusami claimów."],
        claim_ids_used=[item.claim_id for item in selected],
    )


def render_structured_answer(output: WriterOutput) -> str:
    analysis = "\n\n".join(
        f"## {section.title}\n{section.content}"
        for section in output.analysis_sections
    ) or "Brak zatwierdzonej analizy."
    sources = "\n".join(
        f"- {source.label}: {source.citation}" for source in output.sources
    ) or "Nie znaleziono źródła wystarczającego do materialnej konkluzji."
    risks = "\n".join(f"- {item}" for item in output.risks_and_gaps) or "- Brak."
    return (
        f"Teza\n{output.thesis}\n\n"
        f"Analiza\n{analysis}\n\n"
        f"Źródła\n{sources}\n\n"
        f"Ryzyka i luki\n{risks}"
    )


def validate_rendered_answer(
    answer: str,
    *,
    writer_output: WriterOutput,
    claims: list[LegalClaim],
    bundles: list[EvidenceBundle],
) -> ValidationRecord:
    errors: list[str] = []
    positions: list[int] = []
    for heading in ("Teza", "Analiza", "Źródła", "Ryzyka i luki"):
        marker = f"{heading}\n"
        position = answer.find(marker)
        if position < 0:
            errors.append(f"missing_section:{heading}")
        positions.append(position)
    if any(left >= right for left, right in zip(positions, positions[1:]) if left >= 0 and right >= 0):
        errors.append("invalid_section_order")
    errors.extend(
        validate_writer_output(
            writer_output,
            answer_plan=AnswerPlan(
                thesis_claim_ids=writer_output.claim_ids_used,
                sections=[],
                allowed_claim_ids=[
                    item.claim_id
                    for item in claims
                    if item.status in {"approved", "conditional_missing_fact"}
                ]
                or [item.claim_id for item in claims],
                calculation_ids=[],
            ),
            bundles=bundles,
        )
    )
    return ValidationRecord(
        stage="post_render_validation",
        passed=not errors,
        errors=list(dict.fromkeys(errors)),
        warnings=[],
    )


def _validate_authority_spans(
    card: AuthorityCard,
    candidate: RetrievalCandidate,
) -> None:
    for field_name, spans in card.source_spans.model_dump(mode="python").items():
        for raw in spans:
            span = raw if isinstance(raw, DocumentSourceSpan) else DocumentSourceSpan.model_validate(raw)
            if span.document_id != candidate.document_id:
                raise ValueError(f"authority span {field_name} points to another document")
            if span.end > len(candidate.text) or span.start < 0:
                raise ValueError(f"authority span {field_name} is outside the candidate text")
            if span.quote is not None and candidate.text[span.start : span.end] != span.quote:
                raise ValueError(f"authority span {field_name} quote does not match")


def _provision_ids_for_candidate(
    candidate: RetrievalCandidate,
    references: dict[str, ProvisionReference],
) -> list[str]:
    return [
        provision_id
        for provision_id, reference in references.items()
        if reference.source_span
        and reference.source_span.chunk_id == candidate.chunk_id
    ]


def _effective_unit(unit: ProvisionUnit, *, target_date: Optional[str] = None) -> bool:
    try:
        reference_date = (
            date.fromisoformat(target_date[:10]) if target_date else date.today()
        )
        effective_from = (
            date.fromisoformat(str(unit.effective_from)[:10])
            if unit.effective_from
            else None
        )
        effective_to = (
            date.fromisoformat(str(unit.effective_to)[:10])
            if unit.effective_to
            else None
        )
    except ValueError:
        return False
    return (
        (effective_from is None or effective_from <= reference_date)
        and (effective_to is None or reference_date <= effective_to)
    )


def _card_matches_target_date(card: AuthorityCard, target_date: Optional[str]) -> bool:
    if not target_date:
        return True
    legal_state = card.legal_state_date or card.date
    if not legal_state:
        return True
    try:
        return date.fromisoformat(str(legal_state)[:10]) <= date.fromisoformat(target_date[:10])
    except ValueError:
        return True


def _dedupe_provisions(values: Iterable[ProvisionReference]) -> list[ProvisionReference]:
    result: list[ProvisionReference] = []
    seen: set[str] = set()
    for value in values:
        if value.provision_id in seen:
            continue
        seen.add(value.provision_id)
        result.append(value)
    return result


def _dedupe_writer_sources(values: Iterable[WriterSource]) -> list[WriterSource]:
    result: list[WriterSource] = []
    seen: set[str] = set()
    for value in values:
        if value.source_id in seen:
            continue
        seen.add(value.source_id)
        result.append(value)
    return result


def _normalize_citation(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


__all__ = [
    "GLOBAL_LEGAL_SYSTEM_RULES",
    "LegalRagV2Config",
    "LegalRagV2Pipeline",
    "create_default_pipeline",
    "render_structured_answer",
    "validate_claims",
    "validate_rendered_answer",
    "validate_writer_output",
]
