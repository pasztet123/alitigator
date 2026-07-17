"""One evidence-gated production flow for legal RAG v2."""

from __future__ import annotations

import asyncio
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
    ModelFallbackError,
    ModelGateway,
    ModelGatewayError,
    ModelProviderRequestError,
    ModelRateLimitError,
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
from .family_foundation import enrich_family_foundation_plan, family_foundation_issue_kind
from .cit_costs import enrich_cit_cost_plan
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
    MissingPrimaryRequest,
    PrimaryLawGapAssessment,
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
from .transfer_pricing import enrich_transfer_pricing_plan, question_targets_transfer_pricing
from .vat import enrich_input_vat_deduction_plan, enrich_mixed_use_vehicle_vat_plan
from .wht import WhtPayAndRefundCalculationEngine, enrich_crossborder_wht_plan


PIPELINE_VERSION = "legal_rag_v2_2"
SYNTHESIS_PROMPT_VERSION = "legal_claim_synthesis_v1"
ANSWER_PROMPT_VERSION = "legal_answer_writer_v1"
PRIMARY_GAP_ASSESSMENT_PROMPT_VERSION = "primary_law_gap_assessment_v1"
# Keep ordinary cases in one call, but preflight genuinely large structured
# payloads before they reach a provider.  The byte cap also bounds the number
# of claims expected in one structured response; large multi-axis cases are
# independently synthesized with controlled concurrency.
SYNTHESIS_MAX_ISSUES_PER_BATCH = 12
SYNTHESIS_BATCH_CONCURRENCY = 3
SYNTHESIS_MAX_INPUT_BYTES = 70_000
SYNTHESIS_MAX_OUTPUT_TOKENS = 12_000
WRITER_MAX_OUTPUT_TOKENS = 8_000
BEST_EFFORT_MAX_OUTPUT_TOKENS = 4_000


class SynthesisPayloadTooLargeError(ModelGatewayError):
    pass


def _non_splittable_model_failure(error: Optional[ModelGatewayError]) -> bool:
    if error is None:
        return False
    if isinstance(error, ModelFallbackError):
        # Splitting is a decision about the primary request.  An unavailable
        # fallback must not prevent recovery from a primary schema/size error.
        return _non_splittable_model_failure(error.primary_error)
    if isinstance(error, ModelRateLimitError):
        return True
    if isinstance(error, ModelProviderRequestError):
        return error.category in {
            "authentication",
            "billing",
            "model_unavailable",
            "permission",
        }
    return isinstance(error, ModelTechnicalError)


GLOBAL_LEGAL_SYSTEM_RULES = """\
You are a component in an evidence-gated Polish tax-law research pipeline.
Primary law controls the normative rule. Interpretations describe tax-authority
practice and judgments describe court reasoning; neither replaces legislation.
Never add facts, provisions, document IDs or signatures that are absent from
the supplied payload. Distinguish the taxpayer's position from the authority's
or court's holding. Respect the target legal-state date, expose conflicting
evidence and leave unsupported conclusions blocked. Use only validated claims.
"""

PRIMARY_GAP_ASSESSMENT_RULES = GLOBAL_LEGAL_SYSTEM_RULES + """

You are checking whether the retrieved primary-law material is sufficient to
research each issue. Do not answer the legal question and do not state a legal
conclusion. Return a missing_primary_request only when all three conditions
are met: (1) the user question or issue requires a specific Polish primary-law
editorial unit, (2) the supplied candidate citations and excerpts do not
already contain that unit, and (3) you can name its exact article/section/point
reference. Use the issue_id exactly as supplied. The act must be a short label
such as PIT, CIT, VAT, UFR or Ordynacja. The reference must start with "art.".
Do not request broad background provisions, authorities, hypothetical
alternatives or a provision merely because it is commonly related to the
topic. If no exact verified retrieval request is justified, return an empty
list. At most two requests per issue.
"""

CLAIM_SYNTHESIS_RULES = GLOBAL_LEGAL_SYSTEM_RULES + """

Produce only the requested structured claim set. Apply the controlling
provisions to the explicitly grounded facts, but do not write the final answer.
Every material approved or conditional claim needs primary-law IDs and source
spans from the payload. An authority-pattern claim needs concrete authority
document IDs. A numeric result needs a calculation ID produced by code. If the
evidence is insufficient, return a blocked status rather than guessing. Read
the complete text of every bound provision: if it expressly states a statutory
rate, threshold or deadline material to the issue, report that rule and never
claim that the supplied provision omits it. A statutory percentage quoted
directly from primary law is a legal rule, not a newly performed calculation.
The payload may contain ``claim_coverage_requirements`` and a
``completion_request``. Produce separate substantive claims for every listed
requirement, copy its ``requirement_id`` to the claim's
``coverage_requirement_ids``, use one of its exact provision IDs and respect
its allowed claim types and authority IDs. An application requirement must
contain legal subsumption, not another abstract paraphrase of the statute. An
authority requirement must explain the verified holding or reasoning and the
material factual similarity or difference; a signature alone is not a claim.
A definition alone does not complete an issue that also lists a tax charge,
exemption, rate, threshold, filing rule, fact application, evidentiary issue or
authority analysis. Never describe primary law as absent when the current
issue bundle contains it, and never make a global evidence-absence statement
based only on the scope of one issue.
"""

ANSWER_WRITER_RULES = GLOBAL_LEGAL_SYSTEM_RULES + """

Write a structured answer plan result, not free-form Markdown. You may
paraphrase only the supplied validated claims. Do not change claim status,
perform a calculation, create a citation or infer a missing fact. Put material
uncertainty in risks_and_gaps and list every claim ID you actually use. The
thesis must be a short, coherent legal conclusion, not a concatenation of claim
labels, sentence fragments or repeated definitions.
When an issue requests interpretations or judgments, use the approved
authority-pattern claims and distinguish non-binding administrative practice
from court reasoning. If the evidence bundle identifies a missing authority
type, disclose that precise research gap instead of saying that no gaps exist.
"""

BEST_EFFORT_ANSWER_RULES = """\
Jesteś polskim asystentem podatkowym. Udziel użytecznej, wstępnej odpowiedzi
na pytanie użytkownika nawet wtedy, gdy wcześniejsza synteza strukturalna nie
powiodła się. W pierwszej kolejności stosuj przekazane teksty prawa i materiały
urzędowe, a następnie ostrożne rozumowanie prawnicze. Odpowiedz wprost, jaki
wynik jest najbardziej prawdopodobny i dlaczego. Wyjaśnij znaczenie faktów oraz
wskaż praktyczne dokumenty lub działania. Nie wymyślaj przepisów, sygnatur ani
interpretacji. Jeżeli materiałów wtórnych nie przekazano, powiedz to zamiast je
tworzyć. Jeżeli przekazano zweryfikowane interpretacje lub orzeczenia, omów ich
rzeczywiste tezy, rozumowanie i istotne podobieństwa albo różnice faktyczne.
Gdy dostępne są oba rodzaje źródeł, odróżnij praktykę organów od stanowiska
sądów i wykorzystaj co najmniej po jednym relewantnym materiale każdego typu.
Pole ``evidence_relation`` odróżnia wsparcie bezpośrednie, analogiczne i
kontekst dotyczący pełnego odliczenia; nie odrzucaj analogii wyłącznie dlatego,
że dotyczy motocykla albo leasingu, jeżeli mechanizm prawny jest ten sam. Nie
opisuj wyniku wyroku, gdy ``holding_verified`` jest false; możesz wtedy podać
wyłącznie ostrożną informację o jego zakresie tematycznym.
Nie przedstawiaj samego numeru dokumentu jako wsparcia bez jego materialnej
treści. W treści nie podawaj numerów artykułów ani sygnatur; zweryfikowana
lista źródeł zostanie dodana przez aplikację. Każdy materiał w payloadzie ma
``source_id``. W ostatnim polu podaj wyłącznie identyfikatory materiałów, na
których rzeczywiście opierasz analizę. Nie wskazuj materiału tylko zbadanego i
odrzuconego jako nierelewantny. Zwróć wyłącznie:
TEZA: jednoznaczna, zwięzła odpowiedź
ANALIZA: konkretne uzasadnienie odpowiadające na wszystkie elementy pytania
DOKUMENTY: praktyczna lista dowodów lub działań, jeżeli jest relewantna
WYKORZYSTANE_ŹRÓDŁA: source_id rozdzielone przecinkami albo BRAK
"""


def _git_commit() -> str:
    # The source tree is mounted at different depths locally and in Cloud
    # Run (where it is normally /app/app/...).  Never derive a repository
    # root from a fixed parent index: diagnostics must not be able to abort a
    # legal answer before retrieval begins.
    source = Path(__file__).resolve()
    repository_root = next(
        (directory for directory in source.parents if (directory / ".git").exists()),
        None,
    )
    if repository_root is None:
        return os.getenv("K_REVISION", "unknown")
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
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
    authority_extraction_candidates_per_issue: int = 4
    authority_extraction_concurrency: int = 4
    model_authority_extraction: bool = False
    model_primary_gap_recovery: bool = True
    primary_gap_recovery_requests_per_issue: int = 2
    primary_gap_recovery_requests_total: int = 8
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
            # Authority cards are supporting evidence, never a reason to make
            # the whole answer unavailable. Bound model extraction separately
            # from retrieval so a multi-issue question cannot fan out into
            # dozens of serial model calls.
            authority_extraction_candidates_per_issue=max(
                1, int(os.getenv("LEGAL_RAG_V2_AUTHORITY_EXTRACTION_LIMIT_PER_ISSUE", "4"))
            ),
            authority_extraction_concurrency=max(
                1, int(os.getenv("LEGAL_RAG_V2_AUTHORITY_EXTRACTION_CONCURRENCY", "4"))
            ),
            # Retrieval of interpretations, judgments and MF guidance stays
            # enabled.  The model-based card summariser is opt-in because it
            # can otherwise fan a single multi-payment WHT request into many
            # slow provider calls; the conservative extractor preserves source
            # spans without making the answer path dependent on them.
            model_authority_extraction=_env_bool(
                "LEGAL_RAG_V2_MODEL_AUTHORITY_EXTRACTION", False
            ),
            model_primary_gap_recovery=_env_bool(
                "LEGAL_RAG_V2_MODEL_PRIMARY_GAP_RECOVERY", True
            ),
            primary_gap_recovery_requests_per_issue=max(
                1, int(os.getenv("LEGAL_RAG_V2_PRIMARY_GAP_REQUESTS_PER_ISSUE", "2"))
            ),
            primary_gap_recovery_requests_total=max(
                1, int(os.getenv("LEGAL_RAG_V2_PRIMARY_GAP_REQUESTS_TOTAL", "8"))
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

    async def _assess_missing_primary_law(
        self,
        question: str,
        plan: LegalResearchPlan,
        retrieval: LegalRetrievalResult,
    ) -> tuple[tuple[MissingPrimaryRequest, ...], dict[str, Any]]:
        """Ask for bounded retrieval hypotheses after the first pass.

        This stage is intentionally non-authoritative and non-fatal.  It sees
        only the research issue plus compact primary-source identities and
        excerpts; it cannot create evidence or write a legal conclusion.
        """
        if not self.config.model_primary_gap_recovery:
            return (), {"executed": False, "reason": "disabled"}

        lanes = {lane.issue_id: lane for lane in retrieval.primary_law}
        issue_payload: list[dict[str, Any]] = []
        for issue in plan.issues:
            lane = lanes.get(issue.issue_id)
            candidates = []
            for candidate in (lane.candidates if lane else ())[:6]:
                raw_references = candidate.metadata.get("legal_provisions") or []
                if isinstance(raw_references, str):
                    raw_references = [raw_references]
                candidates.append(
                    {
                        "document_id": candidate.document_id,
                        "source_type": candidate.source_type,
                        "tax_domains": list(candidate.metadata.get("tax_domains") or []),
                        "references": [str(value) for value in raw_references][:8],
                        "display_reference": str(candidate.metadata.get("display_reference") or ""),
                        "text_excerpt": " ".join(candidate.text.split())[:700],
                    }
                )
            issue_payload.append(
                {
                    "issue_id": issue.issue_id,
                    "label": issue.label,
                    "tax_domains": issue.tax_domains,
                    "legal_mechanism": issue.legal_mechanism,
                    "possible_provision_concepts": issue.possible_provision_concepts[:12],
                    "positive_fact_constraints": issue.positive_fact_constraints[:8],
                    "negative_fact_constraints": issue.negative_fact_constraints[:8],
                    "retrieved_primary_candidates": candidates,
                }
            )
        payload = {
            "question": question,
            "target_date": plan.target_date,
            "issues": issue_payload,
        }
        try:
            generated = await self.gateway.generate_structured(
                response_model=PrimaryLawGapAssessment,
                input=json.dumps(payload, ensure_ascii=False),
                system_prompt=PRIMARY_GAP_ASSESSMENT_RULES,
                model=self.config.planner_model,
                reasoning_effort="low",
                max_output_tokens=1_200,
            )
            assessment = (
                generated
                if isinstance(generated, PrimaryLawGapAssessment)
                else PrimaryLawGapAssessment.model_validate(generated)
            )
        except (asyncio.TimeoutError, TimeoutError, ModelGatewayError, ValueError) as exc:
            return (), {
                "executed": True,
                "recovered": False,
                "error": type(exc).__name__,
            }

        issue_ids = {issue.issue_id for issue in plan.issues}
        accepted: list[MissingPrimaryRequest] = []
        rejected: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        per_issue: dict[str, int] = {}
        for request in assessment.missing_primary_requests:
            reference = _primary_gap_reference(request.reference)
            key = (request.issue_id, reference.casefold() if reference else "")
            if (
                request.issue_id not in issue_ids
                or reference is None
                or key in seen
                or per_issue.get(request.issue_id, 0)
                >= self.config.primary_gap_recovery_requests_per_issue
                or len(accepted) >= self.config.primary_gap_recovery_requests_total
            ):
                rejected.append(
                    {
                        "issue_id": request.issue_id,
                        "reference": request.reference,
                        "reason": "invalid_or_out_of_budget",
                    }
                )
                continue
            seen.add(key)
            per_issue[request.issue_id] = per_issue.get(request.issue_id, 0) + 1
            accepted.append(
                request.model_copy(update={"reference": reference})
            )
        return tuple(accepted), {
            "executed": True,
            "requested": len(assessment.missing_primary_requests),
            "accepted": len(accepted),
            "rejected": rejected,
        }

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
                "retrieval_mode": "issue_scoped_bidirectional_primary_gap_recovery",
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
                "primary_gap_assessment_model": self.config.planner_model,
                "legal_synthesis_model": self.config.synthesis_model,
                "answer_writer_model": self.config.answer_writer_model,
                "planner_reasoning_effort": self.planner.reasoning_effort,
                "authority_reasoning_effort": "low",
                "synthesis_reasoning_effort": "medium",
                "answer_reasoning_effort": "medium",
                "prompt_versions": {
                    "planner": "legal_query_planner_v2_1",
                    "primary_gap_assessment": PRIMARY_GAP_ASSESSMENT_PROMPT_VERSION,
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
        plan = enrich_mixed_use_vehicle_vat_plan(
            enrich_input_vat_deduction_plan(
                enrich_cit_cost_plan(
                    enrich_transfer_pricing_plan(
                        enrich_family_foundation_plan(
                            enrich_crossborder_wht_plan(planner_outcome.plan, question),
                            question,
                        ),
                        question,
                    ),
                    question,
                ),
                question,
            ),
            question,
        )
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
                plan = enrich_mixed_use_vehicle_vat_plan(
                    enrich_input_vat_deduction_plan(
                        enrich_cit_cost_plan(
                            enrich_transfer_pricing_plan(
                                enrich_family_foundation_plan(
                                    enrich_crossborder_wht_plan(augmented.plan, question),
                                    question,
                                ),
                                question,
                            ),
                            question,
                        ),
                        question,
                    ),
                    question,
                )
                trace.write_json("legal_research_plan.json", plan)
                trace.write_json("fallback_trace.json", augmented.fallback_trace)
                trace.write_json("planner_fallback.json", augmented.fallback_trace)
                retrieval = await self.retriever.retrieve(plan)
                trace.write_json(
                    "runtime.json",
                    {
                        "pipeline_mode": mode,
                        "retrieval_mode": "issue_scoped_bidirectional_primary_gap_recovery",
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
        primary_gap_started = time.monotonic()
        missing_primary_requests, primary_gap_assessment = await self._assess_missing_primary_law(
            question,
            plan,
            retrieval,
        )
        retrieval, primary_gap_recovery_events = await self.retriever.recover_missing_primary_law(
            plan,
            retrieval,
            missing_primary_requests,
            max_requests_per_issue=self.config.primary_gap_recovery_requests_per_issue,
            max_requests_total=self.config.primary_gap_recovery_requests_total,
        )
        timings["primary_gap_recovery"] = _elapsed_ms(primary_gap_started)
        timings["retrieval"] = _elapsed_ms(stage)
        self._write_retrieval_trace(trace, retrieval)
        trace.write_json(
            "backreferences.json",
            [item for item in retrieval.trace if "backreference" in str(item.get("event", "")) or "primary_to_authority" in str(item.get("event", ""))],
        )
        second_pass_events = [
            item for item in retrieval.trace
            if item.get("event") in {
                "authority_backreference_retry",
                "primary_to_authority_retry",
                "model_primary_gap_recovery",
            }
        ]
        trace.write_json("second_pass_queries.json", second_pass_events)
        trace.write_json(
            "second_pass_candidates.json",
            [item for item in second_pass_events if item.get("executed")],
        )
        trace.write_json(
            "missing_evidence_requests.json",
            {
                "model_primary_gap_assessment": primary_gap_assessment,
                "accepted_requests": [item.model_dump(mode="json") for item in missing_primary_requests],
                "recovery_events": primary_gap_recovery_events,
            },
        )

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
        if not _should_use_best_effort_writer(
            plan=plan,
            bundles=bundles,
            claims=claims,
            answer_plan=answer_plan,
        ):
            writer_output, writer_validation = await self._write_answer(writer_payload)
        else:
            writer_output, writer_validation = await self._write_best_effort_answer(
                question=question,
                plan=plan,
                bundles=bundles,
            )
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
        # The user-facing integrity gate remains strict, but a malformed
        # model writer payload must not turn a complete, source-bound result
        # into a 502.  Re-render from validated claims and serve that output
        # only when it passes the exact same post-render validation.
        if not render_validation.passed:
            deterministic_output = _deterministic_writer_output(writer_payload)
            deterministic_answer = render_structured_answer(deterministic_output)
            deterministic_validation = validate_rendered_answer(
                deterministic_answer,
                writer_output=deterministic_output,
                claims=claims,
                bundles=bundles,
            )
            if deterministic_validation.passed:
                writer_output = deterministic_output
                final_answer = deterministic_answer
                render_validation = deterministic_validation.model_copy(
                    update={
                        "warnings": [
                            "model_render_failed_integrity_check",
                            "deterministic_render_revalidated",
                        ]
                    }
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

        semaphore = asyncio.Semaphore(self.config.authority_extraction_concurrency)

        async def extract_one(
            issue_id: str,
            candidate: RetrievalCandidate,
        ) -> tuple[str, Optional[AuthorityCard], dict[str, Any]]:
            try:
                async with semaphore:
                    source_candidate = candidate
                    hydration = getattr(
                        self.retriever.authority_lane.backend,
                        "hydrate_document",
                        None,
                    )
                    hydrated = False
                    if hydration is not None:
                        try:
                            source_candidate = await asyncio.wait_for(
                                hydration(candidate),
                                timeout=8.0,
                            )
                            hydrated = len(source_candidate.text) > len(candidate.text)
                        except Exception:
                            source_candidate = candidate
                    # A missing authority card is non-fatal. The controlling
                    # primary-law bundle remains available to the renderer.
                    extracted = await asyncio.wait_for(
                        self.authority_extractor.extract(source_candidate),
                        timeout=12.0,
                    )
                card = getattr(extracted, "card", extracted)
                if not isinstance(card, AuthorityCard):
                    card = AuthorityCard.model_validate(card)
                _validate_authority_spans(card, source_candidate)
                trace_payload = getattr(extracted, "trace", {})
                return issue_id, card, {
                    "issue_id": issue_id,
                    "document_id": candidate.document_id,
                    "document_hydrated": hydrated,
                    "source_characters": len(source_candidate.text),
                    **dict(trace_payload or {}),
                }
            except Exception as exc:
                return issue_id, None, {
                    "issue_id": issue_id,
                    "document_id": candidate.document_id,
                    "extractor_error": type(exc).__name__,
                }

        tasks = []
        for lane in retrieval.authorities:
            for candidate in _balanced_authority_extraction_candidates(
                lane.candidates,
                limit=self.config.authority_extraction_candidates_per_issue,
            ):
                tasks.append(extract_one(lane.issue_id, candidate))

        for lane in retrieval.authorities:
            cards[lane.issue_id] = []

        for issue_id, card, extraction_trace in await asyncio.gather(*tasks):
            traces.append(extraction_trace)
            if card is not None:
                cards[issue_id].append(card)

        return cards, traces

    async def _synthesize_and_validate_claims(
        self,
        question: str,
        plan: LegalResearchPlan,
        bundles: list[EvidenceBundle],
        calculations: list[CalculationRecord],
    ) -> tuple[list[LegalClaim], ValidationRecord]:
        bundle_by_issue = {item.issue_id: item for item in bundles}
        synthesis_payload_sizes: list[int] = []
        batches = [
            plan.issues[index : index + SYNTHESIS_MAX_ISSUES_PER_BATCH]
            for index in range(0, len(plan.issues), SYNTHESIS_MAX_ISSUES_PER_BATCH)
        ]
        semaphore = asyncio.Semaphore(SYNTHESIS_BATCH_CONCURRENCY)

        def subplan(issues: list[Any]) -> LegalResearchPlan:
            payload = plan.model_dump(mode="python")
            payload["issues"] = issues
            return LegalResearchPlan.model_validate(payload)

        def batch_calculations(issue_ids: set[str]) -> list[CalculationRecord]:
            return [
                item
                for item in calculations
                if not item.dependencies or bool(issue_ids.intersection(item.dependencies))
            ]

        async def call_once(
            issues: list[Any],
            *,
            completion_request: Optional[dict[str, list[dict[str, Any]]]] = None,
            existing_claims: Optional[list[LegalClaim]] = None,
        ) -> list[LegalClaim]:
            issue_ids = {item.issue_id for item in issues}
            scoped_plan = subplan(issues)
            scoped_bundles = [
                bundle_by_issue[issue_id]
                for issue_id in issue_ids
                if issue_id in bundle_by_issue
            ]
            payload = {
                "prompt_version": SYNTHESIS_PROMPT_VERSION,
                "question": question,
                "plan": scoped_plan.model_dump(mode="json"),
                "evidence_bundles": [item.model_dump(mode="json") for item in scoped_bundles],
                "calculation_records": [
                    item.model_dump(mode="json")
                    for item in batch_calculations(issue_ids)
                ],
                "claim_coverage_requirements": {
                    issue_id: requirements
                    for issue_id, requirements in _claim_coverage_requirements(
                        scoped_plan,
                        scoped_bundles,
                    ).items()
                },
            }
            if completion_request:
                payload["completion_request"] = completion_request
                payload["existing_validated_claims"] = [
                    item.model_dump(mode="json") for item in existing_claims or []
                ]
            compact_payload = _compact_model_payload(payload)
            serialized_payload = json.dumps(compact_payload, ensure_ascii=False)
            payload_bytes = len(serialized_payload.encode("utf-8"))
            synthesis_payload_sizes.append(payload_bytes)
            if len(issues) > 1 and payload_bytes > SYNTHESIS_MAX_INPUT_BYTES:
                raise SynthesisPayloadTooLargeError(
                    f"structured synthesis payload exceeds {SYNTHESIS_MAX_INPUT_BYTES} bytes"
                )
            async with semaphore:
                output = await self.gateway.generate_structured(
                    response_model=ClaimSet,
                    input=serialized_payload,
                    system_prompt=CLAIM_SYNTHESIS_RULES,
                    model=self.config.synthesis_model,
                    reasoning_effort="medium",
                    max_output_tokens=SYNTHESIS_MAX_OUTPUT_TOKENS,
                )
            return [item for item in output.claims if item.issue_id in issue_ids]

        async def synthesize_batch(
            issues: list[Any],
        ) -> tuple[list[LegalClaim], list[str], list[str]]:
            failure: Optional[ModelGatewayError] = None
            try:
                batch_claims = await call_once(issues)
                if batch_claims:
                    return batch_claims, [], []
                error = "empty_issue_batch_claim_set"
            except ModelGatewayError as exc:
                failure = exc
                error = f"{type(exc).__name__}:{exc}"

            # A provider can reject a combined payload even though every
            # individual issue is valid. Split the failed batch so one axis
            # cannot erase the remaining legal analysis. Capacity, billing or
            # transport failures must never be split: doing so turns one 429
            # into a burst of additional calls.
            if len(issues) > 1 and not _non_splittable_model_failure(failure):
                children = await asyncio.gather(
                    *(synthesize_batch([issue]) for issue in issues)
                )
                return (
                    [claim for claims, _, _ in children for claim in claims],
                    [item for _, errors, _ in children for item in errors],
                    [
                        f"batch_split_after:{error}",
                        *(item for _, _, warnings in children for item in warnings),
                    ],
                )

            scoped_plan = subplan(issues)
            issue_ids = {item.issue_id for item in issues}
            scoped_bundles = [
                bundle_by_issue[issue_id]
                for issue_id in issue_ids
                if issue_id in bundle_by_issue
            ]
            return (
                _blocked_claims(scoped_plan, scoped_bundles, reason=error),
                [
                    f"structured_claim_synthesis_failed:{issue.issue_id}"
                    for issue in issues
                ],
                [f"{issue.issue_id}:{error}" for issue in issues],
            )

        synthesized = []
        for batch in batches:
            synthesized.append(await synthesize_batch(batch))
        claims = [claim for batch_claims, _, _ in synthesized for claim in batch_claims]
        synthesis_errors = [item for _, errors, _ in synthesized for item in errors]
        synthesis_warnings = [item for _, _, warnings in synthesized for item in warnings]

        claims, errors, warnings = validate_claims(
            claims,
            plan=plan,
            bundles=bundles,
            calculations=calculations,
        )
        errors = [*synthesis_errors, *errors]
        warnings = [
            f"issue_scoped_synthesis_batches:{len(batches)}",
            "synthesis_payload_bytes_max:"
            f"{max(synthesis_payload_sizes, default=0)}",
            *synthesis_warnings,
            *warnings,
        ]
        missing_claim_coverage = _claim_coverage_requirements(plan, bundles, claims)
        if missing_claim_coverage and not synthesis_errors:
            repair_issues = [
                issue for issue in plan.issues if issue.issue_id in missing_claim_coverage
            ]
            try:
                repair_claims = await call_once(
                    repair_issues,
                    completion_request=missing_claim_coverage,
                    existing_claims=[
                        claim
                        for claim in claims
                        if claim.issue_id in missing_claim_coverage
                        and claim.status in {"approved", "conditional_missing_fact"}
                    ],
                )
            except ModelGatewayError as exc:
                warnings.append(f"claim_coverage_repair_failed:{type(exc).__name__}")
            else:
                retained = [
                    claim
                    for claim in claims
                    if claim.issue_id not in missing_claim_coverage
                    or claim.status in {"approved", "conditional_missing_fact"}
                ]
                merged_by_id = {claim.claim_id: claim for claim in retained}
                merged_by_id.update({claim.claim_id: claim for claim in repair_claims})
                repaired, repair_errors, repair_warnings = validate_claims(
                    list(merged_by_id.values()),
                    plan=plan,
                    bundles=bundles,
                    calculations=calculations,
                )
                repaired_gaps = _claim_coverage_requirements(plan, bundles, repaired)
                if sum(map(len, repaired_gaps.values())) < sum(
                    map(len, missing_claim_coverage.values())
                ):
                    claims = repaired
                    errors = repair_errors
                    warnings.extend(repair_warnings)
                    warnings.append(
                        "claim_coverage_repair_applied:"
                        + ",".join(sorted(missing_claim_coverage))
                    )
                else:
                    warnings.append("claim_coverage_repair_no_improvement")
        claims, completion_warnings = _ensure_required_issue_claims(
            claims,
            plan=plan,
            bundles=bundles,
            calculations=calculations,
        )
        warnings.extend(completion_warnings)
        remaining_coverage = _claim_coverage_requirements(plan, bundles, claims)
        if remaining_coverage:
            warnings.append(
                "claim_coverage_remaining:"
                + ",".join(
                    f"{issue_id}="
                    + "|".join(
                        str(requirement["requirement_id"])
                        for requirement in requirements
                    )
                    for issue_id, requirements in sorted(remaining_coverage.items())
                )
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
        compact_payload = _compact_model_payload(serializable)
        try:
            output = await self.gateway.generate_structured(
                response_model=WriterOutput,
                input=json.dumps(compact_payload, ensure_ascii=False),
                system_prompt=ANSWER_WRITER_RULES,
                model=self.config.answer_writer_model,
                reasoning_effort="medium",
                max_output_tokens=WRITER_MAX_OUTPUT_TOKENS,
            )
        except ModelGatewayError as exc:
            output = _deterministic_writer_output(writer_payload)
            return output, ValidationRecord(
                stage="writer_validation",
                passed=False,
                errors=["structured_answer_writer_failed"],
                warnings=[
                    f"{type(exc).__name__}:{exc}",
                    "deterministic_renderer_fallback",
                ],
            )

        errors = validate_writer_output(
            output,
            answer_plan=writer_payload["answer_plan"],
            bundles=writer_payload["evidence_bundles"],
        )
        required_claim_ids = {
            item.claim_id
            for item in writer_payload["validated_claims"]
            if item.claim_id.startswith("deterministic_")
        }
        rendered_claim_ids = {
            *output.claim_ids_used,
            *(claim_id for section in output.analysis_sections for claim_id in section.claim_ids_used),
        }
        if required_claim_ids - rendered_claim_ids:
            errors.append("writer_omitted_required_deterministic_claim")
        if errors:
            output = _deterministic_writer_output(writer_payload)
        return output, ValidationRecord(
            stage="writer_validation",
            passed=not errors,
            errors=errors,
            warnings=["deterministic_renderer_fallback"] if errors else [],
        )

    async def _write_best_effort_answer(
        self,
        *,
        question: str,
        plan: LegalResearchPlan,
        bundles: list[EvidenceBundle],
    ) -> tuple[WriterOutput, ValidationRecord]:
        """Return a useful source-bounded answer when strict claims are incomplete.

        This lane is intentionally less authoritative than validated claims,
        but it remains source-bounded: the model writes only the reasoning and
        the application adds the retrieved source list deterministically.
        """

        model_evidence = _best_effort_model_evidence(plan, bundles)
        try:
            raw = await self.gateway.generate_text(
                input=json.dumps(
                    {"question": question, "evidence": model_evidence},
                    ensure_ascii=False,
                ),
                system_prompt=BEST_EFFORT_ANSWER_RULES,
                model=self.config.answer_writer_model,
                reasoning_effort="low",
                max_output_tokens=BEST_EFFORT_MAX_OUTPUT_TOKENS,
            )
        except ModelGatewayError as exc:
            return _deterministic_writer_output(
                {
                    "validated_claims": [],
                    "legal_research_plan": plan,
                    "evidence_bundles": bundles,
                    "answer_plan": AnswerPlan(),
                }
            ), ValidationRecord(
                stage="writer_validation",
                passed=False,
                errors=["best_effort_writer_failed"],
                warnings=[f"{type(exc).__name__}:{exc}"],
            )

        output = _best_effort_writer_output(raw, plan=plan, bundles=bundles)
        errors = validate_writer_output(output, answer_plan=AnswerPlan(), bundles=bundles)
        return output, ValidationRecord(
            stage="writer_validation",
            passed=not errors,
            errors=errors,
            warnings=[
                "best_effort_mode",
                "strict_claim_synthesis_incomplete",
            ],
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

        heuristic = HeuristicAuthorityExtractor()
        authority_extractor = (
            ModelAuthorityExtractor(
                gateway,
                model=config.authority_extractor_model,
                heuristic_fallback=heuristic,
            )
            if config.model_authority_extraction
            else heuristic
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


def _compact_model_payload(value: Any) -> Any:
    """Remove retrieval-only and duplicated provenance data from LLM inputs.

    The canonical plan and evidence artifacts remain lossless.  Model stages,
    however, do not need every retrieval query or a second copy of source text
    inside ``source_span.quote``: the provision ``text`` plus exact offsets and
    document IDs are sufficient for evidence binding.  Long multi-issue cases
    previously repeated the entire user question in every fallback fact and
    query family, inflating one synthesis request by hundreds of kilobytes.
    """

    if isinstance(value, dict):
        return {
            key: _compact_model_payload(item)
            for key, item in value.items()
            if key not in {"quote", "query_families", "user_query"}
        }
    if isinstance(value, list):
        return [_compact_model_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_model_payload(item) for item in value]
    return value


def _primary_gap_reference(value: str) -> Optional[str]:
    """Validate a model-suggested exact primary-law editorial reference."""
    match = re.search(
        r"\bart\.\s*\d+[a-z]*"
        r"(?:\s*(?:ust\.\s*\d+[a-z]*|§\s*\d+[a-z]*))?"
        r"(?:\s*pkt\s*\d+[a-z]*)?"
        r"(?:\s*lit\.\s*[a-z])?",
        str(value or ""),
        re.IGNORECASE,
    )
    if match is None:
        return None
    return " ".join(match.group(0).casefold().split()).strip(" .;:,") or None


def _claim_coverage_requirements(
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
    claims: Optional[list[LegalClaim]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """List evidence-backed reasoning steps that still need a claim.

    Requirements never encode the legal outcome. They select exact provisions,
    authority cards and types of reasoning already justified by the issue and
    its evidence bundle. The second synthesis pass can therefore repair a
    shallow answer without hard-coding the answer to a benchmark.
    """

    bundle_by_issue = {bundle.issue_id: bundle for bundle in bundles}
    approved_statuses = {"approved", "conditional_missing_fact"}
    claims_by_issue: dict[str, list[LegalClaim]] = {}
    for claim in claims or []:
        if claim.status in approved_statuses:
            claims_by_issue.setdefault(claim.issue_id, []).append(claim)

    result: dict[str, list[dict[str, Any]]] = {}
    for issue in plan.issues:
        family_kind = family_foundation_issue_kind(issue)
        transfer_pricing = question_targets_transfer_pricing(
            " ".join(
                (
                    issue.issue_id,
                    issue.label,
                    issue.legal_mechanism,
                    *issue.possible_provision_concepts,
                )
            )
        )
        required_vat_timing = issue.issue_id == "vat_input_deduction_timing"
        vehicle_vat_issue = issue.issue_id == "mixed_use_vehicle_vat" or (
            "mixed_use_vehicle_vat" in str(issue.legal_mechanism).casefold()
        )
        cost_issue = _is_income_tax_cost_issue(issue)
        if (
            not family_kind
            and not transfer_pricing
            and not required_vat_timing
            and not vehicle_vat_issue
            and not cost_issue
        ):
            continue
        bundle = bundle_by_issue.get(issue.issue_id)
        if bundle is None:
            continue
        provisions = [
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        ]
        requirements: list[dict[str, Any]] = []
        for (
            requirement_id,
            citation_pattern,
            document_pattern,
        ) in _required_issue_dependency_patterns(issue):
            matches = [
                provision
                for provision in provisions
                if re.search(citation_pattern, provision.citation, re.I)
                and re.search(document_pattern, provision.document_id, re.I)
            ]
            if not matches:
                continue
            representative = min(matches, key=lambda item: (len(item.citation), item.citation))
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "citation": representative.citation,
                    "allowed_provision_ids": [item.provision_id for item in matches],
                    "allowed_claim_types": ["normative_rule", "application"],
                    "allowed_authority_ids": [],
                    "purpose": "Wyjaśnij materialną regułę wynikającą z tej jednostki.",
                    "requires_explicit_acknowledgement": False,
                }
            )
        if cost_issue and provisions:
            allowed_provision_ids = list(
                dict.fromkeys(item.provision_id for item in provisions)
            )
            controlling_citation = next(
                (
                    item.citation
                    for item in provisions
                    if re.search(r"art\.\s*(?:15|22)\s+ust\.\s*1", item.citation, re.I)
                ),
                provisions[0].citation,
            )
            cost_reasoning = (
                (
                    "cost_fact_application",
                    ["application"],
                    "Zastosuj regułę kosztową do wszystkich istotnych faktów z pytania, "
                    "wskaż najbardziej prawdopodobny wynik i wyjaśnij związek wydatku "
                    "z osiąganiem, zachowaniem albo zabezpieczeniem przychodów.",
                ),
                (
                    "cost_personal_boundary",
                    ["application"],
                    "Oceń granicę między wydatkiem gospodarczym i osobistym, w tym "
                    "prywatną użyteczność, mieszane wykorzystanie i znaczenie "
                    "konkretnie wskazanych faktów zawodowych.",
                ),
                (
                    "cost_evidence_and_documentation",
                    ["application", "risk"],
                    "Wyjaśnij ciężar dowodu, znaczenie faktury oraz konkretne dokumenty "
                    "potwierdzające cel, racjonalność i faktyczne wykorzystanie wydatku.",
                ),
                (
                    "cost_material_fact_variants",
                    ["application", "risk"],
                    "Porównaj warianty stanu faktycznego wyraźnie wskazane przez "
                    "użytkownika i wyjaśnij, które różnice mogą zmienić wynik.",
                ),
            )
            requirements.extend(
                {
                    "requirement_id": requirement_id,
                    "citation": controlling_citation,
                    "allowed_provision_ids": allowed_provision_ids,
                    "allowed_claim_types": claim_types,
                    "allowed_authority_ids": [],
                    "purpose": purpose,
                    "requires_explicit_acknowledgement": True,
                }
                for requirement_id, claim_types, purpose in cost_reasoning
            )
            authority_groups = {
                "cost_interpretation_analysis": [
                    card
                    for card in bundle.supporting_authorities
                    if _authority_source_kind(card) == "interpretation"
                ],
                "cost_judgment_analysis": [
                    card
                    for card in bundle.supporting_authorities
                    if _authority_source_kind(card) == "judgment"
                ],
            }
            authority_purposes = {
                "cost_interpretation_analysis": (
                    "Omów zweryfikowane stanowisko organu podatkowego, jego tok "
                    "rozumowania oraz podobieństwa i różnice względem faktów użytkownika; "
                    "zaznacz niewiążący charakter interpretacji w cudzej sprawie."
                ),
                "cost_judgment_analysis": (
                    "Omów zweryfikowaną tezę lub rozumowanie sądu oraz podobieństwa i "
                    "różnice względem faktów użytkownika; nie przedstawiaj wyroku jako "
                    "źródła powszechnie obowiązującego prawa."
                ),
            }
            for requirement_id, cards in authority_groups.items():
                if not cards:
                    continue
                requirements.append(
                    {
                        "requirement_id": requirement_id,
                        "citation": controlling_citation,
                        "allowed_provision_ids": allowed_provision_ids,
                        "allowed_claim_types": ["authority_pattern"],
                        "allowed_authority_ids": list(
                            dict.fromkeys(card.document_id for card in cards)
                        ),
                        "purpose": authority_purposes[requirement_id],
                        "requires_explicit_acknowledgement": True,
                    }
                )
        if vehicle_vat_issue and provisions:
            allowed_provision_ids = list(
                dict.fromkeys(item.provision_id for item in provisions)
            )
            controlling_citation = next(
                (
                    item.citation
                    for item in provisions
                    if re.search(r"art\.\s*86a\s+ust\.\s*1", item.citation, re.I)
                ),
                provisions[0].citation,
            )
            vehicle_reasoning = (
                (
                    "vehicle_vat_actual_use_first",
                    ["application"],
                    "Najpierw oceń faktyczny sposób używania pojazdu. Wyjaśnij, czy "
                    "jawny użytek prywatny pozwala uznać pojazd za wykorzystywany "
                    "wyłącznie gospodarczo, zanim omówisz ewidencję i VAT-26.",
                ),
                (
                    "vehicle_vat_mixed_use_and_fuel",
                    ["application"],
                    "Zastosuj limit dla użytku mieszanego i wyjaśnij, czy obejmuje "
                    "paliwo oraz czy własność albo ujęcie auta w środkach trwałych "
                    "stanowi warunek odliczenia.",
                ),
                (
                    "vehicle_vat_full_deduction_conditions",
                    ["application", "risk"],
                    "Rozdziel materialny warunek wyłącznego użytku gospodarczego od "
                    "formalnych obowiązków ewidencji przebiegu i informacji VAT-26; "
                    "nie przedstawiaj samych formalności jako wystarczających do 100%.",
                ),
                (
                    "vehicle_vat_invoice_and_evidence",
                    ["application", "risk"],
                    "Wyjaśnij znaczenie faktury i dowodów związku zakupu paliwa z "
                    "czynnościami opodatkowanymi.",
                ),
            )
            requirements.extend(
                {
                    "requirement_id": requirement_id,
                    "citation": controlling_citation,
                    "allowed_provision_ids": allowed_provision_ids,
                    "allowed_claim_types": claim_types,
                    "allowed_authority_ids": [],
                    "purpose": purpose,
                    "requires_explicit_acknowledgement": True,
                }
                for requirement_id, claim_types, purpose in vehicle_reasoning
            )
            for requirement_id, source_kind, purpose in (
                (
                    "vehicle_vat_interpretation_analysis",
                    "interpretation",
                    "Omów direct, analogous lub contextual support z interpretacji "
                    "dotyczących tego samego mechanizmu art. 86a; różny pojazd albo "
                    "leasing nie wystarcza do odrzucenia analogii.",
                ),
                (
                    "vehicle_vat_judgment_analysis",
                    "judgment",
                    "Omów wyłącznie zweryfikowaną tezę lub rozumowanie sądu. Nie "
                    "przypisuj sądowi wyniku, którego nie potwierdza source span.",
                ),
            ):
                cards = [
                    card
                    for card in bundle.supporting_authorities
                    if _authority_source_kind(card) == source_kind
                ]
                if not cards:
                    continue
                requirements.append(
                    {
                        "requirement_id": requirement_id,
                        "citation": controlling_citation,
                        "allowed_provision_ids": allowed_provision_ids,
                        "allowed_claim_types": ["authority_pattern"],
                        "allowed_authority_ids": list(
                            dict.fromkeys(card.document_id for card in cards)
                        ),
                        "purpose": purpose,
                        "requires_explicit_acknowledgement": True,
                    }
                )

        issue_claims = claims_by_issue.get(issue.issue_id, [])
        missing_requirements: list[dict[str, Any]] = []
        for requirement in requirements:
            allowed_provisions = set(requirement["allowed_provision_ids"])
            allowed_authorities = set(requirement.get("allowed_authority_ids", []))
            allowed_claim_types = set(requirement.get("allowed_claim_types", []))
            explicit = bool(requirement.get("requires_explicit_acknowledgement"))
            fulfilled = any(
                (not allowed_claim_types or claim.claim_type in allowed_claim_types)
                and bool(allowed_provisions.intersection(claim.controlling_provision_ids))
                and (
                    not allowed_authorities
                    or bool(allowed_authorities.intersection(claim.supporting_authority_ids))
                )
                and (
                    not explicit
                    or requirement["requirement_id"]
                    in claim.coverage_requirement_ids
                )
                for claim in issue_claims
            )
            if not fulfilled:
                missing_requirements.append(requirement)
        if missing_requirements:
            result[issue.issue_id] = missing_requirements
    return result


def _should_use_best_effort_writer(
    *,
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
    claims: list[LegalClaim],
    answer_plan: AnswerPlan,
) -> bool:
    """Do not present a detected reasoning omission as a finished answer."""

    if not answer_plan.allowed_claim_ids:
        return True
    return bool(_claim_coverage_requirements(plan, bundles, claims))


def _is_income_tax_cost_issue(issue: Any) -> bool:
    text = " ".join(
        (
            str(issue.issue_id),
            str(issue.label),
            str(issue.legal_mechanism),
            *(str(item) for item in issue.possible_provision_concepts),
        )
    ).casefold()
    return bool(
        "cost_deductibility" in text
        or "contractual_penalty_cost" in text
        or "koszt uzyskania przychod" in text
        or "koszt podatkow" in text
    )


def _authority_source_kind(card: AuthorityCard) -> str:
    document_type = card.document_type.casefold()
    if document_type in {"interpretation", "general_interpretation"}:
        return "interpretation"
    if document_type in {"judgment", "resolution"}:
        return "judgment"
    return "other"


def _balanced_authority_extraction_candidates(
    candidates: Iterable[RetrievalCandidate],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    """Preserve at least one administrative and judicial source when found."""

    ordered = list(candidates)
    selected: list[RetrievalCandidate] = []
    selected_ids: set[str] = set()

    def candidate_kind(candidate: RetrievalCandidate) -> str:
        source_type = candidate.source_type.casefold()
        if source_type in {"interpretation", "general_interpretation"}:
            return "interpretation"
        if source_type in {"judgment", "resolution"}:
            return "judgment"
        return "other"

    for required_kind in ("interpretation", "judgment"):
        match = next(
            (
                candidate
                for candidate in ordered
                if candidate_kind(candidate) == required_kind
                and candidate.candidate_id not in selected_ids
            ),
            None,
        )
        if match is not None and len(selected) < limit:
            selected.append(match)
            selected_ids.add(match.candidate_id)
    for candidate in ordered:
        if len(selected) >= limit:
            break
        if candidate.candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)
    return selected


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
            # Source chunks often render a provision hierarchy as separate
            # lines ("Art. 17." / "1." / "4)").  The generic parser then
            # safely identifies the article but cannot reconstruct the full
            # displayed editorial unit.  Preserve that verified unit from the
            # corpus metadata as well, otherwise a request for art. 17(1)(4)
            # is incorrectly validated as a bare art. 17.
            displayed_reference = str(candidate.metadata.get("display_reference") or "").strip()
            if not units or (
                displayed_reference
                and not any(
                    _normalise_provision_citation(unit.citation)
                    == _normalise_provision_citation(displayed_reference)
                    for unit in units
                )
            ):
                units = (*units, _synthetic_provision_unit(candidate, version_id))
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


def _normalise_provision_citation(value: str) -> str:
    return " ".join(str(value).casefold().split()).strip(" .;:,")


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
            for match in [re.search(r"\bart\.\s*(\d+[a-z]*)", str(value), re.IGNORECASE)]
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

        classified_cards = [
            _classify_authority_for_issue(issue, card)
            for card in authority_cards.get(issue.issue_id, [])
            if _authority_card_has_material(card)
        ]
        vehicle_vat_issue = issue.issue_id == "mixed_use_vehicle_vat" or (
            "mixed_use_vehicle_vat" in str(issue.legal_mechanism).casefold()
        )
        cards = [
            card
            for card in classified_cards
            if not vehicle_vat_issue or card.evidence_relation != "unclassified"
        ]
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
        elif not cards:
            missing_sources.append("authority_material")
        requested_authorities = {
            str(value).casefold()
            for value in issue.requested_source_types
            if str(value).casefold()
            in {
                "interpretation",
                "general_interpretation",
                "guidance",
                "judgment",
                "resolution",
            }
        }
        if requested_authorities.intersection(
            {"interpretation", "general_interpretation"}
        ) and not any(_authority_source_kind(card) == "interpretation" for card in current_cards):
            missing_sources.append("authority_interpretation")
        if requested_authorities.intersection(
            {"judgment", "resolution"}
        ) and not any(_authority_source_kind(card) == "judgment" for card in current_cards):
            missing_sources.append("authority_judgment")
        required_dependency_patterns = _required_issue_dependency_patterns(issue)
        all_issue_provisions = (*controlling, *dependencies, *exceptions)
        missing_dependencies = [
            label
            for label, citation_pattern, document_pattern in required_dependency_patterns
            if not any(
                re.search(citation_pattern, item.citation, re.IGNORECASE)
                and re.search(document_pattern, item.document_id, re.IGNORECASE)
                for item in all_issue_provisions
            )
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


def _required_issue_dependency_patterns(
    issue: Any,
) -> tuple[tuple[str, str, str], ...]:
    issue_id = str(issue.issue_id)
    cit_act = r"podatku-dochodowym-od-osob-prawnych"
    pit_act = r"podatku-dochodowym-od-osob-fizycznych"
    vat_act = r"podatku-od-towarow"
    ufr_act = r"fundacji-rodzinnej"
    issue_concepts = " ".join(
        (
            issue_id,
            str(issue.legal_mechanism),
            *(str(item) for item in issue.possible_provision_concepts),
        )
    ).casefold()
    if issue_id == "cit_contractual_penalty_cost" or (
        "contractual_penalty" in issue_concepts
        or re.search(r"art\.\s*16\s+ust\.\s*1\s+pkt\s*22", issue_concepts)
    ):
        return (
            ("cit_art_15_1", r"art\.\s*15\s+ust\.\s*1(?:\s|$)", cit_act),
            ("cit_art_16_1_22", r"art\.\s*16\s+ust\.\s*1\s+pkt\s*22", cit_act),
        )
    if issue_id == "cit_cost_deductibility" or "cit_cost_deductibility" in issue_concepts:
        return (
            ("cit_art_15_1", r"art\.\s*15\s+ust\.\s*1(?:\s|$)", cit_act),
            ("cit_art_16_1", r"art\.\s*16\s+ust\.\s*1(?:\s|$)", cit_act),
        )
    if issue_id == "pit_cost_deductibility" or "pit_cost_deductibility" in issue_concepts:
        return (
            ("pit_art_22_1", r"art\.\s*22\s+ust\.\s*1(?:\s|$)", pit_act),
            ("pit_art_23_1", r"art\.\s*23\s+ust\.\s*1(?:\s|$)", pit_act),
        )
    if issue_id == "mixed_use_vehicle_vat" or "mixed_use_vehicle_vat" in issue_concepts:
        return (
            ("vat_art_86_1", r"art\.\s*86\s+ust\.\s*1(?:\s|$)", vat_act),
            ("vat_art_86a_1", r"art\.\s*86a\s+ust\.\s*1(?:\s|$)", vat_act),
            ("vat_art_86a_2_3", r"art\.\s*86a\s+ust\.\s*2\s+pkt\s*3", vat_act),
            (
                "vat_art_86a_3_1_a",
                r"art\.\s*86a\s+ust\.\s*3\s+pkt\s*1\s+lit\.\s*a",
                vat_act,
            ),
            ("vat_art_86a_4_1", r"art\.\s*86a\s+ust\.\s*4\s+pkt\s*1", vat_act),
            ("vat_art_86a_6", r"art\.\s*86a\s+ust\.\s*6(?:\s|$)", vat_act),
            ("vat_art_86a_12", r"art\.\s*86a\s+ust\.\s*12(?:\s|$)", vat_act),
        )
    if issue_id == "vat_input_deduction_timing" or "input_vat_deduction_timing" in issue_concepts:
        return (
            ("vat_art_86_1", r"art\.\s*86\s+ust\.\s*1(?:\s|$)", vat_act),
            ("vat_art_86_2_1", r"art\.\s*86\s+ust\.\s*2\s+pkt\s*1(?:\s|$)", vat_act),
            ("vat_art_86_10", r"art\.\s*86\s+ust\.\s*10(?:\s|$)", vat_act),
            ("vat_art_86_10b_1", r"art\.\s*86\s+ust\.\s*10b\s+pkt\s*1", vat_act),
            ("vat_art_86_10e", r"art\.\s*86\s+ust\.\s*10e(?:\s|$)", vat_act),
            ("vat_art_86_11", r"art\.\s*86\s+ust\.\s*11(?:\s|$)", vat_act),
            ("vat_art_86_13", r"art\.\s*86\s+ust\.\s*13(?:\s|$)", vat_act),
            ("vat_art_19a_1", r"art\.\s*19a\s+ust\.\s*1(?:\s|$)", vat_act),
        )
    if issue_id == "vat_invoice_channel_2026" or "invoice_delivery_channel_classification" in issue_concepts:
        return (
            ("vat_art_106ga_1", r"art\.\s*106ga\s+ust\.\s*1(?:\s|$)", vat_act),
            ("vat_art_106ga_2_1", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*1", vat_act),
            ("vat_art_106ga_2_2", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*2", vat_act),
            ("vat_art_106ga_2_3", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*3", vat_act),
            ("vat_art_106ga_2_4", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*4", vat_act),
            ("vat_art_106ga_2_5", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*5", vat_act),
            ("vat_art_106ga_2_6", r"art\.\s*106ga\s+ust\.\s*2\s+pkt\s*6", vat_act),
            ("vat_art_145m_1", r"art\.\s*145m\s+ust\.\s*1(?:\s|$)", vat_act),
            ("vat_art_145m_2", r"art\.\s*145m\s+ust\.\s*2(?:\s|$)", vat_act),
            ("vat_art_106na_3", r"art\.\s*106na\s+ust\.\s*3(?:\s|$)", vat_act),
            ("vat_art_106na_4", r"art\.\s*106na\s+ust\.\s*4(?:\s|$)", vat_act),
            ("vat_art_106nda_11", r"art\.\s*106nda\s+ust\.\s*11(?:\s|$)", vat_act),
            ("vat_art_106nf_10", r"art\.\s*106nf\s+ust\.\s*10(?:\s|$)", vat_act),
            ("vat_art_106nh_4", r"art\.\s*106nh\s+ust\.\s*4(?:\s|$)", vat_act),
            ("vat_art_106ng", r"art\.\s*106ng(?:\s|$)", vat_act),
        )
    if issue_id == "wht_pay_and_refund_procedure":
        return (
            ("art_26_2e", r"art\.\s*26\s+ust\.\s*2e", cit_act),
            ("art_26_2g", r"art\.\s*26\s+ust\.\s*2g", cit_act),
            ("art_26_7a", r"art\.\s*26\s+ust\.\s*7a", cit_act),
            ("art_26_7b", r"art\.\s*26\s+ust\.\s*7b", cit_act),
            ("art_26_7c", r"art\.\s*26\s+ust\.\s*7c", cit_act),
            ("art_26b", r"art\.\s*26b", cit_act),
            ("art_28b", r"art\.\s*28b", cit_act),
        )
    if issue_id == "vat_interest_financial_service":
        return (
            ("vat_art_28b", r"art\.\s*28b", vat_act),
            ("vat_art_17_1_4", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4", vat_act),
            ("vat_financial_exemption", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*38", vat_act),
        )
    if issue_id in {"vat_royalty_crossborder_service", "vat_management_crossborder_service"}:
        return (
            ("vat_art_28b", r"art\.\s*28b", vat_act),
            ("vat_art_17_1_4", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4", vat_act),
        )
    family_issue_id = family_foundation_issue_kind(issue)
    if family_issue_id == "family_foundation_investment_income":
        return (
            ("ufr_art_5_1_3", r"art\.\s*5\s+ust\.\s*1\s+pkt\s*3", ufr_act),
            ("ufr_art_5_1_4", r"art\.\s*5\s+ust\.\s*1\s+pkt\s*4", ufr_act),
            ("cit_art_6_1_25", r"art\.\s*6\s+ust\.\s*1\s+pkt\s*25", cit_act),
            ("cit_art_6_6", r"art\.\s*6\s+ust\.\s*6", cit_act),
            ("cit_art_6_7", r"art\.\s*6\s+ust\.\s*7", cit_act),
        )
    if family_issue_id == "family_foundation_related_party_rent":
        return (
            ("ufr_art_5_1_2", r"art\.\s*5\s+ust\.\s*1\s+pkt\s*2", ufr_act),
            ("cit_art_6_8", r"art\.\s*6\s+ust\.\s*8", cit_act),
            ("cit_art_19_1", r"art\.\s*19\s+ust\.\s*1", cit_act),
            ("cit_art_24q_8", r"art\.\s*24q\s+ust\.\s*8", cit_act),
        )
    if family_issue_id == "family_foundation_related_party_services":
        return (
            ("cit_art_24q_1", r"art\.\s*24q\s+ust\.\s*1(?:\s|$)", cit_act),
            ("cit_art_24q_1a_3", r"art\.\s*24q\s+ust\.\s*1a\s+pkt\s*3", cit_act),
        )
    if family_issue_id == "family_foundation_borrowing_from_related_party":
        return (
            ("cit_art_24q_1", r"art\.\s*24q\s+ust\.\s*1(?:\s|$)", cit_act),
            ("cit_art_24q_1a_1", r"art\.\s*24q\s+ust\.\s*1a\s+pkt\s*1", cit_act),
        )
    if family_issue_id == "family_foundation_beneficiary_loan":
        return (
            ("ufr_art_5_1_5_c", r"art\.\s*5\s+ust\.\s*1\s+pkt\s*5\s+lit\.\s*c", ufr_act),
            ("cit_art_24q_1", r"art\.\s*24q\s+ust\.\s*1(?:\s|$)", cit_act),
            ("cit_art_24q_1a_2", r"art\.\s*24q\s+ust\.\s*1a\s+pkt\s*2", cit_act),
            ("cit_art_24q_1a_5", r"art\.\s*24q\s+ust\.\s*1a\s+pkt\s*5", cit_act),
            ("cit_art_24q_1a_6", r"art\.\s*24q\s+ust\.\s*1a\s+pkt\s*6", cit_act),
            ("cit_art_24q_2", r"art\.\s*24q\s+ust\.\s*2", cit_act),
        )
    if family_issue_id == "family_foundation_beneficiary_benefit":
        return (
            ("ufr_art_2_2", r"art\.\s*2\s+ust\.\s*2", ufr_act),
            ("cit_art_24q_1", r"art\.\s*24q\s+ust\.\s*1(?:\s|$)", cit_act),
            ("pit_art_21_1_157", r"art\.\s*21\s+ust\.\s*1\s+pkt\s*157", pit_act),
            ("pit_art_21_49", r"art\.\s*21\s+ust\.\s*49", pit_act),
            ("pit_art_30_1_17", r"art\.\s*30\s+ust\.\s*1\s+pkt\s*17", pit_act),
        )
    if family_issue_id == "family_foundation_real_estate_activity":
        return (
            ("ufr_art_5_1_1", r"art\.\s*5\s+ust\.\s*1\s+pkt\s*1", ufr_act),
            ("cit_art_6_7", r"art\.\s*6\s+ust\.\s*7", cit_act),
            ("cit_art_24r_1", r"art\.\s*24r\s+ust\.\s*1", cit_act),
            ("cit_art_15_2", r"art\.\s*15\s+ust\.\s*2", cit_act),
        )
    if family_issue_id == "family_foundation_common_costs":
        return (
            ("cit_art_15_2", r"art\.\s*15\s+ust\.\s*2", cit_act),
            ("cit_art_24r_2", r"art\.\s*24r\s+ust\.\s*2", cit_act),
        )
    if family_issue_id == "family_foundation_tax_credit_and_reporting":
        return (
            ("cit_art_24q_6", r"art\.\s*24q\s+ust\.\s*6", cit_act),
            ("cit_art_24q_8", r"art\.\s*24q\s+ust\.\s*8", cit_act),
            ("cit_art_24q_9", r"art\.\s*24q\s+ust\.\s*9", cit_act),
            ("cit_art_24s_1", r"art\.\s*24s\s+ust\.\s*1", cit_act),
        )
    if family_issue_id == "family_foundation_vat_transactions":
        return (
            ("vat_art_15_1", r"art\.\s*15\s+ust\.\s*1", vat_act),
            ("vat_art_15_2", r"art\.\s*15\s+ust\.\s*2", vat_act),
            ("vat_art_32_1", r"art\.\s*32\s+ust\.\s*1", vat_act),
            ("vat_art_43_1_10", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*10(?:\s|$)", vat_act),
            ("vat_art_43_1_10a", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*10a", vat_act),
            ("vat_art_43_1_36", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*36", vat_act),
            ("vat_art_86_1", r"art\.\s*86\s+ust\.\s*1", vat_act),
            ("vat_art_90_1", r"art\.\s*90\s+ust\.\s*1", vat_act),
            ("vat_art_90_2", r"art\.\s*90\s+ust\.\s*2", vat_act),
        )
    if family_issue_id == "family_foundation_allowed_activity_catalog":
        return (("ufr_art_5", r"art\.\s*5(?:\s|$)", ufr_act),)
    if family_issue_id == "family_foundation_cit_hidden_profit":
        return (
            ("cit_art_24q_1", r"art\.\s*24q\s+ust\.\s*1(?:\s|$)", cit_act),
            ("ufr_art_2_2", r"art\.\s*2\s+ust\.\s*2", ufr_act),
        )
    if family_issue_id == "family_foundation_disallowed_income_25_percent":
        return (
            ("cit_art_24r", r"art\.\s*24r(?:\s|$)", cit_act),
            ("cit_art_15_2", r"art\.\s*15\s+ust\.\s*2", cit_act),
            ("ufr_art_5", r"art\.\s*5(?:\s|$)", ufr_act),
        )
    if family_issue_id == "family_foundation_beneficiary_pit":
        return (
            ("pit_art_20_1g", r"art\.\s*20\s+ust\.\s*1g", pit_act),
            ("pit_art_21_1_157", r"art\.\s*21\s+ust\.\s*1\s+pkt\s*157", pit_act),
            ("pit_art_30_1_17", r"art\.\s*30\s+ust\.\s*1\s+pkt\s*17", pit_act),
            ("ufr_art_2_2", r"art\.\s*2\s+ust\.\s*2", ufr_act),
            ("ufr_art_27_4", r"art\.\s*27\s+ust\.\s*4", ufr_act),
            ("ufr_art_28_1", r"art\.\s*28\s+ust\.\s*1", ufr_act),
            ("ufr_art_29_1", r"art\.\s*29\s+ust\.\s*1", ufr_act),
        )
    if family_issue_id == "family_foundation_vat_related_party":
        return (
            ("vat_art_15_1", r"art\.\s*15\s+ust\.\s*1", vat_act),
            ("vat_art_15_2", r"art\.\s*15\s+ust\.\s*2", vat_act),
            ("vat_art_29a_1", r"art\.\s*29a\s+ust\.\s*1", vat_act),
            ("vat_art_32_1", r"art\.\s*32\s+ust\.\s*1", vat_act),
            ("vat_art_32_2", r"art\.\s*32\s+ust\.\s*2", vat_act),
            ("vat_art_43_1_36", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*36", vat_act),
            ("vat_art_86_1", r"art\.\s*86\s+ust\.\s*1", vat_act),
        )
    transfer_pricing_haystack = " ".join(
        (
            issue.issue_id,
            issue.label,
            issue.legal_mechanism,
            *issue.possible_provision_concepts,
        )
    )
    if question_targets_transfer_pricing(transfer_pricing_haystack):
        return (
            ("cit_art_11k_1", r"art\.\s*11k\s+ust\.\s*1", cit_act),
            ("cit_art_11k_2", r"art\.\s*11k\s+ust\.\s*2", cit_act),
            ("cit_art_11k_3", r"art\.\s*11k\s+ust\.\s*3", cit_act),
            ("cit_art_11l_1", r"art\.\s*11l\s+ust\.\s*1", cit_act),
            ("cit_art_11n_1", r"art\.\s*11n\s+pkt\s*1", cit_act),
            ("cit_art_11t_1", r"art\.\s*11t\s+ust\.\s*1", cit_act),
        )
    # Never certify an arbitrary top semantic hit as controlling law for an
    # issue that the planner itself left completely unscoped.  Mechanism
    # enrichers resolve recognized questions before this point; an unknown
    # generic issue remains visibly partial instead of citing a random article.
    has_exact_target = any(
        family.lane in {"primary_law", "both"}
        and family.family in {"explicit_provision_reference", "explicit_provision"}
        for family in issue.query_families
    )
    if issue.legal_mechanism == "general_tax_analysis" and not has_exact_target:
        return (
            ("unresolved_generic_issue", r"(?!)", r".*"),
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
    provision_references_by_issue = {
        bundle.issue_id: {
            item.provision_id: item
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
        bundle = bundle_by_issue.get(claim.issue_id)
        claim, auto_bound = _auto_bind_exact_claim_references(
            claim,
            provision_references_by_issue.get(claim.issue_id, {}),
        )
        warnings.extend(
            f"{claim.claim_id}:auto_bound_exact_provision:{citation}"
            for citation in auto_bound
        )
        if claim.claim_id in seen:
            claim_errors.append("duplicate_claim_id")
        seen.add(claim.claim_id)
        if claim.issue_id not in issues:
            claim_errors.append("unknown_issue_id")
        if not set(claim.controlling_provision_ids).issubset(
            provisions_by_issue.get(claim.issue_id, set())
        ):
            claim_errors.append("unknown_controlling_provision")
        if (
            claim.status in {"approved", "conditional_missing_fact"}
            and not claim.controlling_provision_ids
        ):
            claim_errors.append("material_claim_without_primary_law")
        bound_references = [
            provision_references_by_issue.get(claim.issue_id, {}).get(provision_id)
            for provision_id in claim.controlling_provision_ids
        ]
        unbound_textual_references = _unbound_claim_provision_references(
            f"{claim.text} {claim.result}",
            [item.citation for item in bound_references if item is not None],
        )
        if unbound_textual_references:
            claim_errors.append(
                "unbound_textual_provision_reference:"
                + ",".join(unbound_textual_references)
            )
        claim_text = f"{claim.text} {claim.result}"
        bound_text = " ".join(
            item.text or "" for item in bound_references if item is not None
        )
        bundle_text = " ".join(
            item.text or ""
            for item in (
                *bundle.controlling_provisions,
                *bundle.dependency_provisions,
                *bundle.exception_provisions,
            )
        ) if bundle is not None else bound_text
        if (
            re.search(r"\b(?:nie\s+wynika|brak\w*|nie\s+określa\w*).{0,80}\bstawk", claim_text, re.I)
            and re.search(r"\bwynosi\s+\d+(?:[,.]\d+)?\s*(?:%|procent)", bundle_text, re.I)
        ):
            claim_errors.append("claim_denies_rate_expressly_present_in_primary_law")
        bundle_has_complete_primary = bool(bundle) and not any(
            source == "primary_law" or source.startswith("required_primary:")
            for source in bundle.missing_sources
        )
        if bundle_has_complete_primary and re.search(
            r"(?:brak\s+(?:jest\s+)?(?:pierwotn\w*\s+)?przepis\w*|"
            r"materiał\s+nie\s+zawiera\s+(?:pełn\w*\s+)?art\.|"
            r"niewłączon\w*\s+do\s+materiału\s+(?:przesłan\w*|przepis\w*))",
            claim_text,
            re.I,
        ):
            claim_errors.append("claim_denies_primary_law_present_in_issue_bundle")
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
                result=(
                    "Wniosek pozostaje zablokowany do czasu uzyskania kompletnego, "
                    "zweryfikowanego łańcucha źródeł."
                ),
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


def _ensure_required_issue_claims(
    claims: list[LegalClaim],
    *,
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
    calculations: list[CalculationRecord],
) -> tuple[list[LegalClaim], list[str]]:
    """Close mandatory, source-complete issue outputs deterministically.

    The model is allowed to qualify or block a legal conclusion, but it must
    not silently omit an entire issue whose EvidenceBundle is complete.  In
    particular, this guarantees the B2B VAT licence rule and code-produced
    pay-and-refund calculation are rendered even when structured synthesis
    decides to focus on a different payment stream.
    """
    result = list(claims)
    warnings: list[str] = []
    allowed_statuses = {"approved", "conditional_missing_fact"}
    approved_by_issue = {
        item.issue_id for item in result if item.status in allowed_statuses
    }
    bundles_by_issue = {item.issue_id: item for item in bundles}

    # A model-produced claim can be rejected because it bound the right legal
    # proposition to a wrong provision ID.  That must not erase a complete,
    # independently verified WHT bundle.  These minimum claims are generated
    # only from the exact primary-law references already selected for the
    # issue; they never infer beneficial-owner or permanent-establishment
    # facts and remain explicitly conditional.
    wht_templates = {
        "wht_interest_pl_de_treaty": {
            "required": (r"art\.\s*11", r"art\.\s*21\s+ust\.\s*1\s+pkt\s*1"),
            "text": (
                "Odsetki wypłacane niemieckiej GmbH należą do krajowej kategorii WHT z "
                "art. 21 ust. 1 pkt 1 ustawy o CIT. Art. 11 UPO Polska–Niemcy "
                "dopuszcza opodatkowanie w państwie źródła, lecz przewiduje preferencję "
                "wyłącznie dla osoby uprawnionej mającej rezydencję w drugim państwie."
            ),
            "result": (
                "Stawka traktatowa {rate} ma charakter warunkowy; bez potwierdzenia statusu "
                "beneficial owner nie można zatwierdzić jej zastosowania."
            ),
        },
        "wht_royalties_pl_de_treaty": {
            "required": (r"art\.\s*12", r"art\.\s*21\s+ust\.\s*1\s+pkt\s*1"),
            "text": (
                "Należności za korzystanie z praw własności intelektualnej mogą należeć do "
                "krajowej kategorii WHT z art. 21 ust. 1 pkt 1 ustawy o CIT. Art. 12 UPO "
                "Polska–Niemcy ogranicza podatek u źródła, gdy odbiorca jest osobą uprawnioną "
                "mającą rezydencję w drugim państwie."
            ),
            "result": (
                "Stawka traktatowa {rate} ma charakter warunkowy; potwierdź zakres licencji "
                "oraz status beneficial owner przed zastosowaniem preferencji."
            ),
        },
        "wht_services_pl_de_business_profits": {
            "required": (r"art\.\s*7", r"art\.\s*21\s+ust\.\s*1\s+pkt\s*2a"),
            "text": (
                "Usługi zarządzania i kontroli są objęte krajowym reżimem art. 21 ust. 1 pkt 2a "
                "ustawy o CIT. Dla zastosowania UPO Polska–Niemcy należy odrębnie ocenić "
                "regułę zysków przedsiębiorstw z art. 7 i związek świadczenia z zakładem."
            ),
            "result": (
                "Bez ustalenia, czy GmbH prowadzi działalność przez zakład w Polsce i czy "
                "wynagrodzenie jest z nim związane, nie można zatwierdzić niepobrania WHT na "
                "podstawie art. 7 UPO."
            ),
        },
    }
    for issue_id, template in wht_templates.items():
        bundle = bundles_by_issue.get(issue_id)
        if (
            bundle is None
            or bundle.coverage_status != "complete"
            or issue_id in approved_by_issue
        ):
            continue
        references = [
            item
            for item in (
                *bundle.controlling_provisions,
                *bundle.dependency_provisions,
                *bundle.exception_provisions,
            )
            if item.source_span is not None
        ]
        if not references or not all(
            any(re.search(pattern, item.citation, re.I) for item in references)
            for pattern in template["required"]
        ):
            continue
        treaty_article = "11" if issue_id == "wht_interest_pl_de_treaty" else "12"
        source_text = "\n".join(
            item.text or ""
            for item in references
            if re.search(rf"art\.\s*{treaty_article}(?!\d)", item.citation, re.I)
        )
        # OCR commonly joins the Polish words (``5procentkwoty``).  The
        # official PL-DE PDF is bilingual and its German column can be the
        # only intact rendering of the same norm (``5 vom Hundert``).
        rate_match = re.search(
            r"(?<!\d)(\d{1,2})\s*(?:procent|%|vom\s+Hundert)",
            source_text,
            re.I,
        )
        rate = f"{rate_match.group(1)}%" if rate_match else "przewidziana w UPO"
        result.append(
            LegalClaim(
                claim_id=f"deterministic_{issue_id}_primary_bundle",
                issue_id=issue_id,
                claim_type="application",
                text=template["text"],
                status="conditional_missing_fact",
                result=template["result"].format(rate=rate),
                controlling_provision_ids=[item.provision_id for item in references],
                source_spans=[item.source_span for item in references if item.source_span],
                confidence=0.93,
            )
        )
        approved_by_issue.add(issue_id)
        warnings.append(f"{issue_id}:deterministic_complete_primary_bundle_claim")

    vat_templates = {
        "vat_interest_financial_service": {
            "required": (r"art\.\s*28b", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4", r"art\.\s*43\s+ust\.\s*1\s+pkt\s*38"),
            "text": (
                "Jeżeli odsetki są wynagrodzeniem za udzielenie pożyczki przez GmbH, "
                "usługa korzysta ze zwolnienia z art. 43 ust. 1 pkt 38 ustawy o VAT. "
                "W relacji B2B miejsce świadczenia jest co do zasady w Polsce, a przy "
                "spełnieniu art. 17 ust. 1 pkt 4 polska spółka rozlicza import usług."
            ),
            "result": (
                "Potwierdź status VAT stron, rolę GmbH jako usługodawcy pożyczki oraz brak "
                "jej polskiego stałego miejsca prowadzenia działalności dla tej transakcji."
            ),
        },
        "vat_royalty_crossborder_service": {
            "required": (r"art\.\s*28b", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4"),
            "text": (
                "Jeżeli polska spółka jest podatnikiem VAT nabywającym licencję od niemieckiej "
                "GmbH, miejsce świadczenia B2B jest co do zasady w Polsce, a nabywca rozlicza "
                "import usług po spełnieniu ustawowych warunków."
            ),
            "result": (
                "Dla licencji należy przyjąć warunkowo polskie miejsce świadczenia i import usług; "
                "potwierdź status VAT stron oraz brak polskiego stałego miejsca prowadzenia "
                "działalności GmbH."
            ),
        },
        "vat_management_crossborder_service": {
            "required": (r"art\.\s*28b", r"art\.\s*17\s+ust\.\s*1\s+pkt\s*4"),
            "text": (
                "Dla usług zarządzania świadczonych na rzecz polskiej spółki będącej podatnikiem, "
                "miejscem świadczenia jest co do zasady Polska. Przy spełnieniu warunków art. 17 "
                "ust. 1 pkt 4 polska spółka rozlicza VAT jako usługobiorca."
            ),
            "result": (
                "Import usług jest warunkowo właściwy; potwierdź status VAT polskiej spółki oraz "
                "brak siedziby lub stałego miejsca prowadzenia działalności GmbH w Polsce."
            ),
        },
    }
    for issue_id, template in vat_templates.items():
        vat_bundle = bundles_by_issue.get(issue_id)
        if (
            vat_bundle is None
            or vat_bundle.coverage_status != "complete"
            or issue_id in approved_by_issue
        ):
            continue
        references = [
            item
            for item in (
                *vat_bundle.controlling_provisions,
                *vat_bundle.dependency_provisions,
                *vat_bundle.exception_provisions,
            )
            if item.source_span is not None
        ]
        if references and all(
            any(re.search(pattern, item.citation, re.I) for item in references)
            for pattern in template["required"]
        ):
            result.append(
                LegalClaim(
                    claim_id=f"deterministic_{issue_id}_primary_bundle",
                    issue_id=issue_id,
                    claim_type="application",
                    text=template["text"],
                    status="conditional_missing_fact",
                    result=template["result"],
                    controlling_provision_ids=[item.provision_id for item in references],
                    source_spans=[item.source_span for item in references if item.source_span],
                    confidence=0.82,
                )
            )
            approved_by_issue.add(issue_id)
            warnings.append(f"{issue_id}:deterministic_complete_bundle_claim")

    used_calculations = {
        calculation_id
        for item in result
        if item.status in allowed_statuses
        for calculation_id in item.calculation_ids
    }
    for calculation in calculations:
        if calculation.calculation_id in used_calculations:
            continue
        issue_id = next(
            (item for item in calculation.dependencies if item in bundles_by_issue),
            "wht_pay_and_refund_procedure",
        )
        references = [item for item in calculation.legal_basis if item.source_span is not None]
        if not references:
            continue
        total = calculation.inputs.get("aggregate_payments")
        threshold = calculation.inputs.get("threshold_base")
        excess = calculation.inputs.get("excess")
        domestic_wht = calculation.inputs.get("domestic_wht", calculation.result)
        if not all(isinstance(item, (int, float)) for item in (total, threshold, excess, domestic_wht)):
            continue
        result.append(
            LegalClaim(
                claim_id=f"deterministic_{calculation.calculation_id}",
                issue_id=issue_id,
                claim_type="calculation",
                text=(
                    "Łączne płatności wynoszą "
                    f"{_format_pln(total)}, więc przekraczają próg {_format_pln(threshold)} "
                    f"o {_format_pln(excess)}."
                ),
                status="approved",
                result=(
                    "Przy krajowej stawce 20% obowiązkowy pobór od nadwyżki wynosi "
                    f"{_format_pln(domestic_wht)}."
                ),
                controlling_provision_ids=[item.provision_id for item in references],
                calculation_ids=[calculation.calculation_id],
                source_spans=[item.source_span for item in references if item.source_span],
                confidence=0.99,
            )
        )
        warnings.append(f"{calculation.calculation_id}:deterministic_calculation_claim")
    return result, warnings


def _format_pln(value: int | float) -> str:
    return f"{int(round(value)):,}".replace(",", " ") + " zł"


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
    sections: list[AnswerSection] = []
    for issue in plan.issues:
        issue_claim_ids = [
            claim.claim_id
            for claim in claims
            if claim.issue_id == issue.issue_id
            and claim.status in {"approved", "conditional_missing_fact"}
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
    required_claims = {
        claim_id
        for section in answer_plan.sections
        for claim_id in section.required_claim_ids
        if claim_id in allowed_claims
    }
    if required_claims - used:
        errors.append("writer_omitted_required_claim")
    # A provision ID can identify a source record that exposes several
    # editorial units (for example art. 21 and art. 26 from the same statute
    # source). Keep every verified citation for that ID; a scalar map silently
    # overwrote earlier units and caused false integrity failures.
    allowed_source_citations: dict[str, set[str]] = {}
    for item in (
        provision
        for bundle in bundles
        for provision in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        )
    ):
        allowed_source_citations.setdefault(item.provision_id, set()).add(
            _normalize_citation(item.citation)
        )
    for item in (
        authority
        for bundle in bundles
        for authority in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
            *bundle.historical_authorities,
        )
    ):
        allowed_source_citations.setdefault(item.document_id, set()).add(
            _normalize_citation(item.signature or item.document_id)
        )
    if any(item.source_id not in allowed_source_citations for item in output.sources):
        errors.append("writer_used_unknown_source")
    if any(
        item.source_id in allowed_source_citations
        and _normalize_citation(item.citation)
        not in allowed_source_citations[item.source_id]
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


def _best_effort_model_evidence(
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
) -> dict[str, Any]:
    provisions: list[dict[str, str]] = []
    authorities: list[dict[str, str]] = []
    seen_provisions: set[tuple[str, str]] = set()
    seen_authorities: set[str] = set()
    remaining_chars = 42_000
    for bundle in bundles:
        for provision in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        ):
            key = (provision.provision_id, _normalize_citation(provision.citation))
            if key in seen_provisions or remaining_chars <= 0:
                continue
            seen_provisions.add(key)
            text = " ".join((provision.text or "").split())[:5_000]
            remaining_chars -= len(text)
            provisions.append(
                {
                    "source_id": provision.provision_id,
                    "issue_id": bundle.issue_id,
                    "act": _writer_source_label(provision),
                    "citation": provision.citation,
                    "text": text,
                }
            )
        for authority in (*bundle.supporting_authorities, *bundle.contrary_authorities):
            if authority.document_id in seen_authorities or remaining_chars <= 0:
                continue
            seen_authorities.add(authority.document_id)
            material = " ".join(
                value
                for value in (
                    *authority.facts,
                    *authority.issues,
                    authority.authority_holding,
                    authority.court_holding,
                    authority.reasoning,
                    authority.outcome,
                    *authority.distinguishing_facts,
                )
                if value
            )[:3_000]
            remaining_chars -= len(material)
            authorities.append(
                {
                    "source_id": authority.document_id,
                    "issue_id": bundle.issue_id,
                    "type": authority.document_type,
                    "evidence_relation": authority.evidence_relation,
                    "signature": authority.signature or authority.document_id,
                    "authority": authority.authority,
                    "court": authority.court,
                    "date": authority.date,
                    "facts": authority.facts,
                    "issues": authority.issues,
                    "authority_holding": authority.authority_holding or "",
                    "court_holding": authority.court_holding or "",
                    "reasoning": authority.reasoning or "",
                    "outcome": authority.outcome or "",
                    "distinguishing_facts": authority.distinguishing_facts,
                    "holding_verified": bool(
                        authority.authority_holding
                        or authority.court_holding
                        or authority.reasoning
                    ),
                    "material": material,
                }
            )
    return {
        "issues": [
            {
                "issue_id": issue.issue_id,
                "label": issue.label,
                "mechanism": issue.legal_mechanism,
            }
            for issue in plan.issues
        ],
        "primary_law": provisions,
        "authorities": authorities,
        "missing_facts": [
            {
                "fact_id": fact.fact_id,
                "question": fact.question,
                "materiality": fact.materiality,
            }
            for fact in plan.missing_facts
        ],
    }


def _authority_card_has_material(card: AuthorityCard) -> bool:
    return any(
        str(value or "").strip()
        for value in (
            card.authority_holding,
            card.court_holding,
            card.reasoning,
            card.outcome,
        )
    )


def _classify_authority_for_issue(issue: Any, card: AuthorityCard) -> AuthorityCard:
    if not (
        str(issue.issue_id) == "mixed_use_vehicle_vat"
        or "mixed_use_vehicle_vat" in str(issue.legal_mechanism).casefold()
    ):
        return card
    text = " ".join(
        str(value or "")
        for value in (
            *card.facts,
            *card.issues,
            *card.cited_provisions,
            card.authority_holding,
            card.court_holding,
            card.reasoning,
            card.outcome,
            *card.distinguishing_facts,
        )
    ).casefold()
    has_vehicle_mechanism = bool(
        re.search(r"art\.\s*86a|pojazd\w*|samoch[oó]d\w*|motocykl\w*", text, re.I)
    )
    has_fifty_percent_rule = bool(
        re.search(r"(?:50\s*%|pięćdziesi[ąa]t\w*\s+procent|połow\w*\s+podat)", text, re.I)
    )
    has_mixed_use = bool(
        re.search(r"użytk\w*\s+mieszan\w*|cel\w*\s+prywatn\w*|prywatn\w*", text, re.I)
    )
    has_fuel_or_operation = bool(
        re.search(r"paliw\w*|wydatk\w*\s+eksploatacyjn\w*|eksploatacj\w*", text, re.I)
    )
    analogous_fact = bool(re.search(r"motocykl\w*|leasing\w*", text, re.I))
    full_deduction_context = bool(
        re.search(
            r"100\s*%|pełn\w*\s+odliczeni\w*|wyłącznie\s+do\s+działalno\w*|"
            r"VAT-?26|ewidencj\w*\s+przebieg\w*",
            text,
            re.I,
        )
    )
    relation = "unclassified"
    if has_vehicle_mechanism and has_fifty_percent_rule and (
        has_mixed_use or has_fuel_or_operation
    ):
        relation = "analogous_support" if analogous_fact else "direct_support"
    elif has_vehicle_mechanism and full_deduction_context:
        relation = "context_for_full_deduction"
    return card.model_copy(update={"evidence_relation": relation})


def _scrub_unverified_best_effort_references(
    text: str,
    *,
    citations: Iterable[str],
    signatures: Iterable[str],
) -> str:
    allowed_citations = list(citations)

    def provision_replacement(match: re.Match[str]) -> str:
        reference = match.group(0)
        if any(_citations_are_compatible(reference, citation) for citation in allowed_citations):
            return reference
        return "właściwy przepis"

    result = _RENDERED_PROVISION_RE.sub(provision_replacement, text)
    known_signatures = {" ".join(value.casefold().split()) for value in signatures if value}

    def signature_replacement(match: re.Match[str]) -> str:
        signature = " ".join(match.group(0).casefold().split())
        return match.group(0) if signature in known_signatures else "orzecznictwo sądowe"

    return re.sub(
        r"\b(?:I|II)?\s*(?:FSK|SA/[A-Z]{1,3})\s+\d+/\d+\b",
        signature_replacement,
        result,
        flags=re.I,
    )


def _best_effort_writer_output(
    raw: str,
    *,
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
) -> WriterOutput:
    cleaned = re.sub(r"```(?:json|markdown)?|```", "", raw, flags=re.I)
    cleaned = cleaned.replace("**", "").strip()
    parts = re.split(
        r"(?im)^\s*(TEZA|ANALIZA|DOKUMENTY|WYKORZYSTANE_ŹRÓDŁA)\s*:\s*",
        cleaned,
    )
    parsed: dict[str, str] = {}
    for index in range(1, len(parts) - 1, 2):
        parsed[parts[index].casefold()] = parts[index + 1].strip()
    thesis = parsed.get("teza") or cleaned.splitlines()[0].strip()
    analysis = parsed.get("analiza") or cleaned
    documents = parsed.get("dokumenty", "")
    if documents:
        analysis = f"{analysis}\n\nDokumentacja i działania:\n{documents}"

    valid_source_ids = {
        source_id
        for bundle in bundles
        for source_id in (
            *(
                item.provision_id
                for item in (
                    *bundle.controlling_provisions,
                    *bundle.dependency_provisions,
                    *bundle.exception_provisions,
                )
            ),
            *(
                item.document_id
                for item in (
                    *bundle.supporting_authorities,
                    *bundle.contrary_authorities,
                )
            ),
        )
    }
    raw_used_sources = parsed.get("wykorzystane_źródła", "")
    used_source_tokens = {
        token.strip().strip("-*` ").casefold()
        for token in re.split(r"[,;\n]", raw_used_sources)
        if token.strip()
    }
    used_source_ids = {
        source_id
        for source_id in valid_source_ids
        if source_id.casefold() in used_source_tokens
    }
    required_primary_ids = _best_effort_required_primary_source_ids(plan, bundles)
    issue_by_id = {issue.issue_id: issue for issue in plan.issues}
    sources: list[WriterSource] = []
    for bundle in bundles:
        issue = issue_by_id.get(bundle.issue_id)
        cost_issue = bool(issue and _is_income_tax_cost_issue(issue))
        for provision in (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        ):
            if provision.provision_id not in required_primary_ids and (
                provision.provision_id not in used_source_ids
                or cost_issue
            ):
                continue
            sources.append(
                WriterSource(
                    source_id=provision.provision_id,
                    label=_writer_source_label(provision),
                    citation=provision.citation,
                    claim_ids=[],
                )
            )
        for authority in (*bundle.supporting_authorities, *bundle.contrary_authorities):
            if authority.document_id not in used_source_ids:
                continue
            signature = authority.signature or authority.document_id
            sources.append(
                WriterSource(
                    source_id=authority.document_id,
                    label=authority.document_type,
                    citation=signature,
                    claim_ids=[],
                )
            )
    sources = _dedupe_writer_sources(sources)[:30]
    rendered_citations = [source.citation for source in sources]
    rendered_signatures = [
        source.citation for source in sources if source.label not in {"Ustawa o CIT", "Ustawa o PIT", "Ustawa o VAT", "Ustawa o fundacji rodzinnej", "UPO", "UPO Polska–Niemcy", "Przepis"}
    ]
    thesis = _scrub_unverified_best_effort_references(
        thesis, citations=rendered_citations, signatures=rendered_signatures
    )
    analysis = _scrub_unverified_best_effort_references(
        analysis, citations=rendered_citations, signatures=rendered_signatures
    )
    risks = [
        "Wniosek ma charakter wstępny: część subsumpcji została przygotowana "
        "poza ścisłym trybem weryfikacji źródłowej i wymaga sprawdzenia przed "
        "podjęciem decyzji podatkowej."
    ]
    missing = {
        source
        for bundle in bundles
        for source in bundle.missing_sources
        if source
    }
    if "authority_interpretation" in missing:
        risks.append(
            "Nie znaleziono interpretacji podatkowej zawierającej materialne "
            "stanowisko dostatecznie związane z analizowanym problemem."
        )
    if "authority_judgment" in missing:
        risks.append(
            "Nie znaleziono orzeczenia sądu zawierającego materialne rozumowanie "
            "dostatecznie związane z analizowanym problemem."
        )
    if "primary_law" in missing or any(
        source.startswith("required_primary:") for source in missing
    ):
        risks.append(
            "Nie potwierdzono pełnego zestawu wymaganych przepisów prawa pierwotnego."
        )
    return WriterOutput(
        thesis=thesis.strip(),
        analysis_sections=[
            WriterAnalysisSection(
                section_id="best_effort_analysis",
                title="Wstępna analiza materialna",
                content=analysis.strip(),
                claim_ids_used=[],
            )
        ],
        sources=sources,
        risks_and_gaps=risks,
        claim_ids_used=[],
    )


def _best_effort_required_primary_source_ids(
    plan: LegalResearchPlan,
    bundles: list[EvidenceBundle],
) -> set[str]:
    """Keep the controlling statutory chain without dumping neighbouring units."""

    issue_by_id = {issue.issue_id: issue for issue in plan.issues}
    result: set[str] = set()
    for bundle in bundles:
        provisions = (
            *bundle.controlling_provisions,
            *bundle.dependency_provisions,
            *bundle.exception_provisions,
        )
        issue = issue_by_id.get(bundle.issue_id)
        if issue is None:
            continue
        matches = [
            min(
                matching,
                key=lambda item: (len(_normalize_citation(item.citation)), item.citation),
            )
            for _, citation_pattern, document_pattern in _required_issue_dependency_patterns(issue)
            if (
                matching := [
                    provision
                    for provision in provisions
                    if re.search(citation_pattern, provision.citation, re.I)
                    and re.search(document_pattern, provision.document_id, re.I)
                ]
            )
        ]
        if matches:
            result.update(item.provision_id for item in matches)
        elif bundle.controlling_provisions:
            result.add(bundle.controlling_provisions[0].provision_id)
    return result


def _deterministic_writer_output(payload: dict[str, Any]) -> WriterOutput:
    claims: list[LegalClaim] = payload["validated_claims"]
    plan: LegalResearchPlan = payload["legal_research_plan"]
    bundles: list[EvidenceBundle] = payload["evidence_bundles"]
    answer_plan: AnswerPlan = payload["answer_plan"]
    allowed = set(answer_plan.allowed_claim_ids)
    selected = [item for item in claims if item.claim_id in allowed]
    coverage_requirements = _claim_coverage_requirements(plan, bundles)
    complete_primary_issues = {
        bundle.issue_id
        for bundle in bundles
        if bundle.coverage_status == "complete" and bundle.controlling_provisions
    }
    fallback_source_ids = {
        provision_id
        for requirements in coverage_requirements.values()
        for requirement in requirements
        for provision_id in requirement["allowed_provision_ids"][:1]
    }
    fallback_source_ids.update(
        bundle.controlling_provisions[0].provision_id
        for bundle in bundles
        if bundle.issue_id in complete_primary_issues and bundle.controlling_provisions
    )
    # A provider/schema failure must not turn the thesis into a dump of every
    # normative definition.  Prefer one applied outcome per issue, then fall
    # back to a normative result only when no application claim exists.
    thesis_claims: list[LegalClaim] = []
    for issue in plan.issues:
        issue_claims = [item for item in selected if item.issue_id == issue.issue_id]
        preferred = [
            item for item in issue_claims if item.claim_type in {"application", "calculation"}
        ]
        candidates = preferred or [
            item for item in issue_claims if item.claim_type == "normative_rule"
        ]
        if candidates:
            thesis_claims.append(max(candidates, key=lambda item: item.confidence))
    thesis_parts = list(
        dict.fromkeys(
            " ".join(item.result.split()).rstrip(". ") + "."
            for item in thesis_claims
            if item.result.strip()
        )
    )
    thesis = " ".join(thesis_parts[:6])
    if len(thesis) > 1_500:
        thesis = thesis[:1_497].rsplit(" ", 1)[0].rstrip(".,;: ") + "…"
    if not thesis:
        thesis = (
            "Źródła pierwotne zostały zweryfikowane, lecz synteza materialnych "
            "konkluzji nie została ukończona. Tego wyniku nie należy traktować "
            "jako zakończonej analizy podatkowej."
            if complete_primary_issues
            else "Nie uzyskano zweryfikowanych źródeł wystarczających do materialnej konkluzji."
        )

    def section_content(issue: Any) -> str:
        approved = [claim for claim in selected if claim.issue_id == issue.issue_id]
        if approved:
            return "\n".join(f"- {claim.text} {claim.result}" for claim in approved)
        requirements = coverage_requirements.get(issue.issue_id, [])
        citations = list(
            dict.fromkeys(
                str(requirement["citation"])
                for requirement in requirements
                if requirement.get("citation")
            )
        )
        if issue.issue_id in complete_primary_issues:
            suffix = f" Zweryfikowane przepisy: {', '.join(citations)}." if citations else ""
            return (
                "Nie ukończono syntezy materialnego wniosku mimo kompletnego "
                f"bundla prawa pierwotnego.{suffix}"
            )
        return "Brak kompletnego bundla prawa pierwotnego dla tej osi."

    sections = [
        WriterAnalysisSection(
            section_id=f"analysis_{issue.issue_id}",
            title=issue.label,
            content=section_content(issue),
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
            claim_ids = [
                claim.claim_id
                for claim in selected
                if provision.provision_id in claim.controlling_provision_ids
            ]
            # Do not emit unrelated entries from another issue bundle.  Apart
            # from being noisy, mixing them into the final source list can
            # make two editorial units of one technical record look like a
            # citation substitution to the integrity gate.
            if not claim_ids and not (
                not selected and provision.provision_id in fallback_source_ids
            ):
                continue
            sources.append(
                WriterSource(
                    source_id=provision.provision_id,
                    label=_writer_source_label(provision),
                    citation=provision.citation,
                    claim_ids=claim_ids,
                )
            )
        for authority in (
            *bundle.supporting_authorities,
            *bundle.contrary_authorities,
        ):
            claim_ids = [
                claim.claim_id
                for claim in selected
                if authority.document_id
                in (
                    *claim.supporting_authority_ids,
                    *claim.contrary_authority_ids,
                )
            ]
            if not claim_ids:
                continue
            sources.append(
                WriterSource(
                    source_id=authority.document_id,
                    label=authority.document_type,
                    citation=authority.signature or authority.document_id,
                    claim_ids=claim_ids,
                )
            )
    missing_questions = [item.question for item in plan.missing_facts]
    issue_labels = {issue.issue_id: issue.label for issue in plan.issues}
    source_gaps = [
        "Nie udało się potwierdzić kompletnego zestawu przepisów pierwotnych dla: "
        f"{issue_labels.get(bundle.issue_id, bundle.issue_id)}."
        for bundle in bundles
        if any(
            source == "primary_law" or source.startswith("required_primary:")
            for source in bundle.missing_sources
        )
    ]
    authority_gaps = [
        f"Nie znaleziono materialnej treści {source_label} dla: "
        f"{issue_labels.get(bundle.issue_id, bundle.issue_id)}."
        for bundle in bundles
        for missing_source, source_label in (
            ("authority_interpretation", "interpretacji podatkowej"),
            ("authority_judgment", "orzeczenia sądu"),
        )
        if missing_source in bundle.missing_sources
    ]
    remaining_coverage = _claim_coverage_requirements(plan, bundles, selected)
    reasoning_gaps = [
        "Nie ukończono wymaganych elementów rozumowania dla "
        f"{issue_labels.get(issue_id, issue_id)}: "
        + ", ".join(
            str(requirement["requirement_id"])
            for requirement in requirements
        )
        + "."
        for issue_id, requirements in remaining_coverage.items()
    ]
    claim_gaps = [
        "Nie ukończono syntezy materialnej konkluzji dla: "
        f"{issue.label}."
        for issue in plan.issues
        if not any(claim.issue_id == issue.issue_id for claim in selected)
        and issue.issue_id in complete_primary_issues
    ]
    conditional_gaps = [
        "Wniosek warunkowy wymaga uzupełnienia faktów dla: "
        f"{issue_labels.get(claim.issue_id, claim.issue_id)}."
        for claim in selected
        if claim.status == "conditional_missing_fact"
    ]
    synthesis_gap = (
        [
            "Kompletne źródła prawa pierwotnego są dostępne, ale synteza modelowa "
            "nie została ukończona; należy ponowić analizę po przywróceniu providera."
        ]
        if not selected and complete_primary_issues
        else []
    )
    return WriterOutput(
        thesis=thesis,
        analysis_sections=sections,
        sources=_dedupe_writer_sources(sources),
        risks_and_gaps=list(
            dict.fromkeys(
                [
                    *missing_questions,
                    *source_gaps,
                    *authority_gaps,
                    *reasoning_gaps,
                    *claim_gaps,
                    *conditional_gaps,
                    *synthesis_gap,
                ]
            )
        )
        or ["Nie znaleziono dodatkowych luk poza oznaczonymi statusami claimów."],
        claim_ids_used=[item.claim_id for item in selected],
    )


_RENDERED_PROVISION_RE = re.compile(
    r"\bart\.\s*\d+[a-z]*"
    r"(?:\s+ust\.\s*\d+[a-z]*(?:\s*[–-]\s*\d+[a-z]*)?)?"
    r"(?:\s+pkt\s*\d+[a-z]*(?:\s*[–-]\s*\d+[a-z]*)?)?"
    r"(?:\s+lit\.\s*[a-z])?",
    re.I,
)
_RANGED_SECTION_RE = re.compile(
    r"^(?P<prefix>art\.\s*\d+[a-z]*\s+ust\.\s*)"
    r"(?P<start>\d+[a-z]*)\s*[–-]\s*(?P<end>\d+[a-z]*)$",
    re.I,
)


def _citations_are_compatible(reference: str, citation: str) -> bool:
    """Require the bound source to be exact or finer than the written label.

    A verified paragraph may support the statement that its article applies,
    but a whole-article ID cannot support an invented paragraph or point.
    """

    normalized_reference = _normalize_citation(reference).rstrip(".,;:")
    normalized_citation = _normalize_citation(citation).rstrip(".,;:")
    return (
        normalized_reference == normalized_citation
        or normalized_citation.startswith(f"{normalized_reference} ")
    )


def _unbound_claim_provision_references(
    text: str,
    citations: Iterable[str],
) -> list[str]:
    """Return article labels not backed by this claim's exact source IDs."""

    allowed = [value for value in citations if str(value).strip()]
    result: list[str] = []
    seen: set[str] = set()
    for match in _RENDERED_PROVISION_RE.finditer(text):
        reference = _normalize_citation(match.group(0)).rstrip(".,;:")
        if reference in seen:
            continue
        seen.add(reference)
        range_match = _RANGED_SECTION_RE.fullmatch(reference)
        required_references = (
            [
                f"{range_match.group('prefix')}{range_match.group('start')}",
                f"{range_match.group('prefix')}{range_match.group('end')}",
            ]
            if range_match
            else [reference]
        )
        if not all(
            any(_citations_are_compatible(required, citation) for citation in allowed)
            for required in required_references
        ):
            result.append(reference)
    return result


def _expanded_rendered_provision_references(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for match in _RENDERED_PROVISION_RE.finditer(text):
        reference = _normalize_citation(match.group(0)).rstrip(".,;:")
        range_match = _RANGED_SECTION_RE.fullmatch(reference)
        values = (
            [
                f"{range_match.group('prefix')}{range_match.group('start')}",
                f"{range_match.group('prefix')}{range_match.group('end')}",
            ]
            if range_match
            else [reference]
        )
        for value in values:
            normalized = _normalize_citation(value).rstrip(".,;:")
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result


def _auto_bind_exact_claim_references(
    claim: LegalClaim,
    available: dict[str, ProvisionReference],
) -> tuple[LegalClaim, list[str]]:
    """Attach an exact retrieved unit when the model bound only its ancestor.

    The repair is deliberately conservative.  It never searches another issue
    or creates a citation: an exact normalized citation must occur uniquely in
    the current EvidenceBundle.  A single finer unit is accepted only when no
    exact unit exists.  Ambiguous article numbers remain invalid.
    """

    bound_ids = list(claim.controlling_provision_ids)
    bound = [available[item] for item in bound_ids if item in available]
    source_spans = list(claim.source_spans)
    repaired: list[str] = []
    for reference in _expanded_rendered_provision_references(
        f"{claim.text} {claim.result}"
    ):
        if any(_citations_are_compatible(reference, item.citation) for item in bound):
            continue
        compatible = [
            item
            for item in available.values()
            if _citations_are_compatible(reference, item.citation)
        ]
        exact = [
            item
            for item in compatible
            if _normalize_citation(item.citation).rstrip(".,;:") == reference
        ]
        candidates = exact if exact else compatible
        unique = {item.provision_id: item for item in candidates}
        if len(unique) != 1:
            continue
        provision = next(iter(unique.values()))
        if provision.provision_id not in bound_ids:
            bound_ids.append(provision.provision_id)
            bound.append(provision)
        if provision.source_span is not None and provision.source_span not in source_spans:
            source_spans.append(provision.source_span)
        repaired.append(reference)
    if not repaired:
        return claim, []
    return (
        claim.model_copy(
            update={
                "controlling_provision_ids": bound_ids,
                "source_spans": source_spans,
            }
        ),
        repaired,
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
    by_citation: dict[tuple[str, str], WriterSource] = {}
    for value in values:
        key = (value.label, _normalize_citation(value.citation))
        existing = by_citation.get(key)
        if existing is not None:
            existing.claim_ids = list(dict.fromkeys([*existing.claim_ids, *value.claim_ids]))
            continue
        by_citation[key] = value
        result.append(value)
    return result


def _writer_source_label(provision: ProvisionReference) -> str:
    document_id = provision.document_id.casefold()
    if "pl-upo" in document_id:
        return "UPO Polska–Niemcy" if "niemcy" in document_id else "UPO"
    if "fundacji-rodzinnej" in document_id:
        return "Ustawa o fundacji rodzinnej"
    if "podatku-od-towarow" in document_id:
        return "Ustawa o VAT"
    if "podatku-dochodowym-od-osob-prawnych" in document_id:
        return "Ustawa o CIT"
    if "podatku-dochodowym-od-osob-fizycznych" in document_id:
        return "Ustawa o PIT"
    return "Przepis"


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
