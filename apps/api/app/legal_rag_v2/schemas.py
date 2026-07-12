"""Strict, provider-neutral contracts used by the legal RAG v2 pipeline.

The models in this module deliberately describe evidence and provenance, not
provider payloads.  They can therefore be passed directly to a structured
output capable :class:`app.model_gateway.ModelGateway` and reused by offline
tests and deterministic validators.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyStr = Annotated[str, Field(min_length=1)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
JsonScalar = Union[str, int, float, bool, None]


class V2Schema(BaseModel):
    """Base class which keeps structured outputs closed and deterministic."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class SourceSpan(V2Schema):
    """Half-open character range in the source identified by ``source_id``."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: Optional[str] = None
    source_id: str = "user_question"

    @model_validator(mode="after")
    def validate_range(self) -> "SourceSpan":
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class DocumentSourceSpan(SourceSpan):
    """A span in a retrieved source, with enough identity to audit it later."""

    source_id: str = "document"
    document_id: NonEmptyStr
    chunk_id: Optional[str] = None


FactStatus = Literal["explicit", "inferred_from_language", "missing"]
FactMateriality = Literal["outcome_determinative", "retrieval_relevant", "minor"]
SourceType = Literal[
    "statute",
    "regulation",
    "tax_treaty",
    "interpretation",
    "general_interpretation",
    "guidance",
    "judgment",
    "resolution",
]


class ResearchIntent(V2Schema):
    mode: Literal["authority_research", "mixed_analysis", "rule_first"]
    needs_normative_answer: bool = True
    needs_interpretations: bool = True
    needs_case_law: bool = True
    needs_conflict_analysis: bool = True
    needs_calculations: bool = False


LegalResearchIntent = ResearchIntent


class Fact(V2Schema):
    fact_id: NonEmptyStr
    subject: NonEmptyStr
    role: NonEmptyStr
    predicate: NonEmptyStr
    value: NonEmptyStr
    status: FactStatus
    source_span: Optional[SourceSpan] = None

    @model_validator(mode="after")
    def sourced_when_observed(self) -> "Fact":
        if self.status != "missing" and self.source_span is None:
            raise ValueError("an observed or inferred fact requires a source_span")
        return self


# Explicit alias used in documentation and downstream type annotations.
FactSourceSpan = SourceSpan


class MissingFact(V2Schema):
    fact_id: NonEmptyStr
    question: NonEmptyStr
    materiality: FactMateriality


QueryFamilyName = Literal[
    "natural_language",
    "legal_concept",
    "user_terminology",
    "explicit_provision_reference",
    "known_provision_synonym",
    "citation_backreference",
    "issue_signature",
    "factual_contrast",
    "quoted_statutory_language",
    "cited_judgment_signature",
    # Names used by the public retrieval contract.  Older aliases above are
    # retained so persisted plans from the first v2 iteration stay readable.
    "statutory_concept",
    "fact_contrast",
    "explicit_provision",
    "authority_backreference",
    "quoted_holding_language",
]


class QueryFamily(V2Schema):
    family: QueryFamilyName
    query: NonEmptyStr
    lane: Literal["primary_law", "authority", "both"] = "both"
    origin: Literal["model", "user", "fallback"] = "model"


class LegalIssue(V2Schema):
    issue_id: NonEmptyStr
    label: NonEmptyStr
    tax_domains: list[NonEmptyStr] = Field(default_factory=list)
    legal_mechanism: NonEmptyStr
    taxpayer_roles: list[str] = Field(default_factory=list)
    transactions: list[str] = Field(default_factory=list)
    payments: list[str] = Field(default_factory=list)
    jurisdictions: list[str] = Field(default_factory=list)
    relevant_dates: list[str] = Field(default_factory=list)
    possible_provision_concepts: list[str] = Field(default_factory=list)
    positive_fact_constraints: list[str] = Field(default_factory=list)
    negative_fact_constraints: list[str] = Field(default_factory=list)
    requested_source_types: list[SourceType] = Field(
        default_factory=lambda: ["statute", "interpretation", "judgment"]
    )
    query_families: list[QueryFamily] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"


# The shorter name is convenient in retrieval modules and remains explicit.
Issue = LegalIssue


class Clarification(V2Schema):
    should_ask: bool = False
    questions: list[NonEmptyStr] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_gate(self) -> "Clarification":
        if self.should_ask != bool(self.questions):
            raise ValueError("should_ask must match the presence of questions")
        return self


class LegalResearchPlan(V2Schema):
    """Planner output.  It intentionally has no legal answer or conclusion."""

    user_query: str = ""
    intent: ResearchIntent
    target_date: Optional[str] = None
    facts: list[Fact] = Field(default_factory=list)
    missing_facts: list[MissingFact] = Field(default_factory=list)
    issues: list[LegalIssue] = Field(min_length=1)
    clarification: Clarification = Field(default_factory=Clarification)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_internal_references(self) -> "LegalResearchPlan":
        fact_ids = [item.fact_id for item in self.facts]
        missing_ids = [item.fact_id for item in self.missing_facts]
        issue_ids = [item.issue_id for item in self.issues]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique")
        if len(missing_ids) != len(set(missing_ids)):
            raise ValueError("missing fact_id values must be unique")
        if set(fact_ids) & set(missing_ids):
            raise ValueError("facts and missing_facts must not reuse fact_id")
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("issue_id values must be unique")
        if self.clarification.should_ask and not any(
            item.materiality in {"outcome_determinative", "retrieval_relevant"}
            for item in self.missing_facts
        ):
            raise ValueError("clarification requires a material missing fact")
        return self


FallbackReason = Literal[
    "planner_timeout",
    "provider_unavailable",
    "invalid_schema",
    "low_confidence",
    "insufficient_recall",
    "forced",
]


class FallbackCandidate(V2Schema):
    """A query/provision hint from legacy rules, never a legal conclusion."""

    candidate_type: Literal["query_hint", "provision_hint", "issue_hint"]
    value: NonEmptyStr
    issue_id: Optional[str] = None
    fallback_added: Literal[True] = True


class FallbackTrace(V2Schema):
    fallback_used: bool = False
    fallback_reason: Optional[FallbackReason] = None
    fallback_rules: list[str] = Field(default_factory=list)
    fallback_candidates_added: list[FallbackCandidate] = Field(default_factory=list)
    primary_planner_error: Optional[str] = None

    @model_validator(mode="after")
    def validate_reason(self) -> "FallbackTrace":
        if self.fallback_used and self.fallback_reason is None:
            raise ValueError("fallback_reason is required when fallback is used")
        if not self.fallback_used and (
            self.fallback_reason is not None
            or self.fallback_rules
            or self.fallback_candidates_added
        ):
            raise ValueError("unused fallback cannot contain fallback decisions")
        return self


ProvisionStatus = Literal["active", "repealed", "historical", "unknown"]


class ProvisionReference(V2Schema):
    provision_id: NonEmptyStr
    document_id: NonEmptyStr
    version_id: Optional[str] = None
    citation: NonEmptyStr
    article: Optional[str] = None
    paragraph: Optional[str] = None
    point: Optional[str] = None
    letter: Optional[str] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    status: ProvisionStatus = "unknown"
    text: Optional[str] = None
    source_span: Optional[DocumentSourceSpan] = None

    @model_validator(mode="after")
    def validate_effective_period(self) -> "ProvisionReference":
        if self.effective_from and self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        return self


ProvisionRef = ProvisionReference


ProvisionRelationship = Literal[
    "references",
    "defines",
    "exception_to",
    "special_rule_for",
    "overrides",
    "temporal_successor",
    "neighbor",
    "transitional_rule_for",
]


class ProvisionGraphEdge(V2Schema):
    source_provision_id: NonEmptyStr
    target_provision_id: NonEmptyStr
    relationship: ProvisionRelationship
    source_span: Optional[DocumentSourceSpan] = None
    verified: bool = False


class ProvisionGraph(V2Schema):
    provisions: list[ProvisionReference] = Field(default_factory=list)
    edges: list[ProvisionGraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edges(self) -> "ProvisionGraph":
        ids = [item.provision_id for item in self.provisions]
        if len(ids) != len(set(ids)):
            raise ValueError("provision graph node IDs must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source_provision_id not in known or edge.target_provision_id not in known:
                raise ValueError("provision graph edge references an unknown node")
        return self


class AuthoritySourceSpans(V2Schema):
    signature: list[DocumentSourceSpan] = Field(default_factory=list)
    authority: list[DocumentSourceSpan] = Field(default_factory=list)
    court: list[DocumentSourceSpan] = Field(default_factory=list)
    date: list[DocumentSourceSpan] = Field(default_factory=list)
    facts: list[DocumentSourceSpan] = Field(default_factory=list)
    issues: list[DocumentSourceSpan] = Field(default_factory=list)
    cited_provisions: list[DocumentSourceSpan] = Field(default_factory=list)
    taxpayer_position: list[DocumentSourceSpan] = Field(default_factory=list)
    authority_holding: list[DocumentSourceSpan] = Field(default_factory=list)
    court_holding: list[DocumentSourceSpan] = Field(default_factory=list)
    outcome: list[DocumentSourceSpan] = Field(default_factory=list)
    result_for_taxpayer: list[DocumentSourceSpan] = Field(default_factory=list)
    reasoning: list[DocumentSourceSpan] = Field(default_factory=list)
    distinguishing_facts: list[DocumentSourceSpan] = Field(default_factory=list)


AuthorityCardSourceSpans = AuthoritySourceSpans


class AuthorityCard(V2Schema):
    document_id: NonEmptyStr
    signature: str = ""
    document_type: NonEmptyStr
    authority: str = ""
    court: str = ""
    date: str = ""
    legal_state_date: Optional[str] = None
    tax_domains: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    cited_provisions: list[str] = Field(default_factory=list)
    taxpayer_position: Optional[str] = None
    authority_holding: Optional[str] = None
    court_holding: Optional[str] = None
    outcome: Optional[str] = None
    result_for_taxpayer: Optional[str] = None
    reasoning: Optional[str] = None
    distinguishing_facts: list[str] = Field(default_factory=list)
    source_spans: AuthoritySourceSpans
    extraction_confidence: Confidence

    @model_validator(mode="after")
    def require_material_source_spans(self) -> "AuthorityCard":
        required = (
            (self.facts, self.source_spans.facts, "facts"),
            (self.issues, self.source_spans.issues, "issues"),
            (self.cited_provisions, self.source_spans.cited_provisions, "cited_provisions"),
            (self.taxpayer_position, self.source_spans.taxpayer_position, "taxpayer_position"),
            (self.authority_holding, self.source_spans.authority_holding, "authority_holding"),
            (self.court_holding, self.source_spans.court_holding, "court_holding"),
            (self.outcome, self.source_spans.outcome, "outcome"),
            (self.result_for_taxpayer, self.source_spans.result_for_taxpayer, "result_for_taxpayer"),
            (self.reasoning, self.source_spans.reasoning, "reasoning"),
            (self.distinguishing_facts, self.source_spans.distinguishing_facts, "distinguishing_facts"),
        )
        missing = [name for value, spans, name in required if value and not spans]
        if missing:
            raise ValueError(f"material authority fields lack source spans: {', '.join(missing)}")
        return self


class RerankScore(V2Schema):
    final_score: float
    component_scores: dict[str, float] = Field(default_factory=dict)
    positive_reasons: list[str] = Field(default_factory=list)
    negative_reasons: list[str] = Field(default_factory=list)


class RerankedAuthority(V2Schema):
    issue_id: NonEmptyStr
    document_id: NonEmptyStr
    candidate_rank: int = Field(ge=1)
    final_rank: int = Field(ge=1)
    candidate_recall_measured: bool = False
    score: RerankScore


RerankResult = RerankScore


class EvidenceBundle(V2Schema):
    issue_id: NonEmptyStr
    controlling_provisions: list[ProvisionReference] = Field(default_factory=list)
    dependency_provisions: list[ProvisionReference] = Field(default_factory=list)
    exception_provisions: list[ProvisionReference] = Field(default_factory=list)
    supporting_authorities: list[AuthorityCard] = Field(default_factory=list)
    contrary_authorities: list[AuthorityCard] = Field(default_factory=list)
    historical_authorities: list[AuthorityCard] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    retrieval_confidence: Confidence = 0.0
    coverage_status: Literal["complete", "partial", "missing"] = "missing"
    controlling_provision_present: bool = False
    dependency_coverage: Confidence = 0.0
    exception_coverage: Confidence = 0.0
    temporal_validation_passed: bool = False
    authority_candidates_present: bool = False
    supporting_authorities_present: bool = False
    contrary_authorities_present: bool = False


ClaimStatus = Literal[
    "approved",
    "conditional_missing_fact",
    "blocked_missing_primary_law",
    "blocked_incomplete_dependency_bundle",
    "blocked_insufficient_authority",
    "blocked_conflicting_evidence",
    "blocked_invalid_provision",
    "blocked_out_of_scope",
]


class LegalClaim(V2Schema):
    claim_id: NonEmptyStr
    issue_id: NonEmptyStr
    claim_type: Literal["normative_rule", "application", "authority_pattern", "calculation", "risk"]
    text: NonEmptyStr
    status: ClaimStatus
    result: NonEmptyStr
    controlling_provision_ids: list[str] = Field(default_factory=list)
    supporting_authority_ids: list[str] = Field(default_factory=list)
    contrary_authority_ids: list[str] = Field(default_factory=list)
    fact_dependencies: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    source_spans: list[DocumentSourceSpan] = Field(default_factory=list)
    confidence: Confidence
    material: bool = True

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> "LegalClaim":
        if self.material and self.status in {"approved", "conditional_missing_fact"}:
            if not self.controlling_provision_ids:
                raise ValueError("a material unblocked claim requires controlling primary law")
            if not self.source_spans:
                raise ValueError("a material unblocked claim requires source spans")
        if self.claim_type == "authority_pattern" and self.status == "approved" and not self.supporting_authority_ids:
            raise ValueError("an approved authority-pattern claim requires authority document IDs")
        if self.claim_type == "calculation" and self.status == "approved" and not self.calculation_ids:
            raise ValueError("an approved calculation claim requires calculation IDs")
        return self


class CalculationRecord(V2Schema):
    calculation_id: NonEmptyStr
    inputs: dict[str, JsonScalar] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    operation: NonEmptyStr
    formula: NonEmptyStr
    result: JsonScalar
    rounding: NonEmptyStr
    legal_basis: list[ProvisionReference] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class AnswerSection(V2Schema):
    section_id: NonEmptyStr
    title: NonEmptyStr
    purpose: NonEmptyStr
    required_claim_ids: list[str] = Field(default_factory=list)


class AnswerPlan(V2Schema):
    thesis_claim_ids: list[str] = Field(default_factory=list)
    sections: list[AnswerSection] = Field(default_factory=list)
    allowed_claim_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    source_order: list[Literal["primary_law", "authority", "risk"]] = Field(
        default_factory=lambda: ["primary_law", "authority", "risk"]
    )


class WriterAnalysisSection(V2Schema):
    section_id: NonEmptyStr
    title: NonEmptyStr
    content: NonEmptyStr
    claim_ids_used: list[str] = Field(default_factory=list)


class WriterSource(V2Schema):
    source_id: NonEmptyStr
    label: NonEmptyStr
    citation: NonEmptyStr
    claim_ids: list[str] = Field(default_factory=list)


class WriterOutput(V2Schema):
    thesis: NonEmptyStr
    analysis_sections: list[WriterAnalysisSection] = Field(default_factory=list)
    sources: list[WriterSource] = Field(default_factory=list)
    risks_and_gaps: list[str] = Field(default_factory=list)
    claim_ids_used: list[str] = Field(default_factory=list)


class ValidationRecord(V2Schema):
    stage: NonEmptyStr
    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PipelineResult(V2Schema):
    request_id: NonEmptyStr
    run_id: NonEmptyStr
    mode: Literal["legal_rag_v2", "shadow"] = "legal_rag_v2"
    legal_research_plan: LegalResearchPlan
    fallback_trace: FallbackTrace = Field(default_factory=FallbackTrace)
    provision_graph: ProvisionGraph = Field(default_factory=ProvisionGraph)
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    claims: list[LegalClaim] = Field(default_factory=list)
    calculations: list[CalculationRecord] = Field(default_factory=list)
    answer_plan: Optional[AnswerPlan] = None
    writer_output: Optional[WriterOutput] = None
    final_answer: Optional[str] = None
    validation: list[ValidationRecord] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    costs: dict[str, float] = Field(default_factory=dict)
