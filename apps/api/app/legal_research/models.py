from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LegalResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceSpan(LegalResearchModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "SourceSpan":
        if self.end <= self.start:
            raise ValueError("source span must be a non-empty half-open range")
        return self


class ResearchFact(LegalResearchModel):
    fact_id: str
    subject: str
    role: str
    predicate: str
    value: str
    status: Literal["explicit", "missing"]
    source_span: Optional[SourceSpan] = None

    @model_validator(mode="after")
    def explicit_has_span(self) -> "ResearchFact":
        if self.status == "explicit" and self.source_span is None:
            raise ValueError("every explicit fact requires a source_span")
        return self


class MissingFact(LegalResearchModel):
    fact_id: str
    question: str
    materiality: Literal["outcome_determinative", "retrieval_relevant", "minor"]


SourceType = Literal[
    "statute", "regulation", "treaty", "interpretation",
    "general_interpretation", "tax_guidance", "judgment",
]


class ResearchIssue(LegalResearchModel):
    issue_id: str
    label: str
    tax_domains: list[str] = Field(default_factory=list)
    legal_mechanism: str
    material_fact_ids: list[str] = Field(default_factory=list)
    taxpayer_roles: list[str] = Field(default_factory=list)
    transactions: list[str] = Field(default_factory=list)
    payments: list[str] = Field(default_factory=list)
    jurisdictions: list[str] = Field(default_factory=list)
    relevant_dates: list[str] = Field(default_factory=list)
    possible_legal_concepts: list[str] = Field(default_factory=list)
    possible_provision_hints: list[str] = Field(default_factory=list)
    positive_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    requested_source_types: list[SourceType] = Field(
        default_factory=lambda: ["statute", "interpretation", "judgment"]
    )
    priority: Literal["high", "medium", "low"] = "medium"


class ResearchIntent(LegalResearchModel):
    mode: Literal["authority_research", "mixed_analysis", "rule_first"]
    needs_primary_law: bool = True
    needs_interpretations: bool = True
    needs_case_law: bool = True
    needs_conflict_analysis: bool = True
    needs_calculations: bool = False


class LegalResearchPlan(LegalResearchModel):
    intent: ResearchIntent
    target_date: Optional[str] = None
    facts: list[ResearchFact] = Field(default_factory=list)
    missing_facts: list[MissingFact] = Field(default_factory=list)
    issues: list[ResearchIssue] = Field(default_factory=list)
    should_ask_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1)


class RetrievalQuery(LegalResearchModel):
    query_id: str
    issue_id: str
    lane: Literal["primary", "authority"]
    family: Literal[
        "natural_language", "legal_concept", "fact_signature", "fact_contrast",
        "explicit_provision", "quoted_statutory_language",
        "authority_backreference", "judgment_signature",
    ]
    query: str
    positive_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    generated_by: Literal[
        "planner", "primary_backreference", "authority_backreference", "legacy_fallback"
    ] = "planner"


class ProvisionCandidate(LegalResearchModel):
    provision_id: str
    document_id: str
    act_id: str
    article: str
    paragraph: Optional[str] = None
    point: Optional[str] = None
    letter: Optional[str] = None
    display_reference: str
    effective_from: str
    effective_to: Optional[str] = None
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_score: float = 0
    query_ids: list[str] = Field(default_factory=list)


class ProvisionRelation(LegalResearchModel):
    source_provision_id: str
    target_provision_id: str
    relation_type: Literal[
        "references", "defines", "exception_to", "special_rule_for", "overrides",
        "neighbor", "temporal_successor", "transitional_rule_for",
    ]
    evidence_span: Optional[str] = None


class DocumentSpan(LegalResearchModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class LegalRuleEvidence(LegalResearchModel):
    rule_id: str
    provision_id: str
    issue_id: str
    rule_type: Literal[
        "general_rule", "condition", "exception", "special_rule", "definition",
        "deadline", "formula", "rate", "transitional_rule",
    ]
    rule_text: str
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    referenced_provision_ids: list[str] = Field(default_factory=list)
    source_span_start: int = Field(ge=0)
    source_span_end: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)


class AuthorityCard(LegalResearchModel):
    document_id: str
    signature: str
    document_type: str
    authority: str
    court: Optional[str] = None
    date: str
    legal_state_date: Optional[str] = None
    tax_domains: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    transactions: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    taxpayer_roles: list[str] = Field(default_factory=list)
    cited_provisions: list[str] = Field(default_factory=list)
    taxpayer_position: Optional[str] = None
    authority_holding: Optional[str] = None
    court_holding: Optional[str] = None
    outcome: Optional[str] = None
    result_for_taxpayer: Optional[str] = None
    reasoning_summary: Optional[str] = None
    distinguishing_facts: list[str] = Field(default_factory=list)
    wrong_neighbor_reasons: list[str] = Field(default_factory=list)
    source_spans: dict[str, list[int]] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)


class LegalRerankResult(LegalResearchModel):
    document_id: str
    issue_id: str
    final_score: float
    issue_match: float = 0
    material_fact_match: float = 0
    provision_match: float = 0
    role_match: float = 0
    transaction_match: float = 0
    temporal_match: float = 0
    holding_relevance: float = 0
    wrong_neighbor_penalty: float = 0
    positive_reasons: list[str] = Field(default_factory=list)
    negative_reasons: list[str] = Field(default_factory=list)


class EvidenceBinding(LegalResearchModel):
    source_id: str
    target_id: str
    target_type: Literal["issue", "claim"]
    relation: Literal["supports", "contradicts", "context_only", "historical_only", "not_relevant"]
    score: float = Field(ge=0, le=1)
    reason: str
    supporting_span_ids: list[str] = Field(default_factory=list)


class EvidenceBundle(LegalResearchModel):
    issue_id: str
    controlling_provision_ids: list[str] = Field(default_factory=list)
    dependency_provision_ids: list[str] = Field(default_factory=list)
    exception_provision_ids: list[str] = Field(default_factory=list)
    special_rule_provision_ids: list[str] = Field(default_factory=list)
    supporting_authority_ids: list[str] = Field(default_factory=list)
    contrary_authority_ids: list[str] = Field(default_factory=list)
    historical_authority_ids: list[str] = Field(default_factory=list)
    missing_primary_sources: list[str] = Field(default_factory=list)
    missing_authority_types: list[str] = Field(default_factory=list)
    missing_fact_ids: list[str] = Field(default_factory=list)
    primary_coverage: Literal["complete", "partial", "missing"]
    authority_coverage: Literal["complete", "partial", "missing"]
    retrieval_confidence: float = Field(ge=0, le=1)


class MissingEvidenceRequest(LegalResearchModel):
    issue_id: str
    reason: str
    missing_evidence_type: str
    proposed_queries: list[RetrievalQuery] = Field(default_factory=list)


class LegalClaim(LegalResearchModel):
    claim_id: str
    issue_id: str
    claim_type: Literal[
        "legal_rule", "application", "calculation", "deadline",
        "authority_summary", "practical_conclusion",
    ]
    text: str
    status: Literal[
        "approved", "conditional_missing_fact", "blocked_missing_primary_law",
        "blocked_incomplete_dependency_bundle", "blocked_conflicting_evidence",
        "blocked_invalid_provision", "blocked_out_of_scope",
    ]
    controlling_provision_ids: list[str] = Field(default_factory=list)
    supporting_authority_ids: list[str] = Field(default_factory=list)
    contrary_authority_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    missing_fact_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class CalculationRecord(LegalResearchModel):
    calculation_id: str
    calculation_type: str
    input_values: dict[str, Union[str, int, float]]
    formula: str
    result: Union[str, int, float]
    units: Optional[str] = None
    rounding_rule: Optional[str] = None
    provision_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    validation_status: Literal["valid", "invalid"]


class AnswerSection(LegalResearchModel):
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class AnswerDraft(LegalResearchModel):
    thesis: list[str] = Field(default_factory=list)
    analysis_sections: list[AnswerSection] = Field(default_factory=list)
    risks_and_gaps: list[str] = Field(default_factory=list)
    claim_ids_used: list[str] = Field(default_factory=list)
    provision_ids_used: list[str] = Field(default_factory=list)
    authority_ids_used: list[str] = Field(default_factory=list)
    calculation_ids_used: list[str] = Field(default_factory=list)
