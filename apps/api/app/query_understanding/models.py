"""Provider-neutral cards and plans for the generic query-understanding flow."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProvisionHint:
    citation: str
    source: str
    verified: bool


@dataclass
class QuestionCard:
    question_id: str = ""
    original_question: str = ""
    normalized_question: str = ""
    tax_domains: list[str] = field(default_factory=list)
    locked_institutions: list[str] = field(default_factory=list)
    detected_concepts: list[str] = field(default_factory=list)
    taxpayer_roles: list[str] = field(default_factory=list)
    counterparty_roles: list[str] = field(default_factory=list)
    payment_direction: str | None = None
    payment_types: list[str] = field(default_factory=list)
    transaction_types: list[str] = field(default_factory=list)
    contract_types: list[str] = field(default_factory=list)
    products_or_services: list[str] = field(default_factory=list)
    explicit_provisions: list[str] = field(default_factory=list)
    verified_provision_hints: list[ProvisionHint] = field(default_factory=list)
    material_facts: list[str] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass
class ModelQueryExpansion:
    primary_issue: str = ""
    secondary_issues: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    statutory_language: list[str] = field(default_factory=list)
    factual_synonyms: list[str] = field(default_factory=list)
    likely_document_wording: list[str] = field(default_factory=list)
    material_distinctions: list[str] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    unverified_provision_suggestions: list[ProvisionHint] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)


@dataclass
class QueryFamilySpec:
    id: str
    type: str
    query: str
    must_any: list[str] = field(default_factory=list)
    context_any: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    tax_domains: list[str] = field(default_factory=list)
    limit: int = 40
    weight: float = 1.0
    hardness: str = "soft"

    @property
    def required_any(self) -> list[str]: return self.must_any
    @property
    def required_context_any(self) -> list[str]: return self.context_any
    @property
    def negative_terms(self) -> list[str]: return self.must_not
    @property
    def hard_requirements(self) -> list[str]:
        return [*self.must_any, *self.context_any] if self.hardness == "hard" else []


@dataclass
class QueryPlan:
    query_plan_version: str = "v2"
    question_card: QuestionCard = field(default_factory=QuestionCard)
    primary_issue: str = ""
    secondary_issues: list[str] = field(default_factory=list)
    locked_institutions: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    statutory_language: list[str] = field(default_factory=list)
    factual_synonyms: list[str] = field(default_factory=list)
    likely_document_wording: list[str] = field(default_factory=list)
    verified_provision_hints: list[ProvisionHint] = field(default_factory=list)
    soft_provision_hints: list[ProvisionHint] = field(default_factory=list)
    material_distinctions: list[str] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    query_families: list[QueryFamilySpec] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]: return asdict(self)
    def has_locked_institution(self, value: str) -> bool: return value in self.locked_institutions
    def has_tax_domain(self, value: str) -> bool: return value in self.question_card.tax_domains
    def has_payment_type(self, value: str) -> bool: return value in self.question_card.payment_types
    def has_contract_type(self, value: str) -> bool: return value in self.question_card.contract_types
    def has_any_legal_concept(self, values: list[str]) -> bool: return bool(set(values) & set(self.legal_concepts))
    def has_any_provision(self, values: list[str]) -> bool:
        return any(any(value.casefold() in hint.citation.casefold() for hint in self.verified_provision_hints) for value in values)
