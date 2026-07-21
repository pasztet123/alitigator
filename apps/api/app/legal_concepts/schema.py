"""Schema for the data-owned Polish tax-law concept taxonomy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ConceptStatus = Literal["active", "shadow", "draft", "disabled"]
ConceptType = Literal[
    "legal_institution", "legal_mechanism", "tax_domain", "act", "provision_family",
    "entity_role", "payment_type", "transaction_type", "contract_type",
    "product_or_service", "procedural_instrument", "form_or_report", "factual_concept",
]


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    concept_type: ConceptType
    canonical_name: str
    status: ConceptStatus = "shadow"
    priority: str = "normal"
    semantic_role: str = ""
    tax_domains: tuple[str, ...] = ()
    exact_aliases: tuple[str, ...] = ()
    lemma_aliases: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()
    colloquial_aliases: tuple[str, ...] = ()
    factual_aliases: tuple[str, ...] = ()
    safe_regexes: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()
    context_any: tuple[str, ...] = ()
    negative_context: tuple[str, ...] = ()
    require_context_for_exact: bool = False
    verified_provision_hints: tuple[str, ...] = ()
    statutory_terms: tuple[str, ...] = ()
    legal_terms: tuple[str, ...] = ()
    related_concepts: tuple[str, ...] = ()
    incompatible_concepts: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()

    @property
    def searchable_phrases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((
            self.canonical_name, *self.exact_aliases, *self.lemma_aliases,
            *self.abbreviations, *self.colloquial_aliases, *self.factual_aliases,
            *self.statutory_terms, *self.legal_terms,
        )))
