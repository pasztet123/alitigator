"""Typed dictionary schema kept separate from the retrieval implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


InstitutionStatus = Literal["active", "shadow", "draft", "disabled"]
MatchType = Literal[
    "exact_phrase",
    "lemma_phrase",
    "safe_regex",
    "abbreviation",
    "colloquial_alias",
    "contextual_inference",
]


@dataclass(frozen=True)
class InstitutionDefinition:
    institution_id: str
    canonical_name: str
    status: InstitutionStatus
    rollout_stage: str
    tax_domains: tuple[str, ...] = ()
    exact_aliases: tuple[str, ...] = ()
    lemma_aliases: tuple[str, ...] = ()
    safe_regexes: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()
    colloquial_aliases: tuple[str, ...] = ()
    provision_hints: tuple[str, ...] = ()
    statutory_phrases: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    contextual_signals: tuple[str, ...] = ()
    context_any_signals: tuple[str, ...] = ()
    require_context_for_exact: bool = False
    negative_context: tuple[str, ...] = ()
    source_preferences: tuple[str, ...] = ()
    query_templates: tuple[str, ...] = ()
    legal_mechanisms: tuple[str, ...] = ()

    @property
    def searchable_phrases(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.canonical_name,
                    *self.exact_aliases,
                    *self.lemma_aliases,
                    *self.colloquial_aliases,
                    *self.statutory_phrases,
                    *self.material_concepts,
                )
            )
        )


@dataclass(frozen=True)
class InstitutionMatchRecord:
    institution_id: str
    canonical_name: str
    confidence: float
    match_type: MatchType
    matched_text: str
    tokens: tuple[str, ...]
    tax_domains: tuple[str, ...]
    locked: bool
    status: InstitutionStatus
    provision_hints: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    source_preferences: tuple[str, ...] = ()
    rollout_stage: str = "C"
    context_satisfied: bool = True
    negative_context_hit: bool = False


@dataclass(frozen=True)
class InstitutionMatchResultRecord:
    dictionary_version: str
    original_question: str
    normalized_question: str
    tokens: tuple[str, ...]
    matches: tuple[InstitutionMatchRecord, ...] = field(default_factory=tuple)
