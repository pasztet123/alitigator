"""Data-driven concept matching. It has no behaviour keyed by concept id."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import ConceptDictionary, load_default_dictionary
from .normalizer import NormalizedText, normalize_text, phrase_present


@dataclass(frozen=True)
class ConceptMatch:
    concept_id: str
    concept_type: str
    matched_text: str
    match_type: str
    confidence: float
    locked: bool
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConceptMatchResult:
    dictionary_version: str
    original_question: str
    normalized_question: str
    matches: tuple[ConceptMatch, ...] = ()
    conflicts: tuple[dict[str, str], ...] = ()

    @property
    def locked_concepts(self) -> tuple[str, ...]:
        return tuple(item.concept_id for item in self.matches if item.locked)

    @property
    def soft_concepts(self) -> tuple[str, ...]:
        return tuple(item.concept_id for item in self.matches if not item.locked)


class ConceptMatcher:
    def __init__(self, dictionary: ConceptDictionary | None = None) -> None:
        self.dictionary = dictionary or load_default_dictionary()

    def match(self, text: str) -> ConceptMatchResult:
        normal = normalize_text(text)
        matches: list[ConceptMatch] = []
        for definition in self.dictionary.concepts:
            if definition.status in {"disabled", "draft"}:
                continue
            negative = any(phrase_present(value, normal) for value in definition.negative_context)
            required = all(phrase_present(value, normal) for value in definition.required_context)
            any_context = not definition.context_any or any(phrase_present(value, normal) for value in definition.context_any)
            evidence = tuple(value for value in (*definition.required_context, *definition.context_any) if phrase_present(value, normal))
            found: tuple[str, str, float] | None = None
            for kind, values, confidence in (
                ("exact_phrase", (definition.canonical_name, *definition.exact_aliases), .99),
                ("lemma_phrase", definition.lemma_aliases, .95),
                ("abbreviation", definition.abbreviations, .93),
                ("colloquial_alias", definition.colloquial_aliases, .84),
                ("factual_alias", definition.factual_aliases, .84),
            ):
                phrase = next((value for value in values if phrase_present(value, normal)), None)
                if phrase:
                    found = (kind, phrase, confidence)
                    break
            if found is None:
                pattern = next((value for value in definition.safe_regexes if re.search(value, normal.normalized, re.I)), None)
                if pattern:
                    found = ("safe_regex", pattern, .94)
            if found is None:
                continue
            kind, phrase, confidence = found
            locked = definition.status == "active" and not negative and required and any_context and (not definition.require_context_for_exact or bool(evidence))
            matches.append(ConceptMatch(definition.concept_id, definition.concept_type, phrase, kind, confidence, locked, evidence))
        by_id: dict[str, ConceptMatch] = {}
        for match in sorted(matches, key=lambda value: (-value.confidence, value.concept_id)):
            by_id.setdefault(match.concept_id, match)
        ordered = tuple(sorted(by_id.values(), key=lambda value: (-value.confidence, value.concept_id)))
        # Curated graph edges are a soft enrichment only.  They retain the
        # originating evidence, never create an institution lock and never
        # decide the underlying tax result.
        derived: list[ConceptMatch] = []
        for item in ordered:
            for related_id in self.dictionary.by_id[item.concept_id].related_concepts:
                related = self.dictionary.by_id.get(related_id)
                if related and related_id not in by_id:
                    derived.append(ConceptMatch(related_id, related.concept_type, item.matched_text, "related_concept", min(.75, item.confidence), False, (item.concept_id, item.matched_text)))
        enriched: dict[str, ConceptMatch] = {item.concept_id: item for item in ordered}
        for item in derived: enriched.setdefault(item.concept_id, item)
        ordered = tuple(sorted(enriched.values(), key=lambda value: (-value.confidence, value.concept_id)))
        present = {item.concept_id for item in ordered}
        conflicts = tuple(
            {"concept_id": item.concept_id, "incompatible_with": incompatible}
            for item in ordered for incompatible in self.dictionary.by_id[item.concept_id].incompatible_concepts
            if incompatible in present
        )
        return ConceptMatchResult(self.dictionary.version, text, normal.normalized, ordered, conflicts)
