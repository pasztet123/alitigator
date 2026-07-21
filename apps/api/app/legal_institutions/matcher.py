"""Deterministic matcher and document-marker gate for named institutions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .dictionary import InstitutionDictionary, load_default_dictionary
from .normalizer import NormalizedText, normalize_polish, phrase_present
from .schema import InstitutionDefinition, InstitutionMatchRecord, InstitutionMatchResultRecord


_HARD_CONFIDENCE = {
    "exact_phrase": 0.99,
    "lemma_phrase": 0.95,
    "safe_regex": 0.94,
    "abbreviation": 0.93,
    "colloquial_alias": 0.84,
    "contextual_inference": 0.61,
}
_ARTICLE_RE = re.compile(r"\bart\.\s*(\d+[a-z]*)", re.IGNORECASE)


def _matched_source_text(question: str, phrase: str) -> str:
    normal = normalize_polish(phrase).normalized
    if not normal:
        return phrase
    match = re.search(re.escape(normal).replace(r"\ ", r"\s+"), question, re.IGNORECASE)
    return match.group(0) if match else phrase


def _context_satisfied(definition: InstitutionDefinition, question: NormalizedText) -> bool:
    required = all(phrase_present(signal, question) for signal in definition.contextual_signals)
    optional_group = (
        not definition.context_any_signals
        or any(phrase_present(signal, question) for signal in definition.context_any_signals)
    )
    return required and optional_group


def _negative_context_hit(definition: InstitutionDefinition, question: NormalizedText) -> bool:
    return any(phrase_present(signal, question) for signal in definition.negative_context)


def _record(
    definition: InstitutionDefinition,
    *,
    match_type: str,
    matched_text: str,
    question: NormalizedText,
    context_satisfied: bool,
    negative_context_hit: bool,
) -> InstitutionMatchRecord:
    locked = (
        definition.status == "active"
        and match_type != "contextual_inference"
        and context_satisfied
        and not negative_context_hit
    )
    return InstitutionMatchRecord(
        institution_id=definition.institution_id,
        canonical_name=definition.canonical_name,
        confidence=_HARD_CONFIDENCE[match_type],  # type: ignore[index]
        match_type=match_type,  # type: ignore[arg-type]
        matched_text=matched_text,
        tokens=question.tokens,
        tax_domains=definition.tax_domains,
        locked=locked,
        status=definition.status,
        provision_hints=definition.provision_hints,
        material_concepts=definition.material_concepts,
        source_preferences=definition.source_preferences,
        rollout_stage=definition.rollout_stage,
        context_satisfied=context_satisfied,
        negative_context_hit=negative_context_hit,
    )


class InstitutionMatcher:
    """Recognise only explicit or bounded named-institution signals.

    It intentionally does not perform semantic expansion, broad prefix
    matching, or a guessed legal conclusion.  Shadow matches are traced but
    never lock a planner or a retrieval result.
    """

    def __init__(self, dictionary: InstitutionDictionary | None = None) -> None:
        self.dictionary = dictionary or load_default_dictionary()

    def match(self, question: str) -> InstitutionMatchResultRecord:
        normalized = normalize_polish(question)
        matches: list[InstitutionMatchRecord] = []
        for definition in self.dictionary.institutions:
            if definition.status in {"disabled", "draft"}:
                continue
            matched = self._match_definition(definition, normalized)
            if matched is not None:
                matches.append(matched)
        # One clear record per canonical institution.  Exact evidence wins if
        # an alias and a statutory phrase happen to occur in the same question.
        precedence = {
            "exact_phrase": 0,
            "lemma_phrase": 1,
            "safe_regex": 2,
            "abbreviation": 3,
            "colloquial_alias": 4,
            "contextual_inference": 5,
        }
        by_id: dict[str, InstitutionMatchRecord] = {}
        for item in sorted(matches, key=lambda value: (precedence[value.match_type], -value.confidence)):
            by_id.setdefault(item.institution_id, item)
        ordered = tuple(
            sorted(
                by_id.values(),
                key=lambda value: (-value.confidence, value.institution_id),
            )
        )
        return InstitutionMatchResultRecord(
            dictionary_version=self.dictionary.version,
            original_question=question,
            normalized_question=normalized.normalized,
            tokens=normalized.tokens,
            matches=ordered,
        )

    def _match_definition(
        self,
        definition: InstitutionDefinition,
        question: NormalizedText,
    ) -> InstitutionMatchRecord | None:
        context_satisfied = _context_satisfied(definition, question)
        negative_context_hit = _negative_context_hit(definition, question)
        # The canonical name itself is an exact phrase in the vocabulary.
        for phrase in (definition.canonical_name, *definition.exact_aliases):
            if phrase_present(phrase, question):
                return _record(
                    definition,
                    match_type="exact_phrase",
                    matched_text=_matched_source_text(question.original, phrase),
                    question=question,
                    context_satisfied=(
                        context_satisfied if definition.require_context_for_exact else True
                    ),
                    negative_context_hit=negative_context_hit,
                )
        # Lemma aliases are explicitly curated full forms.  This is safer than
        # stripping Polish endings or applying an unverified general stemmer.
        for phrase in definition.lemma_aliases:
            if phrase_present(phrase, question):
                return _record(
                    definition,
                    match_type="lemma_phrase",
                    matched_text=_matched_source_text(question.original, phrase),
                    question=question,
                    context_satisfied=context_satisfied,
                    negative_context_hit=negative_context_hit,
                )
        for pattern in definition.safe_regexes:
            if re.search(pattern, question.normalized, flags=re.IGNORECASE):
                return _record(
                    definition,
                    match_type="safe_regex",
                    matched_text=pattern,
                    question=question,
                    context_satisfied=context_satisfied,
                    negative_context_hit=negative_context_hit,
                )
        for abbreviation in definition.abbreviations:
            if phrase_present(abbreviation, question):
                return _record(
                    definition,
                    match_type="abbreviation",
                    matched_text=_matched_source_text(question.original, abbreviation),
                    question=question,
                    context_satisfied=context_satisfied,
                    negative_context_hit=negative_context_hit,
                )
        for phrase in definition.colloquial_aliases:
            if phrase_present(phrase, question):
                return _record(
                    definition,
                    match_type="colloquial_alias",
                    matched_text=_matched_source_text(question.original, phrase),
                    question=question,
                    context_satisfied=context_satisfied,
                    negative_context_hit=negative_context_hit,
                )
        # Contextual inference is deliberately not a lock.  It is trace data
        # for future dictionary curation and can never override the planner.
        if (
            len(definition.material_concepts) >= 2
            and sum(phrase_present(value, question) for value in definition.material_concepts) >= 2
            and context_satisfied
            and not negative_context_hit
        ):
            return _record(
                definition,
                match_type="contextual_inference",
                matched_text="contextual_signals",
                question=question,
                context_satisfied=True,
                negative_context_hit=False,
            )
        return None

    def definitions_for(self, institution_ids: Iterable[str]) -> tuple[InstitutionDefinition, ...]:
        return tuple(
            self.dictionary.by_id[item]
            for item in institution_ids
            if item in self.dictionary.by_id
        )

    def document_markers(
        self,
        definition: InstitutionDefinition,
        *,
        text: str,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[str, ...]:
        metadata = metadata or {}
        raw_provisions = metadata.get("legal_provisions") or metadata.get("provisions") or []
        if isinstance(raw_provisions, str):
            raw_provisions = [raw_provisions]
        marker_text = " ".join(
            [
                text,
                str(metadata.get("subject") or ""),
                str(metadata.get("title") or ""),
                *[str(value) for value in raw_provisions],
            ]
        )
        document = normalize_polish(marker_text)
        raw_domains = metadata.get("tax_domains") or []
        if isinstance(raw_domains, str):
            raw_domains = [raw_domains]
        candidate_domains = {str(value).upper() for value in raw_domains}
        expected_domains = {value.upper() for value in definition.tax_domains}
        expected_domain_present = (
            not expected_domains
            or bool(candidate_domains.intersection(expected_domains))
            or any(phrase_present(domain, document) for domain in definition.tax_domains)
        )
        markers: list[str] = []
        for value in (
            definition.canonical_name,
            *definition.exact_aliases,
            *definition.lemma_aliases,
            *definition.statutory_phrases,
            *definition.material_concepts,
        ):
            if phrase_present(value, document):
                markers.append(value)
        for hint in definition.provision_hints:
            if phrase_present(hint, document):
                markers.append(hint)
                continue
            # Corpus metadata may render a verified hint as ``[VAT] art.
            # 89a`` rather than ``VAT art. 89a``.  Compare the editorial
            # article token as a bounded marker instead of relying on one
            # storage-specific act prefix.
            article = _ARTICLE_RE.search(normalize_polish(hint).normalized)
            if expected_domain_present and article and re.search(
                rf"\bart\.\s*{re.escape(article.group(1))}\b",
                document.normalized,
                flags=re.IGNORECASE,
            ):
                markers.append(hint)
        return tuple(dict.fromkeys(markers))


# Public names make the trace contract explicit without coupling callers to
# the internal dataclass module.
InstitutionMatch = InstitutionMatchRecord
InstitutionMatchResult = InstitutionMatchResultRecord
