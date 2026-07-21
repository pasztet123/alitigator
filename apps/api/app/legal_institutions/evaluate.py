"""Deterministic quality metrics for the named-institution retrieval stage."""

from __future__ import annotations

from typing import Iterable, Mapping

from .dictionary import InstitutionDictionary, load_default_dictionary
from .matcher import InstitutionMatcher
from .merger import institution_query_families


def evaluate_dictionary_cases(
    positive_cases: Iterable[Mapping[str, object]],
    negative_cases: Iterable[Mapping[str, object]],
    *,
    matcher: InstitutionMatcher | None = None,
    dictionary: InstitutionDictionary | None = None,
) -> dict[str, object]:
    """Return reproducible recognition and candidate-generation metrics.

    This deliberately reports only what the deterministic stage can prove.  A
    model answer's legal correctness is outside this metric and must be
    evaluated from evidence separately.
    """

    dictionary = dictionary or load_default_dictionary()
    matcher = matcher or InstitutionMatcher(dictionary)
    positives = list(positive_cases)
    negatives = list(negative_cases)
    positive_hits = 0
    positive_locks = 0
    for case in positives:
        matches = matcher.match(str(case["question"])).matches
        target = str(case["institution_id"])
        hit = next((item for item in matches if item.institution_id == target), None)
        positive_hits += int(hit is not None)
        positive_locks += int(hit is not None and hit.locked == bool(case.get("expects_lock")))
    negative_without_lock = sum(
        not any(item.locked for item in matcher.match(str(case["question"])).matches)
        for case in negatives
    )
    active = [item for item in dictionary.institutions if item.status == "active"]
    channel_counts = [len(institution_query_families(item)) for item in active]
    return {
        "dictionary_version": dictionary.version,
        "institution_count": len(dictionary.institutions),
        "active_institution_count": len(active),
        "shadow_institution_count": sum(item.status == "shadow" for item in dictionary.institutions),
        "positive_case_count": len(positives),
        "positive_recognition_coverage": positive_hits / len(positives) if positives else 0.0,
        "positive_lock_contract_coverage": positive_locks / len(positives) if positives else 0.0,
        "negative_case_count": len(negatives),
        "negative_no_lock_rate": negative_without_lock / len(negatives) if negatives else 0.0,
        "candidate_channels": {
            "mean_per_active_institution": (sum(channel_counts) / len(channel_counts)) if channel_counts else 0.0,
            "min_per_active_institution": min(channel_counts, default=0),
            "required_families": [
                "canonical",
                "aliases",
                "provision_hints",
                "statutory_phrases",
                "material_concepts",
            ],
        },
        "direct_gate": {
            "required_marker_classes": [
                "canonical_name",
                "verified_alias",
                "provision_hint",
                "statutory_phrase",
                "material_concept",
            ],
            "rejection_reason": "missing_locked_institution_markers",
        },
        "first_success_criteria": {
            "all_active_locks_preserved": positive_locks == len(positives),
            "no_false_lock_in_negative_suite": negative_without_lock == len(negatives),
            "direct_gate_enabled": True,
        },
    }
