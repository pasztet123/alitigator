"""Merge deterministic institution locks into a model-created research plan."""

from __future__ import annotations

from typing import Iterable

from app.legal_rag_v2.schemas import (
    DeterministicInstitutionLock,
    InstitutionPlannerConflict,
    LegalIssue,
    LegalResearchPlan,
    QueryFamily,
)

from .dictionary import InstitutionDictionary, load_default_dictionary
from .matcher import InstitutionMatchResult
from .schema import InstitutionDefinition, InstitutionMatchRecord


_GENERIC_MECHANISMS = {
    "",
    "general",
    "analysis",
    "general_tax_analysis",
    "business_expense",
    "pit_cost_deductibility",
    "cit_cost_deductibility",
}
_SOURCE_TYPES = {
    "statute",
    "regulation",
    "treaty",
    "tax_treaty",
    "interpretation",
    "general_interpretation",
    "guidance",
    "tax_guidance",
    "judgment",
    "resolution",
}


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split())
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _lock_record(match: InstitutionMatchRecord) -> DeterministicInstitutionLock:
    return DeterministicInstitutionLock(
        institution_id=match.institution_id,
        canonical_name=match.canonical_name,
        confidence=match.confidence,
        match_type=match.match_type,
        matched_text=match.matched_text,
        tax_domains=list(match.tax_domains),
        provision_hints=list(match.provision_hints),
        material_concepts=list(match.material_concepts),
        source_preferences=list(match.source_preferences),
    )


def institution_query_families(definition: InstitutionDefinition) -> list[QueryFamily]:
    """Produce bounded, source-neutral candidate channels for one lock."""

    families: list[QueryFamily] = []

    def add(family: str, query: str) -> None:
        clean = " ".join(query.split())
        if clean and not any(item.family == family and item.query.casefold() == clean.casefold() for item in families):
            families.append(QueryFamily(
                family=family,  # type: ignore[arg-type]
                query=clean,
                lane="both",
                origin="deterministic",
            ))

    add("named_institution_canonical", definition.canonical_name)
    for alias in definition.exact_aliases[:4]:
        add("named_institution_alias", alias)
    for hint in definition.provision_hints[:2]:
        add("named_institution_provision", hint)
    for phrase in definition.statutory_phrases[:1]:
        add("named_institution_statutory", phrase)
    return families


def _issue_for_lock(issues: list[LegalIssue], definition: InstitutionDefinition) -> int:
    domains = set(definition.tax_domains)
    for index, issue in enumerate(issues):
        if domains and domains.intersection(issue.tax_domains):
            return index
    return 0


def merge_locked_institutions(
    plan: LegalResearchPlan,
    match_result: InstitutionMatchResult,
    *,
    dictionary: InstitutionDictionary | None = None,
) -> LegalResearchPlan:
    """Preserve hard recognitions while retaining all model-generated issues.

    A model may add hypotheses, but it cannot remove or downgrade an active
    deterministic lock.  Conflicts are kept in the plan trace with the fixed
    resolution ``keep_deterministic``.
    """

    dictionary = dictionary or load_default_dictionary()
    locks = [match for match in match_result.matches if match.locked]
    if not locks:
        return plan.model_copy(update={
            "deterministic_institutions": [],
            "institution_dictionary_version": match_result.dictionary_version,
        })

    issues = [issue.model_copy(update={"locked_institution_ids": []}) for issue in plan.issues]
    conflicts: list[InstitutionPlannerConflict] = []
    model_level_ids = set(plan.model_inferred_institutions)
    for issue in plan.issues:
        model_level_ids.update(issue.model_inferred_institution_ids)

    for match in locks:
        definition = dictionary.by_id.get(match.institution_id)
        if definition is None:
            continue
        target_index = _issue_for_lock(issues, definition)
        issue = issues[target_index]
        current_mechanism = issue.legal_mechanism.casefold().strip()
        recognised_by_model = (
            match.institution_id in model_level_ids
            or current_mechanism == match.institution_id
            or current_mechanism in {value.casefold() for value in definition.legal_mechanisms}
        )
        if model_level_ids and not recognised_by_model:
            conflicts.append(InstitutionPlannerConflict(
                institution_id=match.institution_id,
                model_mechanism=issue.legal_mechanism,
                model_inferred_institution_ids=sorted(model_level_ids),
                reason="model_institution_hypothesis_conflicts_with_deterministic_lock",
            ))
        mechanism = issue.legal_mechanism
        if current_mechanism != match.institution_id:
            # A deterministic lock is the primary retrieval mechanism for
            # this issue.  Leaving an unrelated or merely generic model label
            # in place would send the authority lane back to broad cost
            # searches even though the lock remains listed alongside it.
            mechanism = match.institution_id
            conflicts.append(InstitutionPlannerConflict(
                institution_id=match.institution_id,
                model_mechanism=issue.legal_mechanism,
                model_inferred_institution_ids=sorted(model_level_ids),
                reason=(
                    "generic_model_mechanism_cannot_replace_deterministic_lock"
                    if current_mechanism in _GENERIC_MECHANISMS
                    else "model_mechanism_cannot_replace_deterministic_lock"
                ),
            ))
        families = [*issue.query_families, *institution_query_families(definition)]
        issues[target_index] = issue.model_copy(update={
            "legal_mechanism": mechanism,
            "tax_domains": _dedupe([*issue.tax_domains, *definition.tax_domains]),
            "possible_provision_concepts": _dedupe([*issue.possible_provision_concepts, *definition.provision_hints]),
            "possible_legal_concepts": _dedupe([*issue.possible_legal_concepts, definition.canonical_name]),
            "transactions": _dedupe([*issue.transactions, *definition.material_concepts]),
            "requested_source_types": _dedupe([
                *issue.requested_source_types,
                *(value for value in definition.source_preferences if value in _SOURCE_TYPES),
            ]),
            "query_families": families,
            "locked_institution_ids": _dedupe([*issue.locked_institution_ids, match.institution_id]),
        })

    # Do not let a planner-supplied deterministic field create a lock.  Only
    # the matcher result above is authoritative for this run.
    return plan.model_copy(update={
        "issues": issues,
        "deterministic_institutions": [_lock_record(match) for match in locks],
        "institution_conflicts": conflicts,
        "institution_dictionary_version": match_result.dictionary_version,
    })
