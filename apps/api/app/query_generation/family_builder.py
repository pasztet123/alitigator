"""Generate bounded query channels from types and dictionary relationships."""
from __future__ import annotations

from app.legal_concepts import load_default_dictionary
from app.query_understanding.models import QueryFamilySpec, QueryPlan

QUERY_BUILDER_VERSION = "query_family_builder_v2"


def _add(target: list[QueryFamilySpec], family_type: str, query: str, *, hardness: str = "soft", limit: int = 40, domains: list[str] | None = None, source_types: list[str] | None = None) -> None:
    clean = " ".join(query.split())
    if clean and not any(item.type == family_type and item.query.casefold() == clean.casefold() for item in target):
        target.append(QueryFamilySpec(f"{family_type}_{len(target) + 1}", family_type, clean, must_any=clean.split()[:3], tax_domains=domains or [], source_types=source_types or [], limit=limit, hardness=hardness, weight=1.4 if hardness == "hard" else 1.0))


def build_query_families(plan: QueryPlan | dict) -> list[QueryFamilySpec]:
    if isinstance(plan, dict):
        return []
    dictionary = load_default_dictionary()
    card = plan.question_card
    families: list[QueryFamilySpec] = []
    _add(families, "exact_user_phrase", card.original_question, limit=25, domains=card.tax_domains)
    for concept_id in card.locked_institutions:
        definition = dictionary.by_id.get(concept_id)
        if not definition:
            continue
        _add(families, "locked_institution", definition.canonical_name, hardness="hard", domains=card.tax_domains, source_types=list(definition.source_types))
        for hint in definition.verified_provision_hints[:2]: _add(families, "verified_provision", hint, hardness="hard", domains=card.tax_domains)
        for term in definition.statutory_terms[:1]: _add(families, "statutory_language", term, domains=card.tax_domains)
    type_to_family = {"product_or_service": "product_or_service", "contract_type": "contract_type", "transaction_type": "material_facts", "payment_type": "material_facts"}
    for concept_id in card.detected_concepts:
        definition = dictionary.by_id.get(concept_id)
        if definition and definition.concept_type in type_to_family:
            _add(families, type_to_family[definition.concept_type], definition.canonical_name, domains=card.tax_domains)
    for term in plan.legal_concepts[:2]: _add(families, "legal_concepts", term, domains=card.tax_domains)
    for term in plan.factual_synonyms[:2]: _add(families, "model_soft_expansion", term, domains=card.tax_domains)
    if card.taxpayer_roles and card.counterparty_roles:
        labels = [dictionary.by_id[item].canonical_name for item in [*card.taxpayer_roles, *card.counterparty_roles] if item in dictionary.by_id]
        _add(families, "role_and_direction", " ".join(labels[:2]), domains=card.tax_domains)
    plan.query_families = families[:8]
    return plan.query_families
