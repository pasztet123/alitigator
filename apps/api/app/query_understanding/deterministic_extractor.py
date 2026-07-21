"""Build a QuestionCard from generic concept types, aliases and relations."""
from __future__ import annotations

import re
from collections import defaultdict

from app.legal_concepts import ConceptMatcher
from app.legal_concepts.normalizer import normalize_text

from .models import ProvisionHint, QuestionCard

_PROVISION = re.compile(r"\bart\.\s*\d+[a-z]*(?:\s+ust\.\s*\d+[a-z]*)?(?:\s+pkt\s*\d+)?", re.I)


def build_question_card(question: str, *, matcher: ConceptMatcher | None = None) -> tuple[QuestionCard, object]:
    matcher = matcher or ConceptMatcher()
    result = matcher.match(question)
    dictionary = matcher.dictionary
    by_type: dict[str, list[str]] = defaultdict(list)
    hints: list[ProvisionHint] = []
    negative: list[str] = []
    domains: list[str] = []
    for match in result.matches:
        definition = dictionary.by_id[match.concept_id]
        by_type[match.concept_type].append(match.concept_id)
        domains.extend(definition.tax_domains)
        if match.concept_type == "legal_institution" and match.locked:
            negative.extend(definition.incompatible_concepts)
        for citation in definition.verified_provision_hints:
            hints.append(ProvisionHint(citation, "concept_dictionary", True))
    def values_for(concept_type: str) -> list[str]:
        candidates = [item for item in result.matches if item.concept_type == concept_type]
        def rank(item: object) -> tuple[int, int, str]:
            match = item
            direct = 0 if getattr(match, "match_type") != "related_concept" else 1
            source_id = (getattr(match, "evidence", ()) or ("",))[0]
            source = dictionary.by_id.get(source_id)
            material_source = 0 if source and source.concept_type in {"product_or_service", "contract_type", "transaction_type"} else 1
            return direct, material_source, getattr(match, "concept_id")
        return [item.concept_id for item in sorted(candidates, key=rank)]
    normal = normalize_text(question)
    explicit = list(dict.fromkeys(match.group(0) for match in _PROVISION.finditer(normal.original)))
    # Direction is an explicit linguistic fact, not a tax conclusion. It is
    # guarded by the independently matched role categories.
    roles = values_for("entity_role")
    directions = [item.concept_id for item in result.matches if dictionary.by_id[item.concept_id].semantic_role == "payment_direction"]
    direction = directions[0] if directions and len(roles) >= 2 else None
    card = QuestionCard(
        question_id="", original_question=question, normalized_question=result.normalized_question,
        tax_domains=list(dict.fromkeys(domains)), locked_institutions=[
            item.concept_id for item in result.matches if item.locked and item.concept_type == "legal_institution"
        ], detected_concepts=[item.concept_id for item in result.matches],
        taxpayer_roles=roles[:1], counterparty_roles=roles[1:], payment_direction=direction,
        payment_types=values_for("payment_type"), transaction_types=values_for("transaction_type"),
        contract_types=values_for("contract_type"), products_or_services=values_for("product_or_service"),
        explicit_provisions=explicit, verified_provision_hints=list(dict.fromkeys(hints)),
        material_facts=by_type["factual_concept"], negative_concepts=list(dict.fromkeys(negative)),
    )
    return card, result
