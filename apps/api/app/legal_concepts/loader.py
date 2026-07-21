"""Load one versioned JSON taxonomy; groups keep the corpus concise and auditable."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import ConceptDefinition

_DATA_PATH = Path(__file__).parent / "dictionaries" / "pl_tax_concepts_v1.json"
_TYPES = {
    "legal_institution", "legal_mechanism", "tax_domain", "act", "provision_family",
    "entity_role", "payment_type", "transaction_type", "contract_type", "product_or_service",
    "procedural_instrument", "form_or_report", "factual_concept",
}


def _tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()) if str(item).strip())


class ConceptDictionary:
    def __init__(self, version: str, concepts: tuple[ConceptDefinition, ...]) -> None:
        self.version = version
        self.concepts = concepts
        self.by_id = {item.concept_id: item for item in concepts}
        if len(self.by_id) != len(concepts):
            raise ValueError("concept dictionary contains duplicate ids")

    @classmethod
    def from_path(cls, path: Path) -> "ConceptDictionary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = list(payload.get("concepts") or [])
        for group in payload.get("catalogue_groups") or []:
            base = {key: value for key, value in group.items() if key not in {"ids", "overrides"}}
            overrides = group.get("overrides") or {}
            for concept_id in group.get("ids") or []:
                entries.append({"id": concept_id, **base, **dict(overrides.get(concept_id) or {})})
        concepts: list[ConceptDefinition] = []
        seen_ids: set[str] = set()
        for item in entries:
            concept_type = str(item.get("type") or "factual_concept")
            if concept_type not in _TYPES:
                raise ValueError(f"unknown concept type: {concept_type}")
            concept_id = str(item["id"])
            # Explicit entries precede catalogue groups and can promote a
            # generic shadow placeholder without creating a second concept.
            if concept_id in seen_ids and str(item.get("status") or "shadow") == "shadow":
                continue
            if concept_id in seen_ids:
                raise ValueError(f"duplicate concept id: {concept_id}")
            seen_ids.add(concept_id)
            canonical = str(item.get("canonical_name") or concept_id.replace("_", " "))
            aliases = dict(item.get("aliases") or {})
            concepts.append(ConceptDefinition(
                concept_id=concept_id, concept_type=concept_type, canonical_name=canonical,
                status=str(item.get("status") or "shadow"), priority=str(item.get("priority") or "normal"), semantic_role=str(item.get("semantic_role") or ""),
                tax_domains=_tuple(item.get("tax_domains")), exact_aliases=_tuple(aliases.get("exact")),
                lemma_aliases=_tuple(aliases.get("lemma_phrases")), abbreviations=_tuple(aliases.get("abbreviations")),
                colloquial_aliases=_tuple(aliases.get("colloquial")), factual_aliases=_tuple(aliases.get("factual")),
                safe_regexes=_tuple(aliases.get("safe_regexes")), required_context=_tuple(item.get("required_context")),
                context_any=_tuple(item.get("context_any")), negative_context=_tuple(item.get("negative_context")),
                require_context_for_exact=bool(item.get("require_context_for_exact", False)),
                verified_provision_hints=_tuple(item.get("verified_provision_hints")),
                statutory_terms=_tuple(dict(item.get("query_terms") or {}).get("statutory")),
                legal_terms=_tuple(dict(item.get("query_terms") or {}).get("legal")),
                related_concepts=_tuple(item.get("related_concepts")), incompatible_concepts=_tuple(item.get("incompatible_concepts")),
                source_types=_tuple(item.get("source_types")),
            ))
        return cls(str(payload["version"]), tuple(concepts))

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for concept in self.concepts:
            counts[concept.concept_type] = counts.get(concept.concept_type, 0) + 1
            statuses[concept.status] = statuses.get(concept.status, 0) + 1
        return {"version": self.version, "total_entries": len(self.concepts), "entries_by_type": counts, **statuses}


@lru_cache(maxsize=1)
def load_default_dictionary() -> ConceptDictionary:
    return ConceptDictionary.from_path(_DATA_PATH)
