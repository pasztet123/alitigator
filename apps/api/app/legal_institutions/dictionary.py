"""Load and validate the versioned named-institution dictionary."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import InstitutionDefinition


_DATA_PATH = Path(__file__).parent / "dictionaries" / "pl_tax_institutions_v1.json"


def _humanize(identifier: str) -> str:
    return identifier.replace("_", " ")


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if not isinstance(value, list):
        raise ValueError("dictionary list fields must be JSON arrays")
    return tuple(str(item) for item in value if str(item).strip())


class InstitutionDictionary:
    def __init__(self, *, version: str, institutions: tuple[InstitutionDefinition, ...]) -> None:
        self.version = version
        self.institutions = institutions
        self.by_id = {item.institution_id: item for item in institutions}
        if len(self.by_id) != len(institutions):
            raise ValueError("named-institution dictionary contains duplicate ids")
        if len(institutions) < 120:
            raise ValueError("named-institution dictionary must contain at least 120 canonical institutions")
        if sum(item.status == "active" for item in institutions) < 50:
            raise ValueError("named-institution dictionary must activate at least 50 institutions")

    def contains_active(self, institution_id: str) -> bool:
        item = self.by_id.get(institution_id)
        return item is not None and item.status == "active"

    @classmethod
    def from_path(cls, path: Path) -> "InstitutionDictionary":
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = str(raw["version"])
        source_entries: list[dict[str, Any]] = list(raw.get("institutions") or [])
        # Catalogue groups keep the broad, versioned vocabulary data compact.
        # They are expanded here, before validation, into the same complete
        # schema as hand-authored institutions.
        for group in raw.get("catalogue_groups") or []:
            overrides = group.get("overrides") or {}
            defaults = {
                key: value
                for key, value in group.items()
                if key not in {"ids", "overrides"}
            }
            for identifier in group.get("ids") or []:
                source_entries.append({
                    "id": identifier,
                    **defaults,
                    **dict(overrides.get(identifier) or {}),
                })

        entries: list[InstitutionDefinition] = []
        active_ids: set[str] = set()
        for raw_entry in source_entries:
            identifier = str(raw_entry["id"])
            status = str(raw_entry.get("status", "shadow"))
            if status not in {"active", "shadow", "draft", "disabled"}:
                raise ValueError(f"invalid status for {identifier}: {status}")
            # A hand-authored active entry can promote one catalogue member
            # without activating its whole shadow group.  Retain the active
            # definition and ignore the older shadow catalogue placeholder.
            if identifier in active_ids and status == "shadow":
                continue
            if any(item.institution_id == identifier for item in entries):
                raise ValueError(f"named-institution dictionary contains duplicate id: {identifier}")
            entries.append(
                InstitutionDefinition(
                    institution_id=identifier,
                    canonical_name=str(raw_entry.get("canonical_name") or _humanize(identifier)),
                    status=status,  # type: ignore[arg-type]
                    rollout_stage=str(raw_entry.get("rollout_stage", "C")),
                    tax_domains=_as_tuple(raw_entry.get("tax_domains")),
                    exact_aliases=_as_tuple(raw_entry.get("exact_aliases")),
                    lemma_aliases=_as_tuple(raw_entry.get("lemma_aliases")),
                    safe_regexes=_as_tuple(raw_entry.get("safe_regexes")),
                    abbreviations=_as_tuple(raw_entry.get("abbreviations")),
                    colloquial_aliases=_as_tuple(raw_entry.get("colloquial_aliases")),
                    provision_hints=_as_tuple(raw_entry.get("provision_hints")),
                    statutory_phrases=_as_tuple(raw_entry.get("statutory_phrases")),
                    material_concepts=_as_tuple(raw_entry.get("material_concepts")),
                    contextual_signals=_as_tuple(raw_entry.get("contextual_signals")),
                    context_any_signals=_as_tuple(raw_entry.get("context_any_signals")),
                    require_context_for_exact=bool(raw_entry.get("require_context_for_exact", False)),
                    negative_context=_as_tuple(raw_entry.get("negative_context")),
                    source_preferences=_as_tuple(raw_entry.get("source_preferences")),
                    query_templates=_as_tuple(raw_entry.get("query_templates")),
                    legal_mechanisms=_as_tuple(raw_entry.get("legal_mechanisms")),
                )
            )
            if status == "active":
                active_ids.add(identifier)
        return cls(version=version, institutions=tuple(entries))


@lru_cache(maxsize=1)
def load_default_dictionary() -> InstitutionDictionary:
    return InstitutionDictionary.from_path(_DATA_PATH)


def validate_required_active_institutions() -> InstitutionDictionary:
    """Fail fast when an image lacks the MVP's active institution entries."""

    dictionary = load_default_dictionary()
    required = ("csr_sponsorship_relief", "expansion_relief")
    missing = [item for item in required if not dictionary.contains_active(item)]
    if missing:
        raise RuntimeError(
            "Named-institution dictionary is unavailable or missing active MVP entries: "
            + ", ".join(missing)
        )
    return dictionary
