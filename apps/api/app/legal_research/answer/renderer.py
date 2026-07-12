from __future__ import annotations

from typing import Any, Mapping

from app.legal_research.models import AnswerDraft, CalculationRecord


def _source_line(source: Mapping[str, Any]) -> str:
    reference = str(source.get("display_reference") or source.get("signature") or source.get("document_id") or "").strip()
    role = str(source.get("role") or source.get("issue") or "źródło dla zatwierdzonego claimu").strip()
    details = [str(source.get(key) or "").strip() for key in ("date", "document_type", "holding", "result_for_taxpayer", "link")]
    return " — ".join([reference, role, *[item for item in details if item]])


def render_answer(
    draft: AnswerDraft, *, source_registry: Mapping[str, Mapping[str, Any]],
    calculations: list[CalculationRecord], allowed_source_ids: set[str],
) -> str:
    thesis = "\n".join(f"- {item}" for item in draft.thesis) or "Brak zatwierdzonej tezy."
    analysis = "\n\n".join(
        f"## {section.heading}\n" + "\n\n".join(section.paragraphs)
        for section in draft.analysis_sections
    ) or "Brak zatwierdzonej analizy."
    used = [*draft.provision_ids_used, *draft.authority_ids_used]
    sources = [
        _source_line(source_registry[source_id])
        for source_id in dict.fromkeys(used)
        if source_id in allowed_source_ids and source_id in source_registry
    ]
    source_text = "\n".join(f"- {item}" for item in sources) or "Nie znaleziono źródeł dostatecznych do materialnej konkluzji."
    risks = "\n".join(f"- {item}" for item in draft.risks_and_gaps) or "- Brak zidentyfikowanych luk."
    return f"Teza\n{thesis}\n\nAnaliza\n{analysis}\n\nŹródła\n{source_text}\n\nRyzyka i luki\n{risks}"


__all__ = ["render_answer"]
