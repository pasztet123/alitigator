"""Compatibility entry point for deterministic query understanding."""
from __future__ import annotations
from .deterministic_extractor import build_question_card
from .merger import merge_query_plan
from .models import ModelQueryExpansion, QueryPlan


def analyze_query(question: str, *, locked_institutions: list[str] | None = None) -> QueryPlan:
    card, result = build_question_card(question)
    # Compatibility locks never grant a taxonomy fact: only a matcher lock
    # remains authoritative. They are retained solely for pre-existing callers.
    if locked_institutions:
        card.locked_institutions = list(dict.fromkeys([*card.locked_institutions, *locked_institutions]))
    return merge_query_plan(card, result, ModelQueryExpansion())
