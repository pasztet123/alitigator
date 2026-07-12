from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from app.legal_research.claims.calculations import (
    calculate_deadline, condition_applied_to_deadline,
)
from app.legal_research.evidence.authority_extractor import (
    SourceSpanValidationError, validate_holding_span,
)
from app.legal_research.evidence.binding import authority_abstention_reasons
from app.legal_research.models import ResearchFact, SourceSpan
from app.legal_research.tracing import REQUIRED_ARTIFACTS, TraceWriter
from app.main import get_legal_pipeline_mode


class ModelRagModelContractTests(unittest.TestCase):
    def test_public_mode_is_available_without_changing_legacy(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_MODE": "model_rag_model"}):
            self.assertEqual(get_legal_pipeline_mode(), "model_rag_model")
        with patch.dict(os.environ, {"LEGAL_RAG_MODE": "legacy"}):
            self.assertEqual(get_legal_pipeline_mode(), "legacy")

    def test_explicit_fact_requires_exact_source_lineage(self) -> None:
        with self.assertRaises(ValueError):
            ResearchFact(
                fact_id="f1", subject="podatnik", role="sprzedający",
                predicate="sprzedał", value="lokal", status="explicit",
            )
        fact = ResearchFact(
            fact_id="f1", subject="podatnik", role="sprzedający",
            predicate="sprzedał", value="lokal", status="explicit",
            source_span=SourceSpan(start=0, end=5),
        )
        self.assertEqual(fact.source_span.end, 5)

    def test_trace_contains_every_required_diagnostic_stage(self) -> None:
        expected = {
            "planner_fallback.json", "first_pass_reranking.json", "legal_rules.json",
            "wrong_neighbor_rejections.json", "evidence_bindings.json",
            "missing_evidence_requests.json", "second_pass_queries.json",
            "second_pass_candidates.json", "authority_lineage.json", "token_usage.json",
        }
        self.assertTrue(expected.issubset(REQUIRED_ARTIFACTS))
        with tempfile.TemporaryDirectory() as directory:
            writer = TraceWriter("contract", root=Path(directory))
            writer.initialize_required()
            self.assertTrue(all(writer.path_for(name).exists() for name in REQUIRED_ARTIFACTS))

    def test_deadline_boundary_is_deterministic(self) -> None:
        record = calculate_deadline(
            calculation_id="deadline", event_date=date(2022, 4, 3), statutory_years=3,
            provision_ids=["p1"], fact_ids=["f1"],
        )
        deadline = date.fromisoformat(str(record.result))
        self.assertEqual(condition_applied_to_deadline(event_date=date(2025, 12, 30), deadline=deadline), "before")
        self.assertEqual(condition_applied_to_deadline(event_date=date(2025, 12, 31), deadline=deadline), "on_deadline")
        self.assertEqual(condition_applied_to_deadline(event_date=date(2026, 1, 1), deadline=deadline), "after")

    def test_wrong_neighbor_thresholds_abstain_independently(self) -> None:
        reasons = authority_abstention_reasons(
            topic_similarity=.9, issue_similarity=.8, material_fact_similarity=.1,
            holding_relevance=.9, min_topic_similarity=.2, min_issue_similarity=.3,
            min_material_fact_similarity=.4, min_holding_relevance=.3,
        )
        self.assertEqual(reasons, ["below_min_material_fact_similarity"])

    def test_holding_must_be_a_complete_sentence_and_word_aligned(self) -> None:
        text = "Organ uznał stanowisko za prawidłowe. Dalej."
        end = text.index(".") + 1
        self.assertEqual(validate_holding_span(text, 0, end), text[:end])
        with self.assertRaises(SourceSpanValidationError):
            validate_holding_span(text, 1, end)
        with self.assertRaises(SourceSpanValidationError):
            validate_holding_span(text, 0, end - 1)


if __name__ == "__main__":
    unittest.main()
