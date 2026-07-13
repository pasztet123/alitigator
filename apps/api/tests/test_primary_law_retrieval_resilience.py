from __future__ import annotations

import unittest
from unittest.mock import patch

from app.rag import (
    RagChunk,
    add_primary_source_fallback_chunks,
    build_preferred_statute_targets,
    query_targets_mortgage_settlement_refund,
    resolve_statute_tax_domains,
)


def _statute_chunk() -> RagChunk:
    return RagChunk(
        chunk_id="cit-art-3:1",
        document_id="cit-art-3",
        chunk_index=1,
        score=1.0,
        chunk_text="Art. 3 ust. 1. Podatnicy podlegają obowiązkowi podatkowemu od całości dochodów.",
        subject="O podatku dochodowym od osób prawnych - art. 3",
        signature=None,
        published_date="2026-01-01",
        source_url="https://example.test/cit/art-3",
        category=None,
        source_type="statute",
        legal_provisions=["art. 3"],
    )


class PrimaryLawRetrievalResilienceTests(unittest.TestCase):
    def test_local_tax_act_is_not_misrouted_to_mortgage_pit_rules(self) -> None:
        query = "Jak ustawa o podatkach i opłatach lokalnych definiuje budynek?"

        self.assertFalse(query_targets_mortgage_settlement_refund(query))
        self.assertEqual(resolve_statute_tax_domains(query), {"NIERUCHOMOŚCI"})
        self.assertEqual(
            build_preferred_statute_targets(query),
            [("NIERUCHOMOŚCI", "1a")],
        )

    def test_common_benchmark_questions_resolve_controlling_articles(self) -> None:
        cases = {
            "Czy odpłatna dostawa towarów na terytorium kraju podlega VAT?": [
                ("VAT", "5")
            ],
            "Kiedy podatnik CIT podlega obowiązkowi podatkowemu od całości swoich dochodów?": [
                ("CIT", "3")
            ],
            "Na czym polega zasada in dubio pro tributario w Ordynacji podatkowej?": [
                ("ORDYNACJA", "2a")
            ],
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(build_preferred_statute_targets(query), expected)

    def test_fallback_uses_backend_routed_retrieval_not_raw_corpus_files(self) -> None:
        chunk = _statute_chunk()
        with (
            patch(
                "app.rag.retrieve_deterministic_statute_chunks",
                return_value=[chunk],
            ) as retrieve,
            patch(
                "app.rag.load_processed_statute_chunks_by_targets",
                side_effect=AssertionError("raw corpus must not be the request fallback"),
            ),
            patch(
                "app.rag.load_processed_statute_chunks_by_subject_prefix",
                side_effect=AssertionError("raw corpus must not be the request fallback"),
            ),
        ):
            result = add_primary_source_fallback_chunks(
                "Kiedy podatnik CIT podlega obowiązkowi od całości dochodów?",
                [],
            )

        retrieve.assert_called_once()
        self.assertEqual(result, [chunk])

    def test_existing_chunks_are_deduplicated_against_recovered_primary_law(self) -> None:
        chunk = _statute_chunk()
        with patch(
            "app.rag.retrieve_deterministic_statute_chunks",
            return_value=[chunk],
        ):
            result = add_primary_source_fallback_chunks("Pytanie CIT", [chunk])

        self.assertEqual(result, [chunk])
