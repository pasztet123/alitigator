from __future__ import annotations

import unittest
from unittest.mock import patch

from app.controlled_authority_retrieval import audit_judgment_corpus, retrieve_housing_authorities
from app.housing_relief_pipeline import HOUSING_RELIEF_BENCHMARK_QUERY, run_housing_relief_pipeline
from app.rag import RagChunk, RagDocumentContext


def _chunk(*, text: str, provision: str, signature: str) -> RagChunk:
    return RagChunk(
        chunk_id=f"interpretation:{signature}",
        document_id=signature,
        chunk_index=0,
        score=1.0,
        chunk_text=text,
        subject="Ulga mieszkaniowa",
        signature=signature,
        published_date="2026-01-15",
        source_url="https://example.test/authority",
        category=None,
        source_type="interpretation",
        legal_provisions=[provision],
    )


GOOD_CREDIT = _chunk(
    signature="0115-KDIT3.4011.1.2026.1.AK",
    provision="art. 21 ust. 30a ustawy PIT",
    text=(
        "Ocena stanowiska. Stanowisko jest prawidłowe: spłata kredytu zaciągniętego "
        "na zakup sprzedanej nieruchomości może stanowić wydatek na własne cele mieszkaniowe. "
        "Uzasadnienie interpretacji."
    ),
)

WRONG_MORTGAGE_NEIGHBOR = _chunk(
    signature="0115-KDIT3.4011.2.2026.1.AK",
    provision="art. 21 ust. 30a ustawy PIT",
    text=(
        "Ocena stanowiska. Stanowisko jest prawidłowe w sprawie umorzenia kredytu "
        "mieszkaniowego przez bank i zwolnienia z długu. Uzasadnienie interpretacji."
    ),
)

TRUNCATED_HOLDING = _chunk(
    signature="0115-KDIT3.4011.3.2026.1.AK",
    provision="art. 21 ust. 30a ustawy PIT",
    text="Ocena stanowiska. Stanowisko jest prawidłowe w sprawie spłaty kredytu",
)


class ControlledHousingAuthorityRetrievalTests(unittest.TestCase):
    def test_retrieves_per_issue_with_provision_anchored_queries_and_pairwise_binding(self) -> None:
        calls: list[tuple[str, set[str]]] = []

        def search(query: str, *, limit: int, source_types: set[str]):
            calls.append((query, source_types))
            if source_types == {"interpretation"} and "art. 21 ust. 30a" in query:
                return [GOOD_CREDIT, WRONG_MORTGAGE_NEIGHBOR]
            return []

        cards, outcome = retrieve_housing_authorities(
            "Sprzedałem mieszkanie i spłaciłem kredyt.",
            search=search,
            context_fetcher=lambda document_ids, *, seed_chunks: [],
        )

        # Each issue/source lane uses both factual and provision-first recall.
        self.assertEqual(len(calls), 16)
        issued_queries = outcome["authority_queries"]
        self.assertTrue(outcome["authority_queries_per_issue"])
        self.assertFalse(outcome["generic_housing_relief_pool_reused_for_all_claims"])
        self.assertTrue(outcome["authority_provision_match_scored"])
        self.assertTrue(outcome["interpretation_lane_executed"])
        self.assertTrue(outcome["interpretation_candidates_before_filters_recorded"])
        self.assertTrue(outcome["interpretation_lane"]["candidate_waterfall"])
        self.assertTrue(any("art. 21 ust. 30a" in item["query"] for item in issued_queries))
        self.assertTrue(any("art. 21 ust. 25a" in item["query"] for item in issued_queries))
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["issue_id"], "credit_on_sold_property")
        self.assertEqual(card["transaction_type"], "credit_repayment")
        self.assertEqual(card["event_type"], "repayment_from_sale_proceeds")
        self.assertEqual(card["holding_section"], "assessment")
        self.assertTrue(card["holding_complete_sentence"])
        self.assertEqual(card["provision_match_score"], 1.0)
        self.assertGreater(card["authority_score"], 0.7)
        self.assertEqual(card["claim_bindings"], [card["claim_bindings"][0]])
        self.assertEqual(card["claim_bindings"][0]["claim_id"], "claim_credit_scope")
        self.assertGreater(card["claim_bindings"][0]["score"], 0)
        self.assertTrue(card["claim_bindings"][0]["reason"])
        self.assertNotIn(WRONG_MORTGAGE_NEIGHBOR.signature, [item["label"] for item in cards])

        result = run_housing_relief_pipeline(HOUSING_RELIEF_BENCHMARK_QUERY, authority_cards=cards)
        source_line = next(line for line in result.answer.splitlines() if GOOD_CREDIT.signature in line)
        self.assertLessEqual(len(source_line.split()), 120)
        self.assertIn("Holding:", source_line)
        self.assertIn("Wynik:", source_line)
        self.assertIn("Podobieństwo:", source_line)
        self.assertIn("Różnica:", source_line)
        self.assertNotIn("claim_credit_scope", source_line)

    def test_abstains_when_holding_is_incomplete_or_candidate_is_wrong_neighbor(self) -> None:
        def search(query: str, *, limit: int, source_types: set[str]):
            if source_types == {"interpretation"} and "art. 21 ust. 30a" in query:
                return [WRONG_MORTGAGE_NEIGHBOR, TRUNCATED_HOLDING]
            return []

        cards, outcome = retrieve_housing_authorities(
            "Spłata kredytu po sprzedaży mieszkania.",
            search=search,
            context_fetcher=lambda document_ids, *, seed_chunks: [],
        )

        self.assertEqual(cards, [])
        self.assertEqual(outcome["outcome"], "no_high_quality_authorities")
        self.assertTrue(outcome["empty_high_quality_result_supported"])
        self.assertGreater(outcome["filtered_counts"]["interpretation"], 0)

    def test_extracts_holding_from_complete_document_not_seed_chunk(self) -> None:
        seed = _chunk(
            signature="0115-KDIT3.4011.4.2026.1.AK",
            provision="art. 21 ust. 30a ustawy PIT",
            text="Fragment wyszukiwawczy o art. 21 ust. 30a i spłacie kredytu.",
        )
        context = RagDocumentContext(
            document_id=seed.document_id,
            subject=seed.subject,
            signature=seed.signature,
            published_date=seed.published_date,
            source_url=seed.source_url,
            category=None,
            source="eureka",
            source_type="interpretation",
            source_subtype=None,
            authority="Dyrektor KIS",
            publication=None,
            legal_state_date=None,
            source_pages=[],
            legal_provisions=["art. 21 ust. 30a ustawy PIT"],
            text=GOOD_CREDIT.chunk_text,
            seed_chunk_ids=[seed.chunk_id],
        )

        cards, _ = retrieve_housing_authorities(
            "Spłata kredytu po sprzedaży mieszkania.",
            search=lambda query, *, limit, source_types: [seed]
            if source_types == {"interpretation"} and "art. 21 ust. 30a" in query
            else [],
            context_fetcher=lambda document_ids, *, seed_chunks: [context],
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["holding_source_span"]["scope"], "document_context")
        self.assertEqual(cards[0]["holding_source_span"]["document_id"], seed.document_id)

    def test_judgment_audit_records_corpus_index_and_filter_counts(self) -> None:
        cards, outcome = retrieve_housing_authorities(
            "Sprzedaż mieszkania.",
            search=lambda query, *, limit, source_types: [],
            context_fetcher=lambda document_ids, *, seed_chunks: [],
        )

        self.assertEqual(cards, [])
        judgment = outcome["judgment_lane"]
        self.assertTrue(outcome["judgment_corpus_count_recorded"])
        self.assertTrue(outcome["judgment_indexed_count_recorded"])
        self.assertIn("corpus_count", judgment)
        self.assertIn("indexed_count", judgment)
        self.assertIn("filtered_count", judgment)
        self.assertIn("zero_candidates_root_cause", judgment)

    def test_judgment_audit_identifies_unindexed_active_backend(self) -> None:
        with patch(
            "app.controlled_authority_retrieval._local_judgment_corpus_count",
            return_value=(2365, True),
        ), patch(
            "app.controlled_authority_retrieval._active_judgment_index_count",
            return_value=(0, None),
        ):
            audit = audit_judgment_corpus(
                candidate_count=0,
                selected_count=0,
                filtered_count=0,
                errors=[],
            )

        self.assertEqual(audit["corpus_count"], 2365)
        self.assertEqual(audit["indexed_count"], 0)
        self.assertEqual(
            audit["zero_candidates_root_cause"],
            "judgment_corpus_not_indexed_in_active_backend",
        )


if __name__ == "__main__":
    unittest.main()
