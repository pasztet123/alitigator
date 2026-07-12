from __future__ import annotations

import unittest
from unittest.mock import patch

from app.main import retrieve_controlled_authority_lane
from app.rag import RagChunk


def _chunk(source_type: str, signature: str) -> RagChunk:
    return RagChunk(
        chunk_id=f"{source_type}:{signature}",
        document_id=signature,
        chunk_index=0,
        score=1.0,
        chunk_text="Treść authority.",
        subject="Authority",
        signature=signature,
        published_date="2026-01-01",
        source_url="https://example.test/authority",
        category=None,
        source_type=source_type,
    )


class ControlledAuthorityLaneTests(unittest.TestCase):
    def test_lane_runs_both_authority_types_and_records_empty_or_found_outcome(self) -> None:
        with patch(
            "app.main.search_chunks",
            side_effect=[[_chunk("interpretation", "0115-KDIT3.4011.1.2026.1.AK")], []],
        ) as search:
            cards, outcome = retrieve_controlled_authority_lane("Sprzedaż mieszkania i PIT")

        self.assertEqual(search.call_count, 2)
        self.assertTrue(outcome["authority_lane_executed"])
        self.assertTrue(outcome["authority_candidates_count_recorded"])
        self.assertTrue(outcome["judgment_lane_executed"])
        self.assertTrue(outcome["judgment_candidate_count_recorded"])
        self.assertTrue(outcome["judgment_selected_count_recorded"])
        self.assertTrue(outcome["judgment_empty_result_reason_recorded"])
        self.assertEqual(outcome["candidate_counts"], {"interpretation": 1, "judgment": 0})
        self.assertFalse(outcome["empty_authority_result_explained"])
        self.assertEqual(
            outcome["judgment_lane"],
            {
                "executed": True,
                "candidate_count": 0,
                "selected_count": 0,
                "empty_result_reason": "no_candidates_from_corpus",
            },
        )
        self.assertEqual(cards[0]["label"], "0115-KDIT3.4011.1.2026.1.AK")
