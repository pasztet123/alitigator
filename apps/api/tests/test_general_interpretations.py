from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.legal_rag_v2.backends import CorpusFtsBackend, _normalized_source_type
from app.mysql_rag import build_type_and_domain_clause, local_record_to_mysql_document
from app.rag import (
    LegalRetrievalAxis,
    RagChunk,
    chunk_matches_axis_source_type,
    derive_source_subtype,
    normalize_source_type,
)


SCRIPTS_PATH = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))
from sync_general_interpretations_to_mysql import canonical_general_document, general_search_text


class GeneralInterpretationArchitectureTests(unittest.TestCase):
    def test_legacy_general_metadata_normalizes_to_dedicated_authority_type(self) -> None:
        legacy = {
            "source_type": "interpretation",
            "source_subtype": "general",
            "category": "Interpretacja ogólna",
        }
        self.assertEqual("general", derive_source_subtype(legacy))
        self.assertEqual("general_interpretation", normalize_source_type(legacy))
        self.assertEqual("general_interpretation", local_record_to_mysql_document({
            **legacy,
            "document_id": "DD4.8201.2.2026",
            "subject": "Interpretacja ogólna MF",
        })["source_type"])

    def test_generic_interpretation_scope_includes_general_interpretations(self) -> None:
        clause, values, _domains = build_type_and_domain_clause(
            source_types={"interpretation"},
            enforce_query_domain=False,
            tax_domains=None,
            detection_query="koszty uzyskania przychodów",
            config=SimpleNamespace(domain_filter_enabled=False),
        )
        self.assertIn("d.source_type IN", clause)
        self.assertEqual(["general_interpretation", "interpretation"], values)

    def test_general_interpretation_is_kept_for_legacy_interpretation_axis(self) -> None:
        chunk = RagChunk(
            chunk_id="general:1",
            document_id="general",
            chunk_index=0,
            score=1.0,
            chunk_text="Interpretacja ogólna.",
            subject="Interpretacja ogólna MF",
            signature="DD4.8201.2.2026",
            published_date="2026-01-01",
            source_url=None,
            category="Interpretacja ogólna",
            source_type="general_interpretation",
            source_subtype="general",
        )
        axis = LegalRetrievalAxis(
            axis_id="authority",
            label="Interpretacje",
            query="koszt podatkowy",
            source_types={"interpretation"},
        )
        self.assertTrue(chunk_matches_axis_source_type(axis, chunk))

    def test_v2_backend_maps_legacy_general_row_and_queries_compatibly(self) -> None:
        self.assertEqual(
            "general_interpretation",
            _normalized_source_type({"source_type": "interpretation", "source_subtype": "general"}),
        )
        self.assertEqual(
            ["general_interpretation", "interpretation"],
            CorpusFtsBackend._storage_source_types(frozenset({"general_interpretation"})),
        )

    def test_sync_row_is_canonical_and_searchable_by_authority_class(self) -> None:
        raw = {
            "document_id": "DD4.8201.2.2026",
            "content_sha256": "a" * 64,
            "source": "eureka",
            "source_type": "interpretation",
            "source_subtype": "general",
            "authority": "",
            "jurisdiction": "PL",
            "act_title": "",
            "publication": "",
            "legal_state_date": "",
            "source_pages_json": "[]",
            "subject": "Koszty podatkowe",
            "signature": "DD4.8201.2.2026",
            "published_date": "2026-01-01",
            "source_url": "https://example.test",
            "category": "Interpretacja ogólna",
            "keywords_json": "[\"koszty\"]",
            "legal_provisions_json": "[\"art. 22\"]",
            "issues_json": "[]",
            "law_tags_json": "[\"PIT\"]",
            "tax_domain": "PIT",
            "signature_family": "",
            "question_text": "",
            "facts_text": "",
            "decision_text": "",
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "chunk_text": "Minister Finansów wyjaśnia zasady kosztów.",
        }
        canonical = canonical_general_document(raw)
        self.assertEqual("general_interpretation", canonical["source_type"])
        self.assertEqual("general", canonical["source_subtype"])
        self.assertEqual("Minister Finansów", canonical["authority"])
        self.assertIn("Interpretacja ogólna | Minister Finansów", general_search_text(raw))


if __name__ == "__main__":
    unittest.main()
