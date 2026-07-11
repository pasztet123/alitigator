from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from app import law_chunk
from app.treaty_chunk import TreatySource, build_record as build_treaty_record


class ProvisionLevelChunkingTests(unittest.TestCase):
    def build_law_records(self, text: str, *, target_chars: int) -> list[dict]:
        pages = [law_chunk.PageText(number=7, text=text)]
        with patch.object(law_chunk, "extract_act_pages", return_value=pages):
            return law_chunk.build_records(
                Path("synthetic-act.pdf"),
                target_chars=target_chars,
                source_url="https://example.invalid/act.pdf",
                law_id="synthetic",
                short_title="ustawa syntetyczna",
                act_title="Ustawa syntetyczna",
                publication="Dz.U. test",
                legal_state_date="2026-01-01",
                published_date="2026-01-02",
                tax_tag="PIT",
            )

    def assert_unit_integrity(self, record: dict) -> None:
        self.assertEqual("provision_units_v1", record["chunker_version"])
        self.assertTrue(record["provision_units"])
        for unit in record["provision_units"]:
            start = unit["source_span_start"]
            end = unit["source_span_end"]
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(record["content_text"]))
            self.assertEqual(unit["text"], record["content_text"][start:end])
            expected_hash = hashlib.sha256(unit["text"].encode("utf-8")).hexdigest()
            self.assertEqual(expected_hash, unit["content_sha256"])
            self.assertEqual(expected_hash, unit["content_hash"])
            self.assertEqual(record["article_document_id"], unit["document_id"])
            self.assertEqual(record["document_id"], unit["record_document_id"])

    @staticmethod
    def units_by_citation(records: list[dict]) -> dict[str, dict]:
        return {
            unit["citation"]: unit
            for record in records
            for unit in record["provision_units"]
        }

    def test_law_records_expose_stable_nested_units_across_chunk_sizes(self) -> None:
        text = "\n".join(
            (
                "DZIAŁ I",
                "Art. 21. 1. Kosztami są:",
                "1) wydatki:",
                "a) udokumentowane fakturą,",
                "b) poniesione definitywnie;",
                "2) inne wydatki.",
                "2. Wyłączenie stosuje się odpowiednio.",
            )
        )
        split_records = self.build_law_records(text, target_chars=70)
        whole_records = self.build_law_records(text, target_chars=10_000)

        self.assertGreater(len(split_records), 1)
        self.assertEqual("pl-synthetic-art.-21", whole_records[0]["document_id"])
        self.assertTrue(all(record["article_document_id"] == "pl-synthetic-art.-21" for record in split_records))
        for record in [*split_records, *whole_records]:
            self.assert_unit_integrity(record)

        split_units = self.units_by_citation(split_records)
        whole_units = self.units_by_citation(whole_records)
        self.assertEqual(
            {citation: unit["provision_id"] for citation, unit in split_units.items()},
            {citation: unit["provision_id"] for citation, unit in whole_units.items()},
        )
        self.assertEqual(
            whole_units["art. 21"]["provision_id"],
            whole_units["art. 21 ust. 1"]["parent_id"],
        )
        self.assertEqual(
            whole_units["art. 21 ust. 1"]["provision_id"],
            whole_units["art. 21 ust. 1 pkt 1"]["parent_id"],
        )
        self.assertEqual(
            whole_units["art. 21 ust. 1 pkt 1"]["provision_id"],
            whole_units["art. 21 ust. 1 pkt 1 lit. a"]["parent_id"],
        )
        self.assertEqual(
            {"article", "section", "point", "letter"},
            {unit["unit_type"] for unit in whole_units.values()},
        )

    def test_long_provision_is_not_cut_to_reach_target_size(self) -> None:
        long_body = "bardzo długa niepodzielna treść " * 30
        text = f"Art. 7.\n1. {long_body}\n2. Krótka treść."
        records = self.build_law_records(text, target_chars=90)
        units = self.units_by_citation(records)

        long_unit = units["art. 7 ust. 1"]
        self.assertGreater(len(long_unit["text"]), 90)
        self.assertEqual(f"1. {long_body}".strip(), long_unit["text"])
        containing_records = [
            record
            for record in records
            if any(unit["citation"] == "art. 7 ust. 1" for unit in record["provision_units"])
        ]
        self.assertEqual(1, len(containing_records))
        self.assert_unit_integrity(containing_records[0])

    def test_paragraph_units_keep_the_article_parent_chain(self) -> None:
        records = self.build_law_records(
            "Art. 12.\n§ 1. Reguła paragrafowa.\n1. Pierwszy ustęp.\n1) Punkt.\na) Litera.",
            target_chars=10_000,
        )
        self.assertEqual(1, len(records))
        self.assert_unit_integrity(records[0])
        units = self.units_by_citation(records)
        paragraph = units["art. 12 § 1"]
        section = units["art. 12 § 1 ust. 1"]
        self.assertEqual("paragraph", paragraph["unit_type"])
        self.assertEqual(units["art. 12"]["provision_id"], paragraph["parent_id"])
        self.assertEqual(paragraph["provision_id"], section["parent_id"])

    def test_treaty_articles_include_sections_points_and_letters(self) -> None:
        source = TreatySource(
            country="Państwo Testowe",
            slug="test",
            variant="umowa",
            pdf_path=Path("synthetic-treaty.pdf"),
            source_url="https://example.invalid/treaty.pdf",
            act_title="Umowa syntetyczna",
            subject_prefix="UPO Polska - Państwo Testowe",
            publication="Dz.U. test",
            legal_state_date="2026-01-01",
            published_date="2026-01-02",
        )
        text = "\n".join(
            (
                "Artykuł 5",
                "ZAKŁAD",
                "1. W rozumieniu niniejszej Umowy:",
                "1) placówka:",
                "a) miejsce zarządu;",
                "2) biuro.",
                "2. Określenie nie obejmuje magazynu.",
            )
        )
        record = build_treaty_record(
            source,
            {"article": "5", "pages": [2, 3], "text": text},
        )

        self.assertEqual("pl-upo-test-umowa-art.-5", record["document_id"])
        self.assert_unit_integrity(record)
        units = self.units_by_citation([record])
        self.assertIn("art. 5 ust. 1", units)
        self.assertIn("art. 5 ust. 1 pkt 1", units)
        self.assertIn("art. 5 ust. 1 pkt 1 lit. a", units)
        self.assertIn("art. 5 ust. 2", units)


if __name__ == "__main__":
    unittest.main()
