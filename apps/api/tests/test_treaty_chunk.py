from __future__ import annotations

import unittest

from app.treaty_chunk import (
    CORE_TREATY_SOURCES,
    build_outputs,
    iter_article_records,
    load_structured_json_records,
    missing_numeric_articles,
    ocr_pages,
)


class TreatyChunkingTests(unittest.TestCase):
    def test_german_treaty_exposes_article_7_from_cached_official_pdf(self) -> None:
        source = next(
            item
            for item in CORE_TREATY_SOURCES
            if item.slug == "niemcy" and item.variant == "umowa"
        )
        articles = {item["article"]: item for item in iter_article_records(source, ocr_pages(source))}

        self.assertTrue({"7", "11", "12"}.issubset(articles))
        self.assertIn("Zyski", articles["7"]["text"])
        self.assertTrue(articles["7"]["pages"])

    def test_german_article_11_override_restores_the_verified_treaty_rate(self) -> None:
        source = next(
            item
            for item in CORE_TREATY_SOURCES
            if item.slug == "niemcy" and item.variant == "umowa"
        )
        records, manifest = build_outputs([source])
        article_11 = next(record for record in records if record["legal_provisions"] == ["art. 11"])

        self.assertIn("5 procent kwoty brutto tych odsetek", article_11["content_text"])
        self.assertEqual([10, 11], article_11["source_pages"])
        self.assertEqual(source.source_url, article_11["source_url"])
        self.assertIn("verified_article_overrides", manifest[0]["extraction_method"])

    def test_multi_column_order_does_not_relabel_headings(self) -> None:
        source = next(item for item in CORE_TREATY_SOURCES if item.slug == "niemcy")
        pages = [{"number": 1, "text": "Artykut8\nArtykut7\nArtykut9", "raw_chars": 27}]

        self.assertEqual(
            ["8", "7", "9"],
            [item["article"] for item in iter_article_records(source, pages)],
        )

    def test_canonical_transcriptions_are_complete_and_keep_official_citation(self) -> None:
        expected = {
            ("hiszpania", "umowa"): 30,
            ("szwajcaria", "umowa"): 28,
            ("usa", "umowa_1974"): 26,
            ("wielka_brytania", "umowa"): 29,
        }
        for (slug, variant), count in expected.items():
            source = next(item for item in CORE_TREATY_SOURCES if item.slug == slug and item.variant == variant)
            records = load_structured_json_records(source)
            self.assertEqual(count, len(records))
            self.assertEqual(source.source_url, records[0]["source_url"])
            self.assertEqual(f"art. {count}", records[-1]["legal_provisions"][0])

    def test_swiss_corrected_article_15_is_not_lost_as_duplicate_article_14(self) -> None:
        source = next(
            item for item in CORE_TREATY_SOURCES if item.slug == "szwajcaria" and item.variant == "umowa"
        )
        records = load_structured_json_records(source)
        article_15 = next(record for record in records if record["legal_provisions"] == ["art. 15"])
        self.assertIn("Praca najemna", article_15["content_text"])
        self.assertIn("niniejszego artykułu, wynagrodzenie", article_15["content_text"])

    def test_expected_terminal_article_is_a_completeness_requirement(self) -> None:
        source = next(item for item in CORE_TREATY_SOURCES if item.slug == "usa" and item.variant == "umowa_1974")
        articles = [{"article": str(number)} for number in range(1, 26)]
        self.assertEqual([26], missing_numeric_articles(source, articles))
