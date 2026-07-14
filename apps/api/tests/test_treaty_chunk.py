from __future__ import annotations

import unittest

from app.treaty_chunk import CORE_TREATY_SOURCES, iter_article_records, ocr_pages


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

    def test_multi_column_order_does_not_relabel_headings(self) -> None:
        source = next(item for item in CORE_TREATY_SOURCES if item.slug == "niemcy")
        pages = [{"number": 1, "text": "Artykut8\nArtykut7\nArtykut9", "raw_chars": 27}]

        self.assertEqual(
            ["8", "7", "9"],
            [item["article"] for item in iter_article_records(source, pages)],
        )

