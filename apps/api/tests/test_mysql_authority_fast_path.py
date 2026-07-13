from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.mysql_rag import (
    authority_citation_targets,
    authority_metadata_citation_patterns,
    fetch_candidate_rows_mysql,
)
from app.rag import extract_normalized_provision_references


def _row(*, chunk_id: str = "683317:42", document_id: str = "683317") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "tax_domain": "PIT",
    }


class MysqlAuthorityFastPathTests(unittest.TestCase):
    def test_extracts_exact_citation_only_for_authority_scopes(self) -> None:
        query = "art. 21 ust. 30a ustawy PIT spłata kredytu"
        self.assertEqual(
            authority_citation_targets(query, source_types={"interpretation"}),
            ["art. 21 ust. 30a"],
        )
        self.assertEqual(authority_citation_targets(query, source_types={"statute"}), [])
        self.assertEqual(authority_citation_targets(query, source_types=None), [])
        self.assertIn(
            "%art. 21-ust. 30a%",
            authority_metadata_citation_patterns(["art. 21 ust. 30a"]),
        )
        self.assertEqual(
            extract_normalized_provision_references(
                "",
                ["[PIT] Ustawa o PIT-Rozdział 3-art. 21-ust. 1-pkt 131"],
            ),
            ["art. 21 ust. 1 pkt 131"],
        )

    def test_exact_citation_pool_skips_broad_fulltext(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = [_row()]
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        config = SimpleNamespace(
            candidate_pool_limit=120,
            retrieval_max_chunks_per_document=2,
            domain_filter_enabled=False,
        )

        with patch("app.mysql_rag.is_mysql_rag_configured", return_value=True), patch(
            "app.mysql_rag.get_rag_config", return_value=config
        ), patch("app.mysql_rag.ensure_search_schema_ready"), patch(
            "app.mysql_rag.mysql_connection", return_value=nullcontext(connection)
        ), patch(
            "app.mysql_rag.build_mysql_candidate_queries", return_value=["broad-one", "broad-two"]
        ):
            descriptor, rows = fetch_candidate_rows_mysql(
                "art. 21 ust. 30a ustawy PIT spłata kredytu",
                effective_limit=8,
                source_types={"interpretation"},
            )

        sql_calls = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(rows, [_row()])
        self.assertIn("citation:art. 21 ust. 30a", descriptor)
        self.assertTrue(any("_citations" in sql for sql in sql_calls))
        self.assertFalse(any("MATCH(c.search_text" in sql for sql in sql_calls))

    def test_empty_exact_pool_runs_only_one_bounded_fulltext_fallback(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.side_effect = [
            [],
            [],
            [_row(chunk_id="historical:3", document_id="historical")],
        ]
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        config = SimpleNamespace(
            candidate_pool_limit=120,
            retrieval_max_chunks_per_document=2,
            domain_filter_enabled=False,
        )

        with patch("app.mysql_rag.is_mysql_rag_configured", return_value=True), patch(
            "app.mysql_rag.get_rag_config", return_value=config
        ), patch("app.mysql_rag.ensure_search_schema_ready"), patch(
            "app.mysql_rag.mysql_connection", return_value=nullcontext(connection)
        ), patch(
            "app.mysql_rag.build_mysql_candidate_queries", return_value=["broad-one", "broad-two"]
        ):
            _, rows = fetch_candidate_rows_mysql(
                "art. 21 ust. 30a ustawy PIT spłata kredytu",
                effective_limit=6,
                source_types={"judgment"},
            )

        sql_calls = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(rows[0]["document_id"], "historical")
        self.assertEqual(sum("MATCH(c.search_text" in sql for sql in sql_calls), 1)


if __name__ == "__main__":
    unittest.main()
