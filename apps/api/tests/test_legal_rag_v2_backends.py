from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.legal_rag_v2.backends import CorpusFtsBackend


class CorpusFtsBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_mysql_treaty_query_uses_country_prefix_before_article_lookup(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        backend = CorpusFtsBackend(backend="mysql")

        with patch("app.mysql_rag.get_mysql_target", return_value=("documents", "chunks")), patch(
            "app.mysql_rag.mysql_connection", return_value=nullcontext(connection)
        ):
            rows = backend._search_mysql(
                "UPO Polska Niemcy art. 11 odsetki",
                8,
                frozenset({"tax_treaty"}),
                {"tax_domains": ["CIT"]},
            )

        self.assertEqual([], rows)
        statement = str(cursor.execute.call_args.args[0])
        params = cursor.execute.call_args.args[1]
        self.assertIn("LOWER(d.subject) LIKE LOWER(%s)", statement)
        self.assertNotIn("MATCH(c.search_text", statement)
        self.assertEqual("UPO Polska - Niemcy%", params[0])
        self.assertEqual("art. 11%", params[1])

    async def test_sqlite_search_is_typed_filtered_and_policy_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rag.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY, subject TEXT, signature TEXT,
                    published_date TEXT, source_url TEXT, category TEXT,
                    tax_domain TEXT, source TEXT, source_type TEXT,
                    source_subtype TEXT, authority TEXT, act_title TEXT, publication TEXT,
                    legal_state_date TEXT, legal_provisions_json TEXT,
                    source_pages_json TEXT
                );
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT, chunk_index INTEGER,
                    chunk_text TEXT, display_reference TEXT DEFAULT '', provision_id TEXT DEFAULT ''
                );
                CREATE TABLE chunk_citations (
                    chunk_id TEXT, citation TEXT,
                    PRIMARY KEY (chunk_id, citation)
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_text, subject, signature, keywords, legal_provisions,
                    issues, question_text, facts_text, tax_domain
                );
                """
            )
            connection.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "law-a", "Ustawa PIT", "", "2025-01-01", "", "", "PIT",
                    "eli", "statute", "consolidated_text", "Sejm", "Ustawa PIT", "Dz.U.",
                    "2025-01-01", '["art. 21"]', "[]",
                ),
            )
            connection.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                (
                    "chunk-a", "law-a", 0,
                    "art. 21 ust. 30a\nUlga mieszkaniowa obejmuje określone wydatki.",
                    "art. 21 ust. 30a", "pit-21-30a",
                ),
            )
            connection.execute(
                "INSERT INTO chunk_citations VALUES (?,?)",
                ("chunk-a", "art. 21 ust. 30a"),
            )
            connection.execute(
                "INSERT INTO chunks_fts(rowid, chunk_text, subject, signature, keywords, legal_provisions, issues, question_text, facts_text, tax_domain) VALUES (1,?,?,?,?,?,?,?,?,?)",
                (
                    "art. 21 ust. 30a Ulga mieszkaniowa obejmuje określone wydatki.", "Ustawa PIT", "",
                    "", "art. 21", "ulga", "", "", "PIT",
                ),
            )
            connection.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "upo-de-11", "UPO Polska - Niemcy - art. 11", "", "2025-01-01", "", "", "TAX_TREATY",
                    "mf", "statute", "tax_treaty", "MF", "UPO Polska-Niemcy", "Dz.U.",
                    "2025-01-01", '["art. 11"]', "[]",
                ),
            )
            connection.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                ("upo-de-11:1", "upo-de-11", 0, "Art. 11 Odsetki: stawka 5%.", "art. 11", "upo-de-11"),
            )
            connection.execute("INSERT INTO chunk_citations VALUES (?,?)", ("upo-de-11:1", "art. 11"))
            connection.execute(
                "INSERT INTO chunks_fts(rowid, chunk_text, subject, signature, keywords, legal_provisions, issues, question_text, facts_text, tax_domain) VALUES (2,?,?,?,?,?,?,?,?,?)",
                ("Art. 11 Odsetki stawka 5%", "UPO Polska Niemcy art. 11", "", "", "art. 11", "upo", "", "", "TAX_TREATY"),
            )
            connection.commit()
            connection.close()

            backend = CorpusFtsBackend(backend="sqlite", sqlite_path=path)
            hits = await backend.search(
                "wydatki na ulgę mieszkaniową",
                limit=5,
                source_types=frozenset({"statute"}),
                metadata_filters={"tax_domains": ["PIT"]},
            )

            self.assertEqual([item.document_id for item in hits], ["law-a"])
            self.assertEqual(hits[0].source_type, "statute")
            self.assertEqual(hits[0].backend, "legal_rag_v2_generic_fts")
            self.assertEqual(hits[0].metadata["legal_provisions"], ["art. 21 ust. 30a"])

            exact = await backend.search(
                "art. 21 ust. 30a",
                limit=5,
                source_types=frozenset({"statute"}),
                metadata_filters={"tax_domains": ["PIT"]},
            )
            self.assertEqual(exact[0].metadata["provision_id"], "pit-21-30a")
            self.assertEqual(exact[0].text.splitlines()[0], "art. 21 ust. 30a")

            treaty = await backend.search(
                "UPO Polska Niemcy art. 11 odsetki",
                limit=5,
                source_types=frozenset({"tax_treaty"}),
                metadata_filters={"tax_domains": ["CIT"]},
            )
            self.assertEqual([item.document_id for item in treaty], ["upo-de-11"])
            self.assertEqual(treaty[0].source_type, "tax_treaty")


if __name__ == "__main__":
    unittest.main()
