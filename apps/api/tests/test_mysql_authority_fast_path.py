from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.mysql_rag import (
    authority_citation_targets,
    authority_metadata_citation_patterns,
    fetch_candidate_rows_mysql,
    fetch_statute_rows_by_targets_mysql,
    select_wht_primary_bundle,
    statute_target_metadata_patterns,
)
from app.rag import (
    LegalRetrievalAxis,
    RagChunk,
    build_legal_source_plan,
    chunk_has_substantive_axis_preferred_target,
    extract_normalized_provision_references,
    filter_treaty_country_chunks,
    query_is_direct_statute_lookup,
    search_primary_law_chunks,
    treaty_direct_subject_prefix,
)


def _row(*, chunk_id: str = "683317:42", document_id: str = "683317") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "tax_domain": "PIT",
    }


class MysqlAuthorityFastPathTests(unittest.TestCase):
    def test_polish_german_inflection_routes_to_treaty_primary_law(self) -> None:
        plan = build_legal_source_plan(
            "Polska spółka wypłaca odsetki niemieckiej GmbH; "
            "oceń umowę polsko-niemiecką oraz podatek u źródła."
        )

        self.assertIn(("CIT", "11"), plan.statute_targets)
        self.assertTrue(
            any(axis.axis_id == "poland_germany_treaty" for axis in plan.axes)
        )

    def test_every_supported_treaty_country_uses_its_own_direct_lane(self) -> None:
        query = "Polska spółka płaci odsetki do austriackiej spółki; UPO i podatek u źródła."
        plan = build_legal_source_plan(query)

        self.assertEqual("UPO Polska - Austria", treaty_direct_subject_prefix(query))
        self.assertTrue(query_is_direct_statute_lookup(query))
        self.assertTrue(any(axis.axis_id == "poland_austria_treaty" for axis in plan.axes))

        chunks = [
            RagChunk(
                chunk_id="at:11", document_id="at", chunk_index=0, score=1.0,
                chunk_text="Art. 11 Odsetki", subject="UPO Polska - Austria - art. 11",
                signature=None, published_date=None, source_url=None, category=None,
                source_type="statute", source_subtype="tax_treaty", legal_provisions=["art. 11"],
            ),
            RagChunk(
                chunk_id="de:11", document_id="de", chunk_index=0, score=1.0,
                chunk_text="Art. 11 Odsetki", subject="UPO Polska - Niemcy - art. 11",
                signature=None, published_date=None, source_url=None, category=None,
                source_type="statute", source_subtype="tax_treaty", legal_provisions=["art. 11"],
            ),
        ]
        self.assertEqual(["at:11"], [chunk.chunk_id for chunk in filter_treaty_country_chunks(chunks, query)])

    def test_exact_treaty_lane_does_not_wait_for_fulltext_supplement(self) -> None:
        query = "Polska spółka płaci odsetki niemieckiej GmbH; UPO i WHT."
        chunk = RagChunk(
            chunk_id="de:11", document_id="de", chunk_index=0, score=200.0,
            chunk_text="Art. 11 Odsetki", subject="UPO Polska - Niemcy - art. 11",
            signature=None, published_date=None, source_url=None, category=None,
            source_type="statute", source_subtype="tax_treaty", legal_provisions=["art. 11"],
        )
        with patch("app.rag.retrieve_deterministic_statute_chunks", return_value=[chunk]), patch(
            "app.mysql_rag.search_chunks_mysql", side_effect=AssertionError("FULLTEXT must not run")
        ):
            self.assertEqual([chunk], search_primary_law_chunks(query, limit=4))

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

    def test_statute_lookup_uses_substantive_chunks_and_both_metadata_forms(self) -> None:
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch("app.mysql_rag.mysql_connection", return_value=nullcontext(connection)), patch(
            "app.mysql_rag.get_mysql_target", return_value=("documents", "chunks")
        ):
            rows = fetch_statute_rows_by_targets_mysql([("CIT", "26")])

        self.assertEqual(rows, [])
        self.assertIn('%"art. 26"%', statute_target_metadata_patterns("26"))
        self.assertIn("%art. 26-%", statute_target_metadata_patterns("26"))
        statement = next(
            str(call.args[0])
            for call in cursor.execute.call_args_list
            if "FROM `chunks`" in str(call.args[0])
        )
        self.assertIn("CHAR_LENGTH(TRIM(c.chunk_text)) >= 40", statement)
        self.assertIn("COALESCE(d.source_subtype, '') <> 'tax_treaty'", statement)
        self.assertNotIn("c.chunk_index = 0", statement)

    def test_heading_does_not_count_as_substantive_primary_law(self) -> None:
        axis = LegalRetrievalAxis(
            axis_id="wht_interest",
            label="WHT",
            query="WHT",
            source_types={"statute"},
            tax_domains={"CIT"},
            preferred_targets=(("CIT", "21"),),
        )
        common = {
            "document_id": "cit-21",
            "chunk_index": 0,
            "score": 1.0,
            "subject": "O podatku dochodowym od osób prawnych - art. 21",
            "signature": None,
            "published_date": "2026-01-01",
            "source_url": None,
            "category": None,
            "source_type": "statute",
            "legal_provisions": ["art. 21"],
        }
        heading = RagChunk(chunk_id="cit-21:0", chunk_text="art. 21", **common)
        rule = RagChunk(
            chunk_id="cit-21:2",
            chunk_text=(
                "Art. 21 ust. 1 pkt 1: odsetki oraz należności licencyjne "
                "podlegają zryczałtowanemu podatkowi dochodowemu."
            ),
            **common,
        )

        self.assertFalse(chunk_has_substantive_axis_preferred_target(axis, heading))
        self.assertTrue(chunk_has_substantive_axis_preferred_target(axis, rule))

    def test_wht_bundle_keeps_separate_controlling_statute_units(self) -> None:
        def chunk(chunk_id: str, subject: str, text: str) -> RagChunk:
            return RagChunk(
                chunk_id=chunk_id,
                document_id=chunk_id.split(":")[0],
                chunk_index=0,
                chunk_text=text,
                score=1.0,
                subject=subject,
                signature=None,
                published_date="2026-01-01",
                source_url=None,
                category=None,
                source_type="statute",
                legal_provisions=[],
            )

        cit = "O podatku dochodowym od osób prawnych - art. "
        vat = "O podatku od towarów i usług - art. "
        chunks = [
            chunk("cit21-1", f"{cit}21", "art. 21 ust. 1 pkt 1 Odsetki i należności licencyjne."),
            chunk("cit21-2a", f"{cit}21", "art. 21 ust. 1 pkt 2a Usługi zarządzania i kontroli."),
            chunk("cit26-2e", f"{cit}26", "art. 26 ust. 2e Próg 2 000 000 zł."),
            chunk("cit26-7a", f"{cit}26", "art. 26 ust. 7a Art. 26. 7a. Przepisu ust. 2e nie stosuje się."),
            chunk("cit26b", f"{cit}26b", "art. 26b ust. 1 Opinia o stosowaniu preferencji."),
            chunk("cit28b", f"{cit}28b", "art. 28b ust. 1 Zwrot podatku pobranego zgodnie z art. 26 ust. 2e."),
            chunk("upo11", "UPO Polska - Niemcy - art. 11", "Art. 11 Odsetki."),
            chunk("upo12", "UPO Polska - Niemcy - art. 12", "Art. 12 Należności licencyjne."),
            chunk("upo7", "UPO Polska - Niemcy - art. 7", "Art. 7 Zyski przedsiębiorstw."),
            chunk("vat28b", f"{vat}28b", "art. 28b ust. 1 Miejsce świadczenia usług."),
            chunk("vat17", f"{vat}17", "art. 17 ust. 1 pkt 4 Import usług."),
            chunk("vat43", f"{vat}43", "art. 43 ust. 1 pkt 38 Usługi udzielania pożyczek."),
        ]
        query = (
            "Niemiecka GmbH otrzymuje odsetki, licencje i usługi zarządzania; "
            "WHT, 2 000 000 zł, UPO polsko-niemiecka i VAT."
        )

        selected = select_wht_primary_bundle(chunks, query)

        self.assertEqual(
            [item.chunk_id for item in selected[:12]],
            [
                "cit21-1", "cit21-2a", "cit26-2e", "cit26-7a", "cit26b", "cit28b",
                "upo11", "upo12", "upo7", "vat28b", "vat17", "vat43",
            ],
        )


if __name__ == "__main__":
    unittest.main()
