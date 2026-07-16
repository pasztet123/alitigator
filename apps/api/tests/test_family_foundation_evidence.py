from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.law_chunk import build_provision_units
from app.legal_rag_v2.backends import CorpusFtsBackend
from app.legal_rag_v2.family_foundation import enrich_family_foundation_plan
from app.legal_rag_v2.transfer_pricing import enrich_transfer_pricing_plan
from app.legal_rag_v2.schemas import (
    Clarification,
    LegalIssue,
    LegalResearchPlan,
    QueryFamily,
    ResearchIntent,
)
from app.rag import (
    decompose_query_into_legal_axes,
    derive_tax_domain,
    get_connection,
    get_rag_config,
    index_record,
)


UFR_PATH = Path("apps/api/data/laws/processed/family_foundation_primary_bundle.jsonl")


def plan() -> LegalResearchPlan:
    issue_ids = (
        "family_foundation_allowed_activity_catalog",
        "family_foundation_cit_hidden_profit",
        "family_foundation_disallowed_income_25_percent",
        "family_foundation_beneficiary_pit",
        "family_foundation_vat_related_party",
    )
    return LegalResearchPlan(
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id=issue_id,
                label=issue_id,
                tax_domains=[],
                legal_mechanism="family_foundation",
                query_families=[
                    QueryFamily(
                        family="natural_language",
                        query="fundacja rodzinna",
                        lane="both",
                        origin="model",
                    )
                ],
            )
            for issue_id in issue_ids
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


class FamilyFoundationEvidenceTests(unittest.TestCase):
    def test_ufr_corpus_contains_exact_current_editorial_units(self) -> None:
        records = [json.loads(line) for line in UFR_PATH.read_text(encoding="utf-8").splitlines()]
        article_numbers = {
            int(record["legal_provisions"][0].removeprefix("art. "))
            for record in records
        }
        self.assertEqual(set(range(1, 146)), article_numbers)
        self.assertTrue(all(record["source_type"] == "statute" for record in records))
        self.assertTrue(all(record["source_subtype"] == "consolidated_text" for record in records))
        self.assertTrue(all(derive_tax_domain(record) == "UFR" for record in records))

        article_five = next(
            record for record in records if record["legal_provisions"] == ["art. 5"]
        )
        citations = [unit["citation"] for unit in article_five["provision_units"]]
        self.assertEqual(len(citations), len(set(citations)))
        self.assertIn("art. 5 ust. 1 pkt 5 lit. a", citations)
        self.assertIn("art. 5 ust. 3", citations)
        self.assertFalse(
            any("Zmiany tekstu jednolitego" in unit["text"] for unit in article_five["provision_units"])
        )

    def test_family_plan_gets_issue_scoped_exact_primary_targets(self) -> None:
        enriched = enrich_family_foundation_plan(plan(), "Kazus fundacji rodzinnej i UFR")
        by_id = {issue.issue_id: issue for issue in enriched.issues}

        allowed = by_id["family_foundation_allowed_activity_catalog"]
        self.assertEqual(["UFR"], allowed.tax_domains)
        self.assertIn("UFR art. 5", [query.query for query in allowed.query_families])

        hidden = by_id["family_foundation_cit_hidden_profit"]
        self.assertEqual({"CIT", "UFR"}, set(hidden.tax_domains))
        hidden_queries = {query.query for query in hidden.query_families}
        self.assertIn("CIT art. 24q ust. 1a pkt 6", hidden_queries)
        self.assertIn("UFR art. 2 ust. 2", hidden_queries)

        beneficiary = by_id["family_foundation_beneficiary_pit"]
        beneficiary_queries = {query.query for query in beneficiary.query_families}
        self.assertIn("PIT art. 21 ust. 1 pkt 157", beneficiary_queries)
        self.assertIn("UFR art. 29 ust. 1", beneficiary_queries)

    def test_complex_case_is_split_by_transaction_not_five_generic_buckets(self) -> None:
        question = (
            "Fundacja rodzinna otrzymała dywidendę i odsetki od obligacji. "
            "Wynajmuje magazyn spółce fundatora, kupuje usługi doradcze od "
            "podmiotu powiązanego, a fundator udzielił fundacji pożyczki. "
            "Fundacja udzieliła beneficjentowi pożyczki na 12 lat i wypłaciła "
            "mu świadczenie. Kupiła mieszkania częściowo w celu odsprzedaży, "
            "ponosi koszty wspólne, pyta o odliczenie podatku, deklaracje, "
            "terminy oraz VAT od najmu i sprzedaży."
        )
        enriched = enrich_family_foundation_plan(plan(), question)
        by_id = {issue.issue_id: issue for issue in enriched.issues}

        expected = {
            "family_foundation_investment_income",
            "family_foundation_related_party_rent",
            "family_foundation_related_party_services",
            "family_foundation_borrowing_from_related_party",
            "family_foundation_beneficiary_loan",
            "family_foundation_beneficiary_benefit",
            "family_foundation_real_estate_activity",
            "family_foundation_common_costs",
            "family_foundation_tax_credit_and_reporting",
            "family_foundation_vat_transactions",
        }
        self.assertTrue(expected.issubset(by_id))
        generic_ids = {
            "family_foundation_allowed_activity_catalog",
            "family_foundation_cit_hidden_profit",
            "family_foundation_disallowed_income_25_percent",
            "family_foundation_beneficiary_pit",
            "family_foundation_vat_related_party",
        }
        self.assertFalse(generic_ids.intersection(by_id))

        rent_queries = {item.query for item in by_id["family_foundation_related_party_rent"].query_families}
        self.assertIn("CIT art. 6 ust. 8", rent_queries)
        self.assertIn("CIT art. 24q ust. 8", rent_queries)
        benefit_queries = {item.query for item in by_id["family_foundation_beneficiary_benefit"].query_families}
        self.assertIn("PIT art. 21 ust. 49", benefit_queries)
        vat_queries = {item.query for item in by_id["family_foundation_vat_transactions"].query_families}
        self.assertIn("VAT art. 90 ust. 2", vat_queries)

    def test_domestic_family_plan_drops_legacy_wht_noise(self) -> None:
        base = plan()
        wht_issue = LegalIssue(
            issue_id="wht_interest",
            label="WHT: odsetki i należności bierne",
            tax_domains=["CIT"],
            legal_mechanism="wht",
        )
        noisy = base.model_copy(update={"issues": [*base.issues, wht_issue]})

        domestic = enrich_family_foundation_plan(
            noisy,
            "Fundacja rodzinna udziela beneficjentowi krajowej pożyczki.",
        )
        cross_border = enrich_family_foundation_plan(
            noisy,
            "Fundacja rodzinna płaci odsetki nierezydentowi i analizuje WHT.",
        )

        self.assertNotIn("wht_interest", {item.issue_id for item in domestic.issues})
        self.assertIn("wht_interest", {item.issue_id for item in cross_border.issues})

    def test_transfer_pricing_gets_its_own_axis_instead_of_wht(self) -> None:
        axes = decompose_query_into_legal_axes(
            "Fundacja rodzinna pyta o zwolnienie dokumentacyjne z art. 11n CIT."
        )
        by_id = {axis.axis_id: axis for axis in axes}

        self.assertIn("transfer_pricing_documentation", by_id)
        self.assertNotIn("wht_interest", by_id)
        self.assertIn(("CIT", "11n"), by_id["transfer_pricing_documentation"].preferred_targets)

    def test_model_plan_is_augmented_with_transfer_pricing_issue_and_exact_law(self) -> None:
        question = "Czy transakcja kontrolowana korzysta ze zwolnienia dokumentacyjnego z art. 11n?"
        enriched = enrich_transfer_pricing_plan(plan(), question)
        issue = next(
            item for item in enriched.issues if item.issue_id == "transfer_pricing_documentation"
        )
        queries = {item.query for item in issue.query_families}

        self.assertIn("CIT art. 11k ust. 1", queries)
        self.assertIn("CIT art. 11n pkt 1", queries)
        self.assertIn("CIT art. 11t ust. 1", queries)

    def test_numbered_source_note_is_not_a_provision_unit(self) -> None:
        text = """Art. 5. 1. Reguła główna:
1) prawidłowy punkt;
2) Zmiany tekstu jednolitego wymienionej ustawy ogłoszono później;
2) drugi prawidłowy punkt.
"""
        units = build_provision_units(
            text,
            article_document_id="ufr-art-5",
            record_document_id="ufr-art-5",
        )
        self.assertFalse(any("Zmiany tekstu jednolitego" in unit["text"] for unit in units))
        self.assertTrue(any("drugi prawidłowy punkt" in unit["text"] for unit in units))

    def test_wrapped_inline_article_reference_does_not_reset_ancestry(self) -> None:
        units = build_provision_units(
            """Art. 32. 1. Reguła podstawowa.
1) pierwszy przypadek odwołujący się do
art. 92 ust. 3 innej ustawy.
2. Przez powiązania rozumie się określone relacje.
""",
            article_document_id="vat-art-32",
            record_document_id="vat-art-32",
        )
        citations = {unit["citation"] for unit in units}

        self.assertIn("art. 32 ust. 2", citations)
        self.assertFalse(any(citation.startswith("art. 92") for citation in citations))

    def test_point_can_be_a_direct_child_of_article(self) -> None:
        units = build_provision_units(
            """Art. 11n. Obowiązek nie ma zastosowania do transakcji:
1) zawieranych wyłącznie przez podmioty krajowe;
2) objętych innym wyjątkiem.
""",
            article_document_id="cit-art-11n",
            record_document_id="cit-art-11n",
        )
        citations = {unit["citation"] for unit in units}

        self.assertIn("art. 11n pkt 1", citations)
        self.assertIn("art. 11n pkt 2", citations)


class FamilyFoundationRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_ufr_domain_filter_cannot_return_vat_article_five(self) -> None:
        ufr_records = [json.loads(line) for line in UFR_PATH.read_text(encoding="utf-8").splitlines()]
        ufr_article_five = next(
            record for record in ufr_records if record["legal_provisions"] == ["art. 5"]
        )
        vat_path = Path(
            "apps/api/data/laws/processed/vat_act_DU_2025_775_codified_2026-05-05.jsonl"
        )
        vat_article_five = next(
            json.loads(line)
            for line in vat_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("legal_provisions") == ["art. 5"]
        )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "rag.sqlite3"
            config = replace(get_rag_config(), db_path=database)
            connection = get_connection(database)
            try:
                index_record(connection, ufr_article_five, config)
                index_record(connection, vat_article_five, config)
                connection.commit()
            finally:
                connection.close()

            backend = CorpusFtsBackend(backend="sqlite", sqlite_path=database)
            candidates = await backend.search(
                "UFR art. 5",
                limit=10,
                source_types=frozenset({"statute"}),
                metadata_filters={"tax_domains": ["UFR"]},
            )

        self.assertTrue(candidates)
        self.assertTrue(
            all("fundacji-rodzinnej" in candidate.document_id for candidate in candidates)
        )


if __name__ == "__main__":
    unittest.main()
