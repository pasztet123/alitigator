from __future__ import annotations

import unittest

from app.legal_rag_v2.schemas import (
    Clarification,
    DocumentSourceSpan,
    EvidenceBundle,
    LegalClaim,
    LegalIssue,
    LegalResearchPlan,
    ProvisionReference,
    ResearchIntent,
)
from app.legal_rag_v2.pipeline import validate_claims
from app.legal_rag_v2.retrieval import LegalRetriever, RetrievalCandidate, RetrievalConfig
from app.legal_rag_v2.wht import (
    WhtPayAndRefundCalculationEngine,
    enrich_crossborder_wht_plan,
)


QUESTION = (
    "Polska spółka płaci niemieckiej GmbH odsetki, licencje software i usługi zarządzania. "
    "Łącznie płatności wynoszą 2 200 000 zł. Oceń WHT, pay and refund, UPO polsko-niemiecką i VAT."
)


def plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query=QUESTION,
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="cit_core",
                label="CIT krajowy",
                tax_domains=["CIT"],
                legal_mechanism="withholding_tax",
            )
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


class CrossborderWhtEnrichmentTests(unittest.TestCase):
    def test_inflected_polish_wht_wording_starts_the_crossborder_bundle(self) -> None:
        question = (
            "Polska spółka wypłaca niemieckiej GmbH odsetki, licencje i usługi "
            "zarządzania. Które płatności podlegają podatkowi u źródła?"
        )
        enriched = enrich_crossborder_wht_plan(plan(), question)
        issue_ids = {item.issue_id for item in enriched.issues}
        self.assertIn("wht_interest_pl_de_treaty", issue_ids)
        self.assertIn("wht_royalties_pl_de_treaty", issue_ids)
        self.assertIn("wht_services_pl_de_business_profits", issue_ids)
        self.assertNotIn("wht_interest", issue_ids)
        self.assertNotIn("poland_germany_treaty", issue_ids)

    def test_enrichment_creates_separate_treaty_vat_and_procedural_issues(self) -> None:
        enriched = enrich_crossborder_wht_plan(plan(), QUESTION)
        issues = {item.issue_id: item for item in enriched.issues}

        self.assertTrue(enriched.intent.needs_calculations)
        self.assertTrue({
            "wht_interest_pl_de_treaty",
            "wht_royalties_pl_de_treaty",
            "wht_services_pl_de_business_profits",
            "wht_pay_and_refund_procedure",
            "vat_interest_financial_service",
            "vat_royalty_crossborder_service",
            "vat_management_crossborder_service",
        }.issubset(issues))
        self.assertIn(
            "UPO Polska Niemcy art. 11 odsetki stawka beneficial owner",
            [item.query for item in issues["wht_interest_pl_de_treaty"].query_families],
        )
        self.assertIn(
            "UPO Polska Niemcy art. 12 należności licencyjne stawka",
            [item.query for item in issues["wht_royalties_pl_de_treaty"].query_families],
        )
        self.assertIn(
            "UPO Polska Niemcy art. 7 zyski przedsiębiorstw zakład usługi zarządzania",
            [item.query for item in issues["wht_services_pl_de_business_profits"].query_families],
        )
        procedure_queries = [item.query for item in issues["wht_pay_and_refund_procedure"].query_families]
        self.assertTrue(any("art. 26 ust. 2g" in item for item in procedure_queries))
        self.assertTrue(any("art. 26 ust. 7a" in item for item in procedure_queries))
        self.assertTrue(any("art. 26 ust. 7b" in item for item in procedure_queries))
        self.assertTrue(any("art. 26 ust. 7c" in item for item in procedure_queries))
        self.assertTrue(any("art. 28b" in item for item in procedure_queries))
        interest_authority_queries = [
            item.query for item in issues["wht_interest_pl_de_treaty"].query_families
        ]
        self.assertTrue(any("MF beneficial owner" in item for item in interest_authority_queries))

    def test_pay_and_refund_calculation_is_bound_to_art_26_2e(self) -> None:
        enriched = enrich_crossborder_wht_plan(plan(), QUESTION)
        art_26_2e = ProvisionReference(
            provision_id="cit_art_26_ust_2e",
            document_id="cit_act",
            citation="art. 26 ust. 2e",
            article="26",
            paragraph="2e",
            status="active",
        )
        bundle = EvidenceBundle(
            issue_id="wht_pay_and_refund_procedure",
            controlling_provisions=[art_26_2e],
            coverage_status="partial",
            controlling_provision_present=True,
            retrieval_confidence=0.7,
            dependency_coverage=0.5,
            exception_coverage=0.5,
            temporal_validation_passed=True,
        )

        calculations = WhtPayAndRefundCalculationEngine().calculate(enriched, [bundle])

        self.assertEqual(1, len(calculations))
        calculation = calculations[0]
        self.assertEqual(2_200_000, calculation.inputs["aggregate_payments"])
        self.assertEqual(2_000_000, calculation.inputs["threshold_base"])
        self.assertEqual(200_000, calculation.inputs["excess"])
        self.assertEqual(40_000, calculation.result)
        self.assertEqual(["cit_art_26_ust_2e"], [item.provision_id for item in calculation.legal_basis])

    def test_calculation_is_not_created_without_art_26_2e(self) -> None:
        enriched = enrich_crossborder_wht_plan(plan(), QUESTION)
        bundle = EvidenceBundle(issue_id="wht_pay_and_refund_procedure")
        self.assertEqual([], WhtPayAndRefundCalculationEngine().calculate(enriched, [bundle]))

    def test_incomplete_procedural_bundle_blocks_only_its_own_claims(self) -> None:
        enriched = enrich_crossborder_wht_plan(plan(), QUESTION)
        art_26_2e = ProvisionReference(
            provision_id="cit_art_26_ust_2e",
            document_id="cit_act",
            citation="art. 26 ust. 2e",
            status="active",
        )
        cit_core = ProvisionReference(
            provision_id="cit_art_21_ust_1_pkt_1",
            document_id="cit_act",
            citation="art. 21 ust. 1 pkt 1",
            status="active",
        )
        procedure_bundle = EvidenceBundle(
            issue_id="wht_pay_and_refund_procedure",
            controlling_provisions=[art_26_2e],
            missing_sources=["required_primary:art_28b", "required_primary:art_26_7c"],
            coverage_status="partial",
            controlling_provision_present=True,
            retrieval_confidence=0.5,
            dependency_coverage=0.29,
            exception_coverage=0.5,
            temporal_validation_passed=True,
        )
        cit_bundle = EvidenceBundle(
            issue_id="cit_core",
            controlling_provisions=[cit_core],
            coverage_status="complete",
            controlling_provision_present=True,
            retrieval_confidence=0.7,
            dependency_coverage=0.5,
            exception_coverage=0.5,
            temporal_validation_passed=True,
        )
        claims = [
            LegalClaim(
                claim_id="cit_approved",
                issue_id="cit_core",
                claim_type="normative_rule",
                text="Płatność jest objęta krajową regulacją CIT.",
                result="CIT pozostaje zatwierdzony.",
                status="approved",
                controlling_provision_ids=[cit_core.provision_id],
                source_spans=[DocumentSourceSpan(start=0, end=5, document_id="cit_act")],
                confidence=0.8,
            ),
            LegalClaim(
                claim_id="refund_without_28b",
                issue_id="wht_pay_and_refund_procedure",
                claim_type="normative_rule",
                text="Zwrot podatku jest możliwy.",
                result="Wniosek o zwrocie.",
                status="approved",
                controlling_provision_ids=[art_26_2e.provision_id],
                source_spans=[DocumentSourceSpan(start=0, end=5, document_id="cit_act")],
                confidence=0.8,
            ),
        ]

        validated, errors, warnings = validate_claims(
            claims,
            plan=enriched,
            bundles=[cit_bundle, procedure_bundle],
            calculations=[],
        )

        self.assertEqual([], errors)
        self.assertEqual("approved", validated[0].status)
        self.assertEqual("blocked_incomplete_dependency_bundle", validated[1].status)
        self.assertIn("refund_without_28b:blocked_for_incomplete_issue_bundle", warnings)


class CrossborderWhtRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_payment_has_independent_treaty_vat_and_authority_lane(self) -> None:
        class Backend:
            trace_marker = "wht_benchmark_backend"

            async def search(self, query, *, limit, source_types, metadata_filters):
                lowered = query.casefold()
                if "guidance" in source_types:
                    return [
                        RetrievalCandidate(
                            candidate_id="mf-bo-guidance",
                            document_id="mf-bo-guidance",
                            chunk_id="mf-bo-guidance",
                            text="MF beneficial owner WHT odsetki.",
                            source_type="guidance",
                            metadata={"tax_domains": ["CIT"], "legal_state_date": "2026-01-01"},
                        )
                    ]
                if "art. 11" in lowered:
                    return [self.candidate("upo-11", "tax_treaty", "Art. 11 Odsetki: stawka 5%.")]
                if "art. 12" in lowered:
                    return [self.candidate("upo-12", "tax_treaty", "Art. 12 Należności licencyjne: stawka 5%.")]
                if "art. 7" in lowered:
                    return [self.candidate("upo-7", "tax_treaty", "Art. 7 Zyski przedsiębiorstw, chyba że zakład.")]
                if "art. 43" in lowered:
                    return [self.candidate("vat-43", "statute", "Art. 43 ust. 1 pkt 38 Zwolnienie usług finansowych.")]
                if "art. 28b" in lowered:
                    return [self.candidate("vat-28b", "statute", "Art. 28b ust. 1 Miejsce świadczenia usług.")]
                if "art. 17" in lowered:
                    return [self.candidate("vat-17", "statute", "Art. 17 ust. 1 pkt 4 Import usług.")]
                if "art. 26" in lowered:
                    return [self.candidate("cit-26", "statute", "Art. 26 ust. 2e, ust. 2g, ust. 7a-7c, art. 26b i art. 28b.")]
                return [self.candidate("cit-21", "statute", "Art. 21 ust. 1 pkt 1 i pkt 2a.")]

            @staticmethod
            def candidate(candidate_id, source_type, text):
                return RetrievalCandidate(
                    candidate_id=candidate_id,
                    document_id=candidate_id,
                    chunk_id=candidate_id,
                    text=text,
                    source_type=source_type,
                    metadata={"tax_domains": ["CIT", "VAT"], "legal_state_date": "2026-01-01"},
                )

        retrieval = await LegalRetriever(
            Backend(), config=RetrievalConfig(selected_limit_per_issue=12)
        ).retrieve(enrich_crossborder_wht_plan(plan(), QUESTION))
        primary = {item.issue_id: item for item in retrieval.primary_law}
        authorities = {item.issue_id: item for item in retrieval.authorities}

        self.assertIn("upo-11", [item.document_id for item in primary["wht_interest_pl_de_treaty"].candidates])
        self.assertIn("upo-12", [item.document_id for item in primary["wht_royalties_pl_de_treaty"].candidates])
        self.assertIn("upo-7", [item.document_id for item in primary["wht_services_pl_de_business_profits"].candidates])
        vat_candidate_ids = {
            item.document_id
            for issue_id in ("vat_interest_financial_service", "vat_royalty_crossborder_service", "vat_management_crossborder_service")
            for item in primary[issue_id].candidates
        }
        self.assertTrue({"vat-28b", "vat-17", "vat-43"}.issubset(vat_candidate_ids))
        procedure_text = " ".join(item.text for item in primary["wht_pay_and_refund_procedure"].candidates)
        self.assertIn("ust. 2e", procedure_text)
        self.assertIn("ust. 2g", procedure_text)
        self.assertIn("ust. 7a-7c", procedure_text)
        self.assertIn("art. 28b", procedure_text)
        self.assertTrue(all(authorities[issue_id].candidates for issue_id in (
            "wht_interest_pl_de_treaty",
            "wht_royalties_pl_de_treaty",
            "wht_services_pl_de_business_profits",
            "wht_pay_and_refund_procedure",
        )))


if __name__ == "__main__":
    unittest.main()
