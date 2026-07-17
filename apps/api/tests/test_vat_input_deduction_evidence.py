from __future__ import annotations

import unittest

from app.legal_rag_v2.pipeline import (
    _claim_coverage_requirements,
    _required_issue_dependency_patterns,
)
from app.legal_rag_v2.schemas import (
    AuthorityCard,
    AuthoritySourceSpans,
    Clarification,
    DocumentSourceSpan,
    EvidenceBundle,
    LegalIssue,
    LegalResearchPlan,
    ProvisionReference,
    ResearchIntent,
)
from app.legal_rag_v2.vat import (
    enrich_input_vat_deduction_plan,
    enrich_mixed_use_vehicle_vat_plan,
    question_targets_input_vat_deduction_timing,
    question_targets_mixed_use_vehicle_vat,
)
from app.rag import decompose_query_into_legal_axes


QUESTION = """
Prowadzę działalność gospodarczą i jestem czynnym podatnikiem VAT. W 2026 r.
laptop dostarczono 30 czerwca, fakturę otrzymałem e-mailem 4 lipca, a zapłaciłem
10 lipca. Czy VAT naliczony mogę odliczyć za czerwiec czy za lipiec i w jakich
kolejnych okresach mogę dokonać odliczenia?
"""

VEHICLE_QUESTION = """
Prowadzę jednoosobową działalność gospodarczą i jestem czynnym podatnikiem VAT.
Prywatny samochód osobowy wykorzystuję służbowo i prywatnie. Nie złożyłem VAT-26
i nie prowadzę ewidencji przebiegu. Kupiłem paliwo za 500 zł netto plus 115 zł VAT.
Czy mogę odliczyć 50% VAT od paliwa i jakie są warunki odliczenia 100%?
"""


def generic_vat_plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query=QUESTION,
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="vat_general_tax_issue",
                label="VAT: general tax issue",
                tax_domains=["VAT"],
                legal_mechanism="general_tax_analysis",
            )
        ],
        clarification=Clarification(),
        confidence=0.3,
    )


class VatInputDeductionEvidenceTests(unittest.TestCase):
    def test_mixed_use_vehicle_question_gets_exact_art_86a_bundle(self) -> None:
        plan = generic_vat_plan().model_copy(update={"user_query": VEHICLE_QUESTION})

        self.assertTrue(question_targets_mixed_use_vehicle_vat(VEHICLE_QUESTION))
        enriched = enrich_mixed_use_vehicle_vat_plan(plan, VEHICLE_QUESTION)

        self.assertEqual(["mixed_use_vehicle_vat"], [item.issue_id for item in enriched.issues])
        issue = enriched.issues[0]
        queries = {query.query for query in issue.query_families}
        self.assertTrue(any("50%" in query and "samochodu" in query for query in queries))
        self.assertTrue(any("50%" in query and "motocykla" in query for query in queries))
        self.assertTrue(any("VAT-26" in query and "prywatnego" in query for query in queries))
        self.assertIn("statute", issue.requested_source_types)
        self.assertIn("interpretation", issue.requested_source_types)
        self.assertIn("judgment", issue.requested_source_types)

        requirement_ids = {
            requirement_id
            for requirement_id, _citation_pattern, _act_pattern
            in _required_issue_dependency_patterns(issue)
        }
        self.assertEqual(
            {
                "vat_art_86_1",
                "vat_art_86a_1",
                "vat_art_86a_2_3",
                "vat_art_86a_3_1_a",
                "vat_art_86a_4_1",
                "vat_art_86a_6",
                "vat_art_86a_12",
            },
            requirement_ids,
        )

    def test_vehicle_claim_synthesis_requires_substance_before_formalities_and_authority(self) -> None:
        plan = generic_vat_plan().model_copy(update={"user_query": VEHICLE_QUESTION})
        plan = enrich_mixed_use_vehicle_vat_plan(plan, VEHICLE_QUESTION)
        issue = plan.issues[0]
        provisions = [
            ProvisionReference(
                provision_id=f"vat-{index}",
                document_id="pl-ustawa-o-podatku-od-towarow-i-uslug",
                citation=citation,
                status="active",
            )
            for index, citation in enumerate(
                (
                    "art. 86 ust. 1",
                    "art. 86a ust. 1",
                    "art. 86a ust. 2 pkt 3",
                    "art. 86a ust. 3 pkt 1 lit. a",
                    "art. 86a ust. 4 pkt 1",
                    "art. 86a ust. 6",
                    "art. 86a ust. 12",
                ),
                start=1,
            )
        ]
        authority_holding = "Przy użytku mieszanym stosuje się limit 50%."
        bundle = EvidenceBundle(
            issue_id=issue.issue_id,
            controlling_provisions=[provisions[1]],
            dependency_provisions=[provisions[0], *provisions[2:]],
            supporting_authorities=[
                AuthorityCard(
                    document_id="interpretation-vehicle",
                    document_type="interpretation",
                    authority_holding=authority_holding,
                    source_spans=AuthoritySourceSpans(
                        authority_holding=[
                            DocumentSourceSpan(
                                start=0,
                                end=len(authority_holding),
                                document_id="interpretation-vehicle",
                            )
                        ]
                    ),
                    extraction_confidence=0.8,
                )
            ],
            coverage_status="complete",
        )

        requirements = {
            item["requirement_id"]
            for item in _claim_coverage_requirements(plan, [bundle])[issue.issue_id]
        }

        self.assertTrue(
            {
                "vehicle_vat_actual_use_first",
                "vehicle_vat_mixed_use_and_fuel",
                "vehicle_vat_full_deduction_conditions",
                "vehicle_vat_invoice_and_evidence",
                "vehicle_vat_interpretation_analysis",
            }.issubset(requirements)
        )

    def test_timing_question_replaces_generic_vat_issue(self) -> None:
        self.assertTrue(question_targets_input_vat_deduction_timing(QUESTION))
        enriched = enrich_input_vat_deduction_plan(generic_vat_plan(), QUESTION)

        self.assertEqual(2, len(enriched.issues))
        issue = next(
            issue for issue in enriched.issues if issue.issue_id == "vat_input_deduction_timing"
        )
        self.assertEqual("vat_input_deduction_timing", issue.issue_id)
        self.assertEqual(["VAT"], issue.tax_domains)
        queries = {query.query for query in issue.query_families}
        for citation in (
            "VAT art. 86 ust. 10",
            "VAT art. 86 ust. 10b pkt 1",
            "VAT art. 86 ust. 11",
            "VAT art. 86 ust. 13",
            "VAT art. 19a ust. 1",
        ):
            self.assertIn(citation, queries)
        self.assertTrue(any("laptop" in query.casefold() for query in queries))
        ksef_issue = next(
            issue for issue in enriched.issues if issue.issue_id == "vat_invoice_channel_2026"
        )
        ksef_queries = {query.query for query in ksef_issue.query_families}
        self.assertIn("VAT art. 106ga ust. 1", ksef_queries)
        self.assertIn("VAT art. 145m ust. 1", ksef_queries)
        self.assertIn("VAT art. 106na ust. 3", ksef_queries)
        self.assertIn("VAT art. 106nda ust. 11", ksef_queries)
        self.assertIn("VAT art. 106nf ust. 10", ksef_queries)
        self.assertIn("VAT art. 106nh ust. 4", ksef_queries)
        self.assertEqual(
            {
                "invoice_delivery_channel",
                "ksef_number_assignment_date",
                "seller_ksef_exception_status",
                "vat_cash_method_status",
            },
            {fact.fact_id for fact in enriched.missing_facts},
        )

    def test_required_bundle_covers_receipt_payment_and_later_periods(self) -> None:
        issue = next(
            issue
            for issue in enrich_input_vat_deduction_plan(
                generic_vat_plan(), QUESTION
            ).issues
            if issue.issue_id == "vat_input_deduction_timing"
        )
        requirement_ids = {
            requirement_id
            for requirement_id, _citation_pattern, _act_pattern
            in _required_issue_dependency_patterns(issue)
        }

        self.assertTrue(
            {
                "vat_art_86_10",
                "vat_art_86_10b_1",
                "vat_art_86_10e",
                "vat_art_86_11",
                "vat_art_86_13",
                "vat_art_19a_1",
            }.issubset(requirement_ids)
        )
        self.assertNotIn("unresolved_generic_issue", requirement_ids)

    def test_legacy_retrieval_axis_prefers_relevant_vat_articles(self) -> None:
        axis = next(
            axis
            for axis in decompose_query_into_legal_axes(QUESTION)
            if axis.axis_id == "vat_input_deduction_timing"
        )

        self.assertEqual({"VAT"}, axis.tax_domains)
        self.assertEqual(
            (
                ("VAT", "86"),
                ("VAT", "19a"),
            ),
            axis.preferred_targets,
        )
        self.assertIn("art. 86 ust. 10b pkt 1", axis.query)
        ksef_axis = next(
            axis
            for axis in decompose_query_into_legal_axes(QUESTION)
            if axis.axis_id == "vat_invoice_channel_2026"
        )
        self.assertIn(("VAT", "106ga"), ksef_axis.preferred_targets)
        self.assertIn(("VAT", "145m"), ksef_axis.preferred_targets)
        self.assertIn("art. 106nf ust. 10", ksef_axis.query)

    def test_ksef_channel_bundle_covers_online_offline_and_exceptions(self) -> None:
        enriched = enrich_input_vat_deduction_plan(generic_vat_plan(), QUESTION)
        issue = next(
            issue for issue in enriched.issues if issue.issue_id == "vat_invoice_channel_2026"
        )
        requirement_ids = {
            requirement_id
            for requirement_id, _citation_pattern, _act_pattern
            in _required_issue_dependency_patterns(issue)
        }

        self.assertTrue(
            {
                "vat_art_106ga_1",
                "vat_art_106ga_2_1",
                "vat_art_106ga_2_6",
                "vat_art_145m_1",
                "vat_art_145m_2",
                "vat_art_106na_3",
                "vat_art_106nda_11",
                "vat_art_106nf_10",
                "vat_art_106nh_4",
                "vat_art_106ng",
            }.issubset(requirement_ids)
        )

    def test_non_timing_vat_question_is_not_overwritten(self) -> None:
        question = "Czy sprzedaż laptopa podlega VAT według stawki 23%?"
        plan = generic_vat_plan().model_copy(update={"user_query": question})

        self.assertFalse(question_targets_input_vat_deduction_timing(question))
        self.assertIs(plan, enrich_input_vat_deduction_plan(plan, question))


if __name__ == "__main__":
    unittest.main()
