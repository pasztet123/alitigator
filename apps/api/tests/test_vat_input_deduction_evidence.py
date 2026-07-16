from __future__ import annotations

import unittest

from app.legal_rag_v2.pipeline import _required_issue_dependency_patterns
from app.legal_rag_v2.schemas import (
    Clarification,
    LegalIssue,
    LegalResearchPlan,
    ResearchIntent,
)
from app.legal_rag_v2.vat import (
    enrich_input_vat_deduction_plan,
    question_targets_input_vat_deduction_timing,
)
from app.rag import decompose_query_into_legal_axes


QUESTION = """
Prowadzę działalność gospodarczą i jestem czynnym podatnikiem VAT. Laptop
dostarczono 30 czerwca, fakturę otrzymałem e-mailem 4 lipca, a zapłaciłem 10
lipca. Czy VAT naliczony mogę odliczyć za czerwiec czy za lipiec i w jakich
kolejnych okresach mogę dokonać odliczenia?
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
    def test_timing_question_replaces_generic_vat_issue(self) -> None:
        self.assertTrue(question_targets_input_vat_deduction_timing(QUESTION))
        enriched = enrich_input_vat_deduction_plan(generic_vat_plan(), QUESTION)

        self.assertEqual(1, len(enriched.issues))
        issue = enriched.issues[0]
        self.assertEqual("vat_input_deduction_timing", issue.issue_id)
        self.assertEqual(["VAT"], issue.tax_domains)
        queries = {query.query for query in issue.query_families}
        for citation in (
            "VAT art. 86 ust. 10",
            "VAT art. 86 ust. 10b pkt 1",
            "VAT art. 86 ust. 11",
            "VAT art. 86 ust. 13",
            "VAT art. 19a ust. 1",
            "VAT art. 106na ust. 3",
            "VAT art. 106nda ust. 11",
        ):
            self.assertIn(citation, queries)
        self.assertTrue(any("Laptop" in query for query in queries))

    def test_required_bundle_covers_receipt_payment_and_later_periods(self) -> None:
        issue = enrich_input_vat_deduction_plan(
            generic_vat_plan(), QUESTION
        ).issues[0]
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
                "vat_art_106na_3",
                "vat_art_106nda_11",
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
                ("VAT", "106na"),
                ("VAT", "106nda"),
            ),
            axis.preferred_targets,
        )
        self.assertIn("art. 86 ust. 10b pkt 1", axis.query)

    def test_non_timing_vat_question_is_not_overwritten(self) -> None:
        question = "Czy sprzedaż laptopa podlega VAT według stawki 23%?"
        plan = generic_vat_plan().model_copy(update={"user_query": question})

        self.assertFalse(question_targets_input_vat_deduction_timing(question))
        self.assertIs(plan, enrich_input_vat_deduction_plan(plan, question))


if __name__ == "__main__":
    unittest.main()
