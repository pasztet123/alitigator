from __future__ import annotations

import unittest

from app.legal_rag_v2.cit_costs import enrich_cit_cost_plan
from app.legal_rag_v2.schemas import (
    Clarification,
    LegalIssue,
    LegalResearchPlan,
    ResearchIntent,
)
from app.rag import decompose_query_into_legal_axes


QUESTION = """
Polska spółka zapłaciła karę umowną za opóźnienie dostawy prawidłowych,
niewadliwych towarów. Czy może zaliczyć wydatek do kosztów uzyskania
przychodów w CIT? Porównaj opóźnienie z wadami towarów i zwłoką w usunięciu
wad oraz oceń należytą staranność i związek z zachowaniem przychodów.
"""


def generic_plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query=QUESTION,
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="cit_general_tax_issue",
                label="CIT: general tax issue",
                tax_domains=["CIT"],
                legal_mechanism="general_tax_analysis",
            )
        ],
        clarification=Clarification(),
        confidence=0.35,
    )


class CitCostEvidenceTests(unittest.TestCase):
    def test_contractual_penalty_replaces_unscoped_cit_issue(self) -> None:
        enriched = enrich_cit_cost_plan(generic_plan(), QUESTION)

        self.assertEqual(1, len(enriched.issues))
        issue = enriched.issues[0]
        self.assertEqual("cit_contractual_penalty_cost", issue.issue_id)
        self.assertNotIn("general tax issue", issue.label.casefold())
        queries = {query.query for query in issue.query_families}
        self.assertIn("CIT art. 15 ust. 1", queries)
        self.assertIn("CIT art. 16 ust. 1", queries)
        self.assertIn("CIT art. 16 ust. 1 pkt 22", queries)

    def test_legacy_axis_is_specific_and_carries_both_cit_articles(self) -> None:
        axes = decompose_query_into_legal_axes(QUESTION)
        axis = next(item for item in axes if item.axis_id == "cit_contractual_penalty_cost")

        self.assertEqual({"CIT"}, axis.tax_domains)
        self.assertEqual((('CIT', '15'), ('CIT', '16')), axis.preferred_targets)
        self.assertIn("opóźnienie dostawy", axis.query)

    def test_unrelated_cit_question_is_not_rewritten_as_penalty(self) -> None:
        question = "Jaka stawka CIT ma zastosowanie do spółki?"
        plan = generic_plan().model_copy(update={"user_query": question})

        self.assertIs(plan, enrich_cit_cost_plan(plan, question))


if __name__ == "__main__":
    unittest.main()
