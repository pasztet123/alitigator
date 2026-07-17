from __future__ import annotations

import unittest

from app.legal_rag_v2.cit_costs import cost_tax_domain, enrich_cit_cost_plan
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
        self.assertTrue(
            all(
                query.family == "explicit_provision_reference"
                for query in issue.query_families
                if query.lane == "primary_law"
            )
        )

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

    def test_jdg_cost_question_routes_to_pit_in_both_planners(self) -> None:
        question = (
            "Prowadzę jednoosobową działalność gospodarczą. Czy okulary "
            "korekcyjne mogę zaliczyć do kosztów uzyskania przychodów?"
        )
        plan = generic_plan().model_copy(update={"user_query": question})
        enriched = enrich_cit_cost_plan(plan, question)

        self.assertEqual("PIT", cost_tax_domain(question))
        self.assertEqual("pit_cost_deductibility", enriched.issues[0].issue_id)
        self.assertEqual(["PIT"], enriched.issues[0].tax_domains)
        self.assertIn(
            "PIT art. 22 ust. 1",
            {query.query for query in enriched.issues[0].query_families},
        )
        self.assertTrue(
            any("okulary" in query.query for query in enriched.issues[0].query_families)
        )
        axes = decompose_query_into_legal_axes(question)
        axis = next(item for item in axes if item.axis_id == "pit_cost_deductibility")
        self.assertEqual({"PIT"}, axis.tax_domains)
        self.assertEqual((("PIT", "22"), ("PIT", "23")), axis.preferred_targets)
        self.assertNotIn("CIT art. 15", axis.query)

    def test_cost_authority_lane_keeps_concrete_expense_facts(self) -> None:
        question = (
            "Prowadzę JDG jako programista i wykupiłem indywidualny kurs języka "
            "angielskiego do obsługi zagranicznych klientów. Czy jest to koszt "
            "uzyskania przychodów w PIT?"
        )
        enriched = enrich_cit_cost_plan(
            generic_plan().model_copy(update={"user_query": question}),
            question,
        )
        issue = enriched.issues[0]
        authority_query = next(
            query.query
            for query in issue.query_families
            if query.lane in {"authority", "both"}
            and query.family == "fact_signature"
        )

        self.assertTrue(
            {"kurs", "języka", "angielskiego", "programista"}.issubset(
                set(issue.transactions)
            )
        )
        self.assertLess(authority_query.index("kurs"), authority_query.index("PIT:"))
        self.assertNotIn("Podaj aktualną podstawę prawną", authority_query)
        self.assertEqual(
            {
                "fact_signature",
                "legal_concept",
                "quoted_holding_language",
            },
            {
                query.family
                for query in issue.query_families
                if query.lane in {"authority", "both"}
                and query.origin == "fallback"
            },
        )

    def test_explicit_cit_still_wins_for_company_penalty(self) -> None:
        self.assertEqual("CIT", cost_tax_domain(QUESTION))


if __name__ == "__main__":
    unittest.main()
