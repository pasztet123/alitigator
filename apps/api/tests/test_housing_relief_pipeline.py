from __future__ import annotations

import unittest
from dataclasses import replace

from app.controlled_legal_pipeline import END_MARKER, build_renderer_payload, validate_rendered_answer
from app.housing_relief_pipeline import (
    HOUSING_RELIEF_BENCHMARK_QUERY,
    build_housing_relief_registry,
    calculate_housing_relief,
    can_run_housing_relief_pipeline,
    validate_housing_deadline_invariants,
    parse_housing_relief_facts,
    run_housing_relief_pipeline,
)
from app.rag import build_legal_source_plan


NATURAL_LANGUAGE_QUERY = """
Podatnik kupił mieszkanie w 2022 r. za 600 tys. zł i sprzedał je w 2025 r.
za 900 tys. zł. Po sprzedaży spłacił 300 tys. zł kredytu zaciągniętego na zakup
tego sprzedanego mieszkania oraz wpłacił 300 tys. zł deweloperowi na nowe
mieszkanie. Przeniesienie własności nowego mieszkania ma nastąpić w 2029 r.
Jak rozliczyć PIT i ulgę mieszkaniową?
"""

USER_REPORTED_QUERY = """
W 2025 r. sprzedałem za 900 tys. zł mieszkanie, które kupiłem w 2022 r. za 600 tys. zł.
Z pieniędzy ze sprzedaży spłaciłem 300 tys. zł kredytu zaciągniętego na zakup tego
mieszkania i wpłaciłem kolejne 300 tys. zł deweloperowi za nowe mieszkanie, ale akt
przenoszący własność mam dostać dopiero w 2029 r. Czy muszę zapłacić PIT od sprzedaży,
a jeżeli tak, to od jakiej kwoty?
"""


class HousingReliefPipelineTests(unittest.TestCase):
    def test_housing_relief_formula_and_deadline(self) -> None:
        result = run_housing_relief_pipeline(HOUSING_RELIEF_BENCHMARK_QUERY)

        self.assertTrue(result.render_validation.passed)
        self.assertEqual(
            result.claims["claim_formula"].result["exempt_income"],
            100000,
        )
        self.assertEqual(
            result.claims["claim_tax_result"].result["taxable_income"],
            200000,
        )
        self.assertEqual(result.claims["claim_tax_result"].result["tax"], 38000)
        self.assertFalse(
            result.claims["claim_formula"].result["direct_expense_income_offset_used"]
        )
        self.assertEqual(
            result.claims["claim_expense_not_income"].result["qualified_housing_expenses"],
            300000,
        )
        self.assertEqual(
            result.claims["claim_expense_not_income"].result["exempt_income"],
            100000,
        )
        self.assertFalse(
            result.claims["claim_expense_not_income"].result["values_treated_as_identical"]
        )
        self.assertEqual(
            result.claims["claim_developer_deadline"].result["housing_expense_deadline"],
            "2028-12-31",
        )
        self.assertEqual(
            result.claims["claim_developer_deadline"].result["planned_transfer_year"],
            2029,
        )
        self.assertFalse(
            result.claims["claim_developer_deadline"].result["developer_expense_qualifies"]
        )
        self.assertEqual(
            result.claims["claim_developer_deadline"].result["disqualified_developer_expense"],
            300000,
        )
        self.assertEqual(result.claims["claim_developer_deadline"].status, "approved")
        self.assertEqual(
            result.claims["claim_developer_deadline"].result["status"],
            "approved_not_qualifying",
        )
        self.assertIn(
            "Trzyletni termin wynika z art. 21 ust. 1 pkt 131 ustawy PIT.",
            result.answer,
        )
        self.assertIn(
            "Warunek nabycia własności wynika z art. 21 ust. 25a ustawy PIT",
            result.answer,
        )
        self.assertFalse(
            result.claims["claim_developer_deadline"].result["interpretive_risk_status_used"]
        )
        for reference in (
            "art. 10 ust. 1 pkt 8 ustawy PIT",
            "art. 21 ust. 1 pkt 131 ustawy PIT",
            "art. 21 ust. 25 pkt 2 ustawy PIT",
            "art. 21 ust. 25a ustawy PIT",
            "art. 21 ust. 30 ustawy PIT",
            "art. 21 ust. 30a ustawy PIT",
            "art. 30e ust. 1 ustawy PIT",
        ):
            self.assertIn(reference, result.answer)
        self.assertNotIn(
            "Spłata kredytu zaciągniętego na zakup sprzedanego mieszkania nie jest wydatkiem",
            result.answer,
        )
        self.assertNotIn("ugruntowane stanowisko", result.answer.lower())

    def test_natural_language_query_uses_controlled_credit_rule(self) -> None:
        self.assertTrue(can_run_housing_relief_pipeline(NATURAL_LANGUAGE_QUERY))

        result = run_housing_relief_pipeline(NATURAL_LANGUAGE_QUERY)

        self.assertEqual(result.claims["claim_sale_tax_regime"].result["income"], 300000)
        self.assertEqual(result.claims["claim_credit_scope"].result["credit_repayment"], 300000)
        self.assertTrue(result.claims["claim_credit_scope"].result["credit_repayment_qualifies"])
        self.assertEqual(
            result.claims["claim_credit_scope"].result_code,
            "credit_on_sold_property_qualified",
        )
        self.assertEqual(result.claims["claim_tax_result"].result["tax"], 38000)
        self.assertIn("art. 21 ust. 30a ustawy PIT", result.answer)

    def test_reported_act_transferring_ownership_phrase_uses_controlled_pipeline(self) -> None:
        self.assertTrue(can_run_housing_relief_pipeline(USER_REPORTED_QUERY))
        result = run_housing_relief_pipeline(USER_REPORTED_QUERY)
        self.assertTrue(result.render_validation.passed)
        self.assertEqual(result.claims["claim_tax_result"].result["tax"], 38_000)
        self.assertEqual(
            result.claims["claim_developer_deadline"].result["housing_relief_deadline"],
            "2028-12-31",
        )
        self.assertIn("Upływa 2028-12-31.", result.answer)
        self.assertNotIn("Upływa 2025-12-31.", result.answer)

    def test_housing_deadline_has_independent_statutory_provenance(self) -> None:
        facts = parse_housing_relief_facts(HOUSING_RELIEF_BENCHMARK_QUERY)
        calculations = calculate_housing_relief(facts)
        deadline = calculations["calc_housing_relief_deadline"]
        self.assertEqual(facts.sale_year_end, "2025-12-31")
        self.assertEqual(facts.deadline, "2028-12-31")
        self.assertEqual(facts.records["housing_relief_deadline"].value, "2028-12-31")
        self.assertEqual(deadline.result, "2028-12-31")
        self.assertEqual(deadline.inputs["deadline_period_source"], "pit_art_21_ust_1_pkt_131")
        self.assertEqual(deadline.inputs["ownership_condition_source"], "pit_art_21_ust_25a")
        self.assertEqual(validate_housing_deadline_invariants(facts, calculations), ())

        corrupted = dict(calculations)
        corrupted["calc_housing_relief_deadline"] = replace(deadline, result="2025-12-31")
        self.assertIn(
            "deadline_calculation_result_invalid",
            validate_housing_deadline_invariants(facts, corrupted),
        )

    def test_sale_year_is_not_replaced_by_later_purchase_year(self) -> None:
        facts = parse_housing_relief_facts(USER_REPORTED_QUERY)
        calculations = calculate_housing_relief(facts)

        self.assertEqual(facts.sale_year, 2025)
        self.assertEqual(facts.purchase_year, 2022)
        self.assertEqual(facts.sale_year_end, "2025-12-31")
        self.assertEqual(facts.deadline, "2028-12-31")
        self.assertEqual(calculations["calc_housing_relief_deadline"].result, "2028-12-31")

    def test_housing_transfer_date_boundaries(self) -> None:
        for transfer_date, expected in (
            ("2027-12-31", True),
            ("2028-12-31", True),
            ("2029-01-01", False),
        ):
            query = NATURAL_LANGUAGE_QUERY.replace("2029 r.", transfer_date)
            facts = parse_housing_relief_facts(query)
            calculations = calculate_housing_relief(facts)
            self.assertEqual(
                calculations["calc_housing_relief_developer_qualification"].result,
                expected,
                transfer_date,
            )
            result = run_housing_relief_pipeline(query)
            self.assertEqual(
                result.claims["claim_developer_deadline"].result["developer_expense_qualifies"],
                expected,
                transfer_date,
            )

    def test_property_sale_always_plans_general_rule_relief_and_rate(self) -> None:
        plan = build_legal_source_plan(USER_REPORTED_QUERY)
        self.assertEqual(
            [(domain, article) for domain, article in plan.statute_targets if domain == "PIT"],
            [("PIT", "10"), ("PIT", "21"), ("PIT", "30e")],
        )

    def test_user_answer_has_sources_and_no_internal_metadata(self) -> None:
        result = run_housing_relief_pipeline(
            HOUSING_RELIEF_BENCHMARK_QUERY,
            authority_cards=(
                {
                    "source_type": "interpretation",
                    "label": "0115-KDIT3.4011.123.2025.1.AK",
                    "source_url": "https://example.test/interpretation",
                },
            ),
            judgment_lane_outcome={
                "executed": True,
                "candidate_count": 0,
                "selected_count": 0,
                "empty_result_reason": "no_candidates_from_corpus",
            },
        )
        self.assertIn("Źródła\n- art. 10 ust. 1 pkt 8 ustawy PIT.", result.answer)
        self.assertIn("0115-KDIT3.4011.123.2025.1.AK", result.answer)
        self.assertIn("zagadnienie: pit_housing_relief", result.answer)
        self.assertIn("Holding:", result.answer)
        self.assertNotIn("claim_sale_tax_regime", result.answer)
        self.assertIn("Wynik:", result.answer)
        self.assertIn("Podobieństwo:", result.answer)
        self.assertIn("Różnica:", result.answer)
        self.assertIn("Orzeczenia: nie znaleziono dostatecznie relewantnego orzeczenia", result.answer)
        self.assertNotIn("kandydaci:", result.answer)
        self.assertNotIn("Powód braku wyboru:", result.answer)
        authority_card = result.renderer_payload["authority_cards"][0]
        self.assertEqual(authority_card["issue_id"], "pit_housing_relief")
        self.assertIn("claim_sale_tax_regime", authority_card["claim_ids"])
        self.assertEqual(len(authority_card["claim_bindings"]), 1)
        self.assertEqual(
            result.renderer_payload["judgment_lane_outcome"]["empty_result_reason"],
            "no_candidates_from_corpus",
        )
        self.assertIn("Ryzyka i luki\n- Do dokumentacyjnego potwierdzenia", result.answer)
        self.assertIn("38 000 zł", result.answer)
        self.assertIn("Wzór D × W / P", result.answer)
        for forbidden in ("[claim_id:", "[provision_id:", "fact_", "calculation_id:"):
            self.assertNotIn(forbidden, result.answer)
        self.assertLess(result.render_validation.thesis_analysis_duplicate_ratio, 0.35)

    def test_authority_lane_statuses_distinguish_no_match_from_timeout(self) -> None:
        completed = run_housing_relief_pipeline(
            HOUSING_RELIEF_BENCHMARK_QUERY,
            interpretation_lane_outcome={
                "executed": True,
                "status": "completed",
                "selected_count": 0,
            },
            judgment_lane_outcome={
                "executed": True,
                "status": "completed",
                "selected_count": 0,
            },
        )
        self.assertIn("Interpretacje: nie znaleziono dostatecznie relewantnej interpretacji", completed.answer)
        self.assertIn("Orzeczenia: nie znaleziono dostatecznie relewantnego orzeczenia", completed.answer)

        timed_out = run_housing_relief_pipeline(
            HOUSING_RELIEF_BENCHMARK_QUERY,
            interpretation_lane_outcome={
                "executed": True,
                "status": "deadline_exceeded",
                "selected_count": 0,
            },
            judgment_lane_outcome={
                "executed": True,
                "status": "deadline_exceeded",
                "selected_count": 0,
                "empty_result_reason": "retrieval_error",
            },
        )
        self.assertEqual(timed_out.answer.count("brak wyniku nie oznacza braku relewantnych źródeł"), 2)

    def test_explicit_credit_facts_drive_main_scenario_while_documents_remain_conditional(self) -> None:
        result = run_housing_relief_pipeline(HOUSING_RELIEF_BENCHMARK_QUERY)
        self.assertEqual(result.claims["claim_tax_result"].result["tax"], 38_000)
        self.assertTrue(result.facts["credit_for_sold_property"].value)
        self.assertTrue(result.facts["credit_taken_before_sale"].value)
        self.assertEqual(result.facts["credit_not_previously_tax_preferenced"].status, "missing")
        conditional = result.claims["claim_credit_fallback_tax_scenario"]
        self.assertEqual(conditional.status, "conditional_missing_fact")
        self.assertEqual(conditional.result["tax"], 57_000)
        self.assertIn("57 000 zł", result.answer)
        self.assertIn("38 000 zł", result.answer)

    def test_validator_blocks_formula_mismatch_in_text(self) -> None:
        facts = parse_housing_relief_facts(HOUSING_RELIEF_BENCHMARK_QUERY)
        calculations = calculate_housing_relief(facts)
        result = run_housing_relief_pipeline(HOUSING_RELIEF_BENCHMARK_QUERY)
        payload = build_renderer_payload(
            result.claims,
            build_housing_relief_registry(),
            target_date="2026-06-30",
            calculations=calculations,
        )
        broken = result.answer.replace("38 000 zł", "0 zł") + f"\n\n{END_MARKER}"

        validation = validate_rendered_answer(broken, payload)

        self.assertFalse(validation.passed)
        self.assertIn("formula_result_text_mismatch", validation.errors)


if __name__ == "__main__":
    unittest.main()
