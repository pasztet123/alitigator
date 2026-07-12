from __future__ import annotations

import unittest

from app.controlled_legal_pipeline import END_MARKER, build_renderer_payload, validate_rendered_answer
from app.housing_relief_pipeline import (
    HOUSING_RELIEF_BENCHMARK_QUERY,
    build_housing_relief_registry,
    calculate_housing_relief,
    can_run_housing_relief_pipeline,
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
        self.assertFalse(
            result.claims["claim_developer_deadline"].result["interpretive_risk_status_used"]
        )
        for provision_id in (
            "pit_art_10_ust_1_pkt_8",
            "pit_art_21_ust_1_pkt_131",
            "pit_art_21_ust_25_pkt_2",
            "pit_art_21_ust_25a",
            "pit_art_21_ust_30",
            "pit_art_21_ust_30a",
            "pit_art_30e_ust_1",
        ):
            self.assertIn(f"[provision_id:{provision_id}]", result.answer)
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

    def test_property_sale_always_plans_general_rule_relief_and_rate(self) -> None:
        plan = build_legal_source_plan(USER_REPORTED_QUERY)
        self.assertEqual(
            [(domain, article) for domain, article in plan.statute_targets if domain == "PIT"],
            [("PIT", "10"), ("PIT", "21"), ("PIT", "30e")],
        )

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
