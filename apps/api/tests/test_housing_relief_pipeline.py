from __future__ import annotations

import unittest

from app.controlled_legal_pipeline import END_MARKER, build_renderer_payload, validate_rendered_answer
from app.housing_relief_pipeline import (
    HOUSING_RELIEF_BENCHMARK_QUERY,
    build_housing_relief_registry,
    calculate_housing_relief,
    parse_housing_relief_facts,
    run_housing_relief_pipeline,
)


class HousingReliefPipelineTests(unittest.TestCase):
    def test_housing_relief_formula_and_deadline(self) -> None:
        result = run_housing_relief_pipeline(HOUSING_RELIEF_BENCHMARK_QUERY)

        self.assertTrue(result.render_validation.passed)
        self.assertEqual(
            result.claims["claim_formula"].result["exempt_income"],
            200000,
        )
        self.assertEqual(
            result.claims["claim_tax_result"].result["taxable_income"],
            100000,
        )
        self.assertEqual(result.claims["claim_tax_result"].result["tax"], 19000)
        self.assertFalse(
            result.claims["claim_formula"].result["direct_expense_income_offset_used"]
        )
        self.assertEqual(
            result.claims["claim_expense_not_income"].result["qualified_housing_expenses"],
            600000,
        )
        self.assertEqual(
            result.claims["claim_expense_not_income"].result["exempt_income"],
            200000,
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
            "pit_art_21_ust_25_pkt_2_lit_a",
            "pit_art_21_ust_25a",
            "pit_art_21_ust_30a",
            "pit_art_30e_ust_1",
        ):
            self.assertIn(f"[provision_id:{provision_id}]", result.answer)

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
        broken = result.answer.replace("19 000 zł", "0 zł") + f"\n\n{END_MARKER}"

        validation = validate_rendered_answer(broken, payload)

        self.assertFalse(validation.passed)
        self.assertIn("formula_result_text_mismatch", validation.errors)


if __name__ == "__main__":
    unittest.main()
