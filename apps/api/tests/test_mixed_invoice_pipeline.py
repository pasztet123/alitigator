from __future__ import annotations

import unittest

from app.controlled_legal_pipeline import (
    END_MARKER,
    build_mixed_invoice_registry,
    build_renderer_payload,
    render_answer,
    run_legal_pipeline,
    validate_rendered_answer,
)


BENCHMARK_QUERY = """
Faktura mieszana na 30 000 zł netto obejmuje towary z załącznika nr 15
oraz usługę transportową. Płatność A 20 000 zł dotyczy towarów z
załącznika nr 15, a Płatność B 10 000 zł wyłącznie transportu.
Oceń MPP, koszty, odpowiedzialność solidarną i ZAW-NR na 30 czerwca 2026 r.
"""


class MixedInvoiceEndToEndTests(unittest.TestCase):
    def test_mixed_invoice_end_to_end(self) -> None:
        result = run_legal_pipeline(BENCHMARK_QUERY)

        self.assertEqual(len(result.claims), 10)
        self.assertIs(
            result.claims["claim_mpp_payment_a"].result["mpp_mandatory"], True
        )
        self.assertIs(
            result.claims["claim_mpp_payment_b"].result["mpp_mandatory"], False
        )
        self.assertTrue(
            result.claims["claim_mpp_payment_b"].controlling_provisions
        )
        self.assertEqual(
            result.claims["claim_cost_payment_a"].calculation_id,
            "calc_cost_payment_a",
        )
        self.assertTrue(result.render_validation.passed)
        self.assertEqual(result.render_validation.placeholder_count, 0)
        self.assertEqual(result.render_validation.missing_claim_ids, ())
        self.assertEqual(result.render_validation.thesis_contradictions, ())
        self.assertFalse(result.render_validation.truncated)
        self.assertEqual(result.render_validation.missing_required_sections, ())
        self.assertIn("[provision_id:vat_art_108a_ust_1a]", result.answer)
        self.assertIn("[version_id:vat_act_2019-11-01]", result.answer)
        self.assertIn("Fakty: fact_invoice_total", result.answer)
        self.assertIn("Obliczenia: calc_cost_payment_a", result.answer)

        payload = result.renderer_payload
        self.assertNotIn("raw_documents", payload)
        self.assertNotIn("retrieved_chunks", payload)
        self.assertFalse(
            any(
                claim["status"].startswith("blocked")
                for claim in payload["approved_claims"]
            )
        )

    def test_mixed_invoice_mpp_scope_is_per_payment(self) -> None:
        result = run_legal_pipeline(BENCHMARK_QUERY)

        self.assertIs(
            result.claims["claim_mpp_payment_a"].result["mpp_mandatory"], True
        )
        self.assertIs(
            result.claims["claim_mpp_payment_b"].result["mpp_mandatory"], False
        )

    def test_zaw_nr_lookup_excludes_historical_authority(self) -> None:
        registry = build_mixed_invoice_registry()

        self.assertIsNone(
            registry.get("ord_art_117ba_par_4_historical", "2026-06-30")
        )
        current = registry.get("ord_art_117ba_par_4", "2026-06-30")
        self.assertIsNotNone(current)
        self.assertIn("właściwego dla podatnika", current.text)

    def test_validator_fails_closed_on_missing_claim_and_contradiction(self) -> None:
        result = run_legal_pipeline(BENCHMARK_QUERY)
        broken = (
            result.answer.replace(
                "Płatność B nie podlega obowiązkowemu MPP.",
                "Płatność B podlega obowiązkowemu MPP.",
                1,
            )
            .replace("[claim_id:claim_invoice_scope]", "", 1)
            + f"\n\n{END_MARKER}"
        )
        payload_result = run_legal_pipeline(BENCHMARK_QUERY)
        registry = build_mixed_invoice_registry()
        payload = build_renderer_payload(
            payload_result.claims, registry, target_date="2026-06-30"
        )

        validation = validate_rendered_answer(broken, payload)

        self.assertFalse(validation.passed)
        self.assertIn("claim_invoice_scope", validation.missing_claim_ids)
        self.assertIn(
            "claim_mpp_payment_b", validation.thesis_contradictions
        )

    def test_validator_rejects_truncation_and_missing_section(self) -> None:
        result = run_legal_pipeline(BENCHMARK_QUERY)
        registry = build_mixed_invoice_registry()
        payload = build_renderer_payload(
            result.claims, registry, target_date="2026-06-30"
        )
        broken = result.answer.partition("\n\nŹródła\n")[0] + "\nUrwany tekst"

        validation = validate_rendered_answer(broken, payload)

        self.assertFalse(validation.passed)
        self.assertFalse(validation.end_marker_present)
        self.assertTrue(validation.truncated)
        self.assertIn("Źródła", validation.missing_required_sections)
        self.assertIn("Ryzyka i luki", validation.missing_required_sections)


if __name__ == "__main__":
    unittest.main()
