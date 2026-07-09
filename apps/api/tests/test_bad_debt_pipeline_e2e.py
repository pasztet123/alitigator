from __future__ import annotations

import re
import unittest
from pathlib import Path

from app.bad_debt_pipeline import (
    VAT_ART_89A_VERIFIED_SPAN,
    _load_statute_article,
    build_bad_debt_claims,
    build_bad_debt_registry,
    calculate_bad_debt,
    can_run_bad_debt_pipeline,
    parse_bad_debt_facts,
    run_bad_debt_pipeline,
)
from app.controlled_legal_pipeline import (
    END_MARKER,
    build_renderer_payload,
    render_answer,
    validate_rendered_answer,
)
from app.legal_pipeline import (
    LegalClaim,
    ProvisionRecord,
    ProvisionRegistry,
    TemporalConflictError,
    validate_claim,
)


BENCHMARK_QUERY = """
Ulga na złe długi VAT i CIT. Faktura z 10 września 2025 r. opiewa na
200 000 zł netto i 46 000 zł VAT. Termin płatności przypadał 30 września
2025 r. Częściowa zapłata 61 500 zł nastąpiła 15 stycznia 2026 r.
JPK_V7M za grudzień 2025 r. złożono 25 stycznia 2026 r., a CIT-8 za
2025 r. złożono 31 marca 2026 r. Brak informacji o statusie
restrukturyzacyjnym, upadłościowym lub likwidacyjnym dłużnika na
28 lutego 2026 r. Pozostałe 184 500 zł zapłacono 10 maja 2026 r.
Oceń ulgę po upływie 90 dni i późniejszą zapłatę.
"""


class BadDebtPipelineEndToEndTests(unittest.TestCase):
    def test_router_requires_complete_structured_case(self) -> None:
        self.assertTrue(can_run_bad_debt_pipeline(BENCHMARK_QUERY))
        self.assertFalse(
            can_run_bad_debt_pipeline(
                "Jak działa ulga na złe długi w VAT i CIT?"
            )
        )

    def test_benchmark_passes_three_times_with_expected_results(self) -> None:
        for _ in range(3):
            result = run_bad_debt_pipeline(BENCHMARK_QUERY)

            self.assertEqual(
                result.calculations["calc_ninety_day_date"].result,
                "2025-12-29",
            )
            self.assertEqual(
                result.calculations["calc_unpaid_net_amount"].result, 150_000
            )
            self.assertEqual(
                result.calculations["calc_unpaid_vat_amount"].result, 34_500
            )
            self.assertEqual(
                result.calculations["calc_cit_tax_effect"].result, 28_500
            )
            self.assertEqual(
                result.claims["claim_vat_relief"].result["status"], "approved"
            )
            self.assertFalse(
                result.claims["claim_vat_relief"].result[
                    "debtor_insolvency_status_required"
                ]
            )
            self.assertEqual(
                result.claims["claim_cit_relief"].status,
                "conditional_missing_fact",
            )
            self.assertEqual(
                result.claims["claim_cit_relief"].missing_fact_dependencies,
                ("debtor_status_on_2026_02_28",),
            )
            self.assertEqual(
                result.claims["claim_vat_reversal"].result["period"], "2026-05"
            )
            self.assertEqual(
                result.claims["claim_cit_reversal"].result["year"], 2026
            )
            self.assertEqual(
                result.claims["claim_cit_reversal"].controlling_provisions,
                ("cit_art_18f_ust_7",),
            )
            self.assertFalse(
                result.claims["claim_cit_no_retro"].result[
                    "retroactive_correction"
                ]
            )
            self.assertTrue(result.render_validation.passed)
            self.assertFalse(result.render_validation.truncated)
            self.assertEqual(result.render_validation.placeholder_count, 0)
            self.assertEqual(result.render_validation.thesis_contradictions, ())
            self.assertEqual(
                set(result.renderer_payload),
                {"approved_claims", "conditional_claims", "answer_plan", "provisions"},
            )
            self.assertNotIn("raw_documents", result.renderer_payload)
            self.assertNotIn("retrieved_chunks", result.renderer_payload)
            self.assertTrue(
                all(
                    not claim["status"].startswith("blocked")
                    for claim in result.renderer_payload["approved_claims"]
                )
            )
            self.assertNotIn("ten przepis", result.answer.lower())
            self.assertIn("vat_art_89a_ust_3", result.answer)
            self.assertIn("vat_art_89a_ust_2_pkt_3_lit_a", result.answer)
            self.assertIn("Brak uregulowania dla korekty VAT ocenia się do dnia złożenia deklaracji", result.answer)

    def test_all_material_claims_have_real_provenance(self) -> None:
        result = run_bad_debt_pipeline(BENCHMARK_QUERY)
        payload_claims = [
            *result.renderer_payload["approved_claims"],
            *result.renderer_payload["conditional_claims"],
        ]
        for claim in payload_claims:
            self.assertTrue(claim["controlling_provisions"])
            self.assertTrue(claim["provenance"])
            for source in claim["provenance"]:
                self.assertTrue(source["provision_id"])
                self.assertTrue(source["version_id"])
                self.assertTrue(source["display_reference"])
                self.assertTrue(source["source_span"])
        numeric_claims = [
            claim
            for claim in result.claims.values()
            if re.search(r"\d", claim.text)
        ]
        self.assertTrue(all(claim.calculation_ids for claim in numeric_claims))

    def test_historical_vat_provisions_are_unavailable_in_2026(self) -> None:
        registry = build_bad_debt_registry()
        historical = (
            "vat_art_89a_ust_2_pkt_1",
            "vat_art_89a_ust_2_pkt_2",
            "vat_art_89a_ust_2_pkt_3_lit_b",
        )
        self.assertTrue(
            all(registry.get(item, "2026-03-31") is None for item in historical)
        )
        self.assertIsNotNone(registry.get("vat_art_89a_ust_3", "2026-03-31"))
        self.assertIsNotNone(
            registry.get("vat_art_89a_ust_2_pkt_3_lit_a", "2026-03-31")
        )

    def test_registry_works_without_runtime_data_directory(self) -> None:
        source = _load_statute_article(
            Path("/missing-in-cloud-run/vat.jsonl"),
            "89a",
            fallback_document_id="eli:DU:2025:775:art_89a",
            fallback_version_id="dz_u_2025_poz_775@2025-05-16",
            fallback_source_span=VAT_ART_89A_VERIFIED_SPAN,
        )

        self.assertEqual(source["document_id"], "eli:DU:2025:775:art_89a")
        self.assertIn("Art. 89a.", source["source_span"])

    def test_same_missing_fact_is_irrelevant_to_vat_and_conditional_for_cit(self) -> None:
        result = run_bad_debt_pipeline(BENCHMARK_QUERY)
        self.assertNotIn(
            "debtor_status_on_2026_02_28",
            result.claims["claim_vat_relief"].fact_dependencies,
        )
        self.assertIn(
            "debtor_status_on_2026_02_28",
            result.claims["claim_cit_relief"].fact_dependencies,
        )

    def test_placeholder_contradiction_and_truncation_are_rejected(self) -> None:
        facts = parse_bad_debt_facts(BENCHMARK_QUERY)
        calculations = calculate_bad_debt(facts)
        claims = build_bad_debt_claims(facts, calculations)
        payload = build_renderer_payload(
            claims, build_bad_debt_registry(), target_date="2026-03-31"
        )
        valid = render_answer(payload)

        placeholder = valid.replace(
            "art. 89a ust. 1 ustawy VAT",
            "zweryfikowany przepis wskazany w źródłach primary law",
            1,
        )
        self.assertFalse(validate_rendered_answer(placeholder, payload).passed)

        ten_przepis = valid.replace(
            "art. 89a ust. 1 ustawy VAT",
            "ten przepis",
            1,
        )
        self.assertFalse(validate_rendered_answer(ten_przepis, payload).passed)

        empty_sources = re.sub(
            rf"\n\nŹródła\n.*?\n\nRyzyka i luki\n",
            "\n\nŹródła\n\n\nRyzyka i luki\n",
            valid,
            flags=re.S,
        )
        empty_sources_validation = validate_rendered_answer(empty_sources, payload)
        self.assertFalse(empty_sources_validation.passed)
        self.assertIn("required_sections_empty", empty_sources_validation.errors)

        contradictory = valid.replace(
            "Status restrukturyzacyjny, upadłościowy ani likwidacyjny dłużnika nie blokuje ulgi VAT wierzyciela",
            "Status restrukturyzacyjny, upadłościowy ani likwidacyjny dłużnika blokuje ulgę VAT wierzyciela",
            1,
        )
        contradiction_validation = validate_rendered_answer(
            contradictory, payload
        )
        self.assertFalse(contradiction_validation.passed)
        self.assertIn(
            "claim_vat_relief",
            contradiction_validation.thesis_contradictions,
        )

        truncated = (
            valid.partition("### CIT\n")[0]
            + "### VAT\n| kolumna | wartość\n| urwany"
        )
        truncated_validation = validate_rendered_answer(truncated, payload)
        self.assertFalse(truncated_validation.passed)
        self.assertFalse(truncated_validation.end_marker_present)
        self.assertFalse(truncated_validation.tables_closed)
        self.assertIn("CIT", truncated_validation.missing_required_sections)


class RegistryAndEntailmentTests(unittest.TestCase):
    def test_temporal_conflict_fails_closed(self) -> None:
        base = build_bad_debt_registry().get(
            "vat_art_89a_ust_1", "2026-03-31"
        )
        assert base is not None
        duplicate = ProvisionRecord(
            **{**base.__dict__, "version_id": "conflicting_version"}
        )
        registry = ProvisionRegistry(provisions=(base, duplicate))

        with self.assertRaises(TemporalConflictError):
            registry.get("vat_art_89a_ust_1", "2026-03-31")

    def test_source_entailment_rejects_cross_tax_and_wrong_result(self) -> None:
        registry = build_bad_debt_registry()
        facts = parse_bad_debt_facts(BENCHMARK_QUERY)
        bad_claim = LegalClaim(
            claim_id="bad",
            axis_id="cit_bad_debt_creditor",
            claim_type="legal_conclusion",
            text="Claim spoza zakresu źródła.",
            source_provisions=("vat_art_89a_ust_1",),
            controlling_provisions=("vat_art_89a_ust_1",),
            status="approved",
            result_code="cit_relief_available",
            taxpayer_role="creditor",
            legal_mechanism="bad_debt_relief",
        )

        validation = validate_claim(
            bad_claim,
            registry,
            target_date="2026-03-31",
            facts=facts.records,
            calculations={},
        )

        self.assertFalse(validation.claim_supported)
        self.assertIn("source_tax_domain_mismatch", validation.errors)
        self.assertIn("source_does_not_entail_claim", validation.errors)


if __name__ == "__main__":
    unittest.main()
