from __future__ import annotations

import unittest

from app.legal_pipeline import (
    CalculationRecord,
    FactRecord,
    LegalClaim,
    ProvisionRecord,
    ProvisionRegistry,
    build_claims_from_rules,
    build_registry_from_rules,
    validate_claim,
)


class ProvisionRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = [
            {
                "source_id": "cit_act",
                "provision_id": "cit_art_18f_ust_1",
                "citation": "art. 18f ust. 1",
                "article_key": "18f",
                "paragraph": "1",
                "directive": "Podstawa opodatkowania może zostać zmniejszona.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "required_facts": ["receivable_unpaid"],
                "supporting_chunk_ids": ["cit:18f:0"],
            }
        ]

    def test_exact_lookup_is_temporal_and_normalized(self) -> None:
        registry = build_registry_from_rules(self.rules)

        self.assertIsNotNone(
            registry.exact_lookup("cit_act", "Art. 18f, ust. 1", "2026-03-31")
        )
        self.assertIsNone(
            registry.exact_lookup("cit_act", "art. 18f ust. 1", "2024-12-31")
        )

    def test_undated_provision_fails_registry_acceptance(self) -> None:
        rule = dict(self.rules[0], legal_state_date="", publication="")
        registry = build_registry_from_rules([rule])

        self.assertEqual(
            registry.validate()["provisions_without_effective_dates"], 1
        )
        self.assertIsNone(registry.get("cit_art_18f_ust_1", "2026-03-31"))

    def test_repealed_provision_cannot_be_resolved_as_current(self) -> None:
        rule = dict(self.rules[0], rule_type="repealed")
        registry = build_registry_from_rules([rule])

        self.assertIsNone(registry.get("cit_art_18f_ust_1", "2026-03-31"))


class ClaimGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = [
            {
                "source_id": "cit_act",
                "provision_id": "cit_art_18f_ust_1",
                "citation": "art. 18f ust. 1",
                "article_key": "18f",
                "paragraph": "1",
                "directive": "Można zmniejszyć podstawę.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "required_facts": ["receivable_unpaid"],
            }
        ]
        self.registry = build_registry_from_rules(self.rules)

    def test_missing_fact_makes_claim_conditional(self) -> None:
        claim = build_claims_from_rules(
            self.rules,
            axis_ids=["cit_bad_debt"],
            missing_facts=["receivable_unpaid"],
        )[0]

        self.assertEqual(claim.status, "conditional_missing_fact")
        result = validate_claim(
            claim,
            self.registry,
            target_date="2026-03-31",
            facts={},
            calculations={},
        )
        self.assertIn("missing_fact_dependency", result.errors)

    def test_numeric_claim_requires_fact_or_calculation_binding(self) -> None:
        claim = LegalClaim(
            claim_id="cit_reduction",
            axis_id="cit_bad_debt",
            claim_type="calculated_result",
            text="Podstawa może zostać zmniejszona o 150 000 zł.",
            source_provisions=("cit_art_18f_ust_1",),
        )

        result = validate_claim(
            claim,
            self.registry,
            target_date="2026-03-31",
            facts={},
            calculations={},
        )
        self.assertIn("numeric_claim_without_calculation_or_fact", result.errors)

        bound_claim = LegalClaim(
            **{
                **claim.__dict__,
                "calculation_id": "calc_cit_reduction",
            }
        )
        bound = validate_claim(
            bound_claim,
            self.registry,
            target_date="2026-03-31",
            facts={"receivable_unpaid": FactRecord("receivable_unpaid", "bool", True)},
            calculations={
                "calc_cit_reduction": CalculationRecord(
                    "calc_cit_reduction",
                    "subtract",
                    {"invoice": 200_000, "paid": 50_000},
                    150_000,
                )
            },
        )
        self.assertTrue(bound.calculation_bound)

    def test_special_following_provision_is_resolved_for_housing_relief_bundle(self) -> None:
        housing_rules = [
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_25_pkt_2",
                "citation": "art. 21 ust. 25 pkt 2",
                "article_key": "21",
                "paragraph": "25",
                "point": "2",
                "directive": "Za wydatki na własne cele mieszkaniowe uważa się spłatę kredytu wraz z odsetkami.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "legal_mechanism": "housing_relief_credit_repayment",
                "entailed_result_codes": ["credit_repayment_qualified"],
            },
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_30",
                "citation": "art. 21 ust. 30",
                "article_key": "21",
                "paragraph": "30",
                "directive": "Wydatków nie uwzględnia się ponownie.",
                "legal_state_date": "2025-01-01",
                "rule_type": "restriction",
                "rule_relationship": "general_rule",
                "legal_mechanism": "housing_relief_credit_repayment",
            },
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_30a",
                "citation": "art. 21 ust. 30a",
                "article_key": "21",
                "paragraph": "30a",
                "directive": "Przepis ust. 30 nie wyłącza spłaty kredytu zaciągniętego na zbywaną nieruchomość.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "rule_relationship": "special_extension",
                "legal_mechanism": "housing_relief_credit_repayment",
                "entailed_result_codes": ["credit_on_sold_property_qualified"],
            },
        ]
        registry = build_registry_from_rules(housing_rules)
        claim = LegalClaim(
            claim_id="housing_relief_credit",
            axis_id="pit_housing_relief",
            claim_type="legal_conclusion",
            text="Spłata kredytu dotyczącego sprzedawanej nieruchomości może korzystać z ulgi.",
            source_provisions=("pit_art_21_ust_30",),
            controlling_provisions=("pit_art_21_ust_30",),
            dependency_provisions=("pit_art_21_ust_25_pkt_2",),
            status="approved",
            result_code="credit_on_sold_property_qualified",
            legal_mechanism="housing_relief_credit_repayment",
        )

        validation = validate_claim(
            claim,
            registry,
            target_date="2026-03-31",
            facts={},
            calculations={},
        )

        self.assertTrue(validation.claim_supported)
        self.assertIn("pit_art_21_ust_30a", validation.applicable_provisions)
        self.assertNotIn("incomplete_mechanism_source_bundle", validation.errors)
        self.assertNotIn("unresolved_rule_conflict", validation.errors)

    def test_special_rule_overrides_general_disqualification_claim(self) -> None:
        housing_rules = [
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_25_pkt_2",
                "citation": "art. 21 ust. 25 pkt 2",
                "article_key": "21",
                "paragraph": "25",
                "point": "2",
                "directive": "Za wydatki na własne cele mieszkaniowe uważa się spłatę kredytu wraz z odsetkami.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "legal_mechanism": "housing_relief_credit_repayment",
            },
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_30",
                "citation": "art. 21 ust. 30",
                "article_key": "21",
                "paragraph": "30",
                "directive": "Wydatków nie uwzględnia się ponownie.",
                "legal_state_date": "2025-01-01",
                "rule_type": "restriction",
                "rule_relationship": "general_rule",
                "legal_mechanism": "housing_relief_credit_repayment",
                "entailed_result_codes": ["credit_on_sold_property_disqualified"],
            },
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_30a",
                "citation": "art. 21 ust. 30a",
                "article_key": "21",
                "paragraph": "30a",
                "directive": "Przepis ust. 30 nie wyłącza spłaty kredytu zaciągniętego na zbywaną nieruchomość.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "rule_relationship": "special_extension",
                "legal_mechanism": "housing_relief_credit_repayment",
                "entailed_result_codes": ["credit_on_sold_property_qualified"],
            },
        ]
        registry = build_registry_from_rules(housing_rules)
        invalid_claim = LegalClaim(
            claim_id="housing_relief_credit_denied",
            axis_id="pit_housing_relief",
            claim_type="legal_conclusion",
            text="Spłata kredytu dotyczącego sprzedawanej nieruchomości nie kwalifikuje się do ulgi.",
            source_provisions=("pit_art_21_ust_30",),
            controlling_provisions=("pit_art_21_ust_30",),
            dependency_provisions=("pit_art_21_ust_25_pkt_2",),
            status="approved",
            result_code="credit_on_sold_property_disqualified",
            legal_mechanism="housing_relief_credit_repayment",
        )

        validation = validate_claim(
            invalid_claim,
            registry,
            target_date="2026-03-31",
            facts={},
            calculations={},
        )

        self.assertFalse(validation.claim_supported)
        self.assertIn("unresolved_rule_conflict", validation.errors)
        self.assertIn("credit_repayment_disqualification_blocked", validation.errors)

    def test_housing_relief_bundle_is_mandatory(self) -> None:
        housing_rules = [
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_25_pkt_2",
                "citation": "art. 21 ust. 25 pkt 2",
                "article_key": "21",
                "paragraph": "25",
                "point": "2",
                "directive": "Za wydatki na własne cele mieszkaniowe uważa się spłatę kredytu wraz z odsetkami.",
                "legal_state_date": "2025-01-01",
                "rule_type": "permission",
                "legal_mechanism": "housing_relief_credit_repayment",
            },
            {
                "source_id": "pit_act",
                "provision_id": "pit_art_21_ust_30",
                "citation": "art. 21 ust. 30",
                "article_key": "21",
                "paragraph": "30",
                "directive": "Wydatków nie uwzględnia się ponownie.",
                "legal_state_date": "2025-01-01",
                "rule_type": "restriction",
                "rule_relationship": "general_rule",
                "legal_mechanism": "housing_relief_credit_repayment",
            },
        ]
        registry = build_registry_from_rules(housing_rules)
        claim = LegalClaim(
            claim_id="housing_relief_incomplete_bundle",
            axis_id="pit_housing_relief",
            claim_type="legal_conclusion",
            text="Analiza została oparta na niepełnym pakiecie przepisów.",
            source_provisions=("pit_art_21_ust_30",),
            controlling_provisions=("pit_art_21_ust_30",),
            dependency_provisions=("pit_art_21_ust_25_pkt_2",),
            status="approved",
            result_code="credit_on_sold_property_qualified",
            legal_mechanism="housing_relief_credit_repayment",
        )

        validation = validate_claim(
            claim,
            registry,
            target_date="2026-03-31",
            facts={},
            calculations={},
        )

        self.assertFalse(validation.claim_supported)
        self.assertIn("incomplete_mechanism_source_bundle", validation.errors)


if __name__ == "__main__":
    unittest.main()
