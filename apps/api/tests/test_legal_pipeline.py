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


if __name__ == "__main__":
    unittest.main()
