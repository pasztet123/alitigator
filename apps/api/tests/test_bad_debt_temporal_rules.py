from __future__ import annotations

import unittest
from typing import Optional

from app.legal_pipeline import (
    build_bad_debt_rule_bindings,
    build_bad_debt_status_facts,
)
from app.rag import LegalRule, filter_legal_rules_for_target_date


def rule(
    *,
    source_id: str,
    article: str,
    paragraph: str,
    point: str,
    letter: Optional[str] = None,
) -> LegalRule:
    return LegalRule(
        source_id=source_id,
        act_title=(
            "Ustawa o podatku od towarów i usług"
            if source_id == "vat_act"
            else "Ustawa o podatku dochodowym od osób prawnych"
        ),
        publication="2021-01-01",
        legal_state_date="2021-01-01",
        provision_id=f"{source_id}:{article}:{paragraph}:{point}:{letter or ''}",
        citation=f"art. {article} ust. {paragraph} pkt {point}",
        article_key=article,
        paragraph=paragraph,
        point=point,
        letter=letter,
        rule_type="rule",
        condition="status dłużnika",
        directive="Dłużnik nie jest w restrukturyzacji, upadłości ani likwidacji.",
        exact_source_span="historyczna przesłanka",
    )


class BadDebtTemporalRegressionTests(unittest.TestCase):
    def test_historical_vat_debtor_status_rules_are_removed_for_2026(self) -> None:
        candidates = [
            rule(source_id="vat_act", article="89a", paragraph="2", point="1"),
            rule(source_id="vat_act", article="89a", paragraph="2", point="2"),
            rule(
                source_id="vat_act",
                article="89a",
                paragraph="2",
                point="3",
                letter="b",
            ),
        ]

        selected = filter_legal_rules_for_target_date(candidates, "2026-03-31")

        self.assertEqual(selected, [])

    def test_cit_debtor_status_condition_is_preserved(self) -> None:
        cit = rule(
            source_id="cit_act",
            article="18f",
            paragraph="10",
            point="1",
        )

        selected = filter_legal_rules_for_target_date([cit], "2026-03-31")

        self.assertEqual(selected, [cit])

    def test_registration_and_insolvency_are_distinct_unassumed_facts(self) -> None:
        facts = build_bad_debt_status_facts()

        self.assertNotEqual(
            facts["debtor_vat_registration_status"].fact_type,
            facts["debtor_insolvency_status"].fact_type,
        )
        self.assertEqual(facts["debtor_vat_registration_status"].status, "missing")
        self.assertEqual(facts["debtor_insolvency_status"].status, "missing")
        self.assertIsNone(facts["debtor_vat_registration_status"].value)

    def test_insolvency_status_is_not_used_in_current_vat_but_is_used_in_cit(self) -> None:
        rules = build_bad_debt_rule_bindings("2026-03-31")
        vat_rules = [item for item in rules if item.tax_axis == "VAT"]
        cit_rules = [item for item in rules if item.tax_axis == "CIT"]

        self.assertFalse(
            any(
                "debtor_insolvency_status" in item.required_fact_ids
                for item in vat_rules
            )
        )
        self.assertTrue(
            any(
                "debtor_insolvency_status" in item.required_fact_ids
                for item in cit_rules
            )
        )
        self.assertFalse(
            any("historical" in item.rule_id for item in vat_rules)
        )


if __name__ == "__main__":
    unittest.main()
