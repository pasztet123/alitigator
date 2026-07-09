from __future__ import annotations

import unittest

from app.main import (
    RENDER_COMPLETION_MARKER,
    build_chat_system_prompt,
    enforce_reply_guardrails,
    validate_final_output,
)


class ConditionalAnswerGuardrailTests(unittest.TestCase):
    def test_axis_validator_accepts_common_markdown_heading_levels(self) -> None:
        for heading in ("CIT", "## CIT", "#### CIT", "**CIT**", "### CIT – rozliczenie"):
            with self.subTest(heading=heading):
                reply = (
                    "Teza\nWniosek.\n\n"
                    f"Analiza\n{heading}\nAnaliza osi.\n\n"
                    "Źródła\nŹródło.\n\n"
                    "Ryzyka i luki\nBrak.\n\n"
                    f"{RENDER_COMPLETION_MARKER}"
                )
                validation = validate_final_output(
                    reply,
                    axis_coverage=[
                        {"axis_id": "cit_bad_debt_creditor", "label": "CIT"}
                    ],
                    expected_sections=[
                        "Teza",
                        "Analiza",
                        "Źródła",
                        "Ryzyka i luki",
                    ],
                )

                self.assertEqual(validation["rendered_axes"], 1)

    def test_empty_retrieval_prompt_still_requires_complete_contract(self) -> None:
        prompt = build_chat_system_prompt(
            "Pytanie bez trafnych źródeł",
            "",
            [],
        )

        self.assertIn(RENDER_COMPLETION_MARKER, prompt)
        self.assertIn("Zacznij odpowiedź dokładnie", prompt)
        self.assertIn("Teza", prompt)

    def test_writer_is_not_told_to_quote_full_statute_before_thesis(self) -> None:
        prompt = build_chat_system_prompt(
            "Pytanie",
            "Zweryfikowany kontekst źródłowy.",
            [],
        )

        self.assertNotIn(
            "Zacznij odpowiedź od pełnego brzmienia",
            prompt,
        )
        self.assertIn("Nie umieszczaj przed nią cytatu", prompt)

    def test_missing_fact_keeps_required_thesis_heading(self) -> None:
        reply = (
            "Teza\nWynik zależy od brakującego faktu.\n\n"
            "Analiza\nNależy rozważyć dwa warianty.\n\n"
            "Źródła\nZweryfikowane źródła opisano w analizie.\n\n"
            "Ryzyka i luki\nBrakujący fakt wymaga potwierdzenia.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        guarded = enforce_reply_guardrails(
            reply,
            allowed_provision_references=set(),
            missing_required_facts=["status dłużnika"],
            timeline_issues=[],
        )
        validation = validate_final_output(
            guarded,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )

        self.assertTrue(validation["has_completion_marker"])
        self.assertEqual(validation["missing_planned_sections"], 0)
        self.assertNotIn("Teza warunkowa\n", guarded)
        self.assertIn("Teza\nNa obecnym materiale", guarded)


if __name__ == "__main__":
    unittest.main()
