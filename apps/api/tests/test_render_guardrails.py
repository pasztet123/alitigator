from __future__ import annotations

import unittest

from app.main import (
    RENDER_COMPLETION_MARKER,
    enforce_reply_guardrails,
    validate_final_output,
)


class ConditionalAnswerGuardrailTests(unittest.TestCase):
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
