from __future__ import annotations

import unittest

from app.main import (
    CHAT_REQUEST_DEADLINE_SECONDS,
    MODEL_CHAT_TIMEOUT_SECONDS,
    RENDER_COMPLETION_MARKER,
    build_chat_system_prompt,
    complete_empty_sources_section,
    enforce_reply_guardrails,
    repair_empty_legal_reference_slots,
    validate_final_output,
)


class ConditionalAnswerGuardrailTests(unittest.TestCase):
    def test_model_timeout_allows_complex_answers_but_is_bounded(self) -> None:
        self.assertGreaterEqual(MODEL_CHAT_TIMEOUT_SECONDS, 90.0)
        self.assertLessEqual(MODEL_CHAT_TIMEOUT_SECONDS, 180.0)
        self.assertGreaterEqual(CHAT_REQUEST_DEADLINE_SECONDS, 30.0)
        self.assertLessEqual(CHAT_REQUEST_DEADLINE_SECONDS, 180.0)

    def test_axis_validator_accepts_common_markdown_heading_levels(self) -> None:
        for heading in ("CIT", "## CIT", "#### CIT", "**CIT**", "### CIT – rozliczenie"):
            with self.subTest(heading=heading):
                reply = (
                    "Teza\nWniosek.\n\n"
                    f"Analiza\n{heading}\nAnaliza osi.\n\n"
                    "Źródła\nart. 18f ustawy CIT.\n\n"
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
            "Źródła\n[provision_id:cit_art_18f_ust_1].\n\n"
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

    def test_final_validator_rejects_ten_przepis_and_empty_sections(self) -> None:
        placeholder_reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nWynika to z ten przepis.\n\n"
            "Źródła\nart. 89a ustawy VAT.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        with self.assertRaisesRegex(Exception, "placeholder"):
            validate_final_output(
                placeholder_reply,
                axis_coverage=[],
                expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
            )

        empty_sources_reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nAnaliza.\n\n"
            "Źródła\n\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        with self.assertRaisesRegex(Exception, "pusta sekcja"):
            validate_final_output(
                empty_sources_reply,
                axis_coverage=[],
                expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
            )

    def test_empty_sources_section_is_completed_from_verified_retrieval(self) -> None:
        reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nAnaliza.\n\n"
            "Źródła\nBrak zatwierdzonego źródła.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        completed = complete_empty_sources_section(
            reply,
            retrieval_citations="- source_document_id: vat_2025 | art. 86 ustawy VAT | https://example.test/vat",
        )
        validation = validate_final_output(
            completed,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )

        self.assertFalse(validation["sources_without_sources"])
        self.assertIn("source_document_id: vat_2025", completed)

    def test_empty_sources_section_declares_missing_retrieval_sources(self) -> None:
        reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nAnaliza.\n\n"
            "Źródła\nBrak zatwierdzonego źródła.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )
        completed = complete_empty_sources_section(reply, retrieval_citations="")
        validation = validate_final_output(
            completed,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )
        self.assertFalse(validation["sources_without_sources"])
        self.assertIn("Nie znaleziono zweryfikowanych źródeł", completed)

    def test_final_validator_rejects_empty_legal_reference_slots(self) -> None:
        reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nPrawo do korekty powstaje w rozliczeniu za grudzień 2025 r. "
            "( i ust. 2 pkt 5 ustawy o VAT). Ulga regulowana jest ustawy o CIT. "
            "Sprzedaż podlega opodatkowaniu ( ustawy o PIT).\n\n"
            "Źródła\nart. 89a ustawy VAT.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        with self.assertRaisesRegex(Exception, "puste miejsce po referencji"):
            validate_final_output(
                reply,
                axis_coverage=[],
                expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
            )

    def test_final_validator_rejects_unsupported_authority_line_claim(self) -> None:
        reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nJest to ugruntowane stanowisko organów podatkowych i sądów administracyjnych.\n\n"
            "Źródła\nart. 21 ustawy PIT.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )

        with self.assertRaisesRegex(Exception, "linii organów lub sądów bez sygnatur"):
            validate_final_output(
                reply,
                axis_coverage=[],
                expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
            )

    def test_guardrails_repair_empty_legal_reference_slots_from_trace(self) -> None:
        reply = (
            "Teza\nWniosek.\n\n"
            "Analiza\nPrawo do korekty powstaje w grudniu ( i ust. 2 pkt 5 ustawy o VAT). "
            "Ulga regulowana jest ustawy o CIT. Dochód wynika z ustawy o PIT.\n\n"
            "Źródła\nart. 89a ustawy VAT; art. 18f ustawy CIT.\n\n"
            "Ryzyka i luki\nBrak.\n\n"
            f"{RENDER_COMPLETION_MARKER}"
        )
        traces = [
            {"provision_reference": "art. 89a ust. 3 ustawy VAT"},
            {"provision_reference": "art. 18f ust. 5 ustawy CIT"},
            {"provision_reference": "art. 30e ust. 1 ustawy PIT"},
        ]

        repaired = repair_empty_legal_reference_slots(reply, traces)

        self.assertIn("art. 89a ust. 3 ustawy VAT", repaired)
        self.assertIn("regulowana jest art. 18f ust. 5 ustawy CIT", repaired)
        self.assertIn("wynika z art. 30e ust. 1 ustawy PIT", repaired)
        validation = validate_final_output(
            repaired,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )
        self.assertTrue(validation["no_empty_legal_reference_slots"])


if __name__ == "__main__":
    unittest.main()
