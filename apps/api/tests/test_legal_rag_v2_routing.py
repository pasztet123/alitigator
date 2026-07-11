from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.auth import AuthenticatedUser
from app.main import ChatMessage, ChatRequest, chat
from app.legal_rag_v2.schemas import (
    Clarification,
    LegalIssue,
    LegalResearchPlan,
    PipelineResult,
    ResearchIntent,
    ValidationRecord,
    WriterOutput,
)


def _plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="general",
                label="Zagadnienie",
                legal_mechanism="general",
            )
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


class FakePipeline:
    calls = 0

    def __init__(self, *, validation_passed: bool = True) -> None:
        self.validation_passed = validation_passed

    async def run(self, question, **kwargs):
        self.calls += 1
        answer = (
            "Teza\nOdpowiedź v2.\n\n"
            "Analiza\nAnaliza v2.\n\n"
            "Źródła\nBrak zatwierdzonego źródła.\n\n"
            "Ryzyka i luki\nBrak."
        )
        return PipelineResult(
            request_id="request-v2",
            run_id=kwargs["run_id"],
            legal_research_plan=_plan(),
            writer_output=WriterOutput(
                thesis="Odpowiedź v2.",
                risks_and_gaps=["Brak."],
            ),
            final_answer=answer,
            validation=[
                ValidationRecord(
                    stage="post_render",
                    passed=self.validation_passed,
                    errors=[] if self.validation_passed else ["unsupported claim"],
                )
            ],
        )


class LegalRagV2RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_deterministic_validation_blocks_user_response(self) -> None:
        pipeline = FakePipeline(validation_passed=False)
        request = ChatRequest(messages=[ChatMessage(role="user", content="Pytanie")])
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        with (
            patch.dict(os.environ, {"LEGAL_PIPELINE_MODE": "legal_rag_v2"}),
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.get_legal_rag_v2_pipeline", return_value=pipeline),
        ):
            with self.assertRaises(HTTPException) as raised:
                await chat(request, current_user=user)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("zablokował odpowiedź", str(raised.exception.detail))

    async def test_v2_returns_before_every_special_case_router(self) -> None:
        pipeline = FakePipeline()
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Ulga na złe długi")]
        )
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        with (
            patch.dict(os.environ, {"LEGAL_PIPELINE_MODE": "legal_rag_v2"}),
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.get_legal_rag_v2_pipeline", return_value=pipeline),
            patch(
                "app.main.is_bad_debt_benchmark_trace_request",
                side_effect=AssertionError("legacy benchmark router was called"),
            ),
            patch(
                "app.main.can_run_bad_debt_pipeline",
                side_effect=AssertionError("bad-debt router was called"),
            ),
            patch(
                "app.main.is_mixed_invoice_query",
                side_effect=AssertionError("mixed-invoice router was called"),
            ),
            patch(
                "app.main.can_run_housing_relief_pipeline",
                side_effect=AssertionError("housing router was called"),
            ),
        ):
            response = await chat(request, current_user=user)

        self.assertEqual(pipeline.calls, 1)
        self.assertEqual(response.analysis_trace["pipeline"], "legal_rag_v2")
        self.assertEqual(response.reply.splitlines()[0], "Teza")


if __name__ == "__main__":
    unittest.main()
