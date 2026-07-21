from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.auth import AuthenticatedUser
from app.main import ChatMessage, ChatRequest, chat, get_legal_pipeline_mode, health
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

    def __init__(
        self,
        *,
        validation_passed: bool = True,
        validation_stage: str = "post_render",
    ) -> None:
        self.validation_passed = validation_passed
        self.validation_stage = validation_stage

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
                    stage=self.validation_stage,
                    passed=self.validation_passed,
                    errors=[] if self.validation_passed else ["unsupported claim"],
                )
            ],
        )


class LegalRagV2RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_v2_is_the_default_when_no_routing_variable_is_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_legal_pipeline_mode(), "model_rag_model")

    async def test_health_exposes_the_pipeline_that_will_serve_chat(self) -> None:
        with (
            patch.dict(os.environ, {"LEGAL_RAG_MODE": "model_rag_model"}),
            patch("app.main.index_exists", return_value=True),
            patch("app.main.is_model_gateway_configured", return_value=True),
            patch("app.main.is_supabase_configured", return_value=False),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_stripe_configured", return_value=False),
        ):
            response = health()

        self.assertEqual(response.legal_pipeline["served_by"], "legal_rag_v2")
        self.assertEqual(
            response.legal_pipeline["pipeline_mode"], "model_rag_model"
        )
        self.assertIn("pipeline_version", response.legal_pipeline)

    async def test_fallback_plan_accepts_a_question_with_trailing_whitespace(self) -> None:
        from app.legal_rag_v2.planner import LegacyFallbackPlanner

        plan, _ = LegacyFallbackPlanner().plan(
            "Podatnik sprzedaje lokal.\n",
            reason="provider_unavailable",
        )
        for fact in plan.facts:
            self.assertEqual(
                fact.source_span.quote,
                "Podatnik sprzedaje lokal.\n"[
                    fact.source_span.start : fact.source_span.end
                ],
            )

    async def test_public_rag_mode_flag_maps_rag_v2_without_changing_legacy_alias(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_MODE": "rag_v2", "LEGAL_PIPELINE_MODE": "legacy"}):
            self.assertEqual(get_legal_pipeline_mode(), "legal_rag_v2")

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
        self.assertIn("kontroli integralności", str(raised.exception.detail))

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
        self.assertEqual(response.analysis_trace["served_by"], "legal_rag_v2")
        self.assertIn("pipeline_version", response.analysis_trace)
        self.assertIn("retrieval_iterations", response.analysis_trace)
        self.assertEqual(response.reply.splitlines()[0], "Teza")

    async def test_chat_entrypoint_exposes_institution_diagnostic_trace(self) -> None:
        class TracePipeline(FakePipeline):
            async def run(self, question, **kwargs):
                result = await super().run(question, **kwargs)
                return result.model_copy(update={
                    "diagnostic_trace": {
                        "dictionary_loaded": True,
                        "dictionary_version": "test-v2",
                        "institution_matches": [
                            {"id": "csr_sponsorship_relief", "locked": True}
                        ],
                        "locked_institutions_after_merge": ["csr_sponsorship_relief"],
                        "authority_search_input": {
                            "mechanisms": ["csr_sponsorship_relief"],
                            "provision_hints": ["PIT art. 26ha", "CIT art. 18ee"],
                        },
                        "institution_gate_results": [],
                        "candidates_before_gate": [
                            {
                                "signature": "REHAB-1",
                                "relation": "irrelevant",
                                "reject": True,
                                "document_card": {
                                    "detected_institutions": [],
                                    "detected_mechanisms": ["rehabilitation_relief"],
                                },
                            }
                        ],
                        "candidates_after_gate": [
                            {
                                "signature": "WHT-SAAS-1",
                                "relation": "direct",
                                "institution_gate_passed": True,
                                "document_card": {
                                    "detected_institutions": ["withholding_tax"],
                                    "detected_mechanisms": ["withholding_tax"],
                                },
                            }
                        ],
                        "final_results": [],
                    },
                })

        pipeline = TracePipeline()
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Poszukaj interpretacji związanych z ulgą sponsoringową")],
            retrieval_profile="current_legal_rag",
            debug_trace=True,
            disable_cache=True,
        )
        user = AuthenticatedUser(id="user", email=None, full_name=None)
        with (
            patch.dict(os.environ, {"LEGAL_PIPELINE_MODE": "legal_rag_v2"}),
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.get_legal_rag_v2_pipeline", return_value=pipeline),
        ):
            response = await chat(request, current_user=user)

        trace = response.analysis_trace["institution_trace"]
        self.assertTrue(trace["dictionary_loaded"])
        self.assertIn("csr_sponsorship_relief", trace["locked_institutions_after_merge"])
        self.assertIn("csr_sponsorship_relief", trace["authority_search_input"]["mechanisms"])
        self.assertIn("PIT art. 26ha", trace["authority_search_input"]["provision_hints"])
        rejected = trace["candidates_before_gate"][0]
        self.assertTrue(rejected["reject"])
        self.assertNotIn("withholding_tax", rejected["document_card"]["detected_mechanisms"])
        self.assertEqual("direct", trace["candidates_after_gate"][0]["relation"])

    async def test_recovered_claim_validation_is_served_without_a_trace_id(self) -> None:
        pipeline = FakePipeline(
            validation_passed=False,
            validation_stage="claim_validation",
        )
        request = ChatRequest(messages=[ChatMessage(role="user", content="Pytanie")])
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        with (
            patch.dict(os.environ, {"LEGAL_PIPELINE_MODE": "legal_rag_v2"}),
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.get_legal_rag_v2_pipeline", return_value=pipeline),
        ):
            response = await chat(request, current_user=user)

        self.assertEqual(response.reply.splitlines()[0], "Teza")
        self.assertNotIn("Trace diagnostyczny", response.reply)


if __name__ == "__main__":
    unittest.main()
