from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from app.auth import AuthenticatedUser, get_current_user
from app.legal_institutions import InstitutionMatcher
from app.main import ChatMessage, ChatRequest, app, chat, get_legal_pipeline_mode, health
from app.legal_rag_v2.retrieval import LegalRetriever, RetrievalCandidate, RetrievalConfig
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
            self.assertEqual(get_legal_pipeline_mode(), "legal_rag_v2")

    async def test_health_exposes_the_pipeline_that_will_serve_chat(self) -> None:
        with (
            patch.dict(os.environ, {"LEGAL_RAG_MODE": "legacy"}),
            patch("app.main.index_exists", return_value=True),
            patch("app.main.is_model_gateway_configured", return_value=True),
            patch("app.main.is_supabase_configured", return_value=False),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_stripe_configured", return_value=False),
        ):
            response = health()

        self.assertEqual(response.legal_pipeline["served_by"], "legal_rag_v2")
        self.assertEqual(
            response.legal_pipeline["pipeline_mode"], "legal_rag_v2"
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

    async def test_deprecated_rag_mode_cannot_override_the_active_architecture(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_MODE": "rag_v2", "LEGAL_PIPELINE_MODE": "legacy"}):
            self.assertEqual(get_legal_pipeline_mode(), "legal_rag_v2")
        with patch.dict(os.environ, {"LEGAL_RAG_MODE": "legacy", "LEGAL_PIPELINE_MODE": "legacy"}):
            self.assertEqual(get_legal_pipeline_mode(), "legal_rag_v2")

    async def test_only_explicit_architecture_values_enable_rollback_or_shadow(self) -> None:
        with patch.dict(os.environ, {"LEGAL_QUERY_ARCHITECTURE": "v1"}):
            self.assertEqual(get_legal_pipeline_mode(), "legacy")
        with patch.dict(os.environ, {"LEGAL_QUERY_ARCHITECTURE": "v2_shadow"}):
            self.assertEqual(get_legal_pipeline_mode(), "shadow")
        with patch.dict(os.environ, {"LEGAL_QUERY_ARCHITECTURE": "typo"}):
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

    async def test_chat_entrypoint_exposes_only_evidence_backed_direct_authorities(self) -> None:
        question = (
            "Czy polska spółka musi pobrać podatek u źródła od opłaty za dostęp "
            "do zagranicznego SaaS, jeżeli umowa to EULA?"
        )

        class Backend:
            async def search(self, query, *, limit, source_types, metadata_filters):
                return [
                    RetrievalCandidate(
                        candidate_id="housing",
                        document_id="housing",
                        source_type="interpretation",
                        text="Sprzedaż lokalu mieszkalnego i ulga mieszkaniowa.",
                        metadata={"signature": "0115-KDIT1.4011.321.2026.1.MK", "subject": "Ulga mieszkaniowa", "tax_domains": ["PIT"]},
                    ),
                    RetrievalCandidate(
                        candidate_id="wht-context",
                        document_id="wht-context",
                        source_type="interpretation",
                        text="Podatek u źródła od odsetek i konwersji długu na kapitał.",
                        metadata={"signature": "0111-KDIB2-1.4010.117.2026.2.BJ", "subject": "WHT od odsetek", "tax_domains": ["CIT"], "legal_provisions": ["CIT art. 20"]},
                    ),
                    RetrievalCandidate(
                        candidate_id="wht-saas",
                        document_id="wht-saas",
                        source_type="interpretation",
                        text="Polski płatnik płaci za dostęp do oprogramowania SaaS na podstawie EULA.",
                        metadata={"signature": "WHT-SAAS", "subject": "Podatek u źródła od SaaS", "tax_domains": ["CIT"], "legal_provisions": ["CIT art. 21", "CIT art. 26"]},
                    ),
                ]

        class RetrievalBackedPipeline(FakePipeline):
            async def run(self, request_question, **kwargs):
                plan = LegalResearchPlan(
                    user_query=request_question,
                    intent=ResearchIntent(mode="mixed_analysis"),
                    issues=[LegalIssue(
                        issue_id="wht",
                        label="WHT SaaS",
                        tax_domains=["CIT"],
                        legal_mechanism="withholding_tax",
                        locked_institution_ids=["withholding_tax"],
                        possible_provision_concepts=["CIT art. 21", "CIT art. 26"],
                        requested_source_types=["interpretation"],
                    )],
                    clarification=Clarification(),
                    confidence=0.9,
                )
                retrieval = await LegalRetriever(
                    Backend(),
                    primary_enabled=False,
                    config=RetrievalConfig(selected_limit_per_issue=6),
                    institution_matcher=InstitutionMatcher(),
                ).retrieve(plan)
                candidates = list(retrieval.authorities[0].candidates)
                rejections = [item for item in retrieval.trace if item.get("event") == "institution_filter_rejection"]
                result = await super().run(request_question, **kwargs)
                return result.model_copy(update={
                    "diagnostic_trace": {
                        "dictionary_loaded": True,
                        "locked_institutions_after_merge": ["withholding_tax"],
                        "candidates_after_gate": [
                            {
                                "signature": item.metadata["signature"],
                                "mechanism": item.metadata["document_card"]["detected_mechanisms"][0],
                                "institution_gate_passed": item.metadata["document_validation"]["institution_gate_passed"],
                                "relation": item.metadata["document_validation"]["relation"],
                                "document_card": item.metadata["document_card"],
                            }
                            for item in candidates
                        ],
                        "institution_gate_results": rejections,
                    },
                })

        pipeline = RetrievalBackedPipeline()
        user = AuthenticatedUser(id="user", email=None, full_name=None)
        with (
            patch.dict(os.environ, {"LEGAL_PIPELINE_MODE": "legal_rag_v2"}),
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.get_legal_rag_v2_pipeline", return_value=pipeline),
        ):
            app.dependency_overrides[get_current_user] = lambda: user
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client:
                    response = await client.post("/api/chat", json={
                        "messages": [{"role": "user", "content": question}],
                        "retrieval_profile": "current_legal_rag",
                        "debug_trace": True,
                        "disable_cache": True,
                    })
            finally:
                app.dependency_overrides.pop(get_current_user, None)

        self.assertEqual(200, response.status_code, response.text)
        trace = response.json()["analysis_trace"]["institution_trace"]
        direct = trace["candidates_after_gate"]
        assert [item["signature"] for item in direct] == ["WHT-SAAS"]
        assert direct[0]["mechanism"] == "withholding_tax"
        assert direct[0]["institution_gate_passed"] is True
        assert direct[0]["document_card"]["evidence"]
        assert all(item["candidate_signature"]["document_id"] != "wht-saas" for item in trace["institution_gate_results"])

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
