from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.legal_rag_v2.planner import IssueLocatorOutput, LegalQueryPlanner
from app.legal_rag_v2.schemas import (
    Clarification,
    Fact,
    LegalIssue,
    LegalResearchPlan,
    QueryFamily,
    ResearchIntent,
    SourceSpan,
)
from app.legal_rag_v2.trace import REQUIRED_ARTIFACTS, TracePathError, TraceWriter
from app.model_gateway import ModelTransportError


QUESTION = "Spółka wypłaca odsetki kontrahentowi z Niemiec."


def valid_plan(*, confidence: float = 0.91) -> LegalResearchPlan:
    return LegalResearchPlan(
        intent=ResearchIntent(
            mode="mixed_analysis",
            needs_normative_answer=True,
            needs_interpretations=True,
            needs_case_law=True,
            needs_conflict_analysis=True,
            needs_calculations=False,
        ),
        facts=[
            Fact(
                fact_id="payer",
                subject="Spółka",
                role="payer",
                predicate="makes_payment",
                value="Spółka",
                status="explicit",
                source_span=SourceSpan(start=0, end=6, quote="Spółka"),
            )
        ],
        missing_facts=[],
        issues=[
            LegalIssue(
                issue_id="withholding_payment",
                label="Opodatkowanie płatności transgranicznej",
                tax_domains=["CIT"],
                legal_mechanism="withholding_tax",
                taxpayer_roles=["payer", "recipient"],
                transactions=["cross_border_payment"],
                payments=["interest"],
                jurisdictions=["PL", "DE"],
                relevant_dates=[],
                possible_provision_concepts=["withholding tax on interest"],
                positive_fact_constraints=["payment is interest"],
                negative_fact_constraints=["payment is not an advisory fee"],
                requested_source_types=["statute", "interpretation", "judgment"],
                query_families=[
                    QueryFamily(
                        family="natural_language",
                        query=QUESTION,
                        lane="both",
                        origin="model",
                    )
                ],
                priority="high",
            )
        ],
        clarification=Clarification(),
        confidence=confidence,
    )


class FakeGateway:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def generate_structured(
        self,
        *,
        response_model: type[LegalResearchPlan],
        input: object,
        system_prompt: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> object:
        self.calls.append(
            {
                "response_model": response_model,
                "input": input,
                "system_prompt": system_prompt,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class LegalQueryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_gateway_is_the_primary_planner(self) -> None:
        gateway = FakeGateway(valid_plan())
        planner = LegalQueryPlanner(gateway)  # type: ignore[arg-type]

        outcome = await planner.plan(QUESTION, target_date="2026-07-11")

        self.assertTrue(outcome.primary_succeeded)
        self.assertFalse(outcome.fallback_trace.fallback_used)
        self.assertEqual(outcome.plan.user_query, QUESTION)
        self.assertEqual(outcome.plan.target_date, "2026-07-11")
        self.assertEqual(len(gateway.calls), 1)
        self.assertIs(gateway.calls[0]["response_model"], LegalResearchPlan)
        self.assertEqual(gateway.calls[0]["model"], "gpt-5.6-terra")
        self.assertIn("Do not answer", gateway.calls[0]["system_prompt"])

    async def test_timeout_uses_a_traced_legacy_fallback(self) -> None:
        gateway = FakeGateway(ModelTransportError("planner timeout"))
        planner = LegalQueryPlanner(gateway)  # type: ignore[arg-type]

        outcome = await planner.plan(QUESTION)

        self.assertFalse(outcome.primary_succeeded)
        self.assertTrue(outcome.fallback_trace.fallback_used)
        self.assertEqual(outcome.fallback_trace.fallback_reason, "planner_timeout")
        self.assertTrue(outcome.fallback_trace.fallback_rules)
        self.assertTrue(
            all(item.fallback_added for item in outcome.fallback_trace.fallback_candidates_added)
        )
        serialized = outcome.plan.model_dump()
        self.assertNotIn("conclusion", serialized)
        self.assertNotIn("final_answer", serialized)

    async def test_unsupported_fact_span_recovers_the_scoped_issue_plan(self) -> None:
        payload = valid_plan().model_dump()
        payload["facts"][0]["value"] = "Francja"
        payload["facts"][0]["source_span"] = {
            "start": 0,
            "end": 6,
            "quote": "Spółka",
            "source_id": "user_question",
        }
        gateway = FakeGateway(payload)
        planner = LegalQueryPlanner(gateway)  # type: ignore[arg-type]

        outcome = await planner.plan(QUESTION)

        self.assertFalse(outcome.fallback_trace.fallback_used)
        self.assertEqual("withholding_payment", outcome.plan.issues[0].issue_id)
        self.assertEqual([], outcome.plan.facts)
        self.assertEqual(2, len(gateway.calls))

    async def test_low_confidence_and_zero_recall_are_the_only_semantic_fallbacks(self) -> None:
        low_gateway = FakeGateway(valid_plan(confidence=0.2))
        low = await LegalQueryPlanner(low_gateway).plan(QUESTION)  # type: ignore[arg-type]
        self.assertEqual(low.fallback_trace.fallback_reason, "low_confidence")

        recall_gateway = FakeGateway(valid_plan())
        recall_planner = LegalQueryPlanner(recall_gateway)  # type: ignore[arg-type]
        first = await recall_planner.plan(QUESTION)
        augmented = recall_planner.fallback_for_insufficient_recall(
            QUESTION, first.plan, candidate_recall=0.0
        )
        self.assertEqual(len(recall_gateway.calls), 1)
        self.assertEqual(augmented.fallback_trace.fallback_reason, "insufficient_recall")

    async def test_forced_fallback_does_not_call_provider(self) -> None:
        gateway = FakeGateway(AssertionError("provider should not be called"))
        outcome = await LegalQueryPlanner(gateway).plan(QUESTION, force_fallback=True)  # type: ignore[arg-type]

        self.assertEqual(gateway.calls, [])
        self.assertEqual(outcome.fallback_trace.fallback_reason, "forced")
        self.assertLessEqual(len(outcome.plan.clarification.questions), 3)

    async def test_unscoped_model_plan_is_repaired_with_issue_only_schema(self) -> None:
        unscoped = LegalResearchPlan(
            intent=ResearchIntent(mode="mixed_analysis"),
            issues=[
                LegalIssue(
                    issue_id="cit_general_tax_issue",
                    label="CIT: general tax issue",
                    tax_domains=["CIT"],
                    legal_mechanism="general_tax_analysis",
                )
            ],
            clarification=Clarification(),
            confidence=0.9,
        )
        located_issue = LegalIssue(
            issue_id="cit_contractual_penalty_cost",
            label="CIT: kara umowna i koszt podatkowy",
            tax_domains=["CIT"],
            legal_mechanism="contractual_penalty_cost",
            possible_provision_concepts=[
                "CIT art. 15 ust. 1",
                "CIT art. 16 ust. 1 pkt 22",
            ],
        )

        class RepairGateway(FakeGateway):
            async def generate_structured(self, *, response_model, **kwargs):
                self.calls.append({"response_model": response_model, **kwargs})
                if response_model is LegalResearchPlan:
                    return unscoped
                if response_model is IssueLocatorOutput:
                    return IssueLocatorOutput(issues=[located_issue], confidence=0.88)
                raise AssertionError(response_model)

        gateway = RepairGateway(unscoped)
        outcome = await LegalQueryPlanner(gateway).plan(
            "Czy kara umowna może być kosztem uzyskania przychodów w CIT?"
        )

        self.assertFalse(outcome.fallback_trace.fallback_used)
        self.assertEqual("cit_contractual_penalty_cost", outcome.plan.issues[0].issue_id)
        self.assertEqual(
            {"CIT art. 15 ust. 1", "CIT art. 16 ust. 1 pkt 22"},
            {
                family.query
                for family in outcome.plan.issues[0].query_families
                if family.lane == "primary_law"
            },
        )
        self.assertEqual(2, len(gateway.calls))
        self.assertIs(IssueLocatorOutput, gateway.calls[1]["response_model"])


class TraceWriterTests(unittest.TestCase):
    def test_required_artifacts_are_atomic_parseable_and_secrets_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = TraceWriter("run-123", root=temporary)
            writer.initialize_required()
            writer.write_json(
                "request.json",
                {
                    "question": "key sk-proj-abcdefghijklmnopqrstuvwxyz123456",
                    "api_key": "do-not-store",
                },
            )

            self.assertEqual(
                {item.name for item in writer.run_dir.iterdir()}, set(REQUIRED_ARTIFACTS)
            )
            request = json.loads((writer.run_dir / "request.json").read_text("utf-8"))
            self.assertNotIn("sk-proj-", request["question"])
            self.assertEqual(request["api_key"], "[REDACTED]")
            for artifact_name in REQUIRED_ARTIFACTS:
                path = writer.run_dir / artifact_name
                self.assertTrue(path.exists())
                if artifact_name.endswith(".json"):
                    json.loads(path.read_text("utf-8"))
            self.assertFalse(list(writer.run_dir.glob("*.tmp")))

    def test_path_traversal_and_unknown_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TracePathError):
                TraceWriter("../escape", root=temporary)
            writer = TraceWriter("safe", root=temporary)
            with self.assertRaises(TracePathError):
                writer.write_json("../request.json", {})


if __name__ == "__main__":
    unittest.main()
