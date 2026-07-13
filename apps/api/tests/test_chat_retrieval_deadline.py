from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from app.auth import AuthenticatedUser
from app.housing_relief_pipeline import HOUSING_RELIEF_BENCHMARK_QUERY
from app.main import ChatMessage, ChatRequest, chat


class ChatRetrievalDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_housing_reply_survives_authority_retrieval_timeout(self) -> None:
        def slow_authority_retrieval(_: str):
            time.sleep(0.2)
            return [], {}

        request = ChatRequest(
            messages=[ChatMessage(role="user", content=HOUSING_RELIEF_BENCHMARK_QUERY)]
        )
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        with (
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.RETRIEVAL_STAGE_TIMEOUT_SECONDS", 0.01),
            patch("app.main.retrieve_housing_authorities", side_effect=slow_authority_retrieval),
        ):
            response = await chat(request, current_user=user)

        authority_trace = response.analysis_trace["authority_retrieval"]
        self.assertEqual(authority_trace["outcome"], "retrieval_deadline_exceeded")
        self.assertIn("Teza", response.reply)
        self.assertEqual(
            response.reply.count("brak wyniku nie oznacza braku relewantnych źródeł"),
            2,
        )

    async def test_housing_reply_renders_interpretation_and_historical_judgment(self) -> None:
        cards = [
            {
                "source_type": "interpretation",
                "label": "0112-KDIL2-1.4011.509.2026.2.MKA",
                "issue_id": "credit_on_sold_property",
                "issue_label": "spłata kredytu dotyczącego sprzedanego mieszkania",
                "holding": "Spłata kredytu zaciągniętego na sprzedane mieszkanie uprawnia do zwolnienia.",
                "holding_complete_sentence": True,
                "holding_section": "assessment_reasoning",
                "holding_source_span": {"chunk_id": "i:1", "start": 10, "end": 90},
                "outcome": "korzystny_lub_potwierdzający",
                "authority_status": "current_authority",
                "authority_score": 1.0,
                "similarity_reason": "Ten sam kredyt i sprzedawana nieruchomość.",
                "distinguishing_facts": "Brak różnicy materialnej.",
                "claim_bindings": [
                    {
                        "claim_id": "claim_credit_scope",
                        "score": 1.0,
                        "reason": "Aktualna interpretacja dotyczy tego samego mechanizmu.",
                    }
                ],
            },
            {
                "source_type": "judgment",
                "label": "II FSK 1105/25",
                "issue_id": "credit_on_sold_property",
                "issue_label": "spłata kredytu dotyczącego sprzedanego mieszkania",
                "holding": "Spłata kredytu na następnie sprzedaną nieruchomość nie była wydatkiem mieszkaniowym.",
                "holding_complete_sentence": True,
                "holding_section": "judicial_reasoning",
                "holding_source_span": {"chunk_id": "j:1", "start": 20, "end": 110},
                "outcome": "niekorzystny",
                "authority_status": "historical_authority",
                "authority_score": 0.8,
                "similarity_reason": "Ten sam historyczny mechanizm kredytu.",
                "distinguishing_facts": "Spór dotyczył stanu prawnego z 2018 r.",
                "claim_bindings": [
                    {
                        "claim_id": "claim_credit_scope",
                        "score": 0.8,
                        "reason": "Historyczne tło zmiany, nie podstawa obecnego prawa.",
                    }
                ],
            },
        ]
        outcome = {
            "outcome": "high_quality_authorities_found",
            "interpretation_lane": {
                "executed": True,
                "status": "completed",
                "selected_count": 1,
            },
            "judgment_lane": {
                "executed": True,
                "status": "completed",
                "candidate_count": 1,
                "selected_count": 1,
            },
        }
        request = ChatRequest(
            messages=[ChatMessage(role="user", content=HOUSING_RELIEF_BENCHMARK_QUERY)]
        )
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        with (
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.retrieve_housing_authorities", return_value=(cards, outcome)),
        ):
            response = await chat(request, current_user=user)

        self.assertIn("0112-KDIL2-1.4011.509.2026.2.MKA", response.reply)
        self.assertIn("II FSK 1105/25", response.reply)
        self.assertIn("Status: źródło historyczne", response.reply)
        self.assertEqual(
            response.analysis_trace["authority_retrieval"]["interpretation_lane"]["selected_count"],
            1,
        )

    async def test_housing_reply_survives_authority_card_render_failure(self) -> None:
        outcome = {
            "outcome": "high_quality_authorities_found",
            "interpretation_lane": {"executed": True, "status": "completed", "selected_count": 1},
            "judgment_lane": {"executed": True, "status": "completed", "selected_count": 0},
        }
        request = ChatRequest(
            messages=[ChatMessage(role="user", content=HOUSING_RELIEF_BENCHMARK_QUERY)]
        )
        user = AuthenticatedUser(id="user", email=None, full_name=None)

        from app.housing_relief_pipeline import run_housing_relief_pipeline as actual_pipeline

        def fail_only_with_cards(*args, **kwargs):
            if kwargs.get("authority_cards"):
                raise RuntimeError("invalid authority card")
            return actual_pipeline(*args, **kwargs)

        with (
            patch("app.main.ensure_profile"),
            patch("app.main.is_chat_storage_available", return_value=False),
            patch("app.main.is_model_gateway_configured", return_value=False),
            patch("app.main.retrieve_housing_authorities", return_value=([{"label": "bad"}], outcome)),
            patch("app.main.run_housing_relief_pipeline", side_effect=fail_only_with_cards),
        ):
            response = await chat(request, current_user=user)

        self.assertIn("Teza", response.reply)
        self.assertIn("Interpretacje: wyszukiwanie nie zostało ukończone", response.reply)
