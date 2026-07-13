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
