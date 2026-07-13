import asyncio
import unittest
from unittest.mock import patch

from app.main import request_prompt_hints


class _SlowHintsGateway:
    async def generate_structured(self, **_: object) -> object:
        await asyncio.sleep(1)
        raise AssertionError("The interactive hints deadline was not enforced")


class PromptHintsTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_model_returns_fast_fallback_hints(self) -> None:
        with (
            patch("app.main.is_model_gateway_configured", return_value=True),
            patch("app.main.create_model_gateway", return_value=_SlowHintsGateway()),
            patch("app.main.HINTS_REQUEST_TIMEOUT_SECONDS", 0.01),
        ):
            response = await request_prompt_hints(
                "Sprzedałem mieszkanie i pytam o ulgę mieszkaniową.",
                [],
            )

        self.assertEqual(response.mode, "fallback")
        self.assertTrue(response.hints)
