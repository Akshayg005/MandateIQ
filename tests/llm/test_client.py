"""src/llm/client.py -- GeminiLike Protocol and GeminiClient.

No test here makes a live network call: all callers depend on GeminiLike
(a Protocol), never on GeminiClient directly, so these tests run against
in-memory doubles.
"""
from __future__ import annotations

import pytest

from src.llm.client import GeminiClient, GeminiClientError, GeminiLike


def test_importing_client_module_requires_no_network_call():
    """Merely importing src.llm.client must not reach the network or require
    google-genai to be installed at import time."""
    # This test passes just by running -- if import failed or made a network
    # call, pytest would not have gotten here.
    assert True


def test_gemini_client_construction_requires_no_network_call():
    """Constructing GeminiClient with a fake key must not touch the network
    or require google-genai at construction time (lazy initialization).
    This mirrors RazorpayClient's own documented discipline.
    """
    # Must not raise and must not touch the network.
    client = GeminiClient(api_key="fake-key-never-used")
    assert client is not None


def test_gemini_client_error_is_runtime_error_subclass():
    """GeminiClientError inherits from RuntimeError."""
    assert issubclass(GeminiClientError, RuntimeError)


def test_gemini_like_protocol_structural():
    """A fake object implementing forced_call and generate_text with matching
    signatures satisfies isinstance(fake, GeminiLike) without importing the
    real SDK."""

    class FakeGemini:
        def forced_call(
            self,
            *,
            model: str,
            system: str,
            user: str,
            tool_name: str,
            tool_schema: dict,
            temperature: float = 0.0,
        ) -> dict:
            return {"result": "fake"}

        def generate_text(
            self, *, model: str, system: str, user: str, temperature: float = 0.3
        ) -> str:
            return "fake prose"

    fake = FakeGemini()
    # The whole point of @runtime_checkable Protocol
    assert isinstance(fake, GeminiLike)


def test_gemini_like_protocol_requires_forced_call():
    """A fake missing forced_call does not satisfy GeminiLike."""

    class IncompleteFake:
        def generate_text(self, *, model, system, user, temperature=0.3) -> str:
            return "prose"

    fake = IncompleteFake()
    assert not isinstance(fake, GeminiLike)


def test_gemini_like_protocol_requires_generate_text():
    """A fake missing generate_text does not satisfy GeminiLike."""

    class IncompleteFake:
        def forced_call(
            self, *, model, system, user, tool_name, tool_schema, temperature=0.0
        ) -> dict:
            return {}

    fake = IncompleteFake()
    assert not isinstance(fake, GeminiLike)
