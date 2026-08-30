"""src/llm/intent.py -- exit-intent scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pytest

from src.llm.intent import INTENT_MODEL, INTENT_VERSION, intent_score
from src.llm.tools import INTENT_TOOL


@dataclass
class _FakeGemini:
    """In-memory double for GeminiLike."""

    responses: list[dict]
    calls: list[dict] = field(default_factory=list)

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
        self.calls.append(
            {
                "method": "forced_call",
                "model": model,
                "tool_name": tool_name,
                "temperature": temperature,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return {"intent_score": 0.5, "rationale": "default"}

    def generate_text(self, *, model: str, system: str, user: str, temperature: float = 0.3) -> str:
        self.calls.append(
            {
                "method": "generate_text",
                "model": model,
                "temperature": temperature,
            }
        )
        return "fake prose"


# --- Tests ---


def test_intent_score_returns_float():
    """intent_score() returns a float."""
    fake = _FakeGemini(responses=[{"intent_score": 0.73, "rationale": "customer is frustrated"}])
    result = intent_score("don't charge me", client=fake)

    assert isinstance(result, float)


def test_intent_score_basic_value():
    """Fake returns 0.73 -> intent_score returns 0.73."""
    fake = _FakeGemini(responses=[{"intent_score": 0.73, "rationale": "..."}])
    result = intent_score("text", client=fake)

    assert result == 0.73


def test_intent_score_clamped_to_1_0():
    """Fake returns 1.4 (out of range) -> clamped to exactly 1.0."""
    fake = _FakeGemini(responses=[{"intent_score": 1.4, "rationale": "..."}])
    result = intent_score("text", client=fake)

    assert result == 1.0


def test_intent_score_clamped_to_0_0():
    """Fake returns -0.2 (out of range) -> clamped to exactly 0.0."""
    fake = _FakeGemini(responses=[{"intent_score": -0.2, "rationale": "..."}])
    result = intent_score("text", client=fake)

    assert result == 0.0


def test_intent_score_clamped_is_float_not_int():
    """Even when fake returns an int, intent_score() returns a float."""
    fake = _FakeGemini(responses=[{"intent_score": 1, "rationale": "..."}])
    result = intent_score("text", client=fake)

    assert isinstance(result, float)
    assert result == 1.0


def test_fake_double_records_forced_call():
    """The fake double records forced_call invocation with correct args."""
    fake = _FakeGemini(responses=[{"intent_score": 0.5, "rationale": "..."}])
    intent_score("text", client=fake)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["method"] == "forced_call"
    assert call["tool_name"] == INTENT_TOOL["name"]
    assert call["temperature"] == 0.0


def test_intent_temperature_is_zero():
    """The forced_call is made with temperature=0.0 (determinism matters)."""
    fake = _FakeGemini(responses=[{"intent_score": 0.5, "rationale": "..."}])
    intent_score("text", client=fake)

    call = fake.calls[0]
    assert call["temperature"] == 0.0


def test_intent_version_is_12_hex_chars():
    """INTENT_VERSION is a 12-character lowercase-hex string."""
    assert re.fullmatch(r"[0-9a-f]{12}", INTENT_VERSION)


def test_intent_score_zero():
    """Fake returns 0.0 -> returns 0.0."""
    fake = _FakeGemini(responses=[{"intent_score": 0.0, "rationale": "no exit signal"}])
    result = intent_score("keeps paying happily", client=fake)

    assert result == 0.0


def test_intent_score_mid_range():
    """Fake returns 0.5 -> returns 0.5."""
    fake = _FakeGemini(responses=[{"intent_score": 0.5, "rationale": "unclear"}])
    result = intent_score("text", client=fake)

    assert result == 0.5


def test_intent_tool_name_in_call():
    """The fake records tool_name matching INTENT_TOOL["name"]."""
    fake = _FakeGemini(responses=[{"intent_score": 0.6, "rationale": "..."}])
    intent_score("message", client=fake)

    call = fake.calls[0]
    assert call["tool_name"] == INTENT_TOOL["name"]
