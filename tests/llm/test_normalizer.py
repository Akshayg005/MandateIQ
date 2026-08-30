"""src/llm/normalizer.py -- decline string normalization."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pytest

from src.core.types import DeclineClass
from src.llm.normalizer import (
    NORMALIZER_MODEL,
    NORMALIZER_VERSION,
    NormalizedDecline,
    normalize,
)
from src.llm.tools import NORMALIZE_TOOL


# --- Fake double for testing ---


@dataclass
class _FakeGemini:
    """In-memory double for GeminiLike. Records calls and returns scripted
    responses. No unittest.mock -- a real object satisfying the Protocol."""

    responses: list[dict]
    calls: list[tuple] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

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
        return {"decline_class": "UNKNOWN", "confidence": 0.5}

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


def test_normalize_insufficient_funds():
    """Basic flow: fake returns INSUFFICIENT_FUNDS -> NormalizedDecline has that value."""
    fake = _FakeGemini(
        responses=[{"decline_class": "INSUFFICIENT_FUNDS", "confidence": 0.95}]
    )
    result = normalize("INSUFFICIENT FUNDS", client=fake)

    assert result.value == DeclineClass.INSUFFICIENT_FUNDS


def test_normalized_decline_has_normalizer_version():
    """The returned NormalizedDecline.normalizer_version equals NORMALIZER_VERSION."""
    fake = _FakeGemini(
        responses=[{"decline_class": "MANDATE_REVOKED", "confidence": 0.88}]
    )
    result = normalize("MANDATE REVOKED BY BANK", client=fake)

    assert result.normalizer_version == NORMALIZER_VERSION


def test_normalized_decline_has_model_id():
    """The returned NormalizedDecline.model_id equals NORMALIZER_MODEL."""
    fake = _FakeGemini(responses=[{"decline_class": "CARD_EXPIRED", "confidence": 0.92}])
    result = normalize("CARD EXPIRED", client=fake)

    assert result.model_id == NORMALIZER_MODEL


def test_normalized_decline_raw_sha256_matches_independent_hash():
    """The returned raw_sha256 equals hashlib.sha256(raw.encode()).hexdigest()."""
    raw = "INSUFFICIENT FUNDS"
    fake = _FakeGemini(
        responses=[{"decline_class": "INSUFFICIENT_FUNDS", "confidence": 0.9}]
    )
    result = normalize(raw, client=fake)

    expected_hash = hashlib.sha256(raw.encode()).hexdigest()
    assert result.raw_sha256 == expected_hash


def test_normalize_rejects_invalid_decline_class():
    """If the fake returns a decline_class that ISN'T a valid DeclineClass value,
    normalize() raises ValueError (from DeclineClass(...) constructor).
    """
    fake = _FakeGemini(
        responses=[{"decline_class": "NOT_A_REAL_CLASS", "confidence": 0.5}]
    )

    with pytest.raises(ValueError):
        normalize("SOME DECLINE", client=fake)


def test_fake_double_records_forced_call_invocation():
    """The fake double records what forced_call was invoked with."""
    fake = _FakeGemini(responses=[{"decline_class": "UNKNOWN", "confidence": 0.5}])
    normalize("raw decline", client=fake)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["method"] == "forced_call"
    assert call["tool_name"] == NORMALIZE_TOOL["name"]
    assert call["temperature"] == 0.0


def test_fake_tool_name_matches_normalize_tool():
    """Assert the fake recorded tool_name equals NORMALIZE_TOOL["name"]."""
    fake = _FakeGemini(responses=[{"decline_class": "BANK_TIMEOUT", "confidence": 0.7}])
    normalize("TIMEOUT", client=fake)

    call = fake.calls[0]
    assert call["tool_name"] == NORMALIZE_TOOL["name"]


def test_normalizer_version_is_12_hex_chars():
    """NORMALIZER_VERSION is a 12-character lowercase-hex string."""
    assert re.fullmatch(r"[0-9a-f]{12}", NORMALIZER_VERSION)


def test_normalize_with_different_raw_strings_different_hashes():
    """Two different raw strings produce different raw_sha256 values in the result."""
    fake = _FakeGemini(
        responses=[
            {"decline_class": "INSUFFICIENT_FUNDS", "confidence": 0.9},
            {"decline_class": "INSUFFICIENT_FUNDS", "confidence": 0.9},
        ]
    )

    result1 = normalize("RAW_1", client=fake)
    result2 = normalize("RAW_2", client=fake)

    assert result1.raw_sha256 != result2.raw_sha256


def test_normalizer_version_is_same_on_all_calls():
    """Multiple calls to normalize all return the same normalizer_version (it's
    a constant, not computed per-call).
    """
    fake = _FakeGemini(
        responses=[
            {"decline_class": "INSUFFICIENT_FUNDS", "confidence": 0.9},
            {"decline_class": "MANDATE_REVOKED", "confidence": 0.85},
        ]
    )

    result1 = normalize("decline 1", client=fake)
    result2 = normalize("decline 2", client=fake)

    assert result1.normalizer_version == result2.normalizer_version == NORMALIZER_VERSION


def test_all_decline_classes_normalized():
    """Spot check: a fake that returns each DeclineClass value is accepted."""
    for decline_cls in DeclineClass:
        fake = _FakeGemini(
            responses=[{"decline_class": decline_cls.value, "confidence": 0.8}]
        )
        result = normalize(f"input_{decline_cls.value}", client=fake)
        assert result.value == decline_cls
