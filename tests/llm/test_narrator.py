"""src/llm/narrator.py -- batch narratives with claims guard."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.llm.narrator import NARRATOR_MODEL, NarratorClaimError, narrate


@dataclass
class _FakeGemini:
    """In-memory double for GeminiLike."""

    prose: str
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
        # narrator.py should never call forced_call
        raise AssertionError("narrator.py must not call forced_call")

    def generate_text(self, *, model: str, system: str, user: str, temperature: float = 0.3) -> str:
        self.calls.append(
            {
                "method": "generate_text",
                "model": model,
                "temperature": temperature,
            }
        )
        return self.prose


# --- Tests ---


def test_narrate_clean_prose_no_forbidden_claims():
    """Fake returns clean prose with no forbidden claims -> narrate() returns it unchanged."""
    clean = "41 mandates were attempted, 28 recovered. 13 entered pause-and-retry."
    fake = _FakeGemini(prose=clean)
    result = narrate("batch context", client=fake)

    assert result == clean


def test_narrate_rejects_active_cancel_by_system():
    """Measured real-world defect: 'we forced mandate cancellations' -> NarratorClaimError."""
    real_defect = (
        "Continuing to aggressively retry these accounts would have triggered "
        "compliance violations and forced mandate cancellations."
    )
    fake = _FakeGemini(prose=real_defect)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)


def test_narrate_allows_off_ramp_language():
    """Safe sentence using 'cancel' descriptively must NOT raise."""
    safe = (
        "41 mandates were offered an off-ramp: pause, then downgrade, then cancel, "
        "with the customer deciding at each step."
    )
    fake = _FakeGemini(prose=safe)
    result = narrate("batch context", client=fake)

    assert result == safe


def test_narrate_rejects_false_claim_we_cancelled():
    """Direct false claim 'We cancelled X mandates' -> NarratorClaimError."""
    false_claim = "We cancelled 12 mandates this cycle."
    fake = _FakeGemini(prose=false_claim)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)


def test_narrate_rejects_system_cancelled():
    """'The system cancelled mandates' -> NarratorClaimError."""
    false_claim = "The system automatically cancelled 5 mandates for non-payment."
    fake = _FakeGemini(prose=false_claim)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)


def test_narrator_temperature_is_not_zero():
    """The generate_text call is made with temperature != 0.0 (prose allowed variation)."""
    fake = _FakeGemini(prose="clean prose")
    narrate("batch", client=fake)

    call = fake.calls[0]
    # Should be 0.3 based on the implementation, but the test only checks "not 0"
    assert call["temperature"] != 0.0


def test_narrator_temperature_specifically_0_3():
    """The temperature should be 0.3 (verified by checking what was passed)."""
    fake = _FakeGemini(prose="clean prose")
    narrate("batch", client=fake)

    call = fake.calls[0]
    assert call["temperature"] == 0.3


def test_fake_records_model():
    """The fake records the model parameter."""
    fake = _FakeGemini(prose="prose")
    narrate("batch context", client=fake)

    call = fake.calls[0]
    assert call["model"] == NARRATOR_MODEL


def test_narrate_rejects_forced_cancellation():
    """Claim of 'forced cancellation' -> NarratorClaimError."""
    false_claim = "We were forced to cancel these mandates by regulatory action."
    fake = _FakeGemini(prose=false_claim)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)


def test_narrate_rejects_automatic_cancellation():
    """'Automatically cancelled' -> NarratorClaimError."""
    false_claim = "The system automatically cancelled mandates for accounts flagged."
    fake = _FakeGemini(prose=false_claim)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)


def test_narrate_allows_mention_of_cancel_in_context():
    """The word 'cancel' in a safe context (off-ramp stages) is allowed."""
    safe = "Our three-stage off-ramp (pause, downgrade, cancel) is customer-controlled."
    fake = _FakeGemini(prose=safe)
    result = narrate("batch", client=fake)

    assert result == safe


def test_narrator_claim_error_is_runtime_error():
    """NarratorClaimError is a RuntimeError subclass."""
    assert issubclass(NarratorClaimError, RuntimeError)


def test_narrate_multi_paragraph_no_forbidden():
    """Multi-paragraph clean prose should pass."""
    clean = (
        "The batch recovered 67 mandates. We paused 23 and offered downgrade. "
        "Customers declined the offers and we honored that.\n\n"
        "Next cycle focuses on reattempt timing."
    )
    fake = _FakeGemini(prose=clean)
    result = narrate("batch", client=fake)

    assert result == clean


def test_narrate_rejects_our_system_cancelled():
    """'Our system cancelled X' -> NarratorClaimError."""
    false_claim = "Our system cancelled 3 mandates due to repeated failures."
    fake = _FakeGemini(prose=false_claim)

    with pytest.raises(NarratorClaimError):
        narrate("batch", client=fake)
