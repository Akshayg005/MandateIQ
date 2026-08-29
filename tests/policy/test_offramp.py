"""src/policy/offramp.py -- pause/downgrade/cancel constructed as an offer.

Design spec: construct_offer(b, ctx) returns an Offer whose three steps are
ordered PAUSE, DOWNGRADE, CANCEL -- least drastic first. Invariant 6: this
module must never call a cancellation endpoint; it only ever describes one
as a menu item the customer could choose.
"""
from __future__ import annotations

import pathlib

from src.core.types import Cause, MandateState, Profile
from src.policy import belief as belief_mod
from src.policy.offramp import Offer, OffRampStep, construct_offer
from src.policy.stopping_rules import AllocationContext


def _ctx(**overrides) -> AllocationContext:
    base = dict(
        mandate_id="M-1",
        cycle_id=1,
        profile=Profile.strict,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        plan_day=10,
        attempts_used=1,
        committed_days=(1,),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=4,
        quiet_hours_start=21,
        quiet_hours_end=8,
    )
    base.update(overrides)
    return AllocationContext(**base)


def _belief() -> belief_mod.Belief:
    return belief_mod.init({Cause.WONT_PAY: 0.9, Cause.CANT_PAY_NOW: 0.05, Cause.CANT_PAY_EVER: 0.05})


# === structure ================================================================

def test_construct_offer_returns_an_offer():
    offer = construct_offer(_belief(), _ctx())
    assert isinstance(offer, Offer)


def test_offer_has_exactly_three_steps():
    offer = construct_offer(_belief(), _ctx())
    assert len(offer.steps) == 3


def test_steps_are_ordered_pause_downgrade_cancel():
    offer = construct_offer(_belief(), _ctx())
    kinds = [s.kind for s in offer.steps]
    assert kinds == ["PAUSE", "DOWNGRADE", "CANCEL"], f"got order {kinds}"


def test_every_step_is_an_offramp_step_with_a_description():
    offer = construct_offer(_belief(), _ctx())
    for step in offer.steps:
        assert isinstance(step, OffRampStep)
        assert isinstance(step.description, str) and step.description


def test_offer_carries_mandate_identity():
    ctx = _ctx(mandate_id="M-42", cycle_id=7)
    offer = construct_offer(_belief(), ctx)
    assert offer.mandate_id == "M-42"
    assert offer.cycle_id == 7


def test_offer_carries_belief_as_json():
    b = _belief()
    offer = construct_offer(b, _ctx())
    assert offer.belief_json == b.to_json()


def test_offer_expires_after_plan_day():
    ctx = _ctx(plan_day=10)
    offer = construct_offer(_belief(), ctx)
    assert offer.expires_on_day > 10


# === invariant 6: never executes a cancellation ==============================

def test_offramp_module_never_calls_a_cancel_endpoint():
    """Mechanical proof, not just a docstring claim: the source text of
    offramp.py must contain no `.cancel(` or `.cancel_subscription(` call
    -- the same pattern scripts/guard_invariants.py's HARD_CANCEL regex
    checks, reproduced here so this specific invariant has a test living
    next to the file it protects, not only a hook that runs elsewhere."""
    import re

    path = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "policy" / "offramp.py"
    text = path.read_text(encoding="utf-8")
    hard_cancel = re.compile(r"\.(cancel_subscription|cancel)\s*\(", re.IGNORECASE)
    match = hard_cancel.search(text)
    assert match is None, f"offramp.py contains a direct cancellation call: {match.group(0)!r}"


def test_offramp_module_imports_no_razorpay_client():
    """Checks for an actual import statement, not a bare substring -- this
    module's own docstring discusses Razorpay in prose, which a naive
    substring check would trip on."""
    import re

    path = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "policy" / "offramp.py"
    text = path.read_text(encoding="utf-8")
    razorpay_import = re.compile(r"^\s*(?:import|from)\s+razorpay\b", re.MULTILINE)
    assert razorpay_import.search(text) is None


def test_dataclasses_are_frozen():
    from dataclasses import FrozenInstanceError

    import pytest

    offer = construct_offer(_belief(), _ctx())
    with pytest.raises((FrozenInstanceError, AttributeError)):
        offer.mandate_id = "other"  # type: ignore
    with pytest.raises((FrozenInstanceError, AttributeError)):
        offer.steps[0].kind = "OTHER"  # type: ignore
