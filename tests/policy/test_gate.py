"""src/policy/gate.py -- ConformalGate Protocol and the FullSetGate stub.

Design spec: FullSetGate returns all three causes for every belief, which
under the singleton-only offer rule (src.model.conformal.should_act) means
OFFER_OFFRAMP degrades to permanently infeasible. That is the safe
direction for a stub -- proven here, not just claimed.
"""
from __future__ import annotations

from src.core.types import Cause
from src.model.conformal import should_act
from src.policy import belief as belief_mod
from src.policy.gate import ConformalGate, FullSetGate


def _some_belief() -> belief_mod.Belief:
    return belief_mod.init({c: 1.0 / 3.0 for c in Cause})


def _degenerate_belief(dominant: Cause) -> belief_mod.Belief:
    """A belief that is (almost) certain on one cause -- the case most
    likely to tempt a real gate into a singleton, so the strongest possible
    input to prove FullSetGate ignores belief entirely."""
    probs = {c: 0.000001 for c in Cause}
    probs[dominant] = 0.999998
    return belief_mod.init(probs)


# === Protocol conformance ===================================================

def test_full_set_gate_satisfies_the_protocol():
    assert isinstance(FullSetGate(), ConformalGate), \
        "FullSetGate does not satisfy ConformalGate Protocol"


# === FullSetGate behaviour ===================================================

def test_full_set_gate_returns_all_three_causes():
    gate = FullSetGate()
    result = gate.pred_set(_some_belief())
    assert result == frozenset(Cause), \
        f"FullSetGate.pred_set returned {result}, expected all three causes"


def test_full_set_gate_ignores_even_a_near_certain_belief():
    """The one input that would tempt a real conformal predictor into a
    singleton -- FullSetGate must return the full set regardless."""
    gate = FullSetGate()
    for dominant in Cause:
        result = gate.pred_set(_degenerate_belief(dominant))
        assert result == frozenset(Cause), \
            f"FullSetGate.pred_set({dominant}) returned {result}, expected all three"


def test_full_set_gate_never_fires_the_offramp():
    """The actual safety property: should_act() on FullSetGate's output is
    always False, for every possible target cause, on every belief --
    OFFER_OFFRAMP is structurally unreachable under this stub."""
    gate = FullSetGate()
    for dominant in list(Cause) + [None]:
        b = _degenerate_belief(dominant) if dominant else _some_belief()
        pred_set = gate.pred_set(b)
        for target in Cause:
            assert should_act(pred_set, target) is False, \
                f"should_act fired for target={target} on FullSetGate's output {pred_set}"
