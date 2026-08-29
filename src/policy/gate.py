"""The off-ramp gate as an interface -- ConformalGate Protocol plus a safe
stub default, FullSetGate.

Design spec (PLAN_DETAIL.md section 8.1, decision 3): the off-ramp ships
behind a protocol-typed gate so B8 does not depend on B6's real conformal
predictor landing first. FullSetGate returns the full three-cause set for
every belief, which under this project's firing rule -- offer only on the
singleton {WONT_PAY} -- degrades to "never offer an off-ramp." That is the
safe direction: the stub's failure mode is declining to offer, never a
false positive, so a B6 slip costs the off-ramp lane, not the allocator.

src.model.conformal.should_act(s, target) already implements the exact
singleton-set firing rule this gate exists to feed -- allocator.py calls it
against whatever frozenset a ConformalGate produces, so "offer only on a
singleton" is defined in exactly one place in this codebase, not
reimplemented here.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types import Cause
from src.policy.belief import Belief


@runtime_checkable
class ConformalGate(Protocol):
    """A callable mapping a Belief to the conformal prediction set over the
    three causes, at this gate's own coverage level. frozenset() (abstain)
    is a legal return -- src.model.conformal.SplitConformal.pred_set()
    already returns frozenset() to signal abstention, and this Protocol
    must accept whatever a real conformal predictor can return."""

    def pred_set(self, b: Belief) -> frozenset[Cause]:
        ...


class FullSetGate:
    """The B8 default. Always returns all three causes, regardless of `b` --
    never a singleton, so OFFER_OFFRAMP is never feasible under
    allocator.py's action set and the policy degrades to the retry lane.
    Satisfies ConformalGate structurally (isinstance(FullSetGate(),
    ConformalGate) is True) without depending on B6's fitted predictor ever
    being wired in."""

    def pred_set(self, b: Belief) -> frozenset[Cause]:
        return frozenset(Cause)
