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

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.core.types import Cause
from src.model.conformal import lac_scores
from src.policy.belief import Belief

if TYPE_CHECKING:
    from src.model.conformal import SplitConformal


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


class ConformalCauseGate:
    """The real off-ramp gate: a split-conformal predictor over the three
    causes, wrapping a SplitConformal fitted on its own calib_conf split.

    B8 shipped FullSetGate as the safe default, whose failure mode is
    declining to offer. This is the gate that can actually fire, and it is
    deliberately the only place where a Belief becomes a prediction SET
    rather than an argmax. allocator.py then applies conformal.should_act()
    -- offer only on the singleton {WONT_PAY} -- so the firing rule stays
    defined in exactly one place.

    Why a cause-level predictor and not the terminal-Outcome one that
    eval/model_fit_report.py calibrates: the off-ramp decision is about WHY
    the mandate is failing, not about what happens at the next attempt. The
    four-way Outcome predictor answers a different question and its
    singleton sets would mean something else entirely.

    `key` for the smoothed p-value is derived from the belief itself rather
    than supplied by the caller. Two identical beliefs must produce the same
    prediction set -- an off-ramp that fires on a coin flip is not a gate --
    and there is no stable per-decision id available at this layer that is
    not also a leak of mandate identity into the score.
    """

    def __init__(self, predictor: "SplitConformal[Cause]") -> None:
        self._predictor = predictor

    @property
    def predictor(self) -> "SplitConformal[Cause]":
        return self._predictor

    @staticmethod
    def _key(b: Belief) -> str:
        return hashlib.sha256(
            (";".join(f"{p:.12f}" for p in b.probs)).encode()
        ).hexdigest()[:16]

    def pred_set(self, b: Belief) -> frozenset[Cause]:
        scores = lac_scores(np.asarray([b.probs], dtype=float))[0]
        return self._predictor.pred_set(scores, key=self._key(b))
