"""The off-ramp gate as an interface -- ConformalGate Protocol plus a safe
stub default, FullSetGate.

Design spec (the build spec section 8.1, decision 3): the off-ramp ships
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

    **The smoothing key must be a per-DECISION id, never a function of the
    belief.** The first version of this class hashed `b.probs`, reasoning
    that two identical beliefs must give the same set. That reasoning is
    right and the implementation of it was wrong, in a way that silently
    destroyed the coverage guarantee:

    Vovk's smoothed p-value is valid because the tie-breaking draw `u` is
    independent of the score and redrawn per row, so coverage is 1-alpha
    after averaging over `u`. Keying on the score makes `u` a deterministic
    function of it, so every row with the same belief shares one `u` and
    there is nothing left to average. With a belief space this small the
    whole study ran on two `u` values, and for WONT_PAY -- whose calibration
    pool is a single tie atom -- the p-value reduced to `u` exactly. The
    inclusion decision for the one class that can fire the off-ramp was
    literally a hash of a constant string, with no data dependence, and the
    reported coverage was one draw from a range spanning 0.105 to 0.980
    (the statistics review, 2026-08-31; DECISIONS.md).

    So the key is now supplied by the caller via `bind()`, as
    `conformal.SplitConformal.pred_set()` documents ("a stable per-row id,
    e.g. mandate_id"). Determinism is preserved -- the same mandate at the
    same slot always gets the same set -- and `u` no longer depends on the
    score. Binding is REQUIRED: an unbound gate raises rather than falling
    back to a key that would reintroduce the bug quietly.
    """

    def __init__(self, predictor: "SplitConformal[Cause]", key: str | None = None) -> None:
        self._predictor = predictor
        self._key = key

    @property
    def predictor(self) -> "SplitConformal[Cause]":
        return self._predictor

    @property
    def key(self) -> str | None:
        return self._key

    def bind(self, key: str) -> "ConformalCauseGate":
        """A view of this gate for one decision point. Shares the fitted
        predictor -- binding is cheap and must not refit anything."""
        if not key:
            raise ValueError("ConformalCauseGate.bind() requires a non-empty key")
        return ConformalCauseGate(self._predictor, key=key)

    def pred_set(self, b: Belief) -> frozenset[Cause]:
        if self._key is None:
            raise ValueError(
                "ConformalCauseGate was queried without a bound key. Call "
                "gate.bind(<per-decision id>) first -- see this class's "
                "docstring for why the key may not be derived from the belief."
            )
        scores = lac_scores(np.asarray([b.probs], dtype=float))[0]
        return self._predictor.pred_set(scores, key=self._key)
