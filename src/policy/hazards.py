"""Protocol for a cause-conditioned hazard: the four Outcome probabilities
for one (cause, slot, day, amount) combination.

PLAN_DETAIL.md section 4's Q(b, ATTEMPT(d,m)) sums over causes using
exactly this shape -- h_rec(c,d,m,ctx), h_opt(c,d,m,ctx), h_dead(c,d,m,ctx),
each conditioned on a specific cause c. But B5 (the model-fit phase)
shipped hazards MARGINAL over cause: competing_risks.hazards() fits and
returns one set of four outcome probabilities per (mandate, slot), with no
per-cause conditioning anywhere in its signature. It cannot be otherwise --
Cause is latent and has NO PRODUCTION LABEL, ever (DECISIONS.md,
2026-08-28, B6), so there is no ground-truth column any model could be fit
against to produce P(outcome | cause, slot, day, amount) directly.

This file therefore defines ONLY a Protocol -- a type declaration, not an
implementation. It exists so B8's allocator must name its hazard source in
the type system: accepting a CauseConditionedHazard makes substituting
B5's cause-marginal hazards an explicit, visible act, rather than a silent
default nobody has to declare. See reports/gates.md's B8 entry and
DECISIONS.md, 2026-08-29, B7 for the full gap, and for the open question
left to B8: whether it resolves this by fitting something new, or by
having Cause enter only through gating which actions are legal rather than
the hazard arithmetic itself.

No implementation lives here. If this file starts acquiring one, that is
B8's work leaking into B7 -- stop and raise it rather than continuing.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types import Cause


@runtime_checkable
class CauseConditionedHazard(Protocol):
    """A callable taking a specific cause, slot, commit day, and amount,
    returning the four Outcome probabilities in Outcome int order --
    STILL_PENDING, RECOVERED, DEAD, OPTED_OUT -- the same convention
    src/model/cif.py and competing_risks.hazards() already use. Summing to
    1 is the implementation's responsibility, not this Protocol's: a
    Protocol declares a shape, it cannot check a postcondition."""

    def __call__(
        self, *, cause: Cause, slot: int, on_day: int, amount_paise: int
    ) -> tuple[float, float, float, float]:
        ...
