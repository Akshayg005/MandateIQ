"""Pause, then downgrade, then cancel -- constructed as an OFFER the
customer can choose from, never executed by this system.

Invariant 6: the system never cancels a mandate; it only ever offers an
off-ramp. This is the one file scripts/guard_invariants.py exempts from the
HARD_CANCEL check (see that script's OFFRAMP_OK constant), and only so this
module can NAME the last step of the offer as a menu item -- it must never
itself call a cancellation endpoint. There is no Razorpay client import
here at all; that would be a stronger guarantee of invariant 6 than a text
guard could ever be, and this module has no need of one.

Design spec (the build spec section 4): Q(b, OFFER) = offer_value(b, ctx),
"deferred revenue, not lost: pause/downgrade retain LTV" -- constructing an
offer is not a concession of lost revenue, it is this project's thesis in
one sentence: a customer kept at a lower commitment is worth more than a
customer ground down to zero by a retry ladder that cannot tell the
difference between "can't pay now" and "wants out."

allocator.py only calls construct_offer() once OFFER has already been
chosen -- which requires a singleton {WONT_PAY} conformal prediction set
under a real gate (never true under the B8 default FullSetGate stub, see
src/policy/gate.py). This module does not re-decide whether to offer, only
what the offer contains.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.policy.belief import Belief
from src.policy.stopping_rules import AllocationContext

# Order matters: least drastic, most reversible step first. A customer who
# only needed a pause is never shown "cancel" as the headline option.
_STEP_ORDER: tuple[tuple[str, str], ...] = (
    ("PAUSE", "Pause this mandate for one cycle -- no charge, nothing cancelled."),
    ("DOWNGRADE", "Switch to a lower amount or a longer billing interval."),
    ("CANCEL", "End this mandate. The customer's own choice, not ours to make."),
)

# How long the offer stays valid before the cycle's normal attempt budget
# would have run out anyway. Not an RBI clause -- a UX constant, disclosed
# as a placeholder the same way config/policy_costs.yaml's tuning
# parameters are.
_OFFER_VALIDITY_DAYS = 14


@dataclass(frozen=True)
class OffRampStep:
    kind: str
    description: str


@dataclass(frozen=True)
class Offer:
    """steps: PAUSE, then DOWNGRADE, then CANCEL, in that order -- an
    ordered menu presented to the customer, never a single ultimatum.
    belief_json: the belief that triggered this offer, carried through for
    audit (mirrors Plan.belief_json). expires_on_day: day-index after which
    this specific offer is no longer presented -- a fresh solve() call
    would construct a new one if the mandate is still live and still
    routes to OFFER."""

    mandate_id: str
    cycle_id: int
    steps: tuple[OffRampStep, ...]
    belief_json: str
    expires_on_day: int


def construct_offer(b: Belief, ctx: AllocationContext) -> Offer:
    """Build the pause/downgrade/cancel offer for a mandate the allocator
    has already decided to route to OFFER. Never calls a cancellation
    endpoint and never mutates mandate state -- this function only
    describes what the customer could choose from; the choice itself is
    made elsewhere (the customer's own response to the offer), never here.
    The customer decides.
    """
    steps = tuple(OffRampStep(kind=kind, description=desc) for kind, desc in _STEP_ORDER)
    return Offer(
        mandate_id=ctx.mandate_id,
        cycle_id=ctx.cycle_id,
        steps=steps,
        belief_json=b.to_json(),
        expires_on_day=ctx.plan_day + _OFFER_VALIDITY_DAYS,
    )
