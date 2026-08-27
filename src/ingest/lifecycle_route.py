"""Routes Razorpay subscription-family webhook events into mandate_lifecycle
-- the event route RBI clause 6(c) requires: an opt-out or revocation riding
on the T-24h notification has to land somewhere the executor will read.
Recording the transition is this file's job; acting on it (the executor's
pre-call re-read, honouring an opt-out inside the 24h window) is B9's.

Keyed off the subscription entity's own `status`, never the event *name* --
a future Razorpay event this code has never heard of degrades gracefully to
"unmapped, write nothing" rather than raising, because the mapping only
ever inspects the status string.

The Razorpay status vocabulary below is the real, documented set
(independently verified against Razorpay's subscription-states page and the
pause-subscription API's own sample response, 2026-08-27) -- not guessed.
Three of the nine real statuses are deliberately left UNMAPPED, each for a
distinct, disclosed reason rather than an oversight:

- `authenticated` -- a pre-billing state (customer completed authentication,
  billing hasn't started) with no equivalent in this project's simpler
  6-state MandateState model.
- `pending` -- mid-retry: Razorpay's OWN auto-retry mechanism is between
  attempts, not evidence about the mandate itself.
- `halted` -- means Razorpay's own retry budget is exhausted, not that the
  underlying mandate/UPI-AutoPay authorisation is confirmed dead. Reading
  it as REVOKED or EXPIRED would be a guess this file's design explicitly
  refuses to make (see decline_taxonomy.py's identical UNKNOWN principle).

record() returning None and writing nothing is the correct, honest
behaviour for all three -- and for any status this table has never seen.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.core.types import MandateState
from src.ledger.store import record_lifecycle_event

_STATUS_MAP: dict[str, MandateState] = {
    "created": MandateState.CREATED,
    "active": MandateState.ACTIVE,
    "paused": MandateState.PAUSED,
    # This project's clause-6(c) vocabulary calls a dead/cancelled mandate
    # "REVOKED" -- Razorpay's own word for the same transition is "cancelled".
    "cancelled": MandateState.REVOKED,
    "expired": MandateState.EXPIRED,
    "completed": MandateState.COMPLETED,
    # authenticated / pending / halted: deliberately absent -- see module
    # docstring. Any other status Razorpay ever introduces is equally
    # absent, and gets the same honest non-guess.
}


def record(conn, event_id: str, payload: dict) -> MandateState | None:
    """Map a subscription-family webhook envelope's status to a
    MandateState and record it. Returns the recorded MandateState, or None
    (writing nothing) if the status has no mapping.

    effective_at comes ONLY from the envelope's top-level `created_at`
    (Unix seconds -- when Razorpay generated this event, i.e. when the
    transition happened at Razorpay's end), never from src.core.clock.now().
    Freezing our test clock must never change what Razorpay claims
    happened; this is a deliberate, tested design point, not an incidental
    detail.
    """
    entity = payload["payload"]["subscription"]["entity"]
    mapped = _STATUS_MAP.get(entity["status"])
    if mapped is None:
        return None

    effective_at = datetime.fromtimestamp(payload["created_at"], tz=timezone.utc)
    return record_lifecycle_event(
        conn,
        event_id=event_id,
        mandate_id=entity["id"],
        state=mapped.value,
        source="WEBHOOK",
        effective_at=effective_at,
    )
