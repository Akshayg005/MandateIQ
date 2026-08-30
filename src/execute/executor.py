"""Ledger-write-then-money, in that order, exactly per PLAN_DETAIL.md
section 3's write-ordering protocol and CLAUDE.md invariant 3: the ledger
write happens BEFORE the money action, never after.

THE LATE-READ PRINCIPLE (PLAN_DETAIL.md section 1, restated here because a
future session that forgets the asymmetry will either debit a revoked
mandate or reintroduce reactive retry): clause 6(a) constrains committing
an attempt ahead of time; clause 6(c) REQUIRES that an opt-out arriving
inside that window be honoured. Reading lifecycle state late in order to
STOP is therefore always permitted; reading it late in order to ACT --
advance a date, add an attempt, raise an amount -- is not. execute()'s
pre-call check (step 2a below) has exactly one outcome available to it:
abort. It never moves attempt.scheduled_for, never changes
attempt.amount_paise, never adds a retry.

The five steps, matching PLAN_DETAIL.md section 3 one-to-one:

  1. INTENT row (ON CONFLICT DO NOTHING). 0 rows -> this attempt already
     exists -> jump straight to reflecting its current state, never to (3).
  2. Claim the lease. Lost the race -> abort, do not send.
  2a. THE ONE LATE READ. Re-read mandate_lifecycle; re-check the hard
      stopping rules against a real scheduled_for (the first live exercise
      of stopping_rules.py's quiet-hours check -- B8's own call sites never
      had a real timestamp). Either abort path voids the schedule row,
      because in THIS process, at THIS moment, holding the lease, having
      just written the INTENT row ourselves, we know first-hand no
      Razorpay call has been made (src/execute/void.py's SENT-row rule).
  3. SENT (forensic only -- schema.sql's own comment), then the actual
     call. The crash window is here and is unavoidable -- no protocol
     removes it, which is why recover.py exists rather than a retry
     counter.
  4. RESULT, from what the provider reports. Lease released.
  5. (recover.py, not this module) reconciles whatever (1)-(4) left
     dangling.

A RazorpayClientError from the call (step 3) -- ambiguous: we do not know
whether the request reached the provider -- deliberately does NOT release
the lease and does NOT write a further ledger row. It is left to expire
naturally so recover.py's uniform expired-lease scan discovers it exactly
the way it would discover a hard process crash; the two cases are the same
category of ambiguity and get the same resolution mechanism (ask the
provider, never guess).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.classify.decline_taxonomy import classify as classify_decline
from src.core.types import Action, DeclineClass, LedgerState, MandateState, Outcome
from src.execute import lease
from src.execute.keys import ScheduledAttempt
from src.execute.razorpay_client import RazorpayClientError, RazorpayDeclined, RazorpayLike
from src.execute.void import void
from src.ledger.store import LedgerEntry, append, find_by_key, latest_state
from src.policy.stopping_rules import AllocationContext, Verdict, permitted

# Which mandate_lifecycle states make an ATTEMPT illegal to send, discovered
# late. REVOKED/EXPIRED: no instrument to charge. PAUSED: the customer (or
# this system's own offramp) has already asked for no further debits this
# cycle -- charging through a pause would defeat the entire off-ramp thesis.
_TERMINAL_LIFECYCLE_STATES = frozenset(
    {MandateState.REVOKED, MandateState.PAUSED, MandateState.EXPIRED}
)

ABORT_REASON_LEASE_LOST = "ABORTED_LEASE_LOST"

# Declines that confirm the instrument itself is dead -- Outcome.DEAD, the
# person-period terminal state src/model/ trains against. Everything else
# recognised (INSUFFICIENT_FUNDS, ISSUER_DECLINE, BANK_TIMEOUT) or not
# (UNKNOWN) is Outcome.STILL_PENDING: retriable, cause still unresolved.
# NEW domain judgment authored at B9, a different axis from
# src.classify.cause_map.prior() (DeclineClass -> a Cause belief-update
# prior) -- this maps to Outcome, the ledger's own terminal-state column.
# Not independently reviewed the way cause_map.py's mapping was at B3;
# flagged for a payments-domain or stats-reviewer pass, not presented as
# already vetted.
_DEAD_DECLINE_CLASSES = frozenset(
    {DeclineClass.MANDATE_REVOKED, DeclineClass.CARD_EXPIRED, DeclineClass.ACCOUNT_CLOSED}
)

# A synchronous charge response's `status` values this module recognises as
# an immediate, confirmed success. Any other status (including a genuinely
# pending/async one) is recorded honestly as unresolved rather than
# guessed -- see execute()'s docstring and razorpay_client.py's own
# disclosure that charge()'s response shape is not independently spiked.
SUCCESS_STATUSES = frozenset({"captured", "authorized"})

PENDING_WEBHOOK_CONFIRMATION = "PENDING_WEBHOOK_CONFIRMATION"


def abort_reason_lifecycle(state: MandateState) -> str:
    return f"ABORTED_LIFECYCLE_{state.value}"


def abort_reason_stopping_rule(action: Action) -> str:
    return f"ABORTED_STOPPING_RULE_{action.value}"


def _outcome_for_decline(dc: DeclineClass) -> Outcome:
    if dc in _DEAD_DECLINE_CLASSES:
        return Outcome.DEAD
    return Outcome.STILL_PENDING


@dataclass(frozen=True)
class Result:
    """What execute() and recover.py both return -- the ledger row's own
    shape, not a richer domain object, so a caller can tell exactly what
    was (or was not) durably recorded."""

    idempotency_key: str
    state: str
    outcome: str | None
    decline_class: str | None
    provider_ref: str | None
    reason: str | None


def _payload_sha256(attempt: ScheduledAttempt) -> str:
    """A canonical hash of what THIS attempt is -- recorded on the INTENT
    row so a later audit can confirm what was intended before anything was
    sent. Deliberately a small local hash rather than reusing
    src.core.ids.decision_sha256: that function hashes a Plan's DECISION
    content, a different thing from an attempt's own payload, and reusing
    it under this name would misname what is actually being hashed."""
    canonical = json.dumps(
        {
            "mandate_id": attempt.mandate_id,
            "cycle_id": attempt.cycle_id,
            "attempt_index": attempt.attempt_index,
            "generation": attempt.generation,
            "action": attempt.action,
            "amount_paise": attempt.amount_paise,
            "decision_sha256": attempt.decision_sha256,
            "scheduled_for": attempt.scheduled_for.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def result_from_row(row) -> Result:
    return Result(
        idempotency_key=row.idempotency_key,
        state=row.state,
        outcome=row.outcome,
        decline_class=row.decline_class,
        provider_ref=row.provider_ref,
        reason=row.reason,
    )


def _append(conn, attempt: ScheduledAttempt, payload_sha256: str, *, state: str, **fields) -> Result:
    append(
        conn,
        LedgerEntry(
            idempotency_key=attempt.idempotency_key,
            mandate_id=attempt.mandate_id,
            cycle_id=attempt.cycle_id,
            attempt_index=attempt.attempt_index,
            action=attempt.action,
            state=state,
            amount_paise=attempt.amount_paise,
            profile=attempt.profile,
            payload_sha256=payload_sha256,
            decision_sha256=attempt.decision_sha256,
            **fields,
        ),
    )
    return Result(
        idempotency_key=attempt.idempotency_key,
        state=state,
        outcome=fields.get("outcome"),
        decline_class=fields.get("decline_class"),
        provider_ref=fields.get("provider_ref"),
        reason=fields.get("reason"),
    )


def execute(
    conn,
    client: RazorpayLike,
    attempt: ScheduledAttempt,
    ctx: AllocationContext,
    *,
    owner: str,
    lease_ttl_seconds: int = 300,
) -> Result:
    """Execute one committed attempt. See module docstring for the five
    steps this follows exactly. Only ever sends an ATTEMPT action --
    REAUTH/OFFER/STOP never produce a committed_schedule row in the first
    place (src.execute.commit.commit()'s own chosen_action gate), so
    reaching this function with anything else is an upstream contract
    violation, raised loudly rather than silently handled."""
    if attempt.action != Action.ATTEMPT.value:
        raise ValueError(
            f"execute() only sends ATTEMPT actions; got {attempt.action!r} "
            f"for {attempt.idempotency_key}"
        )

    payload_sha256 = _payload_sha256(attempt)

    # --- step 1: INTENT, ON CONFLICT DO NOTHING -----------------------------
    ledger_id = append(
        conn,
        LedgerEntry(
            idempotency_key=attempt.idempotency_key,
            mandate_id=attempt.mandate_id,
            cycle_id=attempt.cycle_id,
            attempt_index=attempt.attempt_index,
            action=attempt.action,
            state=LedgerState.INTENT.value,
            amount_paise=attempt.amount_paise,
            profile=attempt.profile,
            payload_sha256=payload_sha256,
            decision_sha256=attempt.decision_sha256,
        ),
    )
    if ledger_id is None:
        # 0 rows -> this attempt already exists -> reflect it, never send.
        existing = find_by_key(conn, attempt.idempotency_key)
        return result_from_row(existing)

    # --- step 2: claim the lease --------------------------------------------
    if not lease.claim(conn, attempt.idempotency_key, owner=owner, ttl_seconds=lease_ttl_seconds):
        # We just wrote INTENT ourselves (ledger_id above), so by
        # construction no one else can hold a legitimate lease on this
        # exact key yet -- a defensive path per PLAN_DETAIL.md section 3,
        # not one normal operation should reach. We never held the lease,
        # so nothing to release; the row is not voided, since whoever DOES
        # hold it may still be legitimately processing it.
        return _append(conn, attempt, payload_sha256, state=LedgerState.FAILED.value, reason=ABORT_REASON_LEASE_LOST)

    # --- step 2a: THE ONE LATE READ. Exactly one legal outcome: abort. -----
    lifecycle_state = latest_state(conn, attempt.mandate_id)
    if lifecycle_state in _TERMINAL_LIFECYCLE_STATES:
        reason = abort_reason_lifecycle(lifecycle_state)
        void(conn, attempt.idempotency_key, reason=reason)
        result = _append(conn, attempt, payload_sha256, state=LedgerState.FAILED.value, reason=reason)
        lease.release(conn, attempt.idempotency_key)
        return result

    verdict = permitted(Action(attempt.action), ctx, at=attempt.scheduled_for)
    if verdict == Verdict.DENY:
        reason = abort_reason_stopping_rule(Action(attempt.action))
        void(conn, attempt.idempotency_key, reason=reason)
        result = _append(conn, attempt, payload_sha256, state=LedgerState.FAILED.value, reason=reason)
        lease.release(conn, attempt.idempotency_key)
        return result

    # --- step 3: SENT (forensic only), then the call. Crash window here. --
    append(
        conn,
        LedgerEntry(
            idempotency_key=attempt.idempotency_key,
            mandate_id=attempt.mandate_id,
            cycle_id=attempt.cycle_id,
            attempt_index=attempt.attempt_index,
            action=attempt.action,
            state=LedgerState.SENT.value,
            amount_paise=attempt.amount_paise,
            profile=attempt.profile,
            payload_sha256=payload_sha256,
            decision_sha256=attempt.decision_sha256,
        ),
    )

    try:
        response = client.charge(
            amount_paise=attempt.amount_paise,
            receipt=attempt.idempotency_key,
            notes={
                "mandate_id": attempt.mandate_id,
                "cycle_id": str(attempt.cycle_id),
                "attempt_index": str(attempt.attempt_index),
            },
        )
    except RazorpayDeclined as exc:
        # A DEFINITIVE, synchronous rejection -- the provider received the
        # request and stated a reason. Safe to record as a known outcome.
        decline_class = classify_decline(None, str(exc))
        result = _append(
            conn, attempt, payload_sha256, state=LedgerState.RESULT.value,
            outcome=_outcome_for_decline(decline_class).name,
            decline_class=decline_class.value,
        )
        lease.release(conn, attempt.idempotency_key)
        return result
    except RazorpayClientError:
        # AMBIGUOUS -- we do not know if this reached the provider. See
        # module docstring: leave SENT as the last row, do not release the
        # lease, let recover.py's expired-lease scan resolve it by asking.
        return Result(
            idempotency_key=attempt.idempotency_key, state=LedgerState.SENT.value,
            outcome=None, decline_class=None, provider_ref=None, reason=None,
        )

    # --- step 4: RESULT, from what the provider reports. Release lease. ---
    provider_ref = response.get("id")
    status = response.get("status")
    if status in SUCCESS_STATUSES:
        result = _append(
            conn, attempt, payload_sha256, state=LedgerState.RESULT.value,
            provider_ref=provider_ref, outcome=Outcome.RECOVERED.name,
        )
    else:
        # A response came back but not one of the statuses this module
        # recognises as a synchronous success -- e.g. genuinely pending
        # settlement, whose definitive answer arrives later via the B3
        # webhook path. Recorded honestly as unresolved, not guessed.
        result = _append(
            conn, attempt, payload_sha256, state=LedgerState.RESULT.value,
            provider_ref=provider_ref, reason=PENDING_WEBHOOK_CONFIRMATION,
        )
    lease.release(conn, attempt.idempotency_key)
    return result
