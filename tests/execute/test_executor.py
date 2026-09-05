"""src/execute/executor.py -- ledger-write-then-money, and the late-read
principle. This file's centre of gravity is the B9 gate's hardest clause:
an opt-out arriving inside the 24h window must be honoured, PROVEN by a
test that actively constructs the race, not merely one that happens never
to generate a late opt-out.

Every race/abort test below ships as a pair: a POSITIVE case (the abort
condition is present) and, where the risk of a vacuous pass is real, a
NEGATIVE CONTROL (the identical setup minus that one condition) that MUST
reach the provider double. If both branches aborted, the positive result
would prove nothing about the mechanism under test -- it would only prove
the test's own setup denies everything, which is exactly the failure mode
DECISIONS.md's 2026-08-29 vacuous-checks audit found and fixed twice
already in this project (B8's discrimination clause). The negative
control's provider double raises AssertionError if it is asked to charge
and fails to receive the call it expects, so a wrongly-aborting negative
control fails LOUDLY, not silently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import clock
from src.core.types import Action, MandateState, Profile
from src.execute.commit import commit
from src.execute.executor import (
    ABORT_REASON_LEASE_LOST,
    PENDING_WEBHOOK_CONFIRMATION,
    Result,
    execute,
)
from src.execute.lease import claim as lease_claim
from src.execute.razorpay_client import RazorpayClientError, RazorpayDeclined
from src.ledger.store import LedgerEntry, append, record_lifecycle_event, replay
from src.policy.allocator import CommittedAttempt, Plan
from src.policy.stopping_rules import AllocationContext

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)  # hour 10: outside quiet hours


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


# --- fakes -------------------------------------------------------------------

class _FakeClient:
    """Records every call it receives. `charge_response` is returned on
    success; `charge_exception`, if set, is raised instead."""

    def __init__(self, charge_response=None, charge_exception=None):
        self.calls: list[tuple] = []
        self._charge_response = charge_response or {"id": "pay_default", "status": "captured"}
        self._charge_exception = charge_exception

    def create_order(self, *, amount_paise, receipt, notes):
        raise AssertionError("execute() must call charge(), never create_order()")

    def charge(self, *, amount_paise, receipt, notes):
        self.calls.append(("charge", amount_paise, receipt, notes))
        if self._charge_exception is not None:
            raise self._charge_exception
        return self._charge_response

    def pause_subscription(self, subscription_id):
        raise AssertionError("execute() must never call pause_subscription()")

    def find_by_receipt(self, receipt):
        raise AssertionError("execute() must never call find_by_receipt() -- that is recover.py's job")


class _MustNotChargeClient(_FakeClient):
    """For every abort test: proves the provider was never reached, not
    merely that the function returned a FAILED-looking Result."""

    def charge(self, *, amount_paise, receipt, notes):
        raise AssertionError(
            "execute() called charge() on an attempt that should have aborted "
            "before ever reaching the provider"
        )


def _ctx(**overrides) -> AllocationContext:
    base = dict(
        mandate_id="M-EXEC-1",
        cycle_id=1,
        profile=Profile.strict,
        amount_paise=50_000,
        ceiling_paise=200_000,
        category="subscription",
        plan_day=0,
        attempts_used=1,
        committed_days=(),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=4,
        quiet_hours_start=21,
        quiet_hours_end=8,
    )
    base.update(overrides)
    return AllocationContext(**base)


def _plan(*, decision_sha256, mandate_id="M-EXEC-1", on_day=2, amount_paise=50_000):
    return Plan(
        mandate_id=mandate_id, cycle_id=1, profile=Profile.strict,
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=on_day, amount_paise=amount_paise),),
        belief_json="{}", conformal_set=frozenset(), binding_constraint=None,
        solver_version="test-solver-v0", decision_sha256=decision_sha256,
    )


def _mandate_created(conn, mandate_id: str, *, at: datetime) -> None:
    """Every real mandate has a CREATED row (store.latest_state raises
    LookupError otherwise) -- establishes the mandate as ACTIVE as of plan
    time, mirroring how a real mandate's lifecycle starts."""
    record_lifecycle_event(
        conn, event_id=f"evt-created-{mandate_id}", mandate_id=mandate_id,
        state=MandateState.ACTIVE.value, source="INTERNAL", effective_at=at,
    )


def _committed(pg_schema, *, decision_sha256, mandate_id="M-EXEC-1", on_day=2, amount_paise=50_000, committed_at):
    clock.set_frozen(committed_at)
    _mandate_created(pg_schema.conn, mandate_id, at=committed_at - timedelta(days=1))
    plan = _plan(decision_sha256=decision_sha256, mandate_id=mandate_id, on_day=on_day, amount_paise=amount_paise)
    return commit(pg_schema.conn, plan, cycle_start=CYCLE_START)


# === THE GATE'S CENTREPIECE: the 6(c) race, as a discriminating pair =======

def test_late_optout_inside_the_window_aborts_the_attempt(pg_schema):
    """POSITIVE: commit an attempt, advance the frozen clock into the 24h
    window, deliver a REVOKED lifecycle event with effective_at inside
    that window, then execute(). The provider must NEVER be reached."""
    attempt = _committed(pg_schema, decision_sha256="d-race-positive", committed_at=CYCLE_START)

    # Inside the window: scheduled_for minus 2 hours.
    inside_window = attempt.scheduled_for - timedelta(hours=2)
    record_lifecycle_event(
        pg_schema.conn, event_id="evt-late-optout-positive", mandate_id=attempt.mandate_id,
        state=MandateState.REVOKED.value, source="WEBHOOK", effective_at=inside_window,
    )
    clock.set_frozen(inside_window)

    client = _MustNotChargeClient()
    ctx = _ctx(mandate_id=attempt.mandate_id)

    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-a")

    assert result.state == "FAILED"
    assert result.reason == "ABORTED_LIFECYCLE_REVOKED"
    assert client.calls == []

    rows = replay(pg_schema.conn, attempt.mandate_id)
    states = [r.state for r in rows if r.idempotency_key == attempt.idempotency_key]
    assert "SENT" not in states, "the attempt must never reach the provider once opted out"
    assert states == ["INTENT", "FAILED"]

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT voided_at FROM committed_schedule WHERE idempotency_key = %s",
            (attempt.idempotency_key,),
        )
        (voided_at,) = cur.fetchone()
    assert voided_at is not None, "the schedule row must be voided once opted out"


def test_negative_control_identical_setup_without_optout_reaches_the_provider(pg_schema):
    """NEGATIVE CONTROL: byte-for-byte the same setup as the positive case
    above, EXCEPT no REVOKED event is ever delivered. This MUST reach
    charge() -- if it also aborts, the positive test above proves nothing
    about the lifecycle mechanism specifically; it would only prove this
    setup denies everything."""
    attempt = _committed(pg_schema, decision_sha256="d-race-negative", committed_at=CYCLE_START)

    inside_window = attempt.scheduled_for - timedelta(hours=2)
    clock.set_frozen(inside_window)
    # No REVOKED event delivered -- mandate stays ACTIVE (from _mandate_created).

    client = _FakeClient(charge_response={"id": "pay_control", "status": "captured"})
    ctx = _ctx(mandate_id=attempt.mandate_id)

    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-a")

    assert len(client.calls) == 1, "the negative control must reach the provider"
    assert result.state == "RESULT"
    assert result.outcome == "RECOVERED"


def test_optout_arriving_after_execution_does_not_retroactively_change_the_result(pg_schema):
    """Sanity bound on the race: an opt-out delivered AFTER a successful
    charge must not be confused with one delivered before it -- the ledger
    is append-only and the RESULT row, once written, stands."""
    attempt = _committed(pg_schema, decision_sha256="d-race-after", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_response={"id": "pay_after", "status": "captured"})
    ctx = _ctx(mandate_id=attempt.mandate_id)
    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-a")
    assert result.state == "RESULT"

    # Opt-out arrives only now, after the fact.
    record_lifecycle_event(
        pg_schema.conn, event_id="evt-late-optout-after", mandate_id=attempt.mandate_id,
        state=MandateState.REVOKED.value, source="WEBHOOK",
        effective_at=clock.now() + timedelta(minutes=5),
    )
    rows = replay(pg_schema.conn, attempt.mandate_id)
    states = [r.state for r in rows if r.idempotency_key == attempt.idempotency_key]
    assert states == ["INTENT", "SENT", "RESULT"], "a completed RESULT must never be rewritten"


# === the other late-read branch: hard stopping rules, re-checked live =====

def test_stopping_rule_denial_aborts_independently_of_lifecycle(pg_schema):
    """Distinct mechanism from the lifecycle race above: the mandate stays
    ACTIVE, but ctx says the contact-frequency cap is already exhausted.
    Proves the executor's SECOND late-read branch (permitted()) fires on
    its own, not only as a side effect of the lifecycle check."""
    attempt = _committed(pg_schema, decision_sha256="d-stopping-rule", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _MustNotChargeClient()
    ctx = _ctx(mandate_id=attempt.mandate_id, contacts_sent=4, max_contacts_per_cycle=4)

    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-a")

    assert result.state == "FAILED"
    assert result.reason == "ABORTED_STOPPING_RULE_ATTEMPT"
    assert client.calls == []

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT voided_at FROM committed_schedule WHERE idempotency_key = %s",
            (attempt.idempotency_key,),
        )
        (voided_at,) = cur.fetchone()
    assert voided_at is not None


# === step 1: already-exists short-circuit ==================================

def test_execute_never_calls_the_provider_when_intent_already_exists(pg_schema):
    """0 rows from the INTENT insert -> jump straight to reflecting the
    existing row, never to (3) -- the build spec section 3's own words."""
    attempt = _committed(pg_schema, decision_sha256="d-already-intent", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    # Simulate another worker already having written INTENT for this key.
    append(pg_schema.conn, LedgerEntry(
        idempotency_key=attempt.idempotency_key, mandate_id=attempt.mandate_id,
        cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
        action=attempt.action, state="INTENT", amount_paise=attempt.amount_paise,
        profile=attempt.profile, payload_sha256="0" * 64,
        decision_sha256=attempt.decision_sha256,
    ))

    client = _MustNotChargeClient()
    ctx = _ctx(mandate_id=attempt.mandate_id)
    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-b")

    assert result.state == "INTENT"
    assert client.calls == []


# === step 2: lease loss ======================================================

def test_execute_aborts_without_voiding_when_lease_is_already_held(pg_schema):
    """A defensive path (the build spec section 3): by construction, we
    just won the unique INTENT insert for this key, so another owner
    holding a live lease on it already is pathological -- but the
    protocol names it explicitly, so it is honoured. Must NOT void the
    schedule row: whoever holds the lease may still legitimately succeed."""
    attempt = _committed(pg_schema, decision_sha256="d-lease-lost", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    assert lease_claim(pg_schema.conn, attempt.idempotency_key, owner="rival", ttl_seconds=3600)

    client = _MustNotChargeClient()
    ctx = _ctx(mandate_id=attempt.mandate_id)
    result = execute(pg_schema.conn, client, attempt, ctx, owner="worker-a")

    assert result.state == "FAILED"
    assert result.reason == ABORT_REASON_LEASE_LOST
    assert client.calls == []

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT voided_at FROM committed_schedule WHERE idempotency_key = %s",
            (attempt.idempotency_key,),
        )
        (voided_at,) = cur.fetchone()
    assert voided_at is None, "the rival may still be legitimately processing this key"


# === step 3/4: the actual call and its three outcome shapes ================

def test_execute_records_recovered_on_a_captured_response(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-success", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_response={"id": "pay_success", "status": "captured"})
    result = execute(pg_schema.conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")

    assert result.state == "RESULT"
    assert result.outcome == "RECOVERED"
    assert result.provider_ref == "pay_success"

    rows = replay(pg_schema.conn, attempt.mandate_id)
    states = [r.state for r in rows if r.idempotency_key == attempt.idempotency_key]
    assert states == ["INTENT", "SENT", "RESULT"]


def test_execute_records_pending_when_status_is_unrecognised(pg_schema):
    """A response the module does not recognise as a synchronous success
    (e.g. genuinely pending settlement) is recorded honestly as
    unresolved, never guessed as either outcome."""
    attempt = _committed(pg_schema, decision_sha256="d-pending", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_response={"id": "pay_pending", "status": "created"})
    result = execute(pg_schema.conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")

    assert result.state == "RESULT"
    assert result.outcome is None
    assert result.reason == PENDING_WEBHOOK_CONFIRMATION
    assert result.provider_ref == "pay_pending"


def test_execute_records_a_definitive_decline_as_dead(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-declined-dead", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_exception=RazorpayDeclined("mandate revoked by customer"))
    result = execute(pg_schema.conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")

    assert result.state == "RESULT"
    assert result.decline_class == "MANDATE_REVOKED"
    assert result.outcome == "DEAD"


def test_execute_records_a_definitive_decline_as_still_pending(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-declined-pending", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_exception=RazorpayDeclined("insufficient funds in account"))
    result = execute(pg_schema.conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")

    assert result.state == "RESULT"
    assert result.decline_class == "INSUFFICIENT_FUNDS"
    assert result.outcome == "STILL_PENDING"


def test_execute_leaves_sent_and_holds_the_lease_on_ambiguous_failure(pg_schema):
    """The crash-equivalence case: a RazorpayClientError means we do NOT
    know if the request reached the provider. Must not write a further
    ledger row, and must NOT release the lease -- recover.py's
    expired-lease scan is the one mechanism that resolves this, exactly
    like a hard process crash."""
    attempt = _committed(pg_schema, decision_sha256="d-ambiguous", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    client = _FakeClient(charge_exception=RazorpayClientError("simulated timeout"))
    result = execute(pg_schema.conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")

    assert result.state == "SENT"
    assert result.outcome is None
    assert result.reason is None

    rows = replay(pg_schema.conn, attempt.mandate_id)
    states = [r.state for r in rows if r.idempotency_key == attempt.idempotency_key]
    assert states == ["INTENT", "SENT"], "no further ledger row on an ambiguous failure"

    # The lease must still be held (not released, not expired) -- a second
    # claim by anyone else must fail right now.
    assert lease_claim(pg_schema.conn, attempt.idempotency_key, owner="someone-else", ttl_seconds=60) is False


# === invariant 3, proven by call order, not by trusting function names ====

def test_intent_row_is_committed_before_the_provider_is_ever_called(pg_schema):
    """the money audit's own standard: verify by reading the actual call
    order, not the function names. The fake client queries the ledger
    table itself, from inside charge(), before returning -- if execute()
    ever called it before the INTENT row was durably committed, this
    assertion fails from inside the provider call itself."""
    attempt = _committed(pg_schema, decision_sha256="d-ordering", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    conn = pg_schema.conn

    class _OrderCheckingClient(_FakeClient):
        def charge(self, *, amount_paise, receipt, notes):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state FROM ledger WHERE idempotency_key = %s ORDER BY ledger_id",
                    (receipt,),
                )
                states_so_far = [r[0] for r in cur.fetchall()]
            assert "INTENT" in states_so_far, (
                "charge() was called before the INTENT row was committed -- "
                "invariant 3 violated"
            )
            return super().charge(amount_paise=amount_paise, receipt=receipt, notes=notes)

    client = _OrderCheckingClient(charge_response={"id": "pay_order", "status": "captured"})
    result = execute(conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")
    assert result.state == "RESULT"
    assert len(client.calls) == 1


def test_sent_row_is_committed_before_the_provider_is_ever_called(pg_schema):
    """The write ordering src/execute/recover.py's NEVER_SENT branch is a
    PROOF about, pinned here rather than asserted there.

    _resolve_never_sent() frees an NPCI slot -- one of only four, ever --
    on the strength of a single implication: charge() is called only after
    the SENT row is durably committed, so no SENT row means no call was
    ever made. That is true of execute() as written. It would silently
    stop being true if step 3's append were ever moved after the call, or
    the call hoisted above it, and nothing else in the suite would notice:
    every other test would still pass, and recovery would begin voiding
    schedule rows for attempts that may have taken money.

    So this reads the actual call order from inside the provider call,
    exactly as the INTENT test above does. Added at B10.
    """
    attempt = _committed(pg_schema, decision_sha256="d-sent-ordering", committed_at=CYCLE_START)
    clock.set_frozen(attempt.scheduled_for - timedelta(hours=2))

    conn = pg_schema.conn

    class _SentOrderCheckingClient(_FakeClient):
        def charge(self, *, amount_paise, receipt, notes):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state FROM ledger WHERE idempotency_key = %s ORDER BY ledger_id",
                    (receipt,),
                )
                states_so_far = [r[0] for r in cur.fetchall()]
            assert "SENT" in states_so_far, (
                "charge() was called before the SENT row was committed. "
                "recover._resolve_never_sent() infers 'no SENT row => never "
                "sent' and frees the NPCI slot on that basis; with this "
                "ordering reversed, that inference is false and recovery can "
                "void a schedule row for an attempt that took money."
            )
            return super().charge(amount_paise=amount_paise, receipt=receipt, notes=notes)

    client = _SentOrderCheckingClient(charge_response={"id": "pay_sent", "status": "captured"})
    result = execute(conn, client, attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")
    assert result.state == "RESULT"
    assert len(client.calls) == 1


# === contract guard ==========================================================

def test_execute_rejects_a_non_attempt_action(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-non-attempt", committed_at=CYCLE_START)
    from dataclasses import replace as dc_replace

    bad_attempt = dc_replace(attempt, action="REAUTH")
    with pytest.raises(ValueError, match="only sends ATTEMPT"):
        execute(pg_schema.conn, _MustNotChargeClient(), bad_attempt, _ctx(mandate_id=attempt.mandate_id), owner="worker-a")
