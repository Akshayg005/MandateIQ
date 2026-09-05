"""src/execute/cycle.py -- R4's two-phase orchestrator. The centrepiece is
the gate's own required test: a real read -> solve -> commit -> execute
chain, driven end to end against a live (throwaway) schema, with the frozen
clock advanced 24h between the two phases -- proving plan_cycle() and
run_due() are actually two separate entry points that only cooperate through
durable state, never through anything held in a Python variable across the
gap.

Every fake/fixture pattern here is reused verbatim from
tests/execute/test_executor.py and tests/policy/test_allocator.py rather
than reinvented -- see cycle.py's own module docstring for why (this file's
job is to prove the ORCHESTRATION correct; execute()/commit()/solve()'s own
correctness is already gated separately).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from src.core import clock
from src.core.types import Cause, MandateState, Profile
from src.execute.cycle import _read_context, plan_cycle, run_due
from src.ledger.store import record_lifecycle_event, replay
from src.policy.costs import PolicyCosts

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)  # hour 10: outside quiet hours

_COSTS = PolicyCosts(
    attempt_cost_paise=50,
    mandate_ltv_paise=180_000,
    reauth_cost_paise=200,
    reauth_success_prob=0.35,
    quiet_hours_start=21,
    quiet_hours_end=8,
    max_contacts_per_cycle=4,
)


def _flat_hazard(p_pending, p_rec, p_dead, p_opt):
    def h(*, slot: int, on_day: int, amount_paise: int) -> tuple[float, float, float, float]:
        return (p_pending, p_rec, p_dead, p_opt)
    return h


# A decent-recovery-odds hazard -- the same values test_allocator.py's own
# test_now_belief_with_decent_hazard_attempts uses to get ATTEMPT out of
# solve() for an otherwise-unconstrained context. A fresh mandate's belief
# here is uniform (belief_mod.init(REFERENCE_PRIOR), no evidence yet), not
# the skewed CANT_PAY_NOW belief that test uses -- confirmed to still choose
# ATTEMPT by test_allocator.py's own memoisation test, which runs this exact
# (uniform belief, hazard, ctx, costs) tuple with no forcing constraint.
_ATTEMPT_HAZARD = _flat_hazard(0.4, 0.45, 0.1, 0.05)


class _FakeClient:
    """Mirrors tests/execute/test_executor.py's own _FakeClient exactly."""

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
        raise AssertionError("execute() must never call find_by_receipt()")


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


def _insert_mandate(conn, mandate_id, *, amount_paise=50_000, ceiling_paise=200_000, category="subscription"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO mandate (mandate_id, amount_paise, ceiling_paise, category) "
            "VALUES (%s, %s, %s, %s)",
            (mandate_id, amount_paise, ceiling_paise, category),
        )


def _mandate_created(conn, mandate_id: str, *, at: datetime) -> None:
    record_lifecycle_event(
        conn, event_id=f"evt-created-{mandate_id}", mandate_id=mandate_id,
        state=MandateState.ACTIVE.value, source="INTERNAL", effective_at=at,
    )


def _seed_mandate(conn, mandate_id: str, **overrides) -> None:
    _insert_mandate(conn, mandate_id, **overrides)
    _mandate_created(conn, mandate_id, at=CYCLE_START - timedelta(days=1))


def _non_attempt_plan_rows(conn, mandate_id, cycle_id):
    """`plan` has no chosen_action column (schema.sql's own shape) -- a
    non-ATTEMPT plan (STOP/OFFER/REAUTH) is identified indirectly, by the
    absence of a committed_schedule row sharing its decision_sha256
    (commit()'s own chosen_action gate: only ATTEMPT ever gets one)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.belief_json FROM plan p
            WHERE p.mandate_id = %s AND p.cycle_id = %s
              AND NOT EXISTS (
                SELECT 1 FROM committed_schedule cs WHERE cs.decision_sha256 = p.decision_sha256
              )
            """,
            (mandate_id, cycle_id),
        )
        return cur.fetchall()


def _committed_schedule_count(conn, mandate_id, cycle_id) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM committed_schedule WHERE mandate_id = %s AND cycle_id = %s",
            (mandate_id, cycle_id),
        )
        return cur.fetchone()[0]


# === THE GATE'S OWN REQUIRED TEST: read -> solve -> commit -> execute ======

def test_plan_cycle_then_run_due_end_to_end(pg_schema):
    mandate_id = "M-CYCLE-1"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )

    assert len(committed) == 1
    attempt = committed[0]
    assert attempt.mandate_id == mandate_id
    assert attempt.scheduled_for >= CYCLE_START + timedelta(hours=24)

    with pg_schema.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM plan WHERE mandate_id = %s", (mandate_id,))
        assert cur.fetchone()[0] == 1
    assert _committed_schedule_count(pg_schema.conn, mandate_id, 1) == 1

    # Phase 2, >=24h later: advance the frozen clock to the scheduled moment.
    clock.set_frozen(attempt.scheduled_for)
    client = _FakeClient(charge_response={"id": "pay_ok", "status": "captured"})
    results = run_due(pg_schema.conn, client, costs=_COSTS, owner="worker-a")

    assert len(client.calls) == 1, "run_due() must actually reach the provider for a due attempt"
    assert len(results) == 1
    assert results[0].state == "RESULT"
    assert results[0].outcome == "RECOVERED"

    rows = replay(pg_schema.conn, mandate_id)
    states = [r.state for r in rows if r.idempotency_key == attempt.idempotency_key]
    assert states == ["INTENT", "SENT", "RESULT"]


def test_run_due_before_scheduled_time_does_not_execute(pg_schema):
    """A due-row scan at `as_of` before scheduled_for must not pick it up --
    the frozen-clock control is real, not incidental."""
    mandate_id = "M-CYCLE-EARLY"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    attempt = committed[0]

    client = _FakeClient()
    results = run_due(
        pg_schema.conn, client, costs=_COSTS, owner="worker-a",
        as_of=attempt.scheduled_for - timedelta(hours=1),
    )
    assert results == []
    assert client.calls == []


# === the DEAD path: observe_terminal() gets its first production caller ===

def test_plan_cycle_after_a_dead_resolution_uses_observe_terminal_and_reauths(pg_schema):
    mandate_id = "M-CYCLE-DEAD"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    attempt = committed[0]

    clock.set_frozen(attempt.scheduled_for)
    client = _FakeClient(charge_exception=_declined("mandate revoked by customer"))
    results = run_due(pg_schema.conn, client, costs=_COSTS, owner="worker-a")
    assert results[0].outcome == "DEAD"

    # Second planning pass, same cycle: must read the DEAD resolution back
    # from the ledger, call observe_terminal() (not update()), and REAUTH --
    # not another ATTEMPT (permitted() denies it: ctx.instrument_dead).
    second = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    assert second == [], "a REAUTH/STOP decision never produces a committed_schedule row"
    assert _committed_schedule_count(pg_schema.conn, mandate_id, 1) == 1, "no second ATTEMPT was ever committed"

    reauth_rows = _non_attempt_plan_rows(pg_schema.conn, mandate_id, 1)
    assert len(reauth_rows) == 1, "exactly one non-ATTEMPT (REAUTH) plan must have been written"
    belief = json.loads(reauth_rows[0][0])
    # The measured posterior (belief_mod.TERMINAL_OBSERVED_CAUSE_PROBS[DEAD]),
    # not the degenerate 1.0 this project explicitly rejected (reports/
    # gates.md, R2a).
    assert belief[Cause.CANT_PAY_EVER.value] == pytest.approx(0.8991, abs=1e-4)
    assert ";observed=terminal" in belief["provenance"]


def _declined(message: str):
    from src.execute.razorpay_client import RazorpayDeclined
    return RazorpayDeclined(message)


# === eligibility skips ======================================================

def test_in_flight_commitment_is_not_replanned(pg_schema):
    mandate_id = "M-CYCLE-INFLIGHT"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    first = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    assert len(first) == 1
    assert _committed_schedule_count(pg_schema.conn, mandate_id, 1) == 1

    # No run_due() call -- the attempt is still unresolved (INTENT never
    # even written yet). A second plan_cycle() call must not commit another.
    second = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    assert second == []
    assert _committed_schedule_count(pg_schema.conn, mandate_id, 1) == 1


def test_recovered_cycle_is_not_replanned(pg_schema):
    mandate_id = "M-CYCLE-RECOVERED"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    attempt = committed[0]

    clock.set_frozen(attempt.scheduled_for)
    client = _FakeClient(charge_response={"id": "pay_ok", "status": "captured"})
    run_due(pg_schema.conn, client, costs=_COSTS, owner="worker-a")

    second = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    assert second == []
    assert _committed_schedule_count(pg_schema.conn, mandate_id, 1) == 1


# === _read_context() unit tests ============================================

def test_read_context_fresh_mandate_matches_eval_initial_context_convention(pg_schema):
    mandate_id = "M-CTX-FRESH"
    _insert_mandate(pg_schema.conn, mandate_id, amount_paise=75_000, ceiling_paise=150_000, category="subscription")
    _mandate_created(pg_schema.conn, mandate_id, at=CYCLE_START - timedelta(days=1))

    ctx = _read_context(
        pg_schema.conn, mandate_id=mandate_id, cycle_id=1, profile=Profile.strict,
        costs=_COSTS, cycle_start=CYCLE_START,
    )
    assert ctx.amount_paise == 75_000
    assert ctx.ceiling_paise == 150_000
    assert ctx.category == "subscription"
    assert ctx.attempts_used == 1
    assert ctx.committed_days == (1,)
    assert ctx.contacts_sent == 1
    assert ctx.mandate_state == MandateState.ACTIVE
    assert ctx.opted_out is False
    assert ctx.instrument_dead is False


def test_read_context_reflects_one_resolved_attempt(pg_schema):
    mandate_id = "M-CTX-ONE-ATTEMPT"
    _seed_mandate(pg_schema.conn, mandate_id)

    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(
        pg_schema.conn, cycle_id=1, cycle_start=CYCLE_START,
        hazard=_ATTEMPT_HAZARD, costs=_COSTS,
    )
    attempt = committed[0]
    clock.set_frozen(attempt.scheduled_for)
    client = _FakeClient(charge_response={"id": "pay_ok", "status": "captured"})
    run_due(pg_schema.conn, client, costs=_COSTS, owner="worker-a")

    on_day = (attempt.scheduled_for - CYCLE_START).days + 1
    ctx = _read_context(
        pg_schema.conn, mandate_id=mandate_id, cycle_id=1, profile=Profile.strict,
        costs=_COSTS, cycle_start=CYCLE_START,
    )
    assert ctx.attempts_used == 2
    assert ctx.committed_days == (1, on_day)
    assert ctx.contacts_sent == 2
    assert ctx.plan_day == on_day


def test_read_context_without_cycle_start_uses_placeholder_committed_days(pg_schema):
    """run_due()'s own call path -- no cycle_start available, and none
    needed: execute()'s permitted() check never reads committed_days."""
    mandate_id = "M-CTX-NO-START"
    _insert_mandate(pg_schema.conn, mandate_id)
    _mandate_created(pg_schema.conn, mandate_id, at=CYCLE_START - timedelta(days=1))

    ctx = _read_context(
        pg_schema.conn, mandate_id=mandate_id, cycle_id=1, profile=Profile.strict, costs=_COSTS,
    )
    assert ctx.committed_days == (1,)
    assert ctx.plan_day == 1


def test_read_context_raises_lookup_error_for_unregistered_mandate(pg_schema):
    with pytest.raises(LookupError):
        _read_context(
            pg_schema.conn, mandate_id="M-NEVER-REGISTERED", cycle_id=1,
            profile=Profile.strict, costs=_COSTS, cycle_start=CYCLE_START,
        )


# === mandate table CHECK constraints ========================================

def test_mandate_rejects_ceiling_below_amount(pg_schema):
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_mandate(pg_schema.conn, "M-BAD-CEILING", amount_paise=100_000, ceiling_paise=50_000)


def test_mandate_rejects_negative_amount(pg_schema):
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_mandate(pg_schema.conn, "M-BAD-AMOUNT", amount_paise=-1, ceiling_paise=50_000)
