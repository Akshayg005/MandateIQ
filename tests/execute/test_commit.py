"""src/execute/commit.py -- the plan/committed_schedule writer that bridges
B8's pure, DB-free solve() into durable rows.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import clock
from src.core.types import Action, Cause, Profile
from src.execute.commit import CommitError, commit
from src.policy.allocator import CommittedAttempt, Plan

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


def _plan(*, decision_sha256: str, **overrides) -> Plan:
    """Every call site passes its own decision_sha256 explicitly -- it is
    the plan table's primary key, and each test needs a value it does not
    share with any other test to avoid one test's row masking another's
    assertions via ON CONFLICT DO NOTHING."""
    base = dict(
        mandate_id="M-COMMIT-1",
        cycle_id=1,
        profile=Profile.strict,
        chosen_action=Action.STOP,
        committed=(),
        belief_json='{"CANT_PAY_NOW": 0.5}',
        conformal_set=frozenset(),
        binding_constraint=None,
        solver_version="test-solver-v0",
        decision_sha256=decision_sha256,
    )
    base.update(overrides)
    return Plan(**base)


def _columns_exist(conn, table, key_col, key_val):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table} WHERE {key_col} = %s", (key_val,))
        return cur.fetchall()


# --- non-ATTEMPT actions: plan row only, no committed_schedule row ---------

@pytest.mark.parametrize("action", [Action.STOP, Action.OFFER, Action.REAUTH])
def test_commit_non_attempt_action_writes_only_the_plan_row(pg_schema, action):
    clock.set_frozen(CYCLE_START)
    plan = _plan(decision_sha256=f"decision-{action.value}", chosen_action=action, committed=())

    result = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)

    assert result is None
    plan_rows = _columns_exist(pg_schema.conn, "plan", "decision_sha256", plan.decision_sha256)
    assert len(plan_rows) == 1
    schedule_rows = _columns_exist(
        pg_schema.conn, "committed_schedule", "mandate_id", plan.mandate_id
    )
    assert schedule_rows == []


# --- ATTEMPT: writes both, returns a ScheduledAttempt -----------------------

def test_commit_attempt_writes_committed_schedule_row(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-attempt-1",
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=2, amount_paise=50_000),),
    )

    result = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)

    assert result is not None
    assert result.mandate_id == plan.mandate_id
    assert result.attempt_index == 1
    assert result.generation == 0
    assert result.amount_paise == 50_000
    assert result.decision_sha256 == plan.decision_sha256
    assert result.scheduled_for == CYCLE_START + timedelta(days=1)  # on_day=2 -> +1 day

    rows = _columns_exist(pg_schema.conn, "committed_schedule", "idempotency_key", result.idempotency_key)
    assert len(rows) == 1


def test_commit_attempt_on_day_one_maps_to_cycle_start_itself(pg_schema):
    clock.set_frozen(CYCLE_START - timedelta(days=2))
    plan = _plan(
        decision_sha256="decision-on-day-1",
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=1, amount_paise=10_000),),
    )
    result = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)
    assert result.scheduled_for == CYCLE_START


# --- idempotent retry: the crash-recovery case this design exists for ------

def test_commit_is_idempotent_on_identical_retry(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-retry-1",
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=3, amount_paise=20_000),),
    )

    first = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)

    # Simulate a retry well after the original call -- clock.now() at the
    # second call is DIFFERENT from the first, and must NOT be what wins.
    clock.set_frozen(CYCLE_START + timedelta(hours=5))
    second = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)

    assert second.idempotency_key == first.idempotency_key
    assert second.committed_at == first.committed_at
    assert second.scheduled_for == first.scheduled_for

    rows = _columns_exist(pg_schema.conn, "committed_schedule", "idempotency_key", first.idempotency_key)
    assert len(rows) == 1, "a retried commit() must never create a second row"


def test_commit_plan_row_is_idempotent_on_retry(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(decision_sha256="decision-plan-retry")
    commit(pg_schema.conn, plan, cycle_start=CYCLE_START)
    commit(pg_schema.conn, plan, cycle_start=CYCLE_START)  # must not raise

    rows = _columns_exist(pg_schema.conn, "plan", "decision_sha256", plan.decision_sha256)
    assert len(rows) == 1


# --- the disclosed day-index / wall-clock gap: CHECK violation surfaces ----

def test_commit_raises_commit_error_when_mapped_lead_time_is_under_24h(pg_schema):
    """The exact gap this module's docstring names: on_day=1 mapped
    straight onto a committed_at equal to cycle_start gives a zero-hour
    lead, which B8's day-index model could consider legal (permissive's
    "the next slot may land on plan_day itself") but which fails
    committed_schedule's real 24h CHECK. This must surface as a typed
    CommitError, never silently pass or silently reschedule."""
    clock.set_frozen(CYCLE_START)  # committed_at == cycle_start exactly
    plan = _plan(
        decision_sha256="decision-under-24h",
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=1, amount_paise=10_000),),
    )
    with pytest.raises(CommitError, match="not actually committable"):
        commit(pg_schema.conn, plan, cycle_start=CYCLE_START)

    # And the plan row from the SAME call must still have been written --
    # only the committed_schedule half of the decision failed.
    rows = _columns_exist(pg_schema.conn, "plan", "decision_sha256", plan.decision_sha256)
    assert len(rows) == 1


def test_commit_accepts_exactly_the_boundary(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-boundary-24h",
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=2, amount_paise=10_000),),
    )
    # on_day=2 -> scheduled_for = cycle_start + 1 day = committed_at + 24h exactly.
    result = commit(pg_schema.conn, plan, cycle_start=CYCLE_START)
    assert result.scheduled_for == result.committed_at + timedelta(hours=24)


# --- upstream-contract guard: committed must be exactly one entry ----------

def test_commit_asserts_when_attempt_has_no_committed_entries(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-bad-zero",
        chosen_action=Action.ATTEMPT,
        committed=(),
    )
    with pytest.raises(AssertionError):
        commit(pg_schema.conn, plan, cycle_start=CYCLE_START)


def test_commit_asserts_when_attempt_has_multiple_committed_entries(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-bad-multi",
        chosen_action=Action.ATTEMPT,
        committed=(
            CommittedAttempt(slot=1, on_day=2, amount_paise=10_000),
            CommittedAttempt(slot=2, on_day=10, amount_paise=10_000),
        ),
    )
    with pytest.raises(AssertionError):
        commit(pg_schema.conn, plan, cycle_start=CYCLE_START)


# --- conformal_set / binding_constraint round-trip through the plan row ----

def test_commit_stores_conformal_set_and_binding_constraint(pg_schema):
    clock.set_frozen(CYCLE_START)
    plan = _plan(
        decision_sha256="decision-conformal",
        conformal_set=frozenset({Cause.WONT_PAY}),
        binding_constraint="AFA_CLIFF",
    )
    commit(pg_schema.conn, plan, cycle_start=CYCLE_START)
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT conformal_set, binding_constraint FROM plan WHERE decision_sha256 = %s",
            (plan.decision_sha256,),
        )
        conformal_set, binding_constraint = cur.fetchone()
    assert conformal_set == "WONT_PAY"
    assert binding_constraint == "AFA_CLIFF"
