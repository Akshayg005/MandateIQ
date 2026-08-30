"""src/execute/void.py -- void-and-reissue, and the SENT-row rule.

Builds its committed_schedule rows through src.execute.commit.commit()
rather than raw SQL -- commit.py already exists and is the one real
producer of these rows, so going through it here means these tests
exercise the same path production code will.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import clock
from src.core.types import Action, Profile
from src.execute.commit import CommitError, commit
from src.execute.keys import key_for
from src.execute.void import VoidError, reissue, void
from src.ledger.store import LedgerEntry, append
from src.policy.allocator import CommittedAttempt, Plan

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


def _plan(*, decision_sha256, on_day=2, slot=1, amount_paise=50_000, mandate_id="M-VOID-1"):
    return Plan(
        mandate_id=mandate_id, cycle_id=1, profile=Profile.strict,
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=slot, on_day=on_day, amount_paise=amount_paise),),
        belief_json="{}", conformal_set=frozenset(), binding_constraint=None,
        solver_version="test-solver-v0", decision_sha256=decision_sha256,
    )


def _committed(pg_schema, **kwargs):
    clock.set_frozen(CYCLE_START)
    plan = _plan(**kwargs)
    return commit(pg_schema.conn, plan, cycle_start=CYCLE_START)


# --- void(): the SENT-row rule ----------------------------------------------

def test_void_succeeds_when_no_ledger_row_exists_at_all(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-void-1")
    void(pg_schema.conn, attempt.idempotency_key, reason="test-no-ledger-row")
    row = _read_schedule(pg_schema, attempt.idempotency_key)
    assert row["voided_at"] is not None
    assert row["void_reason"] == "test-no-ledger-row"


def test_void_succeeds_when_only_an_intent_row_exists(pg_schema):
    """The exact case executor.py's pre-call abort hits: INTENT was just
    written by this same process, no send has happened."""
    attempt = _committed(pg_schema, decision_sha256="d-void-2")
    _append_ledger(pg_schema, attempt, state="INTENT")

    void(pg_schema.conn, attempt.idempotency_key, reason="lifecycle=REVOKED")

    row = _read_schedule(pg_schema, attempt.idempotency_key)
    assert row["voided_at"] is not None


def test_void_raises_when_a_sent_row_exists(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-void-3")
    _append_ledger(pg_schema, attempt, state="INTENT")
    _append_ledger(pg_schema, attempt, state="SENT")

    with pytest.raises(VoidError, match="SENT row already exists"):
        void(pg_schema.conn, attempt.idempotency_key, reason="should-not-apply")

    row = _read_schedule(pg_schema, attempt.idempotency_key)
    assert row["voided_at"] is None, "a rejected void must not partially apply"


def test_void_is_idempotent_and_preserves_the_first_reason(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-void-4")
    void(pg_schema.conn, attempt.idempotency_key, reason="first-reason")
    void(pg_schema.conn, attempt.idempotency_key, reason="second-reason-must-not-apply")

    row = _read_schedule(pg_schema, attempt.idempotency_key)
    assert row["void_reason"] == "first-reason"


# --- reissue(): generation+1, same attempt_index, no slot spent ------------

def test_reissue_creates_a_new_row_at_generation_plus_one(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-reissue-1")
    void(pg_schema.conn, attempt.idempotency_key, reason="overtaken")

    new_scheduled_for = attempt.scheduled_for + timedelta(days=5)
    reissued = reissue(pg_schema.conn, attempt.idempotency_key, scheduled_for=new_scheduled_for)

    assert reissued.generation == 1
    assert reissued.attempt_index == attempt.attempt_index  # no slot spent
    assert reissued.mandate_id == attempt.mandate_id
    assert reissued.decision_sha256 == attempt.decision_sha256
    assert reissued.scheduled_for == new_scheduled_for
    assert reissued.idempotency_key != attempt.idempotency_key


def test_reissue_key_matches_an_independent_key_for_computation(pg_schema):
    """The key derivation itself is keys.key_for's job, not reissue()'s
    own reimplementation of it -- this pins that reissue()'s internal
    computation doesn't quietly diverge from calling key_for directly."""
    attempt = _committed(pg_schema, decision_sha256="d-reissue-key-oracle", amount_paise=50_000)
    void(pg_schema.conn, attempt.idempotency_key, reason="test")

    reissued = reissue(
        pg_schema.conn, attempt.idempotency_key,
        scheduled_for=attempt.scheduled_for + timedelta(days=1),
    )

    expected = key_for(
        mandate_id=attempt.mandate_id, cycle_id=attempt.cycle_id,
        attempt_index=attempt.attempt_index, generation=1,
        action=attempt.action, amount_paise=attempt.amount_paise,
    )
    assert reissued.idempotency_key == expected


def test_reissue_can_reprice_the_amount(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-reissue-2", amount_paise=50_000)
    void(pg_schema.conn, attempt.idempotency_key, reason="repriced")

    reissued = reissue(
        pg_schema.conn, attempt.idempotency_key,
        scheduled_for=attempt.scheduled_for + timedelta(days=3),
        amount_paise=45_000,
    )
    assert reissued.amount_paise == 45_000
    assert reissued.idempotency_key != attempt.idempotency_key  # amount is key material


def test_reissue_defaults_amount_to_the_original_when_not_repriced(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-reissue-3", amount_paise=33_000)
    void(pg_schema.conn, attempt.idempotency_key, reason="moved-date-only")

    reissued = reissue(
        pg_schema.conn, attempt.idempotency_key,
        scheduled_for=attempt.scheduled_for + timedelta(days=2),
    )
    assert reissued.amount_paise == 33_000


def test_reissue_raises_when_no_such_row_exists(pg_schema):
    with pytest.raises(VoidError, match="no committed_schedule row"):
        reissue(pg_schema.conn, "no-such-key-ever", scheduled_for=CYCLE_START + timedelta(days=10))


def test_reissue_raises_when_original_is_still_live(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-reissue-4")
    with pytest.raises(VoidError, match="still-live row"):
        reissue(pg_schema.conn, attempt.idempotency_key, scheduled_for=attempt.scheduled_for + timedelta(days=1))


def test_reissue_raises_commit_error_under_24h_lead(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-reissue-5")
    void(pg_schema.conn, attempt.idempotency_key, reason="test")

    clock.set_frozen(attempt.committed_at)  # reissue's own committed_at == this
    with pytest.raises(CommitError):
        reissue(
            pg_schema.conn, attempt.idempotency_key,
            scheduled_for=clock.now() + timedelta(hours=1),  # nowhere near 24h
        )


def test_committed_one_live_per_slot_holds_across_void_and_reissue(pg_schema):
    """At most one LIVE row per (mandate_id, cycle_id, attempt_index) at
    any time -- proven by reading the table directly, not just trusting
    reissue() didn't raise."""
    attempt = _committed(pg_schema, decision_sha256="d-reissue-6")
    void(pg_schema.conn, attempt.idempotency_key, reason="test")
    reissued = reissue(pg_schema.conn, attempt.idempotency_key, scheduled_for=attempt.scheduled_for + timedelta(days=1))

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT idempotency_key FROM committed_schedule "
            "WHERE mandate_id = %s AND cycle_id = %s AND attempt_index = %s AND voided_at IS NULL",
            (attempt.mandate_id, attempt.cycle_id, attempt.attempt_index),
        )
        live_keys = [row[0] for row in cur.fetchall()]
    assert live_keys == [reissued.idempotency_key]


# --- helpers -----------------------------------------------------------------

def _append_ledger(pg_schema, attempt, *, state: str) -> None:
    append(
        pg_schema.conn,
        LedgerEntry(
            idempotency_key=attempt.idempotency_key,
            mandate_id=attempt.mandate_id,
            cycle_id=attempt.cycle_id,
            attempt_index=attempt.attempt_index,
            action=attempt.action,
            state=state,
            amount_paise=attempt.amount_paise,
            profile=attempt.profile,
            payload_sha256="0" * 64,
            decision_sha256=attempt.decision_sha256,
        ),
    )


def _read_schedule(pg_schema, key: str) -> dict:
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT voided_at, void_reason FROM committed_schedule WHERE idempotency_key = %s",
            (key,),
        )
        voided_at, void_reason = cur.fetchone()
    return {"voided_at": voided_at, "void_reason": void_reason}
