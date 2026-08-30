"""
src/ledger/schema.sql -- pure Postgres 16 DDL. Tested two ways:

1. Text-level: the file must never contain an UPDATE against `ledger`.
   `ledger` is append-only by construction, not just by convention.
2. Live-apply: the DDL is applied into a throwaway scratch schema on the
   real local Postgres (docker container `mrdb`, confirmed running) and
   exercised for real -- column presence, both partial-unique indexes, the
   24h CHECK constraint, and the plan->ledger FK ordering.

Every live test below uses the pg_schema fixture, which SKIPS if Postgres
itself is unreachable but does NOT skip just because schema.sql doesn't
exist yet -- a missing/broken schema file is the thing under test here and
must fail loudly (see tests/ledger/conftest.py).

Decision recorded here, not implemented (spec item 7 in the B1 test brief):
Postgres will happily let an application UPDATE the `ledger` table -- there
is no trigger blocking it, and the spec explicitly says not to invent one.
Append-only-ness is enforced at two OTHER layers instead, both tested
elsewhere: schema.sql's own text never issuing UPDATE ledger (this file,
below), and src/ledger/store.py's source never issuing it either
(tests/ledger/test_store.py). A DB-level trigger is out of scope by design.
"""
from __future__ import annotations

import pathlib
import re
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT / "src" / "ledger" / "schema.sql"


# --- helpers: raw SQL, deliberately NOT going through store.py -------------
# (store.py doesn't exist yet either, and is exercised separately in
# test_store.py -- these tests are about the DDL itself.)

def _insert_ledger_row(conn, *, key, state, decision_sha256, mandate_id="M-TEST",
                        cycle_id=1, attempt_index=1, action="ATTEMPT",
                        amount_paise=10000, profile="strict",
                        payload_sha256="0" * 64):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ledger (
                idempotency_key, mandate_id, cycle_id, attempt_index, action,
                state, amount_paise, profile, payload_sha256, decision_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING ledger_id
            """,
            (key, mandate_id, cycle_id, attempt_index, action, state,
             amount_paise, profile, payload_sha256, decision_sha256),
        )
        return cur.fetchone()[0]


def _insert_committed_schedule_row(conn, *, key, committed_at, scheduled_for,
                                    decision_sha256, mandate_id="M-TEST", cycle_id=1,
                                    attempt_index=1, action="ATTEMPT",
                                    amount_paise=10000, profile="strict"):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO committed_schedule (
                idempotency_key, mandate_id, cycle_id, attempt_index,
                action, amount_paise, profile, decision_sha256, scheduled_for, committed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING idempotency_key
            """,
            (key, mandate_id, cycle_id, attempt_index, action, amount_paise,
             profile, decision_sha256, scheduled_for, committed_at),
        )
        return cur.fetchone()[0]


def _columns(conn, schema, table):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {row[0] for row in cur.fetchall()}


# --- text-level --------------------------------------------------------------

def test_schema_sql_never_updates_the_ledger_table():
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert re.search(r"UPDATE\s+ledger\b", text, re.IGNORECASE) is None


def test_schema_sql_never_updates_the_normalized_decline_table():
    """normalized_decline is append-only by the same construction principle
    as ledger: every row is immutable, and state changes live in different
    tables. Enforce this by ensuring schema.sql never contains an UPDATE
    against normalized_decline."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert re.search(r"UPDATE\s+normalized_decline\b", text, re.IGNORECASE) is None, \
        "schema.sql must never UPDATE normalized_decline -- it is append-only by construction"


# --- applies cleanly -----------------------------------------------------------

def test_schema_applies_cleanly_to_a_scratch_schema(pg_schema):
    assert pg_schema.conn is not None
    assert pg_schema.schema.startswith("test_b1_")


# --- table / column presence ----------------------------------------------------

def test_ledger_table_has_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "ledger")
    for col in (
        "ledger_id", "idempotency_key", "mandate_id", "cycle_id",
        "attempt_index", "action", "state", "amount_paise", "provider_ref",
        "outcome", "decline_class", "reason", "profile", "payload_sha256",
        "decision_sha256", "created_at",
    ):
        assert col in cols, f"ledger.{col} missing"
    # These three were missing from an earlier version of this schema, per
    # a past reviewer -- pin them staying present.
    assert {"reason", "profile", "decision_sha256"}.issubset(cols)


def test_committed_schedule_table_has_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "committed_schedule")
    for col in (
        "idempotency_key", "mandate_id", "cycle_id", "attempt_index",
        "generation", "action", "amount_paise", "profile", "decision_sha256",
        "scheduled_for", "committed_at", "notification_sent_at", "voided_at",
        "void_reason",
    ):
        assert col in cols, f"committed_schedule.{col} missing"
    assert {"generation", "voided_at", "void_reason", "profile", "decision_sha256"}.issubset(cols)


def test_plan_table_exists_with_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "plan")
    assert cols, "plan table does not exist"
    for col in (
        "decision_sha256", "mandate_id", "cycle_id", "profile",
        "belief_json", "conformal_set", "binding_constraint",
        "solver_version", "created_at",
    ):
        assert col in cols, f"plan.{col} missing"


def test_mandate_lifecycle_table_exists_with_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "mandate_lifecycle")
    assert cols, "mandate_lifecycle table does not exist"
    for col in ("event_id", "mandate_id", "state", "source", "effective_at", "recorded_at"):
        assert col in cols, f"mandate_lifecycle.{col} missing"


def test_attempt_lease_table_exists_with_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "attempt_lease")
    assert cols, "attempt_lease table does not exist"
    for col in ("idempotency_key", "owner", "expires_at"):
        assert col in cols, f"attempt_lease.{col} missing"


def test_normalized_decline_table_exists_with_required_columns(pg_schema):
    cols = _columns(pg_schema.conn, pg_schema.schema, "normalized_decline")
    assert cols, "normalized_decline table does not exist"
    for col in ("event_id", "value", "normalizer_version", "model_id", "raw_sha256", "created_at"):
        assert col in cols, f"normalized_decline.{col} missing"


# --- FK ordering: plan must exist before ledger can reference it -----------

def test_ledger_decision_sha256_fk_rejects_unknown_plan(pg_schema):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_ledger_row(
            pg_schema.conn, key="fk-missing-plan", state="INTENT",
            decision_sha256="no-such-plan-row-exists",
        )


# --- ledger_intent_once: unique, but ONLY while state='INTENT' -------------

def test_ledger_intent_once_rejects_duplicate_intent_same_key(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-dup-1")
    key = "idem-dup-1"
    _insert_ledger_row(pg_schema.conn, key=key, state="INTENT", decision_sha256=decision_sha)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_ledger_row(pg_schema.conn, key=key, state="INTENT", decision_sha256=decision_sha)


def test_ledger_intent_once_allows_later_stage_with_same_key(pg_schema, seed_plan):
    """A RESULT row sharing the INTENT row's key is a later stage of the
    SAME attempt, not a duplicate intent -- the partial index only fires
    on state='INTENT'."""
    decision_sha = seed_plan("plan-dup-2")
    key = "idem-dup-2"
    _insert_ledger_row(pg_schema.conn, key=key, state="INTENT", decision_sha256=decision_sha)
    result_id = _insert_ledger_row(pg_schema.conn, key=key, state="RESULT", decision_sha256=decision_sha)
    assert result_id is not None


# --- committed_schedule: 24h CHECK constraint (RBI clause 6(a)) -------------

def test_committed_schedule_rejects_lead_time_under_24_hours(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-23h")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=23)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_committed_schedule_row(
            pg_schema.conn, key="cs-23h", committed_at=committed_at, scheduled_for=scheduled_for,
            decision_sha256=decision_sha,
        )


def test_committed_schedule_accepts_exactly_24_hours(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-24h")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=24)
    row_id = _insert_committed_schedule_row(
        pg_schema.conn, key="cs-24h", committed_at=committed_at, scheduled_for=scheduled_for,
        decision_sha256=decision_sha,
    )
    assert row_id == "cs-24h"


def test_committed_schedule_accepts_more_than_24_hours(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-48h")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=48)
    row_id = _insert_committed_schedule_row(
        pg_schema.conn, key="cs-48h", committed_at=committed_at, scheduled_for=scheduled_for,
        decision_sha256=decision_sha,
    )
    assert row_id == "cs-48h"


def test_committed_schedule_decision_sha256_fk_rejects_unknown_plan(pg_schema):
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_committed_schedule_row(
            pg_schema.conn, key="cs-fk-missing", committed_at=committed_at,
            scheduled_for=committed_at + timedelta(hours=48),
            decision_sha256="no-such-plan-row-exists",
        )


# --- normalized_decline: append-only LLM normalizer outputs ---------------

def test_normalized_decline_composite_pk_allows_multiple_versions_same_event(pg_schema):
    """normalized_decline has a composite PK of (event_id, normalizer_version).
    Inserting two rows with the SAME event_id but DIFFERENT normalizer_version
    must succeed: this is the whole point -- prompt bumps create NEW rows, never
    destroy old ones. The old fact remains readable for audit trails."""
    with pg_schema.conn.cursor() as cur:
        # Insert two rows for the same event_id but different normalizer versions
        cur.execute(
            """
            INSERT INTO ingested_event (event_id, event_type, raw_payload_sha256)
            VALUES (%s, %s, %s)
            """,
            ("evt-norm-v-test", "payment.failed", "a" * 64),
        )
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO normalized_decline (event_id, value, confidence, normalizer_version, model_id, raw_sha256)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            ("evt-norm-v-test", "INSUFFICIENT_FUNDS", 0.9, "v1", "model-abc", "b" * 64),
        )
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO normalized_decline (event_id, value, confidence, normalizer_version, model_id, raw_sha256)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            ("evt-norm-v-test", "CARD_EXPIRED", 0.85, "v2", "model-def", "c" * 64),
        )
    # Verify both rows exist
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM normalized_decline WHERE event_id = %s",
            ("evt-norm-v-test",),
        )
        count = cur.fetchone()[0]
    assert count == 2, f"Expected 2 rows for event_id, got {count}"


def test_normalized_decline_dedupes_duplicate_event_id_normalizer_version(pg_schema):
    """Inserting the SAME (event_id, normalizer_version) twice with different
    values must be a silent no-op (ON CONFLICT DO NOTHING) -- the FIRST write
    wins, the second is dropped. This is the dedupe discipline every table in
    this schema uses: a retried normalization request of the same event with
    the same normalizer version returns the original verdict, never a second
    row and never an exception."""
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_event (event_id, event_type, raw_payload_sha256)
            VALUES (%s, %s, %s)
            """,
            ("evt-norm-dup", "payment.failed", "d" * 64),
        )
    # First insert
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO normalized_decline (event_id, value, confidence, normalizer_version, model_id, raw_sha256)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id, normalizer_version) DO NOTHING
            """,
            ("evt-norm-dup", "INSUFFICIENT_FUNDS", 0.9, "v1", "model-abc", "e" * 64),
        )
    # Second insert with DIFFERENT value but same key -- should be silently dropped
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO normalized_decline (event_id, value, confidence, normalizer_version, model_id, raw_sha256)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id, normalizer_version) DO NOTHING
            """,
            ("evt-norm-dup", "DIFFERENT_VALUE", 0.4, "v1", "model-xyz", "f" * 64),
        )
    # Verify only the FIRST value persists
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT value, model_id FROM normalized_decline "
            "WHERE event_id = %s AND normalizer_version = %s",
            ("evt-norm-dup", "v1"),
        )
        row = cur.fetchone()
    assert row is not None, "Row should exist"
    assert row[0] == "INSUFFICIENT_FUNDS", "Value should be from FIRST insert"
    assert row[1] == "model-abc", "model_id should be from FIRST insert"


def test_normalized_decline_fk_rejects_nonexistent_event_id(pg_schema):
    """normalized_decline.event_id REFERENCES ingested_event(event_id).
    Attempting to insert a row with an event_id that has no matching
    ingested_event row must raise ForeignKeyViolation."""
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with pg_schema.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO normalized_decline (event_id, value, confidence, normalizer_version, model_id, raw_sha256)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                ("evt-norm-fk-missing", "SOME_VALUE", 0.8, "v1", "model-abc", "a" * 64),
            )


# --- attempt_index CHECK: NPCI's 1..4 cap, defense-in-depth at the DB ------

def test_ledger_rejects_attempt_index_zero(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-attempt-idx-0")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_ledger_row(
            pg_schema.conn, key="idem-attempt-0", state="INTENT",
            decision_sha256=decision_sha, attempt_index=0,
        )


def test_ledger_rejects_attempt_index_above_four(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-attempt-idx-5")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_ledger_row(
            pg_schema.conn, key="idem-attempt-5", state="INTENT",
            decision_sha256=decision_sha, attempt_index=5,
        )


def test_committed_schedule_rejects_attempt_index_zero(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-attempt-0")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=24)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_committed_schedule_row(
            pg_schema.conn, key="cs-attempt-0", attempt_index=0,
            committed_at=committed_at, scheduled_for=scheduled_for,
            decision_sha256=decision_sha,
        )


# --- amount_paise CHECK: never negative --------------------------------------

def test_ledger_rejects_negative_amount_paise(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-neg-amount")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_ledger_row(
            pg_schema.conn, key="idem-neg-amount", state="INTENT",
            decision_sha256=decision_sha, amount_paise=-1,
        )


def test_committed_schedule_rejects_negative_amount_paise(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-neg-amount")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=24)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_committed_schedule_row(
            pg_schema.conn, key="cs-neg-amount", amount_paise=-1,
            committed_at=committed_at, scheduled_for=scheduled_for,
            decision_sha256=decision_sha,
        )


# --- committed_one_live_per_slot --------------------------------------------

def test_committed_one_live_per_slot_rejects_second_live_row(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-cs-slot-ab")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=24)
    _insert_committed_schedule_row(
        pg_schema.conn, key="cs-slot-a", mandate_id="M-SLOT", cycle_id=1,
        attempt_index=1, committed_at=committed_at, scheduled_for=scheduled_for,
        decision_sha256=decision_sha,
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_committed_schedule_row(
            pg_schema.conn, key="cs-slot-b", mandate_id="M-SLOT", cycle_id=1,
            attempt_index=1, committed_at=committed_at, scheduled_for=scheduled_for,
            decision_sha256=decision_sha,
        )


def test_committed_one_live_per_slot_allows_reinsert_after_void(pg_schema, seed_plan):
    """Voiding is an UPDATE -- legal here, because this constraint lives on
    committed_schedule, which is NOT append-only (only `ledger` is)."""
    decision_sha = seed_plan("plan-cs-slot-cd")
    committed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scheduled_for = committed_at + timedelta(hours=24)
    _insert_committed_schedule_row(
        pg_schema.conn, key="cs-slot-c", mandate_id="M-SLOT2", cycle_id=1,
        attempt_index=1, committed_at=committed_at, scheduled_for=scheduled_for,
        decision_sha256=decision_sha,
    )
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "UPDATE committed_schedule SET voided_at = %s, void_reason = %s "
            "WHERE idempotency_key = %s",
            (datetime.now(timezone.utc), "test-void", "cs-slot-c"),
        )
    row_id = _insert_committed_schedule_row(
        pg_schema.conn, key="cs-slot-d", mandate_id="M-SLOT2", cycle_id=1,
        attempt_index=1, committed_at=committed_at, scheduled_for=scheduled_for,
        decision_sha256=decision_sha,
    )
    assert row_id == "cs-slot-d"
