"""
src/ledger/store.py -- append-only writes and replay on top of schema.sql.

Design decisions this test file pins (documented here since this interface
is genuinely new and the B1 spec leaves its exact shape to the
implementer):

- Every public function takes an explicit psycopg connection as its FIRST
  argument (`append(conn, entry)`, `replay(conn, mandate_id)`, etc.) rather
  than managing a module-level connection internally. A bare `append(entry)`
  cannot be pointed at an isolated scratch schema per test, which these
  tests need in order to never touch a shared database state.
- `LedgerEntry` is a plain dataclass covering every NOT NULL column on
  `ledger` except `ledger_id` (DB-assigned via BIGSERIAL) and `created_at`
  (DB DEFAULT now() -- Postgres's own now(), not Python's, so a caller
  never needs src.core.clock to construct one).
- `append` returns `None` when a row with the same idempotency_key and
  state='INTENT' already exists (mirrors "ON CONFLICT DO NOTHING -> 0 rows
  -> this attempt already exists"), and returns the new integer ledger_id
  otherwise. This directly implements the write-ordering protocol's dedup
  step, and is the one CLAUDE.md-relevant behaviour in this file: the same
  INTENT must never insert twice, or a retried write could double-charge.
- `latest_state` raises LookupError for a mandate with no mandate_lifecycle
  rows at all -- every real mandate has at least a CREATED row, so silently
  returning an "unknown" sentinel would hide a real bug.
"""
from __future__ import annotations

import pathlib
import re
from datetime import datetime, timedelta, timezone

import pytest

from src.core.types import MandateState
from src.ledger.store import (
    LedgerEntry, append, find_by_key, latest_state, replay,
    record_lifecycle_event, record_ingested_event,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE_SRC = ROOT / "src" / "ledger" / "store.py"


def _entry(*, key, state, decision_sha256, mandate_id="M-STORE",
           cycle_id=1, attempt_index=1, action="ATTEMPT",
           amount_paise=10000, profile="strict", payload_sha256="0" * 64,
           **overrides) -> LedgerEntry:
    fields = dict(
        idempotency_key=key,
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        attempt_index=attempt_index,
        action=action,
        state=state,
        amount_paise=amount_paise,
        profile=profile,
        payload_sha256=payload_sha256,
        decision_sha256=decision_sha256,
    )
    fields.update(overrides)
    return LedgerEntry(**fields)


def _insert_lifecycle_row(conn, *, event_id, mandate_id, state, effective_at, source="test"):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO mandate_lifecycle (event_id, mandate_id, state, source, effective_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (event_id, mandate_id, state, source, effective_at),
        )


# --- append: the duplicate-intent case is the money-critical one -----------

def test_append_returns_int_ledger_id_for_a_new_intent(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-store-1")
    entry = _entry(key="store-intent-1", state="INTENT", decision_sha256=decision_sha)

    ledger_id = append(pg_schema.conn, entry)

    assert isinstance(ledger_id, int)
    assert ledger_id > 0


def test_append_duplicate_intent_returns_none_and_does_not_double_insert(pg_schema, seed_plan):
    """The case that actually matters: appending the same INTENT twice
    (e.g. a retried write after a crash) must never create two rows -- two
    rows here is exactly what would let a retry double-charge."""
    decision_sha = seed_plan("plan-store-2")
    entry = _entry(key="store-intent-dup", state="INTENT", decision_sha256=decision_sha)

    first = append(pg_schema.conn, entry)
    second = append(pg_schema.conn, entry)

    assert isinstance(first, int)
    assert second is None

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ledger WHERE idempotency_key = %s AND state = 'INTENT'",
            ("store-intent-dup",),
        )
        assert cur.fetchone()[0] == 1


def test_append_sent_then_result_same_key_both_succeed(pg_schema, seed_plan):
    """SENT and RESULT rows sharing the INTENT row's key are later stages
    of the SAME attempt, not duplicates -- the partial unique index only
    fires on state='INTENT', so all three appends must succeed."""
    decision_sha = seed_plan("plan-store-3")
    key = "store-stage-1"

    intent_id = append(pg_schema.conn, _entry(key=key, state="INTENT", decision_sha256=decision_sha))
    sent_id = append(pg_schema.conn, _entry(key=key, state="SENT", decision_sha256=decision_sha))
    result_id = append(
        pg_schema.conn,
        _entry(key=key, state="RESULT", decision_sha256=decision_sha, outcome="RECOVERED"),
    )

    assert None not in (intent_id, sent_id, result_id)
    assert len({intent_id, sent_id, result_id}) == 3


# --- replay -------------------------------------------------------------------

def test_replay_returns_entries_for_mandate_in_insertion_order(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-store-4")
    key = "store-replay-1"
    append(pg_schema.conn, _entry(key=key, state="INTENT", decision_sha256=decision_sha, mandate_id="M-REPLAY"))
    append(pg_schema.conn, _entry(key=key, state="SENT", decision_sha256=decision_sha, mandate_id="M-REPLAY"))
    append(
        pg_schema.conn,
        _entry(key=key, state="RESULT", decision_sha256=decision_sha, mandate_id="M-REPLAY", outcome="RECOVERED"),
    )

    entries = replay(pg_schema.conn, "M-REPLAY")

    assert len(entries) == 3
    assert [e.state for e in entries] == ["INTENT", "SENT", "RESULT"]


def test_replay_empty_mandate_returns_empty_list_not_none(pg_schema):
    entries = replay(pg_schema.conn, "M-NEVER-SEEN")
    assert entries == []


# --- find_by_key: most recent stage for a key, not the first ----------------

def test_find_by_key_returns_none_when_key_unknown(pg_schema):
    assert find_by_key(pg_schema.conn, "no-such-key-at-all") is None


def test_find_by_key_returns_the_most_recent_stage(pg_schema, seed_plan):
    decision_sha = seed_plan("plan-store-5")
    key = "store-findkey-1"
    append(pg_schema.conn, _entry(key=key, state="INTENT", decision_sha256=decision_sha))
    append(pg_schema.conn, _entry(key=key, state="SENT", decision_sha256=decision_sha))
    append(
        pg_schema.conn,
        _entry(key=key, state="RESULT", decision_sha256=decision_sha, outcome="RECOVERED"),
    )

    found = find_by_key(pg_schema.conn, key)

    assert found is not None
    assert found.state == "RESULT"
    assert found.outcome == "RECOVERED"


# --- latest_state: orders by effective_at, not insertion/PK order ----------

def test_latest_state_orders_by_effective_at_not_insertion_order(pg_schema):
    """Insert the LATER effective_at row FIRST and the EARLIER one SECOND
    -- if latest_state ever falls back to insertion or PK order, this is
    the test that catches it."""
    mandate_id = "M-LIFECYCLE-1"
    later = datetime(2026, 6, 1, tzinfo=timezone.utc)
    earlier = later - timedelta(days=10)

    _insert_lifecycle_row(pg_schema.conn, event_id="evt-2", mandate_id=mandate_id,
                           state="ACTIVE", effective_at=later)
    _insert_lifecycle_row(pg_schema.conn, event_id="evt-1", mandate_id=mandate_id,
                           state="CREATED", effective_at=earlier)

    assert latest_state(pg_schema.conn, mandate_id) == MandateState.ACTIVE


def test_latest_state_raises_for_mandate_with_no_lifecycle_rows(pg_schema):
    with pytest.raises(LookupError):
        latest_state(pg_schema.conn, "M-NEVER-SEEN")


# --- record_lifecycle_event: append-only mandate state transitions --------

def test_record_lifecycle_event_inserts_fresh_row_returns_mandate_state(pg_schema):
    """A new event_id should insert one row and return the MandateState
    enum member matching the state string passed in."""
    effective_at = datetime(2026, 6, 1, tzinfo=timezone.utc)

    result = record_lifecycle_event(
        pg_schema.conn,
        event_id="evt-lifecycle-1",
        mandate_id="M-LIFECYCLE-FRESH",
        state="ACTIVE",
        source="WEBHOOK",
        effective_at=effective_at,
    )

    assert result == MandateState.ACTIVE
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, mandate_id, state, source, effective_at FROM mandate_lifecycle "
            "WHERE event_id = %s",
            ("evt-lifecycle-1",),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "evt-lifecycle-1"
    assert row[1] == "M-LIFECYCLE-FRESH"
    assert row[2] == "ACTIVE"
    assert row[3] == "WEBHOOK"


def test_record_lifecycle_event_duplicate_event_id_ignores_second_call_returns_first_state(pg_schema):
    """Simulates a retried webhook delivery: calling with the same event_id
    but different state the second time must return the FIRST call's state,
    ignore the second call's state entirely, and leave exactly one row in
    the table (the ON CONFLICT DO NOTHING behavior)."""
    effective_at_1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    effective_at_2 = datetime(2026, 6, 2, tzinfo=timezone.utc)

    result_1 = record_lifecycle_event(
        pg_schema.conn,
        event_id="evt-lifecycle-dup",
        mandate_id="M-LIFECYCLE-DUP",
        state="CREATED",
        source="WEBHOOK",
        effective_at=effective_at_1,
    )
    result_2 = record_lifecycle_event(
        pg_schema.conn,
        event_id="evt-lifecycle-dup",
        mandate_id="M-LIFECYCLE-DUP-IGNORED",
        state="ACTIVE",
        source="INTERNAL",
        effective_at=effective_at_2,
    )

    assert result_1 == MandateState.CREATED
    assert result_2 == MandateState.CREATED  # Returns what was stored first, not second
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM mandate_lifecycle WHERE event_id = %s",
            ("evt-lifecycle-dup",),
        )
        count = cur.fetchone()[0]
    assert count == 1


def test_record_lifecycle_event_all_mandate_states(pg_schema):
    """Verify that every MandateState enum member can be inserted and
    read back without error."""
    for state in MandateState:
        effective_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        result = record_lifecycle_event(
            pg_schema.conn,
            event_id=f"evt-state-{state.value}",
            mandate_id=f"M-STATE-{state.value}",
            state=state.value,
            source="TEST",
            effective_at=effective_at,
        )
        assert result == state


# --- record_ingested_event: append-only ingest landing zone ---------------

def test_record_ingested_event_minimal_fields_inserts_with_nulls(pg_schema):
    """Call with only required fields (event_id, event_type,
    raw_payload_sha256); all optional fields None. Returns None (fire and
    forget), and the row's nullable columns are actually NULL."""
    record_ingested_event(
        pg_schema.conn,
        event_id="evt-ingest-minimal",
        event_type="payment.failed",
        raw_payload_sha256="a" * 64,
    )

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, event_type, mandate_id, provider_ref, decline_code, "
            "decline_text, decline_class, cause_prior, amount_paise, raw_payload_sha256 "
            "FROM ingested_event WHERE event_id = %s",
            ("evt-ingest-minimal",),
        )
        row = cur.fetchone()
    assert row is not None
    event_id, event_type, mandate_id, provider_ref, decline_code, decline_text, decline_class, cause_prior, amount_paise, raw_payload_sha256 = row
    assert event_id == "evt-ingest-minimal"
    assert event_type == "payment.failed"
    assert mandate_id is None
    assert provider_ref is None
    assert decline_code is None
    assert decline_text is None
    assert decline_class is None
    assert cause_prior is None
    assert amount_paise is None
    assert raw_payload_sha256 == "a" * 64


def test_record_ingested_event_all_fields_populated_roundtrips_exactly(pg_schema):
    """Insert a row with every field populated (realistic decline event).
    Verify that all columns round-trip exactly, and specifically confirm
    that amount_paise comes back as a plain Python int, never float."""
    cause_prior_json = '{"CANT_PAY_NOW": 0.8, "CANT_PAY_EVER": 0.1, "WONT_PAY": 0.1}'
    amount_paise = 150000

    record_ingested_event(
        pg_schema.conn,
        event_id="evt-ingest-full",
        event_type="payment.failed",
        raw_payload_sha256="b" * 64,
        mandate_id="M-INGEST-FULL",
        provider_ref="pay_ABC123DEF456",
        decline_code="INSUFFICIENT_FUNDS",
        decline_text="Insufficient balance",
        decline_class="INSUFFICIENT_FUNDS",
        cause_prior_json=cause_prior_json,
        taxonomy_version="v1",
        prior_version="v2",
        amount_paise=amount_paise,
    )

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, event_type, mandate_id, provider_ref, decline_code, "
            "decline_text, decline_class, cause_prior, taxonomy_version, "
            "prior_version, amount_paise, raw_payload_sha256 "
            "FROM ingested_event WHERE event_id = %s",
            ("evt-ingest-full",),
        )
        row = cur.fetchone()
    assert row is not None
    (event_id, event_type, mandate_id, provider_ref, decline_code, decline_text,
     decline_class, cause_prior, taxonomy_version, prior_version,
     amount_paise_read, raw_payload_sha256) = row
    assert event_id == "evt-ingest-full"
    assert event_type == "payment.failed"
    assert mandate_id == "M-INGEST-FULL"
    assert provider_ref == "pay_ABC123DEF456"
    assert decline_code == "INSUFFICIENT_FUNDS"
    assert decline_text == "Insufficient balance"
    assert decline_class == "INSUFFICIENT_FUNDS"
    assert cause_prior == cause_prior_json
    assert taxonomy_version == "v1"
    assert prior_version == "v2"
    assert isinstance(amount_paise_read, int), f"amount_paise must be int, got {type(amount_paise_read)}"
    assert amount_paise_read == amount_paise
    assert raw_payload_sha256 == "b" * 64


def test_record_ingested_event_duplicate_event_id_ignores_second_call(pg_schema):
    """Call record_ingested_event() twice with the same event_id but
    different field values the second time. Must not raise, SELECT count(*)
    must still be 1, and the row's contents must match the FIRST call."""
    record_ingested_event(
        pg_schema.conn,
        event_id="evt-ingest-dup",
        event_type="payment.failed",
        raw_payload_sha256="c" * 64,
        mandate_id="M-INGEST-DUP",
        amount_paise=100000,
    )
    record_ingested_event(
        pg_schema.conn,
        event_id="evt-ingest-dup",
        event_type="subscription.charged",
        raw_payload_sha256="d" * 64,
        mandate_id="M-INGEST-DUP-IGNORED",
        amount_paise=200000,
    )

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ingested_event WHERE event_id = %s",
            ("evt-ingest-dup",),
        )
        count = cur.fetchone()[0]
    assert count == 1

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT event_type, mandate_id, amount_paise FROM ingested_event "
            "WHERE event_id = %s",
            ("evt-ingest-dup",),
        )
        row = cur.fetchone()
    assert row[0] == "payment.failed"  # First call's event_type, not second
    assert row[1] == "M-INGEST-DUP"  # First call's mandate_id
    assert row[2] == 100000  # First call's amount_paise


def test_record_ingested_event_negative_amount_paise_raises_check_violation(pg_schema):
    """The schema has CHECK (amount_paise IS NULL OR amount_paise >= 0).
    Attempt to insert a negative amount_paise must raise CheckViolation."""
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        record_ingested_event(
            pg_schema.conn,
            event_id="evt-ingest-negative",
            event_type="payment.failed",
            raw_payload_sha256="e" * 64,
            amount_paise=-1000,
        )


# --- record_normalized_decline and find_normalized_decline ---------------

def test_record_normalized_decline_then_find_normalized_decline_round_trips(pg_schema):
    """record_normalized_decline() followed by find_normalized_decline() must
    round-trip all five fields correctly."""
    from src.ledger.store import record_normalized_decline, find_normalized_decline

    # First insert an ingested_event so the FK succeeds
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_event (event_id, event_type, raw_payload_sha256)
            VALUES (%s, %s, %s)
            """,
            ("evt-normalized-roundtrip", "payment.failed", "a" * 64),
        )

    # Record a normalized decline
    record_normalized_decline(
        pg_schema.conn,
        event_id="evt-normalized-roundtrip",
        value="INSUFFICIENT_FUNDS",
        confidence=0.92,
        normalizer_version="v1",
        model_id="model-abc123",
        raw_sha256="b" * 64,
    )

    # Find it back
    row = find_normalized_decline(pg_schema.conn, "evt-normalized-roundtrip", "v1")

    assert row is not None, "Row should be found"
    assert row.event_id == "evt-normalized-roundtrip"
    assert row.value == "INSUFFICIENT_FUNDS"
    assert row.confidence == 0.92
    assert row.normalizer_version == "v1"
    assert row.model_id == "model-abc123"
    assert row.raw_sha256 == "b" * 64


def test_record_normalized_decline_duplicate_silent_no_op_first_write_wins(pg_schema):
    """Calling record_normalized_decline() twice with identical
    (event_id, normalizer_version) but a DIFFERENT value the second time --
    the FIRST write wins (ON CONFLICT DO NOTHING means the second is silently
    dropped). This is intentional dedupe behavior, not a bug: a retried
    normalization with the same normalizer version returns the original
    verdict. Verify both by checking fetch and count."""
    from src.ledger.store import record_normalized_decline, find_normalized_decline

    # Insert ingested_event
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_event (event_id, event_type, raw_payload_sha256)
            VALUES (%s, %s, %s)
            """,
            ("evt-norm-dup-wins", "payment.failed", "c" * 64),
        )

    # First write
    record_normalized_decline(
        pg_schema.conn,
        event_id="evt-norm-dup-wins",
        value="INSUFFICIENT_FUNDS",
        confidence=0.9,
        normalizer_version="v1",
        model_id="model-first",
        raw_sha256="d" * 64,
    )

    # Second write with DIFFERENT value but same key -- silently dropped
    record_normalized_decline(
        pg_schema.conn,
        event_id="evt-norm-dup-wins",
        value="CARD_EXPIRED",
        confidence=0.7,
        normalizer_version="v1",
        model_id="model-second",
        raw_sha256="e" * 64,
    )

    # Verify first value persists
    row = find_normalized_decline(pg_schema.conn, "evt-norm-dup-wins", "v1")
    assert row is not None
    assert row.value == "INSUFFICIENT_FUNDS", "Value should be from FIRST write"
    assert row.confidence == 0.9, "confidence should be from FIRST write"
    assert row.model_id == "model-first", "model_id should be from FIRST write"

    # Verify count is exactly 1
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM normalized_decline WHERE event_id = %s AND normalizer_version = %s",
            ("evt-norm-dup-wins", "v1"),
        )
        count = cur.fetchone()[0]
    assert count == 1, f"Expected 1 row, got {count}"


def test_find_normalized_decline_nonexistent_returns_none(pg_schema):
    """find_normalized_decline() on an (event_id, normalizer_version) pair
    that was never written must return None, not raise an exception."""
    from src.ledger.store import find_normalized_decline

    row = find_normalized_decline(pg_schema.conn, "evt-never-seen", "v999")

    assert row is None, "find_normalized_decline should return None for nonexistent row"


def test_record_normalized_decline_new_version_leaves_old_version_untouched(pg_schema):
    """Prompt bump creates a new normalizer_version and a NEW row for the
    same event_id, without touching or replacing the old row. Both rows must
    coexist and be readable, proving the full lineage of normalization
    decisions for audit trails."""
    from src.ledger.store import record_normalized_decline, find_normalized_decline

    # Insert ingested_event
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_event (event_id, event_type, raw_payload_sha256)
            VALUES (%s, %s, %s)
            """,
            ("evt-norm-lineage", "payment.failed", "f" * 64),
        )

    # v1 normalizer run
    record_normalized_decline(
        pg_schema.conn,
        event_id="evt-norm-lineage",
        value="INSUFFICIENT_FUNDS",
        confidence=0.6,
        normalizer_version="v1",
        model_id="model-v1",
        raw_sha256="g" * 64,
    )

    # v2 normalizer run (prompt bump)
    record_normalized_decline(
        pg_schema.conn,
        event_id="evt-norm-lineage",
        value="LOW_BALANCE",
        confidence=0.95,
        normalizer_version="v2",
        model_id="model-v2",
        raw_sha256="h" * 64,
    )

    # Fetch both explicitly and verify they coexist
    row_v1 = find_normalized_decline(pg_schema.conn, "evt-norm-lineage", "v1")
    row_v2 = find_normalized_decline(pg_schema.conn, "evt-norm-lineage", "v2")

    assert row_v1 is not None, "v1 row should still exist"
    assert row_v2 is not None, "v2 row should exist"
    assert row_v1.value == "INSUFFICIENT_FUNDS", "v1 should have its original value"
    assert row_v2.value == "LOW_BALANCE", "v2 should have its new value"
    assert row_v1.model_id == "model-v1"
    assert row_v2.model_id == "model-v2"


def test_record_normalized_decline_fk_violation_on_nonexistent_event_id(pg_schema):
    """record_normalized_decline() with an event_id that has no matching
    ingested_event row must raise the FK constraint violation, proving the
    referential integrity is enforced."""
    from src.ledger.store import record_normalized_decline
    import psycopg

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        record_normalized_decline(
            pg_schema.conn,
            event_id="evt-norm-fk-fail",
            value="SOME_VALUE",
            confidence=0.8,
            normalizer_version="v1",
            model_id="model-test",
            raw_sha256="i" * 64,
        )


# --- source guard: append-only in the code, not just the DB ----------------

def test_store_never_updates_or_deletes_the_ledger_table():
    """Mirrors the schema.sql source guard. Postgres itself will happily
    allow UPDATE ledger / DELETE FROM ledger -- there is no DB trigger
    blocking it, by design (see test_schema.py's module docstring). This
    is the other half of the enforcement: the application code that would
    have to issue such a statement must never contain one."""
    text = STORE_SRC.read_text(encoding="utf-8")
    banned = [
        r'UPDATE\s+ledger\b',
        r'UPDATE\s+"ledger"',
        r'DELETE\s+FROM\s+ledger\b',
        r'DELETE\s+FROM\s+"ledger"',
    ]
    for pattern in banned:
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None, f"store.py must never UPDATE/DELETE ledger, found: {match.group(0)!r}"


def test_store_never_updates_or_deletes_new_tables():
    """The two new functions (record_lifecycle_event, record_ingested_event)
    must be append-only via ON CONFLICT DO NOTHING, not ON CONFLICT DO UPDATE.
    Verify that the source never contains UPDATE against the new tables."""
    text = STORE_SRC.read_text(encoding="utf-8")
    banned = [
        r'UPDATE\s+ingested_event\b',
        r'UPDATE\s+"ingested_event"',
        r'UPDATE\s+mandate_lifecycle\b',
        r'UPDATE\s+"mandate_lifecycle"',
    ]
    for pattern in banned:
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None, f"store.py must never UPDATE ingested_event or mandate_lifecycle, found: {match.group(0)!r}"


def test_store_never_updates_or_deletes_normalized_decline():
    """normalized_decline is append-only like ledger. Verify that store.py's
    source never contains UPDATE or DELETE against normalized_decline."""
    text = STORE_SRC.read_text(encoding="utf-8")
    banned = [
        r'UPDATE\s+normalized_decline\b',
        r'UPDATE\s+"normalized_decline"',
        r'DELETE\s+FROM\s+normalized_decline\b',
        r'DELETE\s+FROM\s+"normalized_decline"',
    ]
    for pattern in banned:
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is None, f"store.py must never UPDATE/DELETE normalized_decline, found: {match.group(0)!r}"
