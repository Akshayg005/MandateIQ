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
from src.ledger.store import LedgerEntry, append, find_by_key, latest_state, replay

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
