"""
src/ingest/lifecycle_route.py -- record Razorpay subscription state transitions
into mandate_lifecycle, mapping Razorpay's status vocabulary to this system's
MandateState enum.

Design decisions this test file pins:

- record() takes an explicit psycopg connection as first arg, not managing
  a module-level connection (mirrors src/ledger/store.py's pattern).
- record() reads effective_at ONLY from the payload's top-level created_at
  field (Unix seconds), never from src.core.clock.now(), so a frozen test
  clock never changes what the system records as the effective time. This
  matters because Razorpay's created_at is when the status transition
  happened at Razorpay's end, not when we observe/process it.
- Razorpay status strings map to MandateState via a fixed table: created,
  active, paused, cancelled, expired, completed are mapped; authenticated,
  pending, halted, and any unrecognised string yield None and write nothing.
- record() never touches committed_schedule, ever.
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import pytest
from freezegun import freeze_time

from src.core.clock import set_frozen
from src.core.types import MandateState

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LIFECYCLE_ROUTE_SRC = ROOT / "src" / "ingest" / "lifecycle_route.py"


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    """A clock left frozen by one test must never leak into the next."""
    yield
    set_frozen(None)


def _subscription_payload(*, event_id, mandate_id, status, created_at_unix):
    """Build a realistic Razorpay subscription webhook envelope.

    Args:
        event_id: e.g. "sub-evt-123" (for event_id row, not used in payload itself)
        mandate_id: e.g. "sub_ABC123" (the Razorpay subscription id)
        status: e.g. "active", "cancelled", "paused"
        created_at_unix: Unix timestamp (seconds) at envelope's top level
    """
    return {
        "entity": "event",
        "account_id": "acc_test123",
        "event": "subscription.activated",
        "contains": ["subscription"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": mandate_id,
                    "status": status,
                    "notes": {},
                }
            }
        },
        "created_at": created_at_unix,
    }


# --- Mapped statuses: should insert and return the corresponding MandateState ----

@pytest.mark.parametrize("razorpay_status,expected_mandate_state", [
    ("created", MandateState.CREATED),
    ("active", MandateState.ACTIVE),
    ("paused", MandateState.PAUSED),
    ("cancelled", MandateState.REVOKED),
    ("expired", MandateState.EXPIRED),
    ("completed", MandateState.COMPLETED),
])
def test_mapped_status_inserts_and_returns_mandate_state(
    pg_schema, razorpay_status, expected_mandate_state
):
    """For each of the 6 mapped Razorpay statuses, record() should:
    - Return the correct MandateState enum member
    - Insert exactly one row into mandate_lifecycle with source='WEBHOOK'
    - Preserve the mandate_id and effective_at from the payload
    """
    from src.ingest.lifecycle_route import record

    event_id = f"test-mapped-{razorpay_status}"
    mandate_id = f"sub_mapped_{razorpay_status}"
    created_at_unix = 1798412345  # Some fixed timestamp
    effective_dt = datetime.fromtimestamp(created_at_unix, tz=timezone.utc)

    payload = _subscription_payload(
        event_id=event_id,
        mandate_id=mandate_id,
        status=razorpay_status,
        created_at_unix=created_at_unix,
    )

    result = record(pg_schema.conn, event_id, payload)

    assert result == expected_mandate_state

    # Verify the row was inserted
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            SELECT event_id, mandate_id, state, source, effective_at
            FROM mandate_lifecycle
            WHERE event_id = %s
            """,
            (event_id,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[0] == event_id
    assert row[1] == mandate_id
    assert row[2] == expected_mandate_state.value
    assert row[3] == "WEBHOOK"
    assert row[4] == effective_dt


# --- Unmapped statuses: should return None and write nothing ----

@pytest.mark.parametrize("unmapped_status", [
    "authenticated",   # pre-billing state, no equivalent in 6-state model
    "pending",         # mid-retry state (Razorpay's own auto-retry)
    "halted",          # Razorpay's retry budget exhausted
    "unknown_status",  # unrecognised string
])
def test_unmapped_status_returns_none_and_writes_nothing(pg_schema, unmapped_status):
    """For unknown/unmapped statuses, record() should:
    - Return None (honest non-guess)
    - Write NO row to mandate_lifecycle (no call to record_lifecycle_event)
    """
    from src.ingest.lifecycle_route import record

    event_id = f"test-unmapped-{unmapped_status}"
    mandate_id = f"sub_unmapped_{unmapped_status}"
    created_at_unix = 1798412345

    payload = _subscription_payload(
        event_id=event_id,
        mandate_id=mandate_id,
        status=unmapped_status,
        created_at_unix=created_at_unix,
    )

    result = record(pg_schema.conn, event_id, payload)

    assert result is None

    # Verify no row was written
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM mandate_lifecycle WHERE event_id = %s",
            (event_id,),
        )
        count = cur.fetchone()[0]

    assert count == 0


# --- Idempotency: duplicate event_id ----

def test_duplicate_event_id_is_idempotent(pg_schema):
    """Calling record() twice with the same event_id and payload should
    not raise an exception, and should result in exactly one row in the
    mandate_lifecycle table (the second call is a no-op due to the
    PRIMARY KEY on event_id)."""
    from src.ingest.lifecycle_route import record

    event_id = "test-dup-lifecycle"
    mandate_id = "sub_test_dup"
    created_at_unix = 1798412345

    payload = _subscription_payload(
        event_id=event_id,
        mandate_id=mandate_id,
        status="active",
        created_at_unix=created_at_unix,
    )

    # Call twice
    result1 = record(pg_schema.conn, event_id, payload)
    result2 = record(pg_schema.conn, event_id, payload)

    # Both should return the same state
    assert result1 == MandateState.ACTIVE
    assert result2 == MandateState.ACTIVE

    # But only one row should exist
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM mandate_lifecycle WHERE event_id = %s",
            (event_id,),
        )
        count = cur.fetchone()[0]

    assert count == 1


# --- Effective timestamp: comes from payload, not frozen clock ----

def test_effective_at_comes_from_payload_not_frozen_clock(pg_schema):
    """CRITICAL: effective_at must come from the payload's created_at field,
    not from src.core.clock.now(). This ensures that freezing the clock
    (for testing the 24h commitment lag, etc.) never changes what timestamp
    the system records for Razorpay's event.

    Set payload.created_at to 2026-01-01, freeze the clock to 2030-06-15,
    call record(), and verify the DB row has 2026-01-01, not 2030-06-15."""
    from src.ingest.lifecycle_route import record

    event_id = "test-effective-at-payload"
    mandate_id = "sub_test_payload_ts"

    # Payload says the status change happened on 2026-01-01 00:00:00 UTC
    payload_timestamp_unix = int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    expected_effective_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    payload = _subscription_payload(
        event_id=event_id,
        mandate_id=mandate_id,
        status="active",
        created_at_unix=payload_timestamp_unix,
    )

    # Freeze the clock to a wildly different time (2030-06-15)
    frozen_to = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    set_frozen(frozen_to)

    # Call record() with the frozen clock
    result = record(pg_schema.conn, event_id, payload)

    assert result == MandateState.ACTIVE

    # Read back the row and check effective_at is the PAYLOAD timestamp, not the frozen clock
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT effective_at FROM mandate_lifecycle WHERE event_id = %s",
            (event_id,),
        )
        row = cur.fetchone()

    assert row is not None
    recorded_effective_at = row[0]

    # Must be 2026-01-01, not 2030-06-15
    assert recorded_effective_at == expected_effective_at


# --- Source guard: never touches committed_schedule ----

def test_never_touches_committed_schedule():
    """Verify that lifecycle_route.py never reads or writes committed_schedule.
    This is a source-level guard: the string 'committed_schedule' must not
    appear anywhere in the file (module docstring doesn't count, only code)."""
    text = LIFECYCLE_ROUTE_SRC.read_text(encoding="utf-8")

    # Split by module docstring to exclude the docstring itself
    parts = text.split('"""', 2)
    if len(parts) >= 3:
        code_part = parts[2]  # Everything after the module docstring
    else:
        code_part = text

    assert "committed_schedule" not in code_part, (
        "lifecycle_route.py must never touch committed_schedule"
    )
