"""
src/ingest/dedupe.py -- webhook event deduplication via webhook_event table.

Design decisions this test file pins:

- `seen(conn, event_id)` is a pure existence check returning bool. Used to
  skip processing a webhook event if we have already seen this exact event_id
  before (a retried delivery from Razorpay).
- `mark(conn, event_id, event_type)` inserts via ON CONFLICT DO NOTHING and
  returns True iff a new row was actually inserted (this is the first time we
  have seen this event_id), False if the row was already marked. This directly
  implements idempotent webhook processing: the first call to mark() carries
  the real event; retries are no-ops.
- scope is event_id-based only. Different event_types with different event_ids
  are separate events; same event_type with different event_ids do not affect
  each other's seen() / mark() status.
"""
from __future__ import annotations

import pytest

from src.ingest.dedupe import seen, mark


# --- seen: pure existence check ----------------------------------------------

def test_seen_is_false_for_a_never_marked_event(pg_schema):
    """A fresh event_id that was never passed to mark() must return False."""
    result = seen(pg_schema.conn, "evt-never-seen")
    assert result is False


def test_seen_is_true_after_mark(pg_schema):
    """After mark() inserts an event_id, seen() must return True for that
    same event_id."""
    mark(pg_schema.conn, "evt-mark-then-see", "payment.failed")
    result = seen(pg_schema.conn, "evt-mark-then-see")
    assert result is True


# --- mark: idempotent insert via ON CONFLICT DO NOTHING --------------------

def test_mark_returns_true_on_first_call(pg_schema):
    """Calling mark() for the first time with a fresh event_id must return
    True, indicating that a new row was inserted."""
    result = mark(pg_schema.conn, "evt-mark-first", "payment.failed")
    assert result is True


def test_mark_returns_false_on_repeat_call(pg_schema):
    """Calling mark() twice with the same event_id must return False on the
    second call (row already exists, so ON CONFLICT DO NOTHING inserted 0
    rows). The event_type can differ the second time, proving it is ignored
    on a duplicate event_id."""
    first = mark(pg_schema.conn, "evt-mark-repeat", "payment.failed")
    second = mark(pg_schema.conn, "evt-mark-repeat", "subscription.charged")
    assert first is True
    assert second is False


def test_mark_is_idempotent_no_duplicate_row(pg_schema):
    """Calling mark() twice with the same event_id must not create two rows.
    Verify with a raw SELECT count(*)."""
    mark(pg_schema.conn, "evt-mark-once", "payment.failed")
    mark(pg_schema.conn, "evt-mark-once", "payment.failed")

    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM webhook_event WHERE event_id = %s",
            ("evt-mark-once",),
        )
        count = cur.fetchone()[0]
    assert count == 1


def test_seen_and_mark_are_scoped_by_event_id_not_event_type(pg_schema):
    """Marking evt-a with any event_type must not affect the seen() status
    of evt-b, even if evt-b carries the same event_type. Deduplication is
    event_id-scoped, not event_type-scoped."""
    mark(pg_schema.conn, "evt-a", "payment.failed")
    result_b = seen(pg_schema.conn, "evt-b")
    assert result_b is False

    # Also verify evt-b can be marked independently
    mark(pg_schema.conn, "evt-b", "payment.failed")
    result_b_after = seen(pg_schema.conn, "evt-b")
    assert result_b_after is True

    # And evt-a is still marked
    result_a_still = seen(pg_schema.conn, "evt-a")
    assert result_a_still is True
