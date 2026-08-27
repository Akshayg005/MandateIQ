"""Event-level idempotency for inbound webhooks, backed by `webhook_event`.

Deliberately type-agnostic -- dedupe on event_id alone, never on payload
hash (a retried webhook delivery legitimately carries the identical
event_id and the identical body; that's the case this exists to catch,
not a case to be suspicious of) and never on anything from
src/classify/ (coupling this to DeclineClass/Cause would leak taxonomy
concerns into a primitive every event type needs).

This is an optimisation, not the sole safety net: every downstream writer
(record_ingested_event, record_lifecycle_event) carries its own
ON CONFLICT (event_id) DO NOTHING, mirroring how attempt_lease is an
optimisation over ledger_intent_once rather than the concurrency control.
"""
from __future__ import annotations


def seen(conn, event_id: str) -> bool:
    """True iff `event_id` has already been marked."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM webhook_event WHERE event_id = %s", (event_id,))
        return cur.fetchone() is not None


def mark(conn, event_id: str, event_type: str) -> bool:
    """Record `event_id` as seen. Returns True iff this call actually
    inserted a new row (first time this event_id has been marked), False
    if it was already marked -- on a duplicate, event_type is ignored."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO webhook_event (event_id, event_type)
            VALUES (%s, %s)
            ON CONFLICT (event_id) DO NOTHING
            RETURNING event_id
            """,
            (event_id, event_type),
        )
        return cur.fetchone() is not None
