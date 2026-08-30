"""Lease-based claiming over attempt_lease, crash-safe. An OPTIMISATION over
ledger_intent_once, not the concurrency control -- if this table were
dropped entirely, a second worker racing to send the same attempt would
still be stopped by the INTENT row's partial unique index
(ledger_intent_once); what a lease adds is stopping the second worker
BEFORE it wastes a Razorpay call, not after.

claim() is a single atomic UPSERT: INSERT ... ON CONFLICT (idempotency_key)
DO UPDATE ... WHERE attempt_lease.expires_at < %(now)s. When the WHERE
clause is false for an existing row, Postgres treats that conflicting row
as untouched (equivalent to DO NOTHING for it) and RETURNING yields no row
for the statement -- exactly the "0 rows -> lost the race" signal this
function needs, with no separate read-then-write and therefore no window
for a second caller to interleave.

Every comparison against "now" uses src.core.clock.now(), never Postgres's
own now() -- CLAUDE.md: "Nothing else calls datetime.now() -- tests must be
able to freeze the clock." A lease TTL test that could not freeze time
would need real wall-clock sleeps to prove expiry.
"""
from __future__ import annotations

from datetime import timedelta

from src.core import clock


def claim(conn, key: str, owner: str, ttl_seconds: int) -> bool:
    """Attempt to claim `key` for `owner`. Returns True if claimed (either
    the key had no lease yet, or its previous lease had expired), False if
    another owner currently holds a live lease on it."""
    now = clock.now()
    expires_at = now + timedelta(seconds=ttl_seconds)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO attempt_lease (idempotency_key, owner, expires_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (idempotency_key) DO UPDATE
                SET owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at
                WHERE attempt_lease.expires_at < %s
            RETURNING owner
            """,
            (key, owner, expires_at, now),
        )
        row = cur.fetchone()
    return row is not None and row[0] == owner


def release(conn, key: str) -> None:
    """Release `key`'s lease early, once the attempt has resolved (a
    RESULT or FAILED row written) -- so a later reconciliation pass does
    not have to wait out the TTL to notice the key is free to inspect.
    A release of a key with no lease row is a silent no-op, not an error:
    releasing something already released (e.g. by a prior crash-recovery
    pass) is the normal case, not a bug."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM attempt_lease WHERE idempotency_key = %s", (key,))


def expired(conn) -> list[str]:
    """Every idempotency_key whose lease has expired as of clock.now() --
    what recover.py's reconciliation pass scans to find INTENT rows a
    crashed worker abandoned mid-attempt."""
    now = clock.now()
    with conn.cursor() as cur:
        cur.execute("SELECT idempotency_key FROM attempt_lease WHERE expires_at < %s", (now,))
        return [row[0] for row in cur.fetchall()]
