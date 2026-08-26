"""Append-only writes and replay on top of schema.sql. Every public function
takes an explicit connection as its first argument -- there is no
module-level connection to manage, and no attempt to hide the DB.

This file must never UPDATE or DELETE a ledger row. Anything that needs to
change (voiding a schedule, recording a lifecycle transition) lives in a
different table, and a different module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.types import MandateState


@dataclass(frozen=True)
class LedgerEntry:
    """What a caller supplies to append(). Covers every NOT NULL column on
    `ledger` except ledger_id (BIGSERIAL) and created_at (DB DEFAULT now())."""

    idempotency_key: str
    mandate_id: str
    cycle_id: int
    attempt_index: int
    action: str
    state: str
    amount_paise: int
    profile: str
    payload_sha256: str
    decision_sha256: str
    provider_ref: str | None = None
    outcome: str | None = None
    decline_class: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LedgerRow:
    """A row read back from `ledger`."""

    ledger_id: int
    idempotency_key: str
    mandate_id: str
    cycle_id: int
    attempt_index: int
    action: str
    state: str
    amount_paise: int
    provider_ref: str | None
    outcome: str | None
    decline_class: str | None
    reason: str | None
    profile: str
    payload_sha256: str
    decision_sha256: str
    created_at: datetime


_LEDGER_COLUMNS = (
    "ledger_id", "idempotency_key", "mandate_id", "cycle_id", "attempt_index",
    "action", "state", "amount_paise", "provider_ref", "outcome",
    "decline_class", "reason", "profile", "payload_sha256", "decision_sha256",
    "created_at",
)


def _row_to_entry(row) -> LedgerRow:
    return LedgerRow(**dict(zip(_LEDGER_COLUMNS, row)))


def append(conn, entry: LedgerEntry) -> int | None:
    """Insert one ledger row. Returns the new ledger_id, or None if a row
    with the same idempotency_key and state='INTENT' already exists -- the
    "0 rows -> this attempt already exists" step of the write-ordering
    protocol. A retried append of the same INTENT must never create a
    second row: that is exactly what would let a retry double-charge."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ledger (
                idempotency_key, mandate_id, cycle_id, attempt_index, action,
                state, amount_paise, provider_ref, outcome, decline_class,
                reason, profile, payload_sha256, decision_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) WHERE state = 'INTENT' DO NOTHING
            RETURNING ledger_id
            """,
            (
                entry.idempotency_key, entry.mandate_id, entry.cycle_id,
                entry.attempt_index, entry.action, entry.state,
                entry.amount_paise, entry.provider_ref, entry.outcome,
                entry.decline_class, entry.reason, entry.profile,
                entry.payload_sha256, entry.decision_sha256,
            ),
        )
        row = cur.fetchone()
    return row[0] if row else None


def replay(conn, mandate_id: str) -> list[LedgerRow]:
    """Every ledger row for `mandate_id`, in insertion order."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_LEDGER_COLUMNS)} FROM ledger "
            "WHERE mandate_id = %s ORDER BY ledger_id ASC",
            (mandate_id,),
        )
        rows = cur.fetchall()
    return [_row_to_entry(row) for row in rows]


def find_by_key(conn, idempotency_key: str) -> LedgerRow | None:
    """The most recent row for `idempotency_key` -- an attempt can have an
    INTENT, then a SENT, then a RESULT row sharing one key, and this
    returns the latest stage, not the first."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_LEDGER_COLUMNS)} FROM ledger "
            "WHERE idempotency_key = %s ORDER BY ledger_id DESC LIMIT 1",
            (idempotency_key,),
        )
        row = cur.fetchone()
    return _row_to_entry(row) if row else None


def latest_state(conn, mandate_id: str) -> MandateState:
    """The mandate's current lifecycle state -- the latest mandate_lifecycle
    row by effective_at, not by insertion order. Raises LookupError if the
    mandate has no lifecycle rows at all; every real mandate has at least a
    CREATED row, so silently returning an "unknown" sentinel would hide a
    real bug rather than surface one."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state FROM mandate_lifecycle WHERE mandate_id = %s "
            "ORDER BY effective_at DESC LIMIT 1",
            (mandate_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"no mandate_lifecycle rows for mandate_id={mandate_id!r}")
    return MandateState(row[0])
