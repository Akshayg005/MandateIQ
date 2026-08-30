"""Void-and-reissue for a committed_schedule row overtaken by events.

THE SENT-ROW RULE, a deliberate reinterpretation of the letter of
PLAN_DETAIL.md's B9 file table -- logged in DECISIONS.md, 2026-08-30, B9.
The file table's own words are "must not void a key that already has an
INTENT row"; read literally, that is impossible to satisfy at the same
time as section 3's write-ordering protocol, whose step 2a is the late
lifecycle read aborting with "abort AND VOID" -- and step 2a runs strictly
after the INTENT row is already written (step 1). Every key void() is ever
asked to void already has an INTENT row, by construction; the literal rule
would make voiding never legal at all.

What this module actually enforces instead: void() refuses only when a
SENT row exists. The distinction that matters is not "does an INTENT row
exist" but "could a Razorpay call have been made" -- and INTENT alone
answers that "no". src/execute/executor.py's own pre-call abort (2a) holds
the lease and wrote that very INTENT row itself, in the same process, a
moment earlier; it knows FIRST-HAND, not by inference, that no send has
happened yet. That is categorically different from src/execute/recover.py,
which is a separate reconciliation pass that may run in a different
process, any amount of time later, inferring from rows alone -- it must
never assume INTENT-without-SENT means nothing was sent (DECISIONS.md,
2026-08-27, B3's idempotency spike: recovery is by ASKING the provider,
never by guessing from local state), so it never calls void() and only
ever calls find_by_receipt.

reissue() takes the ORIGINAL row's key, not a caller-held ScheduledAttempt
object, and re-reads committed_schedule fresh -- a caller's in-memory copy
could be stale (voided by a concurrent process since it was read), and
this module follows the same "trust the DB, not a belief about it"
discipline PLAN_DETAIL.md's late-read principle applies everywhere else.
"""
from __future__ import annotations

from datetime import datetime

import psycopg

from src.core import clock
from src.execute.keys import ScheduledAttempt, key_for
from src.ledger.store import find_by_key


class VoidError(ValueError):
    """Raised by void() when a SENT row already exists for the key (must be
    resolved by asking, never by voiding), or by reissue() when there is no
    row to reissue from, or the row is not yet voided."""


_CS_COLUMNS = (
    "idempotency_key", "mandate_id", "cycle_id", "attempt_index", "generation",
    "action", "amount_paise", "profile", "decision_sha256", "scheduled_for",
    "committed_at", "voided_at", "void_reason",
)


def _read_committed_schedule(conn, key: str) -> ScheduledAttempt | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_CS_COLUMNS)} FROM committed_schedule "
            "WHERE idempotency_key = %s",
            (key,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return ScheduledAttempt(**dict(zip(_CS_COLUMNS, row)))


def void(conn, key: str, reason: str) -> None:
    """Mark the committed_schedule row for `key` voided. Raises VoidError
    if a SENT row already exists in `ledger` for this key (see module
    docstring for the SENT-row rule). Idempotent: voiding an
    already-voided row is a silent no-op that preserves the FIRST void's
    reason, never overwrites it -- mirroring store.record_lifecycle_event's
    own "first write wins" discipline on a duplicate."""
    latest = find_by_key(conn, key)
    if latest is not None and latest.state == "SENT":
        raise VoidError(
            f"{key}: a SENT row already exists in ledger -- a provider call may "
            "have been made; this must be resolved by asking (recover.py), "
            "never by voiding"
        )
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE committed_schedule SET voided_at = %s, void_reason = %s "
            "WHERE idempotency_key = %s AND voided_at IS NULL",
            (clock.now(), reason, key),
        )


def reissue(
    conn, original_key: str, *, scheduled_for: datetime, amount_paise: int | None = None
) -> ScheduledAttempt:
    """A fresh committed_schedule row at generation+1, same attempt_index,
    same mandate/cycle/decision -- everything else carried over from the
    original row except `scheduled_for` (required -- a reissue exists
    because something changed) and optionally `amount_paise`.

    A reissue does NOT spend a new NPCI slot: the attempt budget
    (src.policy.constraints.MAX_ATTEMPTS) counts distinct attempt_index
    values, never distinct keys (root CLAUDE.md; PLAN_DETAIL.md section 3)
    -- attempt_index is copied unchanged from the original.

    Raises VoidError if no row exists for `original_key`, or if it is not
    yet voided (reissue never operates on a live row -- void() it first;
    committed_one_live_per_slot would reject a live duplicate regardless,
    but this gives a clear error instead of a raw UniqueViolation).
    Raises CommitError -- via the same 24h CHECK psycopg.errors.CheckViolation
    path commit.py uses -- if `scheduled_for` is less than 24h after the
    reissue's own committed_at.
    """
    from src.execute.commit import CommitError  # local: avoids a module cycle
    # (commit.py does not import void.py, so this could live at module
    # level too, but keeping the CommitError re-raise pattern colocated
    # with its one use site makes the dependency obvious at the call site).

    current = _read_committed_schedule(conn, original_key)
    if current is None:
        raise VoidError(f"{original_key}: no committed_schedule row exists to reissue")
    if current.voided_at is None:
        raise VoidError(f"{original_key}: cannot reissue a still-live row -- void() it first")

    new_generation = current.generation + 1
    new_amount = amount_paise if amount_paise is not None else current.amount_paise
    new_key = key_for(
        mandate_id=current.mandate_id,
        cycle_id=current.cycle_id,
        attempt_index=current.attempt_index,
        generation=new_generation,
        action=current.action,
        amount_paise=new_amount,
    )
    committed_at = clock.now()

    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO committed_schedule (
                    idempotency_key, mandate_id, cycle_id, attempt_index,
                    generation, action, amount_paise, profile, decision_sha256,
                    scheduled_for, committed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    new_key, current.mandate_id, current.cycle_id, current.attempt_index,
                    new_generation, current.action, new_amount, current.profile,
                    current.decision_sha256, scheduled_for, committed_at,
                ),
            )
        except psycopg.errors.CheckViolation as exc:
            raise CommitError(
                f"{current.mandate_id}: reissue at generation={new_generation} with "
                f"scheduled_for={scheduled_for.isoformat()} is less than 24h after "
                f"committed_at={committed_at.isoformat()}"
            ) from exc

    return ScheduledAttempt(
        idempotency_key=new_key,
        mandate_id=current.mandate_id,
        cycle_id=current.cycle_id,
        attempt_index=current.attempt_index,
        generation=new_generation,
        action=current.action,
        amount_paise=new_amount,
        profile=current.profile,
        decision_sha256=current.decision_sha256,
        scheduled_for=scheduled_for,
        committed_at=committed_at,
    )
