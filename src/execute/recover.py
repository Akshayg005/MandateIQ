"""INTENT-without-RESULT reconciliation, plus the UNCONFIRMED backoff loop.
Run on every executor start (PLAN_DETAIL.md section 3, step 5 -- "for each
K with an INTENT row, no RESULT row, and an expired lease").

Two passes, both run on every call:

  1. DANGLING -- keys whose latest ledger row is INTENT or SENT (no
     terminal RESULT/FAILED yet) and which no unexpired lease is held on.
     Discovered from the LEDGER, not from the lease table: see
     _dangling_keys' docstring and POSTMORTEM.md incident 4, where keying
     this scan on the lease table lost a job permanently. This is the shape
     a hard process crash leaves, and -- by executor.py's own deliberate
     design (see that module's docstring) -- the exact same shape a
     RazorpayClientError's AMBIGUOUS failure leaves too: the two cases
     share one resolution mechanism on purpose. Resolved by ASKING
     (find_by_receipt), never by resending -- DECISIONS.md's B3
     idempotency spike is why: `receipt` did NOT dedupe Order.create, so a
     second charge attempt against an already-sent key is not a safety
     net, it is a second charge.

     A NEVER_SENT FAST PATH WAS BUILT HERE AT B10 AND REVERTED THE SAME
     DAY. It resolved a key with no SENT row immediately and voided its
     committed_schedule row, freeing the NPCI slot -- on the proof that
     executor.py writes SENT before it charges, so no SENT row means no
     call was issued. The proof is sound about one process's own state and
     WRONG about a concurrent one: a live worker that STALLS between
     claiming its lease and writing SENT (executor.py never re-validates
     lease ownership before step 3) is indistinguishable from a crash, so
     recovery could void its slot while it was still alive. The worker
     then completed its real charge, and the freed slot was reissued at
     generation+1 -- a DIFFERENT key -- and charged again. Two real
     charges for one NPCI slot, invisible to any per-receipt idempotency
     check. Measured: 2 charges with the fast path, 1 without.
     Reverted rather than patched, because the thing it was buying (23 of
     60 chaos runs stopped burning a slot on a provably-unsent attempt)
     is an OPTIMISATION, and what it cost is the invariant this whole
     module exists to hold. Reinstating it needs real lease fencing in
     executor.py, not a guard bolted onto this file. Full sequence:
     POSTMORTEM.md incident 5; DECISIONS.md 2026-08-30 B10, both entries.

  2. STUCK -- keys whose LATEST row is FAILED with reason=UNCONFIRMED.
     Re-queried on backoff, terminating at a RESULT (found) or, after
     `max_unconfirmed_passes`, at reason=UNRESOLVED_FINAL -- the B9 gate's
     own words: "a reported metric, never a silent drop." The pass count
     is read from the ledger itself (a COUNT of this key's prior
     UNCONFIRMED rows), not a separate counter table -- this stays
     append-only, like everything else this layer writes.

The slot stays consumed throughout every step here, including
UNRESOLVED_FINAL: this project's constant refrain (root CLAUDE.md;
PLAN_DETAIL.md section 1) is that a double-charge is worse than ten missed
recoveries, and refunding an unconfirmed slot to the NPCI budget before it
is CONFIRMED clear is exactly the risk that refrain forbids -- as the
reverted NEVER_SENT fast path above demonstrated by breaking it.
"""
from __future__ import annotations

from src.core import clock
from src.core.types import LedgerState
from src.execute import lease
from src.execute.executor import PENDING_WEBHOOK_CONFIRMATION, SUCCESS_STATUSES, Result
from src.execute.razorpay_client import RazorpayLike
from src.ledger.store import LedgerEntry, append, find_by_key

UNCONFIRMED = "UNCONFIRMED"
UNRESOLVED_FINAL = "UNRESOLVED_FINAL"
DEFAULT_MAX_UNCONFIRMED_PASSES = 5


def _dangling_keys(conn) -> list[str]:
    """Every idempotency_key whose LATEST ledger row is INTENT or SENT and
    which is not currently held by an unexpired lease.

    THE SCAN STARTS AT THE LEDGER, NOT THE LEASE TABLE -- changed at B10
    after the first chaos run found a permanently lost job (POSTMORTEM.md
    incident 4). The original version iterated lease.expired() and looked
    up each key's ledger row, which cannot see a key that has no lease row
    AT ALL. That state is not hypothetical: executor.execute() writes the
    INTENT row (step 1) before it claims the lease (step 2), exactly as the
    write-ordering protocol requires, so a process dying between those two
    steps leaves precisely that shape. Such a key matched neither scan here,
    and could not be rescued by re-running the attempt either -- step 1's
    ON CONFLICT DO NOTHING finds the INTENT row and returns without sending.
    The slot stayed consumed forever with nothing reported.

    src/execute/lease.py's own docstring names the layering rule that was
    broken: the lease is "an OPTIMISATION over ledger_intent_once, not the
    concurrency control." Keying recovery on it made a deliberately
    non-authoritative table load-bearing for correctness. The ledger is the
    source of truth, so the ledger is where the scan starts; the lease is
    now only used to EXCLUDE keys some worker may still be legitimately
    mid-attempt on.

    Strictly wider than what it replaced: an expired lease whose latest row
    is INTENT/SENT still matches, since an expired lease is not an
    unexpired one.
    """
    now = clock.now()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT idempotency_key FROM (
                SELECT DISTINCT ON (idempotency_key) idempotency_key, state
                FROM ledger
                ORDER BY idempotency_key, ledger_id DESC
            ) latest
            WHERE state IN (%s, %s)
              AND idempotency_key NOT IN (
                  SELECT idempotency_key FROM attempt_lease WHERE expires_at >= %s
              )
            ORDER BY idempotency_key
            """,
            (LedgerState.INTENT.value, LedgerState.SENT.value, now),
        )
        return [row[0] for row in cur.fetchall()]


def _stuck_keys(conn) -> list[str]:
    """Every idempotency_key whose LATEST ledger row is FAILED with
    reason=UNCONFIRMED. DISTINCT ON (idempotency_key) ... ORDER BY
    idempotency_key, ledger_id DESC picks exactly the latest row per key,
    the same "latest stage wins" reasoning store.find_by_key() already
    documents for a single key, applied here across all of them at once."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT idempotency_key FROM (
                SELECT DISTINCT ON (idempotency_key) idempotency_key, state, reason
                FROM ledger
                ORDER BY idempotency_key, ledger_id DESC
            ) latest
            WHERE state = %s AND reason = %s
            """,
            (LedgerState.FAILED.value, UNCONFIRMED),
        )
        return [row[0] for row in cur.fetchall()]


def _unconfirmed_pass_count(conn, key: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ledger WHERE idempotency_key = %s AND state = %s AND reason = %s",
            (key, LedgerState.FAILED.value, UNCONFIRMED),
        )
        (count,) = cur.fetchone()
    return count


def _append_from_existing(conn, key: str, *, state: str, **fields) -> Result:
    """recover.py has no ScheduledAttempt in hand the way executor.py does
    -- only the key, and whatever the existing rows for it already
    recorded (mandate_id, cycle_id, attempt_index, action, amount_paise,
    profile, decision_sha256 are all immutable per key once the INTENT row
    exists). Re-reads that row rather than inventing a second interface
    for the same data."""
    existing = find_by_key(conn, key)
    if existing is None:
        raise LookupError(f"{key}: no existing ledger row to reconcile from")
    append(
        conn,
        LedgerEntry(
            idempotency_key=key,
            mandate_id=existing.mandate_id,
            cycle_id=existing.cycle_id,
            attempt_index=existing.attempt_index,
            action=existing.action,
            state=state,
            amount_paise=existing.amount_paise,
            profile=existing.profile,
            payload_sha256=existing.payload_sha256,
            decision_sha256=existing.decision_sha256,
            **fields,
        ),
    )
    return Result(
        idempotency_key=key,
        state=state,
        outcome=fields.get("outcome"),
        decline_class=fields.get("decline_class"),
        provider_ref=fields.get("provider_ref"),
        reason=fields.get("reason"),
    )


def _resolve_found(conn, key: str, found: dict) -> Result:
    """A payment was located for `key`. Mirrors executor.py's own status
    interpretation exactly (SUCCESS_STATUSES) -- a genuinely ambiguous or
    still-settling status is recorded honestly as unresolved, the same
    discipline execute() applies to its own synchronous response, never
    guessed just because this is the recovery path."""
    provider_ref = found.get("id")
    status = found.get("status")
    if status in SUCCESS_STATUSES:
        return _append_from_existing(
            conn, key, state=LedgerState.RESULT.value,
            provider_ref=provider_ref, outcome="RECOVERED",
        )
    return _append_from_existing(
        conn, key, state=LedgerState.RESULT.value,
        provider_ref=provider_ref, reason=PENDING_WEBHOOK_CONFIRMATION,
    )


def reconcile(
    conn, client: RazorpayLike, *, max_unconfirmed_passes: int = DEFAULT_MAX_UNCONFIRMED_PASSES
) -> list[Result]:
    """Run both passes once. Safe to call repeatedly and often (e.g. on
    every executor start, per PLAN_DETAIL.md section 3 step 5) -- a key
    with nothing to resolve is simply absent from both scans, never
    re-processed."""
    results: list[Result] = []

    handled_this_call: set[str] = set()

    for key in _dangling_keys(conn):
        found = client.find_by_receipt(key)
        if found is not None:
            results.append(_resolve_found(conn, key, found))
        else:
            results.append(
                _append_from_existing(conn, key, state=LedgerState.FAILED.value, reason=UNCONFIRMED)
            )
        lease.release(conn, key)
        handled_this_call.add(key)

    # A key just written to FAILED/UNCONFIRMED by the dangling loop above
    # would otherwise ALSO match _stuck_keys()'s query right now (writes
    # here are autocommit, so it is visible immediately) -- double-counting
    # one backoff pass as two within a single reconcile() call. Skipping
    # anything the dangling loop already touched keeps "N passes" meaning
    # "N reconcile() calls", not "N reconcile() calls plus one extra the
    # first time a key ever goes dangling."
    for key in _stuck_keys(conn):
        if key in handled_this_call:
            continue
        found = client.find_by_receipt(key)
        if found is not None:
            results.append(_resolve_found(conn, key, found))
            continue

        passes_so_far = _unconfirmed_pass_count(conn, key)
        if passes_so_far >= max_unconfirmed_passes:
            results.append(
                _append_from_existing(conn, key, state=LedgerState.FAILED.value, reason=UNRESOLVED_FINAL)
            )
        else:
            results.append(
                _append_from_existing(conn, key, state=LedgerState.FAILED.value, reason=UNCONFIRMED)
            )

    return results
