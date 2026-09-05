"""Turns one src.policy.allocator.Plan into durable rows: always a `plan`
row, and -- iff the chosen action is ATTEMPT -- one `committed_schedule`
row. This file is new scope beyond the build spec's B9 file table, added
because nothing upstream of it writes either table: B8's solve() is
"pure and DB-free" by its own docstring, so something has to be the first
thing that persists a decision. Approved and logged in DECISIONS.md,
2026-08-30, B9.

allocator.py's own Plan docstring is explicit: `committed` holds "zero or
one CommittedAttempt (never more -- solve() is re-invoked once an attempt
resolves ... it never pre-commits multiple future slots on speculation)".
commit() relies on that contract directly rather than looping over
`plan.committed` -- looping would silently paper over a future violation
of it instead of surfacing one.

THE DAY-INDEX / WALL-CLOCK GAP, disclosed rather than quietly patched:
eval/corpus.py's assert_legal() docstring already warns, in board terms,
"do not port this assertion into B9 as if it were sufficient" -- B8's
`on_day` is a day-index with no hour component, and the real >=24h lead
(clause 6(a)) has to be enforced at the hour level once a real timestamp
exists. This module is where `on_day` first becomes a real timestamp:

    scheduled_for = cycle_start + timedelta(days=on_day - 1)

`cycle_start` is supplied by the caller and must carry the real hour this
mandate's attempts land at (day 1 of the cycle, at whatever hour the
mandate's billing schedule uses) -- on_day=1 maps to cycle_start itself.

This mapping can produce a `scheduled_for` that B8's day-granularity model
considered legal (e.g. `permissive`'s "the next slot may land on plan_day
itself") but that is LESS than 24 real hours after the actual moment this
function runs -- committed_at is `clock.now()`, a real instant, and a
day-index gap of "0 days" is obviously not >=24h once mapped to a clock.
This module does NOT special-case that away by silently pushing the date
out: doing so would be the allocator's decision to make (a different
`on_day`), not this layer's to invent, and it would be exactly the kind of
late read that ACTS rather than stops -- forbidden by the same asymmetry
the build spec section 1's late-read principle states for the executor.
Instead: the INSERT is attempted as computed, and `committed_schedule`'s
own CHECK constraint is the enforcement. A violation is caught narrowly and
re-raised as CommitError -- a clear, typed signal that this specific Plan
was not actually committable in wall-clock terms, for a caller (B13's real
production loop) to feed back rather than something this module papers
over. tests/execute/test_commit.py drives this path explicitly so it is a
proven behaviour, not a hoped-for one.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg

from src.core import clock
from src.core.types import Action
from src.execute.keys import ScheduledAttempt, key_for
from src.policy.allocator import Plan


class CommitError(ValueError):
    """Raised when a Plan cannot actually be committed -- currently, only
    when the computed scheduled_for fails committed_schedule's own 24h
    CHECK constraint (see module docstring). Never raised for an ordinary
    STOP/OFFER/REAUTH plan; those never reach the committed_schedule INSERT
    at all."""


def _insert_plan_row(conn, plan: Plan) -> None:
    """ON CONFLICT DO NOTHING: decision_sha256 is a canonical hash of the
    Plan's own content (src.core.ids.decision_sha256), so re-committing an
    identical Plan is legitimately idempotent -- mirroring store.append()'s
    own INTENT-row dedup discipline, not a new convention invented here."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO plan (
                decision_sha256, mandate_id, cycle_id, profile, belief_json,
                conformal_set, binding_constraint, solver_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (decision_sha256) DO NOTHING
            """,
            (
                plan.decision_sha256,
                plan.mandate_id,
                plan.cycle_id,
                plan.profile.value,
                plan.belief_json,
                # conformal_set is a frozenset[Cause]; stored as a stable,
                # sorted, comma-joined string -- there is no richer column
                # type for it and this module owns no JSON-encoding
                # convention of its own to invent one.
                ",".join(sorted(c.value for c in plan.conformal_set)),
                plan.binding_constraint,
                plan.solver_version,
            ),
        )


def commit(conn, plan: Plan, *, cycle_start: datetime) -> ScheduledAttempt | None:
    """Write `plan` always. If `plan.chosen_action` is ATTEMPT, also write
    the one committed_schedule row for `plan.committed`'s single entry, at
    generation=0, and return it. Otherwise (STOP/OFFER/REAUTH -- none of
    which schedule a debit) return None.

    Takes no AllocationContext: everything this function needs
    (mandate_id, cycle_id, profile, the committed slot/day/amount) is
    already on `plan` itself -- accepting ctx here would invite reading a
    field from it instead of from the Plan that was actually solved,
    which is exactly the kind of drift a single source of truth prevents.

    Raises CommitError if the computed scheduled_for fails the 24h CHECK
    (see module docstring) -- propagated from a caught
    psycopg.errors.CheckViolation, never silently retried with a different
    date.

    Idempotent on retry: ON CONFLICT (idempotency_key) DO NOTHING, then a
    re-read on conflict, mirroring src.ledger.store.record_lifecycle_event's
    own dedup pattern exactly. A retried commit() of the SAME Plan (the
    crash-recovery case this whole layer exists for -- the caller crashed
    after committing but before hearing back) must not raise on the
    idempotency_key's primary key, and must return the ORIGINAL row's
    committed_at/scheduled_for, never silently move them to this retry's
    clock.now().
    """
    _insert_plan_row(conn, plan)

    if plan.chosen_action != Action.ATTEMPT:
        return None

    assert len(plan.committed) == 1, (
        f"Plan.chosen_action is ATTEMPT but committed has "
        f"{len(plan.committed)} entries -- violates allocator.py's own "
        "'zero or one, never more' contract; this is an upstream bug, not "
        "something commit() should paper over."
    )
    committed_attempt = plan.committed[0]

    key = key_for(
        mandate_id=plan.mandate_id,
        cycle_id=plan.cycle_id,
        attempt_index=committed_attempt.slot,
        generation=0,
        action=plan.chosen_action.value,
        amount_paise=committed_attempt.amount_paise,
    )
    scheduled_for = cycle_start + timedelta(days=committed_attempt.on_day - 1)
    committed_at = clock.now()

    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO committed_schedule (
                    idempotency_key, mandate_id, cycle_id, attempt_index,
                    generation, action, amount_paise, profile,
                    decision_sha256, scheduled_for, committed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING scheduled_for, committed_at
                """,
                (
                    key, plan.mandate_id, plan.cycle_id, committed_attempt.slot,
                    0, plan.chosen_action.value, committed_attempt.amount_paise,
                    plan.profile.value, plan.decision_sha256, scheduled_for, committed_at,
                ),
            )
            row = cur.fetchone()
        except psycopg.errors.CheckViolation as exc:
            raise CommitError(
                f"{plan.mandate_id}: on_day={committed_attempt.on_day} mapped to "
                f"scheduled_for={scheduled_for.isoformat()}, which is less than 24h "
                f"after committed_at={committed_at.isoformat()} -- this Plan is not "
                "actually committable in wall-clock terms, even though B8's "
                "day-index model considered it legal (see this module's docstring)"
            ) from exc

        if row is None:
            # Already committed by an earlier call -- re-read rather than
            # trust this call's freshly computed values, which is exactly
            # the retried-after-crash case this idempotent insert exists
            # for (see the docstring above).
            cur.execute(
                "SELECT scheduled_for, committed_at FROM committed_schedule "
                "WHERE idempotency_key = %s",
                (key,),
            )
            row = cur.fetchone()
        scheduled_for, committed_at = row

    return ScheduledAttempt(
        idempotency_key=key,
        mandate_id=plan.mandate_id,
        cycle_id=plan.cycle_id,
        attempt_index=committed_attempt.slot,
        generation=0,
        action=plan.chosen_action.value,
        amount_paise=committed_attempt.amount_paise,
        profile=plan.profile.value,
        decision_sha256=plan.decision_sha256,
        scheduled_for=scheduled_for,
        committed_at=committed_at,
    )
