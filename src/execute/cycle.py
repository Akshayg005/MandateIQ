"""R4: the two-phase cycle orchestrator. `plan_cycle()` reads durable state
and writes a Plan (+ a committed_schedule row, iff ATTEMPT); `run_due()`,
called >=24h later against the same schema, executes whatever that produced.
Two entry points, not one combined pass -- `committed_schedule`'s own CHECK
constraint (`scheduled_for >= committed_at + INTERVAL '24 hours'`,
src/ledger/schema.sql, clause 6(a)) means a single combined function would
either violate that check or silently no-op the execute half on every real
invocation. See R4_PLAN.md for the investigation this design is based on,
and DECISIONS.md, 2026-09-04, "R4", for the scope decisions made before this
file was written.

Every per-mandate control-flow decision here mirrors eval/run.py's
`_run_engine_mandate()` -- solve, act, observe, re-solve on a terminal
outcome, using `observe_terminal()`/`with_terminal()` -- rebuilt to read its
state from Postgres each time instead of from Python loop variables. This
file gives `src.policy.belief.observe_terminal()` and
`AllocationContext.with_terminal()` their first production callers; before
R4 both had zero callers outside `eval/` and `tests/` (reports/gates.md, R2).

Deliberately NOT built here, disclosed rather than guessed at: a fold over
MULTIPLE `normalized_decline` rows for one mandate (only the latest is
applied); belief carried in memory across `plan_cycle()` calls (every call
re-derives belief from durable state instead); a `plan`-table eligibility
check for "a terminal decision was already made this cycle" (re-solving an
already-terminal mandate is provably idempotent -- see `_is_eligible()`'s
docstring -- so skipping it is an efficiency question, not a correctness
one, and adding it would need a race-free "latest plan row" ordering this
schema does not yet provide for free). `recover.py`'s crash-recovery pass is
a separate, already-gated concern (B10); a real deployment runs it alongside
`run_due()`, but R4's own test does not need to exercise it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core import clock
from src.core.types import Action, DeclineClass, LedgerState, Outcome, Profile
from src.execute.commit import commit
from src.execute.executor import Result, execute
from src.execute.keys import ScheduledAttempt
from src.execute.razorpay_client import RazorpayLike
from src.ledger.store import LedgerRow, latest_state, replay
from src.policy import belief as belief_mod
from src.policy.allocator import SlotHazard, solve
from src.policy.belief import Belief
from src.policy.costs import PolicyCosts
from src.policy.gate import ConformalGate
from src.policy.stopping_rules import AllocationContext

_REFERENCE_PRIOR: dict = dict(zip(belief_mod.CAUSE_ORDER, belief_mod.REFERENCE_PRIOR))


@dataclass(frozen=True)
class _MandateRow:
    mandate_id: str
    amount_paise: int
    ceiling_paise: int
    category: str


def _find_mandate(conn, mandate_id: str) -> _MandateRow:
    """The `mandate` row for `mandate_id`. Raises LookupError if none exists
    -- mirroring src.ledger.store.latest_state's own convention: a real
    mandate always has a registry row by the time it reaches planning, and
    silently defaulting one would hide a real bug (a mandate the caller
    forgot to register) rather than surface it."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT mandate_id, amount_paise, ceiling_paise, category "
            "FROM mandate WHERE mandate_id = %s",
            (mandate_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"no mandate row for mandate_id={mandate_id!r}")
    return _MandateRow(mandate_id=row[0], amount_paise=row[1], ceiling_paise=row[2], category=row[3])


def _all_mandate_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT mandate_id FROM mandate ORDER BY mandate_id")
        rows = cur.fetchall()
    return [r[0] for r in rows]


def _has_unresolved_commitment(conn, mandate_id: str, cycle_id: int) -> bool:
    """True if this (mandate_id, cycle_id) has a live committed_schedule row
    -- not voided -- with no RESULT/FAILED ledger row yet for its
    idempotency_key. Such a row is "in flight": a second plan_cycle() call
    must not plan another attempt on top of one already committed and not
    yet resolved."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM committed_schedule cs
            WHERE cs.mandate_id = %s AND cs.cycle_id = %s AND cs.voided_at IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM ledger l
                WHERE l.idempotency_key = cs.idempotency_key
                  AND l.state IN ('RESULT', 'FAILED')
              )
            LIMIT 1
            """,
            (mandate_id, cycle_id),
        )
        return cur.fetchone() is not None


def _cycle_recovered(conn, mandate_id: str, cycle_id: int) -> bool:
    """True if a resolved ATTEMPT in this cycle already produced RECOVERED --
    the cycle is done, nothing left to plan."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM ledger
            WHERE mandate_id = %s AND cycle_id = %s
              AND action = %s AND state = %s AND outcome = %s
            LIMIT 1
            """,
            (mandate_id, cycle_id, Action.ATTEMPT.value, LedgerState.RESULT.value, Outcome.RECOVERED.name),
        )
        return cur.fetchone() is not None


def _is_eligible(conn, mandate_id: str, cycle_id: int) -> bool:
    """Whether plan_cycle() should plan this mandate at all, this cycle.

    Two checks, both required by R4's gate (reports/gates.md): an in-flight
    commitment, or a cycle already RECOVERED. A THIRD case -- a prior
    STOP/REAUTH/OFFER decision already made this cycle -- is NOT checked
    here, disclosed rather than silently added: re-solving such a mandate is
    provably idempotent (identical durable state -> identical belief and
    ctx -> the same decision_sha256 -> commit()'s own ON CONFLICT DO NOTHING
    on the plan row, and no new committed_schedule row since chosen_action
    is never ATTEMPT), so omitting it costs a redundant solve() call, not a
    correctness gap. Adding it would need a race-free "latest plan row for
    this cycle" read, and `plan`'s own created_at is DB-clock DEFAULT now()
    with no serial ordinal to break a tie -- exactly the kind of flake risk
    this project's own B9 history (the ~7-10% flake margin found and fixed
    there) says not to introduce without a real ordering guarantee."""
    if _has_unresolved_commitment(conn, mandate_id, cycle_id):
        return False
    if _cycle_recovered(conn, mandate_id, cycle_id):
        return False
    return True


def _resolved_attempts(conn, mandate_id: str, cycle_id: int) -> list[LedgerRow]:
    """Every RESULT-state ATTEMPT ledger row for this (mandate_id, cycle_id),
    in insertion order (replay()'s own order, filtered)."""
    return [
        r for r in replay(conn, mandate_id)
        if r.cycle_id == cycle_id and r.action == Action.ATTEMPT.value and r.state == LedgerState.RESULT.value
    ]


def _day_index(conn, idempotency_key: str, cycle_start: datetime) -> int:
    """The on_day (1-indexed) a committed attempt actually landed on,
    recovered from its own committed_schedule.scheduled_for -- the inverse
    of src.execute.commit.commit()'s `scheduled_for = cycle_start +
    timedelta(days=on_day - 1)`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT scheduled_for FROM committed_schedule WHERE idempotency_key = %s",
            (idempotency_key,),
        )
        row = cur.fetchone()
    return (row[0] - cycle_start).days + 1


def _latest_normalized_decline(conn, mandate_id: str) -> tuple[str, str] | None:
    """(DeclineClass value, normalizer_version) of the most recent
    normalized_decline row for any ingested_event tied to this mandate, or
    None. The read-back path src/policy/belief.update()'s required
    source_version must come from (PLAN_DETAIL.md B11 gate clause 3): never
    an in-memory classifier result, always a round-trip through the ledger."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT nd.value, nd.normalizer_version
            FROM normalized_decline nd
            JOIN ingested_event ie ON ie.event_id = nd.event_id
            WHERE ie.mandate_id = %s
            ORDER BY nd.created_at DESC
            LIMIT 1
            """,
            (mandate_id,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row is not None else None


def _belief_source(conn, mandate_id: str, cycle_id: int) -> tuple[Belief, Outcome | None]:
    """The Belief to solve() with for this mandate's next decision, and the
    terminal Outcome that produced it (None if not terminal). Mirrors
    eval/run.py's _run_engine_mandate branching exactly:

    1. The latest resolved ATTEMPT this cycle is DEAD/OPTED_OUT ->
       observe_terminal() on the measured posterior (belief_mod.
       TERMINAL_OBSERVED_CAUSE_PROBS) -- R4's real caller for both, per the
       pre-registered gap (reports/gates.md, R2a).
    2. Otherwise, a normalized_decline row exists for this mandate ->
       belief.update() on it (B11's existing versioned-evidence path).
    3. Otherwise -> belief.init(REFERENCE_PRIOR): genuinely no evidence yet.
    """
    resolved = _resolved_attempts(conn, mandate_id, cycle_id)
    if resolved:
        latest = resolved[-1]
        if latest.outcome in (Outcome.DEAD.name, Outcome.OPTED_OUT.name):
            outcome = Outcome[latest.outcome]
            probs = belief_mod.TERMINAL_OBSERVED_CAUSE_PROBS[outcome]
            b = belief_mod.observe_terminal(
                probs, source_version=belief_mod.TERMINAL_OBSERVATION_SOURCE_VERSION,
            )
            return b, outcome

    nd = _latest_normalized_decline(conn, mandate_id)
    if nd is not None:
        value, normalizer_version = nd
        b = belief_mod.init(_REFERENCE_PRIOR)
        b = belief_mod.update(b, DeclineClass(value), source_version=normalizer_version)
        return b, None

    return belief_mod.init(_REFERENCE_PRIOR), None


def _read_context(
    conn, *, mandate_id: str, cycle_id: int, profile: Profile, costs: PolicyCosts,
    cycle_start: datetime | None = None,
) -> AllocationContext:
    """The AllocationContext reconstructed from durable state alone --
    plan_cycle()'s input to solve(), and run_due()'s input to execute()'s
    late permitted() check.

    attempts_used/contacts_sent start at 1: slot 1, the original debit that
    triggered recovery, already happened before this system ever engages
    (CLAUDE.md's own framing -- "failed recurring debits") and is never
    itself committed by this module, mirroring eval/run.py's and
    src/execute/shadow.py's own `_initial_context` exactly. contacts_sent
    tracks ATTEMPT contacts only: REAUTH/OFFER never produce a ledger row
    (commit()'s own chosen_action gate), so they cannot be counted from
    durable state today -- disclosed, not fixed (this cycle design ends
    planning at the first REAUTH/OFFER/STOP anyway, so no contact-cap
    interaction is missed in practice).

    `cycle_start`, if given, lets committed_days/plan_day be recovered
    exactly (via each resolved attempt's own committed_schedule.
    scheduled_for) -- needed by plan_cycle()'s solve() call, where
    stopping_rules.permitted() does not depend on either but
    allocator.solve()'s day search does (`earliest = max(ctx.plan_day + lead,
    committed_days[-1] + 1)`). run_due() has no cycle_start (its
    execute() call never reaches solve()), so it omits this parameter and
    gets a placeholder committed_days=(1,)/plan_day=1 -- inert there, since
    permitted() reads neither field."""
    mandate = _find_mandate(conn, mandate_id)
    resolved = _resolved_attempts(conn, mandate_id, cycle_id)
    attempts_used = 1 + len(resolved)

    if cycle_start is not None:
        days = sorted(_day_index(conn, r.idempotency_key, cycle_start) for r in resolved)
        committed_days = (1,) + tuple(days)
    else:
        committed_days = (1,)

    return AllocationContext(
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        profile=profile,
        amount_paise=mandate.amount_paise,
        ceiling_paise=mandate.ceiling_paise,
        category=mandate.category,
        plan_day=committed_days[-1],
        attempts_used=attempts_used,
        committed_days=committed_days,
        contacts_sent=attempts_used,
        mandate_state=latest_state(conn, mandate_id),
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def plan_cycle(
    conn, *, cycle_id: int, cycle_start: datetime,
    hazard: SlotHazard, costs: PolicyCosts, gate: ConformalGate | None = None,
    profile: Profile = Profile.strict,
) -> list[ScheduledAttempt]:
    """For every registered mandate eligible this cycle (see _is_eligible):
    read its belief source and context, solve(), and commit() the result.
    Returns the ScheduledAttempts actually committed (chosen_action ==
    ATTEMPT) -- commit() still writes every other decision's `plan` row for
    audit either way, per its own docstring."""
    committed: list[ScheduledAttempt] = []
    for mandate_id in _all_mandate_ids(conn):
        if not _is_eligible(conn, mandate_id, cycle_id):
            continue

        belief, terminal_outcome = _belief_source(conn, mandate_id, cycle_id)
        ctx = _read_context(
            conn, mandate_id=mandate_id, cycle_id=cycle_id, profile=profile,
            costs=costs, cycle_start=cycle_start,
        )
        if terminal_outcome is not None:
            ctx = ctx.with_terminal(terminal_outcome)

        plan = solve(belief, ctx, hazard=hazard, costs=costs, gate=gate)
        scheduled = commit(conn, plan, cycle_start=cycle_start)
        if scheduled is not None:
            committed.append(scheduled)
    return committed


def _due_rows(conn, as_of: datetime) -> list[ScheduledAttempt]:
    """Every committed_schedule row that is due (scheduled_for <= as_of),
    not voided, and not yet resolved (no RESULT/FAILED ledger row for its
    idempotency_key) -- the first "scan committed_schedule for due,
    unresolved rows" query in the codebase (R4_PLAN.md)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT idempotency_key, mandate_id, cycle_id, attempt_index, generation,
                   action, amount_paise, profile, decision_sha256, scheduled_for,
                   committed_at, voided_at, void_reason
            FROM committed_schedule
            WHERE voided_at IS NULL AND scheduled_for <= %s
              AND NOT EXISTS (
                SELECT 1 FROM ledger l
                WHERE l.idempotency_key = committed_schedule.idempotency_key
                  AND l.state IN ('RESULT', 'FAILED')
              )
            ORDER BY scheduled_for ASC
            """,
            (as_of,),
        )
        rows = cur.fetchall()
    return [
        ScheduledAttempt(
            idempotency_key=r[0], mandate_id=r[1], cycle_id=r[2], attempt_index=r[3],
            generation=r[4], action=r[5], amount_paise=r[6], profile=r[7],
            decision_sha256=r[8], scheduled_for=r[9], committed_at=r[10],
            voided_at=r[11], void_reason=r[12],
        )
        for r in rows
    ]


def run_due(
    conn, client: RazorpayLike, *, costs: PolicyCosts, as_of: datetime | None = None,
    owner: str, lease_ttl_seconds: int = 300,
) -> list[Result]:
    """Execute every committed attempt that is due as of `as_of` (default:
    src.core.clock.now(), so a frozen test clock controls what counts as
    due). For each: reconstruct its ScheduledAttempt and AllocationContext
    from durable state (the same _read_context() plan_cycle() uses, minus
    cycle_start -- see that function's docstring) and call execute(), which
    performs its own late permitted()/mandate_lifecycle check before ever
    sending anything (src/execute/executor.py's own late-read principle)."""
    cutoff = as_of if as_of is not None else clock.now()
    results: list[Result] = []
    for attempt in _due_rows(conn, cutoff):
        ctx = _read_context(
            conn, mandate_id=attempt.mandate_id, cycle_id=attempt.cycle_id,
            profile=Profile(attempt.profile), costs=costs,
        )
        results.append(
            execute(conn, client, attempt, ctx, owner=owner, lease_ttl_seconds=lease_ttl_seconds)
        )
    return results
