"""Induced-kill chaos harness (B10). Kills the executor at uniformly
sampled points and checks what recovery does with the wreckage.

WHY THE DENOMINATOR IS THE WHOLE POINT (the build spec section 8.2,
finding 2). "50 kills, zero double-charges" is not evidence. The unsafe
window -- SENT row written, provider call in flight -- is a few
milliseconds wide inside an operation that spends most of its life doing
other things. Uniform kills therefore sample the SAFE regions almost every
time, and a harness that reports only the headline is reporting the
sampling distribution, not a property of the system. So this module
partitions every kill by the window it actually landed in, counts them,
and prints the partition next to the headline. If the unsafe count is
zero, the headline means nothing, and the report says so in those words.

WINDOWS ARE OBSERVED, NEVER ASSUMED. Each kill is classified after the
fact from durable state -- what is in `ledger`, whether a lease row
exists, and whether the provider double actually accepted a charge -- not
from where the kill was aimed. A harness that trusted its own aim would
report the partition it intended rather than the one it produced.

TWO WINDOWS THE LEDGER CANNOT TELL APART, which is itself a finding worth
stating rather than modelling around. the build spec names "SENT ->
provider-ack" and "ack -> RESULT-commit" as separate windows. From the
ledger they are one: the ack arriving is not written anywhere until the
RESULT row is written, so "accepted, response still in flight" and
"accepted, response received, not yet recorded" leave byte-identical
durable state. They are merged here into SENT_ACCEPTED deliberately. The
system genuinely cannot distinguish them, which is precisely why both must
be resolved by ASKING the provider rather than by reasoning about local
state -- and why recover.py never resends.

TWO KILL MECHANISMS, REPORTED SEPARATELY, because they answer different
questions and averaging them would hide both:

  1. UNIFORM STATEMENT KILLS (`run_uniform`) -- the process dies before a
     uniformly sampled database statement. This is the honest sampler and
     the source of the denominator. It cannot reach the most dangerous
     state at all, which is the finding, not a limitation to apologise
     for: a kill signal to our own process can only ever destroy work we
     have not yet done. It cannot make the provider accept money and then
     lose the answer.

  2. FAULT-SEAM RUNS (`run_fault_seam`) -- src.execute.razorpay_client's
     FaultSpec: the provider genuinely accepts, then the response is
     dropped. Every one of these lands in the unsafe window by
     construction, so they prove nothing about frequency and everything
     about correctness. Reported as a separate block with its own count,
     never folded into the 50.

`_InducedKill` derives from BaseException on purpose -- see its docstring.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from src.core import clock, db
from src.core.types import Action, LedgerState, MandateState, Profile
from src.execute.commit import commit
from src.execute.executor import execute
from src.execute.razorpay_client import FaultSpec, RazorpayClientError
from src.execute.recover import UNCONFIRMED, UNRESOLVED_FINAL, reconcile
from src.ledger.store import record_lifecycle_event, replay
from src.policy.allocator import CommittedAttempt, Plan
from src.policy.stopping_rules import AllocationContext

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
LEASE_TTL_SECONDS = 300

# How many reconcile() passes the harness runs after each kill. Must exceed
# recover.DEFAULT_MAX_UNCONFIRMED_PASSES so a key that goes UNCONFIRMED is
# driven all the way to its own terminal answer (a RESULT once the provider
# index catches up, or UNRESOLVED_FINAL) rather than being scored as
# "unresolved" merely because the harness stopped asking too early.
RECOVERY_PASSES = 8

# The measured Orders-index lag from B9 (POSTMORTEM.md incident 3): a real
# order was invisible to find_by_receipt at 0s/3s/8s and appeared minutes
# later. Modelled as "the first N lookups miss", so the UNCONFIRMED backoff
# is actually exercised rather than every recovery resolving on pass one.
DEFAULT_INDEX_LAG_PASSES = 2

_TERMINAL_STATES = frozenset({LedgerState.RESULT.value, LedgerState.FAILED.value})


class Window(Enum):
    """Where a kill landed, judged from durable state after the fact."""

    PRE_INTENT = "pre-INTENT"
    INTENT_TO_LEASE = "INTENT->lease"
    LEASE_TO_SENT = "lease->SENT"
    SENT_NOT_ACCEPTED = "SENT->ack (provider did NOT accept)"
    SENT_ACCEPTED = "SENT->RESULT (provider ACCEPTED)"
    POST_RESULT = "post-RESULT"


#: The unsafe window: money may have moved and nothing local records it.
UNSAFE_WINDOW = Window.SENT_ACCEPTED


class _InducedKill(BaseException):
    """A simulated SIGKILL.

    Derives from BaseException, not Exception, deliberately. Production
    code on this path has broad `except Exception` handlers (see
    razorpay_client.charge's own AMBIGUOUS branch). A kill that those
    handlers could catch would be converted into an orderly failure and
    land in whatever window the handler wrote a row for -- so the harness
    would report a partition manufactured by its own exception type. A
    real `kill -9` cannot be caught, and neither can this.
    """


# --- the provider double -----------------------------------------------------

class ChaosClient:
    """A RazorpayLike double that is also the DOUBLE-CHARGE ORACLE.

    `accepted` counts, per receipt, how many times the provider committed
    to moving money. It is incremented BEFORE anything else can go wrong,
    because that is the truth on the far side of the wire: once the
    provider has accepted, a lost response, a crash, or a dropped
    connection does not un-charge the customer. Any receipt whose count
    reaches 2 is a real double-charge, no matter what our ledger says
    about it.
    """

    def __init__(
        self,
        *,
        fault: FaultSpec | None = None,
        index_lag_passes: int = DEFAULT_INDEX_LAG_PASSES,
        counter: "_KillCounter | None" = None,
    ) -> None:
        self.accepted: Counter[str] = Counter()
        self.charge_calls = 0
        self.lookups: Counter[str] = Counter()
        self._fault = fault
        self._payments: dict[str, dict] = {}
        self._index_lag_passes = index_lag_passes
        self._counter = counter

    def charge(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        # Ticked BEFORE acceptance, so a kill here models "the process died
        # on the way to the provider" -- SENT row written, nothing sent.
        if self._counter is not None:
            self._counter.tick("the provider charge")
        self.charge_calls += 1
        # THE ORACLE. Recorded first, unconditionally.
        self.accepted[receipt] += 1
        self._payments[receipt] = {
            "id": f"pay_chaos_{self.accepted[receipt]}_{receipt[:10]}",
            "status": "captured",
            "amount": amount_paise,
        }
        if self._fault is not None and self._fault.drop_response_after_accept:
            raise RazorpayClientError(
                f"charge({receipt!r}): response dropped in flight after the "
                "provider accepted (injected fault, B10)"
            )
        return self._payments[receipt]

    def find_by_receipt(self, receipt: str) -> dict | None:
        """Honours the measured index lag: the first `index_lag_passes`
        lookups miss even when the payment exists. None means "not found
        YET", never "never sent" -- razorpay_client.py's own contract."""
        self.lookups[receipt] += 1
        if self.lookups[receipt] <= self._index_lag_passes:
            return None
        return self._payments.get(receipt)

    def create_order(self, **kwargs):
        raise AssertionError("execute() must call charge(), never create_order()")

    def pause_subscription(self, *args, **kwargs):
        raise AssertionError("the executor path must never pause a subscription")


# --- the kill mechanism ------------------------------------------------------

class _KillCounter:
    """The shared kill index, ticked by every OBSERVABLE SIDE EFFECT the
    attempt performs -- every database statement, and the provider charge.

    The charge has to be in the same sequence as the statements, not a
    separate mode. Without it the sampler cannot reach the state "SENT row
    written, provider never contacted": there is no database statement
    between executor.py's SENT append and its client.charge() call, so a
    statement-only index skips straight over that boundary and the window
    is structurally unreachable rather than merely rare. Sampling a space
    that excludes a real crash state, and then reporting the partition as
    a denominator, would be the exact error this module exists to avoid.
    """

    def __init__(self, kill_at: int | None) -> None:
        self.kill_at = kill_at
        self.ticks = 0

    def tick(self, what: str) -> None:
        self.ticks += 1
        if self.kill_at is not None and self.ticks >= self.kill_at:
            raise _InducedKill(f"induced kill before {what} (tick {self.ticks})")


class _KillingCursor:
    """Counts statements and raises _InducedKill before the target one.

    Wrapping the CONNECTION rather than monkeypatching src/ledger/store.py
    matters: this harness must INTERRUPT production code, never alter its
    behaviour. A patched `append` would be a different program under test.
    A connection that stops answering is what a dying process actually
    experiences.
    """

    def __init__(self, counter: _KillCounter, cur) -> None:
        self._counter = counter
        self._cur = cur

    def __enter__(self):
        self._cur.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._cur.__exit__(*exc_info)

    def execute(self, *args, **kwargs):
        self._counter.tick("a database statement")
        return self._cur.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _KillingConnection:
    """Proxies a real connection, counting statements. All writes on this
    path are autocommit, so everything issued before the kill is durable
    exactly as it would be after a real crash."""

    def __init__(self, conn, counter: _KillCounter) -> None:
        self._conn = conn
        self._counter = counter

    @property
    def statements(self) -> int:
        return self._counter.ticks

    def cursor(self, *args, **kwargs):
        return _KillingCursor(self._counter, self._conn.cursor(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._conn, name)


# --- scenario construction ---------------------------------------------------

def _decision_sha256(mandate_id: str) -> str:
    return hashlib.sha256(mandate_id.encode()).hexdigest()


def _plan_for(mandate_id: str, *, amount_paise: int) -> Plan:
    return Plan(
        mandate_id=mandate_id,
        cycle_id=1,
        profile=Profile.strict,
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=2, amount_paise=amount_paise),),
        belief_json="{}",
        conformal_set=frozenset(),
        binding_constraint=None,
        solver_version="chaos-b10",
        decision_sha256=_decision_sha256(mandate_id),
    )


def _ctx_for(mandate_id: str, *, amount_paise: int) -> AllocationContext:
    return AllocationContext(
        mandate_id=mandate_id,
        cycle_id=1,
        profile=Profile.strict,
        amount_paise=amount_paise,
        ceiling_paise=200_000,
        category="subscription",
        plan_day=0,
        attempts_used=1,
        committed_days=(),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=4,
        quiet_hours_start=21,
        quiet_hours_end=8,
    )


def _build_scenario(conn, mandate_id: str, *, amount_paise: int = 50_000):
    """One fresh, committed, ready-to-execute attempt. Every kill gets its
    own mandate so no two runs can interact through a shared key."""
    clock.set_frozen(CYCLE_START)
    record_lifecycle_event(
        conn,
        event_id=f"evt-created-{mandate_id}",
        mandate_id=mandate_id,
        state=MandateState.ACTIVE.value,
        source="INTERNAL",
        effective_at=CYCLE_START - timedelta(days=1),
    )
    attempt = commit(conn, _plan_for(mandate_id, amount_paise=amount_paise), cycle_start=CYCLE_START)
    assert attempt is not None, "an ATTEMPT plan must produce a committed_schedule row"
    return attempt


# --- observing what a kill left behind ---------------------------------------

def _states_for(conn, mandate_id: str, key: str) -> list[str]:
    return [r.state for r in replay(conn, mandate_id) if r.idempotency_key == key]


def _has_lease(conn, key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM attempt_lease WHERE idempotency_key = %s", (key,))
        return cur.fetchone() is not None


def _classify_window(conn, mandate_id: str, key: str, client: ChaosClient) -> Window:
    """Judged entirely from durable state plus the provider's own record of
    what it accepted -- never from where the kill was aimed."""
    states = _states_for(conn, mandate_id, key)
    if not states:
        return Window.PRE_INTENT
    if any(s in _TERMINAL_STATES for s in states):
        return Window.POST_RESULT
    if LedgerState.SENT.value in states:
        if client.accepted[key]:
            return Window.SENT_ACCEPTED
        return Window.SENT_NOT_ACCEPTED
    return Window.LEASE_TO_SENT if _has_lease(conn, key) else Window.INTENT_TO_LEASE


def _ledger_violation(states: list[str]) -> str | None:
    """Is this key's row sequence a legal prefix of the state machine?

    Legal: INTENT -> [SENT] -> (RESULT | FAILED), with FAILED reachable
    directly from INTENT (executor.py's pre-call aborts, which hold the
    lease and know first-hand nothing was sent) and repeated FAILED rows
    legal on the UNCONFIRMED backoff walk.
    """
    if not states:
        return None
    if states[0] != LedgerState.INTENT.value:
        return f"first row is {states[0]}, not INTENT"
    if states.count(LedgerState.INTENT.value) > 1:
        return "more than one INTENT row (ledger_intent_once should forbid this)"
    if LedgerState.RESULT.value in states and LedgerState.SENT.value not in states:
        return "RESULT row without a preceding SENT row"
    first_terminal = next(
        (i for i, s in enumerate(states) if s in _TERMINAL_STATES), None
    )
    if first_terminal is not None:
        after = states[first_terminal + 1:]
        if any(s not in _TERMINAL_STATES for s in after):
            return f"non-terminal row after a terminal one: {states}"
    return None


# --- one kill ----------------------------------------------------------------

@dataclass(frozen=True)
class KillOutcome:
    index: int
    kill_at_statement: int | None
    window: Window
    provider_accepted: int
    final_state: str | None
    final_reason: str | None
    double_charged: bool
    lost_job: bool
    ledger_violation: str | None
    unconfirmed_passes: int = 0
    had_sent_row: bool = False

    @property
    def ok(self) -> bool:
        return not (self.double_charged or self.lost_job or self.ledger_violation)

    @property
    def slot_burned_unsent(self) -> bool:
        """This attempt ended at UNRESOLVED_FINAL -- one of only four NPCI
        slots consumed forever -- with NO SENT row in the ledger.

        Absence of a SENT row on re-read is not a guess, it is proof.
        executor.py writes SENT (autocommit, durable) and only then calls
        charge(), so `charge() was called` implies `SENT committed`; the
        contrapositive is that no SENT row means no call was ever issued.
        A SENT insert still in flight when the process died rolls back, and
        charge() sits after that append returns, so it was never reached
        either.

        REPORTED, NOT SCORED. It violates no gate clause and breaks no
        invariant; it is the price of recovery refusing to conclude
        anything it cannot confirm.

        A fast path that resolved these immediately and voided the slot was
        built at B10 and REVERTED the same day -- it took this number to
        0/60 and introduced a cross-generation double charge to do it
        (POSTMORTEM.md incident 5). The proof it rested on ("no SENT row
        means no call was issued") holds for one process's own state and
        fails against a live worker stalled past its lease TTL. So this
        number is a standing measurement of a real cost that is not
        currently worth what removing it costs, and it stays visible
        rather than being quietly accepted. Reinstating the optimisation
        needs lease fencing in executor.py first.
        """
        return (
            not self.had_sent_row
            and self.provider_accepted == 0
            and self.final_reason == UNRESOLVED_FINAL
        )


def _run_one(
    conn,
    index: int,
    *,
    kill_at: int | None,
    fault: FaultSpec | None,
    index_lag_passes: int,
    label: str,
) -> KillOutcome:
    mandate_id = f"M-CHAOS-{label}-{index:05d}"
    attempt = _build_scenario(conn, mandate_id)
    key = attempt.idempotency_key
    counter = _KillCounter(kill_at)
    client = ChaosClient(
        fault=fault, index_lag_passes=index_lag_passes, counter=counter
    )

    # --- the crash --------------------------------------------------------
    clock.set_frozen(attempt.scheduled_for)
    killing = _KillingConnection(conn, counter)
    try:
        execute(
            killing,
            client,
            attempt,
            _ctx_for(mandate_id, amount_paise=attempt.amount_paise),
            owner=f"chaos-{index}",
            lease_ttl_seconds=LEASE_TTL_SECONDS,
        )
    except _InducedKill:
        pass  # the process is "dead"; everything already written is durable

    window = _classify_window(conn, mandate_id, key, client)

    # --- the restart ------------------------------------------------------
    # The kill is spent: the restarted process is a NEW process and must
    # not inherit the dead one's fate.
    counter.kill_at = None
    # Time passes: the lease expires, which is how recover.py discovers
    # abandoned work at all.
    clock.set_frozen(attempt.scheduled_for + timedelta(seconds=LEASE_TTL_SECONDS + 60))
    for _ in range(RECOVERY_PASSES):
        reconcile(conn, client)
        states = _states_for(conn, mandate_id, key)
        if states and states[-1] in _TERMINAL_STATES and states[-1] != LedgerState.FAILED.value:
            break

    # The scheduler re-queues the same committed attempt after the restart.
    # This is where a double-charge would actually happen if INTENT-first
    # ordering did not hold, so it is not optional.
    try:
        execute(
            conn,
            client,
            attempt,
            _ctx_for(mandate_id, amount_paise=attempt.amount_paise),
            owner=f"chaos-{index}-retry",
            lease_ttl_seconds=LEASE_TTL_SECONDS,
        )
    except _InducedKill:  # pragma: no cover -- no kill armed on this connection
        pass

    states = _states_for(conn, mandate_id, key)
    final_row = None
    rows = [r for r in replay(conn, mandate_id) if r.idempotency_key == key]
    if rows:
        final_row = rows[-1]

    # A job is LOST if, after full recovery and a re-queue, its last word is
    # still INTENT or SENT -- nothing will ever resolve it, and the slot is
    # consumed. UNCONFIRMED and UNRESOLVED_FINAL are FAILED rows: unresolved
    # but REPORTED, which is the opposite of lost.
    lost_job = bool(states) and states[-1] not in _TERMINAL_STATES

    # How many UNCONFIRMED rows this key accumulated -- proof the backoff
    # walk was actually exercised rather than every recovery resolving on
    # the first ask. Zero across the board would mean the modelled index
    # lag never bit, and the B9 resolution path was never really driven.
    unconfirmed_passes = sum(
        1 for r in rows
        if r.state == LedgerState.FAILED.value and r.reason == UNCONFIRMED
    )

    return KillOutcome(
        index=index,
        kill_at_statement=kill_at,
        window=window,
        provider_accepted=client.accepted[key],
        final_state=final_row.state if final_row else None,
        final_reason=final_row.reason if final_row else None,
        double_charged=client.accepted[key] > 1,
        lost_job=lost_job,
        ledger_violation=_ledger_violation(states),
        unconfirmed_passes=unconfirmed_passes,
        had_sent_row=LedgerState.SENT.value in states,
    )


# --- the report --------------------------------------------------------------

@dataclass
class ChaosReport:
    kills: int
    seed: int
    baseline_statements: int
    uniform: list[KillOutcome] = field(default_factory=list)
    fault_seam: list[KillOutcome] = field(default_factory=list)

    @property
    def all_outcomes(self) -> list[KillOutcome]:
        return self.uniform + self.fault_seam

    @property
    def window_counts(self) -> dict[Window, int]:
        counts = {w: 0 for w in Window}
        for o in self.uniform:
            counts[o.window] += 1
        return counts

    @property
    def unsafe_window_kills(self) -> int:
        """THE DENOMINATOR. How many of the uniform kills landed with the
        provider having accepted and nothing local recording it."""
        return self.window_counts[UNSAFE_WINDOW]

    @property
    def double_charges(self) -> int:
        return sum(1 for o in self.all_outcomes if o.double_charged)

    @property
    def lost_jobs(self) -> int:
        return sum(1 for o in self.all_outcomes if o.lost_job)

    @property
    def ledger_violations(self) -> int:
        return sum(1 for o in self.all_outcomes if o.ledger_violation)

    @property
    def unsafe_window_covered(self) -> int:
        """Total kills that reached the unsafe window from EITHER mechanism.
        The gate's "zero double-charges" is only meaningful if this is > 0."""
        return self.unsafe_window_kills + sum(
            1 for o in self.fault_seam if o.window is UNSAFE_WINDOW
        )

    @property
    def slots_burned_unsent(self) -> int:
        return sum(1 for o in self.all_outcomes if o.slot_burned_unsent)

    @property
    def backoff_exercised(self) -> int:
        """Outcomes that actually walked the UNCONFIRMED backoff. Zero would
        mean the modelled index lag never bit and B9's resolution path was
        never really driven by this run."""
        return sum(1 for o in self.all_outcomes if o.unconfirmed_passes > 0)

    @property
    def passed(self) -> bool:
        return (
            self.double_charges == 0
            and self.lost_jobs == 0
            and self.ledger_violations == 0
            and self.unsafe_window_covered > 0
        )

    def render(self) -> str:
        lines: list[str] = []
        add = lines.append
        add("=" * 68)
        add(f"CHAOS REPORT (B10) -- {self.kills} uniform kills, seed={self.seed}")
        add("=" * 68)
        add("")
        add(f"Kill sampling: uniform over [1, {self.baseline_statements}] database")
        add("statements on the happy path. A sample past the last statement of a")
        add("shorter path simply lets the run finish -- scored as post-RESULT.")
        add("")
        add("WINDOW PARTITION (the denominator the gate asks for)")
        add("-" * 68)
        for window in Window:
            count = self.window_counts[window]
            marker = "  <-- UNSAFE" if window is UNSAFE_WINDOW else ""
            add(f"  {window.value:<42} {count:>4}{marker}")
        add("-" * 68)
        add(f"  {'total':<42} {len(self.uniform):>4}")
        add("")

        if self.unsafe_window_kills == 0:
            add("READ THIS BEFORE READING THE HEADLINE:")
            add("  ZERO uniform kills landed in the unsafe window. A kill signal to")
            add("  our own process cannot make the provider accept money and then")
            add("  lose the answer -- it can only destroy work not yet done. So on")
            add("  the uniform block alone, 'zero double-charges' is a statement")
            add("  about the sampler, not about the system.")
            add("")

        add(f"FAULT-SEAM RUNS (provider accepts, response dropped): {len(self.fault_seam)}")
        add("-" * 68)
        add("  Every one lands in the unsafe window by construction, so these")
        add("  prove correctness, not frequency. Reported separately and never")
        add("  folded into the uniform count above.")
        fault_windows = Counter(o.window for o in self.fault_seam)
        for window, count in fault_windows.items():
            add(f"  {window.value:<42} {count:>4}")
        add("")

        resolutions = Counter(
            (o.final_state, o.final_reason) for o in self.fault_seam
        )
        if resolutions:
            add("  How the fault-seam runs resolved:")
            for (state, reason), count in resolutions.items():
                add(f"    {state}/{reason or '-':<28} {count:>4}")
            add("")

        add("HEADLINE")
        add("-" * 68)
        add(f"  double-charges          {self.double_charges}")
        add(f"  lost jobs               {self.lost_jobs}")
        add(f"  ledger violations       {self.ledger_violations}")
        add(f"  unsafe-window coverage  {self.unsafe_window_covered}"
            "   (must be > 0 for the above to mean anything)")
        add(f"  backoff exercised       {self.backoff_exercised}"
            "   (runs that walked UNCONFIRMED at least once)")
        add("")
        add("SLOT ACCOUNTING (reported cost, not a gate clause)")
        add("-" * 68)
        add(f"  NPCI slots burned with NO SENT row   {self.slots_burned_unsent}"
            f" / {len(self.all_outcomes)}")
        add("  An attempt ending at UNRESOLVED_FINAL spends one of only four")
        add("  attempts, ever. On a key with no SENT row it is spending it on")
        add("  an attempt that -- from that process's own state -- never")
        add("  reached the provider. A fast path exploiting that was built")
        add("  and REVERTED at B10: it took this to 0 and bought a cross-")
        add("  generation double charge (POSTMORTEM.md incident 5), because")
        add("  the same evidence cannot distinguish a dead worker from a live")
        add("  one stalled past its lease. The cost stands, and stays visible.")
        add("")

        failures = [o for o in self.all_outcomes if not o.ok]
        if failures:
            add(f"FAILURES ({len(failures)})")
            add("-" * 68)
            for o in failures[:20]:
                why = []
                if o.double_charged:
                    why.append(f"DOUBLE-CHARGED x{o.provider_accepted}")
                if o.lost_job:
                    why.append(f"LOST JOB (last row {o.final_state})")
                if o.ledger_violation:
                    why.append(f"LEDGER: {o.ledger_violation}")
                add(f"  #{o.index:<5} {o.window.value:<40} {'; '.join(why)}")
            if len(failures) > 20:
                add(f"  ... and {len(failures) - 20} more")
            add("")

        add("RESULT: " + ("PASS" if self.passed else "FAIL"))
        add("=" * 68)
        return "\n".join(lines)


# --- drivers -----------------------------------------------------------------

def _measure_baseline(conn) -> int:
    """How many database statements one clean, uncrashed attempt issues --
    the range the uniform sampler draws from. Measured, never guessed: the
    count changes whenever the executor's write ordering changes, and a
    hard-coded range would silently stop covering the whole operation."""
    attempt = _build_scenario(conn, "M-CHAOS-BASELINE")
    clock.set_frozen(attempt.scheduled_for)
    counter = _KillCounter(kill_at=None)
    counting = _KillingConnection(conn, counter)
    execute(
        counting,
        ChaosClient(counter=counter),
        attempt,
        _ctx_for("M-CHAOS-BASELINE", amount_paise=attempt.amount_paise),
        owner="chaos-baseline",
        lease_ttl_seconds=LEASE_TTL_SECONDS,
    )
    return counting.statements


def run(
    conn,
    *,
    kills: int = 50,
    seed: int = 0,
    fault_runs: int = 10,
    index_lag_passes: int = DEFAULT_INDEX_LAG_PASSES,
) -> ChaosReport:
    """Run the full chaos suite against `conn`. Leaves the frozen clock
    reset, so a caller (or a test) is never handed a frozen clock it did
    not set."""
    try:
        baseline = _measure_baseline(conn)
        rng = random.Random(seed)
        report = ChaosReport(kills=kills, seed=seed, baseline_statements=baseline)

        for i in range(kills):
            report.uniform.append(
                _run_one(
                    conn, i,
                    kill_at=rng.randint(1, baseline),
                    fault=None,
                    index_lag_passes=index_lag_passes,
                    label="U",
                )
            )

        for i in range(fault_runs):
            report.fault_seam.append(
                _run_one(
                    conn, i,
                    kill_at=None,
                    fault=FaultSpec(drop_response_after_accept=True),
                    index_lag_passes=index_lag_passes,
                    label="F",
                )
            )
        return report
    finally:
        clock.set_frozen(None)


def _scratch_schema(conn) -> str:
    """A throwaway schema, mirroring tests/conftest.py's pg_schema fixture
    -- `python -m eval.chaos` must never write into a real ledger."""
    from pathlib import Path

    schema = f"chaos_b10_{uuid.uuid4().hex[:16]}"
    schema_sql = (Path(__file__).resolve().parent.parent / "src" / "ledger" / "schema.sql")
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
        cur.execute(f'SET search_path TO "{schema}"')
        cur.execute(schema_sql.read_text(encoding="utf-8"))
    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B10 induced-kill chaos run")
    parser.add_argument("--kills", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault-runs", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    conn = db.connect(autocommit=True)
    schema = _scratch_schema(conn)
    try:
        report = run(conn, kills=args.kills, seed=args.seed, fault_runs=args.fault_runs)
        print(report.render())
        return 0 if report.passed else 1
    finally:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
