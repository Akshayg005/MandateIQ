"""eval/chaos.py -- B10's induced-kill harness, and the regression it found.

Marked `chaos`, so excluded from `test-fast`/`ci` by run.ps1's
$TestFastFilter. Each test builds real committed attempts against a real
Postgres schema and kills a real executor mid-flight, which is slower than
the unit path is allowed to be.

WHAT THIS FILE IS FOR, in order of importance:

  1. Pin POSTMORTEM.md incident 4 -- the permanently lost job the first
     chaos run found, where recovery could not see an INTENT row that had
     no lease row. Pinned at the unit level, directly against
     recover._dangling_keys, because that is where the defect was and a
     whole-harness assertion would only find it again by luck of sampling.
  2. Prove the harness's own oracles are not vacuous. "Zero lost jobs" and
     "zero double-charges" are worth exactly nothing unless the detectors
     can be shown to fire on a state that deserves them, so both are
     driven against a state that must trip them. This is the same standard
     B9's gate applied to its 6(c) race (a discriminating pair, not a
     single happy case) -- see reports/gates.md, B9.
  3. Pin the gate's own clauses so a later change cannot quietly satisfy
     them by shrinking what is measured.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from eval import chaos
from eval.chaos import UNSAFE_WINDOW, ChaosClient, Window, _run_one
from src.core import clock
from src.core.types import LedgerState
from src.execute import executor as executor_module
from src.execute import lease, recover
from src.execute.razorpay_client import FaultSpec
from src.execute.void import reissue
from src.ledger.store import LedgerEntry, append, find_by_key, replay

pytestmark = pytest.mark.chaos


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


# --- 1. the incident-4 regression -------------------------------------------

def test_dangling_scan_sees_an_intent_row_with_no_lease_row(pg_schema, seed_plan):
    """POSTMORTEM.md incident 4, pinned at the defect's own level.

    executor.execute() writes the INTENT row (step 1) BEFORE it claims the
    lease (step 2), so a process dying between them leaves an INTENT row
    and no lease row at all -- not an expired lease, no row. The original
    _dangling_keys() iterated lease.expired() and therefore returned [] for
    this state forever, while step 1's ON CONFLICT DO NOTHING stopped any
    re-run from ever sending. The slot was consumed and nothing reported.

    This is the discriminating case: the pre-fix implementation returns []
    here, the fixed one returns the key.
    """
    conn = pg_schema.conn
    sha = seed_plan("a" * 64, mandate_id="M-INC4")
    append(conn, LedgerEntry(
        idempotency_key="K-INC4", mandate_id="M-INC4", cycle_id=1,
        attempt_index=1, action="ATTEMPT", state=LedgerState.INTENT.value,
        amount_paise=50_000, profile="strict", payload_sha256="b" * 64,
        decision_sha256=sha,
    ))

    assert lease.expired(conn) == [], (
        "precondition: this key must have NO lease row at all -- if it had "
        "one, this test would be re-testing the case that already worked"
    )
    assert "K-INC4" in recover._dangling_keys(conn)


def test_unexpired_lease_still_excludes_a_key_from_the_dangling_scan(pg_schema, seed_plan):
    """The other half of the fix: widening the scan to the ledger must not
    have made it steal work from a worker that is still legitimately
    mid-attempt. A LIVE lease still protects the key."""
    conn = pg_schema.conn
    sha = seed_plan("c" * 64, mandate_id="M-LIVE")
    clock.set_frozen(chaos.CYCLE_START)
    append(conn, LedgerEntry(
        idempotency_key="K-LIVE", mandate_id="M-LIVE", cycle_id=1,
        attempt_index=1, action="ATTEMPT", state=LedgerState.INTENT.value,
        amount_paise=50_000, profile="strict", payload_sha256="d" * 64,
        decision_sha256=sha,
    ))
    lease.claim(conn, "K-LIVE", owner="someone-else", ttl_seconds=300)

    assert recover._dangling_keys(conn) == []

    # ... and once it expires, it becomes fair game again.
    clock.set_frozen(chaos.CYCLE_START + timedelta(seconds=301))
    assert "K-LIVE" in recover._dangling_keys(conn)


# --- 2. the oracles must be able to fire ------------------------------------

def test_double_charge_oracle_fires_when_the_provider_accepts_twice():
    """If ChaosClient.accepted could not reach 2, "zero double-charges"
    would be a tautology rather than a measurement. Drive it to 2 directly
    and assert the oracle notices."""
    client = ChaosClient(index_lag_passes=0)
    client.charge(amount_paise=50_000, receipt="K-DUP", notes={})
    client.charge(amount_paise=50_000, receipt="K-DUP", notes={})
    assert client.accepted["K-DUP"] == 2

    assert chaos.KillOutcome(
        index=0, kill_at_statement=None, window=UNSAFE_WINDOW,
        provider_accepted=client.accepted["K-DUP"], final_state="RESULT",
        final_reason=None, double_charged=client.accepted["K-DUP"] > 1,
        lost_job=False, ledger_violation=None,
    ).ok is False


def test_lost_job_detector_fires_on_a_non_terminal_last_row():
    """The lost-job predicate is "the last ledger row is not terminal".
    Assert it actually rejects such a state -- an always-False detector
    would have made the first chaos run pass and incident 4 invisible."""
    stuck = chaos.KillOutcome(
        index=0, kill_at_statement=2, window=Window.INTENT_TO_LEASE,
        provider_accepted=0, final_state=LedgerState.INTENT.value,
        final_reason=None, double_charged=False, lost_job=True,
        ledger_violation=None,
    )
    assert stuck.ok is False


def test_ledger_violation_detector_rejects_illegal_sequences():
    """Legal prefixes pass; the shapes that would mean real corruption do
    not. A detector that returned None unconditionally would make "ledger
    complete" unfalsifiable."""
    ok_intent_sent_result = [
        LedgerState.INTENT.value, LedgerState.SENT.value, LedgerState.RESULT.value,
    ]
    ok_abort = [LedgerState.INTENT.value, LedgerState.FAILED.value]
    assert chaos._ledger_violation(ok_intent_sent_result) is None
    assert chaos._ledger_violation(ok_abort) is None
    assert chaos._ledger_violation([]) is None

    assert chaos._ledger_violation([LedgerState.SENT.value]) is not None
    assert chaos._ledger_violation(
        [LedgerState.INTENT.value, LedgerState.INTENT.value]
    ) is not None
    assert chaos._ledger_violation(
        [LedgerState.INTENT.value, LedgerState.RESULT.value]
    ) is not None
    assert chaos._ledger_violation(
        [LedgerState.INTENT.value, LedgerState.RESULT.value, LedgerState.SENT.value]
    ) is not None


# --- 3. the gate's own clauses ----------------------------------------------

def test_every_kill_index_resolves_without_double_charge_or_loss(pg_schema):
    """Sweep EVERY kill index on the happy path rather than sampling it.
    The 50-kill run is uniformly sampled and can miss an index entirely
    (seed 0 at 24 kills drew no 6, so the unsafe window went unsampled);
    an exhaustive sweep cannot. Each index must survive recovery."""
    conn = pg_schema.conn
    baseline = chaos._measure_baseline(conn)
    assert baseline >= 6, "the executor should issue at least six observable effects"

    seen: dict[int, Window] = {}
    for k in range(1, baseline + 1):
        outcome = _run_one(
            conn, 900 + k, kill_at=k, fault=None, index_lag_passes=2, label=f"S{k}"
        )
        seen[k] = outcome.window
        assert not outcome.double_charged, f"kill_at={k} double-charged"
        assert not outcome.lost_job, f"kill_at={k} lost the job: {outcome}"
        assert outcome.ledger_violation is None, f"kill_at={k}: {outcome.ledger_violation}"

    # The sweep must actually traverse the state machine, not sit in one
    # window -- otherwise "every index survives" says nothing.
    assert Window.PRE_INTENT in seen.values()
    assert Window.INTENT_TO_LEASE in seen.values(), (
        "the incident-4 window must be reachable, or its regression is untested"
    )
    assert UNSAFE_WINDOW in seen.values(), (
        "an exhaustive sweep must reach the unsafe window; if it cannot, the "
        "kill index does not cover the provider boundary"
    )


def test_fault_seam_puts_the_kill_in_the_unsafe_window(pg_schema):
    """The seam exists because no signal to our own process can make the
    provider accept money and then lose the answer. Assert it produces
    exactly that state, and that recovery resolves it by ASKING -- the
    provider must be charged exactly once, and the key must end at a real
    RESULT rather than a guess."""
    conn = pg_schema.conn
    outcome = _run_one(
        conn, 1, kill_at=None, fault=FaultSpec(drop_response_after_accept=True),
        index_lag_passes=2, label="FS",
    )
    assert outcome.window is UNSAFE_WINDOW
    assert outcome.provider_accepted == 1
    assert not outcome.double_charged
    assert outcome.final_state == LedgerState.RESULT.value
    assert outcome.unconfirmed_passes > 0, (
        "with a modelled index lag the resolution must walk the UNCONFIRMED "
        "backoff, not resolve on the first ask -- otherwise B9's backoff path "
        "is not being exercised here at all"
    )


def test_report_refuses_to_pass_when_the_unsafe_window_was_never_reached(pg_schema):
    """The gate's whole point (the build spec section 8.2, finding 2):
    "zero double-charges" is an artifact of kill sampling unless something
    actually landed in the unsafe window. A report with no unsafe-window
    coverage must FAIL even though every counter reads zero."""
    empty = chaos.ChaosReport(kills=0, seed=0, baseline_statements=7)
    assert empty.double_charges == 0
    assert empty.lost_jobs == 0
    assert empty.ledger_violations == 0
    assert empty.unsafe_window_covered == 0
    assert empty.passed is False, (
        "a report that measured nothing must not pass -- that is exactly the "
        "vacuous headline this gate was written to forbid"
    )


def test_full_run_meets_every_gate_clause(pg_schema):
    """The gate itself, at reduced kill count for test runtime. The 50-kill
    figure the gate names is produced by `.\\run.ps1 chaos -Kills 50`; what
    is asserted here is that every clause holds and the denominator is
    actually reported."""
    report = chaos.run(pg_schema.conn, kills=14, seed=7, fault_runs=3)

    assert report.double_charges == 0
    assert report.lost_jobs == 0
    assert report.ledger_violations == 0
    assert report.unsafe_window_covered > 0
    assert report.passed

    # The denominator is a reported number, not a claim in prose.
    assert sum(report.window_counts.values()) == len(report.uniform)
    rendered = report.render()
    assert "WINDOW PARTITION" in rendered
    assert UNSAFE_WINDOW.value in rendered


# --- 4. what the harness structurally cannot see (the chaos harness review) -----
# Found by the chaos harness pass over B10. The first of these describes a
# real double-charge that a per-receipt oracle -- the harness's own
# ChaosClient.accepted, and a receipt-keyed idempotency check at the real
# Razorpay -- can never see, because it spans two idempotency keys.


def test_a_stalled_worker_cannot_have_its_slot_voided_and_reissued(pg_schema):
    """REGRESSION GUARD for POSTMORTEM.md incident 5.

    A NEVER_SENT fast path in recover.py briefly voided the
    committed_schedule row of any key with no SENT row, on the proof that
    executor.py writes SENT before it charges. That proof is sound about
    one process's own state and WRONG about a concurrent one:

      1. A live worker claims its lease, then STALLS before writing SENT
         (executor.py never re-validates lease ownership before step 3).
      2. The lease TTL expires. The worker is slow, not dead -- but from
         durable state alone the two are indistinguishable.
      3. reconcile() sees latest row INTENT, no unexpired lease, asks the
         provider (a true miss -- nothing charged YET), and concludes
         never-sent. Every premise was true at the instant it was checked.
      4. The worker wakes and completes its real charge.
      5. The voided slot is reissued at generation+1 -- a DIFFERENT key --
         and charged again.

    Two real charges for one NPCI slot, while each key individually shows
    exactly one. This asserts the property that makes step 5 impossible:
    recovery must never void a schedule row it merely believes is unsent,
    so a reissue cannot be minted underneath a worker that is still alive.
    """
    conn = pg_schema.conn
    attempt = chaos._build_scenario(conn, "M-STALL-RACE")
    key0 = attempt.idempotency_key
    client = ChaosClient(index_lag_passes=0)

    # A live worker: INTENT written, lease claimed, then it stalls.
    clock.set_frozen(attempt.scheduled_for)
    append(conn, LedgerEntry(
        idempotency_key=key0, mandate_id=attempt.mandate_id,
        cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
        action=attempt.action, state=LedgerState.INTENT.value,
        amount_paise=attempt.amount_paise, profile=attempt.profile,
        payload_sha256=executor_module._payload_sha256(attempt),
        decision_sha256=attempt.decision_sha256,
    ))
    lease.claim(conn, key0, owner="worker-A", ttl_seconds=300)

    # The stall outlasts the lease. A separate reconciliation pass runs.
    clock.set_frozen(attempt.scheduled_for + timedelta(seconds=400))
    recover.reconcile(conn, client)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT voided_at FROM committed_schedule WHERE idempotency_key = %s",
            (key0,),
        )
        (voided_at,) = cur.fetchone()
    assert voided_at is None, (
        "recovery voided the schedule row of a worker that is stalled, not "
        "dead -- the slot can now be reissued at generation+1 under a new "
        "key while the original worker is still able to complete its charge"
    )

    # And therefore the slot cannot be reissued underneath that worker.
    with pytest.raises(Exception):
        reissue(conn, key0, scheduled_for=attempt.scheduled_for + timedelta(days=3))


def test_unresolved_final_is_a_permanent_dead_end_and_that_is_by_design(pg_schema):
    """A DOCUMENTED LIMITATION, asserted so it cannot change unnoticed.

    `_stuck_keys()` matches `reason = UNCONFIRMED` only, so once a key
    reaches UNRESOLVED_FINAL no code path in this module ever asks about
    it again. If the measured Orders-index lag ever outlasts
    DEFAULT_MAX_UNCONFIRMED_PASSES, a charge that becomes findable later
    stays filed as unresolved forever.

    This is B9's stated design, not a defect found here -- its gate calls
    UNRESOLVED_FINAL "terminal and reported, never a silent drop", and the
    slot correctly stays consumed, so no money-safety invariant breaks.
    What IS worth recording: eval.chaos's `lost_job` predicate ("last
    ledger row is not terminal") structurally cannot see this, because the
    last row IS terminal. A job can be effectively abandoned while looking
    resolved. Anything that wants to correct these needs a path outside
    this module -- the B3 webhook, or an operator tool -- which is B13's
    production loop to own, not B10's.
    """
    conn = pg_schema.conn
    attempt = chaos._build_scenario(conn, "M-DEADEND")
    key = attempt.idempotency_key
    client = ChaosClient(index_lag_passes=0)

    clock.set_frozen(attempt.scheduled_for)
    append(conn, LedgerEntry(
        idempotency_key=key, mandate_id=attempt.mandate_id,
        cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
        action=attempt.action, state=LedgerState.INTENT.value,
        amount_paise=attempt.amount_paise, profile=attempt.profile,
        payload_sha256=executor_module._payload_sha256(attempt),
        decision_sha256=attempt.decision_sha256,
    ))
    lease.claim(conn, key, owner="worker-dead", ttl_seconds=60)
    clock.set_frozen(attempt.scheduled_for + timedelta(seconds=120))

    # Drive it all the way to UNRESOLVED_FINAL against a provider that
    # never finds it.
    for _ in range(recover.DEFAULT_MAX_UNCONFIRMED_PASSES + 2):
        recover.reconcile(conn, client)
    assert find_by_key(conn, key).reason == recover.UNRESOLVED_FINAL

    # The charge now becomes findable -- the index lag finally cleared.
    client._payments[key] = {"id": "pay_late", "status": "captured"}
    for _ in range(3):
        recover.reconcile(conn, client)

    row = find_by_key(conn, key)
    assert row.reason == recover.UNRESOLVED_FINAL, (
        "behaviour changed: UNRESOLVED_FINAL is no longer terminal. That may "
        "be an improvement, but it is a deliberate design change to B9's "
        "stated contract and must be made deliberately, not discovered here"
    )
    assert row.state == LedgerState.FAILED.value


def test_the_harness_never_kills_inside_reconcile_itself(pg_schema):
    """A COVERAGE GAP, recorded rather than silently tolerated.

    `_run_one` wraps only `execute()`'s connection in the kill mechanism;
    its recovery loop calls `reconcile(conn, client)` on the plain
    connection. So every crash this harness induces happens during the
    ATTEMPT, never during the RECONCILIATION -- and reconcile() performs
    durable writes of its own that a crash could interrupt.

    Asserted here so the gap is a known, stated limit of the reported
    denominator rather than an assumption. Closing it means threading a
    _KillCounter through the recovery loop too; that is a harness change,
    and B10's gate is about crashes during the money path.
    """
    import inspect

    source = inspect.getsource(chaos._run_one)
    assert "reconcile(conn, client)" in source, (
        "the recovery loop no longer uses the plain connection -- if kills "
        "have been threaded into reconcile(), delete this test and report "
        "the reconcile windows in the partition"
    )
    assert "_KillingConnection(conn, counter)" in source
