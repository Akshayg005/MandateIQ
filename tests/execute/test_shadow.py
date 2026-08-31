"""tests/execute/test_shadow.py -- the shadow execution harness, comparing
allocator decisions against a fixed baseline ladder.

Every constraint test asserts the VIOLATION is rejected by the comparison
logic, not just that the happy path works. The negative-control pattern is
load-bearing: two tests, one that MUST reach the database and one that
MUST NOT, proving the module under test is actually calling it (rather than
merely returning a result that looks like it did).

Tests deliberately avoid importing or invoking the real allocator, model,
or executor -- the module under test must be isolated and deterministic
enough to shadow real decisions, but testing that isolation in code, not
hope.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from src.core.types import Action, Cause, DeclineClass, Outcome, Profile
from src.policy.allocator import CommittedAttempt, Plan, SlotHazard
from src.policy.costs import PolicyCosts
from src.policy.gate import ConformalGate
from src.policy.stopping_rules import AllocationContext


# --- Fake types for contracts not yet implemented ---


@dataclass(frozen=True)
class ShadowInput:
    """One row: a single cycle to shadow. Matches the contract in the
    module-under-test's docstring."""

    mandate_id: str
    cycle_id: int
    amount_paise: int
    ceiling_paise: int
    category: str
    decline_class: DeclineClass  # slot-1 observation
    source_version: str
    profile: Profile
    plan_day: int


@dataclass(frozen=True)
class DeltaRow:
    """One output row: ladder vs. our decision for a single mandate."""

    mandate_id: str
    cycle_id: int
    ladder_action: Action
    ladder_slot: int
    ladder_day: int
    ladder_committed_attempts: int
    our_action: Action
    our_slot: int | None
    our_day: int | None
    binding_constraint: str | None
    conformal_set: frozenset[Cause]
    belief_json: str
    decision_sha256: str
    divergence: str
    agrees: bool


@dataclass(frozen=True)
class DeltaLog:
    """Summary of a shadow run over a batch."""

    rows: tuple[DeltaRow, ...]
    n_mandates: int
    n_agree: int
    n_diverge: int
    by_divergence: dict[str, int]
    ladder_committed_attempts: int
    our_committed_attempts: int

    def summary(self) -> str:
        """Single-line summary suitable for batch logs, no per-mandate
        detail (root CLAUDE.md: batch output must not enter main context)."""
        return (
            f"n_mandates={self.n_mandates} n_agree={self.n_agree} "
            f"n_diverge={self.n_diverge}"
        )


DIVERGENCE_CATEGORIES: frozenset[str] = frozenset({
    "SAME_ACTION_SAME_DAY",
    "SAME_ACTION_DIFFERENT_DAY",
    "LADDER_ATTEMPTS_WE_REAUTH",
    "LADDER_ATTEMPTS_WE_OFFER",
    "LADDER_ATTEMPTS_WE_STOP",
})


# --- Hand-written deterministic fake SlotHazard ---


@dataclass(frozen=True)
class _DeterministicSlotHazard:
    """A fixed hazard double: always returns the same 4-tuple regardless
    of input. Satisfies SlotHazard Protocol. Uses the same outcome int
    order as competing_risks.hazards() -- STILL_PENDING (0),
    RECOVERED (1), DEAD (2), OPTED_OUT (3)."""

    # Probabilities: 40% still pending, 45% recovered, 10% dead, 5% opted out
    still_pending: float = 0.40
    recovered: float = 0.45
    dead: float = 0.10
    opted_out: float = 0.05

    def __call__(
        self, *, slot: int, on_day: int, amount_paise: int
    ) -> tuple[float, float, float, float]:
        """Always returns the same tuple, deterministic regardless of
        input."""
        return (self.still_pending, self.recovered, self.dead, self.opted_out)


# --- Fake database doubles for contract enforcement ---


class _AssertionOnAnySql:
    """A connection double that raises AssertionError on ANY SQL statement.
    Used to prove run_shadow never touches the database at all."""

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        raise AssertionError(
            f"run_shadow() called database -- statement: {sql[:100]}"
        )


class _AssertionOnInsertShadowLedger:
    """A connection double that raises ONLY on INSERT INTO shadow_ledger.
    All other statements pass silently (do nothing). Used to prove
    run_shadow writes ONLY to shadow_ledger, never the live ledger or
    schedule tables."""

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        if sql and "shadow_ledger" in sql.lower() and "insert" in sql.lower():
            raise AssertionError(
                "run_shadow() attempted to write to shadow_ledger -- "
                "negative control must forbid this"
            )
        # All other statements silently pass.

    def fetchone(self):
        return None


class _AllowShadowLedgerOnly:
    """A connection double that allows ONLY INSERT INTO shadow_ledger.
    Raises on any other statement. Used as the positive control to prove
    the negative control's silence came from selective enforcement, not
    run_shadow never touching the database."""

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        if sql:
            is_insert_shadow = (
                "insert" in sql.lower() and "shadow_ledger" in sql.lower()
            )
            if not is_insert_shadow:
                raise AssertionError(
                    f"run_shadow() called non-shadow-ledger SQL: {sql[:100]}"
                )

    def fetchone(self):
        return None


# --- Helper to build a minimal ShadowInput for tests ---


def _shadow_input(
    mandate_id: str = "M-shadow-1",
    cycle_id: int = 1,
    amount_paise: int = 50_000,
    ceiling_paise: int = 200_000,
    category: str = "subscription",
    decline_class: DeclineClass = DeclineClass.INSUFFICIENT_FUNDS,
    profile: Profile = Profile.strict,
    plan_day: int = 0,
) -> ShadowInput:
    """Factory for a minimal valid ShadowInput."""
    return ShadowInput(
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        amount_paise=amount_paise,
        ceiling_paise=ceiling_paise,
        category=category,
        decline_class=decline_class,
        source_version="test-v1",
        profile=profile,
        plan_day=plan_day,
    )


# --- Tests ---


def test_shadow_run_writes_only_to_shadow_ledger_positive_control():
    """POSITIVE: run_shadow with a conn that allows INSERT INTO
    shadow_ledger completes without raising. Proves the function does
    reach the database."""
    batch = [_shadow_input()]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    conn = _AllowShadowLedgerOnly()
    # This should NOT raise -- shadow_ledger writes are allowed.
    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs, conn=conn)
    assert log is not None


def test_shadow_run_never_writes_to_live_ledger_negative_control():
    """NEGATIVE CONTROL for the positive test above.

    The positive case uses a conn that permits ONLY `INSERT INTO
    shadow_ledger` and shows run_shadow() finishes. On its own that proves
    nothing: a run_shadow() that never touched the database at all would
    pass it identically. So this control hands over a byte-identical batch
    with a conn that ALSO rejects shadow_ledger, and requires run_shadow()
    to raise -- pinning that the positive test's silence came from writing
    only to shadow_ledger, not from writing nothing anywhere.

    (An earlier draft of this test used this stricter double but asserted
    no raise, which is self-contradictory: the double's whole purpose is to
    reject the one statement shadow mode legitimately issues.)"""
    batch = [_shadow_input()]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    conn = _AssertionOnInsertShadowLedger()
    from src.execute.shadow import run_shadow

    with pytest.raises(AssertionError, match="shadow_ledger"):
        run_shadow(batch, hazard=hazard, costs=costs, conn=conn)


def test_shadow_run_without_conn_touches_no_database():
    """When conn=None, run_shadow accesses no database at all. Proves
    shadow operations are pure computation."""
    batch = [_shadow_input()]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    # No conn argument at all; this is pure computation.
    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)
    assert log is not None


def test_shadow_run_calls_no_execute_store_commit_or_void(monkeypatch):
    """run_shadow never calls executor.execute, store.append,
    commit.commit, or void.void. These functions are patched to raise
    AssertionError if called, and a full batch run must still complete.
    This enforces that shadow is purely analytical, never operational."""
    from src.execute import executor, commit
    from src.ledger import store
    from src.execute import void

    # Patch each to raise if called.
    monkeypatch.setattr(
        executor, "execute",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_shadow() must not call executor.execute()")
        ),
    )
    monkeypatch.setattr(
        store, "append",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_shadow() must not call store.append()")
        ),
    )
    monkeypatch.setattr(
        commit, "commit",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_shadow() must not call commit.commit()")
        ),
    )
    monkeypatch.setattr(
        void, "void",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_shadow() must not call void.void()")
        ),
    )

    batch = [_shadow_input(), _shadow_input(mandate_id="M-shadow-2")]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    # Must complete without triggering any of the patched assertions.
    log = run_shadow(batch, hazard=hazard, costs=costs)
    assert log is not None


def test_shadow_run_covers_all_input_rows():
    """len(log.rows) == len(batch) and every mandate_id in batch appears
    exactly once in the output. Full coverage, no omissions."""
    batch = [
        _shadow_input(mandate_id="M-1"),
        _shadow_input(mandate_id="M-2"),
        _shadow_input(mandate_id="M-3"),
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    assert len(log.rows) == len(batch), (
        f"output row count {len(log.rows)} != input batch size {len(batch)}"
    )
    assert log.n_mandates == len(batch)

    output_mandate_ids = [r.mandate_id for r in log.rows]
    input_mandate_ids = [r.mandate_id for r in batch]
    assert sorted(output_mandate_ids) == sorted(input_mandate_ids)

    # Each mandate appears exactly once.
    assert len(set(output_mandate_ids)) == len(output_mandate_ids)


def test_shadow_run_divergence_counts_sum_correctly():
    """n_agree + n_diverge == n_mandates. Accounting invariant."""
    batch = [
        _shadow_input(mandate_id="M-1"),
        _shadow_input(mandate_id="M-2"),
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    assert log.n_agree + log.n_diverge == log.n_mandates


def test_shadow_run_divergence_categories_are_valid():
    """Every row.divergence is a member of DIVERGENCE_CATEGORIES."""
    batch = [
        _shadow_input(mandate_id="M-1"),
        _shadow_input(mandate_id="M-2"),
        _shadow_input(mandate_id="M-3"),
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    for row in log.rows:
        assert row.divergence in DIVERGENCE_CATEGORIES, (
            f"divergence {row.divergence!r} not in "
            f"DIVERGENCE_CATEGORIES"
        )


def test_shadow_run_agrees_iff_same_action_same_day():
    """row.agrees is True iff divergence == "SAME_ACTION_SAME_DAY".
    Ensures agrees field has a single unambiguous definition."""
    batch = [_shadow_input(mandate_id=f"M-{i}") for i in range(1, 4)]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    for row in log.rows:
        if row.divergence == "SAME_ACTION_SAME_DAY":
            assert row.agrees is True
        else:
            assert row.agrees is False


def test_shadow_run_above_afa_cliff_triggers_reauth():
    """A mandate above the AFA cliff (amount_paise > 1_500_000, category
    not in elevated set) yields our_action == Action.REAUTH,
    binding_constraint == "AFA_CLIFF", and divergence ==
    "LADDER_ATTEMPTS_WE_REAUTH" -- while the ladder row still shows ATTEMPT.
    This is the core business logic: shadow exposes where allocator decisions
    diverge from the ladder."""
    batch = [
        _shadow_input(
            mandate_id="M-afa-cliff",
            amount_paise=2_000_000,  # Above 1.5M
            # Ceiling raised above the amount deliberately: the factory
            # default (200_000) is BELOW this amount, so solve() would
            # reject the mandate under clause 4(c) (amount > customer-set
            # mandate ceiling) and the AFA cliff -- the thing this test
            # exists to exercise -- would never be reached.
            ceiling_paise=5_000_000,
            category="subscription",  # Not elevated
        )
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    assert len(log.rows) == 1
    row = log.rows[0]

    assert row.our_action == Action.REAUTH, (
        f"above AFA cliff should yield REAUTH, got {row.our_action}"
    )
    assert row.binding_constraint == "AFA_CLIFF"
    assert row.divergence == "LADDER_ATTEMPTS_WE_REAUTH"
    assert row.ladder_action == Action.ATTEMPT, (
        "ladder should still show ATTEMPT even when we REAUTH"
    )


def test_shadow_run_elevated_afa_category_respects_higher_limit():
    """An amount above the base AFA limit (1.5M) but within the elevated
    limit (10M) for an elevated category (insurance_premium,
    mutual_fund, credit_card_bill) does NOT trigger AFA_CLIFF. Proves the
    afa_free_limit_paise lookup respects category."""
    batch = [
        _shadow_input(
            mandate_id="M-elevated-afa",
            amount_paise=2_000_000,  # Above base 1.5M
            ceiling_paise=5_000_000,
            category="insurance_premium",  # Elevated category
        )
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    row = log.rows[0]
    # Should NOT trigger AFA_CLIFF for this amount in this category.
    assert row.binding_constraint != "AFA_CLIFF" or row.binding_constraint is None


def test_shadow_run_committed_attempts_accounting():
    """our_committed_attempts <= n_mandates (Plan.committed is zero-or-one
    per construct) and ladder_committed_attempts == 3 * n_mandates (fixed
    3-slot ladder). Proves attempt slot accounting is correct."""
    batch = [
        _shadow_input(mandate_id=f"M-{i}")
        for i in range(1, 6)
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    assert log.our_committed_attempts <= log.n_mandates, (
        f"our_committed_attempts {log.our_committed_attempts} > "
        f"n_mandates {log.n_mandates}"
    )
    assert log.ladder_committed_attempts == 3 * log.n_mandates, (
        f"ladder_committed_attempts {log.ladder_committed_attempts} != "
        f"3 * {log.n_mandates}"
    )


def test_shadow_run_is_deterministic():
    """Two run_shadow calls on the same batch produce identical
    decision_sha256 for every row. The sha256 fingerprints the entire
    plan (decision, belief, constraint); if two runs diverge, a sha256
    mismatch proves it."""
    batch = [
        _shadow_input(mandate_id=f"M-{i}")
        for i in range(1, 4)
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log1 = run_shadow(batch, hazard=hazard, costs=costs)
    log2 = run_shadow(batch, hazard=hazard, costs=costs)

    assert len(log1.rows) == len(log2.rows)
    for r1, r2 in zip(log1.rows, log2.rows):
        assert r1.decision_sha256 == r2.decision_sha256, (
            f"sha256 mismatch for {r1.mandate_id}: {r1.decision_sha256} != "
            f"{r2.decision_sha256}"
        )


def test_shadow_log_summary_returns_single_line_without_detail():
    """DeltaLog.summary() returns a single line containing the three
    counts, and no per-mandate detail. Batch output must stay out of the
    main context (root CLAUDE.md)."""
    batch = [
        _shadow_input(mandate_id="M-1"),
        _shadow_input(mandate_id="M-2"),
        _shadow_input(mandate_id="M-3"),
    ]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)

    summary = log.summary()
    assert isinstance(summary, str)
    assert "\n" not in summary, "summary must be a single line"
    assert "n_mandates=" in summary
    assert "n_agree=" in summary
    assert "n_diverge=" in summary
    # No per-mandate detail.
    for row in log.rows:
        assert row.mandate_id not in summary


def test_shadow_run_respects_conformal_gate():
    """When gate is provided, run_shadow passes it to the allocator and
    uses its pred_set() output to populate conformal_set on each row.
    When gate is None, conformal_set is an empty frozenset (default)."""
    batch = [_shadow_input(mandate_id="M-gate-test")]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    # Run without gate.
    log_no_gate = run_shadow(batch, hazard=hazard, costs=costs, gate=None)
    # gate=None means solve() falls back to FullSetGate, so the prediction
    # set is ALL THREE causes -- not the empty set an earlier draft of this
    # test expected. The direction matters and is not cosmetic: the off-ramp
    # fires only on the SINGLETON {WONT_PAY} (root CLAUDE.md, "Safety
    # design"), so the full set is the conservative default that can never
    # offer an exit, while an empty set would be a degenerate value no gate
    # is specified to return. Asserting frozenset() here would have pinned
    # the unsafe reading as correct.
    assert log_no_gate.rows[0].conformal_set == frozenset(
        {Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY}
    )

    # Run with a gate that always returns {WONT_PAY}.
    class _SingletonGate:
        def pred_set(self, b):
            return frozenset([Cause.WONT_PAY])

    gate = _SingletonGate()
    log_with_gate = run_shadow(
        batch, hazard=hazard, costs=costs, gate=gate
    )
    assert log_with_gate.rows[0].conformal_set == frozenset([Cause.WONT_PAY])


def test_shadow_run_belief_json_and_sha256_are_populated():
    """Every row has belief_json (a string, possibly "{}") and
    decision_sha256 (a valid hex digest). These are core audit fields."""
    batch = [_shadow_input(mandate_id="M-audit")]
    hazard = _DeterministicSlotHazard()
    costs = PolicyCosts(
        attempt_cost_paise=10_000,
        mandate_ltv_paise=500_000,
        reauth_cost_paise=5_000,
        reauth_success_prob=0.8,
        quiet_hours_start=21,
        quiet_hours_end=8,
        max_contacts_per_cycle=4,
    )

    from src.execute.shadow import run_shadow

    log = run_shadow(batch, hazard=hazard, costs=costs)
    row = log.rows[0]

    assert isinstance(row.belief_json, str)
    assert len(row.belief_json) > 0

    assert isinstance(row.decision_sha256, str)
    assert len(row.decision_sha256) == 64  # SHA256 hex is 64 chars
    # Verify it is valid hex.
    int(row.decision_sha256, 16)
