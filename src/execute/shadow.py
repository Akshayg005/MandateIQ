"""B12 shadow mode: decide without executing, and log the delta against what
the fixed T+1/T+2/T+3 ladder would have done.

This is how payment systems are actually rolled out -- the new policy runs
beside the incumbent over real traffic, its decisions are recorded, and
nobody's money moves until the delta has been read by a human. It is also
the direct answer to "what about real data": every number in reports/ comes
from a frozen simulator, but this module runs unchanged against a live batch.

=== What it must never do ====================================================

Shadow mode touches NO money and writes NO row any real path can read as an
executed attempt. Concretely, run_shadow():

  - never calls src/execute/executor.py's execute(), commit.commit(),
    void.void(), or src/ledger/store.py's append()
  - never writes `ledger`, `committed_schedule`, `attempt_lease` or `plan`
  - never constructs a provider client, so no charge can be issued even by
    accident
  - writes only `shadow_ledger`, a separate table (src/ledger/schema.sql)
    whose decision_sha256 deliberately carries NO foreign key to `plan` --
    honouring that FK would force shadow mode to write real plan rows in
    order to observe

tests/execute/test_shadow.py enforces the first two as a positive/negative
control PAIR: a conn double that permits only `INSERT INTO shadow_ledger`
must let run_shadow() finish, AND a double that also rejects shadow_ledger
must make it raise. Without the second, "no forbidden statement was issued"
would be satisfied by a function that never touched the database at all --
the vacuous shape this repo audited out of its gates on 2026-08-29.

This separation is cheap because src/policy/allocator.py's solve() is
already pure and DB-free by its own docstring. Shadow mode is solve() plus
bookkeeping; it is not a second implementation of the policy, and it must
never become one.

=== What the delta actually compares =========================================

The comparison is at the FIRST DECISION POINT -- the moment a mandate enters
recovery, its original debit having already failed. That is what a shadow
deployment genuinely observes, and it is the only comparison available to a
system that by definition does not execute.

The ladder's side needs no simulation: its schedule is fixed and known in
advance (that is the entire criticism of it). It commits three attempts, at
T+1/T+2/T+3, for every mandate, regardless of cause, amount, or whether the
instrument is already dead. Our side commits at most ONE attempt and
re-decides after each observation -- Plan.committed is "zero or one, never
more".

What this file therefore does NOT produce is a money delta. "We would have
recovered X more" requires executing both policies, which is eval/'s job
(B13) and not shadow mode's. Reporting a recovered-money figure here would
mean inventing outcomes for attempts that were never made.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.core import clock
from src.core.types import Action, Cause, DeclineClass, MandateState, Profile
from src.policy import belief as belief_mod
from src.policy.allocator import Plan, SlotHazard, solve
from src.policy.costs import PolicyCosts
from src.policy.gate import ConformalGate
from src.policy.stopping_rules import AllocationContext

# The incumbent's fixed cadence: slot 2 at T+1, slot 3 at T+2, slot 4 at T+3.
# Mirrors eval/frozen/sim_config.yaml's `baseline_ladder_offsets_days`, which
# is frozen. Restated as a constant here rather than read from eval/ so that
# production code never imports the evaluation harness -- the dependency
# would run backwards, and eval/frozen/ would become load-bearing for a
# module that has to run against live traffic.
LADDER_OFFSETS_DAYS: Mapping[int, int] = {2: 1, 3: 2, 4: 3}
LADDER_FIRST_SLOT = 2
LADDER_COMMITTED_ATTEMPTS = len(LADDER_OFFSETS_DAYS)

DIVERGENCE_CATEGORIES: frozenset[str] = frozenset(
    {
        "SAME_ACTION_SAME_DAY",
        "SAME_ACTION_DIFFERENT_DAY",
        "LADDER_ATTEMPTS_WE_REAUTH",
        "LADDER_ATTEMPTS_WE_OFFER",
        "LADDER_ATTEMPTS_WE_STOP",
    }
)

_ACTION_TO_DIVERGENCE: Mapping[Action, str] = {
    Action.REAUTH: "LADDER_ATTEMPTS_WE_REAUTH",
    Action.OFFER: "LADDER_ATTEMPTS_WE_OFFER",
    Action.STOP: "LADDER_ATTEMPTS_WE_STOP",
}


@dataclass(frozen=True)
class ShadowInput:
    """One mandate at its first decision point.

    `decline_class` and `source_version` are INPUTS, not something this
    module infers. In production they come from a real issuer decline that
    has been round-tripped through the ledger (src/ledger/store.py's
    find_normalized_decline, per B11's gate). Against the frozen batch the
    driver supplies a simulated one and stamps it with a source_version that
    says so. Keeping the observation on the boundary is what lets the same
    function run in both settings without a mode flag.
    """

    mandate_id: str
    cycle_id: int
    amount_paise: int
    ceiling_paise: int
    category: str
    decline_class: DeclineClass
    source_version: str
    profile: Profile
    plan_day: int


@dataclass(frozen=True)
class DeltaRow:
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
    rows: tuple[DeltaRow, ...]
    n_mandates: int
    n_agree: int
    n_diverge: int
    by_divergence: dict[str, int]
    ladder_committed_attempts: int
    our_committed_attempts: int

    def summary(self) -> str:
        """One line. Deliberately carries no per-mandate detail: root
        CLAUDE.md's context discipline makes batch output the single largest
        context cost, and a 200-row dump is exactly what that rule exists to
        prevent. Per-mandate rows go to shadow_ledger and to the report
        file, where a human or a subagent reads them."""
        parts = " ".join(f"{k}={self.by_divergence.get(k, 0)}" for k in sorted(DIVERGENCE_CATEGORIES))
        return (
            f"n_mandates={self.n_mandates} n_agree={self.n_agree} "
            f"n_diverge={self.n_diverge} "
            f"attempts_committed ladder={self.ladder_committed_attempts} "
            f"ours={self.our_committed_attempts} | {parts}"
        )


def _initial_context(item: ShadowInput, costs: PolicyCosts) -> AllocationContext:
    """The context at the first decision point. attempts_used=1 because slot
    1, the original debit, is already spent by definition -- a mandate only
    reaches a recovery engine after its original attempt failed. Getting
    this wrong would hand the allocator a fourth retry it does not have
    (NPCI: 1 original + 3 retries, ever)."""
    return AllocationContext(
        mandate_id=item.mandate_id,
        cycle_id=item.cycle_id,
        profile=item.profile,
        amount_paise=item.amount_paise,
        ceiling_paise=item.ceiling_paise,
        category=item.category,
        plan_day=item.plan_day,
        attempts_used=1,
        committed_days=(),
        contacts_sent=0,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def _classify_divergence(plan: Plan, ladder_day: int) -> str:
    if plan.chosen_action is not Action.ATTEMPT:
        return _ACTION_TO_DIVERGENCE[plan.chosen_action]
    if not plan.committed:
        # ATTEMPT with nothing committed would mean the allocator chose to
        # spend a slot and then named no day for it. Not reachable through
        # solve(); raised rather than silently bucketed so it cannot become
        # a quietly-miscounted row.
        raise ValueError(
            f"{plan.mandate_id}: chosen_action is ATTEMPT but no attempt was committed"
        )
    return (
        "SAME_ACTION_SAME_DAY"
        if plan.committed[0].on_day == ladder_day
        else "SAME_ACTION_DIFFERENT_DAY"
    )


def run_shadow(
    batch: Sequence[ShadowInput],
    *,
    hazard: SlotHazard,
    costs: PolicyCosts,
    gate: ConformalGate | None = None,
    conn=None,
    run_id: str | None = None,
) -> DeltaLog:
    """Decide over `batch` without executing anything, and return the delta
    against the fixed ladder.

    `conn` is optional. When None, no database is touched at all -- the log
    is returned in memory and the caller writes it wherever it likes. That
    is what keeps the B12 gate ("shadow mode produces a delta log over the
    full batch") reproducible on a machine with no Postgres running, rather
    than making the gate depend on Docker being up.

    `run_id` identifies one shadow run over one batch. Defaulted from the
    frozen clock plus a content hash of the batch, so two runs are distinct
    rows rather than a primary-key collision; pass it explicitly to make a
    run content-addressed instead.
    """
    if not batch:
        raise ValueError("run_shadow() called with an empty batch")

    ladder_day = LADDER_OFFSETS_DAYS[LADDER_FIRST_SLOT]
    rows: list[DeltaRow] = []

    for item in batch:
        b = belief_mod.update(
            belief_mod.init(dict(zip(belief_mod.CAUSE_ORDER, belief_mod.REFERENCE_PRIOR))),
            item.decline_class,
            source_version=item.source_version,
        )
        ctx = _initial_context(item, costs)
        plan = solve(b, ctx, hazard=hazard, costs=costs, gate=gate)

        committed = plan.committed[0] if plan.committed else None
        divergence = _classify_divergence(plan, ladder_day)
        rows.append(
            DeltaRow(
                mandate_id=item.mandate_id,
                cycle_id=item.cycle_id,
                ladder_action=Action.ATTEMPT,
                ladder_slot=LADDER_FIRST_SLOT,
                ladder_day=ladder_day,
                ladder_committed_attempts=LADDER_COMMITTED_ATTEMPTS,
                our_action=plan.chosen_action,
                our_slot=committed.slot if committed else None,
                our_day=committed.on_day if committed else None,
                binding_constraint=plan.binding_constraint,
                conformal_set=plan.conformal_set,
                belief_json=plan.belief_json,
                decision_sha256=plan.decision_sha256,
                divergence=divergence,
                agrees=divergence == "SAME_ACTION_SAME_DAY",
            )
        )

    by_divergence: dict[str, int] = {}
    for row in rows:
        by_divergence[row.divergence] = by_divergence.get(row.divergence, 0) + 1

    log = DeltaLog(
        rows=tuple(rows),
        n_mandates=len(rows),
        n_agree=sum(1 for r in rows if r.agrees),
        n_diverge=sum(1 for r in rows if not r.agrees),
        by_divergence=by_divergence,
        ladder_committed_attempts=LADDER_COMMITTED_ATTEMPTS * len(rows),
        our_committed_attempts=sum(1 for r in rows if r.our_slot is not None),
    )

    if conn is not None:
        _persist(conn, log, run_id=run_id or _default_run_id(batch), batch=batch)
    return log


def _default_run_id(batch: Sequence[ShadowInput]) -> str:
    """`shadow-<timestamp>-<batch hash>`.

    money-auditor (2026-08-31) flagged that this is second-precision, so a
    repeat of the same batch under a FROZEN clock yields an identical run_id
    and the second _persist() dies on the shadow_ledger primary key. The
    finding is real; its suggested fix -- adding microseconds -- is not,
    because a frozen clock returns the same microseconds too, which is the
    entire point of freezing it. Verified rather than applied on the
    reviewer's word.

    Left as-is deliberately. Under a frozen clock a deterministic run_id is
    correct behaviour, and the collision is the database saying "this exact
    run at this exact instant is already recorded" -- which is true. The
    real-world case (an advancing clock, two runs inside one second) needs a
    200-mandate batch to complete in under a second, and callers who want a
    guaranteed-fresh or a content-addressed row pass `run_id` explicitly.
    The failure is a loud IntegrityError, never a silent overwrite, which is
    the property that actually matters for an append-only observer.
    """
    sig = hashlib.sha256(
        json.dumps(
            [[i.mandate_id, i.cycle_id, i.profile.value] for i in batch], sort_keys=True
        ).encode()
    ).hexdigest()[:8]
    return f"shadow-{clock.now():%Y%m%dT%H%M%S}-{sig}"


def render_report(log: DeltaLog, *, arm: str, profile: Profile, run_id: str) -> str:
    """The human-readable delta report. Per-mandate rows go to the .jsonl
    beside it; this stays skimmable."""
    lines = [
        "# Shadow-mode delta log",
        "",
        f"run_id `{run_id}` · arm `{arm}` · profile `{profile.value}` · "
        f"{log.n_mandates} mandates",
        "",
        "Decisions only. Nothing was executed, no provider was called, and no row",
        "was written to `ledger`, `committed_schedule`, `attempt_lease` or `plan`.",
        "",
        "## Committed attempts, at the first decision point",
        "",
        f"| | attempts committed |",
        f"|---|---|",
        f"| fixed ladder (T+1/T+2/T+3, every mandate) | {log.ladder_committed_attempts} |",
        f"| this system | {log.our_committed_attempts} |",
        "",
        "The ladder commits three attempts per mandate up front regardless of cause;",
        "this system commits at most one and re-decides after each observation.",
        "",
        "## Where the two policies disagree",
        "",
        "| divergence | mandates |",
        "|---|---|",
    ]
    for name in sorted(DIVERGENCE_CATEGORIES):
        lines.append(f"| `{name}` | {log.by_divergence.get(name, 0)} |")
    binding: dict[str, int] = {}
    for row in log.rows:
        key = row.binding_constraint or "(none -- decided on belief and expected value)"
        binding[key] = binding.get(key, 0) + 1
    lines += [
        "",
        f"Agree: {log.n_agree} · diverge: {log.n_diverge}",
        "",
        "## What bound each decision",
        "",
        "| binding constraint | mandates |",
        "|---|---|",
    ]
    for name, count in sorted(binding.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "This decomposition matters: a hard constraint and a belief are not the",
        "same kind of reason. `AFA_CLIFF` rows are routed by regulation (clause",
        "8(a)) and would be routed identically by any compliant system. Every",
        "other REAUTH is this system's own inference that the instrument is dead",
        "-- which is the part that can be wrong, in both directions.",
        "",
        "## What this report does NOT say",
        "",
        "- **No money delta.** \"We would have recovered X more\" requires executing",
        "  both policies against outcomes; shadow mode by definition executes neither.",
        "  That comparison is the frozen eval's (B13), not this file's.",
        "- **The slot-1 decline signal is simulated here.** Against the frozen batch",
        "  it is drawn from the simulator's own generative parameters, not observed",
        "  from an issuer. `source_version` on every row says so. Against live",
        "  traffic the same function reads a real normalised decline.",
        "- **One decision point per mandate**, not a full retry cycle.",
        "- **Nothing about the off-ramp.** `LADDER_ATTEMPTS_WE_OFFER` is 0 because",
        "  no `ConformalGate` is passed here, so `solve()` falls back to",
        "  `FullSetGate` and the prediction set is never the singleton",
        "  `{WONT_PAY}` an OFFER requires. That is the safe default working as",
        "  specified, not a defect -- but this log is therefore silent on the",
        "  off-ramp and must not be read as evidence about it either way.",
        "- **Nothing about retry timing.** `SAME_ACTION_DIFFERENT_DAY` is 0:",
        "  every attempt this system commits lands on the same slot and day the",
        "  ladder would have picked. The timing discrimination the thesis claims",
        "  is not visible at the first decision point, and this report does not",
        "  demonstrate it.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Drive shadow mode over the frozen 200-mandate batch.

    eval/ is imported HERE, inside the driver, not at module scope: shadow
    mode has to run against live traffic, and a production module that
    imports the evaluation harness at import time would make eval/frozen/
    load-bearing for it. run_shadow() itself has no eval dependency.
    """
    import argparse
    import json as _json
    import pathlib
    import random

    from eval import allocator_sweep
    from eval.frozen.simulator import Simulator, load_config
    from src.policy.costs import load as load_costs

    ap = argparse.ArgumentParser(description="B12 shadow mode over the frozen batch")
    ap.add_argument("--arm", default="nominal")
    ap.add_argument("--profile", default="strict", choices=[p.value for p in Profile])
    ap.add_argument("--seed", type=int, default=0, help="stream for the slot-1 decline draw")
    ap.add_argument("--out", default="reports/shadow_delta")
    args = ap.parse_args(argv)

    profile = Profile(args.profile)
    config = load_config()
    sim = Simulator(args.arm, seed=config["seed"])
    costs = load_costs()
    hazard = allocator_sweep.hazard_from_fit(allocator_sweep.fit_nominal_hazard_model())
    rng = random.Random(args.seed)

    batch = [
        ShadowInput(
            mandate_id=m.mandate_id,
            cycle_id=m.cycle_id,
            amount_paise=m.amount_paise,
            ceiling_paise=m.ceiling_paise,
            category=m.category,
            decline_class=allocator_sweep.draw_slot1_decline(m.initial_cause, config, rng),
            source_version=allocator_sweep.PROXY_SOURCE_VERSION,
            profile=profile,
            plan_day=0,
        )
        for m in sim.mandates
    ]

    log = run_shadow(batch, hazard=hazard, costs=costs)
    run_id = _default_run_id(batch)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(
        render_report(log, arm=args.arm, profile=profile, run_id=run_id), encoding="utf-8"
    )
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for row in log.rows:
            fh.write(
                _json.dumps(
                    {
                        "run_id": run_id,
                        "mandate_id": row.mandate_id,
                        "cycle_id": row.cycle_id,
                        "ladder_action": row.ladder_action.value,
                        "ladder_slot": row.ladder_slot,
                        "ladder_day": row.ladder_day,
                        "our_action": row.our_action.value,
                        "our_slot": row.our_slot,
                        "our_day": row.our_day,
                        "binding_constraint": row.binding_constraint,
                        "conformal_set": sorted(c.value for c in row.conformal_set),
                        "belief_json": row.belief_json,
                        "decision_sha256": row.decision_sha256,
                        "divergence": row.divergence,
                        "agrees": row.agrees,
                    }
                )
                + "\n"
            )

    print(log.summary())
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.jsonl')}")
    return 0


def _persist(conn, log: DeltaLog, *, run_id: str, batch: Sequence[ShadowInput]) -> None:
    """Write every row to shadow_ledger. Imported here rather than at module
    scope so that importing this module never pulls in the ledger layer for
    a caller that only wants the in-memory log."""
    from src.ledger import store

    profiles = {i.mandate_id: i.profile for i in batch}
    for row in log.rows:
        store.append_shadow(
            conn,
            run_id=run_id,
            mandate_id=row.mandate_id,
            cycle_id=row.cycle_id,
            profile=profiles[row.mandate_id].value,
            ladder_action=row.ladder_action.value,
            ladder_slot=row.ladder_slot,
            ladder_day=row.ladder_day,
            ladder_committed_attempts=row.ladder_committed_attempts,
            our_action=row.our_action.value,
            our_slot=row.our_slot,
            our_day=row.our_day,
            binding_constraint=row.binding_constraint,
            conformal_set=",".join(sorted(c.value for c in row.conformal_set)),
            belief_json=row.belief_json,
            decision_sha256=row.decision_sha256,
            divergence=row.divergence,
            agrees=row.agrees,
        )


if __name__ == "__main__":
    raise SystemExit(main())
