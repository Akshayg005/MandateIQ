"""B13 batch driver: every arm x every regime x both compliance profiles.

One command produces every number in the report:

    python -m eval.run --all-regimes --both-profiles

It writes a single machine-readable artifact, `reports/regimes.json`, which
`eval/report.py` then renders. Nothing in the report is computed anywhere
else -- if a number appears in `reports/` it came out of this file, so
"reproducible by one command" is a property of the pipeline rather than a
claim in prose.

Per PLAN_DETAIL.md's B13 row this module must NOT print per-mandate logs to
stdout. It prints one progress line per cell and a final summary; the
per-mandate detail goes to the JSON artifact.

Three things here are deliberate and easy to get wrong:

1. **The hazard model is fit on the NOMINAL corpus, once, and reused under
   every regime.** A regime is a shift the deployed system did not see
   coming; refitting per regime would measure a model that already knew
   about the shift, which is not the question. The engine is therefore
   MISSPECIFIED under every regime except baseline, on purpose.

2. **The conformal gate is calibrated once, on baseline**, from a separate
   calibration draw with its own seed, and the SAME fitted gate is used
   under every regime. Split conformal's 95% coverage guarantee holds under
   exchangeability; a regime breaks exchangeability by construction. Coverage
   is therefore MEASURED per regime rather than assumed, and degradation is a
   result, not a bug. `reports/regimes.json` records which gate was live for
   every cell so report.py can print the coverage claim only where the real
   gate actually ran.

3. **Error costs are exact per-mandate counterfactuals, not estimates.**
   When the engine stops early we deepcopy the simulator at that exact
   moment -- RNG state included -- and keep grinding on the copy. That
   answers "would this mandate have paid if we had not stopped?" from the
   same random draws the real run would have seen, rather than from a
   re-seeded rerun that answers a different question.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from eval import regimes as regimes_mod
from eval.allocator_sweep import (
    PROXY_SOURCE_VERSION,
    _proxy_decline_class,
    fit_nominal_hazard_model,
    hazard_from_fit,
    initial_belief,
)
from eval.frozen.scoring import MandateResult, aggregate, score_mandate
from eval.frozen.simulator import Simulator, load_config
from src.core.types import Action, Cause, MandateState, Outcome, Profile
from src.model import conformal
from src.policy import belief as belief_mod
from src.policy.allocator import AllocationContext, AllocatorError, solve
from src.policy.constraints import MAX_ATTEMPTS, afa_free_limit_paise
from src.policy.costs import PolicyCosts, load as load_costs
from src.policy.gate import ConformalCauseGate, FullSetGate

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "regimes.json"

ALL_ARMS: tuple[str, ...] = ("nominal", "misspecified", "coupled")
ALL_PROFILES: tuple[Profile, ...] = (Profile.strict, Profile.permissive)

# Distinct RNG offsets, so adding a stream can never perturb an existing one.
# 500_000 is allocator_sweep's slot-1 stream; these must not collide with it.
_SLOT1_OFFSET = 700_000
_CALIB_SLOT1_OFFSET = 900_000
# The calibration draw's own seed. Disjoint from any reported seed, and its
# mandate ids are namespaced (see _calib_group_id) so conformal's own
# assert_disjoint can actually check the split rather than trust it.
CALIB_SEED = 424_242

CAUSE_ORDER: tuple[Cause, ...] = tuple(belief_mod.CAUSE_ORDER)


# --- results -----------------------------------------------------------------


@dataclass
class CellResult:
    """One (regime, arm, profile, policy) cell -- the three bars plus
    everything needed to say what the policy actually did and what it cost."""

    regime: str
    arm: str
    profile: str
    policy: str
    seed: int
    gate_kind: str

    n_mandates: int = 0
    # the three bars
    recovered_paise: int = 0
    attempts_spent: int = 0
    mandates_preserved: int = 0
    # outcome breakdown
    recovered: int = 0
    dead: int = 0
    opted_out: int = 0
    censored: int = 0
    iatrogenic_failures: int = 0
    # what the policy chose
    n_attempt: int = 0
    n_offer: int = 0
    n_reauth: int = 0
    n_stop: int = 0
    n_above_afa: int = 0
    # the two error costs (protocol.md: reported alongside, never folded in)
    missed_recovery_count: int = 0
    missed_recovery_paise: int = 0
    false_offramp_count: int = 0
    false_offramp_paise: int = 0
    # gate evidence, engine only
    coverage_marginal: float | None = None
    coverage_n: int = 0
    singleton_wont_pay_rate: float | None = None
    mean_set_size: float | None = None
    violations: list[str] = field(default_factory=list)
    seconds: float = 0.0


# --- the engine policy -------------------------------------------------------


def _initial_context(m, profile: Profile, costs: PolicyCosts) -> AllocationContext:
    return AllocationContext(
        mandate_id=m.mandate_id,
        cycle_id=m.cycle_id,
        profile=profile,
        amount_paise=m.amount_paise,
        ceiling_paise=m.ceiling_paise,
        category=m.category,
        plan_day=1,
        attempts_used=1,
        committed_days=(1,),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def _counterfactual_recovers(sim: Simulator, mandate_id: str, from_slot: int,
                             last_day: int) -> bool:
    """Would this mandate have recovered if we had kept grinding?

    Called with `sim` ALREADY deepcopied by the caller, at the exact moment
    the engine stopped -- so the RNG state, the household balance, and the
    effective cause are the ones the real run would have carried forward.
    The counterfactual policy is the incumbent's: spend every remaining slot,
    on the tightest legal cadence (the day after the last attempt), until a
    terminal outcome or the NPCI budget runs out.

    Ladder day-offsets are NOT reused here: they are absolute days (1/2/3)
    and the engine may already have attempted past them, which the frozen
    simulator rejects as out-of-order. Consecutive days from where we stopped
    is the same question -- "keep trying" -- expressed legally.
    """
    day = last_day
    for slot in range(from_slot, MAX_ATTEMPTS + 1):
        day += 1
        result = sim.attempt(mandate_id, slot=slot, on_day=day)
        if result.outcome == Outcome.RECOVERED:
            return True
        if result.outcome in (Outcome.DEAD, Outcome.OPTED_OUT):
            return False
    return False


def _run_engine_mandate(m, sim: Simulator, profile: Profile, hazard,
                        costs: PolicyCosts, gate, b, cell: CellResult):
    """Drive one mandate through the allocator. Returns the ordered attempts
    actually made; mutates `cell`'s action counters and error costs.

    Adapted from eval/allocator_sweep.py's _run_one_mandate, which answers a
    different question (B8's gate criteria: did we attempt, how often) and so
    throws away the AttemptResults the three bars are computed from.
    """
    ctx = _initial_context(m, profile, costs)
    attempts = []
    stopped_action: Action | None = None
    last_day = 1

    while ctx.attempts_used < MAX_ATTEMPTS:
        try:
            plan = solve(b, ctx, hazard=hazard, costs=costs, gate=gate)
        except AllocatorError as exc:
            cell.violations.append(f"{m.mandate_id}: AllocatorError: {exc}")
            stopped_action = Action.STOP
            break

        if plan.chosen_action != Action.ATTEMPT:
            stopped_action = plan.chosen_action
            break

        committed = plan.committed[0]
        if committed.amount_paise > ctx.ceiling_paise:
            cell.violations.append(
                f"{m.mandate_id}: committed {committed.amount_paise} over "
                f"ceiling {ctx.ceiling_paise}"
            )
        result = sim.attempt(m.mandate_id, slot=committed.slot, on_day=committed.on_day)
        attempts.append(result)
        last_day = committed.on_day
        ctx = ctx.with_attempt(committed.on_day)
        cell.n_attempt += 1

        dc = _proxy_decline_class(result.outcome)
        if result.outcome != Outcome.STILL_PENDING:
            # Terminal. The ATTEMPT sequence is over, but the DECISION
            # sequence is not: a dead instrument is exactly when REAUTH is
            # the right next action (CLAUDE.md's own cause->action table).
            if dc is not None:
                b = belief_mod.update(b, dc, source_version=PROXY_SOURCE_VERSION)
                try:
                    final = solve(b, ctx, hazard=hazard, costs=costs, gate=gate)
                    stopped_action = final.chosen_action
                except AllocatorError as exc:
                    cell.violations.append(f"{m.mandate_id}: final AllocatorError: {exc}")
            break

        if dc is not None:
            b = belief_mod.update(b, dc, source_version=PROXY_SOURCE_VERSION)

    if stopped_action == Action.OFFER:
        cell.n_offer += 1
    elif stopped_action == Action.REAUTH:
        cell.n_reauth += 1
    elif stopped_action == Action.STOP:
        cell.n_stop += 1

    # -- error costs, by exact counterfactual --------------------------------
    resolved = bool(attempts) and attempts[-1].outcome != Outcome.STILL_PENDING
    slots_left = MAX_ATTEMPTS - (1 + len(attempts))
    if not resolved and slots_left > 0:
        shadow = copy.deepcopy(sim)
        would_pay = _counterfactual_recovers(
            shadow, m.mandate_id, from_slot=2 + len(attempts), last_day=last_day
        )
        if would_pay:
            cell.missed_recovery_count += 1
            cell.missed_recovery_paise += m.amount_paise
            if stopped_action == Action.OFFER:
                cell.false_offramp_count += 1
                cell.false_offramp_paise += m.amount_paise

    return attempts


def _result_for(m, attempts) -> MandateResult:
    """score_mandate() raises on a zero-attempt mandate -- reasonably, since
    the ladder can never produce one. This engine can: REAUTH or OFFER at the
    first decision point spends no slot at all, which is the entire point of
    having those actions. Such a mandate is STILL_PENDING and PRESERVED under
    protocol.md's own definitions (budget unspent, right-censored, still an
    active mandate next cycle), so it is scored here rather than by editing
    the frozen scorer.
    """
    if attempts:
        return score_mandate(m, list(attempts))
    return MandateResult(
        mandate_id=m.mandate_id,
        attempts=(),
        final_outcome=Outcome.STILL_PENDING,
        amount_recovered_paise=0,
        preserved=True,
        iatrogenic_failures=0,
    )


# --- the conformal gate ------------------------------------------------------


def _calib_group_id(mandate_id: str) -> str:
    """Simulator mandate ids (M0000...) repeat across seeds, so a bare id
    cannot prove the calibration and report sets are disjoint. Namespacing by
    the calibration seed gives conformal.assert_disjoint() something real to
    check -- the same convention eval/corpus.py already uses."""
    return f"calib{CALIB_SEED}:{mandate_id}"


def fit_gate(base_cfg: dict, *, alpha: float = 0.05):
    """Calibrate the off-ramp gate ONCE, on the baseline regime, from its own
    simulator draw. Returns (gate, kind, diagnostics).

    The predictor is over CAUSES, not terminal Outcomes: the off-ramp asks
    why the mandate is failing, and allocator.py fires only on the singleton
    {WONT_PAY}. Scores are LAC over the belief the system actually holds
    after the slot-1 decline -- i.e. the gate is calibrated on exactly the
    object it will be asked about in production, not on a proxy.

    Falls back to FullSetGate (never offers) if calibration is underpowered,
    and says so. That is the safe direction and B8's documented default.
    """
    sim = Simulator("nominal", seed=CALIB_SEED, config=base_cfg)
    rng = random.Random(CALIB_SEED + _CALIB_SLOT1_OFFSET)
    scores, y, ids = [], [], []
    for m in sim.mandates:
        b = initial_belief(m.initial_cause, base_cfg, rng)
        scores.append(list(b.probs))
        y.append(CAUSE_ORDER.index(m.initial_cause))
        ids.append(_calib_group_id(m.mandate_id))

    score_rows = conformal.lac_scores(np.asarray(scores, dtype=float))
    try:
        predictor = conformal.calibrate(
            scores=score_rows,
            y=np.asarray(y, dtype=int),
            labels=CAUSE_ORDER,
            row_group_ids=ids,
            provenance="calib_conf",
            alpha=alpha,
        )
    except conformal.ConformalUnderpowered as exc:
        return FullSetGate(), "full_set", {"reason": f"underpowered: {exc}"}

    return (
        ConformalCauseGate(predictor),
        "conformal",
        {"alpha": alpha, "n_calib": len(y), "calib_seed": CALIB_SEED},
    )


def _measure_coverage(gate, sim: Simulator, cfg: dict, seed: int, cell: CellResult) -> None:
    """Empirical coverage of the live gate on THIS cell's batch.

    Uses SimMandate.initial_cause, which is privileged ground truth the
    policy itself must never read (simulator.py's own warning). It is read
    here for the same reason eval/gate_criteria.py reads it: to score, not to
    decide. The belief scored is the one the gate was actually asked about --
    the post-slot-1 belief -- reconstructed from the same RNG stream the run
    used, so this measures the deployed gate rather than a fresh one.
    """
    if not isinstance(gate, ConformalCauseGate):
        return
    rng = random.Random(seed + _SLOT1_OFFSET)
    covered = singleton_wp = 0
    sizes = []
    for m in sim.mandates:
        b = initial_belief(m.initial_cause, cfg, rng)
        s = gate.pred_set(b)
        sizes.append(len(s))
        if m.initial_cause in s:
            covered += 1
        if s == frozenset({Cause.WONT_PAY}):
            singleton_wp += 1
    n = len(sizes)
    cell.coverage_n = n
    cell.coverage_marginal = covered / n if n else None
    cell.singleton_wont_pay_rate = singleton_wp / n if n else None
    cell.mean_set_size = sum(sizes) / n if n else None


# --- cells -------------------------------------------------------------------


def _fill_bars(cell: CellResult, batch) -> None:
    cell.n_mandates = batch.n_mandates
    cell.recovered_paise = batch.total_recovered_paise
    cell.attempts_spent = batch.total_attempts_spent
    cell.mandates_preserved = batch.mandates_preserved
    cell.recovered = batch.mandates_recovered
    cell.dead = batch.mandates_dead
    cell.opted_out = batch.mandates_opted_out
    cell.censored = batch.mandates_censored
    cell.iatrogenic_failures = batch.total_iatrogenic_failures


def run_ladder_cell(regime: str, arm: str, profile: Profile, cfg: dict,
                    seed: int) -> CellResult:
    from eval import baseline_ladder

    t0 = time.perf_counter()
    cell = CellResult(regime=regime, arm=arm, profile=profile.value,
                      policy="ladder", seed=seed, gate_kind="n/a")
    sim = Simulator(arm, seed=seed, config=cfg)
    batch = baseline_ladder.run(sim, profile)
    _fill_bars(cell, batch)
    cell.n_attempt = batch.total_attempts_spent
    cell.n_above_afa = sum(
        1 for m in sim.mandates if m.amount_paise > afa_free_limit_paise(m.category)
    )
    cell.seconds = time.perf_counter() - t0
    return cell


def run_engine_cell(regime: str, arm: str, profile: Profile, cfg: dict, seed: int,
                    hazard, costs: PolicyCosts, gate, gate_kind: str) -> CellResult:
    t0 = time.perf_counter()
    cell = CellResult(regime=regime, arm=arm, profile=profile.value,
                      policy="engine", seed=seed, gate_kind=gate_kind)
    sim = Simulator(arm, seed=seed, config=cfg)
    slot1_rng = random.Random(seed + _SLOT1_OFFSET)

    results = []
    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            cell.n_above_afa += 1
        b0 = initial_belief(m.initial_cause, cfg, slot1_rng)
        attempts = _run_engine_mandate(m, sim, profile, hazard, costs, gate, b0, cell)
        results.append(_result_for(m, attempts))

    _fill_bars(cell, aggregate(results, arm=arm, profile=profile.value))
    _measure_coverage(gate, Simulator(arm, seed=seed, config=cfg), cfg, seed, cell)
    cell.seconds = time.perf_counter() - t0
    return cell


# --- driver ------------------------------------------------------------------


def run_all(*, regime_names: Sequence[str], arms: Sequence[str],
            profiles: Sequence[Profile], seed: int,
            verbose: bool = True,
            config_path: pathlib.Path | None = None) -> dict[str, Any]:
    base_cfg = load_config(config_path)
    costs = load_costs()

    if verbose:
        print("fitting hazard model on the nominal corpus (once, reused everywhere)...",
              file=sys.stderr)
    hazard = hazard_from_fit(fit_nominal_hazard_model())

    if verbose:
        print("calibrating the conformal off-ramp gate on baseline...", file=sys.stderr)
    gate, gate_kind, gate_diag = fit_gate(base_cfg)
    if verbose:
        print(f"  gate: {gate_kind} {gate_diag}", file=sys.stderr)

    cells: list[CellResult] = []
    for regime in regime_names:
        cfg = regimes_mod.config_for(regime, base_cfg)
        for arm in regimes_mod.arms_for(regime, tuple(arms)):
            for profile in profiles:
                cells.append(run_ladder_cell(regime, arm, profile, cfg, seed))
                cells.append(run_engine_cell(regime, arm, profile, cfg, seed,
                                             hazard, costs, gate, gate_kind))
                if verbose:
                    lad, eng = cells[-2], cells[-1]
                    print(
                        f"  {regime:16s} {arm:13s} {profile.value:11s} "
                        f"ladder[rec={lad.recovered_paise:>10d} att={lad.attempts_spent:>4d} "
                        f"pres={lad.mandates_preserved:>3d}]  "
                        f"engine[rec={eng.recovered_paise:>10d} att={eng.attempts_spent:>4d} "
                        f"pres={eng.mandates_preserved:>3d}]",
                        file=sys.stderr,
                    )

    return {
        "schema": 1,
        "seed": seed,
        "gate_kind": gate_kind,
        "gate_diagnostics": gate_diag,
        "arms": list(arms),
        "profiles": [p.value for p in profiles],
        "regimes": {
            name: {
                "story": spec.story,
                "hypothesis": spec.hypothesis,
                "approximation": spec.approximation,
                "overlay": spec.overlay,
            }
            for name, spec in regimes_mod.REGIMES.items()
            if name in regime_names
        },
        "cells": [asdict(c) for c in cells],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all-regimes", action="store_true",
                    help="every regime in eval/regimes.py (the reported configuration)")
    ap.add_argument("--regime", action="append", dest="regime_names", default=None)
    ap.add_argument("--both-profiles", action="store_true",
                    help="strict and permissive (the reported configuration)")
    ap.add_argument("--arm", action="append", dest="arms", default=None)
    ap.add_argument("--profile", action="append", dest="profile_names", default=None,
                    choices=[p.value for p in ALL_PROFILES])
    ap.add_argument("--config", type=pathlib.Path, default=None,
                    help="the frozen sim config to overlay regimes onto; "
                         "defaults to eval/frozen/sim_config.yaml. Accepted so "
                         "the run command names its own input explicitly.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=ARTIFACT)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    regime_names = (
        list(regimes_mod.REGIMES) if args.all_regimes
        else (args.regime_names or ["baseline"])
    )
    arms = tuple(args.arms or ALL_ARMS)
    if args.both_profiles:
        profiles = ALL_PROFILES
    elif args.profile_names:
        profiles = tuple(Profile(p) for p in args.profile_names)
    else:
        profiles = (Profile.strict,)

    payload = run_all(regime_names=regime_names, arms=arms, profiles=profiles,
                      seed=args.seed, verbose=not args.quiet,
                      config_path=args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        shown = args.out.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        shown = args.out
    print(f"wrote {shown} "
          f"({len(payload['cells'])} cells, gate={payload['gate_kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
