"""R5 (reports/gates.md, "Post-B16 remediation gates"): what does making
the off-ramp REACHABLE cost, and how does that cost move as the evidence
channel that reaches it gets worse?

The gate: `n_offer > 0` in at least one regime, with **both** the recovery
cost and the false-off-ramp rate reported at **every** point of a
channel-quality sweep, and the synthetic channel's own ROC published beside
them. The conformal singleton stays the only firing rule.

The gate's own closing note, which this file exists to honour rather than
to beat: *"R5 does not promise the off-ramp is correct, only that it is
reachable and measured. Untested-and-central is a weaker position than
tested-and-imperfect; this gate buys the second one, not a good result."*

=== EVERYTHING HERE IS SYNTHETIC, AND READS PRIVILEGED GROUND TRUTH ========

`eval/frozen/simulator.py` emits no decline strings and no support tickets.
The channel swept below FABRICATES both, and it does so by reading
`SimMandate.initial_cause` -- the latent true cause the policy itself must
never see -- and feeding the result into the DECISION path.

That is a materially stronger claim than the score-only privileged read
`false_reauth_count` already makes, and it is exactly why this file exists:
the honest way to publish a number produced by a fabricated signal is to
publish the signal's own quality curve beside it, including points where
the signal is worthless. Nothing here is evidence that a real
`payment_cancelled` feed or a real support-ticket feed carries this much
information. It is a sensitivity study on a synthetic channel.

=== PRE-REGISTERED, BEFORE THE FIRST RUN ==================================

Fixed in this docstring before this script had ever been executed, for the
same reason `eval/ltv_sensitivity.py` fixes its slices and `reports/gates.md`
was written before R1a ran: a slice or an operating point chosen after
seeing which one produced the nicest number is not a result.

**The slice.** `baseline / nominal / strict`, all 8 seeds -- the same "easy
arm" slice this project already uses as its primary comparison everywhere
else (`eval/ltv_sensitivity.py`'s HEADLINE_SLICE, and README's headline).

**The grid.** Eight (tpr, fpr) points per channel, spanning worthless to
oracular, listed in QUALITY_GRID below. Two of them sit at AUC 0.500 by
construction -- a channel that carries no information at all -- because a
sweep that only shows good channels proves nothing. One sits at AUC 1.000,
labelled as the ORACLE it is: an upper bound nobody should read as
attainable.

**The operating point: tpr 0.60 / fpr 0.15** (AUC 0.725). Chosen because it
is unambiguously *not* an oracle: it misses two exit-intent customers in
five, and one non-exit customer in seven trips it falsely. Any headline
number that survives at this point survives on a channel this project would
be embarrassed to call good. `eval/run.py` imports OPERATING_POINT from
here rather than restating it, so the published grid and this sweep cannot
drift apart.

**The two channels.** Both ship (DECISIONS.md, 2026-09-04, R0), each with
its own quality sweep and its own ROC:
  decline -- DeclineClass.CUSTOMER_DECLINED, inverted through
             src/classify/cause_map.py's hand-authored table.
  intent  -- a score, mapped through src/execute/intent_channel.py's
             DECLARED operating point, which is INDEPENDENT of this sweep's
             (tpr, fpr) and therefore misspecified at every point except by
             coincidence. That is the realistic case.

The published 1024-cell grid (`reports/regimes.json`) runs the **decline**
channel only, at the operating point above. Folding a fabricated
support-ticket signal into the headline would be a bigger fabrication than
a fabricated decline string -- R0's own reasoning -- so the intent channel
is measured here and nowhere else.

=== WHAT IS REPORTED AT EVERY POINT =======================================

Per (channel, quality point), aggregated over the slice's 8 seeds:
  * the three bars (recovered / attempts spent / mandates preserved), and
    the ladder's, so the RECOVERY COST of reaching the off-ramp is visible
    as a difference rather than asserted;
  * `n_offer`, `offramp_scored_count`, `false_offramp_count` and
    `true_offramp_count` -- and the false-off-ramp RATE, which before R5
    had no denominator at all, WITH a Wilson interval on it. `n_offer` runs
    as low as 8 at the least informative points; a bare rate there would be
    a number plus or minus thirty printed as if it were a result;
  * the channel's REALISED ROC point (measured from the draws that actually
    happened, never a restatement of the configured rates), its AUC, and a
    MANDATE-level cluster-bootstrap CI on that AUC -- draws are clustered,
    one mandate contributing up to four decision points, so a row-level
    interval would overstate precision (the convention
    `bench/llm_vs_stats.py` already established; its `macro_ovr_auc` and
    `cluster_bootstrap_ci` are imported here rather than reimplemented);
  * the conformal gate's re-calibration diagnostics AT THAT POINT. The
    channel changes the belief distribution, so it changes the calibration
    pool; coverage is re-measured, and degradation is a result, not a bug.

Run:    python -m eval.offramp_channel
Writes: reports/offramp_channel.json. `eval/report.py` READS it (if
present) and renders `reports/regimes.md`'s off-ramp section from it --
this script computes, report.py renders, the same discipline the rest of
the pipeline already follows.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np

from bench.llm_vs_stats import cluster_bootstrap_ci, macro_ovr_auc, wilson_ci
from eval import regimes as regimes_mod
from eval.allocator_sweep import fit_nominal_hazard_model, hazard_from_fit
from eval.frozen.simulator import load_config
from eval.run import (
    ALL_PROFILES, fit_gate, make_channel, run_engine_cell, run_ladder_cell,
)
from src.core.clock import now as clock_now
from src.core.types import Profile
from src.policy.costs import load as load_costs

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = _REPO_ROOT / "reports" / "offramp_channel.json"

HEADLINE_SLICE: tuple[str, str, Profile] = ("baseline", "nominal", Profile.strict)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

CHANNEL_KINDS: tuple[str, ...] = ("decline", "intent")

# (tpr, fpr), worst first. Pre-registered -- see the module docstring.
# Nominal AUC of the two-point ROC through (fpr, tpr) is (1 + tpr - fpr)/2.
QUALITY_GRID: tuple[tuple[float, float], ...] = (
    (0.30, 0.30),   # AUC 0.500 -- no information, and rarely fires
    (0.50, 0.50),   # AUC 0.500 -- no information, and fires constantly
    (0.40, 0.30),   # AUC 0.550 -- barely informative
    (0.50, 0.25),   # AUC 0.625
    (0.60, 0.15),   # AUC 0.725 -- THE OPERATING POINT
    (0.75, 0.10),   # AUC 0.825
    (0.90, 0.05),   # AUC 0.925
    (1.00, 0.00),   # AUC 1.000 -- ORACLE. An upper bound, not a target.
)

# Imported by eval/run.py. Never restate these numbers there.
OPERATING_POINT: tuple[float, float] = (0.60, 0.15)

assert OPERATING_POINT in QUALITY_GRID, (
    "the operating point must be a point the sweep actually measures"
)


def _channel_auc(log: list[tuple[str, bool, bool]]) -> dict[str, Any]:
    """The channel's REALISED ROC, from the draws that actually happened.

    `log` rows are (mandate_id, is_wont_pay, fired). The emitter is binary,
    so the ROC is the two-point curve through the realised (fpr, tpr) and
    the AUC is computed on exactly that -- but through
    `bench.llm_vs_stats.macro_ovr_auc`, not a hand-rolled formula, because
    that implementation is tie-aware (sklearn integrates ties as a diagonal
    segment rather than a step) and this predictor is ALL ties: two distinct
    values across thousands of rows. A step-function AUC here would be
    wrong in exactly the way that module's own docstring warns about.

    The CI resamples MANDATES, not rows.
    """
    if not log:
        return {"n": 0, "auc": None, "auc_ci": None,
                "tpr_realised": None, "fpr_realised": None}
    groups = [row[0] for row in log]
    y_true = np.asarray([int(row[1]) for row in log])
    fired = np.asarray([float(row[2]) for row in log])
    p = np.stack([1.0 - fired, fired], axis=1)

    n_wp = int(y_true.sum())
    n_other = int(len(y_true) - n_wp)
    try:
        auc = macro_ovr_auc(y_true, p)
        ci = cluster_bootstrap_ci(y_true, p, groups)
    except ValueError:
        # A slice with no WONT_PAY mandate at all (or only those): an AUC
        # over one class is not a number. Reported as absent, never as 0.5.
        auc, ci = None, None
    return {
        "n": len(log),
        "n_wont_pay": n_wp,
        "n_other": n_other,
        "tpr_realised": (float(fired[y_true == 1].mean()) if n_wp else None),
        "fpr_realised": (float(fired[y_true == 0].mean()) if n_other else None),
        "auc": auc,
        "auc_ci": list(ci) if ci else None,
    }


def _sum(cells, attr: str) -> int:
    return sum(getattr(c, attr) for c in cells)


def _repeat_rate(log: list[tuple[str, bool, bool]]) -> dict[str, Any]:
    """Among mandates whose TRUE cause is NOT WONT_PAY, how often did the
    channel fire on that mandate TWO OR MORE times?

    This is the number stats-reviewer's HIGH finding (2026-09-05) argues
    the main quality grid never measures: it holds within-mandate
    dependence fixed at zero (`fires()` draws an independent Bernoulli each
    call), while the singleton off-ramp rule needs roughly two coincident
    false firings to open (one CUSTOMER_DECLINED moves belief to ~0.62
    WONT_PAY; the fitted gate's own singleton boundary sits around
    0.80-0.90). A channel that concentrates its false-firing mass into a
    minority of "habitual dismissers" can hold the exact same marginal
    (tpr, fpr) -- the exact same row in the main grid -- while this number
    moves by multiples. See `WontPayChannel.habitual_fraction` and
    `dependence_sweep()` below.
    """
    from collections import defaultdict

    by_mandate: dict[str, list[bool]] = defaultdict(list)
    for mandate_id, is_wont_pay, fired in log:
        if not is_wont_pay:
            by_mandate[mandate_id].append(fired)
    n = len(by_mandate)
    two_plus = sum(1 for draws in by_mandate.values() if sum(draws) >= 2)
    return {
        "n_non_wont_pay_mandates": n,
        "n_with_two_plus_false_fires": two_plus,
        "rate": (two_plus / n) if n else None,
    }


def _point(kind: str, tpr: float, fpr: float, *, base_cfg, costs, hazard,
           ladder_cells, seeds: tuple[int, ...],
           habitual_fraction: float = 1.0) -> dict[str, Any]:
    """One (channel, quality) point across the slice's seeds.

    The gate is REFIT at every point. Re-calibration is not optional: the
    channel changes the belief distribution, so it changes the calibration
    pool, and reusing a pool drawn under a different channel would break the
    exchangeability split conformal's coverage guarantee rests on.

    `habitual_fraction` defaults to 1.0 -- WontPayChannel's own exactly-iid
    default -- so every call from the main QUALITY_GRID sweep is completely
    unaffected; only `dependence_sweep()` below passes a value below 1.0.
    """
    regime, arm, profile = HEADLINE_SLICE
    cfg = regimes_mod.config_for(regime, base_cfg)
    spec = (kind, tpr, fpr, habitual_fraction)

    gate, gate_kind, gate_diag = fit_gate(base_cfg, channel_spec=spec)

    engine_cells, log = [], []
    for seed in seeds:
        channel = make_channel(spec, seed)
        engine_cells.append(run_engine_cell(
            regime, arm, profile, cfg, seed,
            hazard=hazard, costs=costs, gate=gate, gate_kind=gate_kind,
            channel=channel,
        ))
        # R5 review pass, 2026-09-05 (stats-reviewer): namespaced by seed,
        # not the bare mandate_id `channel.log` stores. The frozen
        # simulator reuses "M0000".."M0199" every seed, so 8 seeds'
        # unqualified logs concatenate into 200 clusters instead of 1,600 --
        # `_channel_auc`'s cluster-bootstrap CI was silently resampling
        # SEEDS, not mandates. The same convention `_calib_group_id`
        # (eval/run.py) already uses for the identical reason, applied here
        # rather than a second one invented. Confirmed numerically harmless
        # on today's data (the channel's draws are conditionally
        # independent given cause, by construction) but the bug is real:
        # a channel whose draws ever gained genuine seed-correlated
        # structure would have had that hidden by this collision.
        log.extend((f"s{seed}:{mid}", is_wp, fired) for mid, is_wp, fired in channel.log)

    n_offer = _sum(engine_cells, "n_offer")
    scored = _sum(engine_cells, "offramp_scored_count")
    false_n = _sum(engine_cells, "false_offramp_count")
    true_n = _sum(engine_cells, "true_offramp_count")
    coverages = [c.coverage_marginal for c in engine_cells if c.coverage_marginal is not None]
    singleton_wp = [c.singleton_wont_pay_rate for c in engine_cells
                    if c.singleton_wont_pay_rate is not None]

    return {
        "channel_kind": kind,
        "tpr": tpr,
        "fpr": fpr,
        "nominal_auc": (1.0 + tpr - fpr) / 2.0,
        "is_operating_point": (tpr, fpr) == OPERATING_POINT,
        "seeds": list(seeds),
        "gate_kind": gate_kind,
        "gate_diagnostics": gate_diag,
        # the three bars, engine and ladder, summed over the slice's seeds
        "engine_recovered_paise": _sum(engine_cells, "recovered_paise"),
        "engine_attempts_spent": _sum(engine_cells, "attempts_spent"),
        "engine_mandates_preserved": _sum(engine_cells, "mandates_preserved"),
        "ladder_recovered_paise": _sum(ladder_cells, "recovered_paise"),
        "ladder_attempts_spent": _sum(ladder_cells, "attempts_spent"),
        "ladder_mandates_preserved": _sum(ladder_cells, "mandates_preserved"),
        "billable_paise": _sum(engine_cells, "billable_paise"),
        "n_mandates": _sum(engine_cells, "n_mandates"),
        # what the policy chose
        "n_attempt": _sum(engine_cells, "n_attempt"),
        "n_offer": n_offer,
        "n_reauth": _sum(engine_cells, "n_reauth"),
        "n_stop": _sum(engine_cells, "n_stop"),
        # BOTH error costs, at every point -- the gate's own wording
        "missed_recovery_count": _sum(engine_cells, "missed_recovery_count"),
        "missed_recovery_paise": _sum(engine_cells, "missed_recovery_paise"),
        "offramp_scored_count": scored,
        "false_offramp_count": false_n,
        "false_offramp_paise": _sum(engine_cells, "false_offramp_paise"),
        "true_offramp_count": true_n,
        "true_offramp_paise": _sum(engine_cells, "true_offramp_paise"),
        "false_offramp_rate": (false_n / scored) if scored else None,
        # An INTERVAL, not just the point. n_offer runs as low as 8 at the
        # least informative points, where a rate printed to one decimal
        # place is a number plus or minus thirty. That is precisely the
        # failure bench/llm_vs_stats.py's own wilson_ci() docstring warns
        # about ("three decimals of a number that is plus or minus six
        # points"), so its implementation is imported rather than a third
        # one written. Wilson rather than the normal approximation because
        # it stays inside [0, 1] and behaves at the boundaries -- and 0.0
        # is exactly the value this table would most easily overclaim.
        #
        # It still IGNORES within-mandate clustering: each OFFER here is one
        # mandate, so the rate itself is unclustered, but the eight seeds
        # are pooled by summing, which treats a seed as exchangeable with a
        # mandate. Treat the width as a FLOOR on the true uncertainty, the
        # same caveat that docstring already carries.
        "false_offramp_rate_ci": (list(wilson_ci(false_n, scored)) if scored else None),
        # the channel's own ROC, measured
        "channel_roc": _channel_auc(log),
        # conformal behaviour after re-calibration at THIS point
        "coverage_marginal_mean": (sum(coverages) / len(coverages)) if coverages else None,
        "singleton_wont_pay_rate_mean": (
            sum(singleton_wp) / len(singleton_wp) if singleton_wp else None
        ),
        "coverage_n": _sum(engine_cells, "coverage_n"),
        # 1.0 on every row of the main QUALITY_GRID -- see dependence_sweep()
        # for the rows where this varies.
        "habitual_fraction": habitual_fraction,
        "repeat_false_fire": _repeat_rate(log),
    }


# R5 REVIEW PASS, 2026-09-05 (stats-reviewer, HIGH; see WontPayChannel's
# own docstring for the mechanism). At the pre-registered operating point's
# fpr (0.15), all four values are >= fpr, so `_effective_fpr()`'s marginal
# guarantee holds EXACTLY at every row -- `repeat_false_fire`'s rate is the
# only thing that should move, and does. 1.0 is the row the main
# QUALITY_GRID already measures, repeated here so the four-point curve
# reads as one continuous sweep rather than three new numbers plus an
# implicit fourth.
DEPENDENCE_GRID: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15)


def dependence_sweep(*, grid: tuple[float, ...] = DEPENDENCE_GRID,
                     seeds: tuple[int, ...] = SEEDS,
                     verbose: bool = True) -> dict[str, Any]:
    """How the false-off-ramp rate moves as within-mandate CORRELATION
    increases, holding the marginal (tpr, fpr) FIXED at the pre-registered
    operating point.

    The main QUALITY_GRID answers "what happens as the channel's
    discrimination changes" and holds correlation at exactly zero. This
    answers the question that grid cannot: `should_act()` needs roughly two
    coincident false firings on one mandate to open the off-ramp (see
    WontPayChannel's docstring), and two independent draws from one
    customer's decline history is not a safe assumption. Every point here
    has the IDENTICAL marginal (tpr, fpr) as the operating-point row of the
    main grid -- only `habitual_fraction` moves -- so a difference in the
    false-off-ramp rate is attributable to correlation, not to a
    confounded discrimination change.

    `grid`/`seeds` are overridable for the same reason `sweep()`'s are.
    """
    base_cfg = load_config()
    costs = load_costs()
    tpr, fpr = OPERATING_POINT

    if verbose:
        print("fitting hazard model on the nominal corpus...", file=sys.stderr)
    hazard = hazard_from_fit(fit_nominal_hazard_model())

    regime, arm, profile = HEADLINE_SLICE
    cfg = regimes_mod.config_for(regime, base_cfg)
    ladder_cells = [run_ladder_cell(regime, arm, profile, cfg, s) for s in seeds]

    points: list[dict[str, Any]] = []
    for hf in grid:
        if verbose:
            print(f"  decline  tpr={tpr:.2f} fpr={fpr:.2f} habitual_fraction={hf:.2f} ...",
                  file=sys.stderr)
        pt = _point("decline", tpr, fpr, base_cfg=base_cfg, costs=costs,
                    hazard=hazard, ladder_cells=ladder_cells, seeds=seeds,
                    habitual_fraction=hf)
        if verbose:
            rr = pt["repeat_false_fire"]
            print(f"      n_offer={pt['n_offer']:>4d}  false_offramp={pt['false_offramp_count']:>4d}  "
                  f"rate={pt['false_offramp_rate']}  "
                  f"repeat_false_fire_rate={rr['rate']}  "
                  f"fpr_realised={pt['channel_roc']['fpr_realised']}", file=sys.stderr)
        points.append(pt)

    return {
        "schema": 1,
        "generated": clock_now().isoformat(),
        "slice": {"regime": regime, "arm": arm, "profile": profile.value,
                  "seeds": list(seeds)},
        "operating_point": {"tpr": tpr, "fpr": fpr},
        "dependence_grid": list(grid),
        "synthetic": True,
        "disclosure": (
            "SYNTHETIC, and additionally: the within-mandate correlation "
            "swept here is itself an assumption, not a measurement -- there "
            "is no real decline-string corpus this project has access to "
            "that could calibrate it. This sweep establishes SENSITIVITY "
            "(the false-off-ramp rate is not robust to an assumption the "
            "main grid holds fixed at zero), not a corrected estimate."
        ),
        "points": points,
    }


def sweep(*, grid: tuple[tuple[float, float], ...] = QUALITY_GRID,
          kinds: tuple[str, ...] = CHANNEL_KINDS,
          seeds: tuple[int, ...] = SEEDS,
          verbose: bool = True) -> dict[str, Any]:
    """`grid`, `kinds` and `seeds` default to the pre-registered sweep but
    are overridable -- tests pass a handful of points so the suite does not
    pay for a 16-point x 8-seed sweep to exercise this function's logic
    (the same convention eval/ltv_sensitivity.py's `grid=` kwarg uses)."""
    base_cfg = load_config()
    costs = load_costs()

    if verbose:
        print("fitting hazard model on the nominal corpus...", file=sys.stderr)
    hazard = hazard_from_fit(fit_nominal_hazard_model())

    regime, arm, profile = HEADLINE_SLICE
    cfg = regimes_mod.config_for(regime, base_cfg)
    # The ladder is channel-blind by construction -- it has no belief and no
    # gate -- so it is run ONCE per seed and reused at every point rather
    # than re-run 16 times to produce the identical number.
    ladder_cells = [run_ladder_cell(regime, arm, profile, cfg, s) for s in seeds]

    points: list[dict[str, Any]] = []
    for kind in kinds:
        for tpr, fpr in grid:
            if verbose:
                print(f"  {kind:8s} tpr={tpr:.2f} fpr={fpr:.2f} ...", file=sys.stderr)
            pt = _point(kind, tpr, fpr, base_cfg=base_cfg, costs=costs,
                        hazard=hazard, ladder_cells=ladder_cells, seeds=seeds)
            if verbose:
                print(f"      n_offer={pt['n_offer']:>4d}  "
                      f"false_offramp={pt['false_offramp_count']:>4d}  "
                      f"rate={pt['false_offramp_rate']}  "
                      f"auc={pt['channel_roc']['auc']}", file=sys.stderr)
            points.append(pt)

    return {
        "schema": 1,
        "generated": clock_now().isoformat(),
        "slice": {"regime": regime, "arm": arm, "profile": profile.value,
                  "seeds": list(seeds)},
        "operating_point": {"tpr": OPERATING_POINT[0], "fpr": OPERATING_POINT[1]},
        "quality_grid": [list(g) for g in grid],
        "channel_kinds": list(kinds),
        "synthetic": True,
        "disclosure": (
            "Every channel measured here is SYNTHETIC and reads the "
            "simulator's privileged true cause. These numbers describe how "
            "the off-ramp behaves GIVEN a channel of stated quality; they "
            "are not evidence that any real signal has that quality."
        ),
        "points": points,
    }


def main() -> int:
    payload = sweep()
    print("sweeping within-mandate dependence at the operating point...", file=sys.stderr)
    payload["dependence_sweep"] = dependence_sweep()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JSON} ({len(payload['points'])} points)", file=sys.stderr)
    fired = [p for p in payload["points"] if p["n_offer"] > 0]
    print(f"points with n_offer > 0: {len(fired)}/{len(payload['points'])}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
