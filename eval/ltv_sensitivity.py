"""R3 (reports/gates.md, "Post-B16 remediation gates"): how does the
engine's raw recovered money change as the assumed mandate LTV
(`config/policy_costs.yaml`'s `mandate_ltv_paise`) varies, and at what LTV
-- if any -- does it match the incumbent ladder's recovered money?

Two slices, both fixed before this script's first run, never chosen after
seeing a result:

1. HEADLINE -- baseline/nominal/strict/seed=0, the same "easy arm" slice
   this project already uses as its primary comparison everywhere else.
   Answers the central question directly for the configuration everyone
   already reads as canonical.

2. WORKED EXAMPLE -- the first (regime, arm, profile, seed) cell, in
   orderings that ALREADY EXIST in this codebase (`eval.regimes.REGIMES`'s
   own dict order, `eval.run.ALL_ARMS`, `eval.run.ALL_PROFILES`, ascending
   seed from 0), where the engine already recovers MORE than the ladder at
   the DEFAULT LTV -- one of the 36/256 such cells `reports/gates.md`'s
   B13 entry already measured to exist. Selected this way (never by
   scanning for the "nicest" curve) so `interpolate_crossing()` has a
   genuine swept-and-measured sign change to interpolate on at least one
   slice, without hand-picking which one.

Both slices are swept over the SAME LTV grid, so the two curves are
directly comparable and neither gets resolution the other lacks.

Run: python -m eval.ltv_sensitivity
Writes: reports/ltv_sensitivity.json. `eval/report.py` reads it (if
present) and renders `reports/regimes.md`'s "LTV sensitivity" section from
it -- this script computes nothing at render time, the same discipline
`eval/run.py`/`eval/report.py` already use for the rest of `regimes.md`.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from fractions import Fraction
from typing import Any

from eval import regimes as regimes_mod
from eval.allocator_sweep import fit_nominal_hazard_model, hazard_from_fit
from eval.frozen.simulator import Simulator, load_config
from eval.run import ALL_ARMS, ALL_PROFILES, fit_gate, run_engine_cell, run_ladder_cell
from src.core.clock import now as clock_now
from src.core.money import interpolate_crossing
from src.core.types import Profile
from src.policy.costs import load as load_costs

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = _REPO_ROOT / "reports" / "ltv_sensitivity.json"

HEADLINE_SLICE: tuple[str, str, Profile, int] = ("baseline", "nominal", Profile.strict, 0)

# Scope, disclosed: the worked-example search checks these seeds only, not
# the full 8-seed grid the main B13 sweep uses -- each engine cell costs
# real wall-clock time and this script already runs LTV_GRID_PAISE points
# twice (headline + worked example). If no winning cell turns up in this
# range, _find_worked_example_cell() raises rather than silently widening
# the search -- that would be a real finding, not a bug to route around.
_SEARCH_SEEDS: tuple[int, ...] = (0, 1, 2)

# LTV grid, paise -- identical for both slices. Linear and fine (50k steps)
# up to Rs 30,000 (a realistic multiple of a below-AFA-cliff mandate),
# coarser beyond that to confirm the curve's shape stays monotone-in-sign
# out to a deliberately unrealistic ceiling (Rs 10 lakh) without paying for
# fine resolution nothing needs.
LTV_GRID_PAISE: tuple[int, ...] = tuple(range(0, 3_000_001, 50_000)) + (
    5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000,
)


def _find_worked_example_cell(
    base_cfg: dict, costs, hazard, gate, gate_kind: str,
) -> tuple[str, str, Profile, int]:
    """The first (regime, arm, profile, seed) cell, in this codebase's own
    pre-existing enumeration order, where engine.recovered_paise already
    exceeds ladder.recovered_paise at the DEFAULT LTV. See module
    docstring for why this selection rule is not cherry-picking."""
    for regime in regimes_mod.REGIMES:
        cfg = regimes_mod.config_for(regime, base_cfg)
        for arm in regimes_mod.arms_for(regime, ALL_ARMS):
            for profile in ALL_PROFILES:
                for seed in _SEARCH_SEEDS:
                    ladder = run_ladder_cell(regime, arm, profile, cfg, seed)
                    engine = run_engine_cell(
                        regime, arm, profile, cfg, seed,
                        hazard=hazard, costs=costs, gate=gate, gate_kind=gate_kind,
                    )
                    if engine.recovered_paise > ladder.recovered_paise:
                        return regime, arm, profile, seed
    raise RuntimeError(
        f"no (regime, arm, profile, seed) cell in the searched range "
        f"(seeds {_SEARCH_SEEDS}) has engine.recovered_paise > "
        f"ladder.recovered_paise at the default LTV -- reports/gates.md's "
        f"B13 entry measured 36/256 such cells across the FULL 8-seed grid, "
        f"so this narrower search coming up empty is itself worth "
        f"reporting, not silently retried with a wider range"
    )


def _sweep(
    regime: str, arm: str, profile: Profile, seed: int,
    base_cfg: dict, costs, hazard, gate, gate_kind: str,
    *, grid: tuple[int, ...] = LTV_GRID_PAISE,
) -> dict[str, Any]:
    """`grid` defaults to the full LTV_GRID_PAISE but is overridable --
    tests pass a handful of points so the suite doesn't pay for a 66-point
    sweep to exercise this function's logic."""
    cfg = regimes_mod.config_for(regime, base_cfg)
    ladder = run_ladder_cell(regime, arm, profile, cfg, seed)
    sim = Simulator(arm, seed=seed, config=cfg)
    mean_amount_paise = Fraction(sum(m.amount_paise for m in sim.mandates), len(sim.mandates))

    points: list[dict[str, int]] = []
    for ltv in grid:
        c = dataclasses.replace(costs, mandate_ltv_paise=ltv)
        engine = run_engine_cell(
            regime, arm, profile, cfg, seed,
            hazard=hazard, costs=c, gate=gate, gate_kind=gate_kind,
        )
        points.append({
            "ltv_paise": ltv,
            "engine_recovered_paise": engine.recovered_paise,
            "ladder_recovered_paise": ladder.recovered_paise,
            "diff_paise": engine.recovered_paise - ladder.recovered_paise,
        })

    # interpolate_crossing() itself is the sign-change check -- calling it
    # and catching its ValueError is the whole detector, not a second,
    # hand-rolled copy of the same test.
    crossings: list[dict[str, Any]] = []
    for p0, p1 in zip(points, points[1:]):
        try:
            x = interpolate_crossing(
                p0["ltv_paise"], p0["diff_paise"], p1["ltv_paise"], p1["diff_paise"],
            )
        except ValueError:
            continue
        ratio = x / mean_amount_paise
        crossings.append({
            "bracket_low_paise": p0["ltv_paise"],
            "bracket_high_paise": p1["ltv_paise"],
            "crossing_ltv_paise_exact": str(x),
            "crossing_ltv_paise": float(x),
            "ratio_to_mean_amount_exact": str(ratio),
            "ratio_to_mean_amount": float(ratio),
        })

    return {
        "regime": regime, "arm": arm, "profile": profile.value, "seed": seed,
        "mean_amount_paise": float(mean_amount_paise),
        "n_mandates": len(sim.mandates),
        "points": points,
        "crossings": crossings,
    }


def main() -> int:
    base_cfg = load_config()
    costs = load_costs()

    print("fitting hazard model on the nominal corpus...", file=sys.stderr)
    hazard = hazard_from_fit(fit_nominal_hazard_model())
    print("calibrating the conformal off-ramp gate on baseline...", file=sys.stderr)
    gate, gate_kind, gate_diag = fit_gate(base_cfg)
    print(f"  gate: {gate_kind} {gate_diag}", file=sys.stderr)

    h_regime, h_arm, h_profile, h_seed = HEADLINE_SLICE
    print(f"sweeping HEADLINE slice ({h_regime}/{h_arm}/{h_profile.value}/seed={h_seed})...",
          file=sys.stderr)
    headline = _sweep(h_regime, h_arm, h_profile, h_seed, base_cfg, costs, hazard, gate, gate_kind)

    print("locating the worked-example cell...", file=sys.stderr)
    we_regime, we_arm, we_profile, we_seed = _find_worked_example_cell(
        base_cfg, costs, hazard, gate, gate_kind
    )
    print(f"  found: {we_regime}/{we_arm}/{we_profile.value}/seed={we_seed}", file=sys.stderr)
    print("sweeping WORKED EXAMPLE slice...", file=sys.stderr)
    worked_example = _sweep(
        we_regime, we_arm, we_profile, we_seed, base_cfg, costs, hazard, gate, gate_kind
    )

    out = {
        "schema": 1,
        "generated": clock_now().isoformat(),
        "ltv_grid_paise": list(LTV_GRID_PAISE),
        "default_ltv_paise": costs.mandate_ltv_paise,
        "headline": headline,
        "worked_example": worked_example,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JSON}", file=sys.stderr)
    print(f"headline crossings: {len(headline['crossings'])}", file=sys.stderr)
    print(f"worked-example crossings: {len(worked_example['crossings'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
