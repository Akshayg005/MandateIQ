"""eval/ltv_sensitivity.py (R3, reports/gates.md).

Tests here deliberately use a small, explicit `grid` (via `_sweep()`'s
override parameter) rather than the module's full 66-point
`LTV_GRID_PAISE` -- the full sweep is a real-report-generation cost, not
a unit-test one. The specific numbers asserted below (the crossing
brackets, the "no crossing" result on the headline slice) were measured
directly by running this module before these tests were written; they
pin the current, real behaviour of a real (deterministic, seeded)
simulation, not a hand-guessed expectation.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from eval.ltv_sensitivity import HEADLINE_SLICE, _find_worked_example_cell, _sweep
from eval.allocator_sweep import fit_nominal_hazard_model, hazard_from_fit
from eval.frozen.simulator import load_config
from eval.run import fit_gate
from src.core.types import Profile
from src.policy.costs import load as load_costs


@pytest.fixture(scope="module")
def _fitted():
    """Fit the hazard model and calibrate the gate ONCE for this whole
    test module -- both are deterministic given the frozen config/seeds,
    and refitting per test would multiply this file's runtime for no
    additional coverage."""
    base_cfg = load_config()
    costs = load_costs()
    hazard = hazard_from_fit(fit_nominal_hazard_model())
    gate, gate_kind, _diag = fit_gate(base_cfg)
    return base_cfg, costs, hazard, gate, gate_kind


def test_headline_slice_has_no_crossing_on_a_wide_bracketing_grid(_fitted):
    """The headline (baseline/nominal/strict/seed=0) slice's engine never
    recovers as much money as the ladder anywhere in [0, 100M] paise --
    measured directly (module docstring's DECISIONS.md entry), pinned here
    with a coarse 3-point grid spanning the same range."""
    base_cfg, costs, hazard, gate, gate_kind = _fitted
    regime, arm, profile, seed = HEADLINE_SLICE
    result = _sweep(
        regime, arm, profile, seed, base_cfg, costs, hazard, gate, gate_kind,
        grid=(0, 50_000_000, 100_000_000),
    )
    assert result["crossings"] == []
    # every point's diff must be negative -- engine strictly behind ladder
    assert all(p["diff_paise"] < 0 for p in result["points"])


def test_worked_example_cell_has_two_crossings_in_the_known_brackets(_fitted):
    """issuer_outage/nominal/strict/seed=0 is a pre-existing engine-wins-
    on-money cell at the default LTV (180,000 paise): the diff is negative
    at LTV=0, positive at 180,000, negative again by 500,000 -- a
    rise-then-fall shape with two sign changes. Measured directly before
    writing this test."""
    base_cfg, costs, hazard, gate, gate_kind = _fitted
    result = _sweep(
        "issuer_outage", "nominal", Profile.strict, 0,
        base_cfg, costs, hazard, gate, gate_kind,
        grid=(0, 100_000, 150_000, 450_000, 500_000),
    )
    assert len(result["crossings"]) == 2
    first, second = result["crossings"]
    assert (first["bracket_low_paise"], first["bracket_high_paise"]) == (100_000, 150_000)
    assert (second["bracket_low_paise"], second["bracket_high_paise"]) == (450_000, 500_000)
    # first crossing rises through zero (diff goes negative -> positive),
    # second falls back through it (positive -> negative)
    assert 100_000 < first["crossing_ltv_paise"] < 150_000
    assert 450_000 < second["crossing_ltv_paise"] < 500_000


def test_worked_example_ratio_to_mean_amount_is_exact_fraction_arithmetic(_fitted):
    """ratio_to_mean_amount_exact must be the crossing (an exact Fraction,
    per interpolate_crossing()'s own contract) divided by mean_amount_paise
    -- never a float division re-derivation that could silently disagree
    with the float convenience field."""
    base_cfg, costs, hazard, gate, gate_kind = _fitted
    result = _sweep(
        "issuer_outage", "nominal", Profile.strict, 0,
        base_cfg, costs, hazard, gate, gate_kind,
        grid=(0, 100_000, 150_000),
    )
    assert len(result["crossings"]) == 1
    c = result["crossings"][0]
    x = Fraction(c["crossing_ltv_paise_exact"])
    # float mean_amount_paise (the JSON-friendly field) loses precision vs
    # the true Fraction sum/count _sweep() computes internally -- recompute
    # it exactly the same way, from the same simulator, rather than
    # round-tripping a float.
    from eval import regimes as regimes_mod
    from eval.frozen.simulator import Simulator

    cfg = regimes_mod.config_for("issuer_outage", base_cfg)
    sim = Simulator("nominal", seed=0, config=cfg)
    exact_mean = Fraction(sum(m.amount_paise for m in sim.mandates), len(sim.mandates))
    assert Fraction(c["ratio_to_mean_amount_exact"]) == x / exact_mean


def test_find_worked_example_cell_actually_beats_the_ladder(_fitted):
    """Whatever cell _find_worked_example_cell() returns, re-run it
    directly and confirm engine.recovered_paise > ladder.recovered_paise
    at the default LTV -- the claim the function exists to make, checked
    independently of its own internal logic."""
    from eval.run import run_engine_cell, run_ladder_cell
    from eval import regimes as regimes_mod

    base_cfg, costs, hazard, gate, gate_kind = _fitted
    regime, arm, profile, seed = _find_worked_example_cell(base_cfg, costs, hazard, gate, gate_kind)
    cfg = regimes_mod.config_for(regime, base_cfg)
    ladder = run_ladder_cell(regime, arm, profile, cfg, seed)
    engine = run_engine_cell(regime, arm, profile, cfg, seed,
                             hazard=hazard, costs=costs, gate=gate, gate_kind=gate_kind)
    assert engine.recovered_paise > ladder.recovered_paise


def test_find_worked_example_cell_is_deterministic(_fitted):
    """Same inputs, same result -- no hidden randomness in cell selection."""
    base_cfg, costs, hazard, gate, gate_kind = _fitted
    a = _find_worked_example_cell(base_cfg, costs, hazard, gate, gate_kind)
    b = _find_worked_example_cell(base_cfg, costs, hazard, gate, gate_kind)
    assert a == b


def test_find_worked_example_cell_raises_when_search_range_has_no_winner(_fitted, monkeypatch):
    """A search space shrunk to exactly one cell already known to lose
    (baseline/nominal/strict/seed=0, per the headline-slice test above)
    must raise RuntimeError, not silently return a wrong cell or hang.
    The search space is monkeypatched down to that one cell so this test
    doesn't pay for exhausting the real ~100-cell search range."""
    import eval.ltv_sensitivity as ltv_mod
    from eval import regimes as regimes_mod

    base_cfg, costs, hazard, gate, gate_kind = _fitted
    monkeypatch.setattr(regimes_mod, "REGIMES", {"baseline": regimes_mod.REGIMES["baseline"]})
    monkeypatch.setattr(ltv_mod, "ALL_ARMS", ("nominal",))
    monkeypatch.setattr(ltv_mod, "ALL_PROFILES", (Profile.strict,))
    monkeypatch.setattr(ltv_mod, "_SEARCH_SEEDS", (0,))

    with pytest.raises(RuntimeError, match="no .* cell"):
        ltv_mod._find_worked_example_cell(base_cfg, costs, hazard, gate, gate_kind)
