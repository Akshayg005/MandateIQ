"""eval/offramp_channel.py -- R5's channel-quality sweep.

The gate (reports/gates.md, "Post-B16 remediation gates", R5) requires
BOTH error costs and the channel's own ROC at EVERY point of the sweep, and
that the conformal singleton stays the only firing rule. These tests pin
the shape of that artifact and the pre-registration that makes it a result
rather than a search.

They run a deliberately tiny `grid=`/`seeds=` (the same convention
eval/ltv_sensitivity.py's tests use) -- the suite must not pay for the
full 16-point x 8-seed sweep to check this module's logic.
"""
from __future__ import annotations

import pytest

from eval import offramp_channel as oc


# --- pre-registration -------------------------------------------------------

def test_the_operating_point_is_a_point_the_sweep_measures():
    """An operating point outside the grid would be an unmeasured claim."""
    assert oc.OPERATING_POINT in oc.QUALITY_GRID


def test_the_grid_includes_a_worthless_channel():
    """"A sweep that only shows good channels proves nothing" -- the module
    docstring's own words. At least one point must carry no information at
    all (nominal AUC 0.5, i.e. tpr == fpr)."""
    assert any(tpr == fpr for tpr, fpr in oc.QUALITY_GRID)


def test_the_grid_spans_worthless_to_oracle():
    aucs = [(1.0 + t - f) / 2.0 for t, f in oc.QUALITY_GRID]
    assert min(aucs) == pytest.approx(0.5)
    assert max(aucs) == pytest.approx(1.0)


def test_the_operating_point_is_not_an_oracle():
    """The whole reason for choosing a mid-quality point: no headline
    number may rest on a channel that is assumed excellent."""
    tpr, fpr = oc.OPERATING_POINT
    assert tpr < 0.8
    assert fpr > 0.05


def test_eval_run_imports_the_operating_point_rather_than_restating_it():
    """Two copies of an operating point is one copy too many -- the
    published grid and the sweep that justified its operating point cannot
    be allowed to drift apart silently."""
    import inspect

    import eval.run as run_mod

    src = inspect.getsource(run_mod.channel_spec_from_args)
    assert "from eval.offramp_channel import OPERATING_POINT" in src


def test_both_channels_are_swept():
    """DECISIONS.md, 2026-09-04, R0 pre-registered BOTH channels, not a
    pick between them."""
    assert set(oc.CHANNEL_KINDS) == {"decline", "intent"}


# --- the artifact -----------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_sweep():
    return oc.sweep(grid=((1.00, 0.00), (0.30, 0.30)), kinds=("decline",),
                    seeds=(0,), verbose=False)


def test_the_artifact_declares_itself_synthetic(tiny_sweep):
    """Disclosed in the artifact itself, not only in prose a renderer might
    drop: the channel reads privileged ground truth."""
    assert tiny_sweep["synthetic"] is True
    assert "privileged true cause" in tiny_sweep["disclosure"]


def test_every_point_reports_both_error_costs(tiny_sweep):
    """The gate's literal wording: "both the recovery cost and the
    false-off-ramp rate reported at every point"."""
    for p in tiny_sweep["points"]:
        assert p["missed_recovery_count"] is not None
        assert p["missed_recovery_paise"] is not None
        assert p["engine_recovered_paise"] is not None
        assert p["ladder_recovered_paise"] is not None
        assert "false_offramp_count" in p
        assert "false_offramp_rate" in p


def test_every_point_publishes_the_channels_own_roc(tiny_sweep):
    for p in tiny_sweep["points"]:
        roc = p["channel_roc"]
        assert roc["n"] > 0
        assert roc["tpr_realised"] is not None
        assert roc["fpr_realised"] is not None
        assert roc["auc"] is not None
        lo, hi = roc["auc_ci"]
        assert lo <= roc["auc"] <= hi


def test_the_reported_roc_is_realised_not_the_configured_parameter(tiny_sweep):
    """A ROC that merely echoes the input is not a measurement. At the
    oracle point the two coincide by construction; at the noisy point they
    must not be assumed to."""
    noisy = [p for p in tiny_sweep["points"] if p["tpr"] == 0.30][0]
    roc = noisy["channel_roc"]
    assert roc["tpr_realised"] == pytest.approx(0.30, abs=0.15)
    assert roc["fpr_realised"] == pytest.approx(0.30, abs=0.15)
    # Realised, so equality with the nominal parameter is a coincidence,
    # not a guarantee -- the counts must at least be real.
    assert roc["n_wont_pay"] > 0
    assert roc["n_other"] > 0


def test_the_false_offramp_rate_has_a_real_denominator(tiny_sweep):
    """Before R5 false_offramp_count was computed inside the would_pay
    branch, so an OFFER to a mandate that would NOT have paid was counted
    nowhere and the metric had no denominator at all."""
    for p in tiny_sweep["points"]:
        assert p["offramp_scored_count"] == p["false_offramp_count"] + p["true_offramp_count"]
        if p["offramp_scored_count"]:
            assert p["false_offramp_rate"] == pytest.approx(
                p["false_offramp_count"] / p["offramp_scored_count"]
            )
        else:
            assert p["false_offramp_rate"] is None


def test_the_offramp_actually_fires_under_a_good_channel(tiny_sweep):
    """R5's headline condition, on the slice: n_offer > 0. Before R5 this
    was 0 in all 256 engine cells, by construction."""
    oracle = [p for p in tiny_sweep["points"] if p["tpr"] == 1.00][0]
    assert oracle["n_offer"] > 0


def test_a_worse_channel_costs_more_false_off_ramps_per_offer(tiny_sweep):
    """The point of sweeping quality at all. Not a claim about magnitudes
    -- only that degradation is VISIBLE rather than hidden behind a single
    flattering operating point."""
    by_tpr = {p["tpr"]: p for p in tiny_sweep["points"]}
    assert by_tpr[0.30]["false_offramp_rate"] > by_tpr[1.00]["false_offramp_rate"]


def test_the_gate_is_refit_at_every_point(tiny_sweep):
    """Re-calibration is mandatory: the channel changes the belief
    distribution, so it changes the calibration pool. Each point's
    diagnostics must describe ITS OWN channel."""
    for p in tiny_sweep["points"]:
        diag = p["gate_diagnostics"]
        assert diag["channel"]["tpr"] == p["tpr"]
        assert diag["channel"]["fpr"] == p["fpr"]


def test_the_mondrian_floor_is_reported_and_met(tiny_sweep):
    """ceil(1/alpha) - 1 = 19 per class at alpha=0.05. calibrate() raises
    ConformalUnderpowered below it, so a conformal gate reaching here
    proves the floor holds -- the counts are recorded anyway, because "it
    did not raise" is a weaker artifact than the numbers."""
    for p in tiny_sweep["points"]:
        if p["gate_kind"] != "conformal":
            continue
        diag = p["gate_diagnostics"]
        assert diag["mondrian_floor"] == 19
        for cause, n in diag["calib_per_class"].items():
            assert n >= diag["mondrian_floor"], (cause, n)


def test_coverage_is_re_reported_after_recalibration(tiny_sweep):
    """R5 review pass, 2026-09-05 (stats-reviewer): this test's own FIRST
    version asserted `0.0 <= coverage_marginal_mean <= 1.0`, which cannot
    fail for any probability -- a genuine calibration break (e.g. a
    quantile computation returning garbage) would pass silently. This
    project's stance on the 0.95 TARGET is measured-not-asserted (coverage
    is known to run 0.899-0.986 on the full grid -- see reports/gates.md's
    R5 entry -- and that under-coverage is a disclosed finding, not a bug
    to gate on), so the floor below is NOT 0.95: it is a value comfortably
    under every measured point (0.93-0.95 on this tiny grid) that a real
    breakage (calibration collapsing toward 0.5 or below) would still
    trip."""
    for p in tiny_sweep["points"]:
        assert p["coverage_n"] > 0
        assert p["coverage_marginal_mean"] > 0.75


def test_every_point_carries_habitual_fraction_and_repeat_rate(tiny_sweep):
    """The main QUALITY_GRID always runs at habitual_fraction=1.0 -- the
    dependence dimension is swept separately -- but the field must be
    present on every row so a reader of the artifact can tell the two
    sweeps apart without cross-referencing code."""
    for p in tiny_sweep["points"]:
        assert p["habitual_fraction"] == 1.0
        assert "rate" in p["repeat_false_fire"]


# === R5 review pass, 2026-09-05: dependence_sweep() (stats-reviewer, HIGH) ==

@pytest.fixture(scope="module")
def tiny_dependence_sweep():
    return oc.dependence_sweep(grid=(1.0, 0.3), seeds=(0, 1), verbose=False)


def test_dependence_sweep_runs_at_the_pre_registered_operating_point(tiny_dependence_sweep):
    op = tiny_dependence_sweep["operating_point"]
    assert (op["tpr"], op["fpr"]) == oc.OPERATING_POINT


def test_dependence_sweep_holds_the_marginal_fpr_fixed_across_the_grid(tiny_dependence_sweep):
    """The entire point of the two-point mixture: every row must measure
    approximately the SAME marginal fpr as the main grid's operating-point
    row, so a difference in the false-off-ramp rate is attributable to
    correlation and not to a confounded discrimination change."""
    tpr, fpr = oc.OPERATING_POINT
    for p in tiny_dependence_sweep["points"]:
        realised = p["channel_roc"]["fpr_realised"]
        assert realised == pytest.approx(fpr, abs=0.08), (
            f"habitual_fraction={p['habitual_fraction']}: realised fpr "
            f"{realised} drifted from the fixed marginal {fpr}"
        )


def test_dependence_sweep_reports_habitual_fraction_per_point(tiny_dependence_sweep):
    hfs = [p["habitual_fraction"] for p in tiny_dependence_sweep["points"]]
    assert hfs == [1.0, 0.3]


def test_dependence_sweep_declares_itself_synthetic_and_further_uncalibrated(tiny_dependence_sweep):
    """A second-order disclosure beyond the main grid's: the CORRELATION
    swept here is an assumption with no real corpus behind it, and the
    sweep establishes sensitivity, not a corrected estimate. Losing this
    distinction would let a reader mistake "the rate moves under
    correlation" for "the rate under real dependence is X"."""
    assert tiny_dependence_sweep["synthetic"] is True
    assert "sensitivity" in tiny_dependence_sweep["disclosure"].lower()
    assert "not a corrected estimate" in tiny_dependence_sweep["disclosure"]


def test_the_default_grid_point_matches_the_main_grids_operating_point_row():
    """habitual_fraction=1.0 in dependence_sweep() must be the SAME
    computation as the operating-point row of the main QUALITY_GRID --
    same channel, same seeds, same everything except which function ran
    it. If these ever disagree, the two sweeps are silently measuring
    different things at what claims to be their one shared point."""
    tpr, fpr = oc.OPERATING_POINT
    from_main = oc.sweep(grid=(oc.OPERATING_POINT,), kinds=("decline",),
                         seeds=(0, 1), verbose=False)["points"][0]
    from_dependence = oc.dependence_sweep(grid=(1.0,), seeds=(0, 1), verbose=False)["points"][0]
    assert from_main["n_offer"] == from_dependence["n_offer"]
    assert from_main["false_offramp_count"] == from_dependence["false_offramp_count"]
    assert from_main["channel_roc"]["fpr_realised"] == from_dependence["channel_roc"]["fpr_realised"]
