"""eval/report.py's R5 additions: `_offramp_channel` and the two-way
headline finding 2. This file computes nothing -- these tests are about
RENDERING already-computed numbers correctly, matching the rest of
eval/report.py's "reads, never computes" discipline.
"""
from __future__ import annotations

import json

from eval.report import _finding_2, _offramp_channel


def _point(**kw):
    base = {
        "channel_kind": "decline", "tpr": 0.60, "fpr": 0.15,
        "nominal_auc": 0.725, "is_operating_point": True, "seeds": [0],
        "gate_kind": "conformal", "gate_diagnostics": {},
        "engine_recovered_paise": 12_000_00, "engine_attempts_spent": 300,
        "engine_mandates_preserved": 140,
        "ladder_recovered_paise": 20_000_00, "ladder_attempts_spent": 400,
        "ladder_mandates_preserved": 110,
        "billable_paise": 100_000_00, "n_mandates": 200,
        "n_attempt": 300, "n_offer": 14, "n_reauth": 20, "n_stop": 5,
        "missed_recovery_count": 9, "missed_recovery_paise": 90_000,
        "offramp_scored_count": 14, "false_offramp_count": 2,
        "false_offramp_paise": 20_000, "true_offramp_count": 12,
        "true_offramp_paise": 120_000, "false_offramp_rate": 2 / 14,
        "channel_roc": {"n": 900, "n_wont_pay": 300, "n_other": 600,
                        "tpr_realised": 0.61, "fpr_realised": 0.14,
                        "auc": 0.735, "auc_ci": [0.70, 0.77]},
        "coverage_marginal_mean": 0.948,
        "singleton_wont_pay_rate_mean": 0.047, "coverage_n": 4000,
    }
    base.update(kw)
    return base


def _artifact(tmp_path, points):
    path = tmp_path / "offramp_channel.json"
    path.write_text(json.dumps({
        "schema": 1, "generated": "2026-09-05T00:00:00+05:30",
        "slice": {"regime": "baseline", "arm": "nominal",
                  "profile": "strict", "seeds": [0, 1]},
        "operating_point": {"tpr": 0.60, "fpr": 0.15},
        "quality_grid": [[0.6, 0.15]], "channel_kinds": ["decline"],
        "synthetic": True,
        "disclosure": "reads the simulator's privileged true cause",
        "points": points,
    }), encoding="utf-8")
    return path


# --- the renderer -----------------------------------------------------------

def test_a_missing_artifact_renders_a_placeholder_not_a_crash(tmp_path):
    """A tree that has never run the sweep must still render the rest of
    regimes.md -- the same graceful-absence rule _ltv_sensitivity() follows."""
    lines = _offramp_channel(tmp_path / "nope.json")
    assert "python -m eval.offramp_channel" in "\n".join(lines)


def test_the_section_discloses_the_channel_is_synthetic(tmp_path):
    text = "\n".join(_offramp_channel(_artifact(tmp_path, [_point()])))
    assert "SYNTHETIC" in text
    assert "privileged true cause" in text


def test_the_section_publishes_the_roc_beside_every_row(tmp_path):
    """The gate's literal requirement: the channel's own ROC published
    beside the results, not in a separate section a reader might miss."""
    text = "\n".join(_offramp_channel(_artifact(tmp_path, [_point()])))
    assert "realised AUC" in text
    assert "0.735 [0.700, 0.770]" in text


def test_the_section_reports_both_error_costs(tmp_path):
    text = "\n".join(_offramp_channel(_artifact(tmp_path, [_point()])))
    assert "false off-ramp" in text
    assert "true off-ramp" in text
    assert "14.3%" in text


def test_money_is_rendered_through_the_money_helper(tmp_path):
    """DESIGN.md: nothing but src/core/money.py formats currency, and
    eval/ is inside MONEY_DIRS."""
    text = "\n".join(_offramp_channel(_artifact(tmp_path, [_point()])))
    assert "₹12,000.00" in text           # engine recovered
    assert "-₹8,000.00" in text           # signed delta vs the ladder


def test_the_operating_point_row_is_marked(tmp_path):
    pts = [_point(tpr=0.30, fpr=0.30, is_operating_point=False,
                  n_offer=8, false_offramp_count=5, true_offramp_count=3,
                  offramp_scored_count=8, false_offramp_rate=5 / 8,
                  channel_roc={"n": 900, "n_wont_pay": 300, "n_other": 600,
                               "tpr_realised": 0.31, "fpr_realised": 0.30,
                               "auc": 0.505, "auc_ci": [0.47, 0.54]}),
           _point()]
    text = "\n".join(_offramp_channel(_artifact(tmp_path, pts)))
    assert "0.60 **<-**" in text
    assert "62.5%" in text
    assert "degradation is the point of sweeping quality" in text


# --- headline finding 2 -----------------------------------------------------

def _eng_cells(**kw):
    base = {
        "n_offer": 14, "offramp_scored_count": 14, "false_offramp_count": 2,
        "true_offramp_count": 12, "coverage_n_retrospective": 100,
        "channel_n_wont_pay": 300, "channel_positive_on_wont_pay": 183,
        "channel_n_other": 600, "channel_positive_on_other": 84,
    }
    base.update(kw)
    return [base]


def test_finding_2_says_unreachable_when_the_channel_is_off():
    """`--channel-kind off` still reproduces the pre-R5 configuration
    exactly, so the report must still describe it correctly."""
    text = _finding_2({"wontpay_channel": None}, _eng_cells(n_offer=0),
                      [], 0, [0.0])
    assert "cannot fire" in text
    assert "arithmetic, not measurement" in text


def test_finding_2_reports_real_numbers_when_the_channel_is_live():
    data = {"wontpay_channel": {"kind": "decline", "tpr": 0.60, "fpr": 0.15}}
    text = _finding_2(data, _eng_cells(), [], 14, [0.047])
    assert "now fires" in text
    assert "14.3%" in text
    assert "SYNTHETIC" in text
    assert "privileged ground truth" in text
    # realised, not the configured parameter
    assert "0.610" in text and "0.140" in text


def test_finding_2_never_claims_the_off_ramp_is_correct():
    """R5's gate: reachable and measured, never correct. A report claiming
    more than the gate bought is the exact failure this project has already
    amended two gates over."""
    data = {"wontpay_channel": {"kind": "decline", "tpr": 0.60, "fpr": 0.15}}
    text = _finding_2(data, _eng_cells(), [], 14, [0.047])
    assert "tested-and-imperfect" in text
    assert "not a good result" in text
