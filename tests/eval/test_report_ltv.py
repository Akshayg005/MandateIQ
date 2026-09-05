"""eval/report.py's R3 additions: _signed_rupees, _ltv_slice_table,
_ltv_sensitivity. This file computes nothing -- these tests are about
RENDERING already-computed numbers correctly, matching the rest of
eval/report.py's "reads, never computes" discipline.
"""
from __future__ import annotations

import json

import pytest

from eval.report import _ltv_sensitivity, _ltv_slice_table, _signed_rupees


def test_signed_rupees_negative():
    assert _signed_rupees(-15_047_099) == "-₹1,50,470.99"


def test_signed_rupees_positive():
    assert _signed_rupees(378_237) == "+₹3,782.37"


def test_signed_rupees_zero_is_positive_sign():
    assert _signed_rupees(0) == "+₹0.00"


def test_ltv_slice_table_no_crossings_renders_the_refusal_explanation():
    slice_data = {
        "regime": "baseline", "arm": "nominal", "profile": "strict", "seed": 0,
        "mean_amount_paise": 1_329_241.09, "n_mandates": 200,
        "points": [
            {"ltv_paise": 0, "diff_paise": -15_047_099},
            {"ltv_paise": 100_000_000, "diff_paise": -102_038_783},
        ],
        "crossings": [],
    }
    lines = _ltv_slice_table(slice_data)
    text = "\n".join(lines)
    assert "No crossing anywhere" in text
    assert "interpolate_crossing()" in text
    # negative diffs must render via the signed formatter, not crash
    assert "-₹1,50,470.99" in text
    assert "-₹10,20,387.83" in text


def test_ltv_slice_table_with_crossings_renders_the_bracket_table():
    slice_data = {
        "regime": "issuer_outage", "arm": "nominal", "profile": "strict", "seed": 0,
        "mean_amount_paise": 1_329_241.09, "n_mandates": 200,
        "points": [],
        "crossings": [
            {
                "bracket_low_paise": 100_000, "bracket_high_paise": 150_000,
                "crossing_ltv_paise_exact": "84224330000/706173",
                "crossing_ltv_paise": 119268.69,
                "ratio_to_mean_amount_exact": "8422433000000/93867416824857",
                "ratio_to_mean_amount": 0.0897,
            },
        ],
    }
    lines = _ltv_slice_table(slice_data)
    text = "\n".join(lines)
    assert "[100,000, 150,000]" in text
    assert "0.090" in text
    assert "No crossing" not in text


def test_ltv_sensitivity_missing_artifact_renders_placeholder(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    lines = _ltv_sensitivity(path=missing)
    text = "\n".join(lines)
    assert "Not yet generated" in text
    assert "python -m eval.ltv_sensitivity" in text


def test_ltv_sensitivity_reads_and_renders_a_real_artifact_shape(tmp_path):
    """A minimal but schema-complete artifact -- both slices, one with a
    crossing and one without -- must render without error and mention
    both cells' identities."""
    artifact = {
        "schema": 1,
        "generated": "2026-09-04T00:00:00+00:00",
        "ltv_grid_paise": [0, 100_000_000],
        "default_ltv_paise": 180_000,
        "headline": {
            "regime": "baseline", "arm": "nominal", "profile": "strict", "seed": 0,
            "mean_amount_paise": 1_329_241.09, "n_mandates": 200,
            "points": [
                {"ltv_paise": 0, "diff_paise": -15_047_099},
                {"ltv_paise": 100_000_000, "diff_paise": -102_038_783},
            ],
            "crossings": [],
        },
        "worked_example": {
            "regime": "issuer_outage", "arm": "nominal", "profile": "strict", "seed": 0,
            "mean_amount_paise": 1_329_241.09, "n_mandates": 200,
            "points": [],
            "crossings": [
                {
                    "bracket_low_paise": 100_000, "bracket_high_paise": 150_000,
                    "crossing_ltv_paise_exact": "84224330000/706173",
                    "crossing_ltv_paise": 119268.69,
                    "ratio_to_mean_amount_exact": "8422433000000/93867416824857",
                    "ratio_to_mean_amount": 0.0897,
                },
            ],
        },
    }
    path = tmp_path / "ltv_sensitivity.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    lines = _ltv_sensitivity(path=path)
    text = "\n".join(lines)
    assert "baseline/nominal/strict/seed=0" in text
    assert "issuer_outage/nominal/strict/seed=0" in text
    assert "36/256" in text  # the B13 provenance note for the worked example
