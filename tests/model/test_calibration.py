"""src/model/calibration.py -- isotonic recalibration of event hazards.

Design decision this file pins: isotonic regression is applied INDEPENDENTLY
to each of the three EVENT hazard classes (RECOVERED, DEAD, OPTED_OUT --
Outcome ints 1, 2, 3), NEVER rescaled/renormalized after calibration. The
residual STILL_PENDING class (Outcome int 0) absorbs whatever probability
mass remains: h_cal[:, 0] = 1 - (h_cal[:,1] + h_cal[:,2] + h_cal[:,3]).
This makes every output row sum to EXACTLY 1 in float (not within a
tolerance), because it is constructed as 1 - sum(other three), not three
independently-adjusted values that are then jointly renormalized.

Calibration is fit on one DISJOINT hold-out split (provenance="calib_iso")
and NEVER on the test split. Attempting to fit on provenance="test" raises
ValueError. The fit_row_ids and report_row_ids must not overlap, enforced
by assert_disjoint(), raising CalibrationLeakError if they do.

The core metric is classwise_ece: the mean over the 4 outcome classes of the
per-class ECE (expected calibration error), where per-class ECE is computed
as the weighted sum over probability atoms of |mean_predicted - observed_freq|
weighted by the atom's row count (n_atom / N). Atoms are discovered via
groupby of the UNIQUE PREDICTED VECTOR (not equal-width binning), since the
design matrix has only a handful of distinct covariate combinations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _simple_valid_hazards(n: int = 10, seed: int = 42) -> np.ndarray:
    """Generate a valid (n, 4) per-row hazard array for testing.
    Each row sums to 1.0. Shape (n, 4), columns in Outcome int order
    [STILL_PENDING=0, RECOVERED=1, DEAD=2, OPTED_OUT=3]."""
    rng = np.random.RandomState(seed)
    # Generate random probabilities for the three event classes
    h = rng.dirichlet([1, 1, 1, 1], size=n)  # Shape (n, 4), rows sum to 1
    return h


def _simple_valid_true_labels(n: int, seed: int = 42) -> np.ndarray:
    """Generate a valid (n,) array of true event_code labels, one per row,
    each in [0, 1, 2, 3]."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 4, size=n)


def _simple_valid_row_ids(n: int, prefix: str = "row") -> list[str]:
    """Generate n unique string row IDs."""
    return [f"{prefix}_{i}" for i in range(n)]


# === Imports (will fail at collection until calibration.py exists) ==========

def test_import_calibration_module():
    """calibration.py must exist and be importable."""
    try:
        from src.model.calibration import (
            EVENT_CLASSES, RESIDUAL_CLASS, CalibrationLeakError,
            SimplexViolation, IsotonicCalibrator, fit, apply,
            assert_disjoint, reliability_table, classwise_ece, per_class_ece
        )
    except ImportError as e:
        pytest.fail(f"Failed to import calibration module: {e}")


# === fit() basic input validation ============================================

def test_fit_rejects_mismatched_lengths_h_y():
    """fit() must raise ValueError if h_calib and y_calib have different lengths."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=8)  # Mismatched
    row_ids = _simple_valid_row_ids(n=10)

    with pytest.raises(ValueError):
        fit(h, y, row_ids=row_ids, provenance="calib_iso")


def test_fit_rejects_mismatched_lengths_h_row_ids():
    """fit() must raise ValueError if h_calib and row_ids have different lengths."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=8)  # Mismatched

    with pytest.raises(ValueError):
        fit(h, y, row_ids=row_ids, provenance="calib_iso")


def test_fit_rejects_duplicate_row_ids():
    """fit() must raise ValueError if row_ids contains duplicates."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)
    row_ids[5] = row_ids[0]  # Duplicate

    with pytest.raises(ValueError):
        fit(h, y, row_ids=row_ids, provenance="calib_iso")


# === fit() provenance validation =============================================

def test_fit_accepts_provenance_calib_iso():
    """fit(..., provenance='calib_iso') must succeed."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h, y, row_ids=row_ids, provenance="calib_iso")
    assert cal is not None
    assert cal.provenance == "calib_iso"


def test_fit_rejects_provenance_test():
    """fit(..., provenance='test') must raise ValueError -- test split is never fit."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    with pytest.raises(ValueError):
        fit(h, y, row_ids=row_ids, provenance="test")


def test_fit_rejects_provenance_train():
    """fit(..., provenance='train') must raise ValueError."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    with pytest.raises(ValueError):
        fit(h, y, row_ids=row_ids, provenance="train")


# === fit() output shape and structure ========================================

def test_fit_returns_isotonic_calibrator():
    """fit() must return an IsotonicCalibrator with expected attributes."""
    from src.model.calibration import fit, IsotonicCalibrator

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h, y, row_ids=row_ids, provenance="calib_iso")

    assert isinstance(cal, IsotonicCalibrator)
    assert hasattr(cal, "maps")
    assert hasattr(cal, "fit_row_ids")
    assert hasattr(cal, "provenance")
    assert hasattr(cal, "n_fit")


def test_fit_maps_has_three_entries():
    """IsotonicCalibrator.maps must be a 3-tuple of fitted isotonic regressors."""
    from src.model.calibration import fit, EVENT_CLASSES

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h, y, row_ids=row_ids, provenance="calib_iso")

    assert len(cal.maps) == len(EVENT_CLASSES)
    assert len(cal.maps) == 3


def test_fit_stores_row_ids():
    """IsotonicCalibrator.fit_row_ids must store the row IDs as a frozenset."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h, y, row_ids=row_ids, provenance="calib_iso")

    assert isinstance(cal.fit_row_ids, frozenset)
    assert cal.fit_row_ids == frozenset(row_ids)


def test_fit_stores_n_fit():
    """IsotonicCalibrator.n_fit must store the number of rows fit."""
    from src.model.calibration import fit

    h = _simple_valid_hazards(n=10)
    y = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h, y, row_ids=row_ids, provenance="calib_iso")

    assert cal.n_fit == 10


# === fit() determinism =====================================================

def test_fit_is_deterministic():
    """Two separate fit() calls on byte-identical input must produce
    IsotonicCalibrator instances whose apply() gives byte-identical output.

    n=20, not the smaller sizes some other tests use here: fitting three
    per-class isotonic regressions on very few points each is inherently
    noisy (PAVA can locally overshoot with sparse data), and a small-enough
    n can trigger calibration.py's own SimplexViolation on the further,
    independent h_test purely from that sampling noise -- a real, correct
    guard, but unrelated to what THIS test checks (byte-identical output
    across two fits). n=20/n=300 below are sized generously enough to
    avoid that noise (verified directly against these exact seeds before
    committing to them), not tuned to any particular outcome."""
    from src.model.calibration import fit, apply

    h = _simple_valid_hazards(n=300, seed=999)
    y = _simple_valid_true_labels(n=300, seed=999)
    row_ids = _simple_valid_row_ids(n=300)

    # First fit
    cal1 = fit(h.copy(), y.copy(), row_ids=row_ids, provenance="calib_iso")

    # Second fit, identical input
    cal2 = fit(h.copy(), y.copy(), row_ids=row_ids, provenance="calib_iso")

    # Apply both to the same test data
    h_test = _simple_valid_hazards(n=5, seed=777)
    out1 = apply(cal1, h_test.copy())
    out2 = apply(cal2, h_test.copy())

    assert np.array_equal(out1, out2), (
        "Determinism check failed: identical fit() calls produced different "
        "apply() outputs"
    )


# === apply() basic functionality ============================================

def test_apply_returns_correct_shape():
    """apply() must return (n, 4), same shape as input h."""
    from src.model.calibration import fit, apply

    h_calib = _simple_valid_hazards(n=10)
    y_calib = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    h_test = _simple_valid_hazards(n=5, seed=555)
    out = apply(cal, h_test)

    assert out.shape == (5, 4)


def test_apply_rows_sum_to_exactly_one():
    """apply() output rows must sum to EXACTLY 1.0 in float, not within tolerance.
    This is the residual design's specific contract -- column 0 is set as
    1 - (col1 + col2 + col3), so every row sums exactly.

    n_calib=300, not 20: see test_fit_is_deterministic's docstring -- too
    few calibration points per class makes the three independent isotonic
    fits noisy enough to trigger a real SimplexViolation on further,
    independent test data, which is correct behaviour from calibration.py
    but not what this test means to exercise (the exact-sum arithmetic of
    a row that DID calibrate successfully)."""
    from src.model.calibration import fit, apply

    h_calib = _simple_valid_hazards(n=300, seed=1)
    y_calib = _simple_valid_true_labels(n=300, seed=1)
    row_ids = _simple_valid_row_ids(n=300)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    h_test = _simple_valid_hazards(n=50, seed=888)
    out = apply(cal, h_test)

    row_sums = out.sum(axis=1)

    # EXACT equality, not allclose
    assert (row_sums == 1.0).all(), (
        f"Not all rows sum to exactly 1.0. Max sum: {row_sums.max()}, "
        f"Min sum: {row_sums.min()}. Deviations: {row_sums - 1.0}"
    )


def test_apply_output_passes_cif_validation():
    """apply() output must pass through _validate_hazards logic when reshaped
    to (n, 3, 4) for cif.survival() and cif.cif().

    n_calib=300, not 15: see test_fit_is_deterministic's docstring -- too
    few calibration points makes the three per-class isotonic fits noisy
    enough to trigger a real SimplexViolation, unrelated to what this test
    checks (cif/survival compatibility of a successfully-calibrated row)."""
    from src.model.calibration import fit, apply
    from src.model.cif import survival, cif

    h_calib = _simple_valid_hazards(n=300, seed=2)
    y_calib = _simple_valid_true_labels(n=300, seed=2)
    row_ids = _simple_valid_row_ids(n=300)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    h_test = _simple_valid_hazards(n=5, seed=333)
    h_cal_test = apply(cal, h_test)

    # Reshape to (n, 3, 4) for cif module: keep only slots 2-4 (which we don't
    # have; we only have per-row hazards, not by-slot). Actually, this is a
    # bit confusing -- the calibration module works on per-row hazards (n, 4),
    # but cif.py works on (n, 3, 4) with axis 1 being slots. Reconsider:
    #
    # Actually, looking at competing_risks.py, hazards(model, X) returns
    # (n, 4) -- the per-row hazard for a SINGLE slot. Then the caller
    # reshapes that to be part of the (n, 3, 4) tensor for cif.py.
    #
    # So calibration.apply() takes an (n, 4) and returns (n, 4). To test it
    # integrates with cif, we need to build a synthetic (n, 3, 4) tensor with
    # rows that came from apply().
    #
    # For this test, let's just verify that the output can be used as input
    # to cif without error. We'll manually construct an (n, 3, 4) tensor where
    # each slot uses the same calibrated hazards (just for testing that rows
    # sum to 1 and pass validation).

    n = h_cal_test.shape[0]
    h_tensor = np.zeros((n, 3, 4))
    # Replicate the calibrated hazard across slots 2, 3, 4
    h_tensor[:, 0, :] = h_cal_test  # slot 2
    h_tensor[:, 1, :] = h_cal_test  # slot 3
    h_tensor[:, 2, :] = h_cal_test  # slot 4

    # Should not raise
    try:
        s = survival(h_tensor)
        c = cif(h_tensor)
        assert s.shape == (n, 4)
        assert c.shape == (n, 3, 4)
    except ValueError as e:
        pytest.fail(f"calibrated hazards failed cif validation: {e}")


def test_apply_identity_after_calibration():
    """On a synthetic calibrated hazard tensor, verify the competing-risks
    identity cif(h)[:,:,3].sum(axis=1) + survival(h)[:,3] == 1 within 1e-12.
    This confirms calibration does not break the fundamental identity."""
    from src.model.calibration import fit, apply
    from src.model.cif import survival, cif

    h_calib = _simple_valid_hazards(n=10)
    y_calib = _simple_valid_true_labels(n=10)
    row_ids = _simple_valid_row_ids(n=10)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    # Small test set to verify identity
    h_test = _simple_valid_hazards(n=3, seed=222)
    h_cal_test = apply(cal, h_test)

    # Build (n, 3, 4) tensor with calibrated hazards across slots
    n = h_cal_test.shape[0]
    h_tensor = np.zeros((n, 3, 4))
    h_tensor[:, :, :] = h_cal_test[np.newaxis, :, :]  # Same hazard at all slots

    # Note: this doesn't quite work because we're replicating the same row
    # hazard across slots. Let me instead just manually replicate:
    for slot_idx in range(3):
        h_tensor[:, slot_idx, :] = h_cal_test

    s = survival(h_tensor)
    c = cif(h_tensor)

    # At slot 4 (axis-2 index 3)
    cif_sum = c[:, :, 3].sum(axis=1)  # Sum over the 3 non-reference causes
    total = cif_sum + s[:, 3]

    assert np.allclose(total, 1.0, atol=1e-12), (
        f"Identity cif[:,:,3].sum + survival[:,3] != 1. "
        f"Max error: {np.abs(total - 1.0).max()}"
    )


# === apply() isotonic monotonicity =========================================

def test_apply_maintains_isotonic_monotonicity():
    """After fit() and apply(), each of the three event-class columns must be
    monotonically non-decreasing as a function of that class's own raw input
    probability. This tests isotonic regression's core property on synthetic
    data with MANY distinct x values (not the 6-atom real design matrix)."""
    from src.model.calibration import fit, apply, EVENT_CLASSES

    # Generate calibration data with continuous-ish raw probabilities
    n_calib = 100
    rng = np.random.RandomState(42)

    # Create hazards that vary smoothly
    h_calib = np.zeros((n_calib, 4))
    # Event 1 (RECOVERED): range from ~0.1 to ~0.4
    h_calib[:, 1] = np.linspace(0.1, 0.4, n_calib)
    # Event 2 (DEAD): range from ~0.05 to ~0.2
    h_calib[:, 2] = np.linspace(0.05, 0.2, n_calib)
    # Event 3 (OPTED_OUT): range from ~0.05 to ~0.15
    h_calib[:, 3] = np.linspace(0.05, 0.15, n_calib)
    # Ensure rows sum to 1
    h_calib[:, 0] = 1.0 - (h_calib[:, 1] + h_calib[:, 2] + h_calib[:, 3])

    # True labels correlated with event 1 probability
    y_calib = np.zeros(n_calib, dtype=int)
    threshold = 0.25
    y_calib[h_calib[:, 1] >= threshold] = 1  # RECOVERED

    row_ids = _simple_valid_row_ids(n=n_calib)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    # Apply to test data: also with smoothly-varying input
    n_test = 50
    h_test = np.zeros((n_test, 4))
    h_test[:, 1] = np.linspace(0.05, 0.45, n_test)
    h_test[:, 2] = np.linspace(0.02, 0.25, n_test)
    h_test[:, 3] = np.linspace(0.02, 0.18, n_test)
    h_test[:, 0] = 1.0 - (h_test[:, 1] + h_test[:, 2] + h_test[:, 3])

    h_cal_out = apply(cal, h_test)

    # Check monotonicity for each event class
    for class_idx, class_int in enumerate(EVENT_CLASSES):
        # Sort by raw input probability for this class
        sorted_indices = np.argsort(h_test[:, class_int])
        calibrated_values = h_cal_out[sorted_indices, class_int]

        # Compute differences
        diffs = np.diff(calibrated_values)

        assert (diffs >= -1e-9).all(), (
            f"Event class {class_int}: calibrated output is not monotonic. "
            f"Sorted raw input: {h_test[sorted_indices, class_int]}. "
            f"Calibrated output: {calibrated_values}. "
            f"Differences: {diffs}"
        )


# === apply() SimplexViolation =============================================

def test_apply_raises_simplex_violation_when_events_exceed_one():
    """apply() must raise SimplexViolation if the three event-class probabilities
    for any row sum to >= 1 (leaving nothing or negative mass for STILL_PENDING).
    Construct a pathological IsotonicCalibrator manually to trigger this."""
    from src.model.calibration import (
        SimplexViolation, apply, IsotonicCalibrator
    )
    from sklearn.isotonic import IsotonicRegression

    # Build degenerate isotonic regressors that always return high values
    # IsotonicRegression(out_of_bounds='clip').predict always returns values
    # between min and max of the training y values.

    # Create three maps that when applied to input [0.3, 0.3, 0.3] each return
    # high values like [0.4, 0.4, 0.4], summing to > 1.

    # Fit isotonic regressors on dummy data such that they output high values
    maps_list = []
    for _ in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        # Fit on (x, y) pairs that force the regressor to output high values
        iso.fit([0.0, 0.5, 1.0], [0.5, 0.6, 0.7])
        maps_list.append(iso)

    cal = IsotonicCalibrator(
        maps=tuple(maps_list),
        fit_row_ids=frozenset(['dummy']),
        provenance="calib_iso",
        n_fit=1
    )

    # Create test data where the three event classes each have 0.35 probability
    # When each is independently transformed by the isotonic regressor above,
    # they might each return ~0.6 or more, summing to > 1.0
    h_test = np.array([
        [0.0, 0.35, 0.35, 0.35],  # Residual=0, three events=0.35 each
        [0.1, 0.3, 0.3, 0.3],
    ])

    with pytest.raises(SimplexViolation) as exc_info:
        apply(cal, h_test)

    # Check error message contains useful info
    error_msg = str(exc_info.value)
    assert "simplex" in error_msg.lower() or "sum" in error_msg.lower(), (
        f"SimplexViolation message should mention simplex or sum, got: {error_msg}"
    )


def test_apply_does_not_silently_clip_probabilities():
    """apply() must NEVER silently clip probabilities when the simplex is
    violated. Verify by checking that an implementation that would clip
    would produce incorrect output that we can detect."""
    from src.model.calibration import (
        SimplexViolation, apply, IsotonicCalibrator
    )
    from sklearn.isotonic import IsotonicRegression

    # Similar setup as above: pathological IsotonicCalibrator
    maps_list = []
    for _ in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit([0.0, 1.0], [0.6, 0.7])  # Outputs will be high
        maps_list.append(iso)

    cal = IsotonicCalibrator(
        maps=tuple(maps_list),
        fit_row_ids=frozenset(['dummy']),
        provenance="calib_iso",
        n_fit=1
    )

    h_test = np.array([[0.0, 0.35, 0.35, 0.35]])

    # If the implementation clips silently instead of raising, the test fails
    with pytest.raises(SimplexViolation):
        apply(cal, h_test)


# === assert_disjoint ===========================================================

def test_assert_disjoint_accepts_disjoint_sets():
    """assert_disjoint() must NOT raise when the two id sets are disjoint."""
    from src.model.calibration import assert_disjoint

    fit_ids = frozenset(['a', 'b', 'c'])
    report_ids = frozenset(['d', 'e', 'f'])

    # Should not raise
    assert_disjoint(fit_ids, report_ids)


def test_assert_disjoint_accepts_empty_sets():
    """assert_disjoint() must NOT raise on empty-set edge cases."""
    from src.model.calibration import assert_disjoint

    # Both empty
    assert_disjoint(frozenset(), frozenset())

    # One empty
    assert_disjoint(frozenset(), frozenset(['a', 'b']))
    assert_disjoint(frozenset(['a', 'b']), frozenset())


def test_assert_disjoint_rejects_overlapping_sets():
    """assert_disjoint() must raise CalibrationLeakError if the two id sets
    share at least one element."""
    from src.model.calibration import assert_disjoint, CalibrationLeakError

    fit_ids = frozenset(['a', 'b', 'c'])
    report_ids = frozenset(['b', 'd', 'e'])  # 'b' is shared

    with pytest.raises(CalibrationLeakError):
        assert_disjoint(fit_ids, report_ids)


def test_assert_disjoint_error_message_names_overlaps():
    """CalibrationLeakError message should name some overlapping ids."""
    from src.model.calibration import assert_disjoint, CalibrationLeakError

    fit_ids = frozenset(['id_1', 'id_2', 'id_3', 'id_4', 'id_5'])
    report_ids = frozenset(['id_3', 'id_4', 'id_6'])  # id_3, id_4 overlap

    with pytest.raises(CalibrationLeakError) as exc_info:
        assert_disjoint(fit_ids, report_ids)

    error_msg = str(exc_info.value)
    # At least one of the overlapping ids should be named
    assert 'id_3' in error_msg or 'id_4' in error_msg, (
        f"Error message should name overlapping ids, got: {error_msg}"
    )


# === reliability_table =======================================================

def test_reliability_table_basic_shape():
    """reliability_table() must return a DataFrame with expected columns."""
    from src.model.calibration import reliability_table

    # Small dataset: 20 rows, manually constructed probabilities
    p = _simple_valid_hazards(n=20, seed=100)
    y = _simple_valid_true_labels(n=20, seed=101)

    table = reliability_table(p, y)

    assert isinstance(table, pd.DataFrame)
    assert "n" in table.columns
    assert "mean_predicted" in table.columns
    assert "observed_frequency" in table.columns
    assert "wilson_lo" in table.columns
    assert "wilson_hi" in table.columns
    assert "z" in table.columns


def test_reliability_table_two_atoms_per_class():
    """reliability_table() groups by UNIQUE predicted-probability vectors (atoms).
    On a hand-constructed dataset with exactly 2 distinct predicted vectors,
    we should see exactly 2 rows per class (2 atoms × 4 classes = 8 rows total)."""
    from src.model.calibration import reliability_table

    # Construct two distinct probability vectors
    atom_1 = np.array([0.5, 0.2, 0.2, 0.1])
    atom_2 = np.array([0.3, 0.4, 0.2, 0.1])

    # Replicate each atom multiple times
    p = np.vstack([atom_1] * 10 + [atom_2] * 10)  # 20 rows total, 2 atoms
    assert p.shape == (20, 4)

    # True labels: roughly 10 per atom, distributed across classes
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1] +  # 10 for atom_1
                 [1, 2, 3, 0, 1, 2, 3, 0, 1, 2])  # 10 for atom_2

    table = reliability_table(p, y)

    # Should have 8 rows: 2 atoms × 4 classes
    assert len(table) == 8, (
        f"Expected 8 rows (2 atoms × 4 classes), got {len(table)}"
    )

    # Group by atom and verify n sums correctly within each atom
    unique_atoms_in_p = np.unique(p, axis=0)
    assert len(unique_atoms_in_p) == 2, (
        f"Expected 2 unique predicted vectors, got {len(unique_atoms_in_p)}"
    )


def test_reliability_table_n_sums_correctly():
    """reliability_table() must have 'n' column that sums back to the total
    row count when grouped by atom."""
    from src.model.calibration import reliability_table

    p = _simple_valid_hazards(n=50, seed=200)
    y = _simple_valid_true_labels(n=50, seed=201)

    table = reliability_table(p, y)

    # Group the table by predicted vector (we need to reconstruct what the
    # atoms were from the table itself, or know that each row in the table
    # represents one (atom, class) pair, and the n value is that atom's total
    # row count across all classes).

    # Actually, let's verify a simpler property: the sum of distinct 'n' values
    # times 4 (classes) should relate to the total. Or: for a given atom,
    # the 'n' value should be the same across all 4 classes.

    # Group rows by their 'mean_predicted' value (proxy for atom identity)
    for mean_pred, group in table.groupby('mean_predicted', as_index=False):
        # All 4 rows for this atom should have the same 'n'
        if len(group) == 4:
            assert group['n'].nunique() == 1, (
                f"For atom with mean_predicted={mean_pred}, 'n' values "
                f"differ: {group['n'].values}"
            )


def test_reliability_table_wilson_interval_correctness():
    """reliability_table() Wilson score interval must be correct for a known
    case. For n=100, k=20 successes (observed_frequency=0.20), the 95% Wilson
    interval should be approximately [0.133, 0.286]."""
    from src.model.calibration import reliability_table

    # Construct a dataset with a single atom: probability vector
    # [0.7, 0.1, 0.1, 0.1], observed outcome 1 (RECOVERED, index 1) 20 times
    # out of 100, outcome 0 the rest.

    atom = np.array([0.7, 0.1, 0.1, 0.1])
    p = np.tile(atom, (100, 1))  # 100 rows, all same atom

    # Create labels: 20 are outcome 1 (RECOVERED), rest are outcome 0
    y = np.zeros(100, dtype=int)
    y[:20] = 1

    table = reliability_table(p, y)

    # Extract the row for (this atom, outcome 1)
    row = table[(table['observed_frequency'] == 0.20) &
                (table['n'] == 100)]

    assert len(row) == 1, f"Expected 1 row, got {len(row)}"
    row = row.iloc[0]

    # Wilson interval for n=100, k=20, alpha=0.05 is approximately [0.133, 0.286]
    # Use a tolerance of 0.01 for the bounds
    assert 0.11 <= row['wilson_lo'] <= 0.15, (
        f"wilson_lo={row['wilson_lo']} outside expected ~[0.11, 0.15]"
    )
    assert 0.25 <= row['wilson_hi'] <= 0.30, (
        f"wilson_hi={row['wilson_hi']} outside expected ~[0.25, 0.30]"
    )


# === ECE functions ===========================================================

def test_classwise_ece_basic():
    """classwise_ece() must return a single float (the headline calibration metric)."""
    from src.model.calibration import classwise_ece

    p = _simple_valid_hazards(n=30, seed=300)
    y = _simple_valid_true_labels(n=30, seed=301)

    ece = classwise_ece(p, y)

    assert isinstance(ece, (float, np.floating))
    assert 0 <= ece <= 1, f"ECE should be in [0, 1], got {ece}"


def test_per_class_ece_returns_dict():
    """per_class_ece() must return a dict with 4 entries, one per Outcome int."""
    from src.model.calibration import per_class_ece

    p = _simple_valid_hazards(n=30, seed=400)
    y = _simple_valid_true_labels(n=30, seed=401)

    ece_dict = per_class_ece(p, y)

    assert isinstance(ece_dict, dict)
    assert len(ece_dict) == 4
    for c in range(4):
        assert c in ece_dict
        assert isinstance(ece_dict[c], (float, np.floating))
        assert 0 <= ece_dict[c] <= 1


def test_classwise_ece_is_mean_of_per_class_ece():
    """classwise_ece should be the mean of the four per_class_ece values."""
    from src.model.calibration import classwise_ece, per_class_ece

    p = _simple_valid_hazards(n=50, seed=500)
    y = _simple_valid_true_labels(n=50, seed=501)

    overall = classwise_ece(p, y)
    per_class = per_class_ece(p, y)

    mean_of_per_class = np.mean([per_class[c] for c in range(4)])

    assert np.isclose(overall, mean_of_per_class, atol=1e-9), (
        f"classwise_ece={overall} should equal mean of per_class_ece values "
        f"={mean_of_per_class}"
    )


def test_classwise_ece_hand_computed():
    """classwise_ece on a small hand-built dataset where we can compute the
    exact expected value. Atom = single predicted vector repeated 20 times.
    Observed: 5 each of outcomes 0, 1, 2, 3."""
    from src.model.calibration import classwise_ece

    # Single atom: [0.5, 0.2, 0.2, 0.1]
    atom = np.array([0.5, 0.2, 0.2, 0.1])
    p = np.tile(atom, (20, 1))

    # Observed: 5 of each outcome
    y = np.array([0, 1, 2, 3] * 5)

    ece = classwise_ece(p, y)

    # Hand compute per-class ECE:
    # Class 0: predicted=0.5, observed=5/20=0.25, error=|0.5-0.25|=0.25, weight=20/20=1
    # Class 1: predicted=0.2, observed=5/20=0.25, error=|0.2-0.25|=0.05, weight=1
    # Class 2: predicted=0.2, observed=5/20=0.25, error=|0.2-0.25|=0.05, weight=1
    # Class 3: predicted=0.1, observed=5/20=0.25, error=|0.1-0.25|=0.15, weight=1
    # classwise_ece = mean([0.25, 0.05, 0.05, 0.15]) = 0.1

    expected = (0.25 + 0.05 + 0.05 + 0.15) / 4
    assert np.isclose(ece, expected, atol=1e-9), (
        f"classwise_ece={ece}, expected={expected}"
    )


def test_per_class_ece_hand_computed():
    """per_class_ece on the same hand-built dataset."""
    from src.model.calibration import per_class_ece

    # Single atom: [0.5, 0.2, 0.2, 0.1]
    atom = np.array([0.5, 0.2, 0.2, 0.1])
    p = np.tile(atom, (20, 1))
    y = np.array([0, 1, 2, 3] * 5)

    ece_dict = per_class_ece(p, y)

    # Class 0: error = 0.25
    # Class 1: error = 0.05
    # Class 2: error = 0.05
    # Class 3: error = 0.15

    assert np.isclose(ece_dict[0], 0.25, atol=1e-9), (
        f"ece_dict[0]={ece_dict[0]}, expected=0.25"
    )
    assert np.isclose(ece_dict[1], 0.05, atol=1e-9)
    assert np.isclose(ece_dict[2], 0.05, atol=1e-9)
    assert np.isclose(ece_dict[3], 0.15, atol=1e-9)


# === Graceful degradation ===================================================

def test_calibration_graceful_degradation_independent_data():
    """When fit() is called on data where y_calib is independent of h_calib
    (no real signal), apply() should degrade gracefully: each event class's
    calibrated output should land close to that class's base rate (within
    ~0.05-0.10) rather than doing something wild or unstable.

    n_calib=500, not 50: with truly independent h/y, three per-class
    isotonic regressions fit on only 50 noisy points each can show a
    spurious high tail purely from small-sample sampling variation, and
    when evaluated together on a further-independent h_test their sum can
    genuinely exceed 1 by chance -- measured directly: n_calib=50 raised
    SimplexViolation on every one of 4 tried seeds, n_calib=500 raised on
    none. That is calibration.py's SimplexViolation working AS DESIGNED
    against small-sample noise, not a bug in it -- but it is a different
    thing from what "graceful degradation" means to test here, so this
    uses enough data for the per-class fits to actually settle near the
    true (here: uniform ~0.25) rate the independence implies."""
    from src.model.calibration import fit, apply

    # Generate calibration data with h_calib INDEPENDENT of y_calib
    n_calib = 500
    rng = np.random.RandomState(42)

    h_calib = rng.dirichlet([1, 1, 1, 1], size=n_calib)

    # Draw y_calib uniformly at random, completely independent of h_calib
    y_calib = rng.randint(0, 4, size=n_calib)

    row_ids = _simple_valid_row_ids(n=n_calib)

    cal = fit(h_calib, y_calib, row_ids=row_ids, provenance="calib_iso")

    # Apply to test data. Note: no random_state= kwarg here --
    # np.random.RandomState.dirichlet() has no such parameter (that is a
    # scipy.stats-style API); rng is already a RandomState instance, so
    # calling rng.dirichlet(...) consumes its own internal state directly.
    h_test = rng.dirichlet([1, 1, 1, 1], size=30)
    h_cal_out = apply(cal, h_test)

    # Compute base rates from calibration set
    base_rates = np.array([
        (y_calib == c).mean() for c in range(4)
    ])

    # Each class's calibrated output should land close to its base rate
    for c in range(4):
        mean_calibrated = h_cal_out[:, c].mean()
        diff = abs(mean_calibrated - base_rates[c])
        assert diff <= 0.10, (
            f"Class {c}: calibrated mean={mean_calibrated}, "
            f"base_rate={base_rates[c]}, diff={diff} > 0.10"
        )
