"""Isotonic recalibration of per-row event hazards, and reliability
reporting over the atoms a hazard model like this project's actually
produces (a handful of distinct (slot, in_salary_window) covariate
combinations, not a continuous score).

Design decision this file pins: isotonic regression is fit INDEPENDENTLY
per event class (RECOVERED, DEAD, OPTED_OUT -- Outcome ints 1, 2, 3),
never on the residual STILL_PENDING class (Outcome int 0). The residual
absorbs whatever probability mass remains after the three event maps are
applied: `h_cal[:, 0] = 1 - (h_cal[:,1] + h_cal[:,2] + h_cal[:,3])`, and
the three event columns are NEVER rescaled or jointly renormalised.
`cif.py`'s recursion already treats column 0 this way -- it sums only
columns 1:4 when computing survival (cif.py's `s[:, k-1] = s[:, k-2] *
(1.0 - row_hazard_sum)`, `row_hazard_sum = h[:, k-2, 1:4].sum(...)`) -- so
this is not a compromise, it is making explicit a semantics the recursion
already assumes. The alternative (calibrate all four, then renormalise by
row sum) was measured to reach the same reliability improvement while
perturbing the row by up to ~28% before renormalising papers over it; this
design never renormalises what was actually calibrated.

Calibration is fit on ONE split, disjoint from whatever split any
reliability/coverage number is reported on -- `provenance="calib_iso"` is
the only value `fit()` accepts; `assert_disjoint()` is the machine-checked
guarantee that the ids used to fit are disjoint from the ids a report is
computed over, raising `CalibrationLeakError` rather than trusting the
caller.

No LLM import. No float money -- probabilities are the one place in this
codebase floats are correct, because a probability is not money.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# RECOVERED, DEAD, OPTED_OUT -- the three event classes isotonic is fit on.
# STILL_PENDING (0) is never fit; it is the residual (see module docstring).
EVENT_CLASSES: tuple[int, int, int] = (1, 2, 3)
RESIDUAL_CLASS: int = 0

_ACCEPTED_FIT_PROVENANCE = "calib_iso"

# 95% two-sided z for the Wilson score interval reliability_table() reports.
_WILSON_Z = 1.959963984540054


class CalibrationLeakError(RuntimeError):
    """Raised by assert_disjoint() when the ids used to fit a calibrator
    overlap with the ids a report is being computed over."""


class SimplexViolation(ValueError):
    """Raised by apply() if the three calibrated event-class probabilities
    for any row sum to >= 1, leaving no room for the residual STILL_PENDING
    class. Raised, never clipped -- silently clipping would fake a
    calibration guarantee that no longer holds for that row."""


@dataclass(frozen=True)
class IsotonicCalibrator:
    """maps: one fitted sklearn IsotonicRegression per EVENT_CLASSES member,
    same order. fit_row_ids: the row ids fit() was given, for leak-checking
    via assert_disjoint(). provenance: always "calib_iso" -- fit() rejects
    any other value. n_fit: the row count fit() was called with."""

    maps: tuple[IsotonicRegression, ...]
    fit_row_ids: frozenset[str]
    provenance: str
    n_fit: int


def fit(
    h_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    row_ids: Sequence[str],
    provenance: str,
) -> IsotonicCalibrator:
    """Fit one IsotonicRegression per EVENT_CLASSES member: x = that
    class's own raw probability (h_calib[:, c]), y = the binary indicator
    (y_calib == c). `provenance` must be "calib_iso" -- this is the one
    split reserved for fitting isotonic calibration (src/model/splits.py);
    fitting on "test" (or anything else) would let calibration and the
    number it is meant to fix share data, and is refused outright."""
    if provenance != _ACCEPTED_FIT_PROVENANCE:
        raise ValueError(
            f"calibration.fit() may only be called with "
            f"provenance={_ACCEPTED_FIT_PROVENANCE!r}; got {provenance!r} -- "
            "isotonic calibration must never be fit on the split its "
            "reliability number is reported on"
        )

    h_calib = np.asarray(h_calib, dtype=float)
    y_calib = np.asarray(y_calib)
    row_ids = list(row_ids)

    if len(h_calib) != len(y_calib):
        raise ValueError(
            f"fit() h_calib/y_calib length mismatch: {len(h_calib)} vs {len(y_calib)}"
        )
    if len(row_ids) != len(h_calib):
        raise ValueError(
            f"fit() row_ids length {len(row_ids)} does not match h_calib "
            f"length {len(h_calib)}"
        )
    dupes = [rid for rid, count in Counter(row_ids).items() if count > 1]
    if dupes:
        raise ValueError(f"fit() row_ids contains duplicate(s): {sorted(dupes)[:10]}")

    maps = []
    for c in EVENT_CLASSES:
        x = h_calib[:, c]
        target = (y_calib == c).astype(float)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, target)
        maps.append(iso)

    return IsotonicCalibrator(
        maps=tuple(maps),
        fit_row_ids=frozenset(row_ids),
        provenance=provenance,
        n_fit=len(h_calib),
    )


def apply(cal: IsotonicCalibrator, h: np.ndarray) -> np.ndarray:
    """(n, 4) in, (n, 4) out, same Outcome-int column order. Columns 1-3
    are each cal.maps[i].predict(h[:, EVENT_CLASSES[i]]) -- a monotone
    transform of ONLY that class's own raw probability, independent of the
    other two. Column 0 = 1 - (col1+col2+col3), so every row sums to
    exactly 1.0 in float -- never renormalised.

    Raises SimplexViolation if any row's three calibrated event
    probabilities sum to >= 1.0 (naming the offending row count and the
    max sum observed), rather than silently clipping."""
    h = np.asarray(h, dtype=float)
    n = h.shape[0]

    event_preds = np.empty((n, len(EVENT_CLASSES)), dtype=float)
    for idx, c in enumerate(EVENT_CLASSES):
        event_preds[:, idx] = cal.maps[idx].predict(h[:, c])

    event_sum = event_preds.sum(axis=1)
    # Strictly greater than 1 -- a row summing to EXACTLY 1.0 is a valid
    # boundary case (residual STILL_PENDING probability of exactly 0), not
    # a violation. Only a sum that would make the residual negative is one.
    violating = event_sum > 1.0
    if violating.any():
        raise SimplexViolation(
            f"apply() produced {int(violating.sum())} row(s) whose "
            f"calibrated event-class probabilities sum to > 1.0 (max sum "
            f"observed: {float(event_sum.max())}), leaving no room for the "
            "residual STILL_PENDING class -- refusing to silently clip"
        )

    out = np.empty((n, 4), dtype=float)
    out[:, EVENT_CLASSES[0]] = event_preds[:, 0]
    out[:, EVENT_CLASSES[1]] = event_preds[:, 1]
    out[:, EVENT_CLASSES[2]] = event_preds[:, 2]
    out[:, RESIDUAL_CLASS] = 1.0 - event_sum
    return out


def assert_disjoint(fit_ids: frozenset[str], report_ids: frozenset[str]) -> None:
    """Raise CalibrationLeakError if `fit_ids` and `report_ids` share any
    element -- the machine-checked half of "never fit and report on the
    same split"."""
    overlap = fit_ids & report_ids
    if overlap:
        raise CalibrationLeakError(
            f"{len(overlap)} id(s) used to fit a calibrator also appear in "
            f"the report set: {sorted(overlap)[:10]}"
        )


def _wilson_interval(k: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def reliability_table(p: np.ndarray, y: np.ndarray, *, n_classes: int = 4) -> pd.DataFrame:
    """Group rows of `p` by their UNIQUE PREDICTED VECTOR (atom-grouped,
    not equal-width binned -- this project's hazard model produces only a
    handful of distinct covariate combinations, so equal-width bins would
    leave most empty). One row per (atom, class): n (the atom's total row
    count), mean_predicted, observed_frequency, wilson_lo/wilson_hi (95%
    Wilson score interval on observed_frequency given n), z
    ((observed_frequency - mean_predicted) / se, se from a normal
    approximation under H0: true rate equals mean_predicted -- a
    diagnostic value, not itself a reported guarantee)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    n_rows = p.shape[0]

    atom_keys = [tuple(row) for row in p]
    unique_atoms = sorted(set(atom_keys), key=lambda a: a)

    rows: list[dict] = []
    for atom in unique_atoms:
        mask = np.array([k == atom for k in atom_keys])
        n = int(mask.sum())
        y_atom = y[mask]
        for c in range(n_classes):
            observed_frequency = float((y_atom == c).mean())
            mean_predicted = float(atom[c])
            wilson_lo, wilson_hi = _wilson_interval(int((y_atom == c).sum()), n)
            se = float(np.sqrt(mean_predicted * (1 - mean_predicted) / n)) if n > 0 else float("nan")
            z = (observed_frequency - mean_predicted) / se if se and se > 0 else 0.0
            rows.append({
                "n": n,
                "mean_predicted": mean_predicted,
                "observed_frequency": observed_frequency,
                "wilson_lo": wilson_lo,
                "wilson_hi": wilson_hi,
                "z": z,
                "class": c,
            })

    return pd.DataFrame(rows)


def per_class_ece(p: np.ndarray, y: np.ndarray, *, n_classes: int = 4) -> dict[int, float]:
    """Atom-weighted mean absolute calibration error, one float per class:
    sum over atoms of (n_atom / N) * |mean_predicted - observed_frequency|
    for that class."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    n_total = p.shape[0]

    atom_keys = [tuple(row) for row in p]
    unique_atoms = sorted(set(atom_keys), key=lambda a: a)

    out: dict[int, float] = {c: 0.0 for c in range(n_classes)}
    for atom in unique_atoms:
        mask = np.array([k == atom for k in atom_keys])
        n_atom = int(mask.sum())
        y_atom = y[mask]
        weight = n_atom / n_total
        for c in range(n_classes):
            observed_frequency = float((y_atom == c).mean())
            mean_predicted = float(atom[c])
            out[c] += weight * abs(mean_predicted - observed_frequency)
    return out


def classwise_ece(p: np.ndarray, y: np.ndarray, *, n_classes: int = 4) -> float:
    """The headline calibration metric: the mean, across all n_classes
    outcome classes, of per_class_ece's atom-weighted mean absolute
    calibration error."""
    per_class = per_class_ece(p, y, n_classes=n_classes)
    return float(np.mean(list(per_class.values())))
