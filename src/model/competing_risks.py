"""Multinomial logit cause-specific hazard model over person-period data.

Design decision this file pins: the model is discrete-time competing-risks
(a single MNLogit predicting all 4 outcomes at once), never a binary "did it
recover" target, and is fit on estimable slots 2-4 only -- slot 1 is a
structural zero (src/model/person_period.py's `estimable` flag), never part
of the estimation set. `fit()` filters `df[df.estimable]` INTERNALLY; the
caller need not pre-filter (and per a real B4 incident, must not re-derive
this filter as `df.slot >= 2` independently -- one flag, not two definitions
that can drift apart).

`assemble()` is the ONLY place the target (`event_code`, `estimable`) meets
the feature frame: `featurize()` physically strips those columns, so this
join by `row_id` is required, not optional -- see DECISIONS.md, 2026-08-28,
B4 finding 1.

The design matrix is built identically at fit time and predict time by one
shared private function, `_design_matrix()`, which constructs every column
explicitly (`slot == 3`, `slot == 4`, ...) rather than via `pd.get_dummies`.
`get_dummies` only emits a column for a category actually present in a given
batch -- an all-slot-2 prediction batch would silently drop `slot_3`/
`slot_4` and misalign the fitted coefficient vector. `HazardModel` carries
`feature_columns`, the exact ordered subset of `_design_matrix()`'s output
the fit used (all six columns for a full fit, just `const` for
`intercept_only=True`), so `hazards()` can always reconstruct the identical
matrix a given model was fit against, regardless of what the model was.

Design matrix -- the final, nominal-arm-only spec, as amended 2026-08-28 by
stats-reviewer finding 4 (DECISIONS.md, B5 stats-reviewer entry):
`FEATURE_COLUMNS` (what `fit()` uses by default) is `const`, `slot_3`,
`slot_4` (slot 2 is the reference level), `in_salary_window`. Two columns
present in an earlier version of this design -- `days_since_last_attempt`
and `slot3_x_in_salary_window` -- are excluded from `FEATURE_COLUMNS`.

CAVEAT, added by the B5 stats-reviewer CONFIRMING pass, correcting the
original reasoning above (do not repeat the original argument as
precedent -- it is a category error): "the frozen simulator's `_draw_
outcome` sets these coefficients to exactly zero" is true only CONDITIONAL
on the latent cause, which this model never observes. It fits the
CAUSE-MARGINAL hazard, and the risk set's cause mix shifts by slot
(CANT_PAY_NOW mandates resolve and exit, enriching later slots in WONT_PAY
/ CANT_PAY_EVER) -- a marginal model can show a genuinely non-zero
coefficient on a term that is exactly zero within every latent stratum.
Confirmed directly: within-latent-cause fits keep both terms at |z| <= 1.45,
but the pooled marginal fit on the 40-seed corpus puts
`slot3_x_in_salary_window`->OPTED_OUT at z=2.85 -- not the noise the
original reasoning claimed. The original claim that dropping these two
columns "improves the held-out margin" is also false: over a stability
sweep, the 4-column and 6-column designs differ by a mean of -0.00002 (a
coin flip, 4-col better on 11/20 seeds). The correct, defensible reason to
exclude them is EMPIRICAL NEUTRALITY plus parsimony/DGP-consistency, not "we
measured they were noise and it helped." Do not cite the original
z<=1.34/"improves the margin" framing when B8 next touches this design
matrix -- rederive it, since a cause-marginal model's zero-coefficient
claims need the within-stratum caveat above, not the frozen simulator's
per-cause `_draw_outcome` logic read directly.

`_design_matrix()` still computes all six possible columns (unchanged) so
`HazardModel.feature_columns` can select a different subset if a future
caller needs one -- only the DEFAULT `FEATURE_COLUMNS` used by
`fit(intercept_only=False)` changed.

Still excluded, per the original design and B4's decisions:
`committed_day_of_month` (collinear with `days_since_last_attempt` --
identical on all slot-2 rows, condition number 144 against the six-column
design), `prior_failures_this_cycle` (exactly `slot - 1`, collinear with
the slot dummies), `profile` (constant per call, collinear with the
intercept), `amount_paise`/`category`/`above_afa_cliff` (no true hazard
signal under `nominal` -- `_draw_outcome` never reads them on this arm).

Evaluation functions (`log_loss`, `brier_per_cause`, `calibration_table`)
assume the caller has already selected `df[df.estimable]` -- they raise
`ValueError` if any `estimable=False` row is present, the same discipline as
`fit()`, made explicit rather than silently filtered a second way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURE_COLUMNS: tuple[str, ...] = ("const", "slot_3", "slot_4", "in_salary_window")
INTERCEPT_ONLY_COLUMNS: tuple[str, ...] = ("const",)
# _design_matrix() always computes all six of these; FEATURE_COLUMNS selects
# the four `fit()` actually uses by default -- see module docstring.
_ALL_DESIGN_COLUMNS: tuple[str, ...] = (
    "const", "slot_3", "slot_4", "in_salary_window",
    "days_since_last_attempt", "slot3_x_in_salary_window",
)


@dataclass(frozen=True)
class HazardModel:
    """Wraps a fitted statsmodels MNLogitResults plus the exact ordered
    design-matrix columns it was fit against -- see module docstring."""
    result: object
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    required = {"slot", "in_salary_window", "days_since_last_attempt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"design matrix input is missing required column(s): {sorted(missing)}"
        )
    out = pd.DataFrame(index=df.index)
    out["const"] = 1.0
    slot = df["slot"].astype("int64")
    out["slot_3"] = (slot == 3).astype(float)
    out["slot_4"] = (slot == 4).astype(float)
    out["in_salary_window"] = df["in_salary_window"].astype(float)
    out["days_since_last_attempt"] = df["days_since_last_attempt"].astype(float)
    out["slot3_x_in_salary_window"] = out["slot_3"] * out["in_salary_window"]
    return out[list(_ALL_DESIGN_COLUMNS)]


def assemble(pp_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Join `feat_df` (a `featurize()` output -- no outcome columns) with
    `pp_df` (a `person_period.build()` output -- has `event_code`,
    `estimable`) on `row_id`. Raises ValueError if the two frames' `row_id`
    sets are not identical, or if either has a duplicate `row_id` -- a
    mismatched pair here would otherwise silently corrupt every downstream
    hazard."""
    if pp_df["row_id"].duplicated().any():
        raise ValueError("assemble() pp_df has duplicate row_id values")
    if feat_df["row_id"].duplicated().any():
        raise ValueError("assemble() feat_df has duplicate row_id values")

    pp_ids = set(pp_df["row_id"])
    feat_ids = set(feat_df["row_id"])
    if pp_ids != feat_ids:
        extra_in_feat = sorted(feat_ids - pp_ids)
        extra_in_pp = sorted(pp_ids - feat_ids)
        raise ValueError(
            "assemble() requires identical row_id sets between pp_df and "
            f"feat_df -- extra in feat_df: {extra_in_feat[:5]}"
            f"{'...' if len(extra_in_feat) > 5 else ''}; extra in pp_df: "
            f"{extra_in_pp[:5]}{'...' if len(extra_in_pp) > 5 else ''}"
        )

    target = pp_df[["row_id", "event_code", "estimable"]]
    return feat_df.merge(target, on="row_id", how="inner")


def fit(df: pd.DataFrame, *, intercept_only: bool = False) -> HazardModel:
    """Fit a multinomial logit over `df[df.estimable]` -- `event_code` as
    the 4-category target, `STILL_PENDING` (0) as the MNLogit reference
    category. `df` is `assemble()`'s full output (slot-1 rows included);
    this function filters internally, per the module docstring."""
    required = {"event_code", "estimable"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fit() input is missing required column(s): {sorted(missing)}")

    estimable_df = df[df["estimable"]]
    if len(estimable_df) == 0:
        raise ValueError(
            "fit() found no estimable rows after filtering df.estimable -- nothing to fit"
        )
    n_classes = estimable_df["event_code"].nunique()
    if n_classes != 4:
        raise ValueError(
            f"fit() requires all 4 event_code classes present in the "
            f"estimable rows (statsmodels derives its output column count "
            f"from what's actually observed, and hazards() assumes 4 -- see "
            f"its own shape check); found {n_classes} distinct value(s): "
            f"{sorted(estimable_df['event_code'].unique())}"
        )

    cols = INTERCEPT_ONLY_COLUMNS if intercept_only else FEATURE_COLUMNS
    design = _design_matrix(estimable_df)[list(cols)]
    target = estimable_df["event_code"].astype(int)

    result = sm.MNLogit(target, design).fit(disp=0)
    return HazardModel(result=result, feature_columns=cols)


def hazards(model: HazardModel, X: pd.DataFrame) -> np.ndarray:
    """Rebuild the identical design matrix `model` was fit against and
    return predicted probabilities, shape (len(X), 4), columns in Outcome
    int order [STILL_PENDING, RECOVERED, DEAD, OPTED_OUT], each row
    summing to 1."""
    design = _design_matrix(X)[list(model.feature_columns)]
    probs = np.asarray(model.result.predict(design))
    if probs.ndim != 2 or probs.shape != (len(X), 4):
        raise ValueError(
            f"hazards() expected shape ({len(X)}, 4) -- statsmodels derives "
            f"its column count from the distinct event_code values actually "
            f"present at fit time, so a fold missing an outcome class "
            f"silently returns fewer columns and misaligns Outcome int "
            f"order. Got shape {probs.shape}."
        )
    return probs


def _require_estimable_only(df: pd.DataFrame) -> None:
    if "estimable" not in df.columns:
        raise ValueError("input is missing required column: estimable")
    if not bool(df["estimable"].all()):
        raise ValueError(
            "input contains estimable=False row(s) -- slot-1 structural "
            "zeros are never a valid evaluation target; filter to "
            "df[df.estimable] before calling this function"
        )


def log_loss(model: HazardModel, df: pd.DataFrame) -> float:
    """Mean of -log(predicted probability of the true event_code), over
    `df`'s rows. Requires `df` already filtered to `estimable=True`."""
    _require_estimable_only(df)
    probs = hazards(model, df)
    true_idx = df["event_code"].astype(int).to_numpy()
    picked = probs[np.arange(len(df)), true_idx]
    return float(-np.log(picked).mean())


def brier_per_cause(model: HazardModel, df: pd.DataFrame) -> dict[int, float]:
    """Per-Outcome-int mean squared error of (predicted_prob - indicator).
    Requires `df` already filtered to `estimable=True`."""
    _require_estimable_only(df)
    probs = hazards(model, df)
    true_idx = df["event_code"].astype(int).to_numpy()
    out: dict[int, float] = {}
    for c in range(4):
        indicator = (true_idx == c).astype(float)
        out[c] = float(np.mean((probs[:, c] - indicator) ** 2))
    return out


def calibration_table(model: HazardModel, df: pd.DataFrame) -> pd.DataFrame:
    """Grouped by (slot, in_salary_window), one row per (cell, event_code)
    reporting n_rows (the cell's total row count), mean_predicted_prob, and
    realized_frequency for that event_code. Requires `df` already filtered
    to `estimable=True`."""
    _require_estimable_only(df)
    probs = hazards(model, df)

    base = df[["slot", "in_salary_window"]].reset_index(drop=True).copy()
    base["event_code"] = df["event_code"].astype(int).reset_index(drop=True).to_numpy()
    for c in range(4):
        base[f"_pred_{c}"] = probs[:, c]

    rows: list[dict] = []
    for (slot, in_window), group in base.groupby(["slot", "in_salary_window"], sort=True):
        n = len(group)
        for c in range(4):
            rows.append({
                "slot": slot,
                "in_salary_window": in_window,
                "event_code": c,
                "n_rows": n,
                "mean_predicted_prob": float(group[f"_pred_{c}"].mean()),
                "realized_frequency": float((group["event_code"] == c).mean()),
            })
    return pd.DataFrame(rows)
