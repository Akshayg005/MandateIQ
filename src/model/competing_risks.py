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

`_design_matrix()` still computes all twelve possible columns (unchanged in
spirit; widened 2026-09-04, R1) so `HazardModel.feature_columns` can select a
different subset if a future caller needs one -- only the DEFAULT
`FEATURE_COLUMNS` used by `fit(intercept_only=False)` changed... except it
didn't: see the next paragraph.

Still excluded from the DEFAULT `FEATURE_COLUMNS`, per the original design
and B4's decisions: `committed_day_of_month` (collinear with
`days_since_last_attempt` -- identical on all slot-2 rows, condition number
144 against the six-column design), `prior_failures_this_cycle` (exactly
`slot - 1`, collinear with the slot dummies), `profile` (constant per call,
collinear with the intercept).

`amount_paise`/`category` -- WIDENED (R1, 2026-09-04), not excluded, but
NOT the default either. `_design_matrix()` now also builds `amount_band_2`,
`amount_band_3`, `amount_band_4` (quartile cuts of
`eval/frozen/sim_config.yaml`'s own `below_afa_range` -- NOT "the only
range fit() trains on": that claim was checked and found FALSE by
stats-reviewer, 2026-09-04. `eval/corpus.py generate()` drops a mandate
above ITS OWN category's AFA limit, and the elevated categories carry a
higher one (clause 8(b), Rs 1,00,000) than the standard one these cuts are
quartiles of (clause 8(a), Rs 15,000) -- 316/7154 mandates (4.42%) in the
actual training sample exceed these cuts' Rs 500-14,000 range, up to a
measured Rs 89,785. `amount_band_4` is therefore a wide catch-all,
CONFOUNDED with category by the AFA filter itself: 25.8% of `subscription`
mandates land in it versus 36.6-37.3% of every elevated category. See
`reports/model_defensibility.md`'s Phase A section for the full disclosure
-- this does not change the null finding below, but these six columns are
not yet fit for an amount-independent-of-category defensibility claim) and
`category_insurance_premium`, `category_mutual_fund`,
`category_credit_card_bill` (the non-reference `category_mix` levels from
that same file). `WIDENED_FEATURE_COLUMNS` is `FEATURE_COLUMNS` plus those
six. This existed to answer a real defensibility question -- a reviewer
asking "why doesn't the hazard model use the mandate's own amount or
category at all" deserves a fitted, measured answer, not a design note --
and the measured answer is in `reports/model_defensibility.md`: under the
`nominal` arm (what `fit()`'s default trains on), `_draw_outcome`
(`eval/frozen/simulator.py`) never branches on `category` at all, and
branches on `amount_paise` only inside `coupled`'s household-balance
comparison, not in the base hazard logits. MEASURED
(`eval/design_matrix_comparison.py`, run 2026-09-04, PRIMARY test -- pooled
out-of-fold per-row log-loss differences clustered by mandate_id, the
properly-powered statistic; a naive 5-fold-MEAN t-test the first version of
that script used was underpowered at df=4 and did NOT clear 95%, corrected
after stats-reviewer caught it): all 18 of the new coefficients (6 columns
x 3 non-reference outcomes) have a 95% CI including zero, and
`WIDENED_FEATURE_COLUMNS` has WORSE held-out log-loss than
`FEATURE_COLUMNS` -- mean(widened - narrow) = +0.00103, clustered
SE = 0.00036, t = +2.88, df = 7153, p = 0.0040. That is the correct,
honest result of actually checking, not a symptom of a bad covariate
choice, and it is why `FEATURE_COLUMNS` (the default) stays narrow: adding
six provably-null parameters made the held-out fit MEASURABLY WORSE, not
merely no better -- a stronger, evidence-based version of the identical
"empirical neutrality plus parsimony" reason this file already excludes
`days_since_last_attempt` and `slot3_x_in_salary_window` above.
`eval/sim2.py` (non-frozen, R1 Phase B) is where amount- and
category-like covariates get a corpus that actually
varies outcomes by them.

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

# Amount-band cut points, in paise -- the quartiles of
# eval/frozen/sim_config.yaml's `below_afa_range: [50_000, 1_400_000]`
# (Rs 500-14,000): 50_000 + k*(1_400_000-50_000)/4 for k=1,2,3 -- derived
# from that range alone, not tuned against any fit result.
#
# NOT "the only range fit() sees": a claim to that effect here was checked
# and found FALSE by stats-reviewer (2026-09-04). generate() drops a
# mandate above ITS OWN category's AFA limit, and the elevated categories
# (insurance_premium/mutual_fund/credit_card_bill, clause 8(b)) allow up to
# Rs 1,00,000 -- far above this range. Measured: 316/7154 mandates (4.42%)
# in the actual estimation sample exceed 1_400_000, up to 8_978_529
# (Rs 89,785). amount_band_4 (>= _AMOUNT_BAND_CUT_3) is consequently a wide
# catch-all, CONFOUNDED with category by the AFA filter itself -- see
# reports/model_defensibility.md's Phase A section for the measured
# category x band breakdown. Left uncorrected pending a band redesign
# (disclosed, not silently fixed): Phase A's null finding is unaffected
# either way (both covariates are non-causal under `nominal` regardless of
# where a boundary falls), but Phase B must not inherit this scheme
# unexamined.
_AMOUNT_BAND_CUT_1 = 387_500
_AMOUNT_BAND_CUT_2 = 725_000
_AMOUNT_BAND_CUT_3 = 1_062_500

# category_mix keys from eval/frozen/sim_config.yaml, verbatim, minus
# "subscription" (70% of the mix, and this tuple's reference/omitted level --
# same convention as slot 2 being the reference for slot_3/slot_4 below).
_CATEGORY_LEVELS: tuple[str, ...] = ("insurance_premium", "mutual_fund", "credit_card_bill")

# ISSUER_LEVELS / INSTRUMENT_LEVELS -- R1 Phase B (2026-09-04). Vocabulary
# for eval/sim2.py's covariates: a second, non-frozen simulator whose DGP
# actually varies outcomes by issuer, instrument type and mandate age,
# because eval/frozen/simulator.py never generates any of the three (see
# this file's own module docstring and src/model/features.py's UNSOURCED
# dict). Defined here, not in eval/sim2.py, so there is exactly one source
# of truth for the vocabulary this module's _design_matrix() validates
# against -- eval/sim2.py imports these FROM here, the same direction every
# eval/ module already depends on src/, never the reverse. First level of
# each is the reference/omitted level, same convention as _CATEGORY_LEVELS
# and slot 2 above. `issuer_gamma` (not the reference) is the one eval/sim2.py
# gives an elevated dead-hazard, so its own name doubles up in the resulting
# dummy column (`issuer_issuer_gamma` -- "issuer_" + the level's own full
# name); left as-is rather than stripping the redundant prefix, since a
# stripped name would silently diverge from the level string itself.
ISSUER_LEVELS: tuple[str, ...] = ("issuer_alpha", "issuer_beta", "issuer_gamma", "issuer_delta")
INSTRUMENT_LEVELS: tuple[str, ...] = ("upi_autopay", "debit_card", "credit_card")

# _design_matrix() always computes whichever of these its `columns` argument
# actually asks for; FEATURE_COLUMNS selects the four `fit()` uses by
# DEFAULT -- see module docstring. WIDENED_FEATURE_COLUMNS is the R1
# (2026-09-04) alternative that adds amount and category; pass it to
# fit()'s `feature_columns` parameter explicitly. The default is unchanged.
_ALL_DESIGN_COLUMNS: tuple[str, ...] = (
    "const", "slot_3", "slot_4", "in_salary_window",
    "days_since_last_attempt", "slot3_x_in_salary_window",
    "amount_band_2", "amount_band_3", "amount_band_4",
    "category_insurance_premium", "category_mutual_fund", "category_credit_card_bill",
)

WIDENED_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + (
    "amount_band_2", "amount_band_3", "amount_band_4",
    "category_insurance_premium", "category_mutual_fund", "category_credit_card_bill",
)

# SIM2_FEATURE_COLUMNS -- R1 Phase B (2026-09-04): the design used to fit
# against eval/sim2.py's corpus, never against the real (frozen-simulator or
# eval/corpus.py) one, since issuer_id/instrument_type/mandate_age_days do
# not exist there. Deliberately NOT folded into `_ALL_DESIGN_COLUMNS` below
# (unlike WIDENED_FEATURE_COLUMNS's amount and category groups): those were
# sourceable from the real, already-assembled corpus frame, so growing the
# default was safe. issuer_id/instrument_type/mandate_age_days are not
# sourceable from any real frame this module is ever handed in production,
# so making them part of the DEFAULT no-arg `_design_matrix()` call would
# make that default unusable on real data. SIM2_FEATURE_COLUMNS is a third,
# fully separate alternative -- pass it to fit()'s `feature_columns`
# explicitly, exactly like WIDENED_FEATURE_COLUMNS.
SIM2_FEATURE_COLUMNS: tuple[str, ...] = (
    FEATURE_COLUMNS
    + tuple(f"issuer_{level}" for level in ISSUER_LEVELS[1:])
    + tuple(f"instrument_{level}" for level in INSTRUMENT_LEVELS[1:])
    + ("mandate_age_years",)
)


@dataclass(frozen=True)
class HazardModel:
    """Wraps a fitted statsmodels MNLogitResults plus the exact ordered
    design-matrix columns it was fit against -- see module docstring."""
    result: object
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS


def _design_matrix(df: pd.DataFrame, *, columns: tuple[str, ...] = _ALL_DESIGN_COLUMNS) -> pd.DataFrame:
    """Build exactly `columns` (default: every column this module knows how
    to construct) from `df`, each one explicitly -- never via
    `pd.get_dummies`, for the reason the module docstring gives (a batch
    missing a category silently drops that dummy and misaligns the fitted
    coefficient vector at predict time).

    Computes -- and requires the source column(s) for -- each column GROUP
    only if `columns` actually asks for a member of that group, rather than
    checking one blanket `required` set up front. This is what lets a
    caller using the unchanged default `FEATURE_COLUMNS`
    (`eval/allocator_sweep.py`'s `hazard_from_fit()`, whose row carries only
    `slot`/`in_salary_window`/`days_since_last_attempt` BY DESIGN -- see
    that function's own docstring on why amount_paise never entered its
    design matrix) keep working completely unmodified: it never asks for an
    amount- or category-derived column, so it is never asked to supply
    `amount_paise` or `category`. A caller that DOES request
    `WIDENED_FEATURE_COLUMNS` must supply both, or gets a `ValueError`
    naming exactly which one is missing -- not a blanket list of every
    column this function could ever need.

    Every column within one group is still computed together regardless of
    which VALUES are actually present in `df` -- e.g. requesting any one
    category dummy computes all three from the same three equality checks,
    and requesting any one amount band computes all three -- so widening
    this function does not reopen the get_dummies bug it was written to
    avoid.
    """
    needed = set(columns)
    out = pd.DataFrame(index=df.index)

    if "const" in needed:
        out["const"] = 1.0

    if {"slot_3", "slot_4", "slot3_x_in_salary_window"} & needed:
        if "slot" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['slot']")
        slot = df["slot"].astype("int64")
        out["slot_3"] = (slot == 3).astype(float)
        out["slot_4"] = (slot == 4).astype(float)

    if {"in_salary_window", "slot3_x_in_salary_window"} & needed:
        if "in_salary_window" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['in_salary_window']")
        out["in_salary_window"] = df["in_salary_window"].astype(float)

    if "days_since_last_attempt" in needed:
        if "days_since_last_attempt" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['days_since_last_attempt']")
        out["days_since_last_attempt"] = df["days_since_last_attempt"].astype(float)

    if "slot3_x_in_salary_window" in needed:
        out["slot3_x_in_salary_window"] = out["slot_3"] * out["in_salary_window"]

    if {"amount_band_2", "amount_band_3", "amount_band_4"} & needed:
        if "amount_paise" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['amount_paise']")
        amount = df["amount_paise"].astype("int64")
        out["amount_band_2"] = amount.between(_AMOUNT_BAND_CUT_1, _AMOUNT_BAND_CUT_2 - 1).astype(float)
        out["amount_band_3"] = amount.between(_AMOUNT_BAND_CUT_2, _AMOUNT_BAND_CUT_3 - 1).astype(float)
        out["amount_band_4"] = (amount >= _AMOUNT_BAND_CUT_3).astype(float)

    category_cols = {f"category_{level}" for level in _CATEGORY_LEVELS}
    if category_cols & needed:
        if "category" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['category']")
        category = df["category"].astype(str)
        # An unrecognised value (a typo'd category, or None/NaN --
        # .astype(str) turns either into the literal string "None"/"nan")
        # would otherwise silently score as the reference level with no
        # error: found by stats-reviewer (2026-09-04) as a real, if
        # currently latent, gap -- every category in the frozen corpus IS
        # one of these four, but Phase B's eval/sim2.py will not carry that
        # guarantee automatically. Loud, not silent, per this file's own
        # discipline for everything else it checks.
        _KNOWN_CATEGORIES = frozenset(_CATEGORY_LEVELS) | {"subscription"}
        unknown = sorted(set(category.unique()) - _KNOWN_CATEGORIES)
        if unknown:
            raise ValueError(
                f"design matrix input has categor{'y' if len(unknown) == 1 else 'ies'} "
                f"outside the known vocabulary {sorted(_KNOWN_CATEGORIES)}: {unknown} "
                f"-- a null/typo'd category would otherwise silently score as the "
                f"reference level"
            )
        for level in _CATEGORY_LEVELS:
            out[f"category_{level}"] = (category == level).astype(float)

    # R1 Phase B (2026-09-04): issuer_id / instrument_type / mandate_age_days
    # groups, requested only via SIM2_FEATURE_COLUMNS -- see that constant's
    # own comment for why these are not part of _ALL_DESIGN_COLUMNS's
    # default. Same loud-not-silent unknown-value discipline as the category
    # group above (a stats-reviewer-found gap there, R1a): a typo'd issuer or
    # instrument would otherwise silently score as the reference level.
    issuer_dummy_cols = {f"issuer_{level}" for level in ISSUER_LEVELS[1:]}
    if issuer_dummy_cols & needed:
        if "issuer_id" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['issuer_id']")
        issuer = df["issuer_id"].astype(str)
        unknown = sorted(set(issuer.unique()) - set(ISSUER_LEVELS))
        if unknown:
            raise ValueError(
                f"design matrix input has issuer_id value(s) outside the known "
                f"vocabulary {sorted(ISSUER_LEVELS)}: {unknown} -- a typo'd issuer "
                f"would otherwise silently score as the reference level"
            )
        for level in ISSUER_LEVELS[1:]:
            out[f"issuer_{level}"] = (issuer == level).astype(float)

    instrument_dummy_cols = {f"instrument_{level}" for level in INSTRUMENT_LEVELS[1:]}
    if instrument_dummy_cols & needed:
        if "instrument_type" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['instrument_type']")
        instrument = df["instrument_type"].astype(str)
        unknown = sorted(set(instrument.unique()) - set(INSTRUMENT_LEVELS))
        if unknown:
            raise ValueError(
                f"design matrix input has instrument_type value(s) outside the known "
                f"vocabulary {sorted(INSTRUMENT_LEVELS)}: {unknown} -- a typo'd "
                f"instrument would otherwise silently score as the reference level"
            )
        for level in INSTRUMENT_LEVELS[1:]:
            out[f"instrument_{level}"] = (instrument == level).astype(float)

    if "mandate_age_years" in needed:
        if "mandate_age_days" not in df.columns:
            raise ValueError("design matrix input is missing required column(s): ['mandate_age_days']")
        out["mandate_age_years"] = df["mandate_age_days"].astype(float) / 365.0

    return out[list(columns)]


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


def fit(
    df: pd.DataFrame,
    *,
    intercept_only: bool = False,
    feature_columns: tuple[str, ...] | None = None,
) -> HazardModel:
    """Fit a multinomial logit over `df[df.estimable]` -- `event_code` as
    the 4-category target, `STILL_PENDING` (0) as the MNLogit reference
    category. `df` is `assemble()`'s full output (slot-1 rows included);
    this function filters internally, per the module docstring.

    `feature_columns`, if given, is used verbatim (e.g.
    `WIDENED_FEATURE_COLUMNS` -- R1, 2026-09-04) instead of the
    `intercept_only`-selected default; `df` must then carry every raw
    source column those design columns need (`_design_matrix()` names the
    specific one missing, if any). Passing both `feature_columns` and
    `intercept_only=True` is a contradictory call -- both select design
    columns -- and raises rather than silently picking one."""
    if intercept_only and feature_columns is not None:
        raise ValueError(
            "fit() got both intercept_only=True and an explicit feature_columns -- "
            "these both select design columns; pass at most one"
        )

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

    if feature_columns is not None:
        cols = feature_columns
    elif intercept_only:
        cols = INTERCEPT_ONLY_COLUMNS
    else:
        cols = FEATURE_COLUMNS
    design = _design_matrix(estimable_df, columns=cols)
    target = estimable_df["event_code"].astype(int)

    result = sm.MNLogit(target, design).fit(disp=0)
    return HazardModel(result=result, feature_columns=cols)


def hazards(model: HazardModel, X: pd.DataFrame) -> np.ndarray:
    """Rebuild the identical design matrix `model` was fit against and
    return predicted probabilities, shape (len(X), 4), columns in Outcome
    int order [STILL_PENDING, RECOVERED, DEAD, OPTED_OUT], each row
    summing to 1."""
    design = _design_matrix(X, columns=model.feature_columns)
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
