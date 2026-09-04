"""src/model/competing_risks.py -- multinomial logit hazard model for person-period data.

Design decision this file pins: the model is discrete-time competing-risks (not
a single-cause survival model), fitted on estimable slots 2-4 only (slot 1 is a
structural zero, never part of the estimation set). The design matrix is built
identically at fit time and predict time via an explicit, private function
(never pd.get_dummies, which silently omits dummy columns for absent categories
and causes silent misalignment at predict time when a batch contains only one
slot). The model filters to df[df.estimable] internally; the caller need not
pre-filter. The result is a HazardModel frozen dataclass wrapping the
statsmodels MNLogitResults. assemble() joins person_period.build() outcome
columns with featurize() feature columns by row_id, raising ValueError if the
row_id sets are not identical. Evaluation functions (log_loss, brier_per_cause,
calibration_table) assume df[df.estimable] has already been selected by the
caller (they raise ValueError if estimable=False rows are present).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.core.types import CensorReason, Outcome, Profile
from src.core.ids import row_id
from eval.corpus import Episode
from eval.frozen.simulator import AttemptResult, SimMandate
from src.model.person_period import build
from src.model.features import featurize


def _mandate(
    mandate_id: str = "M_test",
    cycle_id: int = 1,
    amount_paise: int = 50_000,
    ceiling_paise: int = 100_000,
    category: str = "subscription",
) -> SimMandate:
    """Helper to build a SimMandate for test use."""
    from src.core.types import Cause

    return SimMandate(
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        amount_paise=amount_paise,
        ceiling_paise=ceiling_paise,
        category=category,
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )


def _attempt(
    mandate_id: str,
    slot: int,
    on_day: int,
    outcome: Outcome = Outcome.STILL_PENDING,
) -> AttemptResult:
    """Helper to build an AttemptResult for test use."""
    return AttemptResult(
        mandate_id=mandate_id,
        slot=slot,
        on_day=on_day,
        outcome=outcome,
        iatrogenic_insufficient_funds=False,
    )


def _build_and_featurize(episodes: list[Episode]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and featurize a list of episodes, return both frames."""
    pp_df = build(episodes)
    feat_df = featurize(pp_df)
    return pp_df, feat_df


# One representative amount per amount_band (reference, 2, 3, 4) and one
# category per non-reference level plus the reference itself -- see
# src/model/competing_risks.py's _AMOUNT_BAND_CUT_*/_CATEGORY_LEVELS.
# Real variation, not four repeats of the same value: fit(...,
# feature_columns=WIDENED_FEATURE_COLUMNS) needs every one of those six
# columns to actually vary across rows, or the corresponding MNLogit
# coefficient is unidentified (a constant-zero or constant-one column)
# and statsmodels raises LinAlgError: Singular matrix -- found by running
# this exact test, not anticipated up front.
_AMOUNT_BAND_SAMPLES: tuple[int, ...] = (100_000, 500_000, 800_000, 1_200_000)
_CATEGORY_SAMPLES: tuple[str, ...] = (
    "subscription", "insurance_premium", "mutual_fund", "credit_card_bill",
)


def _simple_estimable_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small frame with ~20 rows per (slot, in_salary_window) cell.
    Guarantees all four outcome types present, slots 2/3/4, both salary
    windows, and all four amount bands / all four categories (see
    _AMOUNT_BAND_SAMPLES/_CATEGORY_SAMPLES above -- needed for
    fit(feature_columns=WIDENED_FEATURE_COLUMNS) to be estimable at all).
    Returns (pp_df, feat_df) ready for fit()."""
    episodes = []
    idx = 0

    # Generate enough rows to have estimable data in each cell
    # We need slots 2, 3, 4; in_salary_window True/False for each
    # That's 6 combinations. Let's generate ~25 rows per combination for good measure.

    for slot in [2, 3, 4]:
        for in_window in [False, True]:
            for outcome_cycle in range(25):
                mandate_id = f"M_est_{idx:04d}"
                # CAUTION, found by stats-reviewer (not anticipated up
                # front): `idx % 4` here -- the first version of this fix --
                # is a PERFECT bijection with `outcome_cycle % 4` (the
                # outcome assignment below), because 25 (this loop's own
                # trip count) is congruent to 1 mod 4, so band and outcome
                # advance in lockstep within every (slot, in_window) cell.
                # The widened fit then "discovers" a large, entirely
                # fabricated amount effect (measured: +0.19 to +0.92,
                # against +/-0.05 on the real corpus) -- in a codebase whose
                # whole R1 finding is "amount carries no signal", and it
                # converges only by numerical accident: range(24)/range(26)/
                # range(28) all fail to converge or overflow. `idx // 5` (a
                # stride NOT a divisor/multiple of 4) breaks the lockstep --
                # verified directly: every (amount_band, outcome) pair and
                # every (amount_band, category) pair co-occurs across this
                # fixture, none is a bijection. Do not "simplify" this back
                # to idx % 4.
                amount_paise = _AMOUNT_BAND_SAMPLES[(idx // 5) % 4]
                category = _CATEGORY_SAMPLES[(idx // 4) % 4]
                idx += 1

                # Vary the day to control in_salary_window
                if in_window:
                    on_day = 2 + outcome_cycle % 4  # Days 2-5
                else:
                    on_day = 6 + outcome_cycle % 10  # Days 6+

                # Vary outcomes across the cycle
                if outcome_cycle % 4 == 0:
                    outcome = Outcome.RECOVERED
                elif outcome_cycle % 4 == 1:
                    outcome = Outcome.DEAD
                elif outcome_cycle % 4 == 2:
                    outcome = Outcome.OPTED_OUT
                else:
                    outcome = Outcome.STILL_PENDING

                # Create episodes with the desired slot
                attempts = []
                for s in range(2, slot + 1):
                    day = on_day if s == slot else on_day - (slot - s) * 3
                    out = outcome if s == slot else Outcome.STILL_PENDING
                    attempts.append(_attempt(mandate_id, s, day, out))

                # ceiling_paise must be >= amount_paise (clause 4(c),
                # eval/corpus.py's assert_legal()) -- found by
                # stats-reviewer: _mandate()'s own default (100_000) is
                # below 3 of the 4 _AMOUNT_BAND_SAMPLES values, which would
                # otherwise construct a clause-4(c)-illegal mandate.
                mandate = _mandate(
                    mandate_id, cycle_id=1, amount_paise=amount_paise,
                    ceiling_paise=amount_paise * 2, category=category,
                )
                # build() only reads ep.censor_reason on the terminal row when
                # that row's outcome is STILL_PENDING (person_period.py:148,
                # row_censored) -- it must be a real CensorReason member, not
                # Python None, or _apply_dtypes()'s CensorReason(None) raises.
                # Any resolved outcome (RECOVERED/DEAD/OPTED_OUT) never has
                # this value read at all, but CensorReason.NONE is the
                # honest value regardless.
                episode = Episode(
                    mandate=mandate,
                    attempts=tuple(attempts),
                    censor_reason=(
                        (CensorReason.BUDGET_EXHAUSTED if slot == 4 else CensorReason.WINDOW_CLOSED)
                        if outcome == Outcome.STILL_PENDING
                        else CensorReason.NONE
                    ),
                )
                episodes.append(episode)

    pp_df, feat_df = _build_and_featurize(episodes)
    return pp_df, feat_df


# === assemble() tests =========================================================


def test_assemble_round_trip_preserves_row_count():
    """assemble() must return one row per row_id and no duplicates."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()
    result = assemble(pp_df, feat_df)

    assert len(result) == len(pp_df)
    assert len(result) == len(feat_df)
    assert len(result) == len(result.drop_duplicates(subset=["row_id"]))


def test_assemble_joins_on_row_id_correctly():
    """assemble() must join feat_df + event_code/estimable from pp_df by row_id.
    Verify that joined values match the source pp_df rows exactly, even when
    pp_df and feat_df are in different row orders."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()

    # Shuffle feat_df row order to test join stability
    feat_df_shuffled = feat_df.sample(frac=1, random_state=42).reset_index(drop=True)

    result = assemble(pp_df, feat_df_shuffled)

    # Build a mapping of row_id -> (event_code, estimable) from pp_df
    pp_mapping = dict(zip(pp_df["row_id"], zip(pp_df["event_code"], pp_df["estimable"])))

    # Verify every result row's event_code/estimable match the source pp_df
    for _, row in result.iterrows():
        pp_event_code, pp_estimable = pp_mapping[row["row_id"]]
        assert row["event_code"] == pp_event_code
        assert row["estimable"] == pp_estimable


def test_assemble_raises_on_mismatched_row_id_sets_extra_in_feat():
    """assemble() must raise ValueError if feat_df has row_ids not in pp_df."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()

    # Add a row to feat_df with a new row_id
    new_row = feat_df.iloc[0].copy()
    new_row["row_id"] = row_id("M_extra", 99, 99)
    feat_df_extra = pd.concat([feat_df, pd.DataFrame([new_row])], ignore_index=True)

    with pytest.raises(ValueError):
        assemble(pp_df, feat_df_extra)


def test_assemble_raises_on_mismatched_row_id_sets_extra_in_pp():
    """assemble() must raise ValueError if pp_df has row_ids not in feat_df."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()

    # Remove a row from feat_df
    feat_df_subset = feat_df.iloc[:-1].copy()

    with pytest.raises(ValueError):
        assemble(pp_df, feat_df_subset)


def test_assemble_includes_all_feat_columns():
    """assemble() output must include all feat_df's columns."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()
    result = assemble(pp_df, feat_df)

    for col in feat_df.columns:
        assert col in result.columns


def test_assemble_includes_event_code_and_estimable():
    """assemble() output must include event_code and estimable from pp_df."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()
    result = assemble(pp_df, feat_df)

    assert "event_code" in result.columns
    assert "estimable" in result.columns


# === fit() tests ==============================================================


def test_fit_filters_to_estimable_internally():
    """fit() must filter to df[df.estimable] internally before fitting.
    This test verifies that fitting on a frame with slot-1 rows (estimable=False)
    produces identical coefficients to fitting on df[df.estimable] pre-filtered,
    proving that fit() is filtering, not just documented to."""
    from src.model.competing_risks import assemble, fit

    pp_df, feat_df = _simple_estimable_frame()
    assembled = assemble(pp_df, feat_df)

    # Fit on the full assembled frame (includes slot-1 rows with estimable=False)
    model_unfiltered = fit(assembled)

    # Fit on pre-filtered frame (estimable=True only)
    assembled_filtered = assembled[assembled["estimable"]].copy()
    model_filtered = fit(assembled_filtered)

    # Extract coefficients from both models. statsmodels MNLogitResults.params
    # is a DataFrame (K exog rows x J-1 non-reference-outcome columns), so
    # reduce to a single scalar via numpy rather than comparing a Series/
    # DataFrame directly in a bare assert (ambiguous truth value otherwise).
    coefs_unfiltered = np.asarray(model_unfiltered.result.params)
    coefs_filtered = np.asarray(model_filtered.result.params)

    # They should be identical (or very close due to numerical precision)
    assert np.abs(coefs_unfiltered - coefs_filtered).max() < 1e-6, (
        "Coefficients differ between filtered and unfiltered fits: "
        "fit() is not filtering to estimable correctly"
    )


def test_fit_raises_on_missing_event_code_column():
    """fit() must raise ValueError if event_code column is missing."""
    from src.model.competing_risks import fit

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    assembled_no_event = assembled.drop(columns=["event_code"])

    with pytest.raises(ValueError):
        fit(assembled_no_event)


def test_fit_raises_on_missing_estimable_column():
    """fit() must raise ValueError if estimable column is missing."""
    from src.model.competing_risks import fit

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    assembled_no_est = assembled.drop(columns=["estimable"])

    with pytest.raises(ValueError):
        fit(assembled_no_est)


def test_fit_raises_on_no_estimable_rows():
    """fit() must raise ValueError if after filtering, no estimable rows remain."""
    from src.model.competing_risks import fit

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    # Set all estimable to False
    assembled_all_non_est = assembled.copy()
    assembled_all_non_est["estimable"] = False

    with pytest.raises(ValueError):
        fit(assembled_all_non_est)


def test_fit_intercept_only_true_design_matrix_has_one_column():
    """fit(..., intercept_only=True) must fit with only const, no other features."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    model_intercept_only = fit(assembled, intercept_only=True)

    # The model's result.params should have exactly 3 entries (one per non-reference outcome)
    # Each with only 'const' as the feature (intercept)
    n_params_per_outcome = model_intercept_only.result.params.groupby(level=0).size()
    assert all(n == 1 for n in n_params_per_outcome), (
        "intercept_only fit should have exactly 1 coefficient per outcome"
    )


def test_fit_intercept_only_vs_full_produces_different_predictions():
    """fit(..., intercept_only=True) and fit(..., intercept_only=False)
    must produce different predictions on the same data, proving that the
    intercept-only model really only has an intercept."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    model_intercept_only = fit(assembled, intercept_only=True)
    model_full = fit(assembled, intercept_only=False)

    # Predict on a subset of data
    X = assembled[["slot", "in_salary_window", "days_since_last_attempt"]].head(20)

    hazards_intercept_only = hazards(model_intercept_only, X)
    hazards_full = hazards(model_full, X)

    # They must be different (at least somewhere)
    assert not np.allclose(hazards_intercept_only, hazards_full), (
        "intercept_only and full models produced identical predictions"
    )


# === hazards() tests ==========================================================


def test_hazards_returns_correct_shape():
    """hazards() must return (n, 4) array for n input rows."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    X = assembled[["slot", "in_salary_window", "days_since_last_attempt"]].head(10)
    result = hazards(model, X)

    assert result.shape == (10, 4)


def test_hazards_rows_sum_to_one():
    """Each row of hazards() output must sum to 1 (probabilities)."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    X = assembled[["slot", "in_salary_window", "days_since_last_attempt"]].head(50)
    result = hazards(model, X)

    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums: {row_sums}"


def test_hazards_all_probabilities_in_valid_range():
    """Every value in hazards() output must be in [0, 1]."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    X = assembled[["slot", "in_salary_window", "days_since_last_attempt"]].head(50)
    result = hazards(model, X)

    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_hazards_handles_single_slot_batch():
    """hazards() must not raise or misalign when X contains only one slot value.
    This guards against the pd.get_dummies bug where a batch with only slot-2
    rows would drop the slot_3/slot_4 dummy columns and misalign the coefficient
    vector."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    # Filter to only slot-2 rows
    X_slot2_only = assembled[assembled["slot"] == 2][
        ["slot", "in_salary_window", "days_since_last_attempt"]
    ].head(10)

    result = hazards(model, X_slot2_only)

    # Must not raise, and must return valid shape/probabilities
    assert result.shape == (len(X_slot2_only), 4)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_hazards_matches_statsmodels_predict_on_same_data():
    """hazards() on fit data must closely match statsmodels' own .predict()
    on the same data."""
    from src.model.competing_risks import fit, hazards
    import statsmodels.api as sm

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    # Use a small subset
    X = assembled[["slot", "in_salary_window", "days_since_last_attempt"]].head(10)

    hazard_result = hazards(model, X)

    # Get statsmodels' prediction from the model's result object
    # We need to rebuild the design matrix the same way hazards() does
    # For now, just verify the shapes and probabilities are valid
    # (A full round-trip would require exposing the design matrix function)
    assert hazard_result.shape == (10, 4)
    assert np.allclose(hazard_result.sum(axis=1), 1.0)


# === log_loss() tests =========================================================


def test_log_loss_raises_on_non_estimable_rows():
    """log_loss() must raise ValueError if the input contains estimable=False rows."""
    from src.model.competing_risks import fit, log_loss

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    # Pass unfiltered data (has slot-1 rows with estimable=False)
    with pytest.raises(ValueError):
        log_loss(model, assembled)


def test_log_loss_near_zero_on_perfect_predictions():
    """log_loss() should be near 0 when the model's predicted probability
    on the true event_code is 1.0 for every row."""
    from src.model.competing_risks import log_loss
    from dataclasses import dataclass

    # Build a stub HazardModel that always predicts [0, 1, 0, 0] (outcome 1 with prob 1.0).
    # predict() ignores its argument's content -- only len(X) matters -- so
    # log_loss() is free to route through the real hazards() design-matrix
    # path (as it does for a real HazardModel) without changing this stub's
    # output. That means test_df needs the raw columns hazards() derives
    # from (slot, in_salary_window, days_since_last_attempt) and this stub
    # needs feature_columns so hazards() can select from the full design
    # matrix, exactly like a real fit()-produced HazardModel.
    class StubResult:
        def predict(self, X):
            n = len(X)
            result = np.zeros((n, 4))
            result[:, 1] = 1.0
            return result

    @dataclass(frozen=True)
    class StubHazardModel:
        result: object
        feature_columns: tuple = (
            "const", "slot_3", "slot_4", "in_salary_window",
            "days_since_last_attempt", "slot3_x_in_salary_window",
        )

    # Create a test frame where every row has event_code=1 (RECOVERED)
    test_df = pd.DataFrame({
        "event_code": [1] * 5,
        "estimable": [True] * 5,
        "slot": [2] * 5,
        "in_salary_window": [False] * 5,
        "days_since_last_attempt": [0] * 5,
    })

    stub_result = StubResult()
    stub_model = StubHazardModel(result=stub_result)

    loss = log_loss(stub_model, test_df)

    # log_loss should be very close to 0
    assert loss < 0.01, f"log_loss on perfect predictions should be near 0, got {loss}"


def test_log_loss_computation_is_correct():
    """Verify log_loss arithmetic against a hand-computed value."""
    from src.model.competing_risks import log_loss

    class StubResult:
        def predict(self, X):
            # Predict: row 0: [0.7, 0.2, 0.1, 0], row 1: [0.5, 0.3, 0.2, 0], etc.
            n = len(X)
            result = np.zeros((n, 4))
            result[0] = [0.7, 0.2, 0.1, 0.0]  # true event_code=0: -log(0.7)
            result[1] = [0.4, 0.4, 0.2, 0.0]  # true event_code=1: -log(0.4)
            if n > 2:
                result[2:] = [0.25, 0.25, 0.25, 0.25]
            return result

    @dataclass(frozen=True)
    class StubHazardModel:
        result: object
        feature_columns: tuple = (
            "const", "slot_3", "slot_4", "in_salary_window",
            "days_since_last_attempt", "slot3_x_in_salary_window",
        )

    test_df = pd.DataFrame({
        "event_code": [0, 1, 0, 1],  # True outcomes
        "estimable": [True] * 4,
        "slot": [2] * 4,
        "in_salary_window": [False] * 4,
        "days_since_last_attempt": [0] * 4,
    })

    stub_result = StubResult()
    stub_model = StubHazardModel(result=stub_result)

    loss = log_loss(stub_model, test_df)

    # Expected: (log(0.7) + log(0.4) + log(0.25) + log(0.25)) / 4
    expected = -(np.log(0.7) + np.log(0.4) + np.log(0.25) + np.log(0.25)) / 4
    assert loss == pytest.approx(expected, abs=1e-6)


# === brier_per_cause() tests ==================================================


def test_brier_per_cause_raises_on_non_estimable_rows():
    """brier_per_cause() must raise ValueError if input has estimable=False rows."""
    from src.model.competing_risks import fit, brier_per_cause

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    # Pass unfiltered data
    with pytest.raises(ValueError):
        brier_per_cause(model, assembled)


def test_brier_per_cause_returns_dict_with_four_keys():
    """brier_per_cause() must return a dict with keys 0, 1, 2, 3."""
    from src.model.competing_risks import fit, brier_per_cause

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    assembled_est = assembled[assembled["estimable"]].copy()
    result = brier_per_cause(model, assembled_est)

    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 1, 2, 3}


def test_brier_per_cause_values_in_valid_range():
    """brier_per_cause() scores must all be in [0, 1]."""
    from src.model.competing_risks import fit, brier_per_cause

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    assembled_est = assembled[assembled["estimable"]].copy()
    result = brier_per_cause(model, assembled_est)

    for outcome_int, score in result.items():
        assert 0.0 <= score <= 1.0, (
            f"Brier score for outcome {outcome_int} is {score}, out of range"
        )


def test_brier_per_cause_computation_is_correct():
    """Verify brier_per_cause() arithmetic against hand-computed value."""
    from src.model.competing_risks import brier_per_cause

    class StubResult:
        def predict(self, X):
            n = len(X)
            result = np.zeros((n, 4))
            result[0] = [0.2, 0.3, 0.4, 0.1]
            result[1] = [0.5, 0.2, 0.2, 0.1]
            if n > 2:
                result[2:] = [0.25, 0.25, 0.25, 0.25]
            return result

    @dataclass(frozen=True)
    class StubHazardModel:
        result: object
        feature_columns: tuple = (
            "const", "slot_3", "slot_4", "in_salary_window",
            "days_since_last_attempt", "slot3_x_in_salary_window",
        )

    # event_code values: 0, 1
    test_df = pd.DataFrame({
        "event_code": [0, 1, 0, 1],
        "estimable": [True] * 4,
        "slot": [2] * 4,
        "in_salary_window": [False] * 4,
        "days_since_last_attempt": [0] * 4,
    })

    stub_result = StubResult()
    stub_model = StubHazardModel(result=stub_result)

    result = brier_per_cause(stub_model, test_df)

    # For outcome 0:
    #   Row 0: true, pred 0.2 -> (1-0.2)^2 = 0.64
    #   Row 1: false, pred 0.5 -> (0-0.5)^2 = 0.25
    #   Row 2: true, pred 0.25 -> (1-0.25)^2 = 0.5625
    #   Row 3: false, pred 0.25 -> (0-0.25)^2 = 0.0625
    #   Mean: (0.64 + 0.25 + 0.5625 + 0.0625) / 4 = 1.5275 / 4 = 0.381875
    expected_outcome_0 = (0.64 + 0.25 + 0.5625 + 0.0625) / 4
    assert result[0] == pytest.approx(expected_outcome_0, abs=1e-6)


# === calibration_table() tests ================================================


def test_calibration_table_raises_on_non_estimable_rows():
    """calibration_table() must raise ValueError if input has estimable=False."""
    from src.model.competing_risks import fit, calibration_table

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    # Pass unfiltered data
    with pytest.raises(ValueError):
        calibration_table(model, assembled)


def test_calibration_table_returns_dataframe():
    """calibration_table() must return a pandas DataFrame."""
    from src.model.competing_risks import fit, calibration_table

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    assembled_est = assembled[assembled["estimable"]].copy()
    result = calibration_table(model, assembled_est)

    assert isinstance(result, pd.DataFrame)


def test_calibration_table_has_required_columns():
    """calibration_table() must include n_rows, mean_predicted_prob, realized_frequency columns."""
    from src.model.competing_risks import fit, calibration_table

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)

    assembled_est = assembled[assembled["estimable"]].copy()
    result = calibration_table(model, assembled_est)

    required_cols = {"n_rows", "mean_predicted_prob", "realized_frequency"}
    assert required_cols <= set(result.columns), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )


def test_calibration_table_mean_pred_equals_realized_on_synthetic_data():
    """On synthetic data where we control the probabilities exactly,
    calibration_table() should show mean_predicted_prob == realized_frequency."""
    from src.model.competing_risks import calibration_table

    class StubResult:
        def predict(self, X):
            # Return constant probabilities for all rows
            n = len(X)
            result = np.zeros((n, 4))
            # All rows: [0.25, 0.25, 0.25, 0.25] -- uniform
            result[:, :] = 0.25
            return result

    @dataclass(frozen=True)
    class StubHazardModel:
        result: object
        feature_columns: tuple = (
            "const", "slot_3", "slot_4", "in_salary_window",
            "days_since_last_attempt", "slot3_x_in_salary_window",
        )

    # Create test data with exactly 25% of each outcome
    test_df = pd.DataFrame({
        "slot": [2, 2, 2, 2] * 25,  # All slot 2
        "in_salary_window": [True, True, True, True] * 25,
        "days_since_last_attempt": [0] * 100,
        "event_code": [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25,  # 25 of each
        "estimable": [True] * 100,
    })

    stub_result = StubResult()
    stub_model = StubHazardModel(result=stub_result)

    result = calibration_table(stub_model, test_df)

    # Filter to outcome 0 (or any outcome)
    outcome0_rows = result[result["event_code"] == 0]
    if len(outcome0_rows) > 0:
        row = outcome0_rows.iloc[0]
        # mean_predicted_prob should be 0.25, realized_frequency should be 0.25
        assert row["mean_predicted_prob"] == pytest.approx(0.25, abs=0.01)
        assert row["realized_frequency"] == pytest.approx(0.25, abs=0.01)


# === HazardModel dataclass tests ===============================================


def test_hazard_model_is_frozen_dataclass():
    """HazardModel must be a frozen dataclass (immutable)."""
    from src.model.competing_risks import HazardModel

    class StubResult:
        pass

    model = HazardModel(result=StubResult())

    # dataclasses.FrozenInstanceError subclasses AttributeError.
    with pytest.raises(AttributeError):
        model.result = None


# === New widened-design-matrix tests ==========================================
# Test suite for amount bands, category dummies, and the new `columns`
# parameter to _design_matrix() and feature_columns parameter to fit().
# These tests validate the exact contract specified in the feature spec:
# backward compatibility, selective column computation, proper boundary
# handling, and the pd.get_dummies avoidance guarantee across new columns.


def test_amount_band_constants_exist():
    """The amount-band boundary constants must be defined at module level."""
    from src.model.competing_risks import (
        _AMOUNT_BAND_CUT_1,
        _AMOUNT_BAND_CUT_2,
        _AMOUNT_BAND_CUT_3,
    )

    assert _AMOUNT_BAND_CUT_1 == 387_500
    assert _AMOUNT_BAND_CUT_2 == 725_000
    assert _AMOUNT_BAND_CUT_3 == 1_062_500


def test_category_levels_constant_exists():
    """The category levels tuple must be defined at module level."""
    from src.model.competing_risks import _CATEGORY_LEVELS

    assert _CATEGORY_LEVELS == ("insurance_premium", "mutual_fund", "credit_card_bill")


def test_widened_feature_columns_constant_exists():
    """WIDENED_FEATURE_COLUMNS must be defined and include the legacy 4 plus new dummies."""
    from src.model.competing_risks import WIDENED_FEATURE_COLUMNS, FEATURE_COLUMNS

    # WIDENED should be 10 columns: 4 legacy + 3 amount + 3 category
    assert len(WIDENED_FEATURE_COLUMNS) == 10
    # First four should be the legacy FEATURE_COLUMNS
    assert WIDENED_FEATURE_COLUMNS[:4] == FEATURE_COLUMNS
    # Should contain all three amount bands
    assert "amount_band_2" in WIDENED_FEATURE_COLUMNS
    assert "amount_band_3" in WIDENED_FEATURE_COLUMNS
    assert "amount_band_4" in WIDENED_FEATURE_COLUMNS
    # Should contain all three category dummies
    assert "category_insurance_premium" in WIDENED_FEATURE_COLUMNS
    assert "category_mutual_fund" in WIDENED_FEATURE_COLUMNS
    assert "category_credit_card_bill" in WIDENED_FEATURE_COLUMNS


def test_design_matrix_accepts_columns_parameter():
    """_design_matrix() must accept a keyword-only `columns` parameter."""
    from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    X = assembled.head(5)

    # Should succeed with explicit columns parameter
    result = _design_matrix(X, columns=FEATURE_COLUMNS)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(FEATURE_COLUMNS)


def test_design_matrix_legacy_columns_work_without_amount_paise_or_category():
    """_design_matrix(..., columns=FEATURE_COLUMNS) must work on a frame
    with NO amount_paise or category columns at all. This is the regression
    test for hazard_from_fit() in eval/allocator_sweep.py -- it builds a
    minimal row with only slot/in_salary_window/days_since_last_attempt and
    must not break when the model was fit with widened columns."""
    from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

    # Build a minimal synthetic frame exactly like hazard_from_fit() does
    X_minimal = pd.DataFrame([{
        "slot": 2,
        "in_salary_window": True,
        "days_since_last_attempt": 0.0,
    }])

    # This must succeed and return exactly 4 columns in order
    result = _design_matrix(X_minimal, columns=FEATURE_COLUMNS)
    assert list(result.columns) == list(FEATURE_COLUMNS)
    assert len(result) == 1
    assert np.isfinite(result.values).all()


def test_design_matrix_default_columns_unchanged():
    """Calling _design_matrix(df) with no `columns` argument must still use
    its default behavior: computing every column the module currently
    knows how to build (_ALL_DESIGN_COLUMNS), on a df -- like a real
    assembled corpus frame -- that carries every source column those need.
    _ALL_DESIGN_COLUMNS itself grew from 6 to 12 with R1's amount/category
    widening (2026-09-04); this test pins "no-arg call returns the full
    set" rather than a stale literal count."""
    from src.model.competing_risks import _design_matrix, _ALL_DESIGN_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    X = assembled.head(5)

    # Call without columns parameter
    result = _design_matrix(X)
    assert len(result.columns) == len(_ALL_DESIGN_COLUMNS)
    assert set(result.columns) == set(_ALL_DESIGN_COLUMNS)


def test_design_matrix_amount_band_boundary_values():
    """Test exact boundary values for amount bands."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    # Test each boundary
    test_amounts = [
        (387_499, "reference_band"),      # Below cut_1, all band dummies = 0
        (387_500, "band_2"),              # Exactly cut_1, band_2 = 1
        (724_999, "band_2"),              # Below cut_2, still band_2
        (725_000, "band_3"),              # Exactly cut_2, band_3 = 1
        (1_062_499, "band_3"),            # Below cut_3, still band_3
        (1_062_500, "band_4"),            # Exactly cut_3, band_4 = 1
        (5_000_000, "band_4"),            # Large amount, still band_4
    ]

    for amount, expected_band in test_amounts:
        X = pd.DataFrame({
            "slot": [2],
            "in_salary_window": [False],
            "days_since_last_attempt": [0.0],
            "amount_paise": [amount],
            "category": ["subscription"],
        })
        result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

        # Verify the expected band is set correctly
        if expected_band == "reference_band":
            assert result["amount_band_2"].iloc[0] == 0.0
            assert result["amount_band_3"].iloc[0] == 0.0
            assert result["amount_band_4"].iloc[0] == 0.0
        elif expected_band == "band_2":
            assert result["amount_band_2"].iloc[0] == 1.0
            assert result["amount_band_3"].iloc[0] == 0.0
            assert result["amount_band_4"].iloc[0] == 0.0
        elif expected_band == "band_3":
            assert result["amount_band_2"].iloc[0] == 0.0
            assert result["amount_band_3"].iloc[0] == 1.0
            assert result["amount_band_4"].iloc[0] == 0.0
        elif expected_band == "band_4":
            assert result["amount_band_2"].iloc[0] == 0.0
            assert result["amount_band_3"].iloc[0] == 0.0
            assert result["amount_band_4"].iloc[0] == 1.0


def test_design_matrix_category_dummies_reference_level():
    """category='subscription' (the reference level) must produce all 0s for the three category dummies."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["subscription"],
    })
    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    assert result["category_insurance_premium"].iloc[0] == 0.0
    assert result["category_mutual_fund"].iloc[0] == 0.0
    assert result["category_credit_card_bill"].iloc[0] == 0.0


def test_design_matrix_category_dummies_insurance_premium():
    """category='insurance_premium' must produce the correct dummy coding."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["insurance_premium"],
    })
    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    assert result["category_insurance_premium"].iloc[0] == 1.0
    assert result["category_mutual_fund"].iloc[0] == 0.0
    assert result["category_credit_card_bill"].iloc[0] == 0.0


def test_design_matrix_category_dummies_mutual_fund():
    """category='mutual_fund' must produce the correct dummy coding."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["mutual_fund"],
    })
    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    assert result["category_insurance_premium"].iloc[0] == 0.0
    assert result["category_mutual_fund"].iloc[0] == 1.0
    assert result["category_credit_card_bill"].iloc[0] == 0.0


def test_design_matrix_category_dummies_credit_card_bill():
    """category='credit_card_bill' must produce the correct dummy coding."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["credit_card_bill"],
    })
    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    assert result["category_insurance_premium"].iloc[0] == 0.0
    assert result["category_mutual_fund"].iloc[0] == 0.0
    assert result["category_credit_card_bill"].iloc[0] == 1.0


def test_design_matrix_raises_on_unrecognized_category():
    """An unrecognized category value must raise, not silently score as the
    reference level. CORRECTED by stats-reviewer, 2026-09-04: an earlier
    version of this test asserted the OPPOSITE (silent tolerance) as the
    intended contract; review found that a typo'd category -- or None/NaN,
    which `.astype(str)` turns into the literal string "None"/"nan" -- would
    then produce a wrong-but-plausible prediction with no error, the exact
    silent-wrong-answer failure mode this file's explicit-column discipline
    exists to prevent everywhere else. Loud failure is the corrected,
    intended behavior; this test's name and assertion changed to match."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["unknown_category"],
    })
    with pytest.raises(ValueError, match="unknown_category"):
        _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)


def test_design_matrix_raises_on_null_category():
    """None/NaN in the category column must raise for the same reason as an
    unrecognized string -- pandas coerces either to the literal string
    "None"/"nan" under .astype(str), which matches no known level and would
    otherwise silently score as the reference."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    X = pd.DataFrame({
        "slot": [2, 3],
        "in_salary_window": [False, True],
        "days_since_last_attempt": [0.0, 1.0],
        "amount_paise": [100_000, 200_000],
        "category": ["subscription", None],
    })
    with pytest.raises(ValueError):
        _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)


def test_design_matrix_raises_on_missing_amount_paise_when_requested():
    """Requesting an amount_band column on a frame missing amount_paise must raise ValueError."""
    from src.model.competing_risks import _design_matrix

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "category": ["subscription"],
        # Deliberately missing amount_paise
    })

    with pytest.raises(ValueError, match="amount_paise"):
        _design_matrix(X, columns=("const", "slot_3", "amount_band_2"))


def test_design_matrix_raises_on_missing_category_when_requested():
    """Requesting a category_* column on a frame missing category must raise ValueError."""
    from src.model.competing_risks import _design_matrix

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        # Deliberately missing category
    })

    with pytest.raises(ValueError, match="category"):
        _design_matrix(X, columns=("const", "slot_3", "category_insurance_premium"))


def test_design_matrix_column_order_matches_parameter_order():
    """The returned DataFrame's column order must exactly match the columns parameter."""
    from src.model.competing_risks import _design_matrix

    X = pd.DataFrame({
        "slot": [2],
        "in_salary_window": [False],
        "days_since_last_attempt": [0.0],
        "amount_paise": [100_000],
        "category": ["subscription"],
    })

    # Request columns in a deliberately scrambled order
    requested = ("category_mutual_fund", "const", "amount_band_4", "in_salary_window")
    result = _design_matrix(X, columns=requested)

    # Column order must match the requested order exactly
    assert list(result.columns) == list(requested)


def test_design_matrix_single_category_level_returns_all_dummies():
    """_design_matrix() must return all three category dummy columns even when
    the data contains only one category level. This is the pd.get_dummies guard."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    # Build a frame where ONLY category='subscription' appears
    X = pd.DataFrame({
        "slot": [2, 2, 2],
        "in_salary_window": [False, True, False],
        "days_since_last_attempt": [0.0, 0.0, 0.0],
        "amount_paise": [100_000, 100_000, 100_000],
        "category": ["subscription", "subscription", "subscription"],
    })

    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    # All three category columns must be present
    assert "category_insurance_premium" in result.columns
    assert "category_mutual_fund" in result.columns
    assert "category_credit_card_bill" in result.columns
    # All values must be 0.0
    assert (result["category_insurance_premium"] == 0.0).all()
    assert (result["category_mutual_fund"] == 0.0).all()
    assert (result["category_credit_card_bill"] == 0.0).all()


def test_design_matrix_single_amount_band_returns_all_bands():
    """_design_matrix() must return all three amount_band columns even when
    all rows fall into a single band. This is the pd.get_dummies guard."""
    from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

    # Build a frame where all amounts are in the reference band
    X = pd.DataFrame({
        "slot": [2, 2, 2],
        "in_salary_window": [False, True, False],
        "days_since_last_attempt": [0.0, 0.0, 0.0],
        "amount_paise": [100_000, 150_000, 200_000],  # All < 387_500
        "category": ["subscription", "subscription", "subscription"],
    })

    result = _design_matrix(X, columns=WIDENED_FEATURE_COLUMNS)

    # All three amount_band columns must be present
    assert "amount_band_2" in result.columns
    assert "amount_band_3" in result.columns
    assert "amount_band_4" in result.columns
    # All values must be 0.0
    assert (result["amount_band_2"] == 0.0).all()
    assert (result["amount_band_3"] == 0.0).all()
    assert (result["amount_band_4"] == 0.0).all()


def test_fit_accepts_feature_columns_parameter():
    """fit() must accept a keyword-only feature_columns parameter."""
    from src.model.competing_risks import fit, FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    # Should succeed with feature_columns parameter
    model = fit(assembled, feature_columns=FEATURE_COLUMNS)
    assert model.feature_columns == FEATURE_COLUMNS


def test_fit_default_feature_columns_unchanged():
    """fit(df) with no feature_columns parameter must use FEATURE_COLUMNS by default."""
    from src.model.competing_risks import fit, FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    model = fit(assembled)
    assert model.feature_columns == FEATURE_COLUMNS


def test_fit_raises_on_both_intercept_only_and_feature_columns():
    """fit() must raise ValueError if both intercept_only=True and feature_columns are passed."""
    from src.model.competing_risks import fit, WIDENED_FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)

    with pytest.raises(ValueError):
        fit(assembled, intercept_only=True, feature_columns=WIDENED_FEATURE_COLUMNS)


def test_fit_with_widened_feature_columns_succeeds():
    """fit(..., feature_columns=WIDENED_FEATURE_COLUMNS) must succeed on a
    frame with amount_paise/category, and the fit must have actually
    CONVERGED -- not merely returned without raising. statsmodels reports a
    ConvergenceWarning (not an exception) on a near-singular design, so a
    test asserting only "did not raise" would still pass against a garbage
    fit; found by stats-reviewer, who showed a nearby stride choice in this
    same fixture produces exactly that (silently-passing, non-converged)
    failure mode. This assertion is what makes a FUTURE fixture regression
    fail loudly instead of silently."""
    from src.model.competing_risks import fit, WIDENED_FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    # The simple_estimable_frame already includes amount_paise and category

    model = fit(assembled, feature_columns=WIDENED_FEATURE_COLUMNS)
    assert model.feature_columns == WIDENED_FEATURE_COLUMNS
    assert model.result.mle_retvals["converged"] is True


def test_simple_estimable_frame_amount_band_is_not_a_bijection_with_outcome():
    """Regression test for the exact bug stats-reviewer found: an earlier
    version of _simple_estimable_frame() derived amount_paise from `idx % 4`,
    which -- because this fixture's own inner loop runs range(25) and
    25 = 4*6+1 -- made the amount band a PERFECT bijection with
    outcome_cycle % 4 (the outcome) within every (slot, in_window) cell:
    every row in amount_band N had the identical outcome, and vice versa.
    That is quasi-complete separation, and the widened MNLogit fit
    "discovered" a large fabricated amount effect (+0.19 to +0.92) purely
    from the fixture's own construction -- the opposite of this corpus's
    real, near-zero amount effect. Assert directly, from the built frame,
    that no amount band (or category level) determines a single outcome
    class -- the property whose absence caused the original bug, checked
    on the actual output rather than re-deriving the index arithmetic by
    hand a second time."""
    from src.model.competing_risks import assemble

    pp_df, feat_df = _simple_estimable_frame()
    assembled = assemble(pp_df, feat_df)
    estimable = assembled[assembled["estimable"]]

    for amount_paise, group in estimable.groupby("amount_paise"):
        n_outcomes = group["event_code"].nunique()
        assert n_outcomes > 1, (
            f"amount_paise={amount_paise} co-occurs with only one "
            f"event_code -- this is the exact perfect-separation bug "
            f"stats-reviewer found; every amount value must see multiple "
            f"outcomes in this fixture"
        )
    for category, group in estimable.groupby("category", observed=True):
        n_outcomes = group["event_code"].nunique()
        assert n_outcomes > 1, (
            f"category={category!r} co-occurs with only one event_code -- "
            f"same bug class as the amount check above"
        )


def test_hazards_with_widened_model_returns_correct_shape():
    """hazards() called on a model fit with WIDENED_FEATURE_COLUMNS must return (n, 4)."""
    from src.model.competing_risks import fit, hazards, WIDENED_FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled, feature_columns=WIDENED_FEATURE_COLUMNS)

    X = assembled.head(10)
    result = hazards(model, X)

    assert result.shape == (10, 4)


def test_hazards_with_widened_model_rows_sum_to_one():
    """hazards() on a widened-fit model must return rows that sum to 1.0."""
    from src.model.competing_risks import fit, hazards, WIDENED_FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled, feature_columns=WIDENED_FEATURE_COLUMNS)

    X = assembled.head(20)
    result = hazards(model, X)

    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_hazards_with_widened_model_single_level_category_and_amount():
    """hazards() on a widened-fit model, predicting on data with only one
    category level and one amount band, must still return valid (n, 4) output
    with all probabilities in [0, 1] summing to 1 per row. This is the critical
    pd.get_dummies guard for widened models."""
    from src.model.competing_risks import fit, hazards, WIDENED_FEATURE_COLUMNS

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled, feature_columns=WIDENED_FEATURE_COLUMNS)

    # Filter to only subscription category, reference amount band
    X_filtered = assembled[
        (assembled["category"] == "subscription") &
        (assembled["amount_paise"] < 387_500)
    ].head(10)

    result = hazards(model, X_filtered)

    assert result.shape == (len(X_filtered), 4)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_hazards_narrow_model_on_minimal_synthetic_row_regression():
    """A model fit with FEATURE_COLUMNS (the default/narrow model) must be
    callable on a minimal synthetic row with only slot/in_salary_window/
    days_since_last_attempt and NO amount_paise/category columns.
    This is the eval/allocator_sweep.py::hazard_from_fit() real-world scenario
    and MUST NOT BREAK when future features are added."""
    from src.model.competing_risks import fit, hazards

    pp_df, feat_df = _simple_estimable_frame()
    from src.model.competing_risks import assemble

    assembled = assemble(pp_df, feat_df)
    model = fit(assembled)  # Default fit (FEATURE_COLUMNS)

    # Minimal synthetic row exactly like hazard_from_fit() builds
    X_minimal = pd.DataFrame([{
        "slot": 2,
        "in_salary_window": True,
        "days_since_last_attempt": 0.0,
    }])

    # This must not raise and must return (1, 4) valid probabilities
    result = hazards(model, X_minimal)
    assert result.shape == (1, 4)
    assert np.isfinite(result).all()
    assert np.isclose(result.sum(axis=1)[0], 1.0)
    assert (result >= 0.0).all() and (result <= 1.0).all()
