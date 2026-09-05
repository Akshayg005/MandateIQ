"""src/model/competing_risks.py -- SIM2 feature column enhancements (R1 Phase B).

Tests for the new SIM2_FEATURE_COLUMNS constant and _design_matrix() extensions
that support issuer_id, instrument_type, and mandate_age_days covariates.

Invariants tested:
1. SIM2_FEATURE_COLUMNS includes FEATURE_COLUMNS plus the new covariate columns
2. _design_matrix() raises ValueError with specific messages when required columns missing
3. _design_matrix() raises ValueError on unknown issuer/instrument values
4. Dummy encoding is correct (spot-check specific rows)
5. Existing FEATURE_COLUMNS path (no sim2 columns) remains unaffected (regression)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSim2FeatureColumns:
    """SIM2_FEATURE_COLUMNS constant is correctly defined."""

    def test_sim2_feature_columns_exists(self):
        """SIM2_FEATURE_COLUMNS must be defined and be a tuple."""
        from src.model.competing_risks import SIM2_FEATURE_COLUMNS

        assert isinstance(SIM2_FEATURE_COLUMNS, tuple)
        assert len(SIM2_FEATURE_COLUMNS) > 0

    def test_sim2_feature_columns_includes_base_features(self):
        """SIM2_FEATURE_COLUMNS must include all base FEATURE_COLUMNS."""
        from src.model.competing_risks import FEATURE_COLUMNS, SIM2_FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            assert col in SIM2_FEATURE_COLUMNS, (
                f"Base column {col} not in SIM2_FEATURE_COLUMNS"
            )

    def test_sim2_feature_columns_includes_issuer_dummies(self):
        """SIM2_FEATURE_COLUMNS must include issuer_<level> for non-reference levels."""
        from src.model.competing_risks import SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS

        # Reference is first level, so we expect dummies for levels 1, 2, 3
        expected_issuer_cols = [f"issuer_{level}" for level in ISSUER_LEVELS[1:]]
        for col in expected_issuer_cols:
            assert col in SIM2_FEATURE_COLUMNS, f"Missing {col}"

    def test_sim2_feature_columns_includes_instrument_dummies(self):
        """SIM2_FEATURE_COLUMNS must include instrument_<level> except upi_autopay."""
        from src.model.competing_risks import SIM2_FEATURE_COLUMNS
        from eval.sim2 import INSTRUMENT_LEVELS

        # upi_autopay (first level) is reference
        expected_instrument_cols = [
            f"instrument_{level}" for level in INSTRUMENT_LEVELS[1:]
        ]
        for col in expected_instrument_cols:
            assert col in SIM2_FEATURE_COLUMNS, f"Missing {col}"

    def test_sim2_feature_columns_includes_mandate_age_years(self):
        """SIM2_FEATURE_COLUMNS must include mandate_age_years."""
        from src.model.competing_risks import SIM2_FEATURE_COLUMNS

        assert "mandate_age_years" in SIM2_FEATURE_COLUMNS


class TestDesignMatrixMissingIssuer:
    """_design_matrix() must raise ValueError when issuer_id columns requested but column missing."""

    def test_design_matrix_raises_on_missing_issuer_id(self):
        """Requesting any issuer_* column without issuer_id raises ValueError."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
            # Intentionally missing issuer_id
        })

        with pytest.raises(ValueError, match="issuer_id") as exc_info:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        error_msg = str(exc_info.value)
        assert "issuer_id" in error_msg.lower()

    def test_design_matrix_error_message_names_issuer_id(self):
        """Error message must specifically name issuer_id as the missing column."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
        })

        try:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Error message must follow the pattern: "design matrix input is missing required column(s): ['issuer_id']"
            assert "issuer_id" in str(e).lower()


class TestDesignMatrixMissingInstrument:
    """_design_matrix() must raise ValueError when instrument_type columns requested but column missing."""

    def test_design_matrix_raises_on_missing_instrument_type(self):
        """Requesting any instrument_* column without instrument_type raises ValueError."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[0], ISSUER_LEVELS[1]],
            # Intentionally missing instrument_type
        })

        with pytest.raises(ValueError, match="instrument_type") as exc_info:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        error_msg = str(exc_info.value)
        assert "instrument_type" in error_msg.lower()


class TestDesignMatrixMissingMandateAge:
    """_design_matrix() must raise ValueError when mandate_age_years requested but mandate_age_days missing."""

    def test_design_matrix_raises_on_missing_mandate_age_days(self):
        """Requesting mandate_age_years without mandate_age_days raises ValueError."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[0], ISSUER_LEVELS[1]],
            "instrument_type": [INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[1]],
            # Intentionally missing mandate_age_days
        })

        with pytest.raises(ValueError, match="mandate_age_days") as exc_info:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        error_msg = str(exc_info.value)
        assert "mandate_age_days" in error_msg.lower()


class TestDesignMatrixUnknownIssuer:
    """_design_matrix() must raise ValueError on unknown issuer_id values."""

    def test_design_matrix_raises_on_unknown_issuer_value(self):
        """A value in issuer_id outside ISSUER_LEVELS must raise ValueError."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
            "issuer_id": [ISSUER_LEVELS[0], "unknown_issuer", ISSUER_LEVELS[1]],
            "instrument_type": [INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[1]],
            "mandate_age_days": [180, 180, 365],
        })

        with pytest.raises(ValueError, match="unknown|issuer") as exc_info:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        error_msg = str(exc_info.value)
        assert "unknown_issuer" in error_msg or "issuer" in error_msg.lower()


class TestDesignMatrixUnknownInstrument:
    """_design_matrix() must raise ValueError on unknown instrument_type values."""

    def test_design_matrix_raises_on_unknown_instrument_value(self):
        """A value in instrument_type outside INSTRUMENT_LEVELS must raise ValueError."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 3],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[0], ISSUER_LEVELS[1]],
            "instrument_type": [INSTRUMENT_LEVELS[0], "unknown_instrument", INSTRUMENT_LEVELS[1]],
            "mandate_age_days": [180, 180, 365],
        })

        with pytest.raises(ValueError, match="unknown|instrument") as exc_info:
            _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        error_msg = str(exc_info.value)
        assert "unknown_instrument" in error_msg or "instrument" in error_msg.lower()


class TestDesignMatrixDummyEncoding:
    """_design_matrix() correctly encodes dummy variables for sim2 columns."""

    def test_issuer_dummy_encoding_correct(self):
        """issuer_* dummies must be 1 for that issuer, 0 otherwise (reference omitted)."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 2],
            "in_salary_window": [1, 1, 1],
            "days_since_last_attempt": [1, 1, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[1], ISSUER_LEVELS[2]],
            "instrument_type": [INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0]],
            "mandate_age_days": [180, 180, 180],
        })

        result = _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        # Reference issuer (ISSUER_LEVELS[0]) should have 0 for all issuer_* dummies
        assert all(
            result.loc[0, [col for col in result.columns if col.startswith("issuer_")]]
            == 0
        ), "Reference issuer row should have all issuer_* = 0"

        # ISSUER_LEVELS[1] should have issuer_<ISSUER_LEVELS[1]> = 1, others = 0
        issuer_1_col = f"issuer_{ISSUER_LEVELS[1]}"
        assert result.loc[1, issuer_1_col] == 1
        other_issuer_cols = [
            col for col in result.columns
            if col.startswith("issuer_") and col != issuer_1_col
        ]
        assert all(result.loc[1, other_issuer_cols] == 0)

    def test_instrument_dummy_encoding_correct(self):
        """instrument_* dummies must be 1 for that instrument, 0 otherwise."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 2],
            "in_salary_window": [1, 1, 1],
            "days_since_last_attempt": [1, 1, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[0], ISSUER_LEVELS[0]],
            "instrument_type": [INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[1], INSTRUMENT_LEVELS[2]],
            "mandate_age_days": [180, 180, 180],
        })

        result = _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        # Reference instrument (upi_autopay at [0]) should have 0 for all instrument_* dummies
        assert all(
            result.loc[0, [col for col in result.columns if col.startswith("instrument_")]]
            == 0
        ), "Reference instrument row should have all instrument_* = 0"

        # INSTRUMENT_LEVELS[1] should have instrument_<INSTRUMENT_LEVELS[1]> = 1, others = 0
        instr_1_col = f"instrument_{INSTRUMENT_LEVELS[1]}"
        assert result.loc[1, instr_1_col] == 1
        other_instr_cols = [
            col for col in result.columns
            if col.startswith("instrument_") and col != instr_1_col
        ]
        assert all(result.loc[1, other_instr_cols] == 0)

    def test_mandate_age_years_computed_from_days(self):
        """mandate_age_years must equal mandate_age_days / 365.0."""
        from src.model.competing_risks import _design_matrix, SIM2_FEATURE_COLUMNS
        from eval.sim2 import ISSUER_LEVELS, INSTRUMENT_LEVELS

        df = pd.DataFrame({
            "slot": [2, 2, 2],
            "in_salary_window": [1, 1, 1],
            "days_since_last_attempt": [1, 1, 1],
            "issuer_id": [ISSUER_LEVELS[0], ISSUER_LEVELS[0], ISSUER_LEVELS[0]],
            "instrument_type": [INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0], INSTRUMENT_LEVELS[0]],
            "mandate_age_days": [0, 365, 730],
        })

        result = _design_matrix(df, columns=SIM2_FEATURE_COLUMNS)

        expected_years = np.array([0.0, 1.0, 2.0])
        actual_years = result["mandate_age_years"].values
        assert np.allclose(actual_years, expected_years), (
            f"Expected {expected_years}, got {actual_years}"
        )


class TestDesignMatrixRegression:
    """Existing FEATURE_COLUMNS path (no sim2 columns) must remain unaffected."""

    def test_feature_columns_path_unchanged(self):
        """_design_matrix() with FEATURE_COLUMNS (no sim2) must work unchanged."""
        from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 3, 4],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 3],
        })

        # This should not raise and should produce the four expected columns
        result = _design_matrix(df, columns=FEATURE_COLUMNS)

        assert len(result.columns) == len(FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_feature_columns_does_not_need_issuer_id(self):
        """_design_matrix(columns=FEATURE_COLUMNS) must work without issuer_id."""
        from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 3, 4],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 3],
            # No issuer_id
        })

        # Should not raise
        result = _design_matrix(df, columns=FEATURE_COLUMNS)
        assert len(result) == 3

    def test_feature_columns_does_not_need_instrument_type(self):
        """_design_matrix(columns=FEATURE_COLUMNS) must work without instrument_type."""
        from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 3, 4],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 3],
            # No instrument_type
        })

        # Should not raise
        result = _design_matrix(df, columns=FEATURE_COLUMNS)
        assert len(result) == 3

    def test_feature_columns_does_not_need_mandate_age_days(self):
        """_design_matrix(columns=FEATURE_COLUMNS) must work without mandate_age_days."""
        from src.model.competing_risks import _design_matrix, FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 3, 4],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 3],
            # No mandate_age_days
        })

        # Should not raise
        result = _design_matrix(df, columns=FEATURE_COLUMNS)
        assert len(result) == 3

    def test_widened_feature_columns_still_works(self):
        """WIDENED_FEATURE_COLUMNS path (amount/category, no sim2) must still work."""
        from src.model.competing_risks import _design_matrix, WIDENED_FEATURE_COLUMNS

        df = pd.DataFrame({
            "slot": [2, 3, 4],
            "in_salary_window": [1, 0, 1],
            "days_since_last_attempt": [1, 2, 3],
            "amount_paise": [100_000, 500_000, 1_200_000],
            "category": ["subscription", "insurance_premium", "mutual_fund"],
        })

        result = _design_matrix(df, columns=WIDENED_FEATURE_COLUMNS)

        # Must have all WIDENED columns
        for col in WIDENED_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column {col}"
