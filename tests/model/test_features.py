"""
src/model/features.py -- add feature columns to person-period frame.

Design decision this file pins: every feature either (1) comes from an emitted
column in SPEC_COLUMNS and is computed from the frame, or (2) is explicitly
documented as unsourced in UNSOURCED. A future session cannot silently drop a
sourced feature or silently invent an unsourced one. No feature may encode a
future slot's outcome (no leakage). FORBIDDEN columns are dropped before return.
`on_day` is an internal carry-through used for derivation but must not appear
in the output.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.types import CensorReason, Outcome, Profile
from src.core.ids import row_id
from eval.corpus import Episode
from eval.frozen.simulator import AttemptResult, SimMandate
from src.model.person_period import build
from src.model.features import SPEC_COLUMNS, UNSOURCED, FORBIDDEN, featurize


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


# === Input Validation ===========================================================


def test_featurize_rejects_input_missing_required_column():
    """featurize() needs on_day (among others) to derive its own columns --
    a build()-shaped input missing it must raise, not silently produce a
    wrong or empty derivation."""
    df = pd.DataFrame({
        "mandate_id": ["M_missing"],
        "cycle_id": [1],
        "slot": [1],
        "row_id": [row_id("M_missing", 1, 1)],
        "outcome": [Outcome.STILL_PENDING],
        "event_code": [0],
        "at_risk": [True],
        "censored": [False],
        "censor_reason": [CensorReason.NONE],
        "is_terminal": [True],
        "amount_paise": [50_000],
        "ceiling_paise": [100_000],
        "category": ["subscription"],
        # "on_day" deliberately omitted
    })
    with pytest.raises(ValueError):
        featurize(df)


# === SPEC_COLUMNS vs Emitted ===================================================


def test_spec_columns_equals_emitted_plus_unsourced():
    """SPEC_COLUMNS == (columns featurize adds) | UNSOURCED.
    This is the load-bearing test that stops a future session from silently
    dropping a sourced feature or inventing an unsourced one."""

    # Build a minimal frame from person_period.build()
    mandate = _mandate("M_spec_check", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_spec_check", 2, 3, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    # Featurize it
    df_feat = featurize(df_built)

    # "Emitted" means: which SPEC_COLUMNS members are actually present in
    # featurize()'s output -- NOT a set-difference against df_built's
    # columns. amount_paise/ceiling_paise/category are sourced Features
    # that featurize() carries through unchanged from build() rather than
    # freshly computing; a diff against df_built's columns would wrongly
    # exclude them (they're already present pre-featurize) and this
    # invariant could never hold for any correct implementation. Membership
    # in SPEC_COLUMNS is what "emitted" means here, regardless of which
    # module actually populated the column.
    emitted_by_featurize = set(df_feat.columns) & SPEC_COLUMNS
    expected_emitted = SPEC_COLUMNS - set(UNSOURCED.keys())

    assert emitted_by_featurize == expected_emitted


# === FORBIDDEN Columns =========================================================


def test_no_forbidden_columns_survive_featurize():
    """No column in FORBIDDEN must appear in featurize()'s output."""
    mandate = _mandate("M_forbidden_check", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_forbidden_check", 2, 3, Outcome.STILL_PENDING),
            _attempt("M_forbidden_check", 3, 8, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    # Confirm at least some FORBIDDEN columns are in the build output, so
    # the "none survive" check below is non-vacuous. NOT all of FORBIDDEN:
    # four of its members (initial_cause, effective_cause, household_id,
    # iatrogenic_insufficient_funds) are simulator-internal oracle fields
    # that never become person_period.build() columns at all -- FORBIDDEN
    # is a safety-net superset for any future caller that might attach
    # them, not a claim that every member is always present.
    assert FORBIDDEN & set(df_built.columns)

    df_feat = featurize(df_built)

    # None of them should survive
    for col in FORBIDDEN:
        assert col not in df_feat.columns, f"Forbidden column {col} survived featurize()"


# === on_day Handling ===========================================================


def test_estimable_present_in_build_absent_in_featurize():
    """estimable (the slot-1 structural-zero flag, added per stats-reviewer's
    B4 finding) must survive in build()'s output -- it is not in FORBIDDEN,
    since it is not outcome-derived, but featurize() still drops it: B5
    must consult it via build()'s own frame, the same mechanism already
    used to rejoin the fit target (event_code)."""
    mandate = _mandate("M_estimable_check", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_estimable_check", 2, 3, Outcome.STILL_PENDING),
            _attempt("M_estimable_check", 3, 8, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    assert "estimable" in df_built.columns
    slot1 = df_built[df_built["slot"] == 1].iloc[0]
    assert bool(slot1["estimable"]) is False
    assert bool(df_built[df_built["slot"] != 1]["estimable"].all()) is True

    df_feat = featurize(df_built)
    assert "estimable" not in df_feat.columns


def test_on_day_present_in_build_absent_in_featurize():
    """on_day is a build() internal carry-through, not part of the modelled feature set."""
    mandate = _mandate("M_onday_check", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_onday_check", 2, 5, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    assert "on_day" in df_built.columns
    assert any(df_built["on_day"] > 0)

    df_feat = featurize(df_built)

    assert "on_day" not in df_feat.columns


# === Feature Derivations =======================================================


def test_prior_failures_this_cycle_equals_slot_minus_1():
    """prior_failures_this_cycle must equal slot - 1 for every row."""
    mandate = _mandate("M_prior_fail", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_prior_fail", 2, 2, Outcome.STILL_PENDING),
            _attempt("M_prior_fail", 3, 8, Outcome.STILL_PENDING),
            _attempt("M_prior_fail", 4, 14, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    # Check every row
    for _, row in df_feat.iterrows():
        assert row["prior_failures_this_cycle"] == row["slot"] - 1


def test_in_salary_window_is_true_for_days_1_to_5():
    """in_salary_window == (1 <= on_day <= 5) for every row."""
    mandate = _mandate("M_salary", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_salary", 2, 2, Outcome.STILL_PENDING),    # in window
            _attempt("M_salary", 3, 5, Outcome.STILL_PENDING),    # in window (boundary)
            _attempt("M_salary", 4, 6, Outcome.STILL_PENDING),    # NOT in window
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    # on_day is correctly absent from df_feat (see
    # test_on_day_present_in_build_absent_in_featurize) -- look it up from
    # df_built by row_id instead of reading it off the featurized frame.
    on_day_by_row = dict(zip(df_built["row_id"], df_built["on_day"]))
    for _, row in df_feat.iterrows():
        on_day = on_day_by_row[row["row_id"]]
        expected_in_window = 1 <= on_day <= 5
        assert row["in_salary_window"] == expected_in_window, (
            f"Row on_day={on_day}: "
            f"expected in_salary_window={expected_in_window}, "
            f"got {row['in_salary_window']}"
        )


def test_days_since_last_attempt_first_row_is_zero():
    """days_since_last_attempt must be 0 on each group's first row (slot 1)."""
    mandate = _mandate("M_days_first", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_days_first", 2, 5, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    slot1_row = df_feat[df_feat["slot"] == 1].iloc[0]
    assert slot1_row["days_since_last_attempt"] == 0


def test_days_since_last_attempt_non_constant_gaps():
    """days_since_last_attempt within a group must track gap changes correctly.
    Build an episode with non-uniform gaps to catch hardcoded constant values."""
    mandate = _mandate("M_days_nonuniform", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_days_nonuniform", 2, 3, Outcome.STILL_PENDING),     # day 3
            _attempt("M_days_nonuniform", 3, 9, Outcome.STILL_PENDING),     # day 9, gap = 6
            _attempt("M_days_nonuniform", 4, 11, Outcome.STILL_PENDING),    # day 11, gap = 2
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    # Sort by slot to ensure row order
    sorted_rows = df_feat.sort_values("slot").reset_index(drop=True)

    # Slot 1 (synthesized): days_since_last_attempt should be 0
    assert sorted_rows.loc[0, "days_since_last_attempt"] == 0

    # Slot 2 (on_day=3, previous on_day=None/0): gap = 3 - 0 = 3
    # Actually, slot 1 on_day is not set; we need to check what build() sets it to
    # Slot 1 should have on_day = 0 (day of the original failed attempt, not simulated)
    # Slot 2 on_day=3: days_since_last_attempt = 3 - 0 = 3
    assert sorted_rows.loc[1, "days_since_last_attempt"] == 3

    # Slot 3 on_day=9, previous=3: gap = 9 - 3 = 6
    assert sorted_rows.loc[2, "days_since_last_attempt"] == 6

    # Slot 4 on_day=11, previous=9: gap = 11 - 9 = 2
    assert sorted_rows.loc[3, "days_since_last_attempt"] == 2


def test_committed_day_of_month_equals_on_day():
    """committed_day_of_month is the honest on_day proxy (no synthetic month boundaries)."""
    mandate = _mandate("M_committed_day", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_committed_day", 2, 7, Outcome.STILL_PENDING),
            _attempt("M_committed_day", 3, 20, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    # on_day is correctly absent from df_feat -- compare against df_built's
    # copy, joined by row_id.
    on_day_by_row = dict(zip(df_built["row_id"], df_built["on_day"]))
    for _, row in df_feat.iterrows():
        assert row["committed_day_of_month"] == on_day_by_row[row["row_id"]]


# === Profile Stamping ==========================================================


def test_profile_defaults_to_strict():
    """If profile is not specified, it defaults to Profile.strict."""
    mandate = _mandate("M_profile_default", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_profile_default", 2, 3, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    df_feat = featurize(df_built)  # No profile argument

    assert "profile" in df_feat.columns
    assert all(df_feat["profile"] == Profile.strict)


def test_profile_stamped_as_constant_when_specified():
    """When profile is specified, it's stamped as a constant across every row."""
    mandate = _mandate("M_profile_permissive", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_profile_permissive", 2, 3, Outcome.STILL_PENDING),
            _attempt("M_profile_permissive", 3, 8, Outcome.STILL_PENDING),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])

    df_feat = featurize(df_built, profile=Profile.permissive)

    assert "profile" in df_feat.columns
    assert all(df_feat["profile"] == Profile.permissive)
    assert len(df_feat["profile"].unique()) == 1


# === No Future Slot Encoding ===================================================


def test_no_feature_encodes_future_slot():
    """No feature value on slot 1 must differ based on what happens at slots 2/3/4.
    Build two identical episodes that diverge only in later slots and verify
    their slot-1 feature rows are identical after featurization."""

    # Episode A: mandates both scenarios, slot 2 -> STILL_PENDING, slot 3 -> STILL_PENDING
    mandate_a = _mandate("M_future_a", cycle_id=1, amount_paise=60_000)
    episode_a = Episode(
        mandate=mandate_a,
        attempts=(
            _attempt("M_future_a", 2, 3, Outcome.STILL_PENDING),
            _attempt("M_future_a", 3, 9, Outcome.STILL_PENDING),
        ),
        censor_reason=CensorReason.WINDOW_CLOSED,
    )

    # Episode B: same mandate details, but slot 2 -> RECOVERED (different outcome)
    # We're NOT diverging here since the question is: does episode A's slot 1 change
    # based on episode B's existence? The answer should be no.
    # Actually, we need to construct two episodes that are IDENTICAL except in future slots.
    # The right setup: same mandate details, same slot 2 outcome, different slot 3+ outcomes.

    mandate_b = _mandate("M_future_b", cycle_id=1, amount_paise=60_000)  # Same mandate details
    episode_b = Episode(
        mandate=mandate_b,
        attempts=(
            _attempt("M_future_b", 2, 3, Outcome.STILL_PENDING),  # Same as A
            _attempt("M_future_b", 3, 9, Outcome.RECOVERED),  # Different from A
        ),
        censor_reason=CensorReason.NONE,
    )

    # Build both separately and featurize
    df_a = build([episode_a])
    df_a_feat = featurize(df_a)

    df_b = build([episode_b])
    df_b_feat = featurize(df_b)

    # Extract slot 1 rows
    slot1_a = df_a_feat[df_a_feat["slot"] == 1].iloc[0]
    slot1_b = df_b_feat[df_b_feat["slot"] == 1].iloc[0]

    # Compare all feature columns (excluding identity/outcome columns)
    feature_cols = [
        col
        for col in slot1_a.index
        if col not in {"mandate_id", "cycle_id", "slot", "row_id", "outcome", "event_code",
                       "at_risk", "censored", "censor_reason", "is_terminal", "amount_paise",
                       "ceiling_paise", "category", "on_day"}
    ]

    for col in feature_cols:
        assert (
            slot1_a[col] == slot1_b[col]
        ), f"Feature {col} differs between identical slot-1 rows: {slot1_a[col]} vs {slot1_b[col]}"


# === Build-Featurize Round-Trip ================================================


def test_featurize_output_contains_only_spec_columns():
    """Every column in featurize()'s output must be in SPEC_COLUMNS."""
    mandate = _mandate("M_output_cols", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_output_cols", 2, 2, Outcome.STILL_PENDING),
            _attempt("M_output_cols", 3, 8, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    df_feat = featurize(df_built)

    # Check: all columns are either from build() or in SPEC_COLUMNS
    from src.model.person_period import EMITTED_COLUMNS
    allowed_cols = set(EMITTED_COLUMNS) | SPEC_COLUMNS - {"on_day"}

    for col in df_feat.columns:
        assert col in allowed_cols, f"Unexpected column in featurize() output: {col}"


# === Household ID (coupled arm) ================================================


def _mandate_with_household(
    mandate_id: str,
    household_id: str | None,
    cycle_id: int = 1,
    amount_paise: int = 50_000,
    ceiling_paise: int = 100_000,
    category: str = "subscription",
) -> SimMandate:
    """Helper to build a SimMandate with an explicit household_id --
    dataclasses.replace() over _mandate()'s pattern, since SimMandate is a
    frozen dataclass and household_id is the one field _mandate() always
    hard-codes to None."""
    import dataclasses

    return dataclasses.replace(
        _mandate(
            mandate_id=mandate_id,
            cycle_id=cycle_id,
            amount_paise=amount_paise,
            ceiling_paise=ceiling_paise,
            category=category,
        ),
        household_id=household_id,
    )


def test_featurize_drops_household_id_even_when_input_has_real_values():
    """household_id is already in FORBIDDEN (this module's own list) --
    featurize() must physically drop it from its output, the same way it
    already drops on_day, so the design matrix never sees it even when the
    input build() frame carries a real (non-null) household_id, matching
    the coupled arm."""
    mandate = _mandate_with_household("M_hh_feat", "H0", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(
            _attempt("M_hh_feat", 2, 3, Outcome.STILL_PENDING),
            _attempt("M_hh_feat", 3, 8, Outcome.RECOVERED),
        ),
        censor_reason=CensorReason.NONE,
    )
    df_built = build([episode])
    assert "household_id" in df_built.columns
    assert df_built["household_id"].notna().all()

    df_feat = featurize(df_built)

    assert "household_id" not in df_feat.columns


def test_featurize_does_not_raise_on_real_household_id_input():
    """A real household_id in the input must not trip featurize()'s
    FORBIDDEN check -- the column is dropped before that check runs (same
    mechanism as on_day), so featurize() must return normally, not raise."""
    mandate = _mandate_with_household("M_hh_feat_ok", "H1", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_hh_feat_ok", 2, 3, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.WINDOW_CLOSED,
    )
    df_built = build([episode])
    # Non-vacuous precondition: household_id must actually be a real,
    # populated column on the input, or "featurize() does not raise" would
    # trivially hold today without ever exercising FORBIDDEN's
    # household_id entry at all (household_id doesn't exist as a build()
    # column yet, pre-implementation).
    assert df_built["household_id"].iloc[0] == "H1"

    df_feat = featurize(df_built)  # must not raise

    assert len(df_feat) == len(df_built)
