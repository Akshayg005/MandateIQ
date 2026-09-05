"""
src/model/splits.py -- mandate-level grouped FOUR-way split.

Design decision this file pins: splits are grouped by mandate_id (never row-level),
mandates never straddle splits, and disjointness of mandate_id sets is enforced
by the implementation (raised, not asserted). This is the only split ever applied
to the training corpus; the frozen evaluation batch is never split. FOUR frames,
not three: calib_iso (fits isotonic) and calib_conf (supplies the conformal
quantile) are disjoint mandate sets, per the statistics review's B4 finding that a
shared calib set breaks conformal exchangeability and silently narrows every
prediction set below its stated coverage. Proportions are 70/10/10/10.
"""
from __future__ import annotations

import pytest
import pandas as pd

from src.core.types import CensorReason, Outcome, Profile
from src.core.ids import row_id
from eval.corpus import Episode
from eval.frozen.simulator import AttemptResult, SimMandate
from src.model.person_period import build
from src.model.features import featurize
from src.model.splits import (
    split,
    SplitIntegrityError,
    TRAIN_FRAC,
    CALIB_ISO_FRAC,
    CALIB_CONF_FRAC,
    TEST_FRAC,
)


def _mandate(
    mandate_id: str,
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


def _synthetic_frame(n_mandates: int = 40) -> pd.DataFrame:
    """Build a synthetic featurized frame with n_mandates distinct mandates.
    Each mandate gets 1 cycle with 1-4 slots (varies to simulate different outcomes).
    Returns a featurize(build(...)) frame ready for split()."""

    episodes = []
    for i in range(n_mandates):
        mandate_id = f"M_{i:04d}"
        mandate = _mandate(mandate_id, cycle_id=1)

        # Vary slot count by mandate to create realistic data
        n_slots = (i % 3) + 2  # 2, 3, or 4 attempts
        attempts = []
        for slot in range(2, n_slots + 1):
            on_day = 1 + (slot - 1) * 3  # Spacing out days
            # Vary outcomes
            if slot == n_slots and i % 5 == 0:
                outcome = Outcome.RECOVERED
            elif slot == n_slots and i % 7 == 0:
                outcome = Outcome.DEAD
            else:
                outcome = Outcome.STILL_PENDING

            attempts.append(_attempt(mandate_id, slot, on_day, outcome))

        censor_reason = CensorReason.NONE if outcome != Outcome.STILL_PENDING else CensorReason.BUDGET_EXHAUSTED if n_slots == 4 else CensorReason.WINDOW_CLOSED
        episodes.append(Episode(mandate=mandate, attempts=tuple(attempts), censor_reason=censor_reason))

    # Build and featurize
    df_built = build(episodes)
    df_feat = featurize(df_built)

    return df_feat


# === Basic Mechanics ===========================================================


def test_split_returns_four_frames():
    """split() must return exactly four DataFrames (train, calib_iso, calib_conf, test)."""
    df = _synthetic_frame(n_mandates=30)
    train, calib_iso, calib_conf, test = split(df, seed=1)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(calib_iso, pd.DataFrame)
    assert isinstance(calib_conf, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) > 0
    assert len(calib_iso) > 0
    assert len(calib_conf) > 0
    assert len(test) > 0


# === Disjointness ==============================================================


def test_disjoint_mandate_ids_train_calib_iso():
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(train["mandate_id"]) & set(calib_iso["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both train and calib_iso"


def test_disjoint_mandate_ids_train_calib_conf():
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(train["mandate_id"]) & set(calib_conf["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both train and calib_conf"


def test_disjoint_mandate_ids_train_test():
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(train["mandate_id"]) & set(test["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both train and test"


def test_disjoint_mandate_ids_calib_iso_calib_conf():
    """The finding this pair specifically guards against: calib_iso and
    calib_conf must be genuinely different mandates, or the conformal
    quantile is computed from residuals that are no longer out-of-sample
    for isotonic -- see this module's docstring."""
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(calib_iso["mandate_id"]) & set(calib_conf["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both calib_iso and calib_conf"


def test_disjoint_mandate_ids_calib_iso_test():
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(calib_iso["mandate_id"]) & set(test["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both calib_iso and test"


def test_disjoint_mandate_ids_calib_conf_test():
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=42)

    intersection = set(calib_conf["mandate_id"]) & set(test["mandate_id"])
    assert len(intersection) == 0, f"Found {len(intersection)} mandates in both calib_conf and test"


# === Complete Partition ========================================================


def test_every_row_appears_exactly_once():
    """Every row from the input must appear in exactly one of the four output frames."""
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=123)

    input_row_ids = set(df["row_id"].unique())
    output_row_ids = (
        set(train["row_id"].unique())
        | set(calib_iso["row_id"].unique())
        | set(calib_conf["row_id"].unique())
        | set(test["row_id"].unique())
    )

    assert input_row_ids == output_row_ids, (
        f"Row count mismatch: input {len(input_row_ids)}, output {len(output_row_ids)}"
    )

    train_rows = set(train["row_id"])
    calib_iso_rows = set(calib_iso["row_id"])
    calib_conf_rows = set(calib_conf["row_id"])
    test_rows = set(test["row_id"])

    all_pairs = [
        (train_rows, calib_iso_rows), (train_rows, calib_conf_rows), (train_rows, test_rows),
        (calib_iso_rows, calib_conf_rows), (calib_iso_rows, test_rows),
        (calib_conf_rows, test_rows),
    ]
    for a, b in all_pairs:
        assert len(a & b) == 0


# === Mandate Grouping ==========================================================


def test_mandate_never_straddles_split():
    """No mandate's rows should be split across multiple output frames."""
    df = _synthetic_frame(n_mandates=40)
    train, calib_iso, calib_conf, test = split(df, seed=99)

    for mandate_id in df["mandate_id"].unique():
        input_mask = df["mandate_id"] == mandate_id
        n_input = input_mask.sum()

        counts = {
            "train": (train["mandate_id"] == mandate_id).sum(),
            "calib_iso": (calib_iso["mandate_id"] == mandate_id).sum(),
            "calib_conf": (calib_conf["mandate_id"] == mandate_id).sum(),
            "test": (test["mandate_id"] == mandate_id).sum(),
        }
        total_output = sum(counts.values())

        assert (
            total_output == n_input
        ), f"Mandate {mandate_id}: input {n_input} rows, found {total_output} across splits"

        non_zero_splits = sum(1 for c in counts.values() if c > 0)
        assert (
            non_zero_splits == 1
        ), f"Mandate {mandate_id}: found in {non_zero_splits} splits (expected 1): {counts}"


def test_all_cycles_of_mandate_land_in_same_split():
    """If a mandate has multiple cycles, all cycles must land in the same split."""
    episodes = []
    for i in range(20):
        mandate_id = f"M_{i:03d}"

        mandate_c1 = _mandate(mandate_id, cycle_id=1)
        ep_c1 = Episode(
            mandate=mandate_c1,
            attempts=(_attempt(mandate_id, 2, 2, Outcome.STILL_PENDING),),
            censor_reason=CensorReason.NONE,
        )
        episodes.append(ep_c1)

        mandate_c2 = _mandate(mandate_id, cycle_id=2)
        ep_c2 = Episode(
            mandate=mandate_c2,
            attempts=(_attempt(mandate_id, 2, 2, Outcome.STILL_PENDING),),
            censor_reason=CensorReason.NONE,
        )
        episodes.append(ep_c2)

    df = featurize(build(episodes))
    train, calib_iso, calib_conf, test = split(df, seed=77)

    frames = {"train": train, "calib_iso": calib_iso, "calib_conf": calib_conf, "test": test}

    for mandate_id in df["mandate_id"].unique():
        mandate_rows = df[df["mandate_id"] == mandate_id]
        cycles = set(mandate_rows["cycle_id"].unique())

        homes = [name for name, f in frames.items() if mandate_id in f["mandate_id"].values]
        assert len(homes) == 1, f"Mandate {mandate_id} found in {homes}, expected exactly 1"
        home = frames[homes[0]]
        for cycle in cycles:
            assert ((home["mandate_id"] == mandate_id) & (home["cycle_id"] == cycle)).sum() > 0


# === Reproducibility ===========================================================


def test_same_seed_produces_identical_partition():
    """Calling split() twice with the same seed must produce identical (mandate-level) partitions."""
    df = _synthetic_frame(n_mandates=45)

    r1 = split(df, seed=55)
    r2 = split(df, seed=55)

    for f1, f2 in zip(r1, r2):
        assert set(f1["mandate_id"]) == set(f2["mandate_id"])


def test_different_seeds_produce_different_partitions():
    """Different seeds must produce different partitions (at least one mandate ends up in a different split)."""
    df = _synthetic_frame(n_mandates=50)

    train1, calib_iso1, calib_conf1, test1 = split(df, seed=11)
    train2, calib_iso2, calib_conf2, test2 = split(df, seed=22)

    assert set(train1["mandate_id"]) != set(train2["mandate_id"]), (
        "Different seeds produced identical train sets (statistically suspicious)"
    )


# === Proportions ===============================================================


def test_roughly_correct_proportions():
    """The four splits should be roughly TRAIN_FRAC/CALIB_ISO_FRAC/CALIB_CONF_FRAC/
    TEST_FRAC in mandate count. Tolerance is generous (within a few mandates)
    since this is a small-n synthetic test."""
    df = _synthetic_frame(n_mandates=50)
    train, calib_iso, calib_conf, test = split(df, seed=123)

    total_mandates = len(df["mandate_id"].unique())
    counts = {
        "train": len(train["mandate_id"].unique()),
        "calib_iso": len(calib_iso["mandate_id"].unique()),
        "calib_conf": len(calib_conf["mandate_id"].unique()),
        "test": len(test["mandate_id"].unique()),
    }
    expected = {
        "train": int(total_mandates * TRAIN_FRAC),
        "calib_iso": int(total_mandates * CALIB_ISO_FRAC),
        "calib_conf": int(total_mandates * CALIB_CONF_FRAC),
        "test": int(total_mandates * TEST_FRAC),
    }

    tolerance = 5
    for name in counts:
        assert abs(counts[name] - expected[name]) <= tolerance, (
            f"{name}: expected ~{expected[name]}, got {counts[name]}"
        )


# === Row Counts ================================================================


def test_row_counts_preserved_across_splits():
    """Total row count across all four splits must equal the input row count."""
    df = _synthetic_frame(n_mandates=35)
    input_rows = len(df)

    train, calib_iso, calib_conf, test = split(df, seed=66)
    output_rows = len(train) + len(calib_iso) + len(calib_conf) + len(test)

    assert output_rows == input_rows, (
        f"Row count mismatch: input {input_rows}, output {output_rows}"
    )


# === Fraction constants =========================================================


def test_fractions_sum_to_one():
    """The four FRAC constants must sum to 1.0 -- if they don't, split()'s
    relative-fraction arithmetic silently produces the wrong proportions."""
    assert TRAIN_FRAC + CALIB_ISO_FRAC + CALIB_CONF_FRAC + TEST_FRAC == pytest.approx(1.0)


# === Household-Aware Grouping (coupled arm) ====================================


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


def _synthetic_pp_frame_with_households(n_households: int = 12) -> pd.DataFrame:
    """Like _synthetic_frame(), but every mandate belongs to a real
    household_id, in households of 2-3 mandates each (alternating) -- and
    the frame is returned straight from person_period.build() rather than
    run through featurize(). featurize() correctly drops household_id
    from its own output (see tests/model/test_features.py's
    test_featurize_drops_household_id_even_when_input_has_real_values),
    and the grouping tests below need household_id genuinely present on
    the split() output frames to check containment. split()'s own
    docstring states it accepts either a build() or a featurize() output,
    so this is a legitimate `df` for split() even though every other
    helper in this file passes a featurized one."""
    episodes = []
    i = 0
    for h in range(n_households):
        household_id = f"H{h:03d}"
        household_size = 2 + (h % 2)  # alternates 2, 3, 2, 3, ...
        for _ in range(household_size):
            mandate_id = f"M_hh_{i:04d}"
            mandate = _mandate_with_household(mandate_id, household_id, cycle_id=1)

            n_slots = (i % 3) + 2  # 2, 3, or 4 attempts
            attempts = []
            for slot in range(2, n_slots + 1):
                on_day = 1 + (slot - 1) * 3  # Spacing out days
                if slot == n_slots and i % 5 == 0:
                    outcome = Outcome.RECOVERED
                elif slot == n_slots and i % 7 == 0:
                    outcome = Outcome.DEAD
                else:
                    outcome = Outcome.STILL_PENDING

                attempts.append(_attempt(mandate_id, slot, on_day, outcome))

            censor_reason = CensorReason.NONE if outcome != Outcome.STILL_PENDING else CensorReason.BUDGET_EXHAUSTED if n_slots == 4 else CensorReason.WINDOW_CLOSED
            episodes.append(
                Episode(mandate=mandate, attempts=tuple(attempts), censor_reason=censor_reason)
            )
            i += 1

    return build(episodes)


def test_split_group_key_matches_default_bit_identical_on_synthetic_frame():
    """THE bit-identity test -- most important test in this batch. B5's
    already-reported numbers (held-out log-loss, calibration) are computed
    on exactly this shape: a featurize()'d nominal-arm frame passed to
    split(). featurize() correctly drops household_id from its own output
    (see tests/model/test_features.py's
    test_featurize_drops_household_id_even_when_input_has_real_values), so
    there is no live household_id column on a featurized frame to read a
    group_key from directly -- but every mandate _synthetic_frame()
    generates goes through plain _mandate() (household_id=None), so the
    household-id-falling-back-to-mandate-id construction reduces, for this
    frame, to exactly mandate_id: group_key=df["mandate_id"] IS the value
    that construction would produce here, just without a live column to
    read it from.
    (test_split_group_key_fillna_construction_is_a_noop_when_household_id_
    all_null, below, exercises the literal household_id.fillna(mandate_id)
    expression against a real household_id column instead, on a
    person_period.build() frame.)

    split() with that group_key must be bit-identical to split() with none
    -- proving the household-aware grouping change never touches any
    already-reported nominal-arm number (root DESIGN.md invariant 4 on
    eval/frozen/ immutability is why this must be provable, not just
    plausible).

    Covers a representative subset of seeds rather than the full
    range(20) to keep this a fast unit test; 5 seeds spanning the range is
    enough to catch a systematic bug (e.g. group_key silently reordering
    rows) that a single seed could hide by chance."""
    df = _synthetic_frame(n_mandates=40)
    group_key = df["mandate_id"].copy()

    for seed in [0, 1, 5, 13, 19]:
        default_result = split(df, seed=seed)
        grouped_result = split(df, seed=seed, group_key=group_key)

        for name, default_frame, grouped_frame in zip(
            ("train", "calib_iso", "calib_conf", "test"), default_result, grouped_result
        ):
            try:
                pd.testing.assert_frame_equal(
                    default_frame.reset_index(drop=True),
                    grouped_frame.reset_index(drop=True),
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"seed={seed}, frame={name}: not bit-identical -- {exc}"
                ) from exc


def test_split_group_key_fillna_construction_is_a_noop_when_household_id_all_null():
    """Directly exercises the caller-side group_key construction pattern
    (household_id.fillna(mandate_id)) against a real household_id column,
    on a person_period.build() frame (which -- unlike a featurize()
    output -- still carries household_id; split() accepts either shape
    per its own docstring). Every mandate here goes through plain
    _mandate() (household_id=None), so the constructed group_key must
    literally equal mandate_id row for row, and split() with that
    group_key must be bit-identical to split() with none."""
    episodes = []
    for i in range(30):
        mandate_id = f"M_nullhh_{i:04d}"
        mandate = _mandate(mandate_id, cycle_id=1)
        n_slots = (i % 3) + 2
        attempts = tuple(
            _attempt(mandate_id, slot, 1 + (slot - 1) * 3, Outcome.STILL_PENDING)
            for slot in range(2, n_slots + 1)
        )
        censor_reason = (
            CensorReason.BUDGET_EXHAUSTED if n_slots == 4 else CensorReason.WINDOW_CLOSED
        )
        episodes.append(Episode(mandate=mandate, attempts=attempts, censor_reason=censor_reason))

    df = build(episodes)
    # Non-vacuous precondition: household_id must actually be a real,
    # populated (if all-null) column here, or this test would trivially
    # KeyError before it could exercise anything -- confirming that,
    # rather than papering over it, is the point.
    assert "household_id" in df.columns
    assert df["household_id"].isna().all()

    group_key = df["household_id"].fillna(df["mandate_id"])

    for seed in [0, 1, 5, 13, 19]:
        default_result = split(df, seed=seed)
        grouped_result = split(df, seed=seed, group_key=group_key)

        for name, default_frame, grouped_frame in zip(
            ("train", "calib_iso", "calib_conf", "test"), default_result, grouped_result
        ):
            try:
                pd.testing.assert_frame_equal(
                    default_frame.reset_index(drop=True),
                    grouped_frame.reset_index(drop=True),
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"seed={seed}, frame={name}: not bit-identical -- {exc}"
                ) from exc


def test_household_group_key_keeps_household_within_one_split():
    """With group_key built from household_id (falling back to
    mandate_id), no household's mandates may be split across more than
    one of the four output frames -- mirrors
    test_mandate_never_straddles_split's structure, checked at household
    level instead of mandate level."""
    df = _synthetic_pp_frame_with_households(n_households=12)
    group_key = df["household_id"].fillna(df["mandate_id"])

    train, calib_iso, calib_conf, test = split(df, seed=99, group_key=group_key)
    frames = {"train": train, "calib_iso": calib_iso, "calib_conf": calib_conf, "test": test}

    for household_id in df["household_id"].dropna().unique():
        counts = {
            name: int((frame["household_id"] == household_id).sum())
            for name, frame in frames.items()
        }
        n_input = int((df["household_id"] == household_id).sum())
        total_output = sum(counts.values())

        assert (
            total_output == n_input
        ), f"Household {household_id}: input {n_input} rows, found {total_output} across splits"

        non_zero_splits = sum(1 for c in counts.values() if c > 0)
        assert (
            non_zero_splits == 1
        ), f"Household {household_id}: found in {non_zero_splits} splits (expected 1): {counts}"


def test_mixed_household_and_null_frame_nulls_stay_independent_groups():
    """A frame mixing real households (2+ mandates sharing one household
    id) with household_id=None mandates: with
    group_key=household_id.fillna(mandate_id), the null-household
    mandates are each their own independent group (never straddling, and
    not coalesced into one giant group either), while the same-household
    mandates never straddle a split."""
    household_episodes = []
    i = 0
    for h in range(6):
        household_id = f"HM{h:02d}"
        for _ in range(2):
            mandate_id = f"M_hh_{i:04d}"
            mandate = _mandate_with_household(mandate_id, household_id, cycle_id=1)
            attempts = (_attempt(mandate_id, 2, 2 + (i % 4), Outcome.STILL_PENDING),)
            household_episodes.append(
                Episode(
                    mandate=mandate, attempts=attempts, censor_reason=CensorReason.WINDOW_CLOSED
                )
            )
            i += 1

    null_episodes = []
    for j in range(20):
        mandate_id = f"M_null_{j:04d}"
        mandate = _mandate_with_household(mandate_id, None, cycle_id=1)
        attempts = (_attempt(mandate_id, 2, 2 + (j % 4), Outcome.STILL_PENDING),)
        null_episodes.append(
            Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.WINDOW_CLOSED)
        )

    df = build(household_episodes + null_episodes)
    group_key = df["household_id"].fillna(df["mandate_id"])

    train, calib_iso, calib_conf, test = split(df, seed=7, group_key=group_key)
    frames = {"train": train, "calib_iso": calib_iso, "calib_conf": calib_conf, "test": test}

    # Same-household mandates never straddle a split.
    for household_id in df["household_id"].dropna().unique():
        counts = {
            name: int((frame["household_id"] == household_id).sum())
            for name, frame in frames.items()
        }
        non_zero_splits = sum(1 for c in counts.values() if c > 0)
        assert (
            non_zero_splits == 1
        ), f"Household {household_id}: found in {non_zero_splits} splits (expected 1): {counts}"

    # Null-household mandates never straddle either -- each is its own
    # independent group, exactly like today's mandate_id-only grouping.
    null_mandate_ids = df[df["household_id"].isna()]["mandate_id"].unique()
    for mandate_id in null_mandate_ids:
        counts = {
            name: int((frame["mandate_id"] == mandate_id).sum())
            for name, frame in frames.items()
        }
        non_zero_splits = sum(1 for c in counts.values() if c > 0)
        assert non_zero_splits == 1, (
            f"Null-household mandate {mandate_id}: found in {non_zero_splits} "
            f"splits (expected 1): {counts}"
        )

    # Sanity: null-household mandates must be genuinely independent
    # groups, not silently coalesced into one -- with 20 of them at
    # 70/10/10/10 proportions they should not all land in a single split.
    homes_touched = {
        name for name, frame in frames.items()
        if frame["mandate_id"].isin(null_mandate_ids).any()
    }
    assert len(homes_touched) > 1, (
        "all null-household mandates landed in a single split -- group_key "
        "may have coalesced them instead of treating each as its own "
        "independent group"
    )


def test_household_id_never_reaches_design_matrix_or_feature_columns():
    """Belt-and-suspenders regression guard: household_id must never
    appear in src.model.competing_risks._design_matrix()'s output
    columns, nor in a fitted HazardModel.feature_columns. This should
    already trivially hold once featurize() correctly drops the column
    (see tests/model/test_features.py's
    test_featurize_drops_household_id_even_when_input_has_real_values) --
    it is checked again here, at the far end of the pipeline, because
    household_id leaking into a design matrix would be a serious,
    easy-to-miss bug: nothing downstream raises when it happens, the
    model just silently fits against a column it should never have seen.

    Builds a minimal frame with real (non-null) household ids through the
    full build() -> featurize() -> assemble() -> fit() pipeline,
    mirroring tests/model/test_competing_risks.py's
    _simple_estimable_frame() shape (25 rows per (slot, in_salary_window)
    cell, cycling all 4 event_code classes, since fit() requires every
    class present)."""
    from src.model.competing_risks import assemble, fit, _design_matrix

    episodes = []
    idx = 0
    for slot in [2, 3, 4]:
        for in_window in [False, True]:
            for outcome_cycle in range(25):
                mandate_id = f"M_hh_est_{idx:04d}"
                household_id = f"H{idx % 5}"
                idx += 1

                if in_window:
                    on_day = 2 + outcome_cycle % 4
                else:
                    on_day = 6 + outcome_cycle % 10

                if outcome_cycle % 4 == 0:
                    outcome = Outcome.RECOVERED
                elif outcome_cycle % 4 == 1:
                    outcome = Outcome.DEAD
                elif outcome_cycle % 4 == 2:
                    outcome = Outcome.OPTED_OUT
                else:
                    outcome = Outcome.STILL_PENDING

                attempts = []
                for s in range(2, slot + 1):
                    day = on_day if s == slot else on_day - (slot - s) * 3
                    out = outcome if s == slot else Outcome.STILL_PENDING
                    attempts.append(_attempt(mandate_id, s, day, out))

                mandate = _mandate_with_household(mandate_id, household_id, cycle_id=1)
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

    pp_df = build(episodes)
    # Non-vacuous precondition: household_id must actually reach pp_df as
    # a real, populated column before this test can say anything about
    # whether it then LEAKS further downstream -- without this, the
    # assertions below would trivially hold even today, pre-
    # implementation, since neither _design_matrix() nor FEATURE_COLUMNS
    # mentions household_id by name regardless of what build() emits.
    assert "household_id" in pp_df.columns
    assert pp_df["household_id"].notna().all()

    feat_df = featurize(pp_df)
    assembled = assemble(pp_df, feat_df)

    design = _design_matrix(assembled)
    assert "household_id" not in design.columns

    model = fit(assembled)
    assert "household_id" not in model.feature_columns
