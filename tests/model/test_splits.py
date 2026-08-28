"""
src/model/splits.py -- mandate-level grouped FOUR-way split.

Design decision this file pins: splits are grouped by mandate_id (never row-level),
mandates never straddle splits, and disjointness of mandate_id sets is enforced
by the implementation (raised, not asserted). This is the only split ever applied
to the training corpus; the frozen evaluation batch is never split. FOUR frames,
not three: calib_iso (fits isotonic) and calib_conf (supplies the conformal
quantile) are disjoint mandate sets, per stats-reviewer's B4 finding that a
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
