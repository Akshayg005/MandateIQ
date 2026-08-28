"""Mandate-level grouped splits. eval/frozen/protocol.md:40-47 pre-commits
the mechanism -- a mandate's cycles never appear in both a split used to
choose/tune a policy and the split its number is reported on, grouped by
mandate_id, never row-level -- before this file existed; this module is
that mechanism's implementation, and the proportions below are what this
block adds.

This module splits ONLY the exploring corpus eval/corpus.py generates
(src/model/features.featurize()'s output over it). The frozen evaluation
batch -- the 200 mandates at eval/frozen/sim_config.yaml's own seed, all
three arms -- is never passed to split(): it is reserved untouched as the
reported evaluation batch from B8/B13 onward, read directly by
eval/baseline_ladder.py today and by the allocator later, exactly as
protocol.md's own "B2's baseline-ladder run is not subject to a split at
all" paragraph already establishes for the ladder. See DECISIONS.md,
2026-08-27, B4.

FOUR-WAY split, not three -- changed from PLAN_DETAIL.md's literal
`split(df, seed) -> (train, calib, test)` interface per stats-reviewer's B4
finding (DECISIONS.md, 2026-08-28): `calib` was being asked to do two jobs
-- fit isotonic calibration AND supply the conformal quantile -- and split
conformal's validity requires the quantile's scores to be exchangeable with
the test-time score. Once isotonic has been fit on a row, that row's own
score is no longer an honest out-of-sample residual for it, so a single
shared `calib` silently narrows every conformal prediction set below its
stated 95% coverage. Narrower sets are MORE likely to collapse to the
singleton `{WONT_PAY}` that fires the off-ramp (root CLAUDE.md, "Safety
design") -- so the bug's failure mode is exactly the harm that gate exists
to prevent, while a reliability diagram fit and read on the same rows would
still look diagonal. `calib_iso` and `calib_conf` are disjoint mandate sets
so each conformal quantile is computed against genuinely unseen residuals.

So the split is entirely within training data: `train` fits the
competing-risks hazards (B5); `calib_iso` fits isotonic calibration (B6);
`calib_conf` supplies the conformal quantile (B6); `test` is an internal
model-selection holdout, not a number this project ever reports.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 70/10/10/10 of the exploring corpus. Halving the old 20% `calib` into two
# 10% pieces still clears conformal prediction's n=19 bare validity floor
# with margin (roughly 175-180 mandates each on eval/corpus.py's default
# 2,000-mandate corpus, ~1,769 after the AFA-cliff filter) while keeping
# each quantile estimate's own source data genuinely unseen by the other
# calibration step. Recorded in DECISIONS.md, 2026-08-28, B4.
TRAIN_FRAC = 0.70
CALIB_ISO_FRAC = 0.10
CALIB_CONF_FRAC = 0.10
TEST_FRAC = 0.10


class SplitIntegrityError(RuntimeError):
    """Raised by split() if any disjointness or row-conservation guarantee
    fails. A real exception, not a bare `assert` -- this module's entire
    product is a guarantee, and `assert` is stripped under `python -O`."""


def split(
    df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` (person-period rows, output of person_period.build() or
    features.featurize()) into (train, calib_iso, calib_conf, test) grouped
    on mandate_id via sklearn's GroupShuffleSplit, at TRAIN_FRAC/
    CALIB_ISO_FRAC/CALIB_CONF_FRAC/TEST_FRAC.

    Guarantees, all checked before returning (raising SplitIntegrityError,
    not asserting): the four mandate_id sets are pairwise disjoint; every
    row of `df` appears in exactly one output frame (no row dropped, no row
    duplicated); every (mandate_id, cycle_id) episode's rows all land in
    the same one of the four frames (an episode never straddles a split,
    since GroupShuffleSplit groups on mandate_id and cycle_id is never
    independently split within a mandate here); the same `seed` reproduces
    the identical four-way partition.

    Three GroupShuffleSplit passes, all grouped on mandate_id -- sklearn
    splits by PROPORTION OF GROUPS, which is what makes the FRAC constants
    a statement about mandate counts (what tests/model/test_splits.py
    checks) rather than row counts: peel off TEST_FRAC of the mandates,
    then CALIB_CONF_FRAC's share of what's left, then split what remains
    into train vs calib_iso.
    """
    groups = df["mandate_id"].to_numpy()

    test_split = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    rest_pos, test_pos = next(test_split.split(df, groups=groups))
    rest_df = df.iloc[rest_pos]

    calib_conf_relative_frac = CALIB_CONF_FRAC / (
        TRAIN_FRAC + CALIB_ISO_FRAC + CALIB_CONF_FRAC
    )
    calib_conf_split = GroupShuffleSplit(
        n_splits=1, test_size=calib_conf_relative_frac, random_state=seed
    )
    rest2_pos, calib_conf_pos = next(
        calib_conf_split.split(rest_df, groups=rest_df["mandate_id"].to_numpy())
    )
    rest2_df = rest_df.iloc[rest2_pos]

    calib_iso_relative_frac = CALIB_ISO_FRAC / (TRAIN_FRAC + CALIB_ISO_FRAC)
    calib_iso_split = GroupShuffleSplit(
        n_splits=1, test_size=calib_iso_relative_frac, random_state=seed
    )
    train_rel, calib_iso_rel = next(
        calib_iso_split.split(rest2_df, groups=rest2_df["mandate_id"].to_numpy())
    )

    train_df = rest2_df.iloc[train_rel].reset_index(drop=True)
    calib_iso_df = rest2_df.iloc[calib_iso_rel].reset_index(drop=True)
    calib_conf_df = rest_df.iloc[calib_conf_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos].reset_index(drop=True)

    frames = {
        "train": train_df, "calib_iso": calib_iso_df,
        "calib_conf": calib_conf_df, "test": test_df,
    }
    mandate_sets = {name: set(f["mandate_id"]) for name, f in frames.items()}
    names = list(mandate_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = mandate_sets[a] & mandate_sets[b]
            if overlap:
                raise SplitIntegrityError(
                    f"{a}/{b} mandate_id overlap: {sorted(overlap)[:10]}"
                )

    total_out = sum(len(f) for f in frames.values())
    if total_out != len(df):
        raise SplitIntegrityError(
            f"split() dropped or duplicated rows: input {len(df)}, output {total_out}"
        )

    return train_df, calib_iso_df, calib_conf_df, test_df
