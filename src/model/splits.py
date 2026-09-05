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

FOUR-WAY split, not three -- changed from the build spec's literal
`split(df, seed) -> (train, calib, test)` interface per the statistics review's B4
finding (DECISIONS.md, 2026-08-28): `calib` was being asked to do two jobs
-- fit isotonic calibration AND supply the conformal quantile -- and split
conformal's validity requires the quantile's scores to be exchangeable with
the test-time score. Once isotonic has been fit on a row, that row's own
score is no longer an honest out-of-sample residual for it, so a single
shared `calib` silently narrows every conformal prediction set below its
stated 95% coverage. Narrower sets are MORE likely to collapse to the
singleton `{WONT_PAY}` that fires the off-ramp (root DESIGN.md, "Safety
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
    df: pd.DataFrame, seed: int, *, group_key: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` (person-period rows, output of person_period.build() or
    features.featurize()) into (train, calib_iso, calib_conf, test) grouped
    on `group_key` (default: mandate_id) via sklearn's GroupShuffleSplit, at
    TRAIN_FRAC/CALIB_ISO_FRAC/CALIB_CONF_FRAC/TEST_FRAC.

    `group_key`, added at B6: an optional pd.Series, aligned to `df`'s row
    order/index, giving the grouping unit for each row. Default None groups
    on mandate_id exactly as before B6 -- every call site written before
    this parameter existed keeps its exact prior behaviour, byte for byte.
    The intended caller-side construction, for the `coupled` arm's shared-
    balance households, is `household_id` where non-null, falling back to
    `mandate_id` where null (e.g. `df["household_id"].fillna(df["mandate_id"])`)
    -- this function does not build that fallback itself, since it has no
    opinion on where the grouping key comes from, only that rows sharing a
    key never straddle a split. On `nominal`/`misspecified`, household_id is
    always null (eval/frozen/simulator.py's SimMandate docstring), so that
    construction is elementwise identical to mandate_id and every B5 number
    -- computed before this parameter existed -- is reproduced bit-for-bit;
    see tests/model/test_splits.py's bit-identity tests. On `coupled`,
    grouping on household_id is what keeps a household's mandates -- whose
    outcomes are dependent through shared-balance contention -- from
    straddling train/calib_conf and silently narrowing conformal's
    prediction sets below stated coverage (DECISIONS.md, 2026-08-28, B5
    the statistics review finding 7).

    Guarantees, all checked before returning (raising SplitIntegrityError,
    not asserting): the four group-key sets are pairwise disjoint; every
    row of `df` appears in exactly one output frame (no row dropped, no row
    duplicated); every (mandate_id, cycle_id) episode's rows all land in
    the same one of the four frames. Under the default (group_key=None,
    grouping on mandate_id) this holds because GroupShuffleSplit groups on
    mandate_id and cycle_id is never independently split within a mandate.
    Under an explicit group_key, "no mandate straddles a split" is a
    COROLLARY, not a separately-enforced guarantee: it holds whenever every
    mandate maps to exactly one group-key value, which person_period.
    validate() now enforces upstream for household_id (constant within a
    mandate across all its cycles) -- this function trusts that precondition
    rather than re-checking it, the same discipline competing_risks.py's
    `estimable` filter uses for its own upstream invariant. The same `seed`
    reproduces the identical four-way partition.

    Three GroupShuffleSplit passes, all grouped on the group key -- sklearn
    splits by PROPORTION OF GROUPS, which is what makes the FRAC constants
    a statement about group counts (mandate counts, under the default)
    rather than row counts: peel off TEST_FRAC of the groups, then
    CALIB_CONF_FRAC's share of what's left, then split what remains into
    train vs calib_iso.
    """
    if group_key is None:
        key = df["mandate_id"]
    else:
        # Positional alignment, not just length: every slice below is
        # `.iloc[pos]` on both `df` and `key` in lockstep, which is only
        # correct if row i of `key` really is row i's group -- silently
        # true for the documented construction pattern
        # (`df["household_id"].fillna(df["mandate_id"])`, which inherits
        # df's own index by construction) but not guaranteed for an
        # arbitrary caller-supplied Series. Checked here (the statistics review,
        # B6, DECISIONS.md 2026-08-28 finding 5) rather than trusted.
        if len(group_key) != len(df):
            raise SplitIntegrityError(
                f"group_key length {len(group_key)} does not match df length "
                f"{len(df)}"
            )
        if not group_key.index.equals(df.index):
            raise SplitIntegrityError(
                "group_key's index does not match df's index -- positional "
                "alignment between the two is required (e.g. "
                "df['household_id'].fillna(df['mandate_id']), which "
                "inherits df's index by construction)"
            )
        key = group_key
    groups = key.to_numpy()

    test_split = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    rest_pos, test_pos = next(test_split.split(df, groups=groups))
    rest_df = df.iloc[rest_pos]
    rest_key = key.iloc[rest_pos]

    calib_conf_relative_frac = CALIB_CONF_FRAC / (
        TRAIN_FRAC + CALIB_ISO_FRAC + CALIB_CONF_FRAC
    )
    calib_conf_split = GroupShuffleSplit(
        n_splits=1, test_size=calib_conf_relative_frac, random_state=seed
    )
    rest2_pos, calib_conf_pos = next(
        calib_conf_split.split(rest_df, groups=rest_key.to_numpy())
    )
    rest2_df = rest_df.iloc[rest2_pos]
    rest2_key = rest_key.iloc[rest2_pos]

    calib_iso_relative_frac = CALIB_ISO_FRAC / (TRAIN_FRAC + CALIB_ISO_FRAC)
    calib_iso_split = GroupShuffleSplit(
        n_splits=1, test_size=calib_iso_relative_frac, random_state=seed
    )
    train_rel, calib_iso_rel = next(
        calib_iso_split.split(rest2_df, groups=rest2_key.to_numpy())
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
