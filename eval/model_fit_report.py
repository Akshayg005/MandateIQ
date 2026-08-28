"""Closes B5's gate (DECISIONS.md, 2026-08-28, B5 rebind entry): held-out
multinomial log-loss and per-cause Brier on the `test` split beat an
intercept-only MNLogit null (the ladder's implicit model -- constant
hazard, no covariates, no adaptation); transfer degradation of that same
fit reported on `misspecified` and `coupled` frames; calibration-in-the-
large reported per `slot x in_salary_window` cell.

Deliberately policy-free -- see DECISIONS.md, 2026-08-28, B5 §3: three of
the four original "beats the ladder" clauses turned out to be monotone in
attempt count and clearable by a policy that consults nothing, so B5 ships
no policy at all. This script only ever calls src/model/competing_risks.py
and src/model/cif.py's fitted-model interfaces against frames built by
eval/corpus.py + src/model/{person_period,features,splits}.py -- it never
drives the Simulator's attempt() directly.

The "beats the null" verdict comes from mandate-grouped K-FOLD cross-
validation (`_grouped_cv_diffs`) over disjoint, non-overlapping folds that
together cover the whole corpus -- NOT from repeatedly re-splitting the same
fixed corpus with different random seeds and treating the per-seed results
as independent samples. That was tried first and is a real statistical trap
(DECISIONS.md, B5, stats-reviewer CONFIRMING pass, finding 1): with ~90%
overlapping test sets across seeds, `SD/sqrt(n_seeds)` is not a standard
error at all -- it is a function of how many seeds you choose to loop over,
demonstrated by the reported t-statistic scaling from -9.56 to -18.67 across
10/20/40/60 seeds with ZERO new data added. K-fold CV's folds are mandate-
disjoint and jointly exhaustive, so the K per-fold means genuinely are
independent draws and SD/sqrt(K) is valid. `SPLIT_SEEDS` below is kept only
as a split-STABILITY check (mean, SD, win-count) -- explicitly not a source
of dispersion for any t-stat or verdict.

Transfer-degradation numbers score BOTH the full model and the intercept-
only null on each transfer frame (DECISIONS.md, B5 stats-reviewer entry,
finding 3 -- scoring only the full model made a base-rate shift under
`coupled` look like model skill; reporting relative to the null on the
SAME frame removes that artifact).

Run: python -m eval.model_fit_report
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from eval.corpus import TRAIN_SEEDS, generate
from src.model.competing_risks import assemble, brier_per_cause, calibration_table, fit, log_loss
from src.model.features import featurize
from src.model.person_period import build
from src.model.splits import split

# Disjoint from TRAIN_SEEDS (90001-90040) and the frozen sim_config seed --
# these arms are never trained on, only scored, so any disjoint seeds work;
# a small set keeps runtime reasonable for a diagnostic report.
TRANSFER_SEEDS = (91001, 91002, 91003, 91004, 91005)

# Split-STABILITY check only (mean/SD/win-count) -- NOT a source of
# dispersion for any t-stat. See module docstring: repeated overlapping
# re-splits of one fixed corpus are not independent samples.
SPLIT_SEEDS = tuple(range(20))
PRIMARY_SEED = SPLIT_SEEDS[0]  # the one split used for calibration/transfer detail

N_CV_FOLDS = 5


def _build_corpus():
    episodes = generate(TRAIN_SEEDS)  # nominal arm only -- B4's decision
    pp_df = build(episodes)
    feat_df = featurize(pp_df)
    return assemble(pp_df, feat_df)


def _grouped_cv_diffs(assembled) -> np.ndarray:
    """K-fold CV grouped by mandate_id: K DISJOINT, jointly-exhaustive folds
    over every estimable row in `assembled`. Refits full+null on each
    fold's (K-1)-fold training portion, scores log_loss(full) -
    log_loss(null) on that fold's held-out portion. Returns the K per-fold
    differences -- these ARE independent samples (no two folds share a
    test row, and every row appears in exactly one), unlike repeatedly
    re-splitting the same corpus with different seeds (see module
    docstring). SD(fold_means, ddof=1) / sqrt(K) is therefore a valid
    standard error for the mean difference."""
    estimable = assembled[assembled["estimable"]].copy().reset_index(drop=True)
    groups = estimable["mandate_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_means = []
    for train_idx, test_idx in gkf.split(estimable, groups=groups):
        train_fold = estimable.iloc[train_idx]
        test_fold = estimable.iloc[test_idx]
        model = fit(train_fold)
        null_model = fit(train_fold, intercept_only=True)
        fold_means.append(log_loss(model, test_fold) - log_loss(null_model, test_fold))
    return np.array(fold_means)


def _transfer_frame(arm: str):
    episodes = generate(TRANSFER_SEEDS, arm=arm, check_coverage=False)
    pp_df = build(episodes)
    feat_df = featurize(pp_df)
    assembled = assemble(pp_df, feat_df)
    return assembled[assembled["estimable"]].copy()


def main() -> None:
    assembled = _build_corpus()
    n_mandates = assembled["mandate_id"].nunique()
    n_estimable = int(assembled["estimable"].sum())
    print(f"corpus: {len(TRAIN_SEEDS)} seeds -> {n_mandates} mandates, "
          f"{len(assembled)} person-period rows ({n_estimable} estimable)")
    print()

    print(f"=== held-out log_loss, full vs intercept-only null: "
          f"{N_CV_FOLDS}-fold mandate-grouped CV (the valid inferential test) ===")
    cv_diffs = _grouped_cv_diffs(assembled)
    cv_mean = float(cv_diffs.mean())
    cv_sd = float(cv_diffs.std(ddof=1))
    cv_se = cv_sd / np.sqrt(len(cv_diffs))
    cv_t = cv_mean / cv_se if cv_se > 0 else float("nan")
    cv_wins = int((cv_diffs < 0).sum())
    print(f"per-fold (full - null): {np.round(cv_diffs, 5).tolist()}")
    print(f"mean = {cv_mean:+.5f}   SD = {cv_sd:.5f}   SE = {cv_se:.5f}   "
          f"t = {cv_t:+.2f}   folds = {len(cv_diffs)}   "
          f"folds negative (full beats null) = {cv_wins}/{len(cv_diffs)}")
    verdict = "BEATS" if (cv_mean < 0 and abs(cv_t) > 2) else \
              "DOES NOT BEAT" if (cv_mean > 0 and abs(cv_t) > 2) else "INCONCLUSIVE ON"
    print(f"verdict: full model {verdict} the null at ~95% (|t|>2), lower log_loss is better")

    print()
    print(f"=== split-stability check ONLY (repeated re-splits of the SAME corpus -- "
          f"NOT independent samples, no SE/t derived from this; see module docstring) ===")
    stability_diffs = []
    for seed in SPLIT_SEEDS:
        train, _calib_iso, _calib_conf, test = split(assembled, seed=seed)
        test_est = test[test["estimable"]].copy()
        model = fit(train)
        null_model = fit(train, intercept_only=True)
        stability_diffs.append(log_loss(model, test_est) - log_loss(null_model, test_est))
    stability_diffs = np.array(stability_diffs)
    stability_wins = int((stability_diffs < 0).sum())
    print(f"mean(full - null) across {len(SPLIT_SEEDS)} split seeds = "
          f"{float(stability_diffs.mean()):+.5f}   SD = {float(stability_diffs.std(ddof=1)):.5f}   "
          f"wins = {stability_wins}/{len(SPLIT_SEEDS)}")

    print()
    print(f"=== representative single fit, split seed={PRIMARY_SEED} "
          f"(detail only -- the verdict above is the sweep, not this) ===")
    train, _calib_iso, _calib_conf, test = split(assembled, seed=PRIMARY_SEED)
    test_est = test[test["estimable"]].copy()
    model = fit(train)
    null_model = fit(train, intercept_only=True)
    ll_full, ll_null = log_loss(model, test_est), log_loss(null_model, test_est)
    print(f"log_loss   full={ll_full:.4f}  null={ll_null:.4f}")
    brier_full, brier_null = brier_per_cause(model, test_est), brier_per_cause(null_model, test_est)
    for c in range(4):
        print(f"brier[{c}]  full={brier_full[c]:.4f}  null={brier_null[c]:.4f}  "
              f"{'beats' if brier_full[c] < brier_null[c] else ('ties' if brier_full[c] == brier_null[c] else 'does not beat')}")

    print()
    print("=== calibration-in-the-large, held-out test split (primary seed) ===")
    cal = calibration_table(model, test_est)
    print(cal.to_string(index=False))

    print()
    print("=== transfer degradation: full model AND null, scored on misspecified/coupled ===")
    print("(degradation reported relative to the null on the SAME transfer frame --")
    print(" scoring only the full model conflates model skill with a base-rate shift")
    print(" in the transfer arm's outcome mix; DECISIONS.md B5 stats-reviewer finding 3.")
    print(" NOTE: single point estimate at PRIMARY_SEED, no dispersion -- unlike the")
    print(" nominal-arm claim above, this has not been put through K-fold CV. Flagged,")
    print(" not fixed this session; DECISIONS.md B5 confirming-pass finding 5.)")
    for arm in ("misspecified", "coupled"):
        transfer_est = _transfer_frame(arm)
        ll_transfer_full = log_loss(model, transfer_est)
        ll_transfer_null = log_loss(null_model, transfer_est)
        print(f"{arm:13s} n={len(transfer_est):5d}  "
              f"full={ll_transfer_full:.4f}  null={ll_transfer_null:.4f}  "
              f"full-vs-null on this arm={ll_transfer_full - ll_transfer_null:+.4f}  "
              f"(nominal test full-vs-null was {ll_full - ll_null:+.4f})")


if __name__ == "__main__":
    main()
