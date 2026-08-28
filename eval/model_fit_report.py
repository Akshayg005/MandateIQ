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
import pandas as pd
from sklearn.model_selection import GroupKFold

from eval.corpus import TRAIN_SEEDS, generate
from src.core.types import Outcome
from src.model import calibration, conformal, paths
from src.model.cif import terminal_distribution
from src.model.competing_risks import assemble, brier_per_cause, calibration_table, fit, hazards, log_loss
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


def _schedule_frame(episodes) -> pd.DataFrame:
    """(mandate_id, cycle_id, slot, on_day) for every episode's FULL
    committed schedule (all three of slots 2/3/4, regardless of how many
    were actually attempted) -- what src/model/paths.hazard_tensor()'s
    `schedule=` parameter needs to score an un-attempted slot from the
    real, pre-registered day rather than imputing one. Removes the
    outcome-dependent-imputation leak stats-reviewer caught at B6
    (DECISIONS.md, 2026-08-28): without this, whether a slot's covariates
    are real or imputed depended on whether the episode SURVIVED to that
    slot, which is a deterministic function of the very outcome being
    predicted."""
    rows = [
        {"mandate_id": ep.mandate.mandate_id, "cycle_id": ep.mandate.cycle_id,
         "slot": slot, "on_day": on_day}
        for ep in episodes if ep.schedule is not None
        for slot, on_day in zip((2, 3, 4), ep.schedule)
    ]
    return pd.DataFrame(rows, columns=["mandate_id", "cycle_id", "slot", "on_day"])


def _build_corpus():
    """Returns (pp_df, assembled, schedule_df). pp_df is needed alongside
    assembled from B6 onward: paths.terminal_labels() reads is_terminal/
    censor_reason, which assemble()'s merge (row_id + event_code +
    estimable only) never carries -- see src/model/paths.py. schedule_df
    is B6's leak fix -- see _schedule_frame()'s docstring."""
    episodes = generate(TRAIN_SEEDS)  # nominal arm only -- B4's decision
    pp_df = build(episodes)
    feat_df = featurize(pp_df)
    return pp_df, assemble(pp_df, feat_df), _schedule_frame(episodes)


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


def _terminal_distribution_and_labels(tensor, pp_split):
    """Bridge one hazard_tensor() output to a per-mandate terminal-Outcome
    distribution (cif.terminal_distribution, Outcome int order), joined
    with terminal_labels()'s eligibility/label on (mandate_id, cycle_id),
    restricted to ELIGIBLE mandates only -- an episode censored
    WINDOW_CLOSED before slot 4 has no honest by-slot-4 label
    (src/model/paths.py's module docstring). Returns (h_terminal, y,
    group_ids) ready for calibration.fit()/conformal.calibrate(). Takes an
    already-built tensor rather than (model, assembled_split) so a caller
    that also needs the tensor for something else (the imputation-extent
    count below) does not pay for hazard_tensor() twice.
    """
    term_dist = terminal_distribution(tensor.h)

    keys_df = tensor.keys.to_frame(index=False)
    keys_df["_row"] = np.arange(len(keys_df))

    labels_df = paths.terminal_labels(pp_split)
    merged = keys_df.merge(labels_df, on=["mandate_id", "cycle_id"], how="inner")
    eligible = merged[merged["eligible"]]

    idx = eligible["_row"].to_numpy()
    y = eligible["label"].to_numpy()
    group_ids = eligible["mandate_id"].astype(str).to_numpy()
    return term_dist[idx], y, group_ids


def _calibration_conformal_sweep(pp_df, assembled, schedule_df, seeds):
    """B6's gate evidence, swept over `seeds` split draws rather than one:
    classwise ECE (raw vs isotonic-calibrated) on hazards, and split-
    conformal coverage on the terminal-Outcome distribution -- both fit on
    their own disjoint calib split (calib_iso / calib_conf) and reported
    on `test`, per src/model/splits.py's discipline. Coverage is reported
    per-class AND marginal, alongside mean set size and singleton rate --
    per B6's design notes, coverage alone is trivially satisfiable by a
    predictor that always returns every label, which is close to what
    this model's small hazard-vector vocabulary actually produces.

    `schedule_df` is passed straight through to every hazard_tensor() call
    -- see _schedule_frame()'s docstring for why this is required, not
    optional, from B6 onward.
    """
    raw_eces, cal_eces = [], []
    marginal_covs = []
    per_class_covs: dict[int, list[float]] = {c: [] for c in range(4)}
    mean_set_sizes, singleton_rates = [], []
    imputed_fractions = []

    for seed in seeds:
        train, calib_iso, calib_conf, test = split(assembled, seed=seed)
        # Note: no pp_calib_iso -- isotonic calibration works at the
        # person-period ROW level (hazards(model, calib_iso_est) below),
        # never needs terminal_labels()'s per-mandate frame the way
        # calib_conf/test do for the conformal bridge.
        pp_calib_conf = pp_df[pp_df["row_id"].isin(calib_conf["row_id"])]
        pp_test = pp_df[pp_df["row_id"].isin(test["row_id"])]

        model = fit(train)

        # -- isotonic calibration, hazard level --
        calib_iso_est = calib_iso[calib_iso["estimable"]].copy()
        test_est = test[test["estimable"]].copy()
        h_calib_iso = hazards(model, calib_iso_est)
        y_calib_iso = calib_iso_est["event_code"].astype(int).to_numpy()
        iso_cal = calibration.fit(
            h_calib_iso, y_calib_iso,
            row_ids=calib_iso_est["row_id"].tolist(), provenance="calib_iso",
        )
        calibration.assert_disjoint(iso_cal.fit_row_ids, frozenset(test_est["row_id"]))

        h_test_raw = hazards(model, test_est)
        y_test_hazard = test_est["event_code"].astype(int).to_numpy()
        h_test_cal = calibration.apply(iso_cal, h_test_raw)
        raw_eces.append(calibration.classwise_ece(h_test_raw, y_test_hazard))
        cal_eces.append(calibration.classwise_ece(h_test_cal, y_test_hazard))

        # -- split conformal, terminal-Outcome level --
        calib_conf_tensor = paths.hazard_tensor(model, calib_conf, schedule=schedule_df)
        test_tensor = paths.hazard_tensor(model, test, schedule=schedule_df)
        h_calib_conf, y_calib_conf, ids_calib_conf = _terminal_distribution_and_labels(
            calib_conf_tensor, pp_calib_conf
        )
        h_test_term, y_test_term, ids_test_term = _terminal_distribution_and_labels(
            test_tensor, pp_test
        )
        # Imputation extent, this seed's test split -- no `schedule` frame
        # is threaded through yet (paths.hazard_tensor()'s preferred path;
        # see its module docstring), so every un-attempted slot falls back
        # to the documented imputation rule. Counted, not just disclosed
        # in prose, per that module's own "report the count either way".
        n_cells = test_tensor.observed.size
        n_imputed = sum(1 for row in test_tensor.observed for cell in row if not cell)
        imputed_fractions.append(n_imputed / n_cells if n_cells else float("nan"))
        predictor = conformal.calibrate(
            scores=conformal.lac_scores(h_calib_conf),
            y=y_calib_conf,
            labels=tuple(Outcome),
            row_group_ids=ids_calib_conf.tolist(),
            provenance="calib_conf",
        )
        conformal.assert_disjoint(predictor.fit_group_ids, frozenset(ids_test_term.tolist()))

        report = predictor.coverage_report(
            score_rows=conformal.lac_scores(h_test_term),
            y=y_test_term,
            keys=ids_test_term.tolist(),
        )
        marginal_row = report[report["label"] == "MARGINAL"].iloc[0]
        marginal_covs.append(float(marginal_row["coverage"]))
        mean_set_sizes.append(float(marginal_row["mean_set_size"]))
        n_test_term = int(marginal_row["n"])
        singleton_rates.append(
            float(marginal_row["singleton_count"]) / n_test_term if n_test_term else float("nan")
        )
        for c in range(4):
            row_c = report[report["label"] == Outcome(c)]
            per_class_covs[c].append(
                float(row_c.iloc[0]["coverage"]) if len(row_c) else float("nan")
            )

    return {
        "raw_ece": np.array(raw_eces), "cal_ece": np.array(cal_eces),
        "marginal_coverage": np.array(marginal_covs),
        "mean_set_size": np.array(mean_set_sizes),
        "singleton_rate": np.array(singleton_rates),
        "per_class_coverage": {c: np.array(v) for c, v in per_class_covs.items()},
        "imputed_fraction": np.array(imputed_fractions),
    }


def main() -> None:
    pp_df, assembled, schedule_df = _build_corpus()
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

    print()
    print(f"=== B6: calibration + conformal, swept across {len(SPLIT_SEEDS)} split seeds "
          f"(min/mean/max -- see gate note below) ===")
    print("(isotonic is fit on calib_iso, the conformal quantile on calib_conf, both")
    print(" reported on test -- all three disjoint per split seed, src/model/splits.py.")
    print(" Coverage is never meaningful without set size and singleton rate alongside")
    print(" it: a predictor returning every label clears any coverage target trivially.)")
    sweep = _calibration_conformal_sweep(pp_df, assembled, schedule_df, SPLIT_SEEDS)

    def _mmm(arr: np.ndarray) -> str:
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0:
            return "no data"
        return f"min={finite.min():.4f}  mean={finite.mean():.4f}  max={finite.max():.4f}"

    print()
    print(f"classwise ECE  raw:        {_mmm(sweep['raw_ece'])}")
    print(f"classwise ECE  calibrated: {_mmm(sweep['cal_ece'])}")
    print("(calibrated is NOT expected to beat raw here -- see src/model/calibration.py's")
    print(" module docstring; the regression guard is calibrated <= raw + 0.01, not 'improves')")
    print()
    print(f"conformal marginal coverage: {_mmm(sweep['marginal_coverage'])}  (nominal 0.95)")
    print(f"conformal mean set size:     {_mmm(sweep['mean_set_size'])}  (of 4 labels)")
    print(f"conformal singleton rate:    {_mmm(sweep['singleton_rate'])}")
    for c in range(4):
        print(f"  per-class coverage [{Outcome(c).name:13s}]: {_mmm(sweep['per_class_coverage'][c])}")
    print()
    print(f"un-attempted-slot rate (test split): {_mmm(sweep['imputed_fraction'])}")
    print("(fraction of tensor cells with no REAL attempt row -- a mandate that resolves")
    print(" early naturally has fewer real attempts, so this is expected to be substantial,")
    print(" not a defect. It is NOT a bias-risk number: every one of these cells now gets")
    print(" its in_salary_window from the real committed schedule (schedule_df, threaded")
    print(" through hazard_tensor()), not a guess -- see eval/corpus.py's Episode.schedule")
    print(" field and stats-reviewer's B6 finding 1 (DECISIONS.md, 2026-08-28) for why that")
    print(" fix mattered: the un-fixed fallback picked its guess based on which slot the")
    print(" episode SURVIVED to, which is a function of the very outcome being predicted.)")
    print()
    print("gate criterion is the MIN across the sweep, not the mean -- a single split")
    print("seed's marginal coverage can look nominal while a per-class minimum across")
    print("seeds reveals real instability (this is what motivated smoothed Mondrian")
    print("conformal as the default; see src/model/conformal.py's module docstring).")


if __name__ == "__main__":
    main()
