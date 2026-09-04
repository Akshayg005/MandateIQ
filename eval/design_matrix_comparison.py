"""R1 Phase A gate evidence (reports/gates.md, "Post-B16 remediation gates",
R1a): does widening the hazard model's design matrix with amount and
category terms (`src.model.competing_risks.WIDENED_FEATURE_COLUMNS`) beat
the narrow, slot-only design (`FEATURE_COLUMNS`) that has been the DEFAULT
since B5? The gate requires the result stated honestly whichever way it
goes -- this script exists to measure it, not to justify a decision already
made.

Three comparisons, following eval/model_fit_report.py's own established
statistical discipline (see that module's docstring) rather than inventing
a new one -- PLUS a correction to that discipline, found by stats-reviewer
on this file's first version, which also applies to model_fit_report.py's
own `|t| > 2` check (not fixed there in this pass; disclosed in
DECISIONS.md instead, since that file's own measured t=-6.32 clears both
the correct and the sloppy threshold and nothing published there was
actually wrong):

1. PRIMARY: pooled out-of-fold per-ROW log-loss differences
   (`_pooled_out_of_fold_diffs`), clustered by mandate_id
   (`_clustered_se_of_mean`). GroupKFold scores every one of the ~12k
   person-period rows exactly once, out-of-fold -- pooling them (rather
   than the 5 fold MEANS) uses all the data the CV actually touched, and
   clustering by mandate_id (rather than treating each row as independent)
   is required because one mandate contributes 2-3 correlated rows.
2. SECONDARY/diagnostic: mandate-GROUPED K-fold CV fold-MEAN differences
   (`_grouped_cv_diffs_widened`, mirroring eval/model_fit_report.py's
   `_grouped_cv_diffs`) -- only 5 independent numbers (one per fold), so
   its own t-test needs df=K-1=4's actual critical value
   (`scipy.stats.t.ppf`), NOT the normal-approximation `abs(t) > 2` both
   this file's first version and model_fit_report.py use: at df=4 the 95%
   two-sided critical value is 2.776, not 2.0. This file's own FIRST
   version reported "WIDENED DOES NOT BEAT narrow at ~95% confidence" from
   t=+2.37 on 4 df (p=0.076, NOT significant) -- a real, published,
   corrected-here statistical error. The properly-powered PRIMARY test
   above (t=+2.88, p=0.004 on 7153 clusters) is what actually earns the
   95% claim; the fold-mean view is kept only as a coarser cross-check,
   never as the number a verdict is drawn from.
3. The SPLIT_SEEDS repeated-re-split stability check -- mean/SD/win-count
   over 20 overlapping re-splits of the SAME corpus. Reported because
   src/model/competing_risks.py's own docstring already cites this exact
   style of number for the two previously-excluded columns
   (`days_since_last_attempt`, `slot3_x_in_salary_window`); it is NOT an
   independent-sample statistic and no SE/t is derived from it here either.

Plus the coefficients a defensibility reviewer actually wants to see: every
one of the six new columns' fitted coefficient, standard error, and 95% CI,
across all three non-reference outcome equations (RECOVERED/DEAD/OPTED_OUT
vs STILL_PENDING), from ONE fit on the full corpus -- not averaged across
folds or seeds, matching ordinary practice for reporting a model's own
coefficients (the CV/stability numbers above are for the OUT-OF-SAMPLE
comparison, a different question).

Why a null result here is not evidence of a bad idea: eval/frozen/
simulator.py's `_draw_outcome` never reads `category` in any arm, and reads
`amount_paise` only inside the `coupled` arm's household-balance comparison
(`_apply_household_coupling`) -- never in the base hazard logits `fit()`
actually trains against (`nominal`, per eval/corpus.generate()'s own
default). Every coefficient measured here is therefore EXPECTED to be
statistically indistinguishable from zero on THIS corpus; that expectation
is falsifiable by this script's own output, not assumed by it. See
src/model/competing_risks.py's module docstring for the full argument and
eval/sim2.py (R1 Phase B) for a corpus where these covariates actually
carry signal.

Run: python -m eval.design_matrix_comparison
Writes: reports/model_defensibility.md (Phase A section; Phase B appends).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import GroupKFold

from eval.corpus import TRAIN_SEEDS, generate
from src.core.clock import now as clock_now
from src.model.competing_risks import (
    FEATURE_COLUMNS,
    WIDENED_FEATURE_COLUMNS,
    assemble,
    fit,
    hazards,
    log_loss,
)
from src.model.features import featurize
from src.model.person_period import build
from src.model.splits import split

N_CV_FOLDS = 5
# Same 20-seed range eval/model_fit_report.py's own stability check uses --
# not independently chosen, so the two reports' "stability sweep" language
# means the same thing.
SPLIT_SEEDS = tuple(range(20))

_NEW_COLUMNS: tuple[str, ...] = tuple(c for c in WIDENED_FEATURE_COLUMNS if c not in FEATURE_COLUMNS)
_OUTCOME_EQUATIONS: tuple[str, ...] = ("RECOVERED", "DEAD", "OPTED_OUT")

OUT_MD = pathlib.Path(__file__).resolve().parent.parent / "reports" / "model_defensibility.md"

_SECTION_BEGIN = "<!-- PHASE_A:BEGIN -->"
_SECTION_END = "<!-- PHASE_A:END -->"


def _build_corpus():
    episodes = generate(TRAIN_SEEDS)  # nominal arm only -- matches B5's fit()
    pp_df = build(episodes)
    feat_df = featurize(pp_df)
    return assemble(pp_df, feat_df)


def _per_row_log_loss(model, df: pd.DataFrame) -> np.ndarray:
    """Same picking logic as competing_risks.log_loss(), but returns the
    per-row -log(p_true) array instead of its mean -- log_loss() itself is
    left untouched (it is used, tested, and trusted elsewhere) since a
    mean-only function should not grow a second return shape for one
    caller's convenience."""
    probs = hazards(model, df)
    true_idx = df["event_code"].astype(int).to_numpy()
    picked = probs[np.arange(len(df)), true_idx]
    return -np.log(picked)


def _clustered_se_of_mean(values: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error of values.mean(), clustered by
    `clusters` (one cluster id per row, aligned to `values`) -- the CR1
    cluster-robust variance for a regression of `values` on an intercept
    only (Cameron & Miller 2015's standard formula, degenerate to this
    simple case). Needed because GroupKFold's out-of-fold per-row
    differences are NOT independent across rows sharing a mandate_id (one
    mandate contributes 2-3 person-period rows) even though GroupKFold
    guarantees independence ACROSS mandates (a mandate is never split
    across folds). Treating all rows as independent -- the mistake this
    function exists to avoid -- understates the true standard error.

    variance = [G/(G-1)] * sum_c(cluster residual sum)^2 / N^2, the
    standard CR1 small-sample-corrected formula; degenerates to the usual
    SD/sqrt(N) when every cluster has exactly one row."""
    n = len(values)
    mean = float(values.mean())
    residuals = values - mean
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    g = len(unique_clusters)
    cluster_sums = np.bincount(inverse, weights=residuals, minlength=g)
    correction = g / (g - 1) if g > 1 else 1.0
    variance = correction * float(np.sum(cluster_sums ** 2)) / (n ** 2)
    return float(np.sqrt(variance))


def _pooled_out_of_fold_diffs(assembled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """THE PRIMARY comparison (see module docstring, point 1): for each of
    N_CV_FOLDS mandate-grouped folds, refit narrow and widened on that
    fold's training portion, then compute the PER-ROW (widened - narrow)
    log-loss difference on that fold's held-out portion. Every row of the
    corpus is scored exactly once, out-of-fold, across the whole loop
    (GroupKFold's own guarantee) -- returns (all_row_diffs, all_row_mandate_ids),
    aligned, ready for _clustered_se_of_mean()."""
    estimable = assembled[assembled["estimable"]].copy().reset_index(drop=True)
    groups = estimable["mandate_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    all_diffs = []
    all_mandate_ids = []
    for train_idx, test_idx in gkf.split(estimable, groups=groups):
        train_fold = estimable.iloc[train_idx]
        test_fold = estimable.iloc[test_idx]
        narrow_model = fit(train_fold, feature_columns=FEATURE_COLUMNS)
        widened_model = fit(train_fold, feature_columns=WIDENED_FEATURE_COLUMNS)
        row_diffs = _per_row_log_loss(widened_model, test_fold) - _per_row_log_loss(narrow_model, test_fold)
        all_diffs.append(row_diffs)
        all_mandate_ids.append(test_fold["mandate_id"].to_numpy())
    return np.concatenate(all_diffs), np.concatenate(all_mandate_ids)


def _grouped_cv_diffs_widened(assembled: pd.DataFrame) -> np.ndarray:
    """SECONDARY/diagnostic only (see module docstring, point 2) -- the
    5 fold-MEAN (widened - narrow) log-loss differences. Only 5 independent
    numbers; its own significance check needs df=4's real critical value,
    not a normal-approximation `abs(t) > 2`. Mirrors
    eval/model_fit_report.py's _grouped_cv_diffs() exactly, parameterised
    for widened-vs-narrow instead of full-vs-null. Kept separate from
    _pooled_out_of_fold_diffs() above (rather than just averaging its
    output per fold) so this function's fold split and refits stay
    independently checkable against model_fit_report.py's own pattern."""
    estimable = assembled[assembled["estimable"]].copy().reset_index(drop=True)
    groups = estimable["mandate_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_diffs = []
    for train_idx, test_idx in gkf.split(estimable, groups=groups):
        train_fold = estimable.iloc[train_idx]
        test_fold = estimable.iloc[test_idx]
        narrow_model = fit(train_fold, feature_columns=FEATURE_COLUMNS)
        widened_model = fit(train_fold, feature_columns=WIDENED_FEATURE_COLUMNS)
        fold_diffs.append(log_loss(widened_model, test_fold) - log_loss(narrow_model, test_fold))
    return np.array(fold_diffs)


def _coefficient_table(assembled: pd.DataFrame) -> pd.DataFrame:
    """Fit WIDENED_FEATURE_COLUMNS once on the full corpus; return one row
    per (new column, outcome equation) with the fitted coefficient, its
    standard error, z, p, and 95% CI -- statsmodels' own inference, not
    re-derived here.

    A genuine MNLogitResults quirk, confirmed empirically (not assumed):
    `.params`/`.bse`/`.pvalues` index the fitted non-reference equations by
    INTEGER position (0, 1, 2 -- matching `_OUTCOME_EQUATIONS`'s own order,
    since STILL_PENDING=0 is the MNLogit reference category and RECOVERED/
    DEAD/OPTED_OUT=1/2/3 are fit in that order), but `.conf_int()`'s
    MultiIndex labels the SAME three equations by the 1-indexed STRING
    `"1"`/`"2"`/`"3"` -- i.e. `conf_int` group `str(i + 1)` is
    `params.columns[i]`. Verified directly: `conf_int`'s group `"1"`,
    `const` row reproduces `params[0]["const"] +/- 1.96 * bse[0]["const"]`
    to 6 decimal places.
    """
    estimable = assembled[assembled["estimable"]]
    model = fit(estimable, feature_columns=WIDENED_FEATURE_COLUMNS)
    result = model.result

    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    conf_int = result.conf_int()  # MultiIndex ("1"/"2"/"3", column) x [lower, upper]

    rows = []
    for eq_pos, eq_name in enumerate(_OUTCOME_EQUATIONS):
        ci_label = str(eq_pos + 1)
        for col in _NEW_COLUMNS:
            coef = float(params.loc[col, eq_pos])
            se = float(bse.loc[col, eq_pos])
            p = float(pvalues.loc[col, eq_pos])
            ci_lo = float(conf_int.loc[(ci_label, col), "lower"])
            ci_hi = float(conf_int.loc[(ci_label, col), "upper"])
            rows.append({
                "outcome": eq_name, "column": col, "coef": coef, "se": se,
                "z": coef / se if se > 0 else float("nan"), "p": p,
                "ci_low": ci_lo, "ci_high": ci_hi,
                "excludes_zero": not (ci_lo <= 0.0 <= ci_hi),
            })
    return pd.DataFrame(rows)


def main() -> None:
    assembled = _build_corpus()
    n_mandates = assembled["mandate_id"].nunique()
    estimable = assembled[assembled["estimable"]]
    print(f"corpus: {len(TRAIN_SEEDS)} seeds -> {n_mandates} mandates, "
          f"{len(estimable)} estimable person-period rows")
    print()

    print("=== PRIMARY: pooled out-of-fold log_loss, WIDENED - NARROW, "
          "clustered by mandate_id ===")
    pooled_diffs, pooled_mandate_ids = _pooled_out_of_fold_diffs(assembled)
    pooled_mean = float(pooled_diffs.mean())
    pooled_se = _clustered_se_of_mean(pooled_diffs, pooled_mandate_ids)
    pooled_n_clusters = len(np.unique(pooled_mandate_ids))
    pooled_df = pooled_n_clusters - 1
    pooled_t = pooled_mean / pooled_se if pooled_se > 0 else float("nan")
    pooled_p = float(2 * (1 - scipy_stats.t.cdf(abs(pooled_t), pooled_df))) if pooled_se > 0 else float("nan")
    pooled_crit = float(scipy_stats.t.ppf(0.975, pooled_df))
    print(f"rows = {len(pooled_diffs)}   mandates (clusters) = {pooled_n_clusters}")
    print(f"mean = {pooled_mean:+.5f}   clustered SE = {pooled_se:.5f}   "
          f"t = {pooled_t:+.2f}   df = {pooled_df}   p = {pooled_p:.4f}   "
          f"(95% critical t = {pooled_crit:.3f})")
    if pooled_mean < 0 and pooled_p < 0.05:
        pooled_verdict = "WIDENED BEATS narrow"
    elif pooled_mean > 0 and pooled_p < 0.05:
        pooled_verdict = "WIDENED DOES NOT BEAT (is worse than) narrow"
    else:
        pooled_verdict = "INCONCLUSIVE -- statistically indistinguishable"
    print(f"verdict: {pooled_verdict} at 95% confidence (p < 0.05), lower log_loss is better")

    print()
    print(f"=== SECONDARY/diagnostic: {N_CV_FOLDS}-fold mean log_loss, "
          f"WIDENED - NARROW (only {N_CV_FOLDS} independent numbers -- "
          f"cross-check, not the verdict) ===")
    cv_diffs = _grouped_cv_diffs_widened(assembled)
    cv_mean = float(cv_diffs.mean())
    cv_sd = float(cv_diffs.std(ddof=1))
    cv_se = cv_sd / np.sqrt(len(cv_diffs))
    cv_df = len(cv_diffs) - 1
    cv_t = cv_mean / cv_se if cv_se > 0 else float("nan")
    cv_crit = float(scipy_stats.t.ppf(0.975, cv_df))
    cv_wins = int((cv_diffs < 0).sum())
    print(f"per-fold (widened - narrow): {np.round(cv_diffs, 5).tolist()}")
    print(f"mean = {cv_mean:+.5f}   SD = {cv_sd:.5f}   SE = {cv_se:.5f}   "
          f"t = {cv_t:+.2f}   df = {cv_df}   95% critical t = {cv_crit:.3f}   "
          f"folds negative (widened beats narrow) = {cv_wins}/{len(cv_diffs)}")
    print(f"(this alone is UNDERPOWERED at df={cv_df} -- see the PRIMARY test above for the verdict)")

    print()
    print(f"=== split-STABILITY check only (repeated re-splits of the SAME "
          f"corpus -- NOT independent samples, no SE/t derived; see module "
          f"docstring) ===")
    stability_diffs = []
    for seed in SPLIT_SEEDS:
        train, _calib_iso, _calib_conf, test = split(assembled, seed=seed)
        test_est = test[test["estimable"]].copy()
        narrow_model = fit(train, feature_columns=FEATURE_COLUMNS)
        widened_model = fit(train, feature_columns=WIDENED_FEATURE_COLUMNS)
        stability_diffs.append(log_loss(widened_model, test_est) - log_loss(narrow_model, test_est))
    stability_diffs = np.array(stability_diffs)
    stability_wins = int((stability_diffs < 0).sum())
    print(f"mean(widened - narrow) across {len(SPLIT_SEEDS)} split seeds = "
          f"{float(stability_diffs.mean()):+.5f}   SD = {float(stability_diffs.std(ddof=1)):.5f}   "
          f"wins = {stability_wins}/{len(SPLIT_SEEDS)}")

    print()
    print("=== fitted coefficients, the six new (amount- and category-derived) "
          "columns, full corpus, 95% CI ===")
    coef_df = _coefficient_table(assembled)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(coef_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    n_excludes_zero = int(coef_df["excludes_zero"].sum())
    print(f"\n{n_excludes_zero}/{len(coef_df)} of the 18 new coefficients have a 95% CI excluding zero")

    _write_report(
        n_mandates, len(estimable),
        pooled_mean, pooled_se, pooled_t, pooled_df, pooled_p, pooled_n_clusters, pooled_verdict,
        cv_diffs, cv_mean, cv_sd, cv_se, cv_t, cv_df, cv_crit, cv_wins,
        stability_diffs, coef_df, n_excludes_zero,
    )
    print(f"\nwrote {OUT_MD}")


def _write_report(
    n_mandates, n_estimable,
    pooled_mean, pooled_se, pooled_t, pooled_df, pooled_p, pooled_n_clusters, pooled_verdict,
    cv_diffs, cv_mean, cv_sd, cv_se, cv_t, cv_df, cv_crit, cv_wins,
    stability_diffs, coef_df, n_excludes_zero,
) -> None:
    lines: list[str] = []
    lines.append(_SECTION_BEGIN)
    lines.append("## Phase A: amount + category on the frozen corpus")
    lines.append("")
    lines.append(
        f"_Generated {clock_now().strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`python -m eval.design_matrix_comparison`. Corpus: "
        f"{len(TRAIN_SEEDS)} seeds, {n_mandates} mandates, {n_estimable} "
        f"estimable person-period rows (nominal arm, the same corpus "
        f"`src.model.competing_risks.fit()`'s default trains on)._"
    )
    lines.append("")
    lines.append(
        "Widens `FEATURE_COLUMNS` (`const`, `slot_3`, `slot_4`, "
        "`in_salary_window`) with `amount_band_2`, `amount_band_3`, "
        "`amount_band_4` and `category_insurance_premium`, "
        "`category_mutual_fund`, `category_credit_card_bill` -- "
        "`WIDENED_FEATURE_COLUMNS` in `src/model/competing_risks.py`, "
        "available via `fit()`'s `feature_columns` parameter alongside the "
        "unchanged default. Does widening it help? **This is a gate on "
        "measuring and reporting, not on winning** (`reports/gates.md`, "
        "R1a) -- the result below is reported exactly as measured."
    )
    lines.append("")
    lines.append("### Held-out log-loss, widened vs narrow")
    lines.append("")
    lines.append(
        f"**PRIMARY test** -- pooled out-of-fold per-row log-loss "
        f"differences, clustered by `mandate_id` (a mandate contributes "
        f"2-3 correlated person-period rows, so treating rows as "
        f"independent would understate the standard error): "
        f"{n_estimable} rows, {pooled_n_clusters} mandates (clusters). "
        f"mean(widened - narrow) = `{pooled_mean:+.5f}`, "
        f"clustered SE = `{pooled_se:.5f}`, t = `{pooled_t:+.2f}`, "
        f"df = `{pooled_df}`, p = `{pooled_p:.4f}`."
    )
    lines.append("")
    lines.append(f"**Verdict: {pooled_verdict} at 95% confidence (p < 0.05).**")
    lines.append("")
    lines.append(
        f"Secondary/diagnostic cross-check -- {len(cv_diffs)}-fold "
        f"mandate-grouped CV, fold-MEAN differences (only {len(cv_diffs)} "
        f"independent numbers, so this alone is underpowered; the correct "
        f"df={cv_df} critical t at 95% is `{cv_crit:.3f}`, not the "
        f"normal-approximation 2.0 an earlier version of this script used "
        f"to claim significance it had not earned -- corrected here, "
        f"stats-reviewer finding, 2026-09-04): mean(widened - narrow) = "
        f"`{cv_mean:+.5f}`, SD = `{cv_sd:.5f}`, SE = `{cv_se:.5f}`, "
        f"t = `{cv_t:+.2f}`, widened beats narrow on {cv_wins}/{len(cv_diffs)} "
        f"folds. Per-fold (widened - narrow): `{np.round(cv_diffs, 5).tolist()}`. "
        f"Lower log-loss is better; a negative value means the widened "
        f"design predicted the held-out fold better."
    )
    lines.append("")
    lines.append(
        f"20-seed split-stability check (repeated re-splits of the SAME "
        f"corpus -- NOT independent samples, no SE/t derived from this; see "
        f"`eval/model_fit_report.py`'s own docstring for why): "
        f"mean(widened - narrow) = `{float(stability_diffs.mean()):+.5f}`, "
        f"SD = `{float(stability_diffs.std(ddof=1)):.5f}`, widened wins "
        f"{int((stability_diffs < 0).sum())}/{len(stability_diffs)} seeds."
    )
    lines.append("")
    lines.append("### Fitted coefficients, the six new columns (full-corpus fit, 95% CI)")
    lines.append("")
    lines.append("| Outcome | Column | Coef | SE | z | p | 95% CI | Excludes 0? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in coef_df.iterrows():
        lines.append(
            f"| {row['outcome']} | `{row['column']}` | {row['coef']:+.4f} | "
            f"{row['se']:.4f} | {row['z']:+.2f} | {row['p']:.3f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{'yes' if row['excludes_zero'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        f"{n_excludes_zero}/{len(coef_df)} of these 18 coefficients have a "
        f"95% CI excluding zero."
    )
    lines.append("")
    lines.append("### Why this result, whichever way it went, is not a modeling failure")
    lines.append("")
    lines.append(
        "`eval/frozen/simulator.py`'s `_draw_outcome` never reads `category` "
        "in any arm, and reads `amount_paise` only inside the `coupled` "
        "arm's household-balance comparison (`_apply_household_coupling`) "
        "-- never in the base hazard logits this fit trains against "
        "(`nominal`). A near-zero, wide-CI coefficient here is the DGP "
        "telling the truth about itself, not evidence the covariates were "
        "a bad idea to check. `WIDENED_FEATURE_COLUMNS` stays available "
        "via `fit()`'s `feature_columns` parameter; `FEATURE_COLUMNS` (the "
        "production default) stays narrow for the same empirical-neutrality-"
        "plus-parsimony reason this file already excludes "
        "`days_since_last_attempt` and `slot3_x_in_salary_window`. Phase B "
        "(`eval/sim2.py`, appended below once it lands) is where amount- "
        "and category-like covariates get a corpus that actually varies "
        "outcomes by them."
    )
    lines.append("")
    lines.append("### A documentation error, found by stats-reviewer, disclosed here")
    lines.append("")
    lines.append(
        "The amount-band cut points (`_AMOUNT_BAND_CUT_1/2/3` in "
        "`src/model/competing_risks.py`) were originally described as "
        "\"quartiles of the range `fit()` trains on.\" That is false: "
        "`eval/corpus.py`'s `generate()` drops a mandate above ITS OWN "
        "category's AFA-free limit, and the elevated categories "
        "(`insurance_premium`, `mutual_fund`, `credit_card_bill`) carry a "
        "much higher limit (Rs 1,00,000, clause 8(b)) than `subscription` "
        "does (Rs 15,000, clause 8(a)). 316 of 7154 mandates (4.42%) exceed "
        "the cut points' stated Rs 500-14,000 range, up to a measured "
        "maximum of Rs 89,785. The cuts are genuinely the quartiles of "
        "`below_afa_range` (`sim_config.yaml`) -- the standard-category "
        "range alone -- not of the full estimation sample, and the docstring "
        "now says so.\n\n"
        "This also means `amount_band_4` (>= Rs 10,625) is a wide catch-all "
        "spanning Rs 10,625-89,785, and it is CONFOUNDED with category by "
        "the AFA filter itself, not by chance: 25.8% of `subscription` "
        "mandates land in band 4 versus 36.6-37.3% of every elevated "
        "category. Neither issue changes the null result above (both "
        "covariates are non-causal under `nominal` regardless of exactly "
        "where a band boundary falls), but it means these six columns are "
        "NOT fit for a defensibility claim about amount independent of "
        "category on THIS corpus -- a real limitation to fix before Phase "
        "B's covariates, which are meant to carry actual signal, inherit "
        "the same band scheme."
    )
    lines.append(_SECTION_END)

    new_section = "\n".join(lines) + "\n"

    if OUT_MD.exists():
        existing = OUT_MD.read_text(encoding="utf-8")
        if _SECTION_BEGIN in existing and _SECTION_END in existing:
            pre = existing[: existing.index(_SECTION_BEGIN)]
            post = existing[existing.index(_SECTION_END) + len(_SECTION_END):]
            OUT_MD.write_text(pre + new_section + post.lstrip("\n"), encoding="utf-8")
            return
        OUT_MD.write_text(existing.rstrip("\n") + "\n\n" + new_section, encoding="utf-8")
        return

    header = (
        "# Model defensibility\n\n"
        "Answers a reviewer's question the frozen three-bar headline "
        "doesn't: does the hazard model actually use the mandate's own "
        "covariates, and what happens when it does? Two phases -- Phase A "
        "on the frozen corpus (amount, category); Phase B (R1, in "
        "progress) on `eval/sim2.py`, a non-frozen simulator built to let "
        "issuer, instrument type and mandate age actually vary outcomes, "
        "since the frozen simulator does not generate them at all (see "
        "`src/model/features.py`'s `UNSOURCED`). Neither phase feeds the "
        "three-bar headline in `reports/regimes.md`.\n\n"
    )
    OUT_MD.write_text(header + new_section, encoding="utf-8")


if __name__ == "__main__":
    main()
