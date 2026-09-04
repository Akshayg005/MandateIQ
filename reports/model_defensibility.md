# Model defensibility

Answers a reviewer's question the frozen three-bar headline doesn't: does the hazard model actually use the mandate's own covariates, and what happens when it does? Two phases -- Phase A on the frozen corpus (amount, category); Phase B (R1, in progress) on `eval/sim2.py`, a non-frozen simulator built to let issuer, instrument type and mandate age actually vary outcomes, since the frozen simulator does not generate them at all (see `src/model/features.py`'s `UNSOURCED`). Neither phase feeds the three-bar headline in `reports/regimes.md`.

<!-- PHASE_A:BEGIN -->
## Phase A: amount + category on the frozen corpus

_Generated 2026-09-03 23:00 UTC by `python -m eval.design_matrix_comparison`. Corpus: 40 seeds, 7154 mandates, 12316 estimable person-period rows (nominal arm, the same corpus `src.model.competing_risks.fit()`'s default trains on)._

Widens `FEATURE_COLUMNS` (`const`, `slot_3`, `slot_4`, `in_salary_window`) with `amount_band_2`, `amount_band_3`, `amount_band_4` and `category_insurance_premium`, `category_mutual_fund`, `category_credit_card_bill` -- `WIDENED_FEATURE_COLUMNS` in `src/model/competing_risks.py`, available via `fit()`'s `feature_columns` parameter alongside the unchanged default. Does widening it help? **This is a gate on measuring and reporting, not on winning** (`reports/gates.md`, R1a) -- the result below is reported exactly as measured.

### Held-out log-loss, widened vs narrow

**PRIMARY test** -- pooled out-of-fold per-row log-loss differences, clustered by `mandate_id` (a mandate contributes 2-3 correlated person-period rows, so treating rows as independent would understate the standard error): 12316 rows, 7154 mandates (clusters). mean(widened - narrow) = `+0.00103`, clustered SE = `0.00036`, t = `+2.88`, df = `7153`, p = `0.0040`.

**Verdict: WIDENED DOES NOT BEAT (is worse than) narrow at 95% confidence (p < 0.05).**

Secondary/diagnostic cross-check -- 5-fold mandate-grouped CV, fold-MEAN differences (only 5 independent numbers, so this alone is underpowered; the correct df=4 critical t at 95% is `2.776`, not the normal-approximation 2.0 an earlier version of this script used to claim significance it had not earned -- corrected here, stats-reviewer finding, 2026-09-04): mean(widened - narrow) = `+0.00103`, SD = `0.00097`, SE = `0.00043`, t = `+2.38`, widened beats narrow on 0/5 folds. Per-fold (widened - narrow): `[0.00272, 0.00026, 0.00087, 0.00073, 0.00058]`. Lower log-loss is better; a negative value means the widened design predicted the held-out fold better.

20-seed split-stability check (repeated re-splits of the SAME corpus -- NOT independent samples, no SE/t derived from this; see `eval/model_fit_report.py`'s own docstring for why): mean(widened - narrow) = `+0.00129`, SD = `0.00179`, widened wins 5/20 seeds.

### Fitted coefficients, the six new columns (full-corpus fit, 95% CI)

| Outcome | Column | Coef | SE | z | p | 95% CI | Excludes 0? |
|---|---|---|---|---|---|---|---|
| RECOVERED | `amount_band_2` | +0.0511 | 0.0660 | +0.77 | 0.439 | [-0.0783, +0.1806] | no |
| RECOVERED | `amount_band_3` | +0.0512 | 0.0650 | +0.79 | 0.431 | [-0.0762, +0.1786] | no |
| RECOVERED | `amount_band_4` | -0.0528 | 0.0620 | -0.85 | 0.394 | [-0.1743, +0.0686] | no |
| RECOVERED | `category_insurance_premium` | +0.0514 | 0.0612 | +0.84 | 0.401 | [-0.0686, +0.1713] | no |
| RECOVERED | `category_mutual_fund` | -0.0221 | 0.0717 | -0.31 | 0.758 | [-0.1627, +0.1185] | no |
| RECOVERED | `category_credit_card_bill` | +0.0386 | 0.1040 | +0.37 | 0.711 | [-0.1652, +0.2424] | no |
| DEAD | `amount_band_2` | +0.0605 | 0.0852 | +0.71 | 0.478 | [-0.1065, +0.2275] | no |
| DEAD | `amount_band_3` | -0.0139 | 0.0852 | -0.16 | 0.870 | [-0.1810, +0.1531] | no |
| DEAD | `amount_band_4` | -0.1011 | 0.0812 | -1.24 | 0.213 | [-0.2603, +0.0581] | no |
| DEAD | `category_insurance_premium` | -0.0315 | 0.0817 | -0.39 | 0.700 | [-0.1915, +0.1286] | no |
| DEAD | `category_mutual_fund` | -0.1195 | 0.0968 | -1.24 | 0.217 | [-0.3092, +0.0701] | no |
| DEAD | `category_credit_card_bill` | +0.1217 | 0.1308 | +0.93 | 0.352 | [-0.1346, +0.3780] | no |
| OPTED_OUT | `amount_band_2` | +0.0871 | 0.0802 | +1.09 | 0.278 | [-0.0701, +0.2443] | no |
| OPTED_OUT | `amount_band_3` | +0.0317 | 0.0798 | +0.40 | 0.691 | [-0.1247, +0.1881] | no |
| OPTED_OUT | `amount_band_4` | -0.0248 | 0.0755 | -0.33 | 0.743 | [-0.1728, +0.1232] | no |
| OPTED_OUT | `category_insurance_premium` | +0.0605 | 0.0740 | +0.82 | 0.413 | [-0.0845, +0.2055] | no |
| OPTED_OUT | `category_mutual_fund` | -0.1521 | 0.0907 | -1.68 | 0.094 | [-0.3299, +0.0257] | no |
| OPTED_OUT | `category_credit_card_bill` | +0.0249 | 0.1264 | +0.20 | 0.844 | [-0.2228, +0.2726] | no |

0/18 of these 18 coefficients have a 95% CI excluding zero.

### Why this result, whichever way it went, is not a modeling failure

`eval/frozen/simulator.py`'s `_draw_outcome` never reads `category` in any arm, and reads `amount_paise` only inside the `coupled` arm's household-balance comparison (`_apply_household_coupling`) -- never in the base hazard logits this fit trains against (`nominal`). A near-zero, wide-CI coefficient here is the DGP telling the truth about itself, not evidence the covariates were a bad idea to check. `WIDENED_FEATURE_COLUMNS` stays available via `fit()`'s `feature_columns` parameter; `FEATURE_COLUMNS` (the production default) stays narrow for the same empirical-neutrality-plus-parsimony reason this file already excludes `days_since_last_attempt` and `slot3_x_in_salary_window`. Phase B (`eval/sim2.py`, appended below once it lands) is where amount- and category-like covariates get a corpus that actually varies outcomes by them.

### A documentation error, found by stats-reviewer, disclosed here

The amount-band cut points (`_AMOUNT_BAND_CUT_1/2/3` in `src/model/competing_risks.py`) were originally described as "quartiles of the range `fit()` trains on." That is false: `eval/corpus.py`'s `generate()` drops a mandate above ITS OWN category's AFA-free limit, and the elevated categories (`insurance_premium`, `mutual_fund`, `credit_card_bill`) carry a much higher limit (Rs 1,00,000, clause 8(b)) than `subscription` does (Rs 15,000, clause 8(a)). 316 of 7154 mandates (4.42%) exceed the cut points' stated Rs 500-14,000 range, up to a measured maximum of Rs 89,785. The cuts are genuinely the quartiles of `below_afa_range` (`sim_config.yaml`) -- the standard-category range alone -- not of the full estimation sample, and the docstring now says so.

This also means `amount_band_4` (>= Rs 10,625) is a wide catch-all spanning Rs 10,625-89,785, and it is CONFOUNDED with category by the AFA filter itself, not by chance: 25.8% of `subscription` mandates land in band 4 versus 36.6-37.3% of every elevated category. Neither issue changes the null result above (both covariates are non-causal under `nominal` regardless of exactly where a band boundary falls), but it means these six columns are NOT fit for a defensibility claim about amount independent of category on THIS corpus -- a real limitation to fix before Phase B's covariates, which are meant to carry actual signal, inherit the same band scheme.
<!-- PHASE_A:END -->
