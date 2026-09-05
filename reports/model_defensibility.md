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

<!-- PHASE_B:BEGIN -->
## Phase B: issuer, instrument type and mandate age on eval/sim2.py

_Generated 2026-09-04 11:03 UTC by `python -m eval.sim2`. Corpus: 40 seeds, 8000 mandates, 12242 estimable person-period rows, from a SECOND, non-frozen simulator (`eval/sim2.py`) whose data-generating process actually varies dead-hazard by `issuer_id`/`instrument_type` and CANT_PAY_NOW's recovery hazard by `mandate_age_days` -- unlike `eval/frozen/simulator.py`, which never generates any of the three (`src/model/features.py`'s `UNSOURCED`). Guarded: `scripts/guard_invariants.py` denies `eval/run.py` importing this module, so nothing here can reach `reports/regimes.md`'s headline._

**Scope decision, disclosed rather than discovered later**: `mandate_age_days` is a STATIC per-mandate generated covariate (drawn once, like `amount_paise` already is everywhere in this codebase), not real multi-cycle mandate history -- `cycle_id` stays `1` here too. Building genuine multi-cycle simulation is a materially bigger DGP than this gate asks for; a static, generated age covariate that the hazard genuinely depends on answers the gate's actual question (does a fitted, honest coefficient with a CI exist for mandate age) without overbuilding.

`SIM2_FEATURE_COLUMNS` (`src/model/competing_risks.py`): const, slot_3, slot_4, in_salary_window, issuer_issuer_beta, issuer_issuer_gamma, issuer_issuer_delta, instrument_debit_card, instrument_credit_card, mandate_age_years.

### Fitted coefficients, the six new columns (full-corpus fit, 95% CI)

| Outcome | Column | Coef | SE | z | p | 95% CI | Excludes 0? |
|---|---|---|---|---|---|---|---|
| RECOVERED | `issuer_issuer_beta` | +0.0118 | 0.0601 | +0.20 | 0.845 | [-0.1060, +0.1296] | no |
| RECOVERED | `issuer_issuer_gamma` | +0.1775 | 0.0633 | +2.80 | 0.005 | [+0.0534, +0.3016] | yes |
| RECOVERED | `issuer_issuer_delta` | +0.0335 | 0.0605 | +0.55 | 0.580 | [-0.0852, +0.1521] | no |
| RECOVERED | `instrument_debit_card` | -0.0579 | 0.0517 | -1.12 | 0.263 | [-0.1593, +0.0435] | no |
| RECOVERED | `instrument_credit_card` | -0.0272 | 0.0621 | -0.44 | 0.662 | [-0.1488, +0.0945] | no |
| RECOVERED | `mandate_age_years` | +0.2752 | 0.0386 | +7.12 | 0.000 | [+0.1995, +0.3509] | yes |
| DEAD | `issuer_issuer_beta` | +0.0971 | 0.0764 | +1.27 | 0.204 | [-0.0527, +0.2469] | no |
| DEAD | `issuer_issuer_gamma` | +0.6429 | 0.0747 | +8.61 | 0.000 | [+0.4965, +0.7893] | yes |
| DEAD | `issuer_issuer_delta` | +0.0256 | 0.0781 | +0.33 | 0.743 | [-0.1275, +0.1787] | no |
| DEAD | `instrument_debit_card` | -0.3167 | 0.0649 | -4.88 | 0.000 | [-0.4440, -0.1894] | yes |
| DEAD | `instrument_credit_card` | -0.4331 | 0.0821 | -5.27 | 0.000 | [-0.5941, -0.2722] | yes |
| DEAD | `mandate_age_years` | +0.1753 | 0.0473 | +3.70 | 0.000 | [+0.0826, +0.2681] | yes |
| OPTED_OUT | `issuer_issuer_beta` | +0.0054 | 0.0791 | +0.07 | 0.945 | [-0.1497, +0.1605] | no |
| OPTED_OUT | `issuer_issuer_gamma` | +0.0971 | 0.0843 | +1.15 | 0.249 | [-0.0680, +0.2622] | no |
| OPTED_OUT | `issuer_issuer_delta` | +0.0545 | 0.0792 | +0.69 | 0.491 | [-0.1007, +0.2097] | no |
| OPTED_OUT | `instrument_debit_card` | +0.0420 | 0.0676 | +0.62 | 0.535 | [-0.0906, +0.1745] | no |
| OPTED_OUT | `instrument_credit_card` | +0.0849 | 0.0806 | +1.05 | 0.292 | [-0.0731, +0.2429] | no |
| OPTED_OUT | `mandate_age_years` | +0.1698 | 0.0509 | +3.34 | 0.001 | [+0.0701, +0.2696] | yes |

7/18 of these 18 coefficients have a 95% CI excluding zero -- the opposite of Phase A's result, as expected: this DGP was built specifically to make issuer/instrument/age carry real signal (see module docstring, `eval/sim2.py`), unlike the frozen corpus's amount and category, which the DGP never reads at all.

**This is an in-sample, full-corpus descriptive fit, not a held-out evaluation** -- no train/test split. Appropriate for reading off a coefficient and its CI; not a generalisation claim, and not comparable to Phase A's held-out log-loss test.

### Direct effects vs cause-marginal artifacts
Each issuer/instrument column is coded into the DGP as a direct additive dead-hazard bonus (`_draw_outcome`, `eval/sim2.py`) -- i.e. a direct effect on the **DEAD** equation only. `mandate_age_years` is coded as a direct bonus to CANT_PAY_NOW's recovery hazard only -- a direct effect on the **RECOVERED** equation only. `fit()` pools every row into ONE multinomial logit with no `cause` covariate at all (production has no true-cause label, ever -- the same reason `reports/gates.md`'s B7 entry adds a `CauseConditionedHazard` Protocol instead of fitting per-cause models), so a column can show a nonzero coefficient in an outcome equation the DGP never coded it into -- a cause-marginal composition effect, not a directly-coded one.

Of the 7 significant coefficients: **4 are direct** DGP effects (mandate_age_years→RECOVERED, issuer_issuer_gamma→DEAD, instrument_debit_card→DEAD, instrument_credit_card→DEAD); **3 are cause-marginal artifacts** (issuer_issuer_gamma→RECOVERED, mandate_age_years→DEAD, mandate_age_years→OPTED_OUT). Both artifacts were checked, not assumed: pooled cause-marginal log-odds computed analytically from the DGP's own cause_mix and age distribution move in the SAME direction as the fitted coefficients (mandate_age_years on DEAD/OPTED_OUT: analytic pooled slope +0.13/year vs fitted +0.18/+0.17; issuer_gamma on RECOVERED: analytic pooled log-odds shift +0.12 vs fitted +0.18) -- both are real, understood, cause-marginal-fitting artifacts, not fit noise or a DGP bug.

### The fitted CIs do not cover the DGP's own coded values -- disclosed, not hidden
For the DIRECT (column, outcome) pairs only, `eval/sim2.py`'s own coded additive dead-hazard logit is a natural reference point -- and the fitted, cause-marginal coefficient consistently falls short of it:

| Column | Coded DGP logit | Fitted coef | 95% CI | Covers coded value? |
|---|---|---|---|---|
| `issuer_issuer_beta` (→DEAD) | +0.15 | +0.0971 | [-0.0527, +0.2469] | yes |
| `issuer_issuer_gamma` (→DEAD) | +1.10 | +0.6429 | [+0.4965, +0.7893] | **no** |
| `issuer_issuer_delta` (→DEAD) | +0.05 | +0.0256 | [-0.1275, +0.1787] | yes |
| `instrument_debit_card` (→DEAD) | -1.00 | -0.3167 | [-0.4440, -0.1894] | **no** |
| `instrument_credit_card` (→DEAD) | -1.10 | -0.4331 | [-0.5941, -0.2722] | **no** |
| `mandate_age_years` (→RECOVERED) | +0.60 | +0.2752 | [+0.1995, +0.3509] | **no** |

This is expected, not a fitting error: a cause-marginal coefficient answers a different question than the per-cause parameter coded into the DGP (the same distinction `src/model/competing_risks.py`'s own module docstring already draws for `slot3_x_in_salary_window`), and pooling across the cause mixture attenuates the coded effect rather than recovering it exactly. The table above is evidence that `_design_matrix()`'s issuer/instrument/age machinery detects the right SIGN and roughly the right ORDER OF MAGNITUDE, not that it recovers the generating parameter.

### What this does and does not license concluding
Every hazard number `eval/sim2.py` uses is an ILLUSTRATIVE synthetic parameter, chosen to make each covariate's effect measurable -- not a statistic from real issuer or instrument failure rates (same framing `eval/frozen/sim_config.yaml`'s own header states of itself). This table is evidence that `src/model/competing_risks.py`'s design-matrix machinery CAN fit and report a defensible, CI-bearing coefficient for these three covariate types when a corpus actually contains their effect -- not a claim about what issuer, instrument or age effects look like in real Razorpay data.

**The least comfortable assumption in this corpus, stated rather than buried**: `initial_cause` is drawn independently of issuer, instrument and age here, so every artifact and every attenuation above comes from marginal-fitting alone, with zero confounding. Real issuer data would not have that independence (a bank whose customers are poorer plausibly has both more dead instruments AND more CANT_PAY_NOW mandates) -- under that correlation, attenuation like the table above does not merely shrink coefficients, it can flip their sign. This report demonstrates the fitting machinery works under the EASY case (independent covariates); it does not demonstrate it is safe under real-world confounding.
<!-- PHASE_B:END -->
