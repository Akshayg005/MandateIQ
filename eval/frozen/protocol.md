# Evaluation protocol -- frozen at block B2

Pre-registered before any file under `src/policy/` exists (verify: `git log
--all --oneline -- src/policy/` is empty as of the freeze commit). This
document, `sim_config.yaml`, and `simulator.py` together are what
`guard_frozen.py` protects. If this file is being read after that commit,
its content has not changed since -- see `reports/FREEZE_HASH`.

## What is measured

The headline is **three bars, never one**: money recovered (paise),
attempts spent, and mandates preserved -- read together, on the same batch,
for every policy under comparison. Recovery-rate-alone is the incumbent's
own metric; reporting it alone would just be re-deriving the ladder's
scorecard and calling it ours.

- **Money recovered** -- sum of `amount_paise` over mandates whose episode
  resolved `RECOVERED`.
- **Attempts spent** -- sum of attempts actually made (slots 2/3/4 only;
  slot 1 is given, not a policy choice). A policy that recovers less money
  by spending fewer attempts is not automatically worse -- NPCI caps every
  mandate at 4 attempts ever, and a wasted attempt on a dead instrument or
  a resistant customer is budget that cannot be spent on a mandate that
  would have recovered.
- **Mandates preserved** -- count of mandates whose final outcome is
  `RECOVERED` or `STILL_PENDING` (budget exhausted, right-censored, still an
  active mandate for the next cycle). `DEAD` and `OPTED_OUT` are the only
  two outcomes that are NOT preserved. This is the metric the ladder cannot
  see itself losing on, because it has no off-ramp and no concept of a
  mandate surviving without this cycle's payment recovering.

Two error costs are carried alongside the three bars from B13 onward, once
a real off-ramp gate exists: **missed recovery** (a mandate that would have
paid, that the policy did not attempt) and **false off-ramp** (a mandate
offered an exit that would in fact have paid). Both are reported; neither is
folded into the three headline bars.

## On which split

Every number reported as a result is computed on a **mandate-level held-out
split** -- a mandate's cycles never appear in both a split used to choose or
tune a policy and the split its number is reported on. The exact split
mechanics (train/calibration/test proportions, the grouping key) are
implemented at B4 (`src/model/splits.py`) once real fitted models exist to
split for; this paragraph commits to the mechanism (mandate-level grouping,
never row-level) before that code is written, which is the part that would
be tempting to loosen quietly if it were decided after seeing a result.

**B2's own baseline-ladder run is not subject to a split at all.** The
ladder consults no model and is fit to nothing -- there is nothing for it to
overfit, so its number is computed over the full frozen batch (all 200
mandates, all three arms). This is deliberately the simplest possible
measurement, precisely because it is the one number in this repository nothing
downstream can contaminate.

## What counts as a win, and a loss -- per arm

- **`nominal`.** The arm whose generative story matches what a
  competing-risks MNLogit model (B5) assumes. A win here is necessary, not
  sufficient: it is evidence the model fits its own assumed world, which is
  the easiest bar in this file to clear.
- **`misspecified`.** Same cause mix and base rates, different functional
  form (cloglog hazard instead of logit) plus a heavier-tailed CANT_PAY_NOW
  replenishment curve and per-attempt cause-switching. A win here is the
  first real evidence: the model was never told the world looks like this.
  A **loss or a narrowed margin here is an expected, reportable finding**,
  not a failure of this protocol -- B13's report is required to show at
  least one regime where the policy loses, explained, and this arm is a
  legitimate candidate for that finding.
- **`coupled`.** Independence, not functional form, is what is varied.
  Mandates share a household balance; recovering one can starve a sibling's
  recovery later the same cycle through pure liquidity contention, which
  `AttemptResult.iatrogenic_insufficient_funds` marks explicitly so this
  effect is measurable rather than inferred after the fact. **This arm is
  not scored primarily on whether the policy "wins."** Its purpose is to
  make PLAN_DETAIL.md's finding 1 -- a per-mandate allocator building a
  debit storm and then misreading its own success as customer illiquidity
  -- observable and falsifiable. A policy that never independently verifies
  batch-level contention (all of B5 through B8, as currently planned) is
  expected to show measurably more iatrogenic failures under `coupled` than
  under `nominal`, and B13's report says so plainly rather than omitting the
  arm because it is unflattering.

## The ladder, precisely

`eval/baseline_ladder.py` (not frozen -- its behaviour is simple, fixed, and
externally specified, so there is nothing to tune) implements Razorpay's
documented incumbent: attempt at fixed offsets T+1, T+2, T+3 days after the
original failure, same amount every time, no decline classification, no
re-authorisation, no off-ramp, halt after either a terminal outcome or after
attempt 4. `run(sim, profile)` is invariant to `profile` by construction --
the ladder does not adapt to either RBI compliance interpretation, which is
itself part of what a compliant policy is supposed to improve on. This is
recorded here, not left to look like an oversight when the same
`BatchResult` shape is produced under both profiles.

## What this protocol does not cover

The five stress regimes named in `PLAN.md` (issuer outage, delayed salary,
mandate-stacking spike, festival season, competitor retry storm) are
pre-registered separately, at B13 (`eval/regimes.py`), not here -- they are
a later, independent commitment, not part of this freeze. Golden-set
accuracy thresholds for the LLM edge (B11) are likewise out of scope for
this document.
