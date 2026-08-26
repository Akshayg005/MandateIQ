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
  the easiest bar in this file to clear. In short: this arm tests correctness of cause handling, not scheduling skill — it is timing-invariant by construction (a privileged oracle ties the fixed ladder exactly on recovered money, 20/20 seeds; see DECISIONS.md, 2026-08-27).
- **`misspecified`.** Same cause mix and the same underlying additive
  log-odds scores that reproduce nominal's stated base rates *under a logit
  link* -- but a genuinely different link function, not merely a reshaping
  of the same numbers. Cloglog mathematically dominates logit for any
  shared score (`1 - exp(-exp(s)) >= 1/(1 + exp(-s))` for every real `s`,
  with equality only in the limit), so misspecified's *realized* hazards
  are uniformly higher than nominal's stated base rates -- e.g. CANT_PAY_NOW
  recovery moves from a stated 0.35 to roughly 0.43 at the base score, purely
  from the link. **This is the honest story, not a bug to paper over**: a
  model that assumes logit when the true process is closer to cloglog will
  systematically *under-predict every hazard*, not just get a curve's shape
  slightly wrong -- and that is a stronger, more falsifiable misspecification
  claim than "same rates, different shape" would have been. (An earlier
  draft of this paragraph claimed "same base rates"; that was checked
  against the implementation and found false -- see DECISIONS.md.) Plus a
  heavier-tailed CANT_PAY_NOW replenishment curve and per-attempt
  cause-switching. A win here is the first real evidence: the model was
  never told the world looks like this. A **loss or a narrowed margin here
  is an expected, reportable finding**, not a failure of this protocol --
  B13's report is required to show at least one regime where the policy
  loses, explained, and this arm is a legitimate candidate for that finding. A privileged timing oracle confirms the intended asymmetry directly: it beats the ladder on recovered money decisively and consistently (20/20 seeds, mean/SE=7.66) — this arm is where real scheduling skill is actually measurable (see DECISIONS.md, 2026-08-27).
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
  arm because it is unflattering. In short: this arm tests whether a policy wastes less under contention, not whether it recovers more. A privileged oracle confirms recovered money is a coin flip here even under perfect knowledge (9/11/0 wins, mean/SE=−0.22, 20 seeds) — this arm's "beats the ladder" claim binds on attempts spent and iatrogenic count instead, never on recovered money (see DECISIONS.md, 2026-08-27, and B5's gate in reports/gates.md).

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

## Comparisons are across seeds, not one seed

`sim_config.yaml`'s single `seed` is what B2's own baseline-ladder number
uses (deterministic, reproducible with one command), but it is **not** what
"beats the ladder" means from B5 onward. A 40-seed sweep of the ladder under
`nominal` shows real run-to-run variance (recovered paise, mean/seed batch:
CV ~19%). Any reported comparison against the ladder from B5 onward is
computed across a pre-registered sweep of **seeds 0-19** (distinct from the
frozen config's own seed), reporting mean and standard deviation for both
the candidate and the ladder; a "beats the ladder" claim requires the
candidate's mean to exceed the ladder's mean by more than one pooled
standard deviation. A single-seed comparison is not evidence of anything by
itself and must not be reported as a result.

## Known limitations, disclosed rather than hidden

These are documented here, at the freeze, rather than discovered and
explained away after a result is seen.

- **Household balance in `coupled` has no exogenous competition.** Nothing
  except our own debits ever draws it down -- no EMI, no rent NACH, no other
  merchant's AutoPay. In production our mandates are a minority of what
  competes for a household's balance in the salary window, so `coupled`
  understates real-world contention rather than overstating it. The salary-
  window recovery bonus (`salary_window_bonus_logit`) is consequently pure
  upside in every arm, including `coupled` -- it does not carry the
  offsetting "everyone else is also debiting this window" cost a real
  household would face. A `mandate-stacking spike` stress regime (B13) is
  the more appropriate place to model exogenous competition explicitly,
  not a retrofit onto this frozen arm.
- **Coupling depletes by call order within a cycle, not by calendar
  distance.** Consistent with "fresh balance at cycle start, no mid-cycle
  top-up" (deliberate, stated above) -- but it means spreading a household's
  retries across more days does not by itself reduce contention; only
  attempting fewer members of the same household does. A policy is
  therefore rewarded by this arm for reading `household_id` and avoiding
  siblings, which is NOT a real capability (see `simulator.py`'s
  `SimMandate` docstring: `household_id` is unobservable ground truth,
  a policy under test must never read it) and NOT the same thing as
  solving batch capacity. `iatrogenic_insufficient_funds` totals should
  always be read alongside attempts and money recovered, never alone --
  a policy that trivially wins on iatrogenic count by attempting less is
  not evidence it solved anything.
- **`cause_switch_prob`'s marginal distribution is invariant by
  construction** -- redrawing from the same `cause_mix` at each attempt
  does not shift the population-level hazard rate a pooled MNLogit (B5)
  fits. Its intended target is the *within-mandate* stationarity
  assumption the belief update (`PLAN_DETAIL.md` §4, `update(b, e) ∝
  b[c]·P(e|c)`) relies on, not B5's marginal fit -- expect this arm to
  stress B7/B8 more than B5.
- **`replenishment_exponent`'s heavy-tailed boost is inert under the
  ladder's fixed cadence.** T+1/T+2/T+3 means `days_since_last_attempt`
  is always exactly 1 at every retry, so the boost (which scales with that
  gap) never activates under the only policy that has run against this arm
  so far. It is real and tested in isolation
  (`tests/eval/test_simulator.py`), and will start to matter the moment a
  future policy chooses variable retry spacing -- which is the point.
- **A policy that chooses STOP or OFFER without ever attempting a mandate
  needs its own scoring path.** `score_mandate` requires at least one
  `AttemptResult` because this simulator only models the ATTEMPT action;
  REAUTH and OFFER have no outcome model here at all (their probabilistic
  behaviour is B8's design, per `PLAN_DETAIL.md` §4's `Q(b, REAUTH, ...)`
  and `Q(b, OFFER, ...)`). This is not a defect in this scorer, but B8 must
  design how a batch mixing ATTEMPT/REAUTH/OFFER/STOP gets scored before
  it can be evaluated against this harness. 
- **`coupled`'s contention order is fixed by mandate-generation index, not
  randomized.** `household_id = f"H{i // household_size}"`
  (`simulator.py:181-183`) assigns mandates to households by slicing them
  in the same order they were generated -- so which mandate lands first,
  second, etc. within a household correlates with generation index, never
  with anything a policy under test could observe or be adversarially
  confounded by. Disclosed so it is clear `coupled` tests contention
  itself, not resilience to a hostile ordering.
- **`coupled` does not discriminate policies on recovered money -- it
  discriminates on attempts spent and iatrogenic count instead.** A
  privileged oracle with perfect knowledge of timing, cause, and
  cause-switching (`eval/oracle_policy.py::run`) ties the fixed ladder on
  recovered money to within noise over a 20-seed sweep (9 wins / 11 losses
  / 0 ties, mean/SE=-0.22) -- no real policy can be expected to show a
  money-recovered edge here. A cause-aware extension of the same oracle
  (`run_cause_aware`, same file) that additionally skips `CANT_PAY_EVER`
  and never attempts `WONT_PAY` does discriminate, on attempts spent and
  iatrogenic count (`eval/cause_aware_headroom.py`, 20-seed sweep:
  iatrogenic failures 128.1 -> 119.0, mean/SE~5.6). B5's "beats the ladder"
  gate (`reports/gates.md`) binds on these two metrics for `coupled`, and
  on recovered money for `misspecified` only -- see `DECISIONS.md`,
  2026-08-27.
- **`nominal` shows zero oracle-ladder gap on recovered money, by
  construction.** The same privileged timing oracle ties the fixed ladder
  exactly on every one of 20 seeds (20/20 exact ties) -- `nominal` has no
  timing-sensitive mechanic for a smarter schedule to exploit, so a win
  was never possible here and the tie is not evidence either way, for or
  against a candidate policy. See `DECISIONS.md`, 2026-08-27.

## What this protocol does not cover

The five stress regimes named in `PLAN.md` (issuer outage, delayed salary,
mandate-stacking spike, festival season, competitor retry storm) are
pre-registered separately, at B13 (`eval/regimes.py`), not here -- they are
a later, independent commitment, not part of this freeze. Golden-set
accuracy thresholds for the LLM edge (B11) are likewise out of scope for
this document.
