# POSTMORTEM — what broke during the build

The rubric line is *"Failure recovery — what broke, and what you did about
it."* That is asking about this build, not runtime resilience. Entries are
written **at the moment of breakage**, before the cause is known. Do not
backfill a tidy story, and do not delete an entry because it turned out to
be your own mistake — those are the valuable ones.

Use the `/log-incident` skill. Format:

## Incident 1 — coupled arm fabricated money in the B2 freeze

**When:** Block B2, 2026-08-26, minutes after the freeze commit (`8321406`).

**Symptom:** `payments-domain` — dispatched as B2's required gate review,
same session, before any file under `src/policy/` existed — reported that a
real batch run of the `coupled` arm recovered ₹3,91,412, roughly 1.7× the
total liquidity (₹2,30,732) that existed across all 50 simulated households.
Independently re-derived by direct computation before acting on the review:
`total_recovered_paise=39141154` against `total_household_balance_paise=23073171`.

**Root cause:** `Simulator._apply_household_coupling` had a "partial
liquidity" branch — when a household's balance fell below the mandate's
amount, it rolled a probability weighted by `balance/amount` for "recovers
anyway." On success it credited the mandate's **full** `amount_paise` while
only debiting the household down to zero. The gap between what was credited
and what the household actually had was created from nothing. It also
wasn't modeling anything real: UPI AutoPay has no partial-debit semantics —
a debit either succeeds in full or is declined.

**Why it wasn't caught earlier:** the pre-freeze test suite
(`tests/eval/test_simulator.py`) verified the storm *effect* — balance
depletes monotonically as household members are attempted in order, later
members show more iatrogenic failures than earlier ones, balance never goes
negative — but never asserted the money-conservation invariant itself
(total recovered across a household must never exceed that household's
starting balance). That was a gap in what I chose to test, not something
the harness structurally prevented me from testing.

**Fix:** removed the probabilistic branch entirely. A household debit now
either succeeds in full (`balance >= amount`, deducted exactly) or fails
outright (iatrogenic `STILL_PENDING`, balance unchanged) — commit
`d634346`, which supersedes the original freeze commit `8321406`.
`reports/FREEZE_HASH` updated to `d634346`. Verified post-fix on a real
batch run: `coupled` recovers ₹52,556.67, which is `<=` the ₹2,30,731.71
total household balance. The corrected numbers are a *more* dramatic,
and now mathematically defensible, demonstration of the storm effect:
recovered mandates dropped from 102 (under `nominal`) to 22 (under
`coupled`), and iatrogenic failures rose from 102 to 138 out of 200.

While re-touching the frozen files for this fix, also corrected
`protocol.md`'s false claim that the misspecified arm shares "the same base
rates" as nominal (mathematically impossible — cloglog dominates logit for
any shared score, so it can only be "same rates, same shape" [the earlier
no-op bug DECISIONS.md already caught] or "different shape, different
realized rates," never both); added "must not read" docstring warnings on
the two ground-truth-only fields (`household_id`,
`iatrogenic_insufficient_funds`) mirroring the existing warning on
`initial_cause`/`effective_cause`; added `on_day` monotonicity validation
to `attempt()` (a non-increasing day previously clamped silently via
`max(days_since_last, 1)` instead of raising). Full reasoning for each in
`DECISIONS.md`.

**Guard added:** `tests/eval/test_simulator.py::test_coupled_arm_never_recovers_more_than_the_household_ever_had`
— asserts, per household, across 20 seeds, that total recovered never
exceeds the household's starting balance. This is the exact invariant whose
absence let the bug ship undetected by my own tests; it is now a permanent
regression test, not just a review finding.
