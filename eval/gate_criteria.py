"""B8's gate criteria, computed once (2026-08-29, before any allocator
code exists) and pinned here rather than left as prose in reports/gates.md.

Two clauses, added to close a vacuous-check found the same day: "zero
constraint violations across the eval" is trivially satisfied by an
allocator that never attempts anything -- B5's null-policy finding
(DECISIONS.md, 2026-08-28), recurring in the block about to start.

1. ATTEMPT_RATE_FLOOR -- the allocator must attempt on at least this
   fraction of AFA-eligible mandates. A null (never-attempt) policy
   attempts 0%, failing trivially.

2. DISCRIMINATION_MARGIN -- the allocator's mean attempt rate on true
   CANT_PAY_NOW mandates must exceed its mean attempt rate on true
   CANT_PAY_EVER mandates, averaged across seeds 0-19, by more than this
   margin. A floor alone is vacuous in a subtler way: a policy attempting
   ATTEMPT_RATE_FLOOR uniformly at random, ignoring cause entirely,
   clears clause 1 while demonstrating nothing this project's thesis
   claims -- that knowing WHY a payment failed changes what you do. This
   clause is what actually tests that.

Ground truth for scoring: `SimMandate.initial_cause`, the same field
ATTEMPT_RATE_FLOOR's own denominator (`src/policy/constraints.
afa_free_limit_paise` applied the same way) is computed from -- consistent
with how the floor itself was derived, not a separate convention. This is
eval-only, privileged information: an allocator scored against these
criteria must never read `initial_cause` itself (simulator.py's own
"must never read" warning on that field), exactly as `initial_cause`/
`household_id` are already off-limits everywhere else in this repo.

Both constants are derived from the frozen simulator's own generative
parameters, across the pre-registered seed sweep (seeds 0-19, matching
protocol.md's "beats the ladder" convention), not chosen by hand:

  ATTEMPT_RATE_FLOOR = 0.25 (25%). The true CANT_PAY_NOW fraction of
  AFA-eligible mandates, measured the same way, is 48.86% (1760/3602,
  seeds 0-19, identical across all three frozen arms since cause_mix does
  not vary by arm) -- DELIBERATELY NOT used as the floor. A belief-based
  allocator does not observe cause, and a correct policy that declines a
  handful of true-CANT_PAY_NOW mandates on negative expected value (a
  legitimate stopping decision, not a violation) would land just under
  48.86% and fail a floor set there -- and the cheapest fix would be to
  attempt more often to clear the threshold, tuning the allocator to the
  grading axis rather than to value, the same shape rejected at B5, B7,
  and the paired-criterion reversal (DECISIONS.md, 2026-08-28/29). 25% is
  roughly half the true fraction: unreachable by a null policy, unreachable
  by any policy that ignores ATTEMPT, comfortably clear of legitimate
  stopping behaviour. A tripwire against a degenerate policy, not a
  performance target.

  DISCRIMINATION_MARGIN ~= 0.0808 (full precision below; this paragraph
  rounds for readability). A uniform-random policy attempting at exactly
  ATTEMPT_RATE_FLOOR (25%), independent of cause -- the precise
  borderline case this clause exists to reject -- was simulated across
  seeds 0-19 (see tests/eval/test_gate_criteria.py for the exact
  reproduction). Its own mean discrimination gap across those 20 seeds was
  -0.0068 (~0, as expected: the random draw is independent of cause by
  construction) with a per-seed standard deviation of 0.0876. The margin
  is set at one pooled SD above that random baseline's own mean --
  ~0.0808 -- the same "clear one pooled SD" convention protocol.md already
  uses for "beats the ladder" claims, not a new statistic invented for
  this clause. Expressed against the MEAN gap's own sampling distribution
  (SE = SD/sqrt(20) = 0.0196, since the gate evaluates a 20-seed mean, the
  same way every other "beats X" claim in this project is evaluated): the
  margin sits 4.13 standard errors from zero -- not something a
  non-discriminating policy clears by chance at that granularity.
"""
from __future__ import annotations

from src.core.types import Cause

# Clause 1 of B8's amended gate (reports/gates.md, 2026-08-29). See module
# docstring for the full derivation and why 48.86% was rejected.
ATTEMPT_RATE_FLOOR = 0.25

# Clause 2 of B8's amended gate. See module docstring for the derivation:
# the uniform-random baseline's own mean gap (-0.006778399979983418) plus
# one pooled standard deviation (0.08762616316461531), full precision --
# not rounded for tidiness, matching this project's "no unattributed magic
# numbers" convention for a number that IS the derivation, not a display
# value.
DISCRIMINATION_MARGIN = -0.006778399979983418 + 0.08762616316461531


def attempt_rate(attempted: dict[str, bool], mandate_ids: list[str]) -> float:
    """Fraction of `mandate_ids` present as True in `attempted`. Raises
    KeyError if any id is missing -- silently treating a missing id as
    "not attempted" would let an incomplete batch pass by accident."""
    if not mandate_ids:
        raise ValueError("attempt_rate() called with an empty mandate list")
    return sum(1 for m in mandate_ids if attempted[m]) / len(mandate_ids)


def discrimination_gap(
    attempted: dict[str, bool],
    true_cause: dict[str, Cause],
    mandate_ids: list[str],
) -> float:
    """Mean CANT_PAY_NOW attempt rate minus mean CANT_PAY_EVER attempt
    rate, over `mandate_ids`. `true_cause` is privileged, eval-only ground
    truth -- the allocator under test must never have read it; this
    function is called by the SCORER, not by any policy code."""
    cpn_ids = [m for m in mandate_ids if true_cause[m] == Cause.CANT_PAY_NOW]
    cpe_ids = [m for m in mandate_ids if true_cause[m] == Cause.CANT_PAY_EVER]
    if not cpn_ids or not cpe_ids:
        raise ValueError(
            "discrimination_gap() needs at least one true-CANT_PAY_NOW and "
            "one true-CANT_PAY_EVER mandate in this batch"
        )
    return attempt_rate(attempted, cpn_ids) - attempt_rate(attempted, cpe_ids)
