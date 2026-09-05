"""B8's gate criteria, computed once and pinned here rather than left as
prose in reports/gates.md.

Two clauses, added to close a vacuous check found 2026-08-29, before any
allocator code existed: "zero constraint violations across the eval" is
trivially satisfied by an allocator that never attempts anything -- B5's
null-policy finding (DECISIONS.md, 2026-08-28), recurring in the block
about to start.

1. ATTEMPT_RATE_FLOOR -- the allocator must attempt on at least this
   fraction of AFA-eligible mandates. A null (never-attempt) policy
   attempts 0%, failing trivially. Unchanged since 2026-08-29; this clause
   was never the problem (see clause 2's history below).

2. DISCRIMINATION_MARGIN -- the allocator's mean ATTEMPTS SPENT on true
   CANT_PAY_NOW mandates must exceed its mean attempts spent on true
   CANT_PAY_EVER mandates, averaged across seeds 0-19, by more than this
   margin. A floor alone is vacuous in a subtler way: a policy attempting
   ATTEMPT_RATE_FLOOR uniformly at random, ignoring cause entirely, clears
   clause 1 while demonstrating nothing this project's thesis claims --
   that knowing WHY a payment failed changes what you do. This clause is
   what actually tests that.

Ground truth for scoring: SimMandate.initial_cause, the same field
ATTEMPT_RATE_FLOOR's own denominator (src.policy.constraints.
afa_free_limit_paise applied the same way) is computed from. This is
eval-only, privileged information: an allocator scored against these
criteria must never read initial_cause itself (simulator.py's own "must
never read" warning on that field), exactly as initial_cause and
household_id are already off-limits everywhere else in this repo.

Both constants are derived from the frozen simulator's own generative
behaviour, across the pre-registered seed sweep (seeds 0-19, matching
protocol.md's "beats the ladder" convention), not chosen by hand.

  ATTEMPT_RATE_FLOOR = 0.25 (25%). Unchanged since 2026-08-29 -- see
  reports/gates.md and DECISIONS.md for that derivation (roughly half the
  true CANT_PAY_NOW fraction, deliberately not the fraction itself, so the
  floor is a tripwire against a degenerate policy rather than a
  performance target).

=== 2026-08-29: clause 2's ORIGINAL form was replaced before B8's gate was
ever ticked -- full account below, and in DECISIONS.md ===============

The clause first shipped as attempt-rate discrimination: mean fraction of
mandates EVER attempted (a boolean per mandate), true CANT_PAY_NOW minus
true CANT_PAY_EVER, margin ~0.0808 (one pooled SD above a uniform-random
baseline attempting at exactly the floor rate).

Once an allocator existed to measure it against (this block), that
original form measured a mean gap of 0.0009 -- essentially zero -- despite
the allocator's own unit tests and a separate diagnostic both confirming
real, substantial cause-discrimination was happening (REAUTH correctly
routing 12 of 34 true-CANT_PAY_EVER mandates once evidence existed). The
metric was not detecting a real absence of discrimination; it was
structurally incapable of detecting discrimination AT ALL, for any policy
that has no evidence at the first decision.

Why: this project's allocator (src/policy/allocator.py) starts every
mandate's belief at the uniform reference prior and only updates it from
OBSERVED evidence -- there is no evidence before the first retry attempt,
so the first decision (slot 2) is necessarily cause-blind for any honest
policy. Under reasonable economics, attempting is positive-EV at a neutral
prior for nearly every AFA-eligible mandate, so "ever attempted" is set
True for ~98.6% of BOTH true-CANT_PAY_NOW and true-CANT_PAY_EVER mandates
at that first, uninformed decision -- before cause-discrimination (which
does happen at slot 3+, once a DEAD-type signal updates belief) has any
chance to move a binary that is already pinned to True. A boolean cannot
recover from that; it does not accumulate the information a later, correct
REAUTH decision represents.

This was a process failure, not a bad-luck measurement: the original
clause was approved and pinned as a constant without ever checking that
ANY achievable policy could move it -- specifically, without computing its
value for an ORACLE that reads true cause and acts perfectly, which would
have shown the metric saturates near its ceiling for every policy that
ever attempts a cause-blind first slot, oracle included, leaving no room
between "does nothing" and "perfect" for a real policy to be measured in.
That validation step -- null / cause-blind-random / oracle-ceiling, with
the ceiling required to sit clearly above the random floor -- is now
required of clause 2's replacement below, specifically because skipping it
is what let the first version through unable to measure anything.

=== Replacement, derived and validated BEFORE any allocator score was
consulted (scripts/eval/allocator_sweep.py was not re-run against it until
after this derivation was fixed) =========================================

Candidate: mean ATTEMPTS SPENT (a count in {0,1,2,3} -- retries only, slot
1 is given), true CANT_PAY_NOW minus true CANT_PAY_EVER, over AFA-eligible
mandates. Chosen over the alternative considered (REAUTH-routing rate
conditional on a DEAD-type signal) because it is continuous rather than
binary, so it cannot saturate the way "ever attempted" did -- every slot a
mandate survives, or is correctly routed away from, changes the count. It
is also already one of this project's own three headline bars (recovered,
ATTEMPTS SPENT, mandates preserved -- root DESIGN.md), so a policy that
scores well here is demonstrably practising the thesis, not just satisfying
a bespoke eval statistic.

Validated with the SAME null / random / oracle protocol the original
clause skipped, all three constructed directly against
eval/frozen/simulator.py -- no fitted hazard model, no allocator.py,
computed before this file's own DISCRIMINATION_MARGIN was set or any
allocator score was read:

  NULL   (never attempts): gap = 0.0 exactly, every seed -- trivial, as
         expected; this policy already fails ATTEMPT_RATE_FLOOR regardless.
  RANDOM (a fair coin at each of up to 3 retry points, cause- and
         outcome-blind -- does not call the simulator at all): mean gap
         across seeds 0-19 = -0.015262401928163994, sd = 0.3108523266763266.
         Near zero, as expected: a cause-blind count carries no cause
         information by construction.
  ORACLE (reads true initial_cause; CANT_PAY_EVER spends 0 attempts and
         stops immediately; CANT_PAY_NOW/WONT_PAY attempt every remaining
         slot until a real terminal outcome or the cap, driving the actual
         simulator so "terminal" means what it means everywhere else in
         this project): mean gap = 1.5816 (sd 0.0699) -- the ceiling.

  Separation: oracle - random = 1.5969, or 5.14 standard deviations of the
  random baseline's OWN per-seed spread -- a wide, clearly resolved gap
  between "no information" and "perfect information," unlike the original
  clause where every policy that ever attempts at all lands near the same
  saturated value. This candidate has real resolving power; it is adopted
  on that basis, not on how our own allocator happens to score against it.

  CORRECTION, 2026-08-30 -- the ORACLE row above was the wrong reference,
  and is superseded. It read `initial_cause` directly and assigned
  CANT_PAY_EVER exactly 0 attempts by fiat: zero signal delay, when the
  whole point of the clause is to measure discrimination UNDER delayed
  evidence. That is a category error in a MARGIN's reference point (a
  floor may legitimately cite an unreachable ideal -- that is what makes
  it a tripwire; a margin may not, or it rejects every achievable policy
  by construction). Replaced by a DELAYED-EVIDENCE reference policy: best
  achievable inference given the same observations and the same delay any
  real policy faces -- measured mean gap ~0.8329, separation from random
  2.73 random-SDs, still ample resolving power. Pinned by
  tests/eval/test_gate_criteria.py::
  test_delayed_evidence_reference_clears_the_discrimination_margin.

  DISCRIMINATION_MARGIN's own VALUE is unchanged by that correction, and
  deliberately so: it is derived from the RANDOM baseline (mean + 1 pooled
  SD), and the random baseline ignores the slot-1 signal by construction,
  so its numbers do not move. The correction fixed which policy is cited
  as the achievable ceiling; it did not move the bar. No third
  re-derivation of the threshold occurred.

  DISCRIMINATION_MARGIN = mean + 1 SD of the random baseline (the same
  "clear one pooled SD" convention protocol.md already uses for "beats the
  ladder" claims, reused rather than inventing a new statistic for this
  clause): -0.015262401928163994 + 0.3108523266763266 = 0.29558992474816265.
  Expressed against the mean gap's own sampling distribution (the gate
  evaluates a 20-seed mean): SE = SD/sqrt(20) = 0.06950869334122375, so the
  margin sits 4.25 standard errors from zero.

Full narrative, both required-fail derivations, and the allocator's actual
measured score against this replacement: DECISIONS.md, 2026-08-29.
"""
from __future__ import annotations

from src.core.types import Cause

# Clause 1 of B8's gate (reports/gates.md, 2026-08-29). Unchanged by the
# 2026-08-29 replacement below -- see module docstring for the full
# derivation and why 48.86% was rejected as the floor value.
ATTEMPT_RATE_FLOOR = 0.25

# Clause 2, REPLACED 2026-08-29 (same day as clause 1's original
# derivation -- the first form never survived to a checkpoint). See module
# docstring for the full validation: null=0.0, random mean
# -0.015262401928163994 (sd 0.3108523266763266), oracle ceiling ~1.5816,
# oracle-random separation 5.14 random-SDs. Margin = random mean + 1 random
# SD, full precision, not rounded for tidiness.
DISCRIMINATION_MARGIN = -0.015262401928163994 + 0.3108523266763266


def attempt_rate(attempted: dict[str, bool], mandate_ids: list[str]) -> float:
    """Fraction of `mandate_ids` present as True in `attempted`. Raises
    KeyError if any id is missing -- silently treating a missing id as
    "not attempted" would let an incomplete batch pass by accident."""
    if not mandate_ids:
        raise ValueError("attempt_rate() called with an empty mandate list")
    return sum(1 for m in mandate_ids if attempted[m]) / len(mandate_ids)


def discrimination_gap(
    attempts_spent: dict[str, int],
    true_cause: dict[str, Cause],
    mandate_ids: list[str],
) -> float:
    """Mean ATTEMPTS SPENT (a count, not a boolean -- see the module
    docstring's 2026-08-29 entry for why the original boolean form was
    replaced) on true-CANT_PAY_NOW mandates minus mean attempts spent on
    true-CANT_PAY_EVER mandates, over `mandate_ids`. `true_cause` is
    privileged, eval-only ground truth -- the allocator under test must
    never have read it; this function is called by the SCORER, not by any
    policy code."""
    cpn_ids = [m for m in mandate_ids if true_cause[m] == Cause.CANT_PAY_NOW]
    cpe_ids = [m for m in mandate_ids if true_cause[m] == Cause.CANT_PAY_EVER]
    if not cpn_ids or not cpe_ids:
        raise ValueError(
            "discrimination_gap() needs at least one true-CANT_PAY_NOW and "
            "one true-CANT_PAY_EVER mandate in this batch"
        )
    cpn_mean = sum(attempts_spent[m] for m in cpn_ids) / len(cpn_ids)
    cpe_mean = sum(attempts_spent[m] for m in cpe_ids) / len(cpe_ids)
    return cpn_mean - cpe_mean
