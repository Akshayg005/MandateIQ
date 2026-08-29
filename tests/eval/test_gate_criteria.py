"""eval/gate_criteria.py -- B8's amended gate (reports/gates.md,
2026-08-29). Written before any allocator code exists; a gate without a
proven failing case is not a gate, so this file's two required-fail tests
are the point, not an afterthought.

Design spec this file pins:
- ATTEMPT_RATE_FLOOR = 0.25 must be FAILED by a null (never-attempt)
  policy, trivially -- that is the whole reason the clause exists.
- DISCRIMINATION_MARGIN = 0.0808 must be FAILED by a uniform-random policy
  attempting at exactly ATTEMPT_RATE_FLOOR, independent of cause -- the
  precise borderline case the clause exists to reject (a policy that
  clears the floor while discriminating on nothing).
- Both constants are reproduced here from the frozen simulator's own
  generative parameters, seeds 0-19, so a future change to sim_config.yaml
  (impossible; it is frozen) or to these constants shows up as a failing
  test rather than a silently stale docstring.
"""
from __future__ import annotations

import random
import statistics

import pytest

from src.core.types import Cause
from src.policy.constraints import afa_free_limit_paise
from eval.frozen.simulator import Simulator
from eval.gate_criteria import (
    ATTEMPT_RATE_FLOOR,
    DISCRIMINATION_MARGIN,
    attempt_rate,
    discrimination_gap,
)

SEEDS = range(20)  # seeds 0-19, protocol.md's own "beats the ladder" sweep


def _afa_eligible_mandates(seed: int):
    """AFA-eligible mandates for one seed's `nominal` draw, with ground
    truth. Arm-independent: cause_mix/amount_paise are shared across all
    three frozen arms in sim_config.yaml, and this file never reads a
    hazard or outcome, only initial_cause and amount_paise -- reproduced
    against `nominal` only is representative of all three arms."""
    sim = Simulator("nominal", seed=seed)
    out = []
    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            continue
        out.append(m)
    return out


# === Required-fail test 1: the null policy must fail the floor =============

def test_null_policy_fails_the_attempt_rate_floor():
    """A policy that attempts nothing has attempt_rate 0.0. This is the
    entire reason ATTEMPT_RATE_FLOOR exists: "zero constraint violations"
    was vacuously satisfiable by an allocator that never acts at all
    (B5's null-policy finding, recurring at B8) -- this test is the proof
    the new clause actually closes that gap."""
    mandates = _afa_eligible_mandates(seed=0)
    ids = [m.mandate_id for m in mandates]
    attempted = {m.mandate_id: False for m in mandates}  # never attempts

    rate = attempt_rate(attempted, ids)

    assert rate == 0.0, f"null policy's own attempt rate should be exactly 0.0, got {rate}"
    assert rate < ATTEMPT_RATE_FLOOR, \
        f"null policy (rate={rate}) must FAIL the floor ({ATTEMPT_RATE_FLOOR}) -- it does not"


# === Required-fail test 2: uniform-random must fail the discrimination clause =

def test_uniform_random_at_floor_rate_fails_the_discrimination_clause():
    """A policy attempting ATTEMPT_RATE_FLOOR (25%) of mandates uniformly
    at random, independent of cause, clears clause 1 (attempt rate = the
    floor, exactly) while discriminating on nothing -- the "subtler"
    vacuous case the discrimination clause exists to reject. Reproduces
    the exact simulation eval/gate_criteria.py's docstring cites as
    DISCRIMINATION_MARGIN's derivation: same seed range, same RNG offset
    (seed + 100_000, a stream independent of the simulator's own RNG),
    same floor rate."""
    gaps = []
    for seed in SEEDS:
        mandates = _afa_eligible_mandates(seed)
        rng = random.Random(seed + 100_000)
        attempted = {m.mandate_id: rng.random() < ATTEMPT_RATE_FLOOR for m in mandates}
        true_cause = {m.mandate_id: m.initial_cause for m in mandates}
        ids = [m.mandate_id for m in mandates]
        gaps.append(discrimination_gap(attempted, true_cause, ids))

    mean_gap = statistics.mean(gaps)
    sd_gap = statistics.stdev(gaps)

    # The value cited in eval/gate_criteria.py's docstring as "should be
    # ~0" -- pinned here so a future change to sim_config.yaml's cause_mix
    # (impossible; frozen) or to the RNG scheme shows up as a test failure.
    assert mean_gap == pytest.approx(-0.006778399979983418, abs=1e-9), \
        f"uniform-random's own mean discrimination gap drifted: {mean_gap}"
    assert sd_gap == pytest.approx(0.08762616316461531, abs=1e-9), \
        f"uniform-random's own gap standard deviation drifted: {sd_gap}"

    assert mean_gap < DISCRIMINATION_MARGIN, \
        f"uniform-random (mean gap={mean_gap:.4f}) must FAIL the discrimination " \
        f"margin ({DISCRIMINATION_MARGIN}) -- it does not"


# === Derivation pin: the margin is one pooled SD above the random baseline =

def test_discrimination_margin_is_one_pooled_sd_above_random_baseline():
    """DISCRIMINATION_MARGIN's own value must equal the uniform-random
    baseline's mean gap plus one pooled standard deviation -- the same
    "clear one pooled SD" convention protocol.md already uses for "beats
    the ladder" claims, not a number chosen by hand and then justified
    after the fact."""
    mean_gap = -0.006778399979983418
    sd_gap = 0.08762616316461531
    assert DISCRIMINATION_MARGIN == pytest.approx(mean_gap + sd_gap, abs=1e-9)


# === Sanity: the floor's own derivation, and why 48.86% was rejected =======

def test_true_cant_pay_now_fraction_is_48_86_percent_not_used_as_the_floor():
    """Pins the number ATTEMPT_RATE_FLOOR's docstring cites as measured-
    but-deliberately-not-used: the true CANT_PAY_NOW fraction of
    AFA-eligible mandates, seeds 0-19. ATTEMPT_RATE_FLOOR (0.25) is
    roughly half of this, not equal to it -- see eval/gate_criteria.py's
    module docstring for why using the exact fraction would have made the
    floor a tuning target rather than a tripwire."""
    cpn = total = 0
    for seed in SEEDS:
        for m in _afa_eligible_mandates(seed):
            total += 1
            if m.initial_cause == Cause.CANT_PAY_NOW:
                cpn += 1

    fraction = cpn / total
    assert fraction == pytest.approx(0.48861743475846753, abs=1e-9), \
        f"true CANT_PAY_NOW fraction drifted from the measured 48.86%: {fraction}"
    assert ATTEMPT_RATE_FLOOR < fraction, \
        "the floor must stay below the true fraction -- it is a tripwire, not a target"


# === Module-level helper contracts =========================================

def test_attempt_rate_raises_on_empty_mandate_list():
    with pytest.raises(ValueError):
        attempt_rate({}, [])


def test_discrimination_gap_raises_without_both_causes_present():
    with pytest.raises(ValueError):
        discrimination_gap(
            {"m1": True}, {"m1": Cause.CANT_PAY_NOW}, ["m1"],
        )
