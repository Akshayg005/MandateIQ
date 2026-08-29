"""eval/gate_criteria.py -- B8's gate (reports/gates.md, 2026-08-29;
clause 2 replaced 2026-08-29, same day, before B8's gate was ever ticked).

Design spec this file pins:
- ATTEMPT_RATE_FLOOR = 0.25 must be FAILED by a null (never-attempt)
  policy, trivially -- that is the whole reason the clause exists.
  Unchanged by clause 2's replacement.
- DISCRIMINATION_MARGIN (now mean ATTEMPTS SPENT, true CANT_PAY_NOW minus
  true CANT_PAY_EVER -- see eval/gate_criteria.py's module docstring for
  why the original boolean "ever attempted" form was replaced) must be
  FAILED by a cause-blind random policy, and CLEARED by an oracle that
  reads true cause and acts perfectly -- both required, because the
  original clause's defect was never having checked the oracle side at
  all. A clause with no daylight between "fails" and "cannot be satisfied
  by anything real" is not a clause.
- All three reference numbers (null, cause-blind-random, oracle) are
  reproduced here from the frozen simulator's own generative behaviour,
  seeds 0-19, constructed directly against eval/frozen/simulator.py -- no
  fitted hazard model, no allocator.py -- so a future change to
  sim_config.yaml (impossible; it is frozen) or to these constants shows
  up as a failing test rather than a silently stale docstring.
"""
from __future__ import annotations

import random
import statistics

import pytest

from src.core.types import Cause, Outcome
from src.policy.constraints import afa_free_limit_paise
from eval.frozen.simulator import Simulator, _logits_from_base_rates, _softmax, load_config
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
    hazard or outcome to build the DENOMINATOR, only initial_cause and
    amount_paise -- reproduced against `nominal` only is representative of
    all three arms. (The ORACLE construction below does call attempt() to
    determine attempts spent, which IS arm-sensitive -- it is deliberately
    scoped to `nominal` only, matching how B8's own gate sweep is scoped.)
    """
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
    the new clause actually closes that gap. Unchanged by clause 2's
    2026-08-29 replacement."""
    mandates = _afa_eligible_mandates(seed=0)
    ids = [m.mandate_id for m in mandates]
    attempted = {m.mandate_id: False for m in mandates}  # never attempts

    rate = attempt_rate(attempted, ids)

    assert rate == 0.0, f"null policy's own attempt rate should be exactly 0.0, got {rate}"
    assert rate < ATTEMPT_RATE_FLOOR, \
        f"null policy (rate={rate}) must FAIL the floor ({ATTEMPT_RATE_FLOOR}) -- it does not"


def test_null_policy_also_scores_zero_on_the_discrimination_gap():
    """Not the clause's main required-fail case (that is the cause-blind
    random policy below -- a null policy already fails on the floor
    alone), but the null policy's own discrimination_gap must be exactly
    0.0: zero attempts spent on every mandate regardless of cause."""
    mandates = _afa_eligible_mandates(seed=0)
    ids = [m.mandate_id for m in mandates]
    attempts_spent = {m.mandate_id: 0 for m in mandates}
    true_cause = {m.mandate_id: m.initial_cause for m in mandates}

    gap = discrimination_gap(attempts_spent, true_cause, ids)
    assert gap == 0.0, f"null policy's discrimination gap should be exactly 0.0, got {gap}"


# === Required-fail test 2: a cause-blind random policy must fail =========

def _cause_blind_random_attempts(mandate_ids: list[str], seed: int) -> dict[str, int]:
    """A fair coin at each of up to 3 retry decision points, cause- and
    outcome-blind -- does not call the simulator at all, so the count is
    drawn independent of everything. The zero-information baseline
    DISCRIMINATION_MARGIN is derived against. RNG stream `seed + 300_000`:
    independent of the simulator's own stream and of the ORIGINAL
    (replaced) clause's uniform-random-at-25% stream (`seed + 100_000`,
    still used by ATTEMPT_RATE_FLOOR's own docstring precedent)."""
    rng = random.Random(seed + 300_000)
    out = {}
    for m in mandate_ids:
        n = 0
        for _ in range(3):
            if rng.random() < 0.5:
                n += 1
            else:
                break
        out[m] = n
    return out


def test_cause_blind_random_policy_fails_the_discrimination_clause():
    """A policy that spends a random number of attempts (fair coin per
    retry point) with NO reference to cause at all -- the precise
    zero-information case the discrimination clause exists to reject.
    Reproduces the exact construction eval/gate_criteria.py's docstring
    cites as DISCRIMINATION_MARGIN's derivation."""
    gaps = []
    for seed in SEEDS:
        mandates = _afa_eligible_mandates(seed)
        ids = [m.mandate_id for m in mandates]
        true_cause = {m.mandate_id: m.initial_cause for m in mandates}
        attempts_spent = _cause_blind_random_attempts(ids, seed)
        gaps.append(discrimination_gap(attempts_spent, true_cause, ids))

    mean_gap = statistics.mean(gaps)
    sd_gap = statistics.stdev(gaps)

    # Pinned so a future change to sim_config.yaml's cause_mix (impossible;
    # frozen) or to this RNG scheme shows up as a failing test.
    assert mean_gap == pytest.approx(-0.015262401928163994, abs=1e-9), \
        f"cause-blind random's own mean discrimination gap drifted: {mean_gap}"
    assert sd_gap == pytest.approx(0.3108523266763266, abs=1e-9), \
        f"cause-blind random's own gap standard deviation drifted: {sd_gap}"

    assert mean_gap < DISCRIMINATION_MARGIN, \
        f"cause-blind random (mean gap={mean_gap:.4f}) must FAIL the discrimination " \
        f"margin ({DISCRIMINATION_MARGIN}) -- it does not"


# === The oracle side: the clause must actually be satisfiable ==============

def _outcome_probs(cause: Cause, in_salary_window: bool, retries_so_far: int, cfg: dict) -> dict[str, float]:
    h = cfg["hazards"][cause.value]
    lg = _logits_from_base_rates(h["base_recovery"], h["base_dead"], h["base_optout"])
    if cause == Cause.CANT_PAY_NOW and in_salary_window:
        lg["recover"] += h.get("salary_window_bonus_logit", 0.0)
    if cause == Cause.WONT_PAY:
        lg["optout"] += h.get("optout_escalation_logit_per_attempt", 0.0) * retries_so_far
    return _softmax(lg)


def _slot1_probs(cause: Cause, cfg: dict) -> dict[str, float]:
    p = _outcome_probs(cause, False, 0, cfg)
    dead, pending = p["dead"], p["survive"]
    t = dead + pending
    return {"dead": dead / t, "survive": pending / t}


def _exact_update(belief: dict, obs_key: str, probs_fn) -> dict:
    un = {c: belief[c] * probs_fn(c)[obs_key] for c in Cause}
    t = sum(un.values())
    return {c: v / t for c, v in un.items()}


_OUTCOME_KEY = {
    Outcome.RECOVERED: "recover",
    Outcome.DEAD: "dead",
    Outcome.OPTED_OUT: "optout",
    Outcome.STILL_PENDING: "survive",
}


def _reference_attempts(seed: int, cfg: dict) -> tuple[dict[str, int], dict[str, Cause], list[str]]:
    """The DELAYED-EVIDENCE reference policy. Sees the same slot-1 decline
    observation the allocator sees (drawn from the frozen config's own
    per-cause hazards, same `seed + 500_000` stream eval/allocator_sweep.py
    uses), but inverts it with the TRUE emission model rather than
    cause_map's independent hand-authored table, and updates exactly on
    every subsequent outcome. Stops attempting the moment CANT_PAY_EVER
    becomes its dominant belief.

    Replaces an earlier zero-delay oracle that read `initial_cause`
    directly and assigned CANT_PAY_EVER exactly 0 attempts by fiat (it
    measured ~1.5816). That construction was withdrawn as the wrong
    reference for a MARGIN: it never waits for evidence, so it cannot
    bound what an evidence-based policy could achieve, and a margin
    referenced to it would reject every real policy by construction. Full
    account: DECISIONS.md, 2026-08-30. This is a reference POLICY's value,
    not a proven supremum over all policies."""
    sim = Simulator("nominal", seed=seed)
    slot1_rng = random.Random(seed + 500_000)
    mandate_ids: list[str] = []
    true_cause: dict[str, Cause] = {}
    attempts_spent: dict[str, int] = {}

    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            continue
        mandate_ids.append(m.mandate_id)
        true_cause[m.mandate_id] = m.initial_cause

        belief = {c: 1.0 / 3.0 for c in Cause}
        obs1 = "dead" if slot1_rng.random() < _slot1_probs(m.initial_cause, cfg)["dead"] else "survive"
        belief = _exact_update(belief, obs1, lambda c: _slot1_probs(c, cfg))

        n, day = 0, 1
        for slot in (2, 3, 4):
            if max(Cause, key=lambda c: belief[c]) == Cause.CANT_PAY_EVER:
                break
            day += 1
            in_win = 1 <= day <= 5
            retries = slot - 2
            result = sim.attempt(m.mandate_id, slot=slot, on_day=day)
            n += 1
            if result.outcome in (Outcome.RECOVERED, Outcome.OPTED_OUT, Outcome.DEAD):
                break
            belief = _exact_update(
                belief, _OUTCOME_KEY[result.outcome], lambda c: _outcome_probs(c, in_win, retries, cfg)
            )
        attempts_spent[m.mandate_id] = n

    return attempts_spent, true_cause, mandate_ids


def test_delayed_evidence_reference_clears_the_discrimination_margin():
    """The validation the ORIGINAL clause skipped entirely: a policy doing
    the best achievable inference UNDER THE SAME EVIDENCE DELAY must clear
    the margin, or the clause has no achievable target and rejects every
    real policy rather than only the ones that ignore cause.

    Deliberately not the zero-delay oracle this test previously used --
    see _reference_attempts' docstring and DECISIONS.md, 2026-08-30."""
    cfg = load_config()
    gaps = [discrimination_gap(*_reference_attempts(seed, cfg)) for seed in SEEDS]

    mean_gap = statistics.mean(gaps)

    assert mean_gap == pytest.approx(0.8329, abs=5e-3), \
        f"delayed-evidence reference's own mean discrimination gap drifted: {mean_gap}"

    assert mean_gap > DISCRIMINATION_MARGIN, \
        f"the reference policy (mean gap={mean_gap:.4f}) must CLEAR the margin " \
        f"({DISCRIMINATION_MARGIN}) -- a clause no real policy can pass measures nothing"

    # Not just "clears": clears with real daylight above the random
    # baseline's own noise, which is what gives the clause resolving power.
    separation_in_random_sd = (mean_gap - (-0.015262401928163994)) / 0.3108523266763266
    assert separation_in_random_sd > 2.0, \
        f"reference-random separation ({separation_in_random_sd:.2f} random-SDs) is too " \
        f"thin for this clause to have real resolving power"


# === Derivation pin: the margin is one pooled SD above the random baseline =

def test_discrimination_margin_is_one_pooled_sd_above_random_baseline():
    """DISCRIMINATION_MARGIN's own value must equal the cause-blind random
    baseline's mean gap plus one pooled standard deviation -- the same
    "clear one pooled SD" convention protocol.md already uses for "beats
    the ladder" claims, not a number chosen by hand and then justified
    after the fact."""
    mean_gap = -0.015262401928163994
    sd_gap = 0.3108523266763266
    assert DISCRIMINATION_MARGIN == pytest.approx(mean_gap + sd_gap, abs=1e-9)


# === Sanity: the floor's own derivation, and why 48.86% was rejected =======

def test_true_cant_pay_now_fraction_is_48_86_percent_not_used_as_the_floor():
    """Pins the number ATTEMPT_RATE_FLOOR's docstring cites as measured-
    but-deliberately-not-used: the true CANT_PAY_NOW fraction of
    AFA-eligible mandates, seeds 0-19. ATTEMPT_RATE_FLOOR (0.25) is
    roughly half of this, not equal to it -- see eval/gate_criteria.py's
    module docstring for why using the exact fraction would have made the
    floor a tuning target rather than a tripwire. Unaffected by clause 2's
    replacement."""
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
            {"m1": 1}, {"m1": Cause.CANT_PAY_NOW}, ["m1"],
        )


def test_discrimination_gap_operates_on_counts_not_booleans():
    """Regression pin for the 2026-08-29 replacement: discrimination_gap
    must accept attempts-spent counts (ints, 0-3) and compute a difference
    of means, not a difference of boolean rates. A count > 1 must actually
    change the result, which a boolean-only implementation could not
    express."""
    attempts_spent = {"a": 3, "b": 0}
    true_cause = {"a": Cause.CANT_PAY_NOW, "b": Cause.CANT_PAY_EVER}
    gap = discrimination_gap(attempts_spent, true_cause, ["a", "b"])
    assert gap == pytest.approx(3.0)
