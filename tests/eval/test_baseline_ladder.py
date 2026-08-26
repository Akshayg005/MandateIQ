"""eval/baseline_ladder.py -- the incumbent. Must never consult a model,
belief state, or anything under src/policy/, and must produce a real number
under every arm (this is the literal B2 gate)."""
from __future__ import annotations

import pathlib
import re

import pytest

from src.core.types import Profile
from eval.baseline_ladder import run
from eval.frozen.simulator import ARMS, Simulator

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LADDER_SRC = ROOT / "eval" / "baseline_ladder.py"


@pytest.mark.parametrize("arm", ARMS)
def test_run_produces_a_number_under_every_arm(arm):
    sim = Simulator(arm, seed=100)
    result = run(sim, Profile.strict)
    assert result.n_mandates == len(sim.mandates)
    assert result.total_recovered_paise >= 0
    assert result.total_attempts_spent > 0
    assert 0 <= result.mandates_preserved <= result.n_mandates


def test_run_recovers_a_plausible_share_of_mandates():
    """Sanity floor: with base_recovery=0.35 for the largest cause share
    (CANT_PAY_NOW at 50%) across up to 3 retries, recovering literally zero
    mandates would indicate a wiring bug, not bad luck."""
    sim = Simulator("nominal", seed=100)
    result = run(sim, Profile.strict)
    assert result.mandates_recovered > 10


def test_run_never_exceeds_three_attempts_per_mandate():
    sim = Simulator("nominal", seed=100)
    result = run(sim, Profile.strict)
    for mandate_result in result.per_mandate:
        assert len(mandate_result.attempts) <= 3


def test_run_is_deterministic_given_the_same_seed():
    a = run(Simulator("nominal", seed=55), Profile.strict)
    b = run(Simulator("nominal", seed=55), Profile.strict)
    assert a.total_recovered_paise == b.total_recovered_paise
    assert a.total_attempts_spent == b.total_attempts_spent
    assert a.mandates_preserved == b.mandates_preserved


def test_run_is_profile_invariant_by_design():
    """The ladder does not adapt to either compliance interpretation --
    confirmed by running the identical simulated batch under both profiles
    and checking every number except the label is identical."""
    strict = run(Simulator("nominal", seed=55), Profile.strict)
    permissive = run(Simulator("nominal", seed=55), Profile.permissive)
    assert strict.total_recovered_paise == permissive.total_recovered_paise
    assert strict.total_attempts_spent == permissive.total_attempts_spent
    assert strict.mandates_preserved == permissive.mandates_preserved
    assert strict.profile == "strict"
    assert permissive.profile == "permissive"


def test_run_never_imports_policy_or_model():
    text = LADDER_SRC.read_text(encoding="utf-8")
    forbidden = [r"from\s+src\.policy\b", r"import\s+src\.policy\b",
                 r"from\s+src\.model\b", r"import\s+src\.model\b"]
    for pattern in forbidden:
        match = re.search(pattern, text)
        assert match is None, f"forbidden import found in baseline_ladder.py: {match.group(0)!r}"


def test_coupled_arm_shows_more_iatrogenic_failures_than_nominal_under_the_ladder():
    """The actual policy-relevant version of the storm test: run the SAME
    fixed-cadence ladder (which attempts every mandate's slot 2 on the same
    calendar day, exactly like a batch-scheduling policy with no capacity
    awareness would) over both arms, and confirm the coupled arm's real
    hazard mix still produces measurably more iatrogenic failures than
    nominal -- not just in the isolated, tuned scenario in
    test_simulator.py, but under the actual incumbent policy's schedule."""
    nominal_total_iatrogenic = 0
    coupled_total_iatrogenic = 0
    for seed in range(10):
        nominal_total_iatrogenic += run(Simulator("nominal", seed=seed), Profile.strict).total_iatrogenic_failures
        coupled_total_iatrogenic += run(Simulator("coupled", seed=seed), Profile.strict).total_iatrogenic_failures
    assert nominal_total_iatrogenic == 0  # nominal has no coupling mechanic at all
    assert coupled_total_iatrogenic > 0
