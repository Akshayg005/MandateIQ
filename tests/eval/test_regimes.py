"""B13 -- the pre-registered stress regimes.

These tests protect two things the block's gate depends on: that a regime is
a config OVERLAY and never an edit to the frozen simulator, and that an
overlay which silently does nothing is impossible.
"""
from __future__ import annotations

import copy

import pytest

from eval import regimes as R
from eval.frozen.simulator import Simulator, load_config


@pytest.fixture()
def base():
    return load_config()


def test_every_regime_produces_a_valid_config(base):
    for name in R.REGIMES:
        cfg = R.config_for(name, base)
        R.validate_config(cfg)


def test_overlay_does_not_mutate_the_frozen_config(base):
    before = copy.deepcopy(base)
    for name in R.REGIMES:
        R.config_for(name, base)
    assert base == before, "config_for() mutated the config it was handed"
    assert load_config() == before, "the frozen config on disk changed"


def test_unknown_overlay_key_is_refused(base):
    """A typo'd knob would otherwise be a silent no-op, and a regime that
    changes nothing produces a result that looks like evidence and is not."""
    with pytest.raises(KeyError, match="unknown config key"):
        R.apply_overlay(base, {"hazrads": {"CANT_PAY_NOW": {"base_recovery": 0.1}}})


def test_unknown_nested_overlay_key_is_refused(base):
    with pytest.raises(KeyError, match="unknown config key"):
        R.apply_overlay(base, {"hazards": {"CANT_PAY_NOW": {"base_recovry": 0.1}}})


def test_every_regime_actually_changes_something(base):
    """Every regime except `baseline` must differ from the frozen config.
    A regime identical to baseline would silently contribute a duplicate
    column to the report."""
    for name, spec in R.REGIMES.items():
        cfg = R.config_for(name, base)
        if name == "baseline":
            assert cfg == base
        else:
            assert cfg != base, f"regime {name!r} is a no-op"


def test_every_regime_carries_a_hypothesis_and_an_approximation():
    """Pre-registration is the point. A regime with no stated hypothesis
    cannot be wrong, and a regime with no stated approximation is claiming a
    fidelity the frozen simulator's knobs cannot deliver."""
    for name, spec in R.REGIMES.items():
        assert spec.story.strip(), f"{name}: no story"
        assert len(spec.hypothesis.strip()) > 40, f"{name}: hypothesis too thin"
        if name != "baseline":
            assert len(spec.approximation.strip()) > 40, (
                f"{name}: no approximation stated -- say where the knobs fall "
                f"short of the story, or explain why they do not"
            )


def test_validate_rejects_base_rates_that_leave_no_pending_mass(base):
    bad = copy.deepcopy(base)
    bad["hazards"]["CANT_PAY_NOW"]["base_recovery"] = 0.99
    bad["hazards"]["CANT_PAY_NOW"]["base_dead"] = 0.05
    with pytest.raises(ValueError, match="STILL_PENDING"):
        R.validate_config(bad)


def test_validate_rejects_cause_mix_that_does_not_sum_to_one(base):
    bad = copy.deepcopy(base)
    bad["cause_mix"]["WONT_PAY"] = 0.9
    with pytest.raises(ValueError, match="cause_mix"):
        R.validate_config(bad)


def test_stacking_spike_runs_only_under_the_coupled_arm():
    """Under nominal/misspecified its overlay touches nothing the simulator
    reads, so the column would duplicate baseline and invite being read as a
    result."""
    assert R.arms_for("stacking_spike", R_ALL) == ("coupled",)
    assert R.arms_for("baseline", R_ALL) == R_ALL


R_ALL = ("nominal", "misspecified", "coupled")


@pytest.mark.parametrize("name", sorted(R.REGIMES))
def test_regime_config_drives_the_frozen_simulator(name, base):
    """The overlay must be something the frozen Simulator actually accepts --
    the whole design depends on never editing it."""
    cfg = R.config_for(name, base)
    for arm in R.arms_for(name, R_ALL):
        sim = Simulator(arm, seed=1, config=cfg)
        assert len(sim.mandates) == cfg["n_mandates"]
        r = sim.attempt(sim.mandates[0].mandate_id, slot=2, on_day=1)
        assert r.slot == 2
