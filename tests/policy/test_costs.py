"""src/policy/costs.py -- typed loader for config/policy_costs.yaml.

Design spec: every non-RBI tuning constant the allocator needs (attempt
cost, mandate LTV, re-auth cost/success rate, quiet hours, contact cap)
lives in the YAML, never hard-coded in costs.py itself. load() validates:
every required key present, money fields are non-negative ints, the
probability field is in [0, 1], hour bounds are in 0..23, and the contact
cap is positive.
"""
from __future__ import annotations

import pathlib

import pytest


def _write_yaml(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    p = tmp_path / "policy_costs.yaml"
    p.write_text(text, encoding="utf-8")
    return p


_VALID = """
attempt_cost_paise: 50
mandate_ltv_paise: 180000
reauth_cost_paise: 200
reauth_success_prob: 0.35
quiet_hours_start: 21
quiet_hours_end: 8
max_contacts_per_cycle: 4
"""


# === the real shipped file ==================================================

def test_default_config_loads_and_validates():
    """config/policy_costs.yaml, as actually shipped, must load without
    error and produce a fully-populated PolicyCosts."""
    from src.policy.costs import load, PolicyCosts

    costs = load()
    assert isinstance(costs, PolicyCosts)
    assert isinstance(costs.attempt_cost_paise, int)
    assert isinstance(costs.mandate_ltv_paise, int)
    assert isinstance(costs.reauth_cost_paise, int)
    assert isinstance(costs.reauth_success_prob, float)


def test_money_fields_are_int_never_float():
    """Invariant 2: all money is integer paise. Every *_paise field on the
    loaded PolicyCosts must be a plain int."""
    from src.policy.costs import load

    costs = load()
    for field in ("attempt_cost_paise", "mandate_ltv_paise", "reauth_cost_paise"):
        v = getattr(costs, field)
        assert isinstance(v, int) and not isinstance(v, bool), \
            f"{field} is {type(v).__name__}, not int"


# === load() validation ======================================================

def test_load_a_valid_file(tmp_path):
    from src.policy.costs import load

    costs = load(_write_yaml(tmp_path, _VALID))
    assert costs.attempt_cost_paise == 50
    assert costs.mandate_ltv_paise == 180000
    assert costs.reauth_cost_paise == 200
    assert costs.reauth_success_prob == pytest.approx(0.35)
    assert costs.quiet_hours_start == 21
    assert costs.quiet_hours_end == 8
    assert costs.max_contacts_per_cycle == 4


def test_missing_key_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("attempt_cost_paise: 50\n", "")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_negative_money_field_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("attempt_cost_paise: 50", "attempt_cost_paise: -1")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_float_money_field_raises(tmp_path):
    """A money field written as a float in the YAML must be rejected, not
    silently accepted -- this is exactly the class of bug invariant 2
    exists to catch, one layer up from the guard's static regex."""
    from src.policy.costs import load, CostsError

    text = _VALID.replace("attempt_cost_paise: 50", "attempt_cost_paise: 50.0")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_probability_outside_unit_interval_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("reauth_success_prob: 0.35", "reauth_success_prob: 1.5")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_negative_probability_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("reauth_success_prob: 0.35", "reauth_success_prob: -0.1")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_hour_bound_outside_range_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("quiet_hours_start: 21", "quiet_hours_start: 24")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_zero_contact_cap_raises(tmp_path):
    from src.policy.costs import load, CostsError

    text = _VALID.replace("max_contacts_per_cycle: 4", "max_contacts_per_cycle: 0")
    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, text))


def test_missing_file_raises(tmp_path):
    from src.policy.costs import load, CostsError

    with pytest.raises(CostsError):
        load(tmp_path / "does_not_exist.yaml")


def test_non_mapping_yaml_raises(tmp_path):
    from src.policy.costs import load, CostsError

    with pytest.raises(CostsError):
        load(_write_yaml(tmp_path, "- just\n- a\n- list\n"))


def test_costs_is_frozen(tmp_path):
    from dataclasses import FrozenInstanceError

    from src.policy.costs import load

    costs = load(_write_yaml(tmp_path, _VALID))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        costs.attempt_cost_paise = 999  # type: ignore
