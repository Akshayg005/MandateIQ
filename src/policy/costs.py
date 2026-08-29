"""Non-RBI tuning parameters the allocator's Q-function and stopping rules
need: attempt cost, mandate LTV, re-auth cost/success rate, quiet hours,
contact-frequency cap.

Design spec: src/policy/constraints.py's own module docstring assigns their
placement to B8 and forbids them living there -- "they are tuning parameters
for a config file, not a regulatory constant." This module is the typed
loader for config/policy_costs.yaml, the file that placement points to.
Every field is sourced from that YAML, never hard-coded here, so changing a
cost is a data edit, not a code edit -- the same discipline DECISIONS.md
2026-08-29 (B7) required of belief.py's declined switch_eps parameter,
applied to this layer's own tuning surface.

All money is integer paise. reauth_success_prob is the one float field --
a probability, not a money value.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml

_DEFAULT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "policy_costs.yaml"
)

_REQUIRED_INT_FIELDS = (
    "attempt_cost_paise",
    "mandate_ltv_paise",
    "reauth_cost_paise",
    "quiet_hours_start",
    "quiet_hours_end",
    "max_contacts_per_cycle",
)
_MONEY_FIELDS = ("attempt_cost_paise", "mandate_ltv_paise", "reauth_cost_paise")


class CostsError(ValueError):
    """Raised by load() when policy_costs.yaml is missing a required key, a
    money field is not a non-negative int, a probability field is not in
    [0, 1], or an hour bound is outside 0..23. Never a bare assert -- assert
    is stripped under python -O, and a misconfigured cost silently
    mispricing every Q-value is exactly the kind of bug that must be loud."""


@dataclass(frozen=True)
class PolicyCosts:
    """attempt_cost_paise: flat operational cost charged against every
    ATTEMPT's expected value. mandate_ltv_paise: value treated as lost when
    a mandate opts out (6(c) is terminal) -- charged against Q(ATTEMPT) via
    h_opt, and what OFFER's construction treats as retained rather than
    lost. reauth_cost_paise / reauth_success_prob: price and completion
    rate of the re-authorisation path. quiet_hours_start / quiet_hours_end:
    local-time window (hour, 0-23) stopping_rules.py refuses to contact the
    customer in -- a consumer-protection norm, not an RBI clause.
    max_contacts_per_cycle: contact-frequency cap stopping_rules.py
    enforces."""

    attempt_cost_paise: int
    mandate_ltv_paise: int
    reauth_cost_paise: int
    reauth_success_prob: float
    quiet_hours_start: int
    quiet_hours_end: int
    max_contacts_per_cycle: int


def load(path: pathlib.Path | None = None) -> PolicyCosts:
    """Read and validate config/policy_costs.yaml (or `path`, for tests
    exercising a deliberately broken file). Raises CostsError on any
    missing key, a negative money field, a probability outside [0, 1], an
    hour bound outside 0..23, or a non-positive contact cap."""
    p = path or _DEFAULT_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CostsError(f"cannot read {p}: {exc}") from exc

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise CostsError(f"{p} did not parse to a mapping")

    missing = [k for k in _REQUIRED_INT_FIELDS + ("reauth_success_prob",) if k not in raw]
    if missing:
        raise CostsError(f"{p} is missing required key(s): {missing}")

    for key in _REQUIRED_INT_FIELDS:
        v = raw[key]
        if not isinstance(v, int) or isinstance(v, bool):
            raise CostsError(f"{key} must be an int, got {v!r}")

    for key in _MONEY_FIELDS:
        if raw[key] < 0:
            raise CostsError(f"{key} must be non-negative, got {raw[key]}")

    prob = raw["reauth_success_prob"]
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
        raise CostsError(f"reauth_success_prob must be in [0, 1], got {prob!r}")

    for key in ("quiet_hours_start", "quiet_hours_end"):
        if not (0 <= raw[key] <= 23):
            raise CostsError(f"{key} must be in 0..23, got {raw[key]}")

    if raw["max_contacts_per_cycle"] <= 0:
        raise CostsError(
            f"max_contacts_per_cycle must be positive, got {raw['max_contacts_per_cycle']}"
        )

    return PolicyCosts(
        attempt_cost_paise=raw["attempt_cost_paise"],
        mandate_ltv_paise=raw["mandate_ltv_paise"],
        reauth_cost_paise=raw["reauth_cost_paise"],
        reauth_success_prob=float(prob),
        quiet_hours_start=raw["quiet_hours_start"],
        quiet_hours_end=raw["quiet_hours_end"],
        max_contacts_per_cycle=raw["max_contacts_per_cycle"],
    )
