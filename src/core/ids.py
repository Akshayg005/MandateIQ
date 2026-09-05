"""Deterministic identifier derivation only. Must never depend on wall-clock
time, process identity, or randomness -- the same attempt must derive the
same key on any machine, in any process, after any number of crashes. See
DESIGN.md invariant 3 and the build spec §3.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping


def idempotency_key(
    mandate_id: str,
    cycle_id: int,
    attempt_index: int,
    generation: int,
    action: str,
    amount_paise: int,
) -> str:
    """Stable key for one committed attempt. `generation` is what lets a
    void-and-reissue derive a different key from the original attempt it
    replaces, without colliding on ledger_intent_once; `amount_paise` is
    what stops a repriced attempt from silently reusing a key.

    Numeric fields are coerced with int() before stringifying so that two
    numerically-equal values of different type (e.g. 500 and 500.0) always
    derive the same key -- the same attempt must derive the same key
    regardless of which code path produced its numbers.
    """
    if "|" in mandate_id or "|" in action:
        raise ValueError("mandate_id/action must not contain the '|' delimiter")
    raw = (
        "mr:v1"
        + "|" + mandate_id
        + "|" + str(int(cycle_id))
        + "|" + str(int(attempt_index))
        + "|" + str(int(generation))
        + "|" + action
        + "|" + str(int(amount_paise))
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def row_id(mandate_id: str, cycle_id: int, slot: int) -> str:
    """Unique person-period row identifier."""
    return f"{mandate_id}:{cycle_id}:{slot}"


def decision_sha256(payload: Mapping) -> str:
    """Canonical, order-independent hash of a Plan payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
