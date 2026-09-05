"""
src/core/ids.py -- deterministic identifier derivation only. Must not
depend on wall-clock time, process identity, or randomness in any form
(DESIGN.md invariant 4 / the status notes's B1 gate names this file explicitly).

test_ids_module_imports_no_time_uuid_os_random reads the module's SOURCE
TEXT rather than importing it and inspecting sys.modules, so it still means
something once the module exists and some unrelated import elsewhere in the
process has already pulled in `time` or `random` for other reasons.
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

from src.core.ids import decision_sha256, idempotency_key, row_id

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
IDS_SRC = ROOT / "src" / "core" / "ids.py"


def _expected_key(mandate_id, cycle_id, attempt_index, generation, action, amount_paise):
    raw = (
        "mr:v1" + "|" + mandate_id + "|" + str(cycle_id) + "|" + str(attempt_index)
        + "|" + str(generation) + "|" + action + "|" + str(amount_paise)
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


BASE_ARGS = dict(
    mandate_id="mandate_abc123",
    cycle_id=7,
    attempt_index=2,
    generation=0,
    action="ATTEMPT",
    amount_paise=150000,
)


# --- the derivation is pinned, not just "returns a string" -----------------

def test_idempotency_key_matches_hand_computed_sha256():
    expected = _expected_key(**BASE_ARGS)
    assert idempotency_key(**BASE_ARGS) == expected


def test_idempotency_key_is_32_lowercase_hex_chars():
    key = idempotency_key(**BASE_ARGS)
    assert isinstance(key, str)
    assert re.fullmatch(r"[0-9a-f]{32}", key)


def test_idempotency_key_is_deterministic_across_repeated_calls():
    """No time/uuid/random is involved, so this is trivially true today --
    the point is to catch a future regression where someone salts the key."""
    keys = {idempotency_key(**BASE_ARGS) for _ in range(5)}
    assert len(keys) == 1


# --- sensitivity: every argument must actually participate in the hash -----

@pytest.mark.parametrize("field, other_value", [
    ("mandate_id", "mandate_zzz999"),
    ("cycle_id", 8),
    ("attempt_index", 3),
    ("generation", 1),
    ("action", "OFFER"),
    ("amount_paise", 150001),
])
def test_idempotency_key_changes_when_exactly_one_field_changes(field, other_value):
    changed = dict(BASE_ARGS)
    changed[field] = other_value
    assert idempotency_key(**BASE_ARGS) != idempotency_key(**changed)


def test_idempotency_key_generation_bump_prevents_reissue_collision():
    """The specific bug this exists to prevent: a voided-and-reissued
    attempt must NOT collide with the original attempt's key, or the
    ledger's ledger_intent_once unique constraint blocks the reissue."""
    original = idempotency_key(**{**BASE_ARGS, "generation": 0})
    reissued = idempotency_key(**{**BASE_ARGS, "generation": 1})
    assert original != reissued


def test_idempotency_key_amount_change_alone_changes_key():
    """A repriced attempt must not silently reuse a key."""
    a = idempotency_key(**{**BASE_ARGS, "amount_paise": 100000})
    b = idempotency_key(**{**BASE_ARGS, "amount_paise": 100001})
    assert a != b


def test_idempotency_key_numeric_type_does_not_affect_the_key():
    """500 and 500.0 are the same amount. If two code paths agree
    numerically but differ in type (an int here, a numpy/float value
    there), they must still derive the SAME key -- otherwise the same
    attempt collides with itself under a different key and
    ledger_intent_once never catches the duplicate."""
    as_int = idempotency_key(**{**BASE_ARGS, "amount_paise": 500})
    as_float = idempotency_key(**{**BASE_ARGS, "amount_paise": 500.0})
    assert as_int == as_float


def test_idempotency_key_rejects_pipe_in_mandate_id():
    with pytest.raises(ValueError):
        idempotency_key(**{**BASE_ARGS, "mandate_id": "mandate|evil"})


# --- row_id ------------------------------------------------------------------

def test_row_id_exact_format():
    assert row_id("mandate_abc123", 7, 2) == "mandate_abc123:7:2"
    assert row_id("M-1", 0, 0) == "M-1:0:0"


# --- decision_sha256: canonical, order-independent --------------------------

def test_decision_sha256_is_order_independent():
    a = {"mandate_id": "M1", "cycle_id": 1, "belief": {"x": 1, "y": 2}}
    b = {"cycle_id": 1, "belief": {"y": 2, "x": 1}, "mandate_id": "M1"}
    assert decision_sha256(a) == decision_sha256(b)


def test_decision_sha256_changes_when_a_value_changes():
    a = {"mandate_id": "M1", "cycle_id": 1}
    b = {"mandate_id": "M1", "cycle_id": 2}
    assert decision_sha256(a) != decision_sha256(b)


def test_decision_sha256_returns_full_hex_digest_string():
    result = decision_sha256({"a": 1})
    assert isinstance(result, str)
    assert re.fullmatch(r"[0-9a-f]{64}", result)


# --- source-level guard: no time/uuid/os/random import, in either style ----

def test_ids_module_imports_no_time_uuid_os_random():
    text = IDS_SRC.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"^\s*import\s+time\b",
        r"^\s*import\s+uuid\b",
        r"^\s*import\s+os\b",
        r"^\s*import\s+random\b",
        r"^\s*from\s+time\s+import\b",
        r"^\s*from\s+uuid\s+import\b",
        r"^\s*from\s+os\s+import\b",
        r"^\s*from\s+random\s+import\b",
    ]
    for pattern in forbidden_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        assert match is None, f"forbidden import found in ids.py: {match.group(0)!r}"
