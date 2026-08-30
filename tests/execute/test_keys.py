"""src/execute/keys.py -- the idempotency-key adapter for this layer.

test_keys_module_imports_no_time_uuid_os_random mirrors
tests/core/test_ids.py's own source-level guard, scoped to this file: it
reads SOURCE TEXT rather than importing and inspecting sys.modules, so it
still means something once some unrelated import elsewhere in the process
has already pulled in `time` or `random` for other reasons.
"""
from __future__ import annotations

import pathlib
import re

from src.core.ids import idempotency_key
from src.execute.keys import ScheduledAttempt, key_for

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
KEYS_SRC = ROOT / "src" / "execute" / "keys.py"

BASE_ARGS = dict(
    mandate_id="mandate_abc123",
    cycle_id=1,
    attempt_index=1,
    generation=0,
    action="ATTEMPT",
    amount_paise=50_000,
)


# --- source guard: no clock, no uuid, no pid, no randomness ----------------

def test_keys_module_imports_no_time_uuid_os_random():
    """Mirrors tests/core/test_ids.py's own guard exactly: line-anchored
    import patterns, not a bare substring search -- this file's own
    docstring legitimately contains the words "import time" in prose."""
    text = KEYS_SRC.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"^\s*import\s+time\b",
        r"^\s*import\s+uuid\b",
        r"^\s*import\s+os\b",
        r"^\s*import\s+random\b",
        r"^\s*from\s+time\s+import\b",
        r"^\s*from\s+uuid\s+import\b",
        r"^\s*from\s+os\s+import\b",
        r"^\s*from\s+random\s+import\b",
        r"^\s*from\s+src\.core\.clock\s+import\b",
        r"^\s*import\s+src\.core\.clock\b",
    ]
    for pattern in forbidden_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        assert match is None, f"forbidden import found in keys.py: {match.group(0)!r}"


# --- delegates exactly to src.core.ids.idempotency_key ----------------------

def test_key_for_matches_core_ids_idempotency_key():
    expected = idempotency_key(**BASE_ARGS)
    assert key_for(**BASE_ARGS) == expected


def test_key_for_is_deterministic_across_calls():
    assert key_for(**BASE_ARGS) == key_for(**BASE_ARGS)


# --- the fields that must change the key, per src/core/ids.py's own contract

def test_key_for_changes_with_generation():
    """The whole point of `generation`: a void-and-reissue must derive a
    distinct key from the attempt it replaces, or it collides on
    ledger_intent_once."""
    base = key_for(**BASE_ARGS)
    reissued = key_for(**{**BASE_ARGS, "generation": 1})
    assert base != reissued


def test_key_for_changes_with_amount_paise():
    """A repriced attempt must never silently reuse a key."""
    base = key_for(**BASE_ARGS)
    repriced = key_for(**{**BASE_ARGS, "amount_paise": 60_000})
    assert base != repriced


def test_key_for_changes_with_attempt_index():
    base = key_for(**BASE_ARGS)
    next_slot = key_for(**{**BASE_ARGS, "attempt_index": 2})
    assert base != next_slot


def test_key_for_changes_with_action():
    """ATTEMPT and REAUTH against the same slot must not collide."""
    base = key_for(**BASE_ARGS)
    reauth = key_for(**{**BASE_ARGS, "action": "REAUTH"})
    assert base != reauth


def test_key_for_changes_with_mandate_id():
    base = key_for(**BASE_ARGS)
    other_mandate = key_for(**{**BASE_ARGS, "mandate_id": "mandate_xyz789"})
    assert base != other_mandate


def test_key_for_changes_with_cycle_id():
    base = key_for(**BASE_ARGS)
    next_cycle = key_for(**{**BASE_ARGS, "cycle_id": 2})
    assert base != next_cycle


# --- ScheduledAttempt is a plain, hashable-by-value row shape ---------------

def test_scheduled_attempt_is_frozen():
    from datetime import datetime, timezone

    attempt = ScheduledAttempt(
        idempotency_key=key_for(**BASE_ARGS),
        mandate_id=BASE_ARGS["mandate_id"],
        cycle_id=BASE_ARGS["cycle_id"],
        attempt_index=BASE_ARGS["attempt_index"],
        generation=BASE_ARGS["generation"],
        action=BASE_ARGS["action"],
        amount_paise=BASE_ARGS["amount_paise"],
        profile="strict",
        decision_sha256="a" * 64,
        scheduled_for=datetime(2026, 1, 2, tzinfo=timezone.utc),
        committed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    try:
        attempt.amount_paise = 1  # type: ignore[misc]
        assert False, "ScheduledAttempt must be frozen"
    except AttributeError:
        pass
    assert attempt.voided_at is None
    assert attempt.void_reason is None
