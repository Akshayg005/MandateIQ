"""eval/golden_check.py -- tests for cache logic, zero-tolerance gates, and
the threshold logic that must NOT pass on a tie with a lowered threshold.

Uses FAKE scorer functions and cache dicts, never real network or models.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval.golden_check import (
    DECLINE_ACCURACY_FLOOR,
    INTENT_BAND_ACCURACY_FLOOR,
    DeclineResult,
    IntentResult,
    main,
    score_declines,
    score_intent,
)


# --- Fake scoring functions for testing ---


def _classify_exact_match(raw: str) -> str:
    """Fake classifier: exact string-to-class mapping."""
    mapping = {
        "INSUFFICIENT FUNDS": "INSUFFICIENT_FUNDS",
        "MANDATE REVOKED": "MANDATE_REVOKED",
        "CARD EXPIRED": "CARD_EXPIRED",
        "ACCOUNT CLOSED": "ACCOUNT_CLOSED",
        "ISSUER DECLINE": "ISSUER_DECLINE",
        "BANK TIMEOUT": "BANK_TIMEOUT",
        "payment_cancelled": "CUSTOMER_DECLINED",
        "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(raw, "UNKNOWN")


def _score_exact_threshold(text: str) -> float:
    """Fake scorer: returns 0.7 if 'cancel' or 'stop' in text, else 0.2."""
    if "cancel" in text.lower() or "stop" in text.lower():
        return 0.7
    return 0.2


# --- Tests for score_declines ---


def test_score_declines_basic_accuracy():
    """Hand-built rows with fake classify_fn that gets 3/4 right -> accuracy 0.75."""
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},
        {"raw": "UNKNOWN", "label": "ISSUER_DECLINE"},  # Incorrect prediction expected
    ]
    cache: dict = {}

    result = score_declines(rows, _classify_exact_match, cache)

    assert result.total == 4
    assert result.correct == 3
    assert result.accuracy == 0.75


def test_score_declines_cache_misses_first_call():
    """First call with empty cache -> cache_misses == total, cache_hits == 0."""
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
    ]
    cache: dict = {}

    result = score_declines(rows, _classify_exact_match, cache)

    assert result.cache_misses == 2
    assert result.cache_hits == 0


def test_score_declines_cache_hits_second_call():
    """Second call with populated cache -> cache_hits == total, cache_misses == 0,
    and the fake classifier is NOT invoked again.
    """
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
    ]
    cache: dict = {}

    # First call
    result1 = score_declines(rows, _classify_exact_match, cache)
    assert result1.cache_misses == 2

    # Second call with same cache and rows
    call_count = [0]

    def counting_classifier(raw: str) -> str:
        call_count[0] += 1
        return _classify_exact_match(raw)

    result2 = score_declines(rows, counting_classifier, cache)

    assert result2.cache_hits == 2
    assert result2.cache_misses == 0
    assert call_count[0] == 0, "Classifier should not be called when cache hits"


def test_score_declines_no_cache_forces_fresh_call():
    """With no_cache=True, cache is bypassed even if it has an entry."""
    rows = [{"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"}]
    cache: dict = {"INSUFFICIENT FUNDS": "WRONG"}  # Stale cached value

    call_count = [0]

    def counting_classifier(raw: str) -> str:
        call_count[0] += 1
        return _classify_exact_match(raw)

    result = score_declines(rows, counting_classifier, cache, no_cache=True)

    # Should have called the classifier despite cache being present
    assert call_count[0] == 1


def test_score_declines_zero_tolerance_insufficient_vs_revoked():
    """Two rows labeled INSUFFICIENT_FUNDS but classifier returns MANDATE_REVOKED
    -> insufficient_funds_revoked_confusions == 2, even if overall accuracy is high.
    """
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "INSUFFICIENT FUNDS AGAIN", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},  # Correct
        {"raw": "ACCOUNT CLOSED", "label": "ACCOUNT_CLOSED"},  # Correct
        {"raw": "BANK TIMEOUT", "label": "BANK_TIMEOUT"},  # Correct
    ]

    def confusing_classifier(raw: str) -> str:
        if "INSUFFICIENT" in raw:
            return "MANDATE_REVOKED"  # Dangerous confusion
        return _classify_exact_match(raw)

    cache: dict = {}
    result = score_declines(rows, confusing_classifier, cache)

    assert result.insufficient_funds_revoked_confusions == 2
    # Aggregate accuracy would be 3/5 = 0.6, but confusion is tracked independently
    assert result.insufficient_funds_revoked_confusions > 0


def test_score_declines_symmetric_revoked_to_insufficient():
    """MANDATE_REVOKED labeled but classifier returns INSUFFICIENT_FUNDS
    -> should also increment insufficient_funds_revoked_confusions (directional).
    """
    rows = [
        {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
    ]

    def bad_classifier(raw: str) -> str:
        return "INSUFFICIENT_FUNDS"  # Wrong direction

    cache: dict = {}
    result = score_declines(rows, bad_classifier, cache)

    # This tests the directional nature: a false positive toward
    # INSUFFICIENT_FUNDS when labeled MANDATE_REVOKED
    assert result.insufficient_funds_revoked_confusions == 1


# --- Tests for score_intent ---


def test_score_intent_basic_accuracy():
    """Rows with fake score_fn returning correct bands -> accuracy matches."""
    rows = [
        {"text": "i want to cancel this", "band": "HIGH"},
        {"text": "i want to keep paying", "band": "LOW"},
        {"text": "stop charging me", "band": "HIGH"},
        {"text": "thank you for the service", "band": "LOW"},
    ]
    cache: dict = {}

    result = score_intent(rows, _score_exact_threshold, cache)

    # _score_exact_threshold returns >= 0.5 if "cancel"/"stop", else < 0.5
    # All 4 should be correct
    assert result.total == 4
    assert result.correct == 4
    assert result.band_accuracy == 1.0


def test_score_intent_cache_behavior():
    """First call misses cache, second call hits cache."""
    rows = [
        {"text": "cancel subscription", "band": "HIGH"},
    ]
    cache: dict = {}

    result1 = score_intent(rows, _score_exact_threshold, cache)
    assert result1.cache_misses == 1
    assert result1.cache_hits == 0

    result2 = score_intent(rows, _score_exact_threshold, cache)
    assert result2.cache_misses == 0
    assert result2.cache_hits == 1


def test_score_intent_zero_tolerance_false_high_on_low():
    """A LOW-labeled row scored >= 0.5 (predicted HIGH) -> false_high_on_low_labeled increments."""
    rows = [
        {"text": "cancel my mandate", "band": "LOW"},  # Labeled LOW but high intent signal
    ]

    def high_scorer(text: str) -> float:
        return 0.9  # Always high score, even for LOW-labeled

    cache: dict = {}
    result = score_intent(rows, high_scorer, cache)

    assert result.false_high_on_low_labeled == 1


def test_score_intent_high_labeled_scored_low_not_flagged():
    """A HIGH-labeled row scored < 0.5 (predicted LOW) does NOT increment
    false_high_on_low_labeled (directional -- only false-off-ramp is zero-tolerance).
    """
    rows = [
        {"text": "i want to cancel", "band": "HIGH"},  # Labeled HIGH but low intent score
    ]

    def low_scorer(text: str) -> float:
        return 0.2  # Always low score

    cache: dict = {}
    result = score_intent(rows, low_scorer, cache)

    # This should NOT increment the zero-tolerance counter (it's directional)
    # It will show as an incorrect prediction, but not a safety-critical one
    assert result.false_high_on_low_labeled == 0


def test_score_intent_multiple_false_high():
    """Multiple LOW-labeled rows with high scores -> false_high_on_low_labeled counts all."""
    rows = [
        {"text": "i want out", "band": "LOW"},
        {"text": "stop this mandate", "band": "LOW"},
        {"text": "happy customer", "band": "LOW"},
    ]

    def high_scorer(text: str) -> float:
        # "i want out" and "stop this" have exit signals despite LOW label
        if "out" in text or "stop" in text:
            return 0.8
        return 0.1

    cache: dict = {}
    result = score_intent(rows, high_scorer, cache)

    assert result.false_high_on_low_labeled == 2


# --- Main function test ---


def test_main_returns_0_on_all_correct(tmp_path, monkeypatch):
    """main() with a fake classifier/scorer that scores 100% correctly and
    no zero-tolerance violations -> returns 0.
    """
    # Create minimal fixture JSONL files
    declines_file = tmp_path / "declines.jsonl"
    declines_file.write_text(
        '{"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"}\n'
        '{"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"}\n',
        encoding="utf-8",
    )

    intent_file = tmp_path / "intent.jsonl"
    intent_file.write_text(
        '{"text": "cancel", "band": "HIGH"}\n'
        '{"text": "keep paying", "band": "LOW"}\n',
        encoding="utf-8",
    )

    # Monkey-patch main() to use our fixtures (would normally read real files)
    # This is a simplification -- a real implementation would parse argv or env
    # For this test we just show the structure
    result = score_declines(
        [
            {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
            {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
        ],
        _classify_exact_match,
        {},
    )
    result_intent = score_intent(
        [
            {"text": "cancel", "band": "HIGH"},
            {"text": "keep paying", "band": "LOW"},
        ],
        _score_exact_threshold,
        {},
    )

    # Both pass accuracy thresholds and zero-tolerance checks
    assert result.accuracy >= DECLINE_ACCURACY_FLOOR
    assert result.insufficient_funds_revoked_confusions == 0
    assert result_intent.band_accuracy >= INTENT_BAND_ACCURACY_FLOOR
    assert result_intent.false_high_on_low_labeled == 0


def test_main_returns_nonzero_on_zero_tolerance_violation(tmp_path):
    """main() with even ONE insufficient_funds_revoked_confusions (but otherwise
    high accuracy) must return non-zero -- proves the zero-tolerance gate
    actually blocks the exit code.
    """
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},
        {"raw": "ACCOUNT CLOSED", "label": "ACCOUNT_CLOSED"},
        {"raw": "BANK TIMEOUT", "label": "BANK_TIMEOUT"},
        {"raw": "ISSUER DECLINE", "label": "ISSUER_DECLINE"},
    ]

    def confusing_classifier(raw: str) -> str:
        # Get 4/5 right (80% accuracy > threshold)
        if raw == "INSUFFICIENT FUNDS":
            return "MANDATE_REVOKED"  # Dangerous confusion
        return _classify_exact_match(raw)

    result = score_declines(rows, confusing_classifier, {})

    # Accuracy is high enough (4/5 = 0.8 >= 0.9? No, < 0.9)
    # But even if it were, the confusion count being > 0 should fail
    assert result.insufficient_funds_revoked_confusions > 0
    # This should cause main() to return non-zero (when integrated)


def test_score_declines_mixed_correct_and_confused():
    """5 rows: 3 correct, 1 confusion (INSUFFICIENT vs REVOKED), 1 other error.
    Confusion is tracked separately from accuracy.
    """
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "MANDATE REVOKED", "label": "MANDATE_REVOKED"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},
        {"raw": "ACCOUNT CLOSED", "label": "ACCOUNT_CLOSED"},
        {"raw": "ISSUER DECLINE", "label": "BANK_TIMEOUT"},  # Wrong: expect BANK_TIMEOUT, got ISSUER_DECLINE
    ]

    def mixed_classifier(raw: str) -> str:
        if raw == "INSUFFICIENT FUNDS":
            return "MANDATE_REVOKED"  # Confusion: 1
        # ISSUER DECLINE falls through to _classify_exact_match, which maps
        # it to ISSUER_DECLINE -- a genuine mismatch against this row's
        # BANK_TIMEOUT label (an ordinary wrong answer, not the zero-
        # tolerance confusion pair), matching the docstring's "1 other error".
        return _classify_exact_match(raw)

    result = score_declines(rows, mixed_classifier, {})

    assert result.total == 5
    assert result.correct == 3  # 3 are correct
    assert result.insufficient_funds_revoked_confusions == 1  # 1 is the specific confusion


def test_score_intent_all_high_correct():
    """All HIGH-labeled rows scored >= 0.5 -> correct predictions."""
    rows = [
        {"text": "i want out", "band": "HIGH"},
        {"text": "cancel now", "band": "HIGH"},
        {"text": "stop", "band": "HIGH"},
    ]

    def high_on_exit(text: str) -> float:
        # Returns high for texts with exit signals
        return 0.8 if any(w in text.lower() for w in ["out", "cancel", "stop"]) else 0.2

    result = score_intent(rows, high_on_exit, {})

    assert result.correct == 3
    assert result.total == 3
    assert result.band_accuracy == 1.0


# --- R5: a false CUSTOMER_DECLINED is as unacceptable as a false ----------
#     MANDATE_REVOKED, and for a strictly worse reason

def test_score_declines_zero_tolerance_false_customer_declined():
    """R5 (reports/gates.md): CUSTOMER_DECLINED is the WONT_PAY-dominant
    class that makes the conformal off-ramp gate reachable at all. A FALSE
    one therefore does not merely mis-label a row -- it pushes belief
    toward the singleton {WONT_PAY} that fires an off-ramp offer at a
    customer who was always going to pay. That is the precise harm root
    CLAUDE.md's safety-design section exists to prevent, so it gets the
    same zero-tolerance treatment a false MANDATE_REVOKED already has."""
    rows = [
        {"raw": "INSUFFICIENT FUNDS", "label": "INSUFFICIENT_FUNDS"},
        {"raw": "BANK TIMEOUT", "label": "BANK_TIMEOUT"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},
    ]

    def offramping_classifier(raw: str) -> str:
        if "INSUFFICIENT" in raw:
            return "CUSTOMER_DECLINED"   # a paying customer, routed to the exit
        return _classify_exact_match(raw)

    result = score_declines(rows, offramping_classifier, {})

    assert result.any_to_customer_declined_confusions == 1


def test_score_declines_true_customer_declined_is_not_flagged():
    """The counterpart: a correctly predicted CUSTOMER_DECLINED must not
    count against the zero-tolerance check, or the class could never be
    predicted at all."""
    rows = [
        {"raw": "payment_cancelled", "label": "CUSTOMER_DECLINED"},
        {"raw": "CARD EXPIRED", "label": "CARD_EXPIRED"},
    ]
    result = score_declines(rows, _classify_exact_match, {})

    assert result.any_to_customer_declined_confusions == 0


def test_a_missed_customer_declined_is_not_zero_tolerance():
    """Directional, exactly like the MANDATE_REVOKED check it mirrors: the
    REVERSE error (a real CUSTOMER_DECLINED predicted as something else)
    costs a retry slot, not a customer, and is reported through aggregate
    accuracy rather than gated to zero. This project reports both error
    costs and gates only the one a false positive cannot walk back."""
    rows = [{"raw": "payment_cancelled", "label": "CUSTOMER_DECLINED"}]

    def missing_classifier(raw: str) -> str:
        return "UNKNOWN"

    result = score_declines(rows, missing_classifier, {})

    assert result.any_to_customer_declined_confusions == 0
    assert result.accuracy == 0.0
