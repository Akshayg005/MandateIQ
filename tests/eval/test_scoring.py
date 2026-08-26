"""eval/frozen/scoring.py -- pure aggregation, no randomness."""
from __future__ import annotations

import pytest

from src.core.types import Cause, Outcome
from eval.frozen.scoring import aggregate, score_mandate
from eval.frozen.simulator import AttemptResult, SimMandate

MANDATE = SimMandate(
    mandate_id="M0000", cycle_id=1, amount_paise=100000, ceiling_paise=100000,
    category="subscription", household_id=None, initial_cause=Cause.CANT_PAY_NOW,
)


def _attempt(slot, outcome, iatrogenic=False):
    return AttemptResult(mandate_id=MANDATE.mandate_id, slot=slot, on_day=slot - 1,
                          outcome=outcome, iatrogenic_insufficient_funds=iatrogenic)


def test_score_mandate_recovered():
    result = score_mandate(MANDATE, [_attempt(2, Outcome.STILL_PENDING), _attempt(3, Outcome.RECOVERED)])
    assert result.final_outcome == Outcome.RECOVERED
    assert result.amount_recovered_paise == MANDATE.amount_paise
    assert result.preserved is True


def test_score_mandate_dead():
    result = score_mandate(MANDATE, [_attempt(2, Outcome.DEAD)])
    assert result.final_outcome == Outcome.DEAD
    assert result.amount_recovered_paise == 0
    assert result.preserved is False


def test_score_mandate_opted_out():
    result = score_mandate(MANDATE, [_attempt(2, Outcome.OPTED_OUT)])
    assert result.final_outcome == Outcome.OPTED_OUT
    assert result.preserved is False


def test_score_mandate_censored_when_budget_exhausted():
    attempts = [_attempt(s, Outcome.STILL_PENDING) for s in (2, 3, 4)]
    result = score_mandate(MANDATE, attempts)
    assert result.final_outcome == Outcome.STILL_PENDING
    assert result.amount_recovered_paise == 0
    assert result.preserved is True  # censored, not lost


def test_score_mandate_raises_on_empty_attempts():
    with pytest.raises(ValueError):
        score_mandate(MANDATE, [])


def test_score_mandate_raises_if_attempt_follows_a_terminal_outcome():
    malformed = [_attempt(2, Outcome.DEAD), _attempt(3, Outcome.STILL_PENDING)]
    with pytest.raises(ValueError):
        score_mandate(MANDATE, malformed)


def test_score_mandate_counts_iatrogenic_failures():
    attempts = [
        _attempt(2, Outcome.STILL_PENDING, iatrogenic=True),
        _attempt(3, Outcome.STILL_PENDING, iatrogenic=True),
        _attempt(4, Outcome.RECOVERED),
    ]
    result = score_mandate(MANDATE, attempts)
    assert result.iatrogenic_failures == 2


def test_aggregate_sums_across_mandates():
    m2 = SimMandate(mandate_id="M0001", cycle_id=1, amount_paise=200000, ceiling_paise=200000,
                     category="subscription", household_id=None, initial_cause=Cause.WONT_PAY)
    r1 = score_mandate(MANDATE, [_attempt(2, Outcome.RECOVERED)])
    a2 = AttemptResult(mandate_id=m2.mandate_id, slot=2, on_day=1, outcome=Outcome.OPTED_OUT)
    r2 = score_mandate(m2, [a2])

    batch = aggregate([r1, r2], arm="nominal", profile="strict")

    assert batch.n_mandates == 2
    assert batch.total_recovered_paise == 100000
    assert batch.total_attempts_spent == 2
    assert batch.mandates_recovered == 1
    assert batch.mandates_opted_out == 1
    assert batch.mandates_preserved == 1  # only the recovered one


def test_aggregate_raises_on_empty_results():
    with pytest.raises(ValueError):
        aggregate([], arm="nominal", profile="strict")


def test_aggregate_preserved_excludes_only_dead_and_opted_out():
    censored = score_mandate(MANDATE, [_attempt(s, Outcome.STILL_PENDING) for s in (2, 3, 4)])
    batch = aggregate([censored], arm="nominal", profile="strict")
    assert batch.mandates_preserved == 1
    assert batch.mandates_censored == 1


def test_batch_result_summary_returns_a_string_mentioning_the_arm():
    r = score_mandate(MANDATE, [_attempt(2, Outcome.RECOVERED)])
    batch = aggregate([r], arm="nominal", profile="strict")
    text = batch.summary()
    assert isinstance(text, str)
    assert "nominal" in text
