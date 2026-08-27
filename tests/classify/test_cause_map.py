"""src/classify/cause_map.py -- DeclineClass to Cause prior probability
distribution. Maps observed decline patterns to the probability of each
latent cause: CANT_PAY_NOW, CANT_PAY_EVER, WONT_PAY.

Design choices:
- Returns a full dict over all three Cause members for every DeclineClass,
  never a partial dict. Even if a cause has near-zero mass, it must be
  present as a key.
- Probabilities are Python float, not Decimal or numpy types. This is the
  single place in the codebase floats ARE allowed (they are not money).
- UNKNOWN class must return exactly uniform (1/3 each) because the decline
  is unclassifiable, not a rounding accident. This guarantees the result
  of an uninformed prior lookup is deterministic, not dependent on any
  statistical fit.
- INSUFFICIENT_FUNDS must map dominantly to CANT_PAY_NOW (transient gap).
- MANDATE_REVOKED, CARD_EXPIRED, ACCOUNT_CLOSED must map dominantly to
  CANT_PAY_EVER (dead instrument).
- The separation between CANT_PAY_NOW and CANT_PAY_EVER is the central
  thesis of the recovery engine: never conflate them.
"""
from __future__ import annotations

import pytest

from src.core.types import Cause, DeclineClass


# --- valid distribution tests -----------------------------------------------

@pytest.mark.parametrize("dc", list(DeclineClass))
def test_prior_returns_normalized_distribution(dc):
    """For every DeclineClass, the sum of probabilities must equal 1.0."""
    from src.classify.cause_map import prior

    result = prior(dc)
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("dc", list(DeclineClass))
def test_prior_contains_all_three_causes(dc):
    """The returned dict must have all three Cause members as keys for every
    DeclineClass, even if some have near-zero mass."""
    from src.classify.cause_map import prior

    result = prior(dc)

    for cause in Cause:
        assert cause in result, f"Missing {cause} in prior({dc})"


@pytest.mark.parametrize("dc", list(DeclineClass))
def test_prior_values_are_float_not_other_types(dc):
    """All values must be Python float, never numpy types or Decimal.
    Floats are allowed for probabilities (the single exception in the
    codebase); this test makes that boundary explicit."""
    from src.classify.cause_map import prior

    result = prior(dc)

    for cause, prob in result.items():
        assert isinstance(prob, float), f"prior({dc})[{cause}] = {prob!r}, not float"


# --- semantic correctness ---------------------------------------------------

def test_unknown_skews_cant_pay_now_not_uniform():
    """DeclineClass.UNKNOWN is skewed toward CANT_PAY_NOW, not exactly
    uniform -- per .claude/skills/new-failure-class/SKILL.md's explicit
    policy for a genuinely ambiguous class: map it to the safe default,
    because the three actions this feeds have asymmetric downside
    (CANT_PAY_NOW costs a retry slot; WONT_PAY risks an off-ramp offer).
    Corrected from an earlier exactly-uniform version by payments-domain's
    B3 review -- see cause_map.py's module docstring."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.UNKNOWN)

    assert result[Cause.CANT_PAY_NOW] > result[Cause.CANT_PAY_EVER]
    assert result[Cause.CANT_PAY_NOW] > result[Cause.WONT_PAY]
    # Still an honest abstention, not overconfidence: CANT_PAY_NOW must not
    # claim a supermajority the way a real, specific decline class would.
    assert result[Cause.CANT_PAY_NOW] < 0.75


def test_issuer_decline_skews_cant_pay_now_not_uniform():
    """Same safe-default policy applies to ISSUER_DECLINE -- a generic
    catch-all bin is exactly the "genuinely ambiguous" case the skill's
    policy is about, and it must not carry equal WONT_PAY mass to a
    confident WONT_PAY-dominant class like MANDATE_REVOKED."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.ISSUER_DECLINE)

    assert result[Cause.CANT_PAY_NOW] > result[Cause.CANT_PAY_EVER]
    assert result[Cause.CANT_PAY_NOW] > result[Cause.WONT_PAY]


def test_mandate_revoked_dominant_cause_is_not_cant_pay_now():
    """MANDATE_REVOKED (a dead/revoked instrument) must have strictly lower
    probability for CANT_PAY_NOW than for both CANT_PAY_EVER and WONT_PAY.
    This enforces the critical invariant: never conflate a transient liquidity
    gap (CANT_PAY_NOW) with a dead mandate (CANT_PAY_EVER)."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.MANDATE_REVOKED)
    cant_pay_now = result[Cause.CANT_PAY_NOW]
    cant_pay_ever = result[Cause.CANT_PAY_EVER]
    wont_pay = result[Cause.WONT_PAY]

    assert cant_pay_now < cant_pay_ever
    assert cant_pay_now < wont_pay


def test_insufficient_funds_dominant_cause_is_cant_pay_now():
    """INSUFFICIENT_FUNDS (transient liquidity gap) must have strictly higher
    probability for CANT_PAY_NOW than for the other two causes."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.INSUFFICIENT_FUNDS)
    cant_pay_now = result[Cause.CANT_PAY_NOW]
    cant_pay_ever = result[Cause.CANT_PAY_EVER]
    wont_pay = result[Cause.WONT_PAY]

    assert cant_pay_now > cant_pay_ever
    assert cant_pay_now > wont_pay


def test_card_expired_dominant_cause_is_cant_pay_ever():
    """CARD_EXPIRED (dead instrument) must have strictly higher probability
    for CANT_PAY_EVER than for the other two causes."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.CARD_EXPIRED)
    cant_pay_ever = result[Cause.CANT_PAY_EVER]
    cant_pay_now = result[Cause.CANT_PAY_NOW]
    wont_pay = result[Cause.WONT_PAY]

    assert cant_pay_ever > cant_pay_now
    assert cant_pay_ever > wont_pay


def test_account_closed_dominant_cause_is_cant_pay_ever():
    """ACCOUNT_CLOSED (dead instrument) must have strictly higher probability
    for CANT_PAY_EVER than for the other two causes."""
    from src.classify.cause_map import prior

    result = prior(DeclineClass.ACCOUNT_CLOSED)
    cant_pay_ever = result[Cause.CANT_PAY_EVER]
    cant_pay_now = result[Cause.CANT_PAY_NOW]
    wont_pay = result[Cause.WONT_PAY]

    assert cant_pay_ever > cant_pay_now
    assert cant_pay_ever > wont_pay
