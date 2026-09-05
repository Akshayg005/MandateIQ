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


# --- R5: the WONT_PAY-dominant row -----------------------------------------

def test_customer_declined_is_wont_pay_dominant():
    """R5 (reports/gates.md): before this row existed no DeclineClass had a
    WONT_PAY prior above 0.45, and MANDATE_REVOKED tied WONT_PAY with
    CANT_PAY_EVER at 0.45/0.45 -- a tie is absorbing under Bayes, so belief
    could never reach the {WONT_PAY} singleton ConformalCauseGate fires on.
    This is the row that makes the off-ramp lane REACHABLE at all."""
    from src.classify.cause_map import prior

    p = prior(DeclineClass.CUSTOMER_DECLINED)
    assert p[Cause.WONT_PAY] > p[Cause.CANT_PAY_NOW]
    assert p[Cause.WONT_PAY] > p[Cause.CANT_PAY_EVER]
    assert p[Cause.WONT_PAY] > 0.45


def test_customer_declined_is_not_near_degenerate():
    """Dominant, but deliberately not near-degenerate: ONE observation must
    not be able to slam belief into a singleton by itself. The off-ramp is
    the one action a false positive cannot walk back."""
    from src.classify.cause_map import prior

    p = prior(DeclineClass.CUSTOMER_DECLINED)
    assert p[Cause.WONT_PAY] < 0.90
    assert all(v > 0.0 for v in p.values())


def test_customer_declined_and_mandate_revoked_stay_distinct():
    """Two different events -- one dismissed collect request vs a revoked
    mandate -- must not carry the same cause distribution, or the taxonomy
    split buys nothing downstream."""
    from src.classify.cause_map import prior

    assert prior(DeclineClass.CUSTOMER_DECLINED) != prior(DeclineClass.MANDATE_REVOKED)


def test_prior_version_records_the_new_row():
    """PRIOR_VERSION is stamped per-row in ingested_event.prior_version; a
    changed table under an unchanged version makes a persisted belief
    untraceable."""
    from src.classify.cause_map import PRIOR_VERSION

    assert PRIOR_VERSION == "v3"


def test_a_raw_decline_string_maps_to_the_right_cause():
    """`.claude/skills/new-failure-class` checklist item 6: "a test
    asserting the RAW STRING maps to the right cause" -- end to end through
    classify() and prior(), not just through the class in isolation, so a
    taxonomy rule and a prior row that disagree cannot both pass their own
    tests while the pair is broken."""
    from src.classify.cause_map import prior
    from src.classify.decline_taxonomy import classify

    raw = "payment_cancelled: customer cancelled the UPI AutoPay mandate approval request"
    dominant = max(prior(classify(raw, None)).items(), key=lambda kv: kv[1])[0]
    assert dominant == Cause.WONT_PAY

    # And the two neighbours it must never be confused with, same route.
    revoked = "the customer has cancelled the mandate at their bank"
    assert max(prior(classify(revoked, None)).items(),
               key=lambda kv: kv[1])[0] == Cause.CANT_PAY_EVER
    funds = "insufficient_funds: the account did not have enough funds"
    assert max(prior(classify(funds, None)).items(),
               key=lambda kv: kv[1])[0] == Cause.CANT_PAY_NOW


def test_the_model_design_matrix_does_not_encode_declineclass_at_all():
    """`.claude/skills/new-failure-class` checklist item 4 asks that
    person_period one-hot encode the new class and that an unseen class map
    to an explicit "unknown" bucket rather than silently becoming
    all-zeros. Checked and NOT APPLICABLE here, pinned rather than assumed:
    FEATURE_COLUMNS is ("const", "slot_3", "slot_4", "in_salary_window") --
    the fitted hazard model never sees a DeclineClass in any form, so
    adding an 8th class cannot create an all-zeros row. If that ever
    changes, this test fails and item 4 becomes live work."""
    from src.model.competing_risks import FEATURE_COLUMNS, WIDENED_FEATURE_COLUMNS

    for col in tuple(FEATURE_COLUMNS) + tuple(WIDENED_FEATURE_COLUMNS):
        assert "decline" not in col.lower()
        assert not any(dc.value.lower() in col.lower() for dc in DeclineClass)
