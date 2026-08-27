"""src/classify/decline_taxonomy.py -- Razorpay error-reason strings normalized
into a 7-class taxonomy (DeclineClass) for downstream cause inference.

Design choices:
- Matching is case-insensitive substring matching over both code and text.
- First-match-wins: the order of checks in the implementation determines
  priority when multiple patterns match. Tests assert the ground truth from
  Razorpay's official docs, not assume an implementation strategy.
- None values are handled explicitly: a None code with a None text defaults
  to UNKNOWN.
- The critical test is that payment_collect_request_expired and
  payment_timed_out (both containing "expired" in English) must map to
  BANK_TIMEOUT, not CARD_EXPIRED. This is the one precision trap.
"""
from __future__ import annotations

import pytest

from src.core.types import DeclineClass


# --- parametrized table from Razorpay official docs -------------------------

DECLINE_TABLE = [
    ("insufficient_funds", "The payment did not go through because the customer's bank account did not have enough funds.", DeclineClass.INSUFFICIENT_FUNDS),
    (None, "insufficient balance in account", DeclineClass.INSUFFICIENT_FUNDS),
    ("card_expired", "The payment could not be completed because the customer's card is expired.", DeclineClass.CARD_EXPIRED),
    (None, "customer's card has expired", DeclineClass.CARD_EXPIRED),
    ("invalid_vpa", "The payment was unsuccessful due to the customer not being a valid user on the UPI App.", DeclineClass.ACCOUNT_CLOSED),
    ("debit_instrument_blocked", "The payment could not be processed due to the card being blocked.", DeclineClass.ACCOUNT_CLOSED),
    ("card_not_enrolled", "The payment was unsuccessful as the card was not activated for online transactions.", DeclineClass.ACCOUNT_CLOSED),
    ("card_declined", "The payment was declined by the customer's bank.", DeclineClass.ISSUER_DECLINE),
    ("payment_declined", "The payment did not go through because the funds could not be debited from the customer's account.", DeclineClass.ISSUER_DECLINE),
    ("payment_risk_check_failed", "the customer's bank declined the payment, citing it as fraudulent.", DeclineClass.ISSUER_DECLINE),
    ("incorrect_cvv", "customer entered an incorrect CVV.", DeclineClass.ISSUER_DECLINE),
    ("bank_technical_error", "downtime on the customer's bank.", DeclineClass.BANK_TIMEOUT),
    ("gateway_technical_error", "downtime on our partner bank.", DeclineClass.BANK_TIMEOUT),
    ("payment_timed_out", "customer exceeded the time limit for payment processing.", DeclineClass.BANK_TIMEOUT),
    ("payment_collect_request_expired", "customer exceeded the 10-minute time limit.", DeclineClass.BANK_TIMEOUT),
    (None, "mandate revoked by customer", DeclineClass.MANDATE_REVOKED),
    (None, "the mandate has been cancelled", DeclineClass.MANDATE_REVOKED),
    ("totally_unrecognised_code_xyz", "some novel bank narration nobody has seen before", DeclineClass.UNKNOWN),
    (None, None, DeclineClass.UNKNOWN),
    ("", "", DeclineClass.UNKNOWN),
    # --- text-only (code=None) real Razorpay description strings ----------
    # payments-domain's B3 review demonstrated that with code stripped, the
    # ORIGINAL keyword lists (built mostly from underscored error_reason
    # tokens) collapsed every one of these to UNKNOWN, because the real
    # human-readable description often doesn't contain the enum token at
    # all (e.g. the real insufficient_funds description never says the
    # word "insufficient"). These pin the fix.
    (None, "The payment did not go through because the customer's bank account did not have enough funds.", DeclineClass.INSUFFICIENT_FUNDS),
    (None, "The payment was unsuccessful due to the customer not being a valid user on the UPI App.", DeclineClass.ACCOUNT_CLOSED),
    (None, "The payment could not be processed due to the card being blocked.", DeclineClass.ACCOUNT_CLOSED),
    (None, "The payment was unsuccessful as the card was not activated for online transactions.", DeclineClass.ACCOUNT_CLOSED),
    (None, "There was a downtime on the customer's bank due to which the payment has failed.", DeclineClass.BANK_TIMEOUT),
    (None, "The payment could not be completed as the customer exceeded the time limit for payment processing.", DeclineClass.BANK_TIMEOUT),
    # "account has been closed" -- real prose, not the contiguous phrase
    # "account closed" the original keyword list matched.
    (None, "The payment was declined as the customer's account has been closed.", DeclineClass.ACCOUNT_CLOSED),
]


@pytest.mark.parametrize("code,text,expected_class", DECLINE_TABLE)
def test_classify_matches_razorpay_ground_truth(code, text, expected_class):
    """All entries in the Razorpay official docs must classify correctly."""
    from src.classify.decline_taxonomy import classify

    assert classify(code, text) == expected_class


# --- critical precision trap ------------------------------------------------

def test_collect_request_expired_is_timeout_not_card_expired():
    """payment_collect_request_expired contains the word 'expired' but describes
    a customer-response TIMEOUT (BANK_TIMEOUT), not a dead card (CARD_EXPIRED).
    This is the one semantic trap in the taxonomy."""
    from src.classify.decline_taxonomy import classify

    result = classify("payment_collect_request_expired", "customer exceeded the 10-minute time limit.")
    assert result == DeclineClass.BANK_TIMEOUT
    assert result != DeclineClass.CARD_EXPIRED


# --- case insensitivity ----

def test_case_insensitive_matching():
    """Matching must work regardless of case in code or text."""
    from src.classify.decline_taxonomy import classify

    # Test with lowercase vs uppercase
    lower_result = classify("insufficient_funds", "The payment did not go through")
    upper_result = classify("INSUFFICIENT_FUNDS", "THE PAYMENT DID NOT GO THROUGH")
    assert lower_result == upper_result == DeclineClass.INSUFFICIENT_FUNDS

    # Test mixed case
    mixed_result = classify("Card_Expired", "customer's card has EXPIRED")
    assert mixed_result == DeclineClass.CARD_EXPIRED

    # Test with None code, uppercase text
    none_upper = classify(None, "MANDATE REVOKED BY CUSTOMER")
    assert none_upper == DeclineClass.MANDATE_REVOKED


# --- non-negotiable invariant ------------------------------------------------

def test_insufficient_funds_and_mandate_revoked_are_never_the_same_class():
    """INSUFFICIENT_FUNDS (transient liquidity gap, CANT_PAY_NOW) and
    MANDATE_REVOKED (dead instrument, CANT_PAY_EVER) must NEVER collapse
    into the same DeclineClass. This is a project-wide non-negotiable."""
    from src.classify.decline_taxonomy import classify

    insufficient = classify("insufficient_funds", "The payment did not go through because the customer's bank account did not have enough funds.")
    revoked = classify(None, "mandate revoked by customer")

    assert insufficient != revoked
    assert insufficient == DeclineClass.INSUFFICIENT_FUNDS
    assert revoked == DeclineClass.MANDATE_REVOKED


# --- cross-field contamination (found by payments-domain's B3 review) ------

def test_payment_cancelled_naming_the_mandate_is_not_mandate_revoked():
    """A per-attempt cancel (customer backed out of THIS collect request)
    must not become MANDATE_REVOKED just because its own free text names
    "mandate" -- UPI AutoPay's ordinary product noun -- alongside a form of
    "cancelled". `payment_cancelled` is a real, specific Razorpay reason
    code (verified 2026-08-27) for exactly this per-attempt case; its
    presence must override the mandate+cancelled heuristic."""
    from src.classify.decline_taxonomy import classify

    result = classify(
        "payment_cancelled",
        "The customer cancelled the UPI AutoPay mandate approval request.",
    )
    assert result != DeclineClass.MANDATE_REVOKED
