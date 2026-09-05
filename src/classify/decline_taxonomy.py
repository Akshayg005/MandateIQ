"""Issuer/gateway decline strings, normalised into a fixed 8-class taxonomy.

classify(code, text) never guesses: unrecognised input is DeclineClass.UNKNOWN,
routed downstream to the B11 LLM normaliser rather than defaulted into some
other class. Coverage is a reported metric (the UNKNOWN rate), not a
swallowed one -- see the payments-domain review's of this file.

Keyword grounding: every phrase below is either a real Razorpay `error_reason`
enum value or a close paraphrase of one, independently verified against
Razorpay's own error-reason documentation (the UPI and card payment
error-reason pages) on 2026-08-27 -- not guessed. Two findings from that
research shape this file's design:

1. Razorpay's own reason vocabulary has no dedicated "the customer revoked
   their mandate" value. When a customer cancels a UPI AutoPay mandate at
   their bank, the *next* debit attempt does not surface a distinct decline
   reason for it -- the subscription instead moves to `pending` and
   Razorpay's own auto-retry keeps trying blindly the following day (exactly
   the incumbent behaviour this project exists to replace). MANDATE_REVOKED
   is therefore matched here only against explicit revocation language that
   might appear in a bank's free-text narration -- a best-effort net, not a
   confident classifier -- while the reliable channel for revocation signal
   is src/ingest/lifecycle_route.py reading the subscription entity's own
   `status`, which IS distinct (see that module).

2. "expired" is not a safe bare keyword: `payment_collect_request_expired`
   (a customer-response TIMEOUT -- they didn't approve the UPI collect
   request within 10 minutes) contains the English word "expired" but is
   nothing like a dead card. CARD_EXPIRED therefore requires "card" AND an
   "expir*" fragment to BOTH appear, never "expired" alone.

3. `payment_cancelled` (the customer backed out of THIS attempt's approval
   prompt) must never be folded into MANDATE_REVOKED: declining one collect
   request is not evidence the whole mandate was revoked, and conflating
   the two would violate the one hard invariant this file is tested against
   (INSUFFICIENT_FUNDS and MANDATE_REVOKED, and by the same logic any
   WONT_PAY-flavoured signal, must never collapse together). This needs an
   explicit guard, not just an absent keyword: "mandate" is UPI AutoPay's
   ordinary product noun, so a `payment_cancelled` event's own free text
   routinely names the mandate it belongs to ("...cancelled the UPI AutoPay
   mandate approval request"), which would otherwise satisfy the
   MANDATE_REVOKED check below by accident -- found by the payments-domain review's B3
   review.

   R5 (2026-09-05, v2): that guard is NARROWED, not deleted. Until R5 this
   signal was left unclassified (-> UNKNOWN) purely because no class
   covered it: a real WONT_PAY-flavoured event with nowhere to go. That
   absence had a measured downstream cost -- no DeclineClass had a WONT_PAY
   prior above 0.45, so `src/policy/belief.py`'s posterior could never reach
   the `{WONT_PAY}` singleton `ConformalCauseGate` fires on, and the entire
   off-ramp lane was structurally unreachable (reports/gates.md, R5).
   `DeclineClass.CUSTOMER_DECLINED` now carries it. The guard still stands
   in its original job -- `payment_cancelled` still cannot become
   MANDATE_REVOKED -- it now routes the event to a class of its own instead
   of to UNKNOWN. The CUSTOMER_DECLINED check is ordered BEFORE the
   MANDATE_REVOKED one for exactly that reason, so the guard is expressed
   as a positive classification rather than as a negative lookahead that
   could silently stop applying.

Known, disclosed gaps this file does NOT attempt to close (the payments-domain review's
B3 review, coverage pre-conceded per the build spec): raw NPCI/NACH response
codes (e.g. "51", "U17", two/three-character codes) arrive, if at all,
outside Razorpay's normalised `error_reason` vocabulary entirely, and no
substring rule here reaches them -- widening the matcher to arbitrary short
codes risks matching a payment amount or an id fragment instead. This is
exactly the free-text
normalisation problem B11's LLM layer exists for; `classify()`'s `text`
parameter already accepts arbitrary free text for that reason, this file
just doesn't yet contain rules for that specific vocabulary. Also: a decline
whose real cause is "amount exceeds the mandate ceiling" (clause 4(c)
territory, not a decline-taxonomy concept) has no dedicated class among the
8 -- it lands in ISSUER_DECLINE, the least-wrong available bucket. (This
paragraph previously said "among the 7 ... adding an 8th is out of scope
here". R5 did add an 8th, CUSTOMER_DECLINED, for a different signal
entirely; the ceiling-exceeded gap is still open and still out of scope,
so the sentence is corrected rather than deleted.)
"""
from __future__ import annotations

from src.core.types import DeclineClass

# Bumped by hand whenever the keyword rules below change meaningfully.
# Recorded per-row in ingested_event.taxonomy_version (see schema.sql) so a
# future read can tell which ruleset classified it -- "the taxonomy will
# grow all week" (new-failure-class skill), and an unversioned classifier
# feeding a belief is the same gap B11's normaliser-versioning gate exists
# to close, just for keyword rules instead of an LLM call.
# v2 (R5, 2026-09-05): `payment_cancelled` / "declined the collect request"
# now classify as CUSTOMER_DECLINED instead of falling through to UNKNOWN.
TAXONOMY_VERSION = "v2"


def classify(code: str | None, text: str | None) -> DeclineClass:
    """Normalise a Razorpay decline -- code is the machine `error_reason`
    (or `error_code`), text is the free-text `error_description` -- into a
    DeclineClass. Case-insensitive substring matching over both combined,
    first-match-wins. Unrecognised, or both inputs empty/None -> UNKNOWN,
    never a guessed default."""
    haystack = f"{code or ''} {text or ''}".lower()

    if not haystack.strip():
        return DeclineClass.UNKNOWN

    # The customer dismissed THIS collect request -- checked FIRST, ahead of
    # MANDATE_REVOKED, because it is the more specific reading of the same
    # words and because ordering it here is what keeps docstring finding 3's
    # guard alive as a positive classification rather than a negative
    # lookahead (R5). Every phrase requires the CUSTOMER as the actor: bare
    # "declined" is an ISSUER_DECLINE keyword further down and must stay
    # one.
    if any(kw in haystack for kw in (
        "payment_cancelled",
        "customer cancelled the collect", "customer declined the collect",
        "declined the collect request", "cancelled the collect request",
        "customer declined the mandate approval",
        "customer did not approve", "customer rejected the collect",
    )):
        return DeclineClass.CUSTOMER_DECLINED

    # Highest-consequence, most specific check second -- see docstring
    # finding 1. The `payment_cancelled` exclusion below is retained
    # (docstring finding 3 / the payments-domain review B3 review): that code's own
    # free text routinely names "mandate" as the ordinary UPI AutoPay
    # product noun, which would otherwise satisfy this check for a
    # per-attempt cancel, not a real mandate revocation. Belt and braces
    # with the CUSTOMER_DECLINED block above -- the guard that existed
    # before a class existed to carry the signal is not removed just
    # because a second mechanism now also covers it.
    if (
        "mandate" in haystack
        and ("revoked" in haystack or "cancelled" in haystack)
        and "payment_cancelled" not in haystack
    ):
        return DeclineClass.MANDATE_REVOKED

    # Compound, never bare "expired" -- see docstring finding 2.
    if "card" in haystack and "expir" in haystack:
        return DeclineClass.CARD_EXPIRED

    if any(kw in haystack for kw in (
        "invalid_vpa", "invalid vpa", "vpa_resolution_failed",
        "not being a valid user",  # real invalid_vpa description text
        "account_closed", "account closed",
        "instrument_blocked", "instrument blocked", "instrument_inactive",
        "card being blocked",  # real debit_instrument_blocked description text
        "not_enrolled", "card_disabled", "disabled_for_online",
        "not activated for online",  # real card_not_enrolled description text
        # Compound, not a fixed phrase: real prose says "account has been
        # closed" -- "account" and "closed" don't sit adjacent (found by
        # the payments-domain review's B3 review).
    )) or ("account" in haystack and "closed" in haystack):
        return DeclineClass.ACCOUNT_CLOSED

    if any(kw in haystack for kw in (
        "insufficient_funds", "insufficient funds", "insufficient balance", "low balance",
        "did not have enough funds", "does not have enough funds",  # real description text
    )):
        return DeclineClass.INSUFFICIENT_FUNDS

    if any(kw in haystack for kw in (
        "bank_technical_error", "gateway_technical_error", "technical_error",
        "bank_downtime", "partner_bank", "timed_out", "timeout",
        "collect_request_expired",
        "downtime",  # real bank_technical_error / gateway_technical_error description text
        "time limit",  # real payment_timed_out / collect_request_expired description text
    )):
        return DeclineClass.BANK_TIMEOUT

    if any(kw in haystack for kw in (
        "card_declined", "payment_declined", "declined",
        "risk_check_failed", "incorrect_cvv", "authentication_failed",
        "transaction_limit", "credit_failed", "payment_failed",
        "account_mismatch",
    )):
        return DeclineClass.ISSUER_DECLINE

    return DeclineClass.UNKNOWN
