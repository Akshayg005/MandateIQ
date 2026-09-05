"""src/execute/razorpay_client.py -- the sole Razorpay-facing module.

No test here makes a live network call: executor.py and recover.py depend
on RazorpayLike (a Protocol), never on RazorpayClient directly, exactly so
these tests -- and theirs -- can run against an in-memory double. The one
real test-mode call this block promises (create_order, proving the client
is not a fiction) is a separate, manual verification step, not part of the
automated suite (see PLAN's verification section).
"""
from __future__ import annotations

import re

import pytest

from src.execute.razorpay_client import (
    RazorpayClient,
    RazorpayClientError,
    RazorpayDeclined,
    RazorpayLike,
)


# --- invariant 5, defense in depth: this is the one place a key is USED ----

def test_rejects_a_non_test_key_at_construction():
    # Deliberately NOT a live-shaped key (no "rzp_live_" prefix + id chars)
    # -- scripts/guard_invariants.py's repo-wide LIVE_KEY scan would rightly
    # flag a real-looking one appearing even in a test file. The production
    # check here is simply "does not start with rzp_test_", so any
    # non-test-prefixed string exercises the same rejection path.
    with pytest.raises(RazorpayClientError, match="not a test-mode key"):
        RazorpayClient(key_id="not_a_test_key", key_secret="x")


def test_accepts_a_test_key_at_construction():
    # Must not raise, and must not touch the network (razorpay.Client's own
    # constructor is a pure local object -- see the exploration that
    # grounded this test: it returns in <1ms with no credentials check).
    RazorpayClient(key_id="rzp_test_abc123", key_secret="secret")


# --- RazorpayLike is a structural Protocol -----------------------------------

class _FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def create_order(self, *, amount_paise, receipt, notes):
        self.calls.append(("create_order", amount_paise, receipt, notes))
        return {"id": f"order_{receipt}"}

    def charge(self, *, amount_paise, receipt, notes):
        self.calls.append(("charge", amount_paise, receipt, notes))
        return {"id": f"pay_{receipt}"}

    def pause_subscription(self, subscription_id):
        self.calls.append(("pause_subscription", subscription_id))
        return {"status": "paused"}

    def find_by_receipt(self, receipt):
        self.calls.append(("find_by_receipt", receipt))
        return None


def test_fake_client_structurally_satisfies_razorpay_like():
    assert isinstance(_FakeClient(), RazorpayLike)


def test_real_client_structurally_satisfies_razorpay_like():
    assert isinstance(RazorpayClient(key_id="rzp_test_abc", key_secret="x"), RazorpayLike)


# --- error wrapping: every SDK failure becomes RazorpayClientError ----------
# Each stubs the underlying SDK resource directly (no network) and asserts
# the wrapper both raises the project's own exception type and preserves
# the original as __cause__, so nothing about the underlying failure is
# silently discarded.

def _client_with_stubbed_order(order_stub):
    c = RazorpayClient(key_id="rzp_test_abc", key_secret="x")
    c._client.order = order_stub
    return c


class _RaisingOrder:
    def create(self, body):
        raise RuntimeError("simulated Razorpay 5xx")

    def all(self, params):
        raise RuntimeError("simulated Razorpay 5xx")


def test_create_order_wraps_sdk_failure():
    client = _client_with_stubbed_order(_RaisingOrder())
    with pytest.raises(RazorpayClientError) as exc_info:
        client.create_order(amount_paise=100, receipt="r1", notes={})
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_pause_subscription_wraps_sdk_failure():
    client = RazorpayClient(key_id="rzp_test_abc", key_secret="x")

    class _RaisingSubscription:
        def pause(self, sub_id, data):
            raise RuntimeError("simulated failure")

    client._client.subscription = _RaisingSubscription()
    with pytest.raises(RazorpayClientError):
        client.pause_subscription("sub_123")


class _CreatingOrder:
    """charge() creates an order BEFORE the recurring payment (so a crash
    between the two still leaves a receipt-addressable record). Every
    charge test therefore has to stub the order step as well, or it never
    reaches the payment step it means to exercise."""

    def __init__(self):
        self.created_with = None

    def create(self, body):
        self.created_with = body
        return {"id": "order_stub", "receipt": body.get("receipt")}


def _client_for_charge(payment_stub):
    c = RazorpayClient(key_id="rzp_test_abc", key_secret="x")
    c._client.order = _CreatingOrder()
    c._client.payment = payment_stub
    return c


def test_charge_creates_the_order_before_the_payment():
    """The ordering the crash-recovery design depends on: the receipt-
    carrying record must exist before the ambiguous call is made."""
    class _CapturingPayment:
        def __init__(self):
            self.body = None

        def createRecurring(self, body):
            self.body = body
            return {"id": "pay_1", "status": "captured"}

    payment = _CapturingPayment()
    client = _client_for_charge(payment)
    client.charge(amount_paise=100, receipt="r-order-first", notes={"m": "1"})

    assert client._client.order.created_with["receipt"] == "r-order-first"
    assert payment.body["order_id"] == "order_stub", "payment must bind to the order"
    assert "receipt" not in payment.body, "Payments have no receipt field"


def test_charge_wraps_ambiguous_failure_as_client_error():
    class _RaisingPayment:
        def createRecurring(self, body):
            raise RuntimeError("simulated network/timeout failure")

    client = _client_for_charge(_RaisingPayment())
    with pytest.raises(RazorpayClientError):
        client.charge(amount_paise=100, receipt="r1", notes={})


def test_charge_wraps_a_definitive_decline_as_razorpay_declined():
    """The distinction executor.py's money-safety logic depends on: a
    razorpay.errors.BadRequestError is a DEFINITIVE rejection, safe to
    record as a known outcome -- unlike RazorpayClientError's ambiguous
    everything-else, this must never be confused with 'we don't know if
    it was sent.'"""
    import razorpay.errors

    class _DecliningPayment:
        def createRecurring(self, body):
            raise razorpay.errors.BadRequestError("payment failed: insufficient funds")

    client = _client_for_charge(_DecliningPayment())
    with pytest.raises(RazorpayDeclined, match="insufficient funds"):
        client.charge(amount_paise=100, receipt="r1", notes={})


# --- find_by_receipt: lookup filter, never a dedupe mechanism --------------
# Anchors on ORDERS, then reads that order's payments. Payments carry no
# `receipt` field at all -- filtering them by one is rejected outright by
# the live API ("receipt is/are not required and should not be sent"), which
# is how the previous Payments-based implementation was found to be dead on
# arrival: POSTMORTEM.md incident 3, DECISIONS.md 2026-08-30 B9.
#
# These stubs cannot catch a wrong parameter shape -- a fake accepts
# whatever it is handed, which is exactly why the bug survived a green
# suite. Wire format is verified by the live smoke check
# (scripts/live_smoke_b9.py), not here; these tests guard BEHAVIOUR.

class _StubOrder:
    """Stands in for client.order: `all` filters by receipt, `payments`
    returns that order's payments."""

    def __init__(self, orders, payments=None):
        self._orders = orders
        self._payments = payments if payments is not None else []
        self.payments_called_with = None

    def all(self, params):
        assert "receipt" in params, "find_by_receipt must filter orders by receipt"
        return {"items": self._orders}

    def payments(self, order_id):
        self.payments_called_with = order_id
        return {"items": self._payments}


def _client_with_stubbed_order(order_stub):
    c = RazorpayClient(key_id="rzp_test_abc", key_secret="x")
    c._client.order = order_stub
    c._client.payment = _MustNotCharge()
    return c


class _MustNotCharge:
    def createRecurring(self, body):
        raise AssertionError("find_by_receipt must never call charge()/createRecurring()")

    def all(self, params):
        raise AssertionError(
            "find_by_receipt must not filter PAYMENTS -- the live API rejects a "
            "receipt filter there (POSTMORTEM incident 3)"
        )


def test_find_by_receipt_returns_none_when_no_order_exists():
    """Covers both 'never created' and 'not indexed yet' -- the method
    cannot distinguish them, and its docstring says so."""
    client = _client_with_stubbed_order(_StubOrder(orders=[]))
    assert client.find_by_receipt("r-none") is None


def test_find_by_receipt_returns_none_when_the_order_has_no_payments():
    """The order was created but no payment ever was: nothing charged, but
    still reported as unresolved rather than as proof of a clean slot."""
    client = _client_with_stubbed_order(_StubOrder(orders=[{"id": "order_1"}], payments=[]))
    assert client.find_by_receipt("r-empty") is None


def test_find_by_receipt_returns_the_single_payment_of_the_matching_order():
    stub = _StubOrder(orders=[{"id": "order_1"}], payments=[{"id": "pay_1", "status": "captured"}])
    client = _client_with_stubbed_order(stub)
    assert client.find_by_receipt("r-one") == {"id": "pay_1", "status": "captured"}
    assert stub.payments_called_with == "order_1", "must read payments of the matched order"


def test_find_by_receipt_raises_on_ambiguous_double_create():
    """The B3 spike's own observed failure mode (DOUBLE_CREATED): two
    orders sharing one receipt. Surfaced loudly rather than silently
    picking one, which would hide exactly the provider behaviour B3
    found."""
    client = _client_with_stubbed_order(
        _StubOrder(orders=[{"id": "order_A"}, {"id": "order_B"}])
    )
    with pytest.raises(RazorpayClientError, match="2 orders"):
        client.find_by_receipt("r-double")


def test_find_by_receipt_raises_when_one_order_has_multiple_payments():
    """A genuine double-charge against a single order needs a human, not a
    guess about which payment is the real one."""
    client = _client_with_stubbed_order(
        _StubOrder(orders=[{"id": "order_1"}], payments=[{"id": "pay_A"}, {"id": "pay_B"}])
    )
    with pytest.raises(RazorpayClientError, match="2 payments"):
        client.find_by_receipt("r-multi-pay")


def test_find_by_receipt_never_filters_payments_by_receipt():
    """Regression guard for POSTMORTEM incident 3: the dead-on-arrival
    implementation called payment.all({'receipt': ...}), which the live API
    rejects. _MustNotCharge.all raises if that is ever reintroduced."""
    stub = _StubOrder(orders=[{"id": "order_1"}], payments=[{"id": "pay_1"}])
    client = _client_with_stubbed_order(stub)
    assert client.find_by_receipt("r-guard") == {"id": "pay_1"}


# --- module-level: no cancellation call anywhere in this file ---------------

def test_module_never_calls_a_cancellation_endpoint():
    """Invariant 6, checked at the source level here too (in addition to
    the write-guard) so this test fails loudly in CI even if the
    hook is ever bypassed."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "execute" / "razorpay_client.py"
    text = src.read_text(encoding="utf-8")
    assert re.search(r"\.cancel(_subscription)?\s*\(", text) is None
