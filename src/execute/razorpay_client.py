"""The only module that talks to Razorpay. Two things live here:

  RazorpayLike  -- the Protocol executor.py and recover.py depend on. Both
                   take a client as an argument (dependency injection) so
                   their tests run against an in-memory double, never
                   against the network -- a race-condition or crash-
                   recovery test that depended on a live HTTP round-trip
                   would be flaky by construction and too slow to run on
                   every commit.
  RazorpayClient -- the real SDK-backed implementation, used only at the
                   actual integration edge (a real test-mode call, the
                   same discipline scripts/idempotency_spike.py already
                   established) and, eventually, production wiring.

Must never be imported from src/model/ or src/policy/ -- CLAUDE.md
invariant 1 (no LLM there) is unrelated to this specific rule, but the
dependency-edge discipline is the same: the decision core must not know
this module exists.

Must never expose a method that reaches Razorpay's subscription-cancellation
endpoint -- scripts/guard_invariants.py's invariant-6 guard (HARD_CANCEL)
allows that call shape in exactly one file, src/policy/offramp.py, and even
THAT file never actually calls it (it only names cancellation as a menu
string). This client's own method set (create_order, charge,
pause_subscription, find_by_receipt) has no reason to ever need one.

find_by_receipt anchors on ORDERS, and charge() creates an order before it
creates the recurring payment. This was REVERSED on 2026-08-30 after the
first live test-mode call ever made through this module (POSTMORTEM.md
incident 3, DECISIONS.md 2026-08-30 B9).

An earlier version queried Payments -- payment.all({"receipt": ...}) -- on
the reasoning that recovery must search the entity the ambiguous call
actually creates, and charge() (payment.createRecurring) creates a Payment,
never an Order. The REASONING was right; the MECHANISM does not exist.
Razorpay rejects a receipt filter on the Payments resource outright:

    receipt is/are not required and should not be sent

`receipt` is an Order field. Payments have none. That method could never
have returned anything, for any input, against the real API -- and since
the B3 spike proved `receipt` does not dedupe Order.create, this lookup is
the ENTIRE "recover by asking, never by resending" path. Every
UNCONFIRMED -> RESULT resolution runs through it.

Measured alternatives, both against live test mode:
  - order.all({"receipt": R})  -- WORKS, server-side indexed, exact: three
    known receipts each returned exactly their own order (count=1).
  - payment.all({"from","to"}) + a client-side notes match -- also works,
    but scans a bounded window client-side (count caps at 100/page), so it
    degrades on a busy account exactly when recovery matters most.

The first was chosen. charge() therefore creates an order carrying the
idempotency key as its receipt, then creates the recurring payment against
that order_id -- which is also how Razorpay itself models recurring debits
(the one real Payment in this test account, pay_TUqQ25JYjOyNPD from B3,
carries an order_id). Recovery is then an indexed two-step: find the order
by receipt, then read that order's payments.

KNOWN LIMIT, measured not assumed: the Orders LIST endpoint lags indexing.
An order queried by its own receipt at 0s, 3s and 8s after creation
returned count=0, and was absent from the unfiltered recent list too; it
appeared minutes later. So None from this method means "not found YET", not
"never created" -- which is precisely why recover.py treats a miss as
UNCONFIRMED and keeps asking on backoff rather than concluding anything.
The slot stays consumed throughout.

find_by_receipt's contract, precisely (DECISIONS.md, 2026-08-27, B3
"Provider idempotency spike"): `receipt` does NOT dedupe Order.create --
sending the same receipt twice measurably produced two distinct orders
(DOUBLE_CREATED, order_TUlHyAjGj0hWzK and order_TUlHyP6FTZCKYY). That
finding is about Order.create specifically; whether Payments' own receipt
field behaves the same way (very likely, given the shared platform
convention, but NOT independently spiked the way Order.create was) is a
second, disclosed gap alongside charge()'s own unverified shape below. This
method is therefore a LOOKUP FILTER on a fetch/list call after a crash --
"did a payment with this receipt already get created?" -- never something
relied on to have PREVENTED a duplicate charge from happening. The actual
prevention is invariant 3 (ledger write before the money action) plus
src/execute/executor.py's INTENT-row-first, lease-before-send ordering;
recovery here is by ASKING, never by resending.

RazorpayClientError vs RazorpayDeclined -- the distinction executor.py's
money-safety logic actually depends on, per money-auditor's own checklist
("every Razorpay call has a defined behaviour on timeout, on 5xx, and on a
response that arrives after the client gave up"). The SDK exposes distinct
exception classes (razorpay.errors.BadRequestError, GatewayError,
ServerError), inspected directly against the installed SDK version rather
than assumed from memory:

  RazorpayDeclined    -- razorpay.errors.BadRequestError only: a
                         DEFINITIVE, synchronous rejection. The provider
                         received the request and rejected it for a
                         stated reason. Safe for a caller to record as a
                         known outcome.
  RazorpayClientError -- everything else (GatewayError, ServerError,
                         network/timeout errors). AMBIGUOUS: the caller
                         must NOT assume the request did or did not reach
                         the provider. A double-charge is worse than an
                         extra reconciliation pass -- this is the whole
                         reason recover.py's find_by_receipt exists.

`charge()`'s exact request/response shape against Razorpay's recurring-
payment endpoint (payment.createRecurring, for a merchant-initiated debit
against a saved token -- the UPI AutoPay / e-mandate case this project
exists for, as distinct from Razorpay's own auto-billed Subscriptions
product, which would take timing out of this system's hands entirely) has
NOT been independently spiked against live test-mode traffic the way
Order.create was at B3. Flagged here as a real gap, not guessed at as
verified: the amount_paise-as-paise convention below matches what B3's
spike observed Order.create accepting (`{"amount": 100, ...}` for a
₹1.00 order), and Razorpay's API is documented as paise-denominated
throughout, so the unit convention is on solid ground; the exact field
names createRecurring expects are not independently confirmed here the
way Order.create's were. A follow-up spike before this touches real
test-mode traffic is recommended the same way B3 recommended one for
Order.create, and this is disclosed rather than silently assumed correct.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FaultSpec:
    """A TEST-ONLY fault seam (B10). Exists because `kill -9` on our own
    process reproduces only the BENIGN half of the crash space.

    The dangerous case this project actually has to survive is "the provider
    ACCEPTED the debit, and the response was lost in flight" -- the customer
    is charged and our process never learns it. No signal we can send to our
    own process creates that state, because the state is created on the far
    side of the network. It has to be injected at the client boundary, which
    is this module, which is why the seam lives here rather than in the eval
    harness that uses it.

    `drop_response_after_accept` performs the real create_order and the real
    createRecurring -- a genuine payment exists at the provider afterwards --
    and then raises RazorpayClientError instead of returning. That is exactly
    the wire truth of a lost response, and executor.py's existing AMBIGUOUS
    branch is what must then hold the line: leave SENT as the last row, do
    NOT release the lease, let recover.py resolve it by ASKING.

    Reachable only from eval/chaos.py and tests/ -- enforced by
    scripts/guard_invariants.py, not by trust. A fault seam that production
    code could construct is a production defect, not a test utility.
    """

    drop_response_after_accept: bool = False


class RazorpayClientError(RuntimeError):
    """An AMBIGUOUS failure -- network error, timeout, gateway error, server
    error. We do NOT know whether the provider actually processed the
    request. Callers must treat this the same way a hard process crash is
    treated: never assume nothing happened (see module docstring and the
    B3 DOUBLE_CREATED finding)."""


class RazorpayDeclined(RuntimeError):
    """A DEFINITIVE, synchronous rejection: razorpay.errors.BadRequestError
    only. The provider received the request and stated a reason. Safe to
    record as a known outcome, unlike RazorpayClientError."""


@runtime_checkable
class RazorpayLike(Protocol):
    """What executor.py and recover.py depend on. A test double implementing
    this Protocol structurally satisfies isinstance(double, RazorpayLike)
    without importing the real razorpay package at all."""

    def create_order(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        ...

    def charge(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        ...

    def pause_subscription(self, subscription_id: str) -> dict:
        ...

    def find_by_receipt(self, receipt: str) -> dict | None:
        ...


class RazorpayClient:
    """The real SDK-backed implementation. Constructed lazily -- importing
    this module must not require network access or even the `razorpay`
    package to be importable-and-configured at import time, only when a
    RazorpayClient is actually instantiated, so unit tests can import
    RazorpayLike/RazorpayClient freely while running entirely against
    doubles.
    """

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        *,
        fault: FaultSpec | None = None,
    ) -> None:
        import razorpay  # local import: see class docstring

        # See FaultSpec's docstring. Default None means every production
        # construction of this class is fault-free by omission, not by
        # remembering to pass something.
        self._fault = fault

        key_id = key_id if key_id is not None else os.environ.get("RAZORPAY_KEY_ID", "")
        key_secret = (
            key_secret if key_secret is not None else os.environ.get("RAZORPAY_KEY_SECRET", "")
        )
        # Defense in depth alongside guard_invariants.py's repo-wide
        # rzp_live_ text scan (CLAUDE.md invariant 5) -- this is the one
        # place a live key would actually be USED, not merely present in a
        # file, so it gets its own runtime assertion.
        if not key_id.startswith("rzp_test_"):
            raise RazorpayClientError(
                f"RAZORPAY_KEY_ID is not a test-mode key: {key_id[:12]!r}... "
                "-- this project is test-mode only (CLAUDE.md invariant 5)"
            )
        self._client = razorpay.Client(auth=(key_id, key_secret))

    def create_order(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        try:
            return self._client.order.create(
                {"amount": amount_paise, "currency": "INR", "receipt": receipt, "notes": notes}
            )
        except Exception as exc:  # noqa: BLE001 -- see RazorpayClientError docstring
            raise RazorpayClientError(f"create_order({receipt!r}) failed: {exc}") from exc

    def charge(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        # Two steps, in this order, because find_by_receipt anchors on the
        # ORDER -- see the module docstring. The order is what carries
        # `receipt` (Payments have no such field), so it is what makes this
        # attempt findable after a crash. Creating it FIRST means a crash
        # between the two calls still leaves a receipt-addressable record:
        # recovery finds the order, sees zero payments, and correctly
        # reports "nothing charged yet" rather than finding nothing at all.
        #
        # See module docstring: createRecurring's exact field shape is still
        # unverified against live traffic. Recommend a B3-style spike before
        # this path carries real money.
        import razorpay.errors

        order = self.create_order(amount_paise=amount_paise, receipt=receipt, notes=notes)

        try:
            payment = self._client.payment.createRecurring(
                {
                    "amount": amount_paise,
                    "currency": "INR",
                    "order_id": order["id"],
                    "notes": notes,
                }
            )
        except razorpay.errors.BadRequestError as exc:
            # DEFINITIVE: the provider received and rejected this request.
            raise RazorpayDeclined(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 -- AMBIGUOUS, see RazorpayClientError
            raise RazorpayClientError(f"charge({receipt!r}) failed: {exc}") from exc

        # THE FAULT SEAM. Deliberately placed AFTER the call succeeded: the
        # payment above is real and the money has moved. Only the answer is
        # lost. See FaultSpec.
        if self._fault is not None and self._fault.drop_response_after_accept:
            raise RazorpayClientError(
                f"charge({receipt!r}): response dropped in flight after the provider "
                "accepted (injected fault, B10)"
            )

        return payment

    def pause_subscription(self, subscription_id: str) -> dict:
        try:
            return self._client.subscription.pause(subscription_id, {"pause_at": "now"})
        except Exception as exc:  # noqa: BLE001
            raise RazorpayClientError(
                f"pause_subscription({subscription_id!r}) failed: {exc}"
            ) from exc

    def find_by_receipt(self, receipt: str) -> dict | None:
        """Lookup filter only -- see module docstring. Two indexed steps:
        find the ORDER carrying this receipt (Payments have no receipt
        field; filtering them by one is rejected by the API), then read
        that order's payments.

        Returns the single payment for `receipt`, or None. None is
        deliberately NOT a claim that nothing was charged -- it means "no
        payment found yet", and covers three genuinely different states
        this method cannot distinguish:

          - the crash happened before the order was ever created;
          - the order exists but no payment was created against it;
          - both exist, but the Orders list endpoint has not indexed them
            yet (measured: count=0 at 0s/3s/8s after creation, present
            minutes later).

        recover.py is built for exactly this: a miss becomes UNCONFIRMED
        and is asked again on backoff, never treated as proof nothing was
        sent, and the slot stays consumed throughout.

        Raises RazorpayClientError rather than returning anything if the
        receipt matches MORE than one order, or the order more than one
        payment -- B3's spike proved a duplicate create is reachable
        (DOUBLE_CREATED), and a genuine double is the case that needs a
        human, not a guess."""
        try:
            orders = self._client.order.all({"receipt": receipt})
        except Exception as exc:  # noqa: BLE001
            raise RazorpayClientError(f"find_by_receipt({receipt!r}) failed: {exc}") from exc

        order_items = orders.get("items", [])
        if not order_items:
            return None
        if len(order_items) > 1:
            raise RazorpayClientError(
                f"receipt {receipt!r} matches {len(order_items)} orders -- ambiguous, "
                "needs manual reconciliation (see B3 DOUBLE_CREATED finding)"
            )

        order_id = order_items[0]["id"]
        try:
            payments = self._client.order.payments(order_id)
        except Exception as exc:  # noqa: BLE001
            raise RazorpayClientError(
                f"find_by_receipt({receipt!r}) failed reading payments for {order_id}: {exc}"
            ) from exc

        payment_items = payments.get("items", [])
        if not payment_items:
            return None
        if len(payment_items) > 1:
            raise RazorpayClientError(
                f"order {order_id} (receipt {receipt!r}) has {len(payment_items)} payments -- "
                "ambiguous, needs manual reconciliation"
            )
        return payment_items[0]
