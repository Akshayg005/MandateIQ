"""Standalone live smoke check for src/execute/razorpay_client.py -- NOT a
pytest test, and deliberately not on the default test path (it needs network
and credentials).

Why this exists. Every test in tests/execute/test_razorpay_client.py fakes
the SDK, on purpose: crash-recovery and race tests that depended on a live
HTTP round-trip would be flaky by construction. But a fake accepts whatever
parameter shape it is handed, so a fake can never catch a WRONG shape. That
gap shipped a dead-on-arrival recovery interface -- find_by_receipt() called
payment.all({"receipt": ...}), and Razorpay rejects a receipt filter on
Payments outright ("receipt is/are not required and should not be sent").
The suite was green and guard_invariants was clean the whole time. See
POSTMORTEM.md incident 3 and DECISIONS.md 2026-08-30 B9.

So: fake-based tests guard BEHAVIOUR, this guards WIRE FORMAT. They are
separate risks and neither substitutes for the other. Run this at the end of
any block that touches razorpay_client.py.

What it does NOT cover: charge() itself. Driving payment.createRecurring
needs a real saved token / active mandate, which test mode will not mint on
demand, so charge()'s exact field shape remains unverified against live
traffic -- disclosed in the module docstring rather than assumed correct.
This script exercises the two calls that CAN be driven standalone, which
are also the two the recovery path depends on.

Usage:
    .venv\\Scripts\\python.exe scripts\\live_smoke_b9.py
"""
from __future__ import annotations

import pathlib
import sys
import time

from dotenv import find_dotenv, load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(find_dotenv(usecwd=True))

from src.execute.razorpay_client import RazorpayClient  # noqa: E402

# The Orders LIST endpoint lags indexing: an order queried by its own
# receipt at 0s, 3s and 8s after creation measured count=0, and was absent
# from the unfiltered recent list too, appearing minutes later. That lag is
# a real property of the provider, and recover.py is built for it (a miss
# becomes UNCONFIRMED and is asked again on backoff). This script therefore
# retries rather than declaring a failure the moment the first lookup misses
# -- a miss here is only a failure if it NEVER resolves.
_LOOKUP_ATTEMPTS = 12
_LOOKUP_INTERVAL_SECONDS = 15


def main() -> int:
    client = RazorpayClient()  # asserts rzp_test_ at construction
    receipt = f"b9-smoke-{int(time.time())}"

    print(f"1. create_order(receipt={receipt!r}) ...")
    order = client.create_order(amount_paise=100, receipt=receipt, notes={"block": "B9"})
    print(f"   -> {order.get('id')} status={order.get('status')} "
          f"receipt={order.get('receipt')!r}")

    if order.get("receipt") != receipt:
        print("   FAIL: the order did not come back carrying our receipt")
        return 1

    print(f"\n2. find_by_receipt({receipt!r}) -- the recovery interface")
    print(f"   (retrying up to {_LOOKUP_ATTEMPTS}x every {_LOOKUP_INTERVAL_SECONDS}s; "
          "the Orders list endpoint lags indexing)")

    for attempt in range(1, _LOOKUP_ATTEMPTS + 1):
        found = client.find_by_receipt(receipt)
        if found is not None:
            print(f"   -> attempt {attempt}: found payment {found.get('id')}")
            print("\nRESULT: OK -- receipt lookup resolved to a payment.")
            return 0

        # None is ambiguous by design (see find_by_receipt's docstring): no
        # order yet, no payment on the order, or not indexed yet. Here we
        # know no payment was ever created -- create_order alone creates
        # none -- so the meaningful signal is whether the ORDER became
        # findable, which is what the raw lookup below checks.
        orders = client._client.order.all({"receipt": receipt})
        if orders.get("items"):
            print(f"   -> attempt {attempt}: ORDER indexed and found by receipt "
                  f"({orders['items'][0]['id']}); it has no payments, which is "
                  "correct -- create_order alone creates none.")
            print("\nRESULT: OK -- the receipt->order lookup works against the live API.")
            print("        (find_by_receipt correctly returns None for an order with "
                  "no payments; recover.py treats that as UNCONFIRMED, not as proof "
                  "nothing was charged.)")
            return 0

        if attempt < _LOOKUP_ATTEMPTS:
            print(f"   -> attempt {attempt}: not indexed yet, waiting "
                  f"{_LOOKUP_INTERVAL_SECONDS}s ...")
            time.sleep(_LOOKUP_INTERVAL_SECONDS)

    elapsed = _LOOKUP_ATTEMPTS * _LOOKUP_INTERVAL_SECONDS
    print(f"\nRESULT: FAIL -- the order was never findable by its own receipt "
          f"within ~{elapsed}s. Either the receipt filter stopped being honoured "
          "or indexing lag is far worse than measured; do not trust the recovery "
          "path until this is understood.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
