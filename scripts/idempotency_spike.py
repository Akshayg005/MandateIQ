"""Standalone spike -- NOT a pytest test. Hits the real Razorpay test-mode
API once, deliberately, to answer a question docs alone can't settle:

If the same `receipt` (Razorpay's own docs call it "treated as an
idempotency key") is sent twice on Order.create, does Razorpay (a) dedupe
and hand back the original order, (b) reject the second call outright, or
(c) -- the dangerous case -- silently create two distinct orders?

the build spec's B9 section already names `find_by_receipt`, not "trust
the key", as the future razorpay_client.py's recovery interface. This
script is what that decision is grounded on -- run for real, not cited
from docs alone. See B3 in reports/gates.md and DECISIONS.md for how the
observed result was used.

Usage:
    python scripts\\idempotency_spike.py
"""
from __future__ import annotations

import os
import sys
import time

import razorpay
from dotenv import find_dotenv, load_dotenv
from razorpay.errors import BadRequestError

load_dotenv(find_dotenv(usecwd=True))


def main() -> int:
    key_id = os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
    assert key_id.startswith("rzp_test_"), (
        f"RAZORPAY_KEY_ID is not a test key: {key_id[:12]!r}..."
    )

    client = razorpay.Client(auth=(key_id, key_secret))
    receipt = f"idem-spike-{int(time.time())}"
    body = {"amount": 100, "currency": "INR", "receipt": receipt}

    print(f"Creating order #1, receipt={receipt!r} ...")
    first = client.order.create(body)
    print(f"  -> {first['id']}")

    print("Creating order #2, IDENTICAL body, same receipt ...")
    second_id = None
    try:
        second = client.order.create(body)
        second_id = second["id"]
        outcome = "DEDUPED_SAME_ORDER" if second_id == first["id"] else "DOUBLE_CREATED"
        print(f"  -> {second_id}")
    except BadRequestError as exc:
        outcome = "REJECTED"
        print(f"  -> rejected: {exc}")

    print(f"\nOUTCOME: {outcome}\n")

    if outcome == "DOUBLE_CREATED":
        print(
            "*** DANGER: receipt did NOT dedupe -- two distinct orders were\n"
            "*** created from an identical body. This contradicts Razorpay's\n"
            "*** own docs and changes B9's executor / recovery design.\n"
        )

    print("--- paste into DECISIONS.md ---")
    print(
        f"### {time.strftime('%Y-%m-%d')} · B3 · Provider idempotency spike -- "
        f"observed: {outcome}\n\n"
        f"`scripts/idempotency_spike.py` called `Order.create` twice against the "
        f"real Razorpay test-mode API with an identical body and a fixed "
        f"`receipt` (`{receipt}`).\n\n"
        f"- order #1: `{first['id']}`\n"
        f"- order #2: `{second_id if second_id else '(rejected, no id created)'}`\n"
        f"- outcome: **{outcome}**\n"
    )

    return 1 if outcome == "DOUBLE_CREATED" else 0


if __name__ == "__main__":
    sys.exit(main())
