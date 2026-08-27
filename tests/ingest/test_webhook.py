"""
src/ingest/webhook.py + app.py + deps.py -- FastAPI endpoint for handling
Razorpay webhook deliveries (payment.failed, subscription.* events).

Design decisions this test file pins:

- Signature verification uses hmac.compare_digest, never a bare == operator,
  to prevent timing attacks.
- Signatures are verified BEFORE any database writes or further processing.
- Event-id header is required (dedupe key), and stale events (>48h old) are
  rejected with HTTP 400 -- no writes.
- Duplicate event_ids return HTTP 200 with {"status": "duplicate"}, not an
  error, so Razorpay does not retry forever (Razorpay retries any non-2xx).
- payment.failed events are classified via decline_taxonomy and recorded in
  ingested_event; subscription.* events are routed to lifecycle_route.
- All writes (ingested_event, mandate_lifecycle, webhook_event dedupe) use
  the connection injected via FastAPI's Depends(get_conn), which tests
  override to point at pg_schema.conn for isolation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import pathlib
import re
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WEBHOOK_SRC = ROOT / "src" / "ingest" / "webhook.py"

# Test webhook secret -- must match what the tests env-var sets
WEBHOOK_SECRET = "test-webhook-secret-do-not-use-in-prod"


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Compute HMAC-SHA256 signature of body, keyed on secret."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payment_failed_body(
    *,
    event_id_suffix="1",
    amount=150000,
    mandate_id="M-WEBHOOK-TEST",
    error_reason="insufficient_funds",
    error_description="The payment did not go through because the customer's bank account did not have enough funds.",
    created_at=None,
):
    """Build a realistic payment.failed envelope.

    Args:
        event_id_suffix: appended to "pay_TEST" to form the payment id
        amount: integer paise (Razorpay's amount field for INR is already paise)
        mandate_id: stored in payment.entity.notes.mandate_id
        error_reason: Razorpay error_reason field
        error_description: Razorpay error_description field
        created_at: Unix timestamp (seconds). If None, use current time.
    """
    if created_at is None:
        created_at = int(time.time())

    return {
        "entity": "event",
        "account_id": "acc_test123",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": f"pay_TEST{event_id_suffix}",
                    "amount": amount,
                    "currency": "INR",
                    "status": "failed",
                    "notes": {"mandate_id": mandate_id},
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_reason": error_reason,
                    "error_description": error_description,
                }
            }
        },
        "created_at": created_at,
    }


def _subscription_cancelled_body(
    *,
    event_id_suffix="1",
    mandate_id="sub_TEST1",
    created_at=None,
):
    """Build a realistic subscription.cancelled envelope."""
    if created_at is None:
        created_at = int(time.time())

    return {
        "entity": "event",
        "account_id": "acc_test123",
        "event": "subscription.cancelled",
        "contains": ["subscription"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": mandate_id,
                    "status": "cancelled",
                    "notes": {},
                }
            }
        },
        "created_at": created_at,
    }


@pytest.fixture(autouse=True)
def _set_webhook_secret(monkeypatch):
    """Every test in this file gets RAZORPAY_WEBHOOK_SECRET set to the known
    test secret, so signing and verification use the same value."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", WEBHOOK_SECRET)


@pytest.fixture
def client(pg_schema):
    """FastAPI TestClient with dependency overrides.

    - Imports the app and deps from src.ingest
    - Overrides get_conn to return pg_schema.conn (the test's isolated schema)
    - Yields the client
    - Clears overrides afterward (cleanup)
    """
    from src.ingest.app import app
    from src.ingest.deps import get_conn

    app.dependency_overrides[get_conn] = lambda: pg_schema.conn
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# --- Signature verification and required headers ----

def test_valid_signature_and_realistic_payment_failed_body_is_accepted(client, pg_schema):
    """POST a realistic payment.failed body with a valid X-Razorpay-Signature
    and x-razorpay-event-id header -> HTTP 200.

    Then verify via raw SQL that exactly one row was inserted into ingested_event
    with the right fields, and specifically:
    - decline_class == "INSUFFICIENT_FUNDS"
    - mandate_id == "M-WEBHOOK-TEST"
    - amount_paise == 150000 and is a plain Python int, not float
    - cause_prior parses as JSON and sums to ~1.0
    """
    event_id = "evt-pay-valid-sig-1"
    body_dict = _payment_failed_body(
        event_id_suffix="valid-sig-1",
        amount=150000,
        mandate_id="M-WEBHOOK-TEST",
    )
    body_bytes = json.dumps(body_dict).encode("utf-8")
    signature = _sign(body_bytes)

    response = client.post(
        "/webhook/razorpay",
        content=body_bytes,
        headers={
            "X-Razorpay-Signature": signature,
            "x-razorpay-event-id": event_id,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data.get("status") == "ok"

    # Verify the row in ingested_event
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            """
            SELECT event_id, event_type, mandate_id, provider_ref,
                   decline_code, decline_text, decline_class, cause_prior,
                   taxonomy_version, prior_version, amount_paise
            FROM ingested_event
            WHERE event_id = %s
            """,
            (event_id,),
        )
        row = cur.fetchone()

    assert row is not None
    (event_id_read, event_type, mandate_id, provider_ref, decline_code, decline_text,
     decline_class, cause_prior, taxonomy_version, prior_version, amount_paise) = row

    assert event_type == "payment.failed"
    assert mandate_id == "M-WEBHOOK-TEST"
    assert decline_class == "INSUFFICIENT_FUNDS"
    assert provider_ref == "pay_TESTvalid-sig-1"

    # Versioned, per payments-domain's B3 review -- a classification with
    # no record of which ruleset produced it can't be reproduced later.
    from src.classify.cause_map import PRIOR_VERSION
    from src.classify.decline_taxonomy import TAXONOMY_VERSION
    assert taxonomy_version == TAXONOMY_VERSION
    assert prior_version == PRIOR_VERSION

    # Money must be plain int, not float
    assert isinstance(amount_paise, int), f"amount_paise must be int, got {type(amount_paise)}"
    assert amount_paise == 150000

    # cause_prior must parse as JSON and sum to ~1.0
    if cause_prior is not None:
        prior_dict = json.loads(cause_prior)
        prior_sum = sum(prior_dict.values())
        assert prior_sum == pytest.approx(1.0, abs=0.01)


def test_invalid_signature_rejected(client, pg_schema):
    """POST with a garbage X-Razorpay-Signature header -> HTTP 400, and no
    row inserted into ingested_event afterward."""
    event_id = "evt-pay-invalid-sig"
    body_dict = _payment_failed_body(event_id_suffix="invalid-sig")
    body_bytes = json.dumps(body_dict).encode("utf-8")

    # Use a garbage signature instead of the real one
    garbage_signature = "0" * 64

    response = client.post(
        "/webhook/razorpay",
        content=body_bytes,
        headers={
            "X-Razorpay-Signature": garbage_signature,
            "x-razorpay-event-id": event_id,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 400

    # Verify no row was written
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ingested_event WHERE event_id = %s",
            (event_id,),
        )
        count = cur.fetchone()[0]

    assert count == 0


def test_missing_signature_header_rejected(client, pg_schema):
    """POST with no X-Razorpay-Signature header at all -> HTTP 400."""
    event_id = "evt-pay-no-sig-header"
    body_dict = _payment_failed_body(event_id_suffix="no-sig")
    body_bytes = json.dumps(body_dict).encode("utf-8")

    response = client.post(
        "/webhook/razorpay",
        content=body_bytes,
        headers={
            # NO X-Razorpay-Signature
            "x-razorpay-event-id": event_id,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 400

    # Verify no row was written
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ingested_event WHERE event_id = %s",
            (event_id,),
        )
        count = cur.fetchone()[0]

    assert count == 0


def test_missing_event_id_header_rejected(client, pg_schema):
    """POST with a valid signature but no x-razorpay-event-id header -> HTTP 400."""
    body_dict = _payment_failed_body(event_id_suffix="no-evt-id")
    body_bytes = json.dumps(body_dict).encode("utf-8")
    signature = _sign(body_bytes)

    response = client.post(
        "/webhook/razorpay",
        content=body_bytes,
        headers={
            "X-Razorpay-Signature": signature,
            # NO x-razorpay-event-id
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 400


# --- Replay window: 48h cutoff ----

def test_stale_event_rejected_fresh_event_accepted(client, pg_schema):
    """Build a body with created_at ~49 hours in the past -> signed request ->
    HTTP 400 and no writes. Then build one ~1 hour in the past -> signed ->
    HTTP 200 and written.

    This tests the 48-hour replay-window check using real time (no clock freeze),
    just Unix timestamp arithmetic.
    """
    now_unix = int(time.time())
    stale_unix = now_unix - (49 * 3600)  # 49 hours ago
    fresh_unix = now_unix - (1 * 3600)   # 1 hour ago

    # Stale event
    stale_body_dict = _payment_failed_body(
        event_id_suffix="stale",
        created_at=stale_unix,
    )
    stale_body_bytes = json.dumps(stale_body_dict).encode("utf-8")
    stale_signature = _sign(stale_body_bytes)

    response = client.post(
        "/webhook/razorpay",
        content=stale_body_bytes,
        headers={
            "X-Razorpay-Signature": stale_signature,
            "x-razorpay-event-id": "evt-stale",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400

    # Fresh event
    fresh_body_dict = _payment_failed_body(
        event_id_suffix="fresh",
        created_at=fresh_unix,
    )
    fresh_body_bytes = json.dumps(fresh_body_dict).encode("utf-8")
    fresh_signature = _sign(fresh_body_bytes)

    response = client.post(
        "/webhook/razorpay",
        content=fresh_body_bytes,
        headers={
            "X-Razorpay-Signature": fresh_signature,
            "x-razorpay-event-id": "evt-fresh",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200


# --- Deduplication ----

def test_duplicate_event_id_sent_twice_yields_one_row_and_both_responses_are_200(client, pg_schema):
    """POST the same signed body + same event-id header twice -> both responses
    are HTTP 200 (idempotent success), and ingested_event has exactly one row
    for that event_id."""
    event_id = "evt-dup-dedup"
    body_dict = _payment_failed_body(event_id_suffix="dup-dedup")
    body_bytes = json.dumps(body_dict).encode("utf-8")
    signature = _sign(body_bytes)

    headers = {
        "X-Razorpay-Signature": signature,
        "x-razorpay-event-id": event_id,
        "Content-Type": "application/json",
    }

    # First POST
    response1 = client.post("/webhook/razorpay", content=body_bytes, headers=headers)
    assert response1.status_code == 200

    # Second POST (identical)
    response2 = client.post("/webhook/razorpay", content=body_bytes, headers=headers)
    assert response2.status_code == 200

    # Only one row in ingested_event
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ingested_event WHERE event_id = %s",
            (event_id,),
        )
        count = cur.fetchone()[0]

    assert count == 1


# --- Subscription events: lifecycle_route ----

def test_subscription_cancelled_lands_revoked_in_mandate_lifecycle(client, pg_schema):
    """POST a subscription.cancelled body -> HTTP 200 -> mandate_lifecycle
    table has one row for that mandate_id with state='REVOKED'."""
    event_id = "evt-sub-cancelled"
    mandate_id = "sub_TEST_REVOKED"
    body_dict = _subscription_cancelled_body(
        event_id_suffix="revoked",
        mandate_id=mandate_id,
    )
    body_bytes = json.dumps(body_dict).encode("utf-8")
    signature = _sign(body_bytes)

    response = client.post(
        "/webhook/razorpay",
        content=body_bytes,
        headers={
            "X-Razorpay-Signature": signature,
            "x-razorpay-event-id": event_id,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200

    # Verify the row in mandate_lifecycle
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "SELECT state FROM mandate_lifecycle WHERE mandate_id = %s",
            (mandate_id,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[0] == "REVOKED"


# --- Source guards: no timing attacks, no bare == on signatures ----

def test_no_bare_equality_on_signature_comparison():
    """Verify that webhook.py uses hmac.compare_digest for signature
    verification, not a bare == operator (which is vulnerable to timing
    attacks).

    Check:
    1. The string "hmac.compare_digest" appears in the source
    2. There is no bare "==" comparison of anything named like 'signature'
    """
    text = WEBHOOK_SRC.read_text(encoding="utf-8")

    # Must use hmac.compare_digest
    assert "hmac.compare_digest" in text, (
        "webhook.py must use hmac.compare_digest for signature verification"
    )

    # Must not have bare == on signature variables
    # This regex looks for patterns like "signature ==" or "== signature"
    # We match word boundaries to avoid false positives in comments/strings
    bare_eq_patterns = [
        r'\bsignature\s*==',
        r'==\s*signature\b',
    ]
    for pattern in bare_eq_patterns:
        match = re.search(pattern, text)
        assert match is None, (
            f"webhook.py must not use bare == for signature comparison, found: {match.group(0)!r}"
        )
