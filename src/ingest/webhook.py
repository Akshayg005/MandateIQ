"""FastAPI receiver for Razorpay webhook deliveries -- payment.failed and
subscription.* events. Order of operations, strictly: verify signature ->
parse -> replay-window check -> dedupe -> classify/route -> single write.
Nothing before signature verification touches the database or trusts the
body's contents.

Signature scheme (independently verified against Razorpay's own webhook
docs): X-Razorpay-Signature is the hex HMAC-SHA256 of the RAW request body,
keyed on RAZORPAY_WEBHOOK_SECRET. Verified with hmac.compare_digest, never
a bare == -- a bare comparison leaks timing information about how many
leading bytes matched, exactly the class of bug this file's source-guard
test exists to catch.

The event-id header (x-razorpay-event-id, Razorpay's own recommended
dedupe key) is UNSIGNED -- the signature covers only the body -- so a
captured old body under a relabelled event-id needs the independent
replay-window check on the body's own created_at, not just dedupe.

A duplicate event_id returns HTTP 200, not an error: Razorpay retries any
non-2xx response on backoff for 24h after the event's created_at, so
answering a duplicate with an error would just make it retry forever for
no reason.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from src.classify.cause_map import PRIOR_VERSION, prior
from src.classify.decline_taxonomy import TAXONOMY_VERSION, classify
from src.core.clock import now
from src.ingest import dedupe
from src.ingest.deps import get_conn
from src.ingest.lifecycle_route import record as record_lifecycle
from src.ledger.store import record_ingested_event

# Not a regulatory figure -- no RBI clause sets a webhook replay window.
# An operational choice, roughly 2x Razorpay's own documented 24h
# retry-until-give-up window, so a delivery that arrives late in that
# window (still legitimate) isn't rejected by an unrelated coincidence of
# timing, while a genuinely stale/replayed body still gets caught.
REPLAY_WINDOW_SECONDS = 48 * 3600

router = APIRouter()


@router.post("/webhook/razorpay")
async def receive_webhook(request: Request, conn=Depends(get_conn)):
    raw_body = await request.body()

    received_signature = request.headers.get("X-Razorpay-Signature")
    if not received_signature:
        raise HTTPException(status_code=400, detail="missing signature")

    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    expected_signature = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_signature, received_signature):
        raise HTTPException(status_code=400, detail="invalid signature")

    event_id = request.headers.get("x-razorpay-event-id")
    if not event_id:
        raise HTTPException(status_code=400, detail="missing event id")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    event_time = datetime.fromtimestamp(payload["created_at"], tz=timezone.utc)
    age_seconds = abs((now() - event_time).total_seconds())
    if age_seconds > REPLAY_WINDOW_SECONDS:
        raise HTTPException(status_code=400, detail="event outside replay window")

    if dedupe.seen(conn, event_id):
        return {"status": "duplicate"}
    dedupe.mark(conn, event_id, payload.get("event", ""))

    event_type = payload.get("event", "")
    if event_type == "payment.failed":
        _handle_payment_failed(conn, event_id, event_type, raw_body, payload)
    elif event_type.startswith("subscription."):
        record_lifecycle(conn, event_id, payload)
    # Any other event type: already dedupe-marked above; otherwise a no-op.
    # Returning 200 here (rather than only for handled types) matters --
    # Razorpay retries a non-2xx for 24h, and there is no reason to make it
    # retry an event this receiver has no handler for at all.

    return {"status": "ok"}


def _handle_payment_failed(
    conn, event_id: str, event_type: str, raw_body: bytes, payload: dict,
) -> None:
    """mandate_id resolution order: payload.payment.entity.notes (a
    merchant-set reference, stashed at order-creation time -- Razorpay's
    Payment entity has no subscription_id and no other reliable link back
    to a mandate) -> a sibling payload.subscription.entity.id, for bodies
    that carry one -> None. An honest NULL when neither is present, never
    a guess."""
    entity = payload["payload"]["payment"]["entity"]

    notes = entity.get("notes")
    mandate_id = notes.get("mandate_id") if isinstance(notes, dict) else None
    if mandate_id is None:
        sub_entity = payload.get("payload", {}).get("subscription", {}).get("entity", {})
        mandate_id = sub_entity.get("id")

    decline_class = classify(entity.get("error_reason"), entity.get("error_description"))
    cause_prior_json = json.dumps({cause.value: p for cause, p in prior(decline_class).items()})

    record_ingested_event(
        conn,
        event_id=event_id,
        event_type=event_type,
        # The RAW bytes actually received, not a re-serialization of the
        # parsed dict -- key order/whitespace/escaping are not guaranteed
        # to round-trip, and this hash exists to let an auditor verify the
        # exact bytes that arrived, not a semantic reconstruction of them.
        raw_payload_sha256=hashlib.sha256(raw_body).hexdigest(),
        mandate_id=mandate_id,
        provider_ref=entity.get("id"),
        decline_code=entity.get("error_reason"),
        decline_text=entity.get("error_description"),
        decline_class=decline_class.value,
        cause_prior_json=cause_prior_json,
        taxonomy_version=TAXONOMY_VERSION,
        prior_version=PRIOR_VERSION,
        # Razorpay's `amount` for INR is already an integer in paise --
        # never a float, never divided or multiplied to convert units.
        amount_paise=int(entity["amount"]),
    )
