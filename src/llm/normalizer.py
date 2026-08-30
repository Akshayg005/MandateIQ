"""Decline string normalization. Maps issuer-specific strings to the taxonomy.

Called only on the ESCALATION path: src/classify/decline_taxonomy.py's
classify() is the primary classifier, and this module exists for the
DeclineClass.UNKNOWN it deliberately leaves unresolved rather than guesses
at (decline_taxonomy.py's own docstring: "unrecognised input is
DeclineClass.UNKNOWN, routed downstream to the B11 LLM normaliser"). This
module is general-purpose and does not enforce that policy itself -- the
caller decides when to invoke it, exactly as src/classify/ never imports
src/llm/ (scripts/guard_invariants.py's B11 extension) so that decision
cannot live in the protected core.

One Gemini model call per decline string. Deterministic at temperature=0.0.
Version-hashed for audit: if the system prompt changes, the version changes,
and stale cached results are known to be from a different normalizer.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from src.core.types import DeclineClass
from src.llm.client import GeminiClient, GeminiLike
from src.llm.tools import NORMALIZE_TOOL

NORMALIZER_MODEL = os.environ.get("MODEL_NORMALIZER", "gemini-3.5-flash-lite")

# Below the model-confidence floor, force UNKNOWN regardless of what
# decline_class the model named -- see normalize()'s confidence handling.
# Named rather than a bare literal so it can be folded into
# NORMALIZER_VERSION below: a threshold change is a behaviour change, and
# should invalidate cached golden-set answers exactly like a prompt edit
# does (payments-domain review, 2026-08-31 -- the version hash previously
# covered only the prompt text and tool schema, so this constant, and a
# MODEL_NORMALIZER env override, could both change without busting the
# cache, letting a stale answer silently pass as current).
_CONFIDENCE_FLOOR = 0.5

NORMALIZER_SYSTEM_PROMPT = """You are a decline-string normalizer for a payments recovery system.

Your task: classify an issuer's decline string into the system's fixed taxonomy.

Categories (strict, mutually exclusive):
- INSUFFICIENT_FUNDS: temporary liquidity gap, retry later
- MANDATE_REVOKED: mandate was revoked by customer or issuer, cannot retry
- CARD_EXPIRED: card expiry date passed
- ACCOUNT_CLOSED: account, card, or UPI instrument/handle is closed, blocked,
  disabled, or invalid -- covers a literally closed bank account, a card
  blocked or disabled by the issuer for online use, and a UPI VPA that no
  longer resolves to a valid user. All of these mean the instrument itself
  cannot be charged, same as a closed account, even when the decline text
  says "blocked" or "invalid" rather than "closed".
- ISSUER_DECLINE: issuer-specific decline (not a known category)
- BANK_TIMEOUT: bank did not respond in time
- UNKNOWN: unclassifiable

CRITICAL: Never collapse INSUFFICIENT_FUNDS and MANDATE_REVOKED. They are
opposites: one is transient (keep trying), one is terminal (stop trying).
A false positive on MANDATE_REVOKED cancels a paying customer. A false
negative on INSUFFICIENT_FUNDS wastes an attempt. Both matter.

CRITICAL: a decline meaning the customer dismissed or did not approve ONE
payment collect-request/attempt is NOT the same as MANDATE_REVOKED, even if
its text names "mandate" (mandate is UPI AutoPay's ordinary product noun,
so text about declining one approval prompt routinely mentions it). Only
classify MANDATE_REVOKED when the text says the mandate/standing-instruction
itself was revoked, cancelled, or withdrawn -- not when a single attempt was
merely declined or not approved in time. When genuinely unsure which of
these it is, prefer UNKNOWN over MANDATE_REVOKED: UNKNOWN costs a wasted
retry slot, a false MANDATE_REVOKED stops retrying a mandate that may still
be alive.

Bare or opaque codes with no descriptive text (a short numeric or
alphanumeric code alone, e.g. "51", generic acquirer-referral text like "see
acquirer for details") carry real information only if you recognise the
SPECIFIC code from established payment-processing convention (e.g. ISO 8583
"51" is universally Not Sufficient Funds) -- classify those confidently.
Otherwise, an opaque code you do not specifically recognise is genuinely
ambiguous: report LOW confidence rather than guessing at the closest-sounding
category.

Respond with confidence (0.0-1.0), your genuine estimate of correctness --
not a formality. Confidence below {floor} maps to UNKNOWN regardless of
which class you named.
""".replace("{floor}", str(_CONFIDENCE_FLOOR))

NORMALIZER_VERSION = hashlib.sha256(
    (NORMALIZER_SYSTEM_PROMPT + str(NORMALIZE_TOOL) + NORMALIZER_MODEL
     + str(_CONFIDENCE_FLOOR)).encode()
).hexdigest()[:12]

_singleton_client: GeminiClient | None = None


@dataclass(frozen=True)
class NormalizedDecline:
    value: DeclineClass
    normalizer_version: str
    model_id: str
    raw_sha256: str
    # The model's own self-reported confidence, kept alongside `value`
    # rather than only consumed and discarded -- an auditable verdict must
    # be able to show WHY, not just what (payments-domain review,
    # 2026-08-31: "anything that cannot be replayed cannot be disputed").
    # Persisted to normalized_decline.confidence (src/ledger/schema.sql).
    confidence: float


def normalize(raw: str, *, client: GeminiLike | None = None) -> NormalizedDecline:
    """Normalize a decline string to the taxonomy.

    Args:
        raw: The issuer's decline string
        client: GeminiLike double (tests inject a fake). Defaults to a
               lazily-constructed module-level singleton.

    Returns:
        NormalizedDecline with the mapped class, version audit info, and
        hash of the raw input.

    Raises:
        ValueError: if the model returns an invalid DeclineClass value
    """
    global _singleton_client

    if client is None:
        if _singleton_client is None:
            _singleton_client = GeminiClient()
        client = _singleton_client

    # Call the model with forced function calling
    result = client.forced_call(
        model=NORMALIZER_MODEL,
        system=NORMALIZER_SYSTEM_PROMPT,
        user=raw,
        tool_name=NORMALIZE_TOOL["name"],
        tool_schema=NORMALIZE_TOOL,
        temperature=0.0,
    )

    # Parse and validate the result
    decline_class_str = result["decline_class"]
    decline_class = DeclineClass(decline_class_str)  # Raises ValueError if invalid

    # The prompt asks the model to self-report low confidence as UNKNOWN, but
    # a prompted instruction is not a guarantee (this project's whole thesis
    # is not trusting a model to hold a line unassisted) -- enforced here
    # too, so a confidently-worded but low-confidence answer cannot silently
    # pass as a real classification.
    #
    # Default on a MISSING confidence is 0.0, not 1.0: confidence is a
    # REQUIRED field in NORMALIZE_TOOL's schema, so its absence under forced
    # tool-use is a genuine provider anomaly, not an ordinary case -- and
    # client.py's own discipline is to treat an anomaly as an error, never
    # to coerce it into the MOST PERMISSIVE plausible value. Defaulting to
    # 0.0 forces UNKNOWN on that anomaly instead of silently trusting an
    # answer nothing actually vouched for (payments-domain review,
    # 2026-08-31: the original 1.0 default was "the wrong direction").
    confidence = float(result.get("confidence", 0.0))
    if confidence < _CONFIDENCE_FLOOR:
        decline_class = DeclineClass.UNKNOWN

    raw_sha256 = hashlib.sha256(raw.encode()).hexdigest()

    return NormalizedDecline(
        value=decline_class,
        normalizer_version=NORMALIZER_VERSION,
        model_id=NORMALIZER_MODEL,
        raw_sha256=raw_sha256,
        confidence=confidence,
    )
