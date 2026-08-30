"""Exit-intent extraction from support messages. Weakly identified; gated by
conformal prediction in src/policy/gate.py (singleton {WONT_PAY} at 95% coverage).

Returns a score in [0.0, 1.0], clamped defensively. This module NEVER emits
a decision -- that is src/policy/gate.py's job. It returns a number that
feeds into a gate, not an Action.
"""
from __future__ import annotations

import hashlib
import os

from src.llm.client import GeminiClient, GeminiLike
from src.llm.tools import INTENT_TOOL

INTENT_MODEL = os.environ.get("MODEL_INTENT", "gemini-3.5-flash-lite")

INTENT_SYSTEM_PROMPT = """You are an exit-intent extractor for a payments recovery system.

Your task: analyze customer support text (possibly Hinglish) to assess the
likelihood the customer WONT_PAY -- wants the subscription to end -- as
opposed to CANT_PAY_NOW, a transient liquidity gap where the customer still
wants the service and will pay once money is available. These are DIFFERENT
latent causes with different correct actions (offer an exit vs. simply
retry later), and this system's entire purpose is telling them apart.

Return a score from 0.0 (definitely wants to keep paying -- includes
"can't afford it right now") to 1.0 (definitely wants to cancel/opt out).
Include brief rationale.

CRITICAL, the most common mistake: liquidity language is NOT exit intent.
"Money is tight right now", "salary hasn't come yet", "will pay once I get
paid", "can you retry in a few days" -- these describe CANT_PAY_NOW and
must score LOW (near 0.0), even though the customer is, in some sense,
currently declining to pay. Score HIGH only when the customer wants the
SERVICE itself to stop: explicit cancellation requests, "don't charge me
again", "I don't want this anymore", repeated refusal framed as rejecting
the product rather than lacking funds this cycle.

This score feeds a conformal prediction gate that decides whether to OFFER
an off-ramp (pause -> downgrade -> cancel, customer decides). A false
positive here -- scoring a liquidity-gap message as exit intent -- risks
offering an off-ramp to a customer who was always going to pay, which is
the exact harm this system exists to prevent. Be conservative: 0.7+ should
be rare, and never triggered by financial-hardship language alone.
"""

INTENT_VERSION = hashlib.sha256((INTENT_SYSTEM_PROMPT + str(INTENT_TOOL)).encode()).hexdigest()[
    :12
]

_singleton_client: GeminiClient | None = None


def intent_score(text: str, *, client: GeminiLike | None = None) -> float:
    """Score a support message for exit intent.

    Args:
        text: Support message, possibly Hinglish
        client: GeminiLike double (tests inject). Defaults to lazily-
               constructed module-level singleton.

    Returns:
        Float in [0.0, 1.0], clamped defensively.
    """
    global _singleton_client

    if client is None:
        if _singleton_client is None:
            _singleton_client = GeminiClient()
        client = _singleton_client

    result = client.forced_call(
        model=INTENT_MODEL,
        system=INTENT_SYSTEM_PROMPT,
        user=text,
        tool_name=INTENT_TOOL["name"],
        tool_schema=INTENT_TOOL,
        temperature=0.0,
    )

    score = float(result["intent_score"])
    # Defensive clamp
    score = max(0.0, min(1.0, score))

    return score
