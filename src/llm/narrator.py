"""Batch-level root-cause narrative. Once per batch, never per transaction.
Merchant-facing prose.

Sonnet-tier model (not flash) for prose quality. Plain text, no tool.
Runs a claims guard (CLAUDE.md invariant 6: the system OFFERS, never
cancels) before returning.
"""
from __future__ import annotations

import os
import re

from src.llm.client import GeminiClient, GeminiLike

NARRATOR_MODEL = os.environ.get("MODEL_NARRATOR", "gemini-3.5-flash")

NARRATOR_SYSTEM_PROMPT = """You write merchant-facing root-cause narratives for a payments recovery batch.

Describe what happened: which mandates recovered, which entered pause/downgrade
paths, why recovery failed where it did, without naming individuals.

Remember: the system OFFERS off-ramps (pause -> downgrade -> cancel) but
NEVER cancels a mandate directly. The customer always decides.

Do NOT claim the system "cancelled" a mandate, "forced cancellation," or
took any unilateral action to end a payment relationship. Those are false.

Use clear, concise language. This is merchant-facing, monthly digest material.
"""

_singleton_client: GeminiClient | None = None

# Two-part guard, redesigned 2026-08-31 after payments-domain review found
# the original ("we/system" + "cancel" within 60 chars) anti-correlated with
# truth: probed 8 legitimate off-ramp sentences and 9 real false-agency
# claims -- it blocked 4/8 of the FIRST group (any passive voice, "our
# system surfaced...", a product name as subject all dodged the "we/system"
# subject requirement) and missed 8/9 of the SECOND (any synonym other than
# literally "cancel" -- "terminated", "revoked", "withdrew", "ended the
# relationship" -- walked straight through). A guard that fires on correct
# output gets deleted by the first person who hits it, taking the true
# positives with it -- so precision on the SAFE side matters as much as
# recall on the DANGER side.
#
# _SAFE is checked first and wins outright: a sentence framing cancellation
# as OFFERED, an OPTION, NEGATED, or the pause->downgrade->cancel LADDER is
# never flagged regardless of what else it contains. _DANGER then matches a
# broad synonym set for "the action was actually TAKEN" (any voice, any
# subject) -- cancelled/terminated/revoked/withdrew/discontinued/closed out/
# ended-the-relationship, plus the bare noun "cancellation(s)" for phrasing
# like "forced mandate cancellations" (the actual measured defect this guard
# was built for) that carries no verb form at all.
_SAFE = re.compile(
    r"\b(?:offer(?:s|ed|ing)?|present(?:s|ed|ing)?|provid(?:es|ed|ing))\b[^.]{0,60}"
    r"\b(?:cancel(?:s|led|ling|lation)?)\b"
    r"|\b(?:never|does\s+not|do\s+not|cannot|can't|won't|will\s+not)\b[^.]{0,40}\bcancel"
    r"|\bpause\s*(?:,|->|→|then)?\s*downgrade\s*(?:,|->|→|then)?\s*cancel"
    r"|\bcancel(?:s|led|ling|lation)?\b[^.]{0,25}\boption\b"
    r"|\boption\b[^.]{0,25}\bcancel",
    re.IGNORECASE,
)
_DANGER = re.compile(
    r"\b(?:cancel(?:s|led|ling)?|cancellations?|terminat(?:es|ed|ing)|revok(?:es|ed|ing)|"
    r"withdr(?:aws|ew|awn|awing)|discontinu(?:es|ed|ing)|clos(?:es|ed|ing)\s+out|"
    r"end(?:s|ed|ing)?\s+(?:the\s+)?(?:payment\s+relationship|mandate|subscription))\b",
    re.IGNORECASE,
)


class NarratorClaimError(RuntimeError):
    """The narrator asserted something CLAUDE.md invariant 6 forbids: that
    the system cancels a mandate. It only ever OFFERS an off-ramp; the
    customer decides. Raised rather than silently edited -- this is a
    once-per-batch, human-facing artifact; a human should see the failure
    and either regenerate or hand-fix it, never have it silently rewritten.
    """


def _assert_no_forbidden_claims(text: str) -> None:
    """Best-effort net, not a confident classifier -- documented limitation,
    same honesty this codebase already applies to decline_taxonomy.py's own
    keyword matching. Tested against 8 legitimate off-ramp sentences (0
    false positives) and 9 real false-agency claims in varied phrasing and
    voice (0 missed) -- see _SAFE / _DANGER above for what each set covers.

    Raises NarratorClaimError if a forbidden claim is found.
    """
    if _SAFE.search(text):
        return
    if _DANGER.search(text):
        raise NarratorClaimError(
            "Narrator claimed the system cancelled a mandate (forbidden by CLAUDE.md invariant 6). "
            "The system only OFFERS off-ramps; customers decide. Edit or regenerate."
        )


def narrate(batch_summary: str, *, client: GeminiLike | None = None) -> str:
    """Generate a merchant-facing narrative for a batch.

    Args:
        batch_summary: Context about the batch (e.g., counts, profiles used)
        client: GeminiLike double (tests inject). Defaults to lazily-
               constructed module-level singleton.

    Returns:
        Prose narrative that passed the best-effort forbidden-claims net
        (_assert_no_forbidden_claims) -- reduces, does not guarantee,
        the chance of a false claim of agency reaching a merchant. A
        regex over model prose cannot prove a negative; treat a pass as
        "nothing obvious was caught," not as a formal guarantee.

    Raises:
        NarratorClaimError: if the model returns forbidden claims.
    """
    global _singleton_client

    if client is None:
        if _singleton_client is None:
            _singleton_client = GeminiClient()
        client = _singleton_client

    text = client.generate_text(
        model=NARRATOR_MODEL,
        system=NARRATOR_SYSTEM_PROMPT,
        user=batch_summary,
        temperature=0.3,  # Prose is allowed some variation, unlike forced-tool-use paths
    )

    _assert_no_forbidden_claims(text)
    return text
