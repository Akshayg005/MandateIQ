"""DeclineClass -> prior probability distribution over the three latent
causes. Doubles as the P(e|c) likelihood term the allocator's belief update
will consume (PLAN_DETAIL.md Sec 4: `update(b,e)[c] ~ b[c]*P(e|c)`, comment
"P(e|c) from cause_map.prior()") -- exact under Bayes' rule only given a
flat prior over causes, an explicit, disclosed simplification appropriate
for a hand-authored starting point. B5/B6 supersede this with fitted
hazards; nothing downstream of B5 should still be reading this file.

Every distribution sums to 1.0 and names all three Cause members, even
where one is near-zero -- a partial dict here would be silently
mis-normalised by anything that sums it. Values are plain Python float:
the one place in this codebase floats are correct, because a probability
is not money.

These numbers are a provisional, hand-authored starting point (payments-
domain's review is expected to push on them), not a fitted estimate.

UNKNOWN and ISSUER_DECLINE are skewed toward CANT_PAY_NOW, not uniform,
per .claude/skills/new-failure-class/SKILL.md's explicit policy for a
genuinely ambiguous class: "map it to CANT_PAY_NOW (the safe default: we
retry rather than offer an exit)". This corrects an earlier version of this
file that gave both classes exactly/near-uniform mass, following a
different (also-defensible-sounding, but wrong for THIS project) instinct
that "no signal" should mean "no opinion" -- found by payments-domain's B3
review, which pointed out the three causes are not symmetric in
consequence: CANT_PAY_NOW costs a retry slot (cheap, reversible), WONT_PAY
routes toward an off-ramp offer (not reversible in the same way), so
abstaining from a real class actually IS a bet, and the skill already
settled which side it should fall on.
"""
from __future__ import annotations

from src.core.types import Cause, DeclineClass

# Bumped by hand whenever the numbers below change meaningfully. Recorded
# per-row in ingested_event.prior_version (see schema.sql) -- see
# decline_taxonomy.TAXONOMY_VERSION's docstring for why this matters.
PRIOR_VERSION = "v2"

_PRIORS: dict[DeclineClass, dict[Cause, float]] = {
    # Transient liquidity gap -- retry, timed to replenishment rhythm.
    DeclineClass.INSUFFICIENT_FUNDS: {
        Cause.CANT_PAY_NOW: 0.80, Cause.CANT_PAY_EVER: 0.10, Cause.WONT_PAY: 0.10,
    },
    # Dead instrument -- stop retrying, request re-authorisation.
    DeclineClass.CARD_EXPIRED: {
        Cause.CANT_PAY_EVER: 0.75, Cause.CANT_PAY_NOW: 0.15, Cause.WONT_PAY: 0.10,
    },
    DeclineClass.ACCOUNT_CLOSED: {
        Cause.CANT_PAY_EVER: 0.70, Cause.CANT_PAY_NOW: 0.15, Cause.WONT_PAY: 0.15,
    },
    # Revocation is often deliberate -- split across "instrument genuinely
    # dead" and "customer chose to leave", never read as a liquidity gap.
    DeclineClass.MANDATE_REVOKED: {
        Cause.CANT_PAY_EVER: 0.45, Cause.WONT_PAY: 0.45, Cause.CANT_PAY_NOW: 0.10,
    },
    # Genuinely ambiguous -- a generic issuer/gateway decline carries little
    # cause information on its own. Skewed to the safe default rather than
    # uniform: see module docstring.
    DeclineClass.ISSUER_DECLINE: {
        Cause.CANT_PAY_NOW: 0.60, Cause.CANT_PAY_EVER: 0.20, Cause.WONT_PAY: 0.20,
    },
    # Something didn't complete in time -- closer to a transient
    # availability gap than evidence of intent or a dead instrument.
    DeclineClass.BANK_TIMEOUT: {
        Cause.CANT_PAY_NOW: 0.70, Cause.CANT_PAY_EVER: 0.15, Cause.WONT_PAY: 0.15,
    },
    # Unclassifiable -- skewed to the safe default (see module docstring),
    # not uniform: "no signal" is not "no opinion" when the three actions
    # this feeds have asymmetric downside.
    DeclineClass.UNKNOWN: {
        Cause.CANT_PAY_NOW: 0.60, Cause.CANT_PAY_EVER: 0.20, Cause.WONT_PAY: 0.20,
    },
}


def prior(dc: DeclineClass) -> dict[Cause, float]:
    """Provisional starting distribution over the three causes for a given
    DeclineClass. Every DeclineClass member has an entry; every entry names
    all three Cause members and sums to 1.0. Returns a fresh dict each call
    so a caller mutating its result can never corrupt the shared table."""
    return dict(_PRIORS[dc])
