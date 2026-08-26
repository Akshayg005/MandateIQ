"""The incumbent: fixed T+1/T+2/T+3 retries, same amount, then halt.
Razorpay's documented behaviour -- the baseline this whole project measures
against, not a strawman. Deliberately simple: it consults no model, no
belief state, nothing under src/policy/, and does not adapt to either
compliance profile (see eval/frozen/protocol.md, "The ladder, precisely").
"""
from __future__ import annotations

from src.core.types import Outcome, Profile
from eval.frozen.scoring import BatchResult, aggregate, score_mandate
from eval.frozen.simulator import Simulator


def run(sim: Simulator, profile: Profile) -> BatchResult:
    """Run the fixed-cadence ladder over every mandate in `sim`. `profile`
    only labels the resulting BatchResult -- the ladder's behaviour does not
    depend on it, by design."""
    offsets = sim.config["baseline_ladder_offsets_days"]
    results = []
    for mandate in sim.mandates:
        attempts = []
        for slot in (2, 3, 4):
            attempt = sim.attempt(mandate.mandate_id, slot, offsets[slot])
            attempts.append(attempt)
            if attempt.outcome != Outcome.STILL_PENDING:
                break
        results.append(score_mandate(mandate, attempts))
    return aggregate(results, arm=sim.arm, profile=profile.value)
