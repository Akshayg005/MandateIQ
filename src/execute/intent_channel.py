"""Exit-intent score -> declared likelihood ratio over the three causes.

R5 (reports/gates.md, "Post-B16 remediation gates"; DECISIONS.md,
2026-09-04, R0). `src/llm/intent.py::intent_score()` has existed and been
tested since B11, and its only consumer was the golden set -- a real,
tested component with no path into the system it was built for. This module
is that path, and it is deliberately the thinnest one that can exist.

=== Why an adapter, and why it lives HERE ==================================

`scripts/guard_invariants.py`'s SRC_LLM_IMPORT forbids `src/policy/` from
importing `src.llm` in any form. That rule is correct and is not being
worked around: generative models do not make money decisions (invariant 1).
So the intent score crosses the boundary as a PLAIN FLOAT, and the mapping
from that float to a likelihood ratio happens in `src/execute/` -- the
layer already permitted to touch both sides, and the layer that already
carries every other "read a real-world signal, hand the core a value"
responsibility in this codebase.

`src/policy/belief.py::update_from_likelihood_ratio()` then applies Bayes'
rule to whatever ratio it is handed. It has no LLM knowledge, no Outcome
knowledge and no DeclineClass knowledge -- exactly the separation that
already keeps the Outcome->distribution mapping in observe_terminal()'s
callers rather than in observe_terminal().

**This module itself never calls an LLM.** It imports nothing from
`src.llm` and holds no client. It is a pure function of a float. The caller
obtains the score however it obtains it (a live `intent_score()` call in
production, a synthesised draw in the eval sweep) and passes it in.

=== The operating point is DECLARED, not fitted ============================

There is no labelled corpus of (support message, true latent cause) pairs
in this project, and there is no honest way to manufacture one: `Cause` is
LATENT and has no production label, ever (DECISIONS.md, 2026-08-28) -- the
same fact that forced B5's hazards to be marginal over cause rather than
cause-conditioned. A FITTED operating point would therefore be fitted
against something other than the truth it claims to measure.

So the operating point below is a DECLARED modelling choice, written down
in one place, version-stamped, and deliberately conservative. It is not a
measurement and this file does not present it as one. Its numbers are
policy, in the same sense `config/policy_costs.yaml`'s tuning parameters
are policy, and they are disclosed the same way.

The declared model is the simplest one that can be stated and checked: the
intent channel is a BINARY test for WONT_PAY, with sensitivity `tpr` and
false-positive rate `fpr`, and the score is thresholded at `threshold`. For
a positive reading, P(evidence | WONT_PAY) = tpr and P(evidence | not
WONT_PAY) = fpr; for a negative reading, (1 - tpr) and (1 - fpr). Those
four numbers ARE the likelihood, so the ratio needs no further scaling --
`update_from_likelihood_ratio()` renormalises anyway.

Two consequences, both intended:

* **The two non-WONT_PAY causes get identical entries.** This channel reads
  exit intent; it carries no information about which of CANT_PAY_NOW and
  CANT_PAY_EVER is at work. Inventing an asymmetry here would be
  fabricating evidence, which is the exact failure `src/policy/allocator.py`
  refuses when it declines to update belief on invented survival evidence.

* **One observation cannot reach a singleton.** At the declared point a
  single positive reading moves a uniform prior to ~0.62 on WONT_PAY, well
  short of anything `ConformalCauseGate` would fire on alone. That is the
  same posture `src/classify/cause_map.py`'s CUSTOMER_DECLINED row takes
  and for the same reason: the off-ramp is the one action a false positive
  cannot walk back.

=== What this is NOT ======================================================

Wiring this module into R4's live `src/execute/cycle.py` decision path is
NOT part of R5. The gate asks for the off-ramp to be reachable and
measured, not deployed. Today's callers are `eval/offramp_channel.py`'s
sweep (through a SYNTHESISED score, disclosed as synthetic everywhere it is
reported) and this module's own tests. A production caller needs the
support-ticket ingestion path that does not exist yet, and would need the
ledger round-trip `belief.update()`'s own docstring already names as a
caller obligation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.types import Cause


@dataclass(frozen=True)
class IntentOperatingPoint:
    """A declared binary-test operating point for the exit-intent channel.

    threshold: scores >= this read POSITIVE for WONT_PAY.
    tpr: declared P(positive | WONT_PAY).
    fpr: declared P(positive | not WONT_PAY).
    """

    threshold: float
    tpr: float
    fpr: float

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must lie strictly in (0, 1); got {self.threshold}")
        if not 0.0 < self.fpr < self.tpr < 1.0:
            raise ValueError(
                f"need 0 < fpr < tpr < 1 for the channel to be informative and "
                f"non-degenerate; got tpr={self.tpr}, fpr={self.fpr}"
            )


# DECLARED, 2026-09-05, before any sweep ran -- see the module docstring for
# why this cannot be fitted. 0.70 mirrors `src/llm/intent.py`'s own system
# prompt, which already tells the model "0.7+ should be rare, and never
# triggered by financial-hardship language alone"; using a different
# threshold here would silently disagree with the instruction the score was
# produced under. 0.65/0.20 is a deliberately UNIMPRESSIVE point (AUC 0.725
# as a two-point ROC): a channel this project would be embarrassed to claim
# is good, chosen so no headline number can rest on an assumed-excellent
# signal. The eval sweep additionally varies the channel's TRUE quality
# independently of this declared point, so the adapter is measured while
# MISSPECIFIED -- which is the realistic case and the whole reason the
# sweep exists.
INTENT_OPERATING_POINT = IntentOperatingPoint(threshold=0.70, tpr=0.65, fpr=0.20)

# Stamped into `Belief.provenance` by every caller. Deliberately distinct
# from `decline_taxonomy.TAXONOMY_VERSION` (no taxonomy ran), from
# `PROXY_SOURCE_VERSION` (a different fabricated channel) and from
# `TERMINAL_OBSERVATION_SOURCE_VERSION` (an observed ledger fact): a reader
# grepping the ledger must be able to tell which KIND of evidence produced
# a given belief. Bump this whenever INTENT_OPERATING_POINT changes -- a
# changed operating point under an unchanged version makes a persisted
# belief untraceable, the same gap `PRIOR_VERSION`/`TAXONOMY_VERSION` exist
# to close for their own tables.
INTENT_CHANNEL_SOURCE_VERSION = "intent-channel-op-v1"


def likelihood_ratio_from_intent_score(
    score: float, *, operating_point: IntentOperatingPoint = INTENT_OPERATING_POINT,
) -> dict[Cause, float]:
    """P(this reading | cause) for each of the three causes, under
    `operating_point`'s DECLARED binary-test model. Feed the result to
    `src.policy.belief.update_from_likelihood_ratio()` with
    INTENT_CHANNEL_SOURCE_VERSION.

    Raises ValueError on a score outside [0, 1] or a NaN rather than
    clamping. `src/llm/intent.py::intent_score()` already clamps
    defensively at its own boundary, so an out-of-range value arriving here
    means something OTHER than intent_score() produced it -- exactly when a
    silent clamp would hide the bug instead of surfacing it.
    """
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"intent score must be a number in [0.0, 1.0]; got {score!r}. "
            "src/llm/intent.py clamps its own output, so a value outside the "
            "range means this score did not come from there -- raising rather "
            "than clamping so that stays visible."
        )
    positive = score >= operating_point.threshold
    p_wont_pay = operating_point.tpr if positive else 1.0 - operating_point.tpr
    p_other = operating_point.fpr if positive else 1.0 - operating_point.fpr
    return {
        Cause.WONT_PAY: p_wont_pay,
        Cause.CANT_PAY_NOW: p_other,
        Cause.CANT_PAY_EVER: p_other,
    }
