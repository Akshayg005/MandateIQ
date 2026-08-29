"""Posterior distribution over the three latent causes (CANT_PAY_NOW,
CANT_PAY_EVER, WONT_PAY), updated by Bayes' rule as decline observations
arrive. This is the belief half of the state PLAN_DETAIL.md section 4's
backward induction searches over: `(b, r, ctx)`.

The likelihood P(decline | cause) does not exist anywhere in this codebase
as a fitted quantity -- src/classify/cause_map.prior() gives the OPPOSITE
direction, P(cause | decline), a hand-authored posterior that is exact only
"given a flat prior over causes" (its own docstring's words). This module
inverts it explicitly rather than reusing it as-is:

    likelihood(dc)[c] = cause_map.prior(dc)[c] / REFERENCE_PRIOR[c]

deliberately left UNNORMALISED -- any factor depending only on `dc` cancels
inside Bayes' rule, so normalising here would be pure busywork. What this
buys: the identity below holds for ANY REFERENCE_PRIOR, which is exactly
what breaks the moment REFERENCE_PRIOR and likelihood() drift apart --

    update(init(REFERENCE_PRIOR-as-a-dict), dc) == cause_map.prior(dc)

-- see tests/policy/test_belief.py's round-trip test, the load-bearing one
in this module's suite.

cause_map.py's docstring previously read "nothing downstream of B5 should
still be reading this file" -- written about the OUTCOME HAZARDS cause_map
was superseded by, and it now contradicts PLAN_DETAIL.md section 4:999,
which names cause_map.prior() as exactly this update's likelihood source.
Narrowed, not deleted: see cause_map.py's current docstring and
DECISIONS.md, 2026-08-29, B7. The prior() read in likelihood() below is the
one permitted exception that narrowing carves out.

update() is PURE STATIC BAYES -- b[c] * likelihood(dc)[c], renormalised.
No damping, no tempering, no cause-switch leak, not even as a parameter
that defaults to off. This is measurably overconfident relative to
eval/frozen/sim_config.yaml's cause_switch_prob (the misspecified arm's
per-attempt cause-drift rate) -- three identical declines drive the belief
to ~99.6% confidence in one cause, while the probability the cause even
stayed the same across those three attempts is only ~61.4%. Measured and
disclosed in tests/policy/test_belief.py's characterisation tests and
DECISIONS.md, 2026-08-29, B7 -- not damped away. A dial that exists gets
turned on using results from the arm it was built to fix; disclosing the
gap and bringing it back as a finding is the discipline this project has
applied at B5 (the stopping-threshold scalar) and is applying again here.

Belief is frozen and hashable -- B8 memoises backward induction on
`(quantised(b, 1e-6), r, ctx.signature())` (PLAN_DETAIL.md:1022), so this
module must produce a stable, hashable key from a Belief. Zero I/O in this
file. Probabilities are plain float, not integer paise -- a probability is
not money (the same reasoning cause_map.py's docstring already gives).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from src.classify.cause_map import PRIOR_VERSION, prior as _cause_map_prior
from src.core.types import Cause, DeclineClass

# Fixed positional order every tuple-shaped Belief quantity uses. An
# ordering bug here would silently mislabel causes in every consumer.
CAUSE_ORDER: tuple[Cause, Cause, Cause] = (
    Cause.CANT_PAY_NOW,
    Cause.CANT_PAY_EVER,
    Cause.WONT_PAY,
)

# The prior the likelihood inversion assumes. cause_map.prior()'s own
# docstring admits its numbers are exact only "given a flat prior over
# causes" -- this constant names that assumption instead of leaving it
# implicit. Uniform today; if it ever changes, bump REFERENCE_PRIOR_VERSION
# so a belief's provenance records which convention produced it.
REFERENCE_PRIOR: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
REFERENCE_PRIOR_VERSION = "ref-v1"

_SUM_TOL = 1e-9


class BeliefError(ValueError):
    """A prior or update input cannot form a valid distribution over the
    three causes. Never a bare assert -- assert is stripped under
    python -O, and this module's entire product is a guarantee about what
    a Belief can contain."""


@dataclass(frozen=True)
class Belief:
    """probs: the three cause probabilities, positionally indexed by
    CAUSE_ORDER. provenance: which normaliser and reference-prior versions
    produced this belief (cause_map.PRIOR_VERSION plus
    REFERENCE_PRIOR_VERSION) -- what makes a belief written to
    plan.belief_json traceable, per PLAN_DETAIL.md section 8.1's B11 gate:
    "a belief that cannot be traced to a specific normaliser version is not
    auditable"."""

    probs: tuple[float, float, float]
    provenance: str

    def __getitem__(self, cause: Cause) -> float:
        return self.probs[CAUSE_ORDER.index(cause)]

    def as_dict(self) -> dict[Cause, float]:
        """All three cause probabilities as a fresh dict, CAUSE_ORDER keys."""
        return {c: self.probs[i] for i, c in enumerate(CAUSE_ORDER)}

    def dominant(self) -> Cause:
        """The cause with the highest posterior probability."""
        best = max(range(len(CAUSE_ORDER)), key=lambda i: self.probs[i])
        return CAUSE_ORDER[best]

    def to_json(self) -> str:
        """JSON string for plan.belief_json -- each cause's plain string
        value mapped to its probability, plus provenance."""
        payload: dict[str, object] = {
            c.value: self.probs[i] for i, c in enumerate(CAUSE_ORDER)
        }
        payload["provenance"] = self.provenance
        return json.dumps(payload)


def _provenance() -> str:
    return f"cause_map={PRIOR_VERSION};reference_prior={REFERENCE_PRIOR_VERSION}"


def init(prior: Mapping[Cause, float]) -> Belief:
    """A Belief from a starting distribution over the three causes -- either
    the flat REFERENCE_PRIOR, or a previously observed posterior (e.g.
    cause_map.prior(dc) used as a later mandate-cycle's starting point,
    exactly the case the explicit likelihood inversion exists to make
    correct). Raises BeliefError if `prior` does not name every Cause
    exactly once, contains a negative probability, or does not sum to 1."""
    if set(prior.keys()) != set(Cause):
        raise BeliefError(
            f"prior must name every Cause exactly once; got keys "
            f"{set(prior.keys())!r}, expected {set(Cause)!r}"
        )
    probs = tuple(float(prior[c]) for c in CAUSE_ORDER)
    if any(p < 0.0 for p in probs):
        raise BeliefError(f"prior contains a negative probability: {probs}")
    total = sum(probs)
    if abs(total - 1.0) > _SUM_TOL:
        raise BeliefError(f"prior must sum to 1.0, got {total} from {probs}")
    return Belief(probs=probs, provenance=_provenance())


def likelihood(dc: DeclineClass) -> tuple[float, float, float]:
    """P(dc | cause), up to a dc-only normalising factor -- the explicit
    inversion of cause_map.prior(dc) (which is P(cause | dc)) through
    REFERENCE_PRIOR. Deliberately unnormalised: update() renormalises after
    multiplying, so a constant depending only on `dc` cancels there and
    carrying it here would be pure busywork."""
    p = _cause_map_prior(dc)
    return tuple(p[c] / REFERENCE_PRIOR[i] for i, c in enumerate(CAUSE_ORDER))


def update(b: Belief, obs: DeclineClass) -> Belief:
    """Bayes' rule on an observed DeclineClass: update(b, obs)[c] is
    proportional to b[c] * likelihood(obs)[c], then renormalised. Pure
    static Bayes -- no damping, no tempering, no cause-switch leak. See this
    module's docstring and DECISIONS.md, 2026-08-29, B7 for why the
    resulting overconfidence is disclosed, not mitigated."""
    lik = likelihood(obs)
    unnorm = tuple(b.probs[i] * lik[i] for i in range(len(CAUSE_ORDER)))
    total = sum(unnorm)
    if total <= 0.0:
        raise BeliefError(
            f"update({b.probs}, {obs}) leaves no support: every cause obs "
            "is consistent with already had zero prior mass"
        )
    probs = tuple(u / total for u in unnorm)
    return Belief(probs=probs, provenance=_provenance())


def quantised(b: Belief, step: float) -> tuple[int, int, int]:
    """b's three probabilities, each rounded to the nearest multiple of
    `step` and expressed as an integer count of steps -- a stable, hashable
    key that collapses near-identical beliefs together. This is what B8's
    backward-induction memoisation keys on (PLAN_DETAIL.md:1022):
    `(quantised(b, 1e-6), r, ctx.signature())`."""
    return tuple(round(p / step) for p in b.probs)
