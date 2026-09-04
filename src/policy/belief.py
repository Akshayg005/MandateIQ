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
    CAUSE_ORDER. provenance: which classifier and reference-prior versions
    produced this belief -- cause_map.PRIOR_VERSION and
    REFERENCE_PRIOR_VERSION always; from update(), also the required
    source_version identifying which classifier (deterministic taxonomy or
    LLM normaliser, and which version of it) produced the DeclineClass
    observation, as `;source=<source_version>` -- what makes a belief
    written to plan.belief_json traceable, per PLAN_DETAIL.md section 8.1's
    B11 gate: "a belief that cannot be traced to a specific normaliser
    version is not auditable". init() never has an observation to trace,
    so its provenance omits the source field entirely."""

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


def _validate_source_version(source_version: str) -> None:
    if not source_version:
        raise BeliefError(
            "source_version is required and must be non-empty -- a Belief "
            "updated from an observation whose classifier cannot be named is "
            "not auditable (PLAN_DETAIL.md B11 gate, clause 3). Pass "
            "decline_taxonomy.TAXONOMY_VERSION for the deterministic path, or "
            "a NormalizedDecline.normalizer_version read back from the "
            "ledger's normalized_decline row for the LLM path."
        )


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


def update(b: Belief, obs: DeclineClass, *, source_version: str) -> Belief:
    """Bayes' rule on an observed DeclineClass: update(b, obs)[c] is
    proportional to b[c] * likelihood(obs)[c], then renormalised. Pure
    static Bayes -- no damping, no tempering, no cause-switch leak. See this
    module's docstring and DECISIONS.md, 2026-08-29, B7 for why the
    resulting overconfidence is disclosed, not mitigated.

    source_version is REQUIRED and keyword-only: which classifier produced
    `obs` -- decline_taxonomy.TAXONOMY_VERSION for the deterministic path,
    or a NormalizedDecline.normalizer_version (read back from the ledger's
    normalized_decline row, never passed in directly from src/llm/ -- this
    module must stay import-free of that package) for the LLM path. Before
    this parameter existed, a belief updated from an LLM-normalised decline
    and one updated from the deterministic taxonomy produced byte-identical
    provenance -- exactly the gap this Belief's own docstring already named
    as unacceptable ("a belief that cannot be traced to a specific
    normaliser version is not auditable", PLAN_DETAIL.md B11 gate). Required
    rather than defaulted so that gap cannot silently reopen: a caller
    cannot construct a belief without stating where its evidence came from.

    What this DOES NOT prove (payments-domain review, 2026-08-31): that the
    string is honest. `update(b, nd.value, source_version=nd.normalizer_version)`
    on an in-memory NormalizedDecline that was never actually written to
    `normalized_decline` satisfies this signature while defeating the whole
    point -- the ledger round-trip is a documented caller obligation, not a
    structural one enforced by a type here. Closing that gap needs the
    observation and its provenance to arrive as ONE object a ledger read
    produces (e.g. store.find_normalized_decline's return value, or an
    equivalent for the deterministic path), not two independent arguments a
    caller can source separately -- real design work for whichever block
    wires the executor to this function, deferred rather than guessed at
    now since update() has no production caller yet to prove the right
    shape against. See DECISIONS.md, 2026-08-31.
    """
    _validate_source_version(source_version)
    lik = likelihood(obs)
    unnorm = tuple(b.probs[i] * lik[i] for i in range(len(CAUSE_ORDER)))
    total = sum(unnorm)
    if total <= 0.0:
        raise BeliefError(
            f"update({b.probs}, {obs}) leaves no support: every cause obs "
            "is consistent with already had zero prior mass"
        )
    probs = tuple(u / total for u in unnorm)
    provenance = f"{_provenance()};source={source_version}"
    return Belief(probs=probs, provenance=provenance)


def observe_terminal(cause_probs: Mapping[Cause, float], *, source_version: str) -> Belief:
    """A Belief from a MEASURED posterior over the three causes, given an
    OBSERVED terminal outcome (DEAD or OPTED_OUT) -- conditioning on a
    ledger fact, not inferring from an ambiguous decline signal the way
    update() does. No PRIOR belief is accepted as a parameter (unlike
    update()) -- the terminal observation supersedes whatever came before,
    the same way init() needs no prior either.

    R2, 2026-09-04 (reports/gates.md, "Post-B16 remediation gates", R2a):
    before this function existed, the only belief-update path was update()
    with a DeclineClass likelihood, and after a couple of INSUFFICIENT_FUNDS
    updates the belief could sit at ~99% CANT_PAY_NOW -- one further,
    ordinary CARD_EXPIRED-shaped update (prior 0.75 toward CANT_PAY_EVER)
    could not move `b.dominant()` to CANT_PAY_EVER even once the instrument
    was actually confirmed dead, so allocator.py's REAUTH inference path
    (`_best_action`'s `elif b.dominant() == Cause.CANT_PAY_EVER`) was never
    entered and ATTEMPT kept winning by default.

    CORRECTED same day (stats-reviewer / payments-domain review, before this
    gate was ticked): the first version of this function took a single
    `cause` and returned a DEGENERATE (1.0/0/0) posterior, on the reasoning
    that "an observed DEAD outcome means CANT_PAY_EVER ... that is what the
    cause label MEANS, not a hypothesis about it." Checked against this
    project's own frozen generative process (`eval/frozen/sim_config.yaml`,
    the `nominal` arm) and found FALSE: direct 200-seed simulation measures
    P(CANT_PAY_EVER | DEAD) = 0.899 and P(WONT_PAY | OPTED_OUT) = 0.904 --
    roughly 10% of each terminal outcome has a DIFFERENT true cause, drawn
    by chance from a cause whose own dead/opt-out base rate is low, not
    zero (`CANT_PAY_NOW`/`WONT_PAY` both carry `base_dead: 0.02` against
    `CANT_PAY_EVER`'s `0.55`; `CANT_PAY_NOW`/`CANT_PAY_EVER` both carry a
    non-zero `base_optout`). A degenerate 1.0 was ALSO irreversible in a way
    nothing else in this module is: `cause_map._PRIORS` contains no zeros,
    so `update()` on an exact `(0, 1, 0)` belief returns that identical
    belief regardless of any later evidence (`0 * anything = 0`) -- an
    absorbing state no other belief in this system can reach, dormant today
    only because no mandate's belief survives past its own terminal outcome
    in this eval harness, but a real hazard for any future multi-cycle
    persistence (R4).

    `cause_probs` is therefore the CALLER's measured distribution, not an
    assumption this module makes -- see `eval/run.py`'s
    `_TERMINAL_OBSERVED_CAUSE_PROBS`, derived directly from the frozen
    simulator's own generative process and cited there. The Outcome-to-
    distribution mapping is the CALLER's decision, not this module's:
    belief.py stays free of any Outcome-shaped knowledge, the same
    separation that already keeps the Outcome->DeclineClass proxy in
    eval/allocator_sweep.py's `_proxy_decline_class()` rather than here.
    Validated exactly like `init()`'s prior (every Cause named once,
    non-negative, sums to 1.0) -- raises `BeliefError` on the same terms.

    This is NOT the "pure static Bayes, no damping" behaviour update()
    documents -- it never touches likelihood()/REFERENCE_PRIOR. It is
    still an inference under (now honestly quantified) uncertainty, not
    the certain, definitional collapse the first version claimed.

    source_version is REQUIRED and keyword-only, the same discipline as
    update(): which observation channel produced this measurement, so a
    belief built this way stays traceable to what produced it (B11's "a
    belief that cannot be traced to a specific normaliser version is not
    auditable" -- generalised here to "traced to a specific OBSERVATION
    source", the same requirement applied to a different evidence kind)."""
    if set(cause_probs.keys()) != set(Cause):
        raise BeliefError(
            f"cause_probs must name every Cause exactly once; got keys "
            f"{set(cause_probs.keys())!r}, expected {set(Cause)!r}"
        )
    probs = tuple(float(cause_probs[c]) for c in CAUSE_ORDER)
    if any(p < 0.0 for p in probs):
        raise BeliefError(f"cause_probs contains a negative probability: {probs}")
    total = sum(probs)
    if abs(total - 1.0) > _SUM_TOL:
        raise BeliefError(f"cause_probs must sum to 1.0, got {total} from {probs}")
    _validate_source_version(source_version)
    provenance = f"{_provenance()};source={source_version};observed=terminal"
    return Belief(probs=probs, provenance=provenance)


def quantised(b: Belief, step: float) -> tuple[int, int, int]:
    """b's three probabilities, each rounded to the nearest multiple of
    `step` and expressed as an integer count of steps -- a stable, hashable
    key that collapses near-identical beliefs together. This is what B8's
    backward-induction memoisation keys on (PLAN_DETAIL.md:1022):
    `(quantised(b, 1e-6), r, ctx.signature())`."""
    return tuple(round(p / step) for p in b.probs)
