"""B8's own gate-measurement harness: attempt rate and discrimination
margin (eval/gate_criteria.py) across seeds 0-19, both compliance profiles,
nominal arm -- matching tests/eval/test_gate_criteria.py's own convention
(`Simulator("nominal", seed=seed)`, AFA-eligible mandates only).

This is NOT eval/run.py -- that is B13's file (PLAN_DETAIL.md section 1,
B13 entry) and has never existed; .\\run.ps1's own `ci` comment explains
why it must stay that way until B13. This script is scoped to exactly what
B8's gate needs: a summary table, printed once, never a per-mandate log --
batch output must never enter the main Claude Code context (root
CLAUDE.md, "Keeping the window usable within a session"). Run via the
`eval-runner` subagent, per that same section.

=== The outcome -> DeclineClass proxy, and why it exists ===================

allocator.py's solve() is deliberately pure: it never calls
belief.update() (see that module's own docstring for why belief stays
fixed within one solve() call -- the cause-conditioned-hazard gap B7 left
open, resolved at B8 via action-gating only). Driving a full multi-slot
retry cycle, though -- which this harness must do, to measure attempt rate
and discrimination honestly across up to three retries -- needs
*something* to update belief BETWEEN solve() calls, the same way B9's real
executor will from a real issuer decline string.

eval/frozen/simulator.py's AttemptResult carries only `outcome: Outcome`
(STILL_PENDING / RECOVERED / DEAD / OPTED_OUT) -- no DeclineClass. There is
no decline-string layer in this simulator; B3/B11's normaliser sits
between a real Razorpay webhook and a DeclineClass, and the frozen
simulator was never built to produce one. Fabricating a DeclineClass from
a bare Outcome is therefore a real, disclosed simplification --
`_proxy_decline_class()` below -- confined to this eval harness, never to
allocator.py itself:

    DEAD          -> CARD_EXPIRED        (0.75 prior toward CANT_PAY_EVER --
                                           matches "instrument confirmed dead")
    STILL_PENDING -> INSUFFICIENT_FUNDS  (0.80 prior toward CANT_PAY_NOW --
                                           matches "failed this round, will
                                           retry")
    RECOVERED, OPTED_OUT -> None         (terminal; the cycle ends, no
                                           further belief update needed)

This is coarser than a real issuer decline string, but the underlying
generative separation is large enough to carry real information on its
own: eval/frozen/sim_config.yaml's own hazards give CANT_PAY_EVER a 0.55
base_dead rate against CANT_PAY_NOW's 0.02 -- roughly 27x -- so a single
observed DEAD outcome is already strong evidence before any DeclineClass-
level nuance is even applied. A real deployment never uses this proxy; B9
reads a genuine issuer string.

=== Fitting the hazard model =================================================

The sweep fits competing_risks on a corpus draw from eval.corpus's own
TRAIN_SEEDS (90001-90040) -- disjoint from this sweep's own seeds 0-19.
Fitting and scoring on the same seeds would be exactly the leakage
src.model.calibration/conformal's own assert_disjoint() exists to catch
elsewhere in this project; this script keeps the same discipline by
construction (a different seed range), not by a runtime check, because
scoring here is a live simulation loop, not a static held-out frame.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field

from eval import corpus
from eval.frozen.simulator import Simulator, _logits_from_base_rates, _softmax, load_config
from eval.gate_criteria import ATTEMPT_RATE_FLOOR, DISCRIMINATION_MARGIN, attempt_rate, discrimination_gap
from src.core.types import Action, Cause, DeclineClass, MandateState, Outcome, Profile
from src.execute.intent_channel import likelihood_ratio_from_intent_score
from src.model import competing_risks, features, person_period
from src.policy import belief as belief_mod
from src.policy.allocator import AllocatorError, solve
from src.policy.constraints import MAX_ATTEMPTS, afa_free_limit_paise
from src.policy.costs import PolicyCosts, load as load_costs
from src.policy.gate import FullSetGate
from src.policy.stopping_rules import AllocationContext

SEEDS = range(20)  # seeds 0-19 -- matches tests/eval/test_gate_criteria.py

# The provenance string every belief.update() in this harness is stamped
# with. Deliberately NOT decline_taxonomy.TAXONOMY_VERSION: no taxonomy runs
# here. Every DeclineClass in this file is FABRICATED -- either by
# _proxy_decline_class() from a bare Outcome, or by draw_slot1_decline() from
# the frozen simulator's own generative parameters -- and neither ever saw an
# issuer string. Stamping them as taxonomy output would be exactly the
# provenance lie B11 added source_version to make impossible
# (src/policy/belief.py:172, "a belief that cannot be traced to a specific
# normaliser version is not auditable"). A reader grepping the ledger for
# this string finds simulated evidence, not observed evidence.
PROXY_SOURCE_VERSION = "eval-allocator-sweep-proxy-v1"

# === R5: the synthetic WONT_PAY evidence channel ============================
#
# reports/gates.md, "Post-B16 remediation gates", R5. Everything in this
# section is FABRICATED and disclosed as such: it reads the simulator's
# PRIVILEGED true cause (`SimMandate.initial_cause`) and feeds the result
# into the DECISION path. That is a materially stronger claim than the
# score-only privileged read `false_reauth_count` already makes -- which is
# exactly why R5's gate requires the channel's own ROC to be published
# beside every number it produces, and why `eval/offramp_channel.py` sweeps
# channel QUALITY rather than asserting one.
#
# Why it exists. Before R5 this harness had a two-symbol decline alphabet
# (CARD_EXPIRED / INSUFFICIENT_FUNDS), whose WONT_PAY likelihood components
# are IDENTICAL (0.30 each) -- so the WONT_PAY likelihood ratio against the
# other causes is monotone non-increasing, and exhaustive enumeration over
# every sequence reachable within the NPCI cap gives max P(WONT_PAY) = 0.10
# (re-derived, not quoted, in tests/eval/test_wontpay_channel.py).
# `ConformalCauseGate` can never return the `{WONT_PAY}` singleton
# `allocator.py` fires on, so `n_offer` was 0 in all 256 engine cells and
# `false_offramp_count` was a structural zero rather than a measurement.
# The off-ramp -- the lane this entire project exists to defend -- was
# untested-and-central. R5 buys tested-and-imperfect, which is a weaker
# claim honestly made instead of a stronger one nobody checked.
#
# Two channels, per DECISIONS.md (2026-09-04, R0), because picking one
# would have been a worse answer either way:
#
#   "decline" -- emits DeclineClass.CUSTOMER_DECLINED (R5's new taxonomy
#                class, src/classify/), which belief.update() inverts
#                through src/classify/cause_map.py's hand-authored table.
#                Stays entirely inside the payments story: a real Razorpay
#                `payment_cancelled` event, given a class to land in.
#   "intent"   -- emits a SCORE, which src/execute/intent_channel.py maps
#                to a DECLARED likelihood ratio for
#                belief.update_from_likelihood_ratio(). This is the honest
#                real-world channel for exit intent and the only consumer
#                src/llm/intent.py has ever had outside the golden set --
#                but it needs a fabricated support-ticket signal in eval,
#                which is a bigger fabrication than a decline string. It is
#                therefore measured in the sweep and NOT folded into the
#                published headline grid.
#
# Both are MISSPECIFIED on purpose, in the same way the pre-existing slot-1
# signal already is: the channel's true (tpr, fpr) is a sweep parameter,
# while the allocator's inference runs through cause_map's independent
# hand-authored numbers (decline) or intent_channel's independently
# declared operating point (intent). The allocator can therefore still be
# wrong, in both directions.
#
# `channel=None` is the pre-R5 path, unchanged -- every number this project
# has already published came from it, and tests/eval/test_wontpay_channel.py
# pins that equivalence rather than trusting it.

# Provenance stamps. Deliberately NOT PROXY_SOURCE_VERSION and deliberately
# NOT a taxonomy version: a reader grepping the ledger for either string
# must be able to tell WHICH fabricated channel produced a belief, and
# stamping this as taxonomy output would be the provenance lie
# PROXY_SOURCE_VERSION's own comment above already refuses to make.
WONTPAY_CHANNEL_SOURCE_VERSION = "eval-wontpay-channel-v1"
INTENT_CHANNEL_SOURCE_VERSION = "eval-intent-channel-v1"

CHANNEL_KINDS: tuple[str, ...] = ("decline", "intent")

# The two score values the "intent" channel emits, straddling
# src/execute/intent_channel.py's DECLARED threshold. Two values, not a
# continuous draw: the channel's quality is fully described by (tpr, fpr),
# and adding score-magnitude structure would invent a second, unmeasurable
# dimension of realism on top of an already-fabricated signal. Its realised
# ROC is therefore exactly the two-point curve through (fpr, tpr), which is
# what gets published.
_INTENT_SCORE_POSITIVE = 0.90
_INTENT_SCORE_NEGATIVE = 0.10


@dataclass
class WontPayChannel:
    """A quality-parameterised, cause-aware synthetic evidence channel.

    kind: "decline" or "intent" -- see the section comment above.
    tpr:  P(this channel emits positive evidence | true cause is WONT_PAY).
    fpr:  P(this channel emits positive evidence | true cause is not).
    rng:  MUST be a stream independent of both the simulator's own RNG and
          the slot-1 decline stream, so switching the channel on cannot
          perturb the outcome draws or the slot-1 draws every previously
          reported number depends on.

    Mutable (the rng advances); never shared across cells.
    """

    kind: str
    tpr: float
    fpr: float
    rng: random.Random

    # R5 REVIEW PASS, 2026-09-05 (stats-reviewer, HIGH). The published grid
    # sweeps the MARGINAL (tpr, fpr) while `fires()` draws an independent
    # Bernoulli each call -- holding fixed, at exactly zero, the one axis
    # the singleton firing rule is actually sensitive to. A single
    # CUSTOMER_DECLINED observation moves belief to ~0.62 WONT_PAY (see
    # tests/eval/test_wontpay_channel.py); the fitted gate's own singleton
    # boundary sits around p(WONT_PAY) 0.80-0.90 REGARDLESS of alpha
    # (verified 2026-09-05: alpha 0.05/0.20/0.30/0.40 all fire the
    # singleton somewhere in that narrow band -- see the calibration-atom
    # finding in reports/gates.md's R5 entry), so it takes roughly TWO
    # coincident firings on the SAME mandate to open the off-ramp. Two
    # independent draws from a real customer's decline history is not a
    # safe assumption: a customer who dismisses one collect request is
    # measurably more likely to dismiss the next one, for reasons that have
    # nothing to do with wanting to leave (a bad app UX, a bill they have
    # forgotten about, one bad week).
    #
    # `habitual_fraction` is a SEPARATE sweep dimension for exactly this.
    # It holds the MARGINAL fpr fixed (proof below) while concentrating the
    # false-firing mass into a shrinking sub-population of mandates that,
    # once habitual, fire almost every time -- the standard two-point
    # mixture for inducing over-dispersion in a Bernoulli sequence without
    # moving its mean, so the channel's realised ROC stays comparable
    # across dependence levels.
    #
    # 1.0, the DEFAULT, is EXACTLY today's iid behaviour: `_effective_fpr()`
    # below reduces to a constant `fpr` at habitual_fraction=1.0, `fires()`
    # takes the identical branch it always took, and not one existing
    # published number moves. Values below 1.0 are additive, swept
    # separately (see eval/offramp_channel.py's `dependence_sweep()`), and
    # touch nothing in the main 1024-cell grid.
    habitual_fraction: float = 1.0
    _habitual: dict = field(default_factory=dict)

    # --- realised-ROC bookkeeping, filled in by fires() ------------------
    # R5's gate requires "the synthetic channel's own ROC published beside"
    # every result. These are the REALISED rates, counted from the draws
    # that actually happened, not the nominal (tpr, fpr) parameters -- a
    # channel that was configured at 0.60 and realised 0.57 on this cell's
    # 200 mandates must publish 0.57, or the ROC is a restatement of the
    # input rather than a measurement of the output.
    #
    # `log` carries one (mandate_id, is_wont_pay, fired) row per draw so
    # eval/offramp_channel.py can compute a MANDATE-level cluster bootstrap
    # CI -- draws are clustered (one mandate contributes up to four
    # decision points), so a naive row-level interval overstates precision,
    # the same reasoning bench/llm_vs_stats.py's cluster_bootstrap_ci()
    # already documents.
    mandate_id: str = ""
    n_wont_pay: int = 0
    n_positive_on_wont_pay: int = 0
    n_other: int = 0
    n_positive_on_other: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in CHANNEL_KINDS:
            raise ValueError(
                f"unknown channel kind {self.kind!r}; expected one of {CHANNEL_KINDS}"
            )
        for name, v in (("tpr", self.tpr), ("fpr", self.fpr)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {v}")
        if not 0.0 < self.habitual_fraction <= 1.0:
            raise ValueError(
                f"habitual_fraction must lie in (0, 1]; got {self.habitual_fraction}"
            )

    def _effective_fpr(self) -> float:
        """The per-observation false-fire rate for THIS mandate under the
        two-point habitual-dismisser mixture, persistent across calls for
        the same mandate_id (drawn once, from `self.rng`, so the mixture's
        own randomness shares the channel's single stream rather than
        opening a second one).

        At habitual_fraction=1.0 this returns `self.fpr` unconditionally --
        every mandate is "habitual" by construction, so the mixture
        degenerates to the plain iid case and this function is provably
        equivalent to the constant `self.fpr` `fires()` always used.

        At habitual_fraction=h<1.0: with probability h this mandate is
        "habitual" and fires at rate min(1, fpr/h) on every call; otherwise
        it never fires falsely. E[fire] = h * min(1, fpr/h) + (1-h)*0 = fpr
        EXACTLY whenever h >= fpr, so the sweep's own ROC stays comparable
        across dependence levels -- only WITHIN-mandate correlation moves.
        Below h < fpr the mixture caps at rate 1.0 and understates the true
        marginal; `WontPayChannel.realised()`'s own `fpr_realised` already
        surfaces that as a measured discrepancy, so it is not separately
        guarded here -- a caller sweeping h < fpr sees its own mistake in
        the artifact rather than a silent wrong number.
        """
        if self.habitual_fraction >= 1.0:
            return self.fpr
        if self.mandate_id not in self._habitual:
            self._habitual[self.mandate_id] = self.rng.random() < self.habitual_fraction
        if not self._habitual[self.mandate_id]:
            return 0.0
        return min(1.0, self.fpr / self.habitual_fraction)

    def fires(self, cause: Cause) -> bool:
        """Draw one observation. Reads the PRIVILEGED true cause -- see the
        section comment. Called once per decision point, so evidence
        accumulates across a mandate's retries exactly as repeated real
        observations would, and a false-positive channel accumulates FALSE
        evidence at the same rate.

        The habitual-dismisser mixture applies only to the FALSE-positive
        side (cause is not WONT_PAY) -- the reviewer's finding is
        specifically that repeat false firings threaten a paying customer;
        WONT_PAY's own tpr draw is unchanged in every case.
        """
        is_wont_pay = cause is Cause.WONT_PAY
        p = self.tpr if is_wont_pay else self._effective_fpr()
        fired = self.rng.random() < p
        if is_wont_pay:
            self.n_wont_pay += 1
            self.n_positive_on_wont_pay += int(fired)
        else:
            self.n_other += 1
            self.n_positive_on_other += int(fired)
        self.log.append((self.mandate_id, is_wont_pay, fired))
        return fired

    def for_mandate(self, mandate_id: str) -> "WontPayChannel":
        """Tag subsequent draws with the mandate they belong to. Mutates
        and returns self -- the channel is per-cell by construction (its
        rng must not be shared), so there is nothing to copy."""
        self.mandate_id = mandate_id
        return self

    def realised(self) -> dict:
        """The measured ROC point, plus the counts it was computed from.
        None where the denominator is zero -- never 0.0, which would read
        as "measured, and it was nothing"."""
        return {
            "n_wont_pay": self.n_wont_pay,
            "n_other": self.n_other,
            "positive_on_wont_pay": self.n_positive_on_wont_pay,
            "positive_on_other": self.n_positive_on_other,
            "tpr_realised": (
                self.n_positive_on_wont_pay / self.n_wont_pay if self.n_wont_pay else None
            ),
            "fpr_realised": (
                self.n_positive_on_other / self.n_other if self.n_other else None
            ),
        }

    def intent_score(self, cause: Cause) -> float:
        """One synthetic exit-intent score, for kind="intent"."""
        return _INTENT_SCORE_POSITIVE if self.fires(cause) else _INTENT_SCORE_NEGATIVE

    def describe(self) -> dict:
        return {"kind": self.kind, "tpr": self.tpr, "fpr": self.fpr}


def channel_decline_class(outcome: Outcome, *, cause: Cause, channel) -> DeclineClass | None:
    """The DeclineClass this harness treats as observed after `outcome`.

    Identical to `_proxy_decline_class(outcome)` unless a "decline" channel
    is live AND fires for this mandate's true cause, in which case
    CUSTOMER_DECLINED is emitted instead. A terminal outcome whose proxy is
    None stays None: RECOVERED ends the cycle, and DEAD/OPTED_OUT are
    handled by `belief.observe_terminal()` in eval/run.py, which conditions
    on an OBSERVED ledger fact rather than on a decline string (R2a).

    An "intent" channel deliberately returns the plain proxy: the two
    channels are different evidence KINDS, and emitting both from one draw
    would double-count the same signal.
    """
    proxy = _proxy_decline_class(outcome)
    if channel is None or channel.kind != "decline" or proxy is None:
        return proxy
    return DeclineClass.CUSTOMER_DECLINED if channel.fires(cause) else proxy


def apply_intent_channel(b, cause: Cause, channel):
    """Fold one synthetic exit-intent observation into `b`, if an "intent"
    channel is live. Returns `b` unchanged otherwise.

    The score crosses into the decision core exactly as a production one
    would: through src/execute/intent_channel.py's DECLARED operating
    point, never through the sweep's own (tpr, fpr). Those two are
    independent on purpose -- the adapter is therefore MISSPECIFIED at
    every sweep point except by coincidence, which is the realistic case
    and the reason the sweep exists.
    """
    if channel is None or channel.kind != "intent":
        return b
    lr = likelihood_ratio_from_intent_score(channel.intent_score(cause))
    return belief_mod.update_from_likelihood_ratio(
        b, lr, source_version=INTENT_CHANNEL_SOURCE_VERSION,
    )


_OUTCOME_TO_DECLINE_CLASS: dict[Outcome, DeclineClass | None] = {
    Outcome.DEAD: DeclineClass.CARD_EXPIRED,
    Outcome.STILL_PENDING: DeclineClass.INSUFFICIENT_FUNDS,
    Outcome.RECOVERED: None,
    Outcome.OPTED_OUT: None,
}


def _proxy_decline_class(outcome: Outcome) -> DeclineClass | None:
    return _OUTCOME_TO_DECLINE_CLASS[outcome]


# === The slot-1 decline signal ==============================================
#
# The frozen simulator does not emit one: its own module docstring records
# that "Every mandate entering this simulator has already had its slot-1
# (original) attempt fail ... Only slots 2/3/4 are simulated as decisions;
# slot 1 is given." That was frozen at B2, before B7's belief layer and
# B3's DeclineClass taxonomy existed to consume such a signal -- a scope
# boundary, not an oversight, and Simulator.attempt() enforces it by
# raising on slot 1.
#
# But the slot-1 decline reason is the ENTIRE premise of this system: a
# failed debit with a reason attached is what puts a mandate into a
# recovery engine, and it is the primary evidence the cause-inference
# layer exists to read. Without it the allocator's belief is uniform at
# every first decision, cause cannot influence any action taken BEFORE an
# outcome is revealed, and no discrimination metric of any shape can
# measure anything (DECISIONS.md, 2026-08-29/30, three entries).
#
# This harness therefore reconstructs that signal, using ONLY the frozen
# simulator's own generative parameters:
#
#   EMISSION  -- P(slot-1 failure mode | cause) is read from
#                sim_config.yaml's own per-cause hazards via the
#                simulator's own _logits_from_base_rates/_softmax helpers
#                (imported, not reimplemented, so there is no
#                transcription drift), conditioned on the two failure
#                modes that can actually put a mandate into recovery:
#                DEAD (instrument gone) and STILL_PENDING (it simply did
#                not go through). RECOVERED is excluded -- a recovered
#                slot-1 never enters recovery at all -- and so is
#                OPTED_OUT, which ends the relationship rather than
#                starting a retry cycle.
#
#   INFERENCE -- the allocator inverts that observation through
#                src/classify/cause_map.py, a SEPARATE, hand-authored
#                table with no relationship to sim_config.yaml's numbers.
#
# The two tables being independent is the point: the allocator's resulting
# belief is realistically MISCALIBRATED, not oracular. Observing
# CARD_EXPIRED yields 0.75 on CANT_PAY_EVER, never 1.0, and the emission
# rates that produced it (~58% vs ~3%) are not the rates cause_map assumes.
# The allocator can therefore still be wrong, in both directions -- which
# is exactly what makes the belief-weighted REAUTH safety rule in
# src/policy/allocator.py load-bearing rather than decorative.
#
# Nothing under eval/frozen/ is modified or re-derived; this reads it.

_SLOT1_DEAD = "dead"
_SLOT1_PENDING = "survive"


def slot1_failure_probs(cause: Cause, config: dict) -> dict[str, float]:
    """P(slot-1 failure mode | cause) over {dead, survive}, from the frozen
    config's own hazards, conditioned on the mandate having entered
    recovery (i.e. renormalised over the two non-terminal-for-us modes).
    No salary-window bonus and no optout escalation are applied: slot 1 is
    the original scheduled debit, not a retry this system chose the timing
    of, and retries_so_far is 0 there by definition."""
    h = config["hazards"][cause.value]
    probs = _softmax(_logits_from_base_rates(h["base_recovery"], h["base_dead"], h["base_optout"]))
    dead, pending = probs[_SLOT1_DEAD], probs[_SLOT1_PENDING]
    total = dead + pending
    return {_SLOT1_DEAD: dead / total, _SLOT1_PENDING: pending / total}


def draw_slot1_decline(
    cause: Cause, config: dict, rng: random.Random, *, channel=None,
) -> DeclineClass:
    """Draw the decline class the mandate's already-failed slot-1 attempt
    would have carried. `rng` must be a stream independent of the
    simulator's own, so adding this signal cannot perturb the outcome draws
    every previously-reported number depends on.

    R5: a live "decline" channel that fires REPLACES the draw with
    CUSTOMER_DECLINED -- it does not add to it. A slot-1 attempt the
    customer dismissed did not ALSO fail for insufficient funds; one
    attempt has one decline reason. The channel draws from its OWN rng
    (see WontPayChannel.rng), so the `rng.random()` call below still
    happens and consumes the same value it always did -- switching the
    channel on cannot shift this stream and desynchronise every other
    mandate's slot-1 draw.
    """
    p = slot1_failure_probs(cause, config)
    drawn = rng.random()
    if channel is not None and channel.kind == "decline" and channel.fires(cause):
        return DeclineClass.CUSTOMER_DECLINED
    if drawn < p[_SLOT1_DEAD]:
        return DeclineClass.CARD_EXPIRED
    return DeclineClass.INSUFFICIENT_FUNDS


def initial_belief(
    cause: Cause, config: dict, rng: random.Random, *, channel=None,
) -> belief_mod.Belief:
    """The belief the allocator starts a cycle with: the uniform reference
    prior updated by the slot-1 decline observation. Equivalent to
    cause_map.prior(dc) by belief.py's own documented round-trip identity,
    written as an explicit update() so the evidence step is visible rather
    than implied.

    R5: with an "intent" channel live, one synthetic exit-intent
    observation is folded in ON TOP of the ordinary slot-1 decline --
    additional evidence of a different kind, not a replacement for it,
    which is how a real support ticket would arrive alongside a real
    decline string. The "decline" channel instead replaces the decline
    itself (see draw_slot1_decline). The resulting Belief carries the
    channel's own provenance stamp either way.
    """
    dc = draw_slot1_decline(cause, config, rng, channel=channel)
    uniform = belief_mod.init(dict(zip(belief_mod.CAUSE_ORDER, belief_mod.REFERENCE_PRIOR)))
    source = (
        WONTPAY_CHANNEL_SOURCE_VERSION
        if dc == DeclineClass.CUSTOMER_DECLINED
        else PROXY_SOURCE_VERSION
    )
    b = belief_mod.update(uniform, dc, source_version=source)
    return apply_intent_channel(b, cause, channel)


@dataclass
class SweepResult:
    seed: int
    profile: Profile
    attempted: dict[str, bool] = field(default_factory=dict)
    attempts_spent: dict[str, int] = field(default_factory=dict)
    reauth: dict[str, bool] = field(default_factory=dict)
    true_cause: dict[str, Cause] = field(default_factory=dict)
    mandate_ids: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def fit_nominal_hazard_model() -> competing_risks.HazardModel:
    """Fit once, on a real corpus draw (eval.corpus's own TRAIN_SEEDS), so
    the sweep scores against a genuine fitted model rather than a
    hand-picked stub -- consistent with how B8 is actually used."""
    episodes = corpus.generate(arm="nominal")
    pp = person_period.build(episodes)
    feat = features.featurize(pp)
    assembled = competing_risks.assemble(pp, feat)
    return competing_risks.fit(assembled)


def hazard_from_fit(model: competing_risks.HazardModel):
    """Wrap a fitted HazardModel as the SlotHazard-shaped callable
    allocator.solve() takes -- one row at a time. See allocator.py's own
    module docstring for why the allocator takes this narrow Protocol
    rather than a CIF object.

    Memoised on (slot, in_salary_window): FEATURE_COLUMNS
    (competing_risks.py) is ("const", "slot_3", "slot_4",
    "in_salary_window") -- amount_paise and the exact on_day never enter
    the design matrix at all, only which of the 3 slots and whether on_day
    falls in the 1-5 salary window. There are therefore at most 6 distinct
    outputs this function can ever produce; recomputing a fresh
    single-row statsmodels prediction per call (thousands of times across
    one seed's mandates, since backward induction explores every candidate
    day at every depth) measured at ~50s/seed -- this cache is what makes
    the 20-seed x 2-profile sweep tractable to actually run."""
    import pandas as pd

    cache: dict[tuple[int, bool], tuple[float, float, float, float]] = {}

    def h(*, slot: int, on_day: int, amount_paise: int) -> tuple[float, float, float, float]:
        key = (slot, 1 <= on_day <= 5)
        cached = cache.get(key)
        if cached is not None:
            return cached
        row = pd.DataFrame([{
            "slot": slot,
            "in_salary_window": key[1],
            "days_since_last_attempt": 0.0,
        }])
        probs = competing_risks.hazards(model, row)
        result = tuple(float(x) for x in probs[0])
        cache[key] = result
        return result

    return h


def _initial_context(m, profile: Profile, costs: PolicyCosts) -> AllocationContext:
    return AllocationContext(
        mandate_id=m.mandate_id,
        cycle_id=m.cycle_id,
        profile=profile,
        amount_paise=m.amount_paise,
        ceiling_paise=m.ceiling_paise,
        category=m.category,
        plan_day=1,
        attempts_used=1,
        committed_days=(1,),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def _run_one_mandate(m, sim: Simulator, profile: Profile, hazard, costs: PolicyCosts, gate, violations: list[str], b) -> tuple[bool, int, bool]:
    """Drive one AFA-eligible mandate through up to three retries
    (slots 2, 3, 4), calling solve() for each decision and the frozen
    Simulator for the real outcome. Returns (ever attempted, attempts
    spent, ever re-authorised).

    `b` is the mandate's STARTING belief, already carrying the slot-1
    decline observation (see initial_belief() above) -- passed in rather
    than constructed here so the caller owns the RNG stream that produced
    it."""
    ctx = _initial_context(m, profile, costs)
    attempted = False
    n_attempts = 0
    reauthed = False

    while ctx.attempts_used < MAX_ATTEMPTS:
        try:
            plan = solve(b, ctx, hazard=hazard, costs=costs, gate=gate)
        except AllocatorError as exc:
            violations.append(f"{m.mandate_id}: AllocatorError: {exc}")
            break

        if plan.chosen_action == Action.REAUTH:
            reauthed = True
            break
        if plan.chosen_action != Action.ATTEMPT:
            break

        committed = plan.committed[0]
        if len(plan.committed) != 1:
            violations.append(f"{m.mandate_id}: ATTEMPT plan committed {len(plan.committed)} attempts, expected 1")
        if committed.amount_paise > ctx.ceiling_paise:
            violations.append(f"{m.mandate_id}: committed amount {committed.amount_paise} exceeds ceiling {ctx.ceiling_paise}")
        if committed.slot != ctx.attempts_used + 1:
            violations.append(f"{m.mandate_id}: committed slot {committed.slot} != expected {ctx.attempts_used + 1}")
        if ctx.committed_days and committed.on_day <= ctx.committed_days[-1]:
            violations.append(f"{m.mandate_id}: committed day {committed.on_day} not after previous {ctx.committed_days[-1]}")

        attempted = True
        n_attempts += 1
        result = sim.attempt(m.mandate_id, slot=committed.slot, on_day=committed.on_day)
        ctx = ctx.with_attempt(committed.on_day)

        if result.outcome in (Outcome.RECOVERED, Outcome.OPTED_OUT, Outcome.DEAD):
            # All three end the ATTEMPT sequence -- a DEAD instrument does
            # not become un-dead on a re-attempt; only STILL_PENDING
            # continues the retry cycle. (Re-attempting after DEAD was a
            # bug in this harness, found while diagnosing the B8 gate
            # sweep; see DECISIONS.md.)
            #
            # But the ATTEMPT sequence ending is not the same as the
            # DECISION sequence ending. A dead instrument is precisely
            # when re-authorisation is the correct next action -- the
            # CANT_PAY_EVER -> REAUTH row of root CLAUDE.md's own table --
            # and this harness previously just `break`-ed, never asking
            # the allocator what to do next and so never recording the
            # one action that lane exists to produce. Ask once more, with
            # the terminal observation folded into belief, and record the
            # decision without executing any further debit.
            dc = _proxy_decline_class(result.outcome)
            if dc is not None:
                b = belief_mod.update(b, dc, source_version=PROXY_SOURCE_VERSION)
                final = solve(b, ctx, hazard=hazard, costs=costs, gate=gate)
                if final.chosen_action == Action.REAUTH:
                    reauthed = True
            break

        dc = _proxy_decline_class(result.outcome)
        if dc is not None:
            b = belief_mod.update(b, dc, source_version=PROXY_SOURCE_VERSION)

    return attempted, n_attempts, reauthed


def sweep_one(seed: int, profile: Profile, hazard, costs: PolicyCosts, config: dict | None = None) -> SweepResult:
    sim = Simulator("nominal", seed=seed)
    gate = FullSetGate()
    result = SweepResult(seed=seed, profile=profile)
    cfg = config if config is not None else load_config()
    # Independent of the simulator's own RNG, so adding the slot-1 signal
    # cannot perturb the outcome draws. Offset distinct from every other
    # stream in this project (100_000 = the original uniform-random gate
    # baseline, 300_000 = the cause-blind-random baseline).
    slot1_rng = random.Random(seed + 500_000)

    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            continue  # above the AFA cliff -- excluded from the eligible denominator
        result.mandate_ids.append(m.mandate_id)
        result.true_cause[m.mandate_id] = m.initial_cause
        b0 = initial_belief(m.initial_cause, cfg, slot1_rng)
        attempted, n_attempts, reauthed = _run_one_mandate(
            m, sim, profile, hazard, costs, gate, result.violations, b0
        )
        result.attempted[m.mandate_id] = attempted
        result.attempts_spent[m.mandate_id] = n_attempts
        result.reauth[m.mandate_id] = reauthed

    return result


def main() -> int:
    costs = load_costs()
    model = fit_nominal_hazard_model()
    hazard = hazard_from_fit(model)

    overall_ok = True
    print(f"{'profile':<11} {'attempt_rate (mean)':>20} {'floor':>8} {'ok':>4}   "
          f"{'discrimination (mean)':>22} {'margin':>8} {'ok':>4}   {'violations':>10}")

    for profile in (Profile.strict, Profile.permissive):
        rates: list[float] = []
        gaps: list[float] = []
        total_violations = 0

        for seed in SEEDS:
            res = sweep_one(seed, profile, hazard, costs)
            rates.append(attempt_rate(res.attempted, res.mandate_ids))
            gaps.append(discrimination_gap(res.attempts_spent, res.true_cause, res.mandate_ids))
            total_violations += len(res.violations)
            for v in res.violations[:5]:
                print(f"  VIOLATION seed={seed} profile={profile.value}: {v}", file=sys.stderr)

        mean_rate = sum(rates) / len(rates)
        mean_gap = sum(gaps) / len(gaps)
        rate_ok = mean_rate >= ATTEMPT_RATE_FLOOR
        gap_ok = mean_gap > DISCRIMINATION_MARGIN
        overall_ok = overall_ok and rate_ok and gap_ok and total_violations == 0

        print(
            f"{profile.value:<11} {mean_rate:>20.4f} {ATTEMPT_RATE_FLOOR:>8.2f} "
            f"{'OK' if rate_ok else 'FAIL':>4}   "
            f"{mean_gap:>22.4f} {DISCRIMINATION_MARGIN:>8.4f} {'OK' if gap_ok else 'FAIL':>4}   "
            f"{total_violations:>10}"
        )

    print()
    print("GATE: " + ("PASS" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
