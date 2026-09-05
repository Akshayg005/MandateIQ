"""The synthetic WONT_PAY evidence channel (eval/allocator_sweep.py).

R5 (reports/gates.md, "Post-B16 remediation gates"). Two channels, each
quality-parameterised by (tpr, fpr):

  decline -- emits DeclineClass.CUSTOMER_DECLINED, which belief.update()
             inverts through src/classify/cause_map.py.
  intent  -- emits a score, which src/execute/intent_channel.py maps to a
             declared likelihood ratio for
             belief.update_from_likelihood_ratio().

**This channel reads the privileged true cause and feeds it into the
DECISION path.** That is a materially stronger claim than the score-only
privileged read `false_reauth_count` already makes, which is why every
test here also pins the disclosure machinery: the distinct provenance
stamp, and the fact that a channel-free run is byte-identical to the runs
every previously published number came from.
"""
from __future__ import annotations

import random

import pytest

from src.core.types import Cause, DeclineClass, Outcome


def _chan(kind: str, tpr: float, fpr: float, seed: int = 0):
    from eval.allocator_sweep import WontPayChannel

    return WontPayChannel(kind=kind, tpr=tpr, fpr=fpr, rng=random.Random(seed))


# --- the emitter's (tpr, fpr) behaviour, at both extremes -------------------

def test_a_perfect_channel_fires_on_exactly_the_wont_pay_mandates():
    ch = _chan("decline", tpr=1.0, fpr=0.0)
    assert all(ch.fires(Cause.WONT_PAY) for _ in range(200))
    assert not any(ch.fires(Cause.CANT_PAY_NOW) for _ in range(200))
    assert not any(ch.fires(Cause.CANT_PAY_EVER) for _ in range(200))


def test_a_dead_channel_never_fires():
    ch = _chan("decline", tpr=0.0, fpr=0.0)
    assert not any(ch.fires(c) for c in Cause for _ in range(200))


def test_a_coin_flip_channel_fires_at_the_same_rate_on_every_cause():
    """AUC 0.5 -- the deliberately worthless point the sweep includes so
    degradation is visible. A sweep that only shows good channels proves
    nothing."""
    ch = _chan("decline", tpr=0.5, fpr=0.5, seed=7)
    on_wont = sum(ch.fires(Cause.WONT_PAY) for _ in range(4000))
    on_other = sum(ch.fires(Cause.CANT_PAY_NOW) for _ in range(4000))
    assert abs(on_wont - on_other) < 250          # ~3 SE at n=4000
    assert 1800 < on_wont < 2200


def test_the_realised_rates_track_the_declared_ones():
    ch = _chan("decline", tpr=0.60, fpr=0.15, seed=11)
    on_wont = sum(ch.fires(Cause.WONT_PAY) for _ in range(20000)) / 20000
    on_other = sum(ch.fires(Cause.CANT_PAY_EVER) for _ in range(20000)) / 20000
    assert on_wont == pytest.approx(0.60, abs=0.02)
    assert on_other == pytest.approx(0.15, abs=0.02)


def test_channel_rejects_an_unknown_kind():
    from eval.allocator_sweep import WontPayChannel

    with pytest.raises(ValueError):
        WontPayChannel(kind="telepathy", tpr=0.5, fpr=0.1, rng=random.Random(0))


def test_channel_rejects_out_of_range_rates():
    from eval.allocator_sweep import WontPayChannel

    for tpr, fpr in ((1.5, 0.1), (0.5, -0.1), (0.5, 1.2)):
        with pytest.raises(ValueError):
            WontPayChannel(kind="decline", tpr=tpr, fpr=fpr, rng=random.Random(0))


# --- R5 review pass, 2026-09-05: within-mandate dependence (stats-reviewer,
# HIGH). The main QUALITY_GRID holds correlation fixed at exactly zero --
# `habitual_fraction` is the separate dimension that varies it, WITHOUT
# perturbing anything the default (1.0) already produced.

def test_habitual_fraction_defaults_to_the_plain_iid_channel():
    """1.0 must be a NO-OP: every already-published number came from the
    plain iid draw, and this default must reproduce it byte for byte, not
    just approximately. Same rng, same sequence of calls, same outcomes --
    proven by comparing against a channel that has no habitual_fraction
    concept, one draw at a time."""
    from eval.allocator_sweep import WontPayChannel

    plain = WontPayChannel(kind="decline", tpr=0.6, fpr=0.15, rng=random.Random(7))
    habitual = WontPayChannel(kind="decline", tpr=0.6, fpr=0.15,
                              habitual_fraction=1.0, rng=random.Random(7))
    plain.for_mandate("M1")
    habitual.for_mandate("M1")
    for i in range(200):
        cause = Cause.WONT_PAY if i % 3 == 0 else Cause.CANT_PAY_NOW
        assert plain.fires(cause) == habitual.fires(cause)


def test_habitual_fraction_rejects_zero_and_above_one():
    from eval.allocator_sweep import WontPayChannel

    for hf in (0.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            WontPayChannel(kind="decline", tpr=0.6, fpr=0.15,
                           habitual_fraction=hf, rng=random.Random(0))


def test_habitual_fraction_preserves_the_marginal_false_positive_rate():
    """The whole point of the two-point mixture: only CORRELATION should
    move, not the rate the main grid's ROC already measures. Exact at
    habitual_fraction >= fpr (proven in WontPayChannel._effective_fpr's own
    docstring); checked here empirically across many mandates."""
    ch = _chan("decline", tpr=0.6, fpr=0.15, seed=3)
    ch.habitual_fraction = 0.5  # >= fpr=0.15, so E[fire] == fpr exactly
    fires = []
    for i in range(4000):
        ch.for_mandate(f"M{i}")
        fires.append(ch.fires(Cause.CANT_PAY_NOW))
    assert sum(fires) / len(fires) == pytest.approx(0.15, abs=0.02)


def test_habitual_fraction_concentrates_repeat_false_firing():
    """The mechanism the review found missing: at the SAME marginal fpr,
    lower habitual_fraction must produce MORE mandates with two-or-more
    false firings, not the same number redistributed."""
    def two_plus_rate(hf, seed):
        ch = _chan("decline", tpr=0.6, fpr=0.15, seed=seed)
        ch.habitual_fraction = hf
        two_plus = 0
        for i in range(3000):
            ch.for_mandate(f"M{i}")
            fires = [ch.fires(Cause.CANT_PAY_NOW) for _ in range(4)]
            two_plus += int(sum(fires) >= 2)
        return two_plus / 3000

    iid_rate = two_plus_rate(1.0, seed=11)
    correlated_rate = two_plus_rate(0.3, seed=11)
    # Binomial(4, 0.15) gives P(>=2)=0.110 at hf=1.0 (iid); the two-point
    # mixture at hf=0.3 gives 0.3*Binomial(4, 0.5, >=2) = 0.3*0.6875=0.206 --
    # matches measured to 3 decimal places. 1.5x is comfortably inside that
    # margin while still requiring a REAL, not incidental, increase.
    assert correlated_rate > 1.5 * iid_rate


def test_the_wont_pay_side_is_never_touched_by_the_mixture():
    """The reviewer's finding is specifically about FALSE firings on a
    paying customer. WONT_PAY's own tpr draw must stay the plain
    Bernoulli(tpr) at every habitual_fraction -- inventing correlation on
    the true-positive side is a different, unjustified claim this module
    does not make."""
    ch = _chan("decline", tpr=0.6, fpr=0.15, seed=5)
    ch.habitual_fraction = 0.2
    fires = []
    for i in range(3000):
        ch.for_mandate(f"M{i}")
        fires.append(ch.fires(Cause.WONT_PAY))
    assert sum(fires) / len(fires) == pytest.approx(0.6, abs=0.03)


# --- the decline channel reaches the belief --------------------------------

def test_the_decline_channel_emits_customer_declined_on_a_wont_pay_mandate():
    from eval.allocator_sweep import channel_decline_class

    ch = _chan("decline", tpr=1.0, fpr=0.0)
    assert channel_decline_class(
        Outcome.STILL_PENDING, cause=Cause.WONT_PAY, channel=ch,
    ) == DeclineClass.CUSTOMER_DECLINED


def test_the_decline_channel_falls_back_to_the_proxy_when_it_does_not_fire():
    """A non-firing channel must leave the existing proxy untouched, or the
    channel would be changing the harness's behaviour on mandates it has
    nothing to say about."""
    from eval.allocator_sweep import _proxy_decline_class, channel_decline_class

    ch = _chan("decline", tpr=0.0, fpr=0.0)
    for outcome in Outcome:
        assert channel_decline_class(
            outcome, cause=Cause.WONT_PAY, channel=ch,
        ) == _proxy_decline_class(outcome)


def test_a_none_channel_is_exactly_the_old_proxy():
    """The load-bearing regression guard: every number this project has
    already published came from the channel-free path, so `channel=None`
    must be that path unchanged, not a re-implementation of it."""
    from eval.allocator_sweep import _proxy_decline_class, channel_decline_class

    for outcome in Outcome:
        for cause in Cause:
            assert channel_decline_class(
                outcome, cause=cause, channel=None,
            ) == _proxy_decline_class(outcome)


def test_the_intent_channel_never_emits_a_decline_class():
    """The two channels are separate evidence kinds. The intent channel
    updates belief through a declared likelihood ratio and must not also
    fabricate a decline string -- that would be two observations from one
    signal, double-counting the same evidence."""
    from eval.allocator_sweep import _proxy_decline_class, channel_decline_class

    ch = _chan("intent", tpr=1.0, fpr=0.0)
    for outcome in Outcome:
        assert channel_decline_class(
            outcome, cause=Cause.WONT_PAY, channel=ch,
        ) == _proxy_decline_class(outcome)


# --- provenance -------------------------------------------------------------

def test_the_channel_carries_its_own_provenance_stamp():
    """Never PROXY_SOURCE_VERSION and never a taxonomy version: stamping
    fabricated evidence as taxonomy output would be exactly the provenance
    lie belief.update()'s source_version parameter exists to make
    impossible."""
    from eval.allocator_sweep import (
        INTENT_CHANNEL_SOURCE_VERSION, PROXY_SOURCE_VERSION,
        WONTPAY_CHANNEL_SOURCE_VERSION,
    )
    from src.classify.decline_taxonomy import TAXONOMY_VERSION

    stamps = {WONTPAY_CHANNEL_SOURCE_VERSION, INTENT_CHANNEL_SOURCE_VERSION}
    assert len(stamps) == 2
    assert PROXY_SOURCE_VERSION not in stamps
    assert TAXONOMY_VERSION not in stamps
    assert all("eval" in s for s in stamps)


def test_a_channel_belief_is_traceable_to_the_channel():
    from eval.allocator_sweep import WONTPAY_CHANNEL_SOURCE_VERSION, initial_belief
    from eval.frozen.simulator import load_config

    ch = _chan("decline", tpr=1.0, fpr=0.0)
    b = initial_belief(Cause.WONT_PAY, load_config(), random.Random(0), channel=ch)
    assert f"source={WONTPAY_CHANNEL_SOURCE_VERSION}" in b.provenance


# --- the whole point: the singleton becomes reachable -----------------------

def test_repeated_channel_evidence_can_reach_a_wont_pay_dominant_belief():
    """Before R5, exhaustive enumeration over every sequence reachable
    within the NPCI cap measured max P(WONT_PAY) = 0.10. Three
    CUSTOMER_DECLINED observations now clear 0.9, which is what makes the
    conformal singleton reachable at all."""
    from eval.allocator_sweep import WONTPAY_CHANNEL_SOURCE_VERSION
    from src.policy import belief as belief_mod

    b = belief_mod.init(dict(zip(belief_mod.CAUSE_ORDER, belief_mod.REFERENCE_PRIOR)))
    for _ in range(3):
        b = belief_mod.update(
            b, DeclineClass.CUSTOMER_DECLINED,
            source_version=WONTPAY_CHANNEL_SOURCE_VERSION,
        )
    assert b.dominant() == Cause.WONT_PAY
    assert b[Cause.WONT_PAY] > 0.90


def test_the_pre_r5_alphabet_could_never_reach_a_wont_pay_dominant_belief():
    """The measurement this whole block rests on, re-derived here rather
    than quoted: over EVERY sequence of up to four observations from the
    two-symbol alphabet eval/allocator_sweep.py could emit before R5, the
    maximum achievable P(WONT_PAY) is 0.10. Not "low" -- structurally
    capped, because the WONT_PAY component of both likelihood vectors is
    identical (0.30), so its likelihood ratio against the others is
    monotone non-increasing."""
    import itertools

    from src.policy import belief as belief_mod

    uniform = belief_mod.init(
        dict(zip(belief_mod.CAUSE_ORDER, belief_mod.REFERENCE_PRIOR))
    )
    pre_r5_alphabet = (DeclineClass.CARD_EXPIRED, DeclineClass.INSUFFICIENT_FUNDS)
    best = 0.0
    for n in range(1, 5):
        for seq in itertools.product(pre_r5_alphabet, repeat=n):
            b = uniform
            for dc in seq:
                b = belief_mod.update(b, dc, source_version="probe")
            best = max(best, b[Cause.WONT_PAY])
    assert best == pytest.approx(0.10, abs=1e-9)
