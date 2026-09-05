"""src/policy/belief.py -- posterior distribution over the three latent causes
(CANT_PAY_NOW, CANT_PAY_EVER, WONT_PAY). Bayesian belief update at each decline
observation.

Design spec: The belief state is a normalized distribution P(cause | history) over
the three causes, updated by observing DeclineClass via Bayes' rule:

    update(b, dc)[c] ∝ b[c] · P(dc | c)

where P(dc | c) is the "likelihood" -- conventionally computed as an inversion of
the prior:

    P(dc | c) = prior(dc)[c] / REFERENCE_PRIOR[c]

The inversion is exact under the RBI's pre-registered Bayesian model, which assumes
a flat (uniform) starting prior over causes and a static cause per mandate-cycle.
The cause_switch_prob parameter (0.15 per attempt) in sim_config.yaml documents
that this stationarity assumption is known to be violated; B7 measures and discloses
the gap, but does not dampen or switch-leak away the update.

Belief objects are hashable (B8 memoises backward induction on quantised beliefs),
immutable (frozen=True), and carry provenance (normaliser version + reference
prior version) so the ledger can audit them.
"""
from __future__ import annotations

import json
import pytest

from src.core.types import Cause, DeclineClass


# === Helper functions ===========================================================

def _read_cause_map_source() -> str:
    """Read cause_map.py source to grep for implementation details."""
    from pathlib import Path
    return Path("src/classify/cause_map.py").read_text()


def _read_sim_config_source() -> str:
    """Read sim_config.yaml source to find cause_switch_prob."""
    from pathlib import Path
    return Path("eval/frozen/sim_config.yaml").read_text()


# === reference prior validation ================================================

def test_reference_prior_sums_to_one_and_names_all_three_causes():
    """The reference prior must be a valid probability distribution that names
    all three Cause members and sums to exactly 1.0."""
    from src.policy.belief import REFERENCE_PRIOR, CAUSE_ORDER

    assert len(REFERENCE_PRIOR) == 3, \
        f"REFERENCE_PRIOR has {len(REFERENCE_PRIOR)} elements, not 3"
    assert sum(REFERENCE_PRIOR) == pytest.approx(1.0, abs=1e-9), \
        f"REFERENCE_PRIOR sum = {sum(REFERENCE_PRIOR)}, not 1.0"

    for cause in Cause:
        idx = CAUSE_ORDER.index(cause)
        assert REFERENCE_PRIOR[idx] >= 0.0, \
            f"REFERENCE_PRIOR[{cause}] = {REFERENCE_PRIOR[idx]} < 0"


def test_reference_prior_is_uniform_today():
    """The current REFERENCE_PRIOR is exactly uniform: (1/3, 1/3, 1/3).
    This is a distinct, deliberately narrow test so changing it to a non-uniform
    value is a visible gate failure, not a silent shift in every belief."""
    from src.policy.belief import REFERENCE_PRIOR

    expected = 1.0 / 3.0
    for prob in REFERENCE_PRIOR:
        assert prob == pytest.approx(expected, abs=1e-9), \
            f"REFERENCE_PRIOR element {prob} != uniform {expected}"


# === cause order validation ===================================================

def test_cause_order_is_fixed_and_names_all_three_causes():
    """CAUSE_ORDER must be a 3-tuple naming each Cause member exactly once.
    Downstream code (allocator, CIF, conformal prediction) indexes beliefs via
    CAUSE_ORDER, so an ordering bug silently mislabels causes everywhere."""
    from src.policy.belief import CAUSE_ORDER

    assert isinstance(CAUSE_ORDER, tuple), \
        f"CAUSE_ORDER is {type(CAUSE_ORDER).__name__}, not tuple"
    assert len(CAUSE_ORDER) == 3, \
        f"CAUSE_ORDER has {len(CAUSE_ORDER)} members, not 3"
    assert set(CAUSE_ORDER) == set(Cause), \
        f"CAUSE_ORDER {CAUSE_ORDER} does not name all three Cause members"


# === likelihood inversion ====================================================

@pytest.mark.parametrize("dc", list(DeclineClass))
def test_inversion_round_trips_to_prior_exactly(dc):
    """The single most important test in this file: the load-bearing invariant
    that cause_map.prior() (a posterior in Bayes' terms) and belief.likelihood()
    (its inversion) are two representations of one mathematical object and must
    never drift apart.

    This test takes likelihood(dc), applies Bayes with REFERENCE_PRIOR as the
    prior (i.e., computes REFERENCE_PRIOR[c] * likelihood(dc)[c] for each c,
    then normalises), and asserts it recovers cause_map.prior(dc) exactly. The
    test is parametrized over EVERY DeclineClass member and would still pass
    with a non-uniform REFERENCE_PRIOR, because the round-trip is an identity
    in Bayes' rule, not a feature of uniformity.

    If this test fails:
    - Someone changed cause_map.prior() without updating likelihood()
    - Someone changed REFERENCE_PRIOR without updating likelihood()
    - Someone changed the update() formula
    Do not rationalize the failure as a "rounding issue" -- it is a definition
    mismatch, and the fix is to reconcile the two sources against the spec."""
    from src.policy.belief import likelihood, REFERENCE_PRIOR, CAUSE_ORDER
    from src.classify.cause_map import prior as cause_map_prior

    # Get the likelihood (unnormalised Bayes inverse)
    lik = likelihood(dc)
    assert len(lik) == 3, f"likelihood({dc}) has {len(lik)} elements, not 3"

    # Apply Bayes with the reference prior: P(c|dc) ∝ P(c) * P(dc|c)
    unnormalized = [REFERENCE_PRIOR[i] * lik[i] for i in range(3)]
    total = sum(unnormalized)
    assert total > 0, f"likelihood({dc}) is all zeros: {lik}"

    posterior = [unnormalized[i] / total for i in range(3)]

    # Compare to cause_map.prior(dc)
    cause_map_result = cause_map_prior(dc)
    for i, cause in enumerate(CAUSE_ORDER):
        expected = cause_map_result[cause]
        observed = posterior[i]
        assert observed == pytest.approx(expected, abs=1e-9), \
            f"Bayes(REFERENCE_PRIOR, likelihood({dc})) -- " \
            f"cause {cause}: observed {observed}, expected {expected}"


@pytest.mark.parametrize("dc", list(DeclineClass))
def test_likelihood_returns_a_3_tuple_of_floats(dc):
    """likelihood(dc) must return a tuple with exactly 3 float elements,
    indexed in CAUSE_ORDER."""
    from src.policy.belief import likelihood

    result = likelihood(dc)
    assert isinstance(result, tuple), \
        f"likelihood({dc}) is {type(result).__name__}, not tuple"
    assert len(result) == 3, \
        f"likelihood({dc}) has {len(result)} elements, not 3"

    for i, val in enumerate(result):
        assert isinstance(val, float), \
            f"likelihood({dc})[{i}] = {val!r}, not float"


# === Belief class tests ======================================================

def test_reference_prior_version_is_a_string():
    """REFERENCE_PRIOR_VERSION names the version of the uniform prior used
    in the normalizer's output. It must be a non-empty string."""
    from src.policy.belief import REFERENCE_PRIOR_VERSION

    assert isinstance(REFERENCE_PRIOR_VERSION, str), \
        f"REFERENCE_PRIOR_VERSION is {type(REFERENCE_PRIOR_VERSION).__name__}, not str"
    assert len(REFERENCE_PRIOR_VERSION) > 0, \
        f"REFERENCE_PRIOR_VERSION is empty"


def test_belief_is_hashable():
    """Belief must be hashable. B8's backward-induction memoization uses
    quantised(b) as part of a cache key, so the object itself must support
    hash(b) and {b: value} dict insertion."""
    from src.policy.belief import init, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    # Must not raise
    h = hash(b)
    assert isinstance(h, int), f"hash(b) returned {type(h).__name__}, not int"

    # Must be insertable as a dict key
    d = {b: 1}
    assert b in d, f"Belief is not usable as a dict key"


def test_update_result_is_a_valid_distribution():
    """After update(), the returned Belief must represent a valid probability
    distribution: all probabilities in [0, 1], sum to 1.0, and all three Cause
    members are named."""
    from src.policy.belief import init, update, REFERENCE_PRIOR, CAUSE_ORDER
    from src.classify.cause_map import prior as cause_map_prior

    b0 = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                       REFERENCE_PRIOR)))

    for dc in DeclineClass:
        b1 = update(b0, dc, source_version="taxonomy=v1")

        # Check it's a valid distribution
        for cause in CAUSE_ORDER:
            prob = b1[cause]
            assert 0.0 <= prob <= 1.0, \
                f"update(b0, {dc})[{cause}] = {prob} not in [0, 1]"

        # Sum to 1
        total = sum(b1[cause] for cause in CAUSE_ORDER)
        assert total == pytest.approx(1.0, abs=1e-9), \
            f"update(b0, {dc}) sums to {total}, not 1.0"


def test_update_is_order_invariant():
    """Updating with DeclineClass A then B must equal updating B then A,
    because multiplication (the update operator) commutes. This is a core
    Bayesian identity: the order of observations does not affect the posterior
    when likelihood is memoryless."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b0 = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                       REFERENCE_PRIOR)))

    dc_a = DeclineClass.INSUFFICIENT_FUNDS
    dc_b = DeclineClass.CARD_EXPIRED

    b_ab = update(update(b0, dc_a, source_version="taxonomy=v1"), dc_b, source_version="taxonomy=v1")
    b_ba = update(update(b0, dc_b, source_version="taxonomy=v1"), dc_a, source_version="taxonomy=v1")

    for cause in Cause:
        assert b_ab[cause] == pytest.approx(b_ba[cause], abs=1e-9), \
            f"Order dependence at {cause}: A-then-B={b_ab[cause]}, " \
            f"B-then-A={b_ba[cause]}"


def test_update_from_flat_prior_equals_the_cause_map_prior():
    """A flat prior updated once with a DeclineClass must equal cause_map.prior()
    of that class. This is the identity that the likelihood inversion enforces:
    if you start with uniform and observe, you should get the same answer as
    querying cause_map directly."""
    from src.policy.belief import init, update, REFERENCE_PRIOR
    from src.classify.cause_map import prior as cause_map_prior

    b0 = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                       REFERENCE_PRIOR)))

    for dc in DeclineClass:
        b1 = update(b0, dc, source_version="taxonomy=v1")
        expected = cause_map_prior(dc)

        for cause in Cause:
            assert b1[cause] == pytest.approx(expected[cause], abs=1e-9), \
                f"update(flat, {dc})[{cause}] = {b1[cause]}, " \
                f"cause_map.prior({dc})[{cause}] = {expected[cause]}"


def test_update_never_moves_mass_to_a_zero_probability_cause():
    """If a Belief has exactly 0.0 mass on a cause, Bayes' rule cannot
    resurrect it. A zero probability is absorbing: update(b, dc)[c] must
    remain 0.0 if b[c] == 0.0."""
    from src.policy.belief import init, update, CAUSE_ORDER

    # Create a belief with zero mass on WONT_PAY
    probs = [0.7, 0.3, 0.0]  # CANT_PAY_NOW, CANT_PAY_EVER, WONT_PAY
    b = init(dict(zip(CAUSE_ORDER, probs)))

    # Update with any decline class
    for dc in [DeclineClass.INSUFFICIENT_FUNDS, DeclineClass.CARD_EXPIRED]:
        b_updated = update(b, dc, source_version="taxonomy=v1")
        assert b_updated[Cause.WONT_PAY] == pytest.approx(0.0, abs=1e-9), \
            f"update(zero-WONT_PAY, {dc}) resurrected mass: {b_updated[Cause.WONT_PAY]}"


def test_update_returns_a_new_object_and_does_not_mutate_its_input():
    """update(b, dc) must return a distinct Belief object and leave the input
    unchanged. Belief is frozen; the implementation should never modify the
    input in place."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b0 = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                       REFERENCE_PRIOR)))

    b0_before = {c: b0[c] for c in Cause}

    b1 = update(b0, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")

    # Check b0 is unchanged
    b0_after = {c: b0[c] for c in Cause}
    assert b0_before == b0_after, \
        f"update() mutated its input: before={b0_before}, after={b0_after}"

    # Check b1 is distinct
    assert b1 is not b0, f"update() returned the same object, not a new one"


def test_update_requires_source_version():
    """update() must require the source_version keyword argument. Calling
    update(b, dc) with only two positional args must raise TypeError because
    source_version has no default -- this enforces that every update is
    traced to the classifier that produced its observation."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    # Calling without source_version must raise TypeError
    with pytest.raises(TypeError):
        update(b, DeclineClass.INSUFFICIENT_FUNDS)


def test_update_rejects_empty_source_version():
    """update() must reject an empty string for source_version, raising
    BeliefError. A belief cannot be traced to a non-existent classifier
    version -- that is the entire reason source_version is required and must
    be validated at runtime."""
    from src.policy.belief import init, update, BeliefError, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    # Empty string must raise BeliefError
    with pytest.raises(BeliefError):
        update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="")


def test_update_provenance_records_which_source_produced_the_observation():
    """update(b, dc, source_version=v1) and update(b, dc, source_version=v2)
    produce beliefs with identical probs (source doesn't affect the math) but
    different provenance strings. The provenance must record which classifier
    produced the observation, proving that the belief is auditable to a
    specific normaliser version. This is the traceability property B11's gate
    requires: 'normaliser output is versioned in the ledger before it can
    touch a belief ... a belief that cannot be traced to a specific normaliser
    version is not auditable'."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    dc = DeclineClass.INSUFFICIENT_FUNDS
    v1 = "taxonomy=v1"
    v2 = "normalizer=abc123def456"

    b_v1 = update(b, dc, source_version=v1)
    b_v2 = update(b, dc, source_version=v2)

    # Probabilities must be identical (source doesn't affect the math)
    for cause in Cause:
        assert b_v1[cause] == pytest.approx(b_v2[cause], abs=1e-9), \
            f"Source version changed the math for {cause}: {b_v1[cause]} vs {b_v2[cause]}"

    # Provenance strings must differ and each contain its own source_version
    assert b_v1.provenance != b_v2.provenance, \
        f"Provenance should differ for different source_version values"
    assert f"source={v1}" in b_v1.provenance, \
        f"Provenance '{b_v1.provenance}' does not contain 'source={v1}'"
    assert f"source={v2}" in b_v2.provenance, \
        f"Provenance '{b_v2.provenance}' does not contain 'source={v2}'"


def test_update_provenance_still_contains_cause_map_and_reference_prior_versions():
    """update() adds source_version to provenance, but must NOT replace the
    existing cause_map and reference_prior version fields. The provenance is
    additive: it contains cause_map=<version>, reference_prior=<version>,
    AND source=<version>, all three pieces of traceability together."""
    from src.policy.belief import init, update, REFERENCE_PRIOR, REFERENCE_PRIOR_VERSION
    from src.classify.cause_map import PRIOR_VERSION

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    b_updated = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")

    provenance = b_updated.provenance
    assert isinstance(provenance, str), \
        f"provenance is {type(provenance).__name__}, not str"

    # Must contain ALL THREE version fields
    assert f"cause_map={PRIOR_VERSION}" in provenance, \
        f"provenance '{provenance}' missing 'cause_map={PRIOR_VERSION}'"
    assert f"reference_prior={REFERENCE_PRIOR_VERSION}" in provenance, \
        f"provenance '{provenance}' missing 'reference_prior={REFERENCE_PRIOR_VERSION}'"
    assert "source=taxonomy=v1" in provenance, \
        f"provenance '{provenance}' missing 'source=taxonomy=v1'"


def test_update_getitem_access_works():
    """Belief must support __getitem__(cause) to retrieve the probability for
    a given cause. This is the primary access pattern."""
    from src.policy.belief import init, REFERENCE_PRIOR, CAUSE_ORDER

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))

    for cause in Cause:
        prob = b[cause]
        assert isinstance(prob, float), \
            f"b[{cause}] = {prob!r}, not float"
        assert 0.0 <= prob <= 1.0, \
            f"b[{cause}] = {prob} not in [0, 1]"


# === init() validation =======================================================

def test_init_rejects_a_prior_that_does_not_sum_to_one():
    """init() must raise BeliefError if the input prior does not sum to 1.0."""
    from src.policy.belief import init, BeliefError

    bad_prior = {
        Cause.CANT_PAY_NOW: 0.5,
        Cause.CANT_PAY_EVER: 0.3,
        Cause.WONT_PAY: 0.1,  # sum = 0.9
    }

    with pytest.raises(BeliefError):
        init(bad_prior)


def test_init_rejects_a_prior_missing_a_cause():
    """init() must raise BeliefError if a Cause is missing from the prior dict.
    A partial dict would be silently mis-normalised or confuse downstream code."""
    from src.policy.belief import init, BeliefError

    incomplete = {
        Cause.CANT_PAY_NOW: 0.5,
        Cause.CANT_PAY_EVER: 0.5,
        # WONT_PAY is missing
    }

    with pytest.raises(BeliefError):
        init(incomplete)


def test_init_rejects_a_negative_probability():
    """init() must raise BeliefError if any probability is negative."""
    from src.policy.belief import init, BeliefError

    bad_prior = {
        Cause.CANT_PAY_NOW: 0.6,
        Cause.CANT_PAY_EVER: -0.1,  # negative
        Cause.WONT_PAY: 0.5,
    }

    with pytest.raises(BeliefError):
        init(bad_prior)


def test_init_accepts_a_non_flat_prior():
    """init() must accept and normalize a non-uniform prior, such as the
    posterior from cause_map.prior() for a specific DeclineClass. This is the
    entire reason the explicit likelihood inversion exists: to make init() usable
    with a previously-observed posterior as the new starting point."""
    from src.policy.belief import init
    from src.classify.cause_map import prior as cause_map_prior

    # Use the posterior for CARD_EXPIRED as a prior
    new_prior = cause_map_prior(DeclineClass.CARD_EXPIRED)

    b = init(new_prior)

    # Should have preserved the probabilities
    for cause in Cause:
        assert b[cause] == pytest.approx(new_prior[cause], abs=1e-9)


# === Characterisation tests (measured, not tuned away) =====================

def test_three_insufficient_funds_declines_reach_99_6_percent():
    """Characterisation test: starting from a flat uniform prior and observing
    three INSUFFICIENT_FUNDS declines in a row, the belief's confidence in
    CANT_PAY_NOW reaches approximately 99.6%.

    The exact value (256/257 = 0.996108949416...) comes from:
    - Unnormalized posterior after 3 updates: (0.8^3, 0.1^3, 0.1^3)
    - = (0.512, 0.001, 0.001), sum = 0.514
    - Normalized: 0.512/0.514 = 256/257

    This is a CHARACTERISATION test, not an endorsement. It records what the
    pre-registered static-cause update actually does. Comment in §2's
    cause_switch_prob discussion (protocol.md lines 153-159) points out this
    overconfidence relative to actual cause stationarity."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")

    assert b[Cause.CANT_PAY_NOW] == pytest.approx(0.996108949416, abs=1e-9), \
        f"Three INSUFFICIENT_FUNDS: CANT_PAY_NOW = {b[Cause.CANT_PAY_NOW]}, " \
        f"expected 0.996108949416"


def test_cause_switch_survival_after_three_attempts_is_0_61():
    """Characterisation test: eval/frozen/sim_config.yaml line 99 pre-registers
    `cause_switch_prob: 0.15`, the per-attempt probability that a mandate's
    effective cause is redrawn and STAYS switched (Markov transition).

    Therefore, P(cause unchanged across three attempts) = (1 - 0.15)^3
    = 0.85^3 = 0.614125.

    This constant is used in test_static_cause_belief_is_overconfident_relative_to_cause_persistence()
    to measure the gap between what the belief update claims (0.9961 after 3
    identical declines) and what the cause-switching model allows (0.6141
    probability the cause even stayed the same)."""
    from src.policy.belief import CAUSE_ORDER

    # Read the pre-registered value from sim_config.yaml
    sim_config = _read_sim_config_source()
    assert "cause_switch_prob: 0.15" in sim_config, \
        f"Could not find 'cause_switch_prob: 0.15' in sim_config.yaml"

    # Verify the arithmetic
    cause_switch_prob = 0.15
    survival_prob = (1 - cause_switch_prob) ** 3
    assert survival_prob == pytest.approx(0.614125, abs=1e-9), \
        f"(1 - {cause_switch_prob})^3 = {survival_prob}, expected 0.614125"


def test_static_cause_belief_is_overconfident_relative_to_cause_persistence():
    """Measured gap: After three identical INSUFFICIENT_FUNDS declines, the
    belief update claims 99.61% confidence in CANT_PAY_NOW. But the
    cause_switch_prob=0.15 parameter (sim_config.yaml line 99) means the
    probability the cause even STAYED the same across all three attempts is
    only 0.85^3 = 61.41%.

    This gap was PRE-REGISTERED in protocol.md lines 153-159: '[cause_switch_prob]
    intended target is the within-mandate stationarity assumption the belief
    update relies on ... expect this arm to stress B7/B8 more than B5.'

    The system discloses and measures this gap rather than dampening or
    switch-leaking the update away. Assert both numbers explicitly: the
    overconfidence is real, and it is disclosed."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    # Get the belief after 3 identical declines
    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")

    belief_confidence = b[Cause.CANT_PAY_NOW]
    cause_survival = 0.614125

    # The gap must exist
    assert belief_confidence > cause_survival, \
        f"Belief confidence ({belief_confidence}) should exceed cause survival " \
        f"({cause_survival}), but it doesn't"

    # Pin both numbers
    assert belief_confidence == pytest.approx(0.996108949416, abs=1e-9), \
        f"Belief confidence = {belief_confidence}, expected 0.9961"
    assert cause_survival == pytest.approx(0.614125, abs=1e-9), \
        f"Cause survival = {cause_survival}, expected 0.6141"

    # Verify the gap is real
    gap = belief_confidence - cause_survival
    assert gap == pytest.approx(0.381983949416, abs=1e-9), \
        f"Gap = {gap}"


# === Quantisation tests ======================================================

def test_quantised_is_stable_and_collapses_near_identical_beliefs():
    """quantised(b, step) returns a tuple of ints. This is the property B8's
    backward-induction memoisation actually depends on (the build spec,
    memoisation key is `(quantised(b, 1e-6), r, ctx.signature())`) -- so this
    test constructs real belief pairs, not just a self-consistency check:

    - Two beliefs differing by 1e-6 (far less than step=0.01) must quantise
      to the SAME key, so B8's cache actually collapses near-duplicate
      belief states instead of treating each floating-point belief as a
      distinct, uncached one.
    - Two beliefs differing by 0.8 (far more than step) must quantise to
      DIFFERENT keys, so quantisation is not so coarse it conflates
      genuinely different beliefs (e.g. dominant-cause flips)."""
    from src.policy.belief import init, quantised, CAUSE_ORDER

    step = 0.01

    # Two beliefs 1e-6 apart -- must collapse to the same quantised key.
    close_a = init(dict(zip(CAUSE_ORDER, [1 / 3, 1 / 3, 1 / 3])))
    close_b = init(dict(zip(CAUSE_ORDER, [1 / 3 + 1e-6, 1 / 3 - 5e-7, 1 / 3 - 5e-7])))
    qa = quantised(close_a, step)
    qb = quantised(close_b, step)
    assert qa == qb, \
        f"beliefs 1e-6 apart (<< step={step}) quantised differently: {qa} vs {qb}"

    # Two beliefs 0.8 apart -- must quantise to different keys.
    far_a = init(dict(zip(CAUSE_ORDER, [0.9, 0.05, 0.05])))
    far_b = init(dict(zip(CAUSE_ORDER, [0.1, 0.45, 0.45])))
    qfar_a = quantised(far_a, step)
    qfar_b = quantised(far_b, step)
    assert qfar_a != qfar_b, \
        f"beliefs 0.8 apart (>> step={step}) quantised identically: {qfar_a}"

    # Determinism: repeated calls on the same belief give the same key.
    assert quantised(close_a, step) == quantised(close_a, step), \
        "quantised() is not deterministic across repeated calls"

    # Shape: a tuple of exactly 3 ints.
    q0 = quantised(close_a, step)
    assert isinstance(q0, tuple), f"quantised() returned {type(q0).__name__}, not tuple"
    assert len(q0) == 3, f"quantised() returned {len(q0)} elements, not 3"
    for val in q0:
        assert isinstance(val, int), f"quantised() element {val!r} is not int"


def test_dominant_returns_the_argmax_cause():
    """Belief.dominant() returns the Cause with the highest probability."""
    from src.policy.belief import init, update, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    # After INSUFFICIENT_FUNDS, CANT_PAY_NOW dominates
    b = update(b, DeclineClass.INSUFFICIENT_FUNDS, source_version="taxonomy=v1")
    assert b.dominant() == Cause.CANT_PAY_NOW, \
        f"After INSUFFICIENT_FUNDS, dominant should be CANT_PAY_NOW, got {b.dominant()}"

    # After CARD_EXPIRED, CANT_PAY_EVER dominates
    from src.policy.belief import init
    b2 = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                       REFERENCE_PRIOR)))
    b2 = update(b2, DeclineClass.CARD_EXPIRED, source_version="taxonomy=v1")
    assert b2.dominant() == Cause.CANT_PAY_EVER, \
        f"After CARD_EXPIRED, dominant should be CANT_PAY_EVER, got {b2.dominant()}"


def test_to_json_round_trips_and_is_a_string():
    """Belief.to_json() returns a JSON string that can be parsed back and
    recovers all three causes and the provenance field. The output is written
    to plan.belief_json TEXT NOT NULL (schema.sql line 19)."""
    from src.policy.belief import init, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    json_str = b.to_json()
    assert isinstance(json_str, str), f"to_json() returned {type(json_str).__name__}, not str"

    # Round-trip
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict), f"JSON parsed to {type(parsed).__name__}, not dict"

    # Must have all three causes
    for cause in Cause:
        assert cause in parsed, f"JSON missing cause {cause}"
        assert isinstance(parsed[cause], (int, float)), \
            f"JSON[{cause}] = {parsed[cause]!r}, not numeric"

    # Must have provenance
    assert "provenance" in parsed, f"JSON missing provenance field"


def test_provenance_names_both_versions():
    """Belief.provenance must contain both cause_map.PRIOR_VERSION (currently
    'v2') and REFERENCE_PRIOR_VERSION, so an auditor can trace the belief to
    both the normalizer that produced the decline classification AND the
    reference-prior convention in effect when the belief was created.

    Per the build spec §8.1 (B11 gate): 'normaliser output is versioned in
    the ledger before it can touch a belief ... A belief that cannot be
    traced to a specific normaliser version is not auditable.'"""
    from src.policy.belief import init, REFERENCE_PRIOR, REFERENCE_PRIOR_VERSION
    from src.classify.cause_map import PRIOR_VERSION

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    provenance = b.provenance
    assert isinstance(provenance, str), \
        f"provenance is {type(provenance).__name__}, not str"

    # Must name both version strings
    assert PRIOR_VERSION in provenance, \
        f"provenance '{provenance}' does not contain PRIOR_VERSION '{PRIOR_VERSION}'"
    assert REFERENCE_PRIOR_VERSION in provenance, \
        f"provenance '{provenance}' does not contain REFERENCE_PRIOR_VERSION '{REFERENCE_PRIOR_VERSION}'"


# === Module-level invariants =================================================

def test_belief_module_performs_no_io():
    """Per the build spec's B7 table, belief.py must NOT contain any I/O:
    no open(), psycopg, requests, import os, or datetime.now. It is a pure
    statistical computation layer."""
    from pathlib import Path

    source = Path("src/policy/belief.py").read_text()

    forbidden_patterns = [
        "open(",
        "psycopg",
        "requests",
        "import os",
        "datetime.now",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in source, \
            f"belief.py contains forbidden I/O pattern: {pattern}"


def test_as_dict_returns_cause_dict():
    """Belief.as_dict() returns a dict mapping each Cause to its probability."""
    from src.policy.belief import init, REFERENCE_PRIOR

    b = init(dict(zip([Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY],
                      REFERENCE_PRIOR)))

    d = b.as_dict()
    assert isinstance(d, dict), f"as_dict() returned {type(d).__name__}, not dict"

    # Must have all three causes
    for cause in Cause:
        assert cause in d, f"as_dict() missing {cause}"
        assert d[cause] == pytest.approx(b[cause], abs=1e-9), \
            f"as_dict()[{cause}] differs from __getitem__"


# === observe_terminal() tests ================================================
#
# CORRECTED, same session, before R2's gate was ticked: observe_terminal()'s
# first version took (b, cause, *, source_version) and always returned a
# DEGENERATE (1.0/0/0) posterior. The statistics review/the payments-domain review found that
# claim false against eval/frozen/sim_config.yaml's own generative process
# (P(CANT_PAY_EVER|DEAD) measures ~0.90, not 1.0) and additionally
# irreversible (cause_map._PRIORS has no zeros, so update() on an exact
# (0,1,0) belief can never move away from it). The signature changed to
# observe_terminal(cause_probs: Mapping[Cause, float], *, source_version) --
# no prior belief parameter at all (matching init()'s own shape, since the
# whole point is that the prior no longer matters), and the caller supplies
# a MEASURED distribution rather than the module assuming a degenerate one.
# These tests were rewritten to match, not just patched to compile.

_CPN, _CPE, _WP = Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY


def _measured_dead_probs() -> dict[Cause, float]:
    """A representative non-degenerate measured distribution, shaped like
    this module's own real TERMINAL_OBSERVED_CAUSE_PROBS[Outcome.DEAD] but
    defined locally so tests of the general observe_terminal() contract
    stay independent of that specific measured table."""
    return {_CPE: 0.90, _WP: 0.06, _CPN: 0.04}


# === TERMINAL_OBSERVED_CAUSE_PROBS / TERMINAL_OBSERVATION_SOURCE_VERSION ===
#
# R4, 2026-09-04 (reports/gates.md, "Post-B16 remediation gates"): relocated
# here from eval/run.py -- src/execute/cycle.py (R4) is the first PRODUCTION
# caller of observe_terminal(), and src/ must never import eval/, so a value
# only eval/run.py defined would be unreachable from there. eval/run.py now
# imports these under their original names as aliases; these tests pin the
# values and shape at their real home, not just at the alias.

def test_terminal_observed_cause_probs_has_exactly_dead_and_optout():
    from src.core.types import Outcome
    from src.policy.belief import TERMINAL_OBSERVED_CAUSE_PROBS

    assert set(TERMINAL_OBSERVED_CAUSE_PROBS.keys()) == {Outcome.DEAD, Outcome.OPTED_OUT}, \
        "RECOVERED must never appear here -- no cause left to decide once recovered"


@pytest.mark.parametrize("outcome_name, expected", [
    ("DEAD", {Cause.CANT_PAY_EVER: 0.8991, Cause.WONT_PAY: 0.0512, Cause.CANT_PAY_NOW: 0.0497}),
    ("OPTED_OUT", {Cause.WONT_PAY: 0.9040, Cause.CANT_PAY_NOW: 0.0684, Cause.CANT_PAY_EVER: 0.0276}),
])
def test_terminal_observed_cause_probs_matches_the_measured_table(outcome_name, expected):
    from src.core.types import Outcome
    from src.policy.belief import TERMINAL_OBSERVED_CAUSE_PROBS

    outcome = Outcome[outcome_name]
    probs = TERMINAL_OBSERVED_CAUSE_PROBS[outcome]
    assert set(probs.keys()) == set(Cause)
    for cause, value in expected.items():
        assert probs[cause] == pytest.approx(value, abs=1e-6)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_eval_run_aliases_the_same_objects_belief_defines():
    """Not merely equal-valued copies -- eval/run.py must reference belief.py's
    own objects, so a future edit to one cannot silently drift from the other."""
    import eval.run as run_mod
    from src.policy import belief as belief_mod

    assert run_mod._TERMINAL_OBSERVED_CAUSE_PROBS is belief_mod.TERMINAL_OBSERVED_CAUSE_PROBS
    assert run_mod.TERMINAL_OBSERVATION_SOURCE_VERSION == belief_mod.TERMINAL_OBSERVATION_SOURCE_VERSION


def test_observe_terminal_produces_the_measured_distribution_exactly():
    """observe_terminal(cause_probs, source_version=v) must return a Belief
    whose probabilities are EXACTLY cause_probs -- not a degenerate 1.0/0.0
    collapse. This is the corrected contract: the caller's measured
    distribution passes through unchanged, in CAUSE_ORDER."""
    from src.policy.belief import CAUSE_ORDER, observe_terminal

    probs = _measured_dead_probs()
    observed = observe_terminal(probs, source_version="test=v1")

    for cause in CAUSE_ORDER:
        assert observed[cause] == pytest.approx(probs[cause], abs=1e-9), (
            f"observe_terminal()[{cause}] = {observed[cause]}, "
            f"expected {probs[cause]} (the measured input, unchanged)"
        )


def test_observe_terminal_rejects_a_degenerate_distribution_only_if_invalid():
    """A caller MAY still pass an exactly degenerate distribution (1.0 on one
    cause) -- that is a valid distribution, just no longer the ONLY one this
    function can produce. observe_terminal() itself does not forbid it; it
    only validates the shape (sums to 1, non-negative, every Cause named)."""
    from src.policy.belief import CAUSE_ORDER, observe_terminal

    degenerate = {_CPE: 1.0, _WP: 0.0, _CPN: 0.0}
    observed = observe_terminal(degenerate, source_version="test=v1")
    assert observed[_CPE] == pytest.approx(1.0, abs=1e-9)
    assert observed[_WP] == pytest.approx(0.0, abs=1e-9)
    assert observed[_CPN] == pytest.approx(0.0, abs=1e-9)


def test_observe_terminal_rejects_missing_cause():
    """cause_probs must name every Cause exactly once -- same validation as
    init()'s own prior argument."""
    from src.policy.belief import BeliefError, observe_terminal

    incomplete = {_CPE: 0.9, _WP: 0.1}  # CANT_PAY_NOW missing
    with pytest.raises(BeliefError):
        observe_terminal(incomplete, source_version="test=v1")


def test_observe_terminal_rejects_negative_probability():
    from src.policy.belief import BeliefError, observe_terminal

    bad = {_CPE: 1.1, _WP: -0.1, _CPN: 0.0}
    with pytest.raises(BeliefError):
        observe_terminal(bad, source_version="test=v1")


def test_observe_terminal_rejects_probabilities_not_summing_to_one():
    from src.policy.belief import BeliefError, observe_terminal

    bad = {_CPE: 0.5, _WP: 0.3, _CPN: 0.1}  # sums to 0.9
    with pytest.raises(BeliefError):
        observe_terminal(bad, source_version="test=v1")


def test_observe_terminal_requires_source_version():
    """observe_terminal() must require the source_version keyword argument.
    Calling without it must raise TypeError because source_version has no
    default -- this enforces traceability of the observation source."""
    from src.policy.belief import observe_terminal

    with pytest.raises(TypeError):
        observe_terminal(_measured_dead_probs())


def test_observe_terminal_rejects_empty_source_version():
    """observe_terminal() must reject an empty string for source_version,
    raising BeliefError. The observation must be traceable to a source."""
    from src.policy.belief import BeliefError, observe_terminal

    with pytest.raises(BeliefError):
        observe_terminal(_measured_dead_probs(), source_version="")


def test_observe_terminal_provenance_records_terminal_observation():
    """The provenance string must contain the exact substring ';observed=terminal'
    to mark that this belief was created by observing a terminal fact, not by
    Bayesian updating from a decline signal."""
    from src.policy.belief import observe_terminal

    observed = observe_terminal(_measured_dead_probs(), source_version="test=v1")

    assert ";observed=terminal" in observed.provenance, \
        f"provenance '{observed.provenance}' missing ';observed=terminal'"


def test_observe_terminal_provenance_records_source_version():
    """The provenance string must also contain ';source=<source_version>' to
    record which observation source produced this terminal fact."""
    from src.policy.belief import observe_terminal

    source = "my-observation-v42"
    observed = observe_terminal(_measured_dead_probs(), source_version=source)

    assert f"source={source}" in observed.provenance, \
        f"provenance '{observed.provenance}' missing 'source={source}'"


def test_observe_terminal_provenance_still_contains_versions():
    """observe_terminal() adds the observation marker and source to provenance,
    but must NOT replace the existing cause_map and reference_prior version
    fields. The provenance is additive."""
    from src.policy.belief import REFERENCE_PRIOR_VERSION, observe_terminal
    from src.classify.cause_map import PRIOR_VERSION

    observed = observe_terminal(_measured_dead_probs(), source_version="test=v1")

    provenance = observed.provenance
    assert isinstance(provenance, str), \
        f"provenance is {type(provenance).__name__}, not str"

    assert f"cause_map={PRIOR_VERSION}" in provenance, \
        f"provenance '{provenance}' missing 'cause_map={PRIOR_VERSION}'"
    assert f"reference_prior={REFERENCE_PRIOR_VERSION}" in provenance, \
        f"provenance '{provenance}' missing 'reference_prior={REFERENCE_PRIOR_VERSION}'"
    assert "source=test=v1" in provenance, \
        f"provenance '{provenance}' missing 'source=test=v1'"
    assert ";observed=terminal" in provenance, \
        f"provenance '{provenance}' missing ';observed=terminal'"


def test_observe_terminal_is_order_independent_of_cause():
    """Two different measured distributions must produce different, but each
    individually deterministic, beliefs."""
    from src.policy.belief import observe_terminal

    probs_now = {_CPN: 0.90, _CPE: 0.06, _WP: 0.04}
    probs_ever = {_CPE: 0.90, _WP: 0.06, _CPN: 0.04}

    obs_now = observe_terminal(probs_now, source_version="test=v1")
    obs_ever = observe_terminal(probs_ever, source_version="test=v1")

    assert obs_now[_CPN] == pytest.approx(0.90, abs=1e-9)
    assert obs_ever[_CPE] == pytest.approx(0.90, abs=1e-9)
    assert obs_now != obs_ever, "Different measured distributions should produce different beliefs"


def test_observe_terminal_is_hashable():
    """The output of observe_terminal() must be hashable, just like any Belief,
    since B8's backward-induction memoisation uses quantised(b) as part of the key."""
    from src.policy.belief import observe_terminal

    observed = observe_terminal(_measured_dead_probs(), source_version="test=v1")

    h = hash(observed)
    assert isinstance(h, int), f"hash(observed) returned {type(h).__name__}, not int"


# === R5: update_from_likelihood_ratio ========================================
#
# The generic, evidence-agnostic Bayes update R5's intent channel needs.
# `scripts/guard_invariants.py`'s SRC_LLM_IMPORT forbids src/policy/ from
# importing src.llm in any form, so an intent score cannot arrive as an LLM
# call here -- it arrives as a plain declared likelihood ratio, computed by
# an adapter in src/execute/ (the layer already permitted to touch both
# sides). This function knows nothing about LLMs, nothing about Outcome,
# and nothing about DeclineClass: the CALLER declares the ratio.


def _lr(**kw) -> dict:
    return {Cause[k]: v for k, v in kw.items()}


def test_update_from_likelihood_ratio_applies_bayes():
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    out = update_from_likelihood_ratio(
        b, _lr(CANT_PAY_NOW=1.0, CANT_PAY_EVER=1.0, WONT_PAY=4.0),
        source_version="test-v1",
    )
    # Uniform prior x (1, 1, 4) -> (1/6, 1/6, 4/6).
    assert out[Cause.WONT_PAY] == pytest.approx(4.0 / 6.0)
    assert out[Cause.CANT_PAY_NOW] == pytest.approx(1.0 / 6.0)
    assert sum(out.probs) == pytest.approx(1.0)


def test_update_from_likelihood_ratio_is_scale_invariant():
    """Only ratios matter -- any factor common to all three causes cancels
    inside Bayes' rule, exactly as likelihood()'s deliberate lack of
    normalisation already relies on."""
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    a = update_from_likelihood_ratio(
        b, _lr(CANT_PAY_NOW=0.2, CANT_PAY_EVER=0.2, WONT_PAY=0.8), source_version="v")
    c = update_from_likelihood_ratio(
        b, _lr(CANT_PAY_NOW=20.0, CANT_PAY_EVER=20.0, WONT_PAY=80.0), source_version="v")
    assert a.probs == pytest.approx(c.probs)


def test_update_from_likelihood_ratio_matches_the_declineclass_path():
    """The generic function must agree with update() when handed exactly
    the likelihood vector update() would have used -- otherwise this is a
    second, silently different inference path rather than the same one
    with a wider input."""
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, init, likelihood, update,
        update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    dc = DeclineClass.CUSTOMER_DECLINED
    lik = likelihood(dc)
    via_generic = update_from_likelihood_ratio(
        b, dict(zip(CAUSE_ORDER, lik)), source_version="x")
    via_dc = update(b, dc, source_version="x")
    assert via_generic.probs == pytest.approx(via_dc.probs)


def test_update_from_likelihood_ratio_requires_every_cause():
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, BeliefError, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    with pytest.raises(BeliefError):
        update_from_likelihood_ratio(
            b, {Cause.WONT_PAY: 2.0}, source_version="v")


def test_update_from_likelihood_ratio_rejects_negative_and_all_zero():
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, BeliefError, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    with pytest.raises(BeliefError):
        update_from_likelihood_ratio(
            b, _lr(CANT_PAY_NOW=-1.0, CANT_PAY_EVER=1.0, WONT_PAY=1.0), source_version="v")
    with pytest.raises(BeliefError):
        update_from_likelihood_ratio(
            b, _lr(CANT_PAY_NOW=0.0, CANT_PAY_EVER=0.0, WONT_PAY=0.0), source_version="v")


def test_update_from_likelihood_ratio_requires_source_version():
    """Same discipline as update()/observe_terminal(): a belief whose
    evidence channel cannot be named is not auditable."""
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, BeliefError, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    with pytest.raises(BeliefError):
        update_from_likelihood_ratio(
            b, _lr(CANT_PAY_NOW=1.0, CANT_PAY_EVER=1.0, WONT_PAY=2.0), source_version="")


def test_update_from_likelihood_ratio_stamps_provenance():
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    out = update_from_likelihood_ratio(
        b, _lr(CANT_PAY_NOW=1.0, CANT_PAY_EVER=1.0, WONT_PAY=2.0),
        source_version="eval-intent-channel-v1")
    assert "source=eval-intent-channel-v1" in out.provenance
    assert "cause_map=" in out.provenance
    assert ";observed=terminal" not in out.provenance


def test_update_from_likelihood_ratio_cannot_reach_a_degenerate_belief():
    """A single observation must never drive a cause's posterior to exactly
    zero -- that is the absorbing state observe_terminal()'s own docstring
    records this project building and reversing on the same day. A finite,
    positive ratio cannot do it, and this pins that."""
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    for _ in range(20):
        b = update_from_likelihood_ratio(
            b, _lr(CANT_PAY_NOW=0.05, CANT_PAY_EVER=0.05, WONT_PAY=0.90),
            source_version="v")
    assert all(p > 0.0 for p in b.probs)
    assert b[Cause.WONT_PAY] > 0.99
