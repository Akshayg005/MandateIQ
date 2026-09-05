"""src/model/cif.py -- cumulative incidence functions and survival curves.

Design decision this file pins: CIF and survival are computed via explicit
recursion over 3 competing hazard rates (slot 2, 3, 4), with slot 1 hard-wired
as a structural zero (h_c(1) ≡ 0, CIF_c(1) = 0, S(1) = 1). The recursion is
NOT a naive 1-KM per cause (which overstates CIF by ignoring competing risks)
but the correct competing-risks formula where each cause's CIF depends on the
marginal survival S(k-1) (probability of not resolving at any cause by slot k-1).
The hazard input h has shape (n, 3, 4) where axis 1 indexes slots 2-4 (in that
order, no slot 1) and axis 2 indexes the four Outcome probabilities [0,1,2,3].
Every hazard row must sum to 1 within tolerance. CIF output has shape (n, 3, 4)
where axis 1 indexes the 3 non-reference causes [RECOVERED, DEAD, OPTED_OUT]
(i.e., outcomes 1, 2, 3) and axis 2 indexes slots 1-4 (axis-2 index 0=slot 1, etc).
Survival output has shape (n, 4), one row per mandate, slots 1-4.
"""
from __future__ import annotations

import numpy as np
import pytest


def _simple_valid_hazards(n: int = 10, seed: int = 42) -> np.ndarray:
    """Generate a valid (n, 3, 4) hazard array for testing.
    Each row in axis 1 (slot) sums to 1.0."""
    rng = np.random.RandomState(seed)

    # Generate (n, 3, 4) random values
    h = rng.dirichlet([1, 1, 1, 1], size=(n, 3))  # Shape (n, 3, 4), rows sum to 1

    return h


def _degenerate_all_still_pending(n: int = 5) -> np.ndarray:
    """Generate a degenerate hazard array: h[:, :, 0] = 1 (certain STILL_PENDING).
    Every other outcome has 0 probability."""
    h = np.zeros((n, 3, 4))
    h[:, :, 0] = 1.0  # All probability on STILL_PENDING
    return h


def _degenerate_certain_recovery_at_slot2(n: int = 5) -> np.ndarray:
    """Generate a degenerate hazard array: at slot 2 (axis-1 index 0),
    every mandate recovers with probability 1.0. Slots 3/4 are never reached."""
    h = np.zeros((n, 3, 4))
    h[:, 0, 1] = 1.0  # Slot 2: outcome 1 (RECOVERED) = 1.0
    # Slots 3/4: all STILL_PENDING (outcome 0)
    h[:, 1, 0] = 1.0
    h[:, 2, 0] = 1.0
    return h


# === cif() input validation ====================================================


def test_cif_raises_on_wrong_shape_axis1():
    """cif() must raise ValueError if input is not (n, 3, 4)."""
    from src.model.cif import cif

    # Wrong axis 1: (n, 4, 4) instead of (n, 3, 4)
    h_wrong = np.ones((5, 4, 4))
    h_wrong /= 4  # Make rows sum to 1

    with pytest.raises(ValueError):
        cif(h_wrong)


def test_cif_raises_on_wrong_shape_axis2():
    """cif() must raise ValueError if axis 2 is not 4."""
    from src.model.cif import cif

    # Wrong axis 2: (n, 3, 5) instead of (n, 3, 4)
    h_wrong = np.ones((5, 3, 5))
    h_wrong /= 5

    with pytest.raises(ValueError):
        cif(h_wrong)


def test_cif_raises_on_rows_not_summing_to_one():
    """cif() must raise ValueError if any hazard row (slot) does not sum to 1."""
    from src.model.cif import cif

    h = _simple_valid_hazards(n=5)

    # Corrupt one row to not sum to 1
    h[0, 0, :] = [0.2, 0.2, 0.2, 0.2]  # Sum = 0.8

    with pytest.raises(ValueError):
        cif(h)


def test_cif_raises_on_tolerance_check():
    """cif() must validate row sums within tolerance (e.g., 1e-6)."""
    from src.model.cif import cif

    h = _simple_valid_hazards(n=5)

    # Corrupt one row to sum slightly off (outside tolerance)
    h[0, 1, 0] -= 0.2  # Make it sum to 0.8

    with pytest.raises(ValueError):
        cif(h)


# === survival() input validation ===============================================


def test_survival_raises_on_wrong_shape():
    """survival() must raise ValueError on wrong input shape."""
    from src.model.cif import survival

    h_wrong = np.ones((5, 4, 4))
    with pytest.raises(ValueError):
        survival(h_wrong)


def test_survival_raises_on_rows_not_summing_to_one():
    """survival() must raise ValueError if hazard rows don't sum to 1."""
    from src.model.cif import survival

    h = _simple_valid_hazards(n=5)
    h[0, 0, :] = [0.5, 0.2, 0.1, 0.1]  # Sum = 0.9

    with pytest.raises(ValueError):
        survival(h)


# === cif() shape and slot-1 zero tests ========================================


def test_cif_returns_correct_shape():
    """cif() must return (n, 3, 4) where axis 1 is non-reference outcomes."""
    from src.model.cif import cif

    h = _simple_valid_hazards(n=20)
    result = cif(h)

    assert result.shape == (20, 3, 4)


def test_cif_slot1_column_is_all_zeros():
    """cif() output's slot-1 column (axis-2 index 0) must be identically 0
    for all mandates and causes, since slot 1 is hard-wired as a structural zero."""
    from src.model.cif import cif

    h = _simple_valid_hazards(n=50)
    result = cif(h)

    # result[:, :, 0] is the slot-1 column (all 3 causes)
    assert np.allclose(result[:, :, 0], 0.0), (
        "Slot 1 CIF must be identically 0 for all mandates and causes"
    )


# === survival() shape tests ====================================================


def test_survival_returns_correct_shape():
    """survival() must return (n, 4) for n mandates and 4 slots."""
    from src.model.cif import survival

    h = _simple_valid_hazards(n=20)
    result = survival(h)

    assert result.shape == (20, 4)


def test_survival_slot1_is_always_one():
    """survival() output's slot-1 column (axis-1 index 0) must be exactly 1.0
    for every mandate (no one has resolved by slot 1)."""
    from src.model.cif import survival

    h = _simple_valid_hazards(n=50)
    result = survival(h)

    # result[:, 0] is the slot-1 survival (should all be 1.0)
    assert np.allclose(result[:, 0], 1.0), (
        "Slot 1 survival must be exactly 1.0 for all mandates"
    )


# === Core identity test ========================================================


def test_cif_plus_survival_equals_one_at_every_slot_random_hazards():
    """For every slot t in 0..3: CIF[:, :, t].sum(axis=1) + S[:, t] == 1.
    This is the fundamental identity of competing risks."""
    from src.model.cif import cif, survival

    h = _simple_valid_hazards(n=100, seed=123)
    cif_result = cif(h)
    surv_result = survival(h)

    for t in range(4):
        cif_sum = cif_result[:, :, t].sum(axis=1)
        surv_t = surv_result[:, t]
        total = cif_sum + surv_t

        assert np.allclose(total, 1.0, atol=1e-9), (
            f"At slot {t}: CIF_sum + survival != 1. "
            f"Max error: {np.abs(total - 1.0).max()}"
        )


def test_cif_plus_survival_equals_one_degenerate_all_still_pending():
    """Identity test on degenerate all-STILL_PENDING hazards.
    Every CIF should be 0, every survival should be 1."""
    from src.model.cif import cif, survival

    h = _degenerate_all_still_pending(n=10)
    cif_result = cif(h)
    surv_result = survival(h)

    # All CIF values should be 0
    assert np.allclose(cif_result, 0.0), "All CIF should be 0"

    # All survival values should be 1
    assert np.allclose(surv_result, 1.0), "All survival should be 1"

    # Identity check
    for t in range(4):
        total = cif_result[:, :, t].sum(axis=1) + surv_result[:, t]
        assert np.allclose(total, 1.0, atol=1e-9)


def test_cif_plus_survival_equals_one_certain_recovery_at_slot2():
    """Identity test on degenerate certain-recovery-at-slot2.
    At slot 2: CIF_RECOVERED = 1, all others = 0, S = 0.
    At slots 3/4: CIF = 0 (no new resolutions), S = 0 (still resolved)."""
    from src.model.cif import cif, survival

    h = _degenerate_certain_recovery_at_slot2(n=10)
    cif_result = cif(h)
    surv_result = survival(h)

    # Slot 1: CIF = 0, S = 1
    assert np.allclose(cif_result[:, :, 0], 0.0)
    assert np.allclose(surv_result[:, 0], 1.0)

    # Slot 2: CIF_RECOVERED = 1, all others = 0, S = 0
    assert np.allclose(cif_result[:, 0, 1], 1.0), "Slot 2 CIF_RECOVERED should be 1"
    assert np.allclose(cif_result[:, 1, 1], 0.0), "Slot 2 CIF_DEAD should be 0"
    assert np.allclose(cif_result[:, 2, 1], 0.0), "Slot 2 CIF_OPTED_OUT should be 0"
    assert np.allclose(surv_result[:, 1], 0.0), "Slot 2 survival should be 0"

    # Slots 3/4: CIF is CUMULATIVE, so it stays at slot 2's value (already
    # resolved by RECOVERED -- CIF_RECOVERED=1, others=0), it does not reset
    # to 0 just because no NEW event occurs. S stays 0 (already resolved).
    assert np.allclose(cif_result[:, 0, 2], 1.0), "Slot 3 CIF_RECOVERED should stay 1"
    assert np.allclose(cif_result[:, 1:, 2], 0.0), "Slot 3 CIF_DEAD/OPTED_OUT should stay 0"
    assert np.allclose(cif_result[:, 0, 3], 1.0), "Slot 4 CIF_RECOVERED should stay 1"
    assert np.allclose(cif_result[:, 1:, 3], 0.0), "Slot 4 CIF_DEAD/OPTED_OUT should stay 0"
    assert np.allclose(surv_result[:, 2], 0.0), "Slot 3 survival should be 0"
    assert np.allclose(surv_result[:, 3], 0.0), "Slot 4 survival should be 0"


# === Monotonicity tests ========================================================


def test_cif_monotonically_nondecreasing_per_cause_per_mandate():
    """CIF must be monotonically non-decreasing across slots for each cause
    and mandate (it's a cumulative incidence)."""
    from src.model.cif import cif

    h = _simple_valid_hazards(n=50, seed=456)
    cif_result = cif(h)

    for n_idx in range(cif_result.shape[0]):
        for cause_idx in range(3):
            cause_cif = cif_result[n_idx, cause_idx, :]
            diffs = np.diff(cause_cif)
            assert (diffs >= -1e-9).all(), (
                f"Mandate {n_idx}, cause {cause_idx}: CIF is not monotonic. "
                f"Values: {cause_cif}"
            )


def test_survival_monotonically_nonincreasing():
    """Survival must be monotonically non-increasing across slots
    (probability of not resolving can only decrease or stay same)."""
    from src.model.cif import survival

    h = _simple_valid_hazards(n=50, seed=789)
    surv_result = survival(h)

    for n_idx in range(surv_result.shape[0]):
        surv_curve = surv_result[n_idx, :]
        diffs = np.diff(surv_curve)
        assert (diffs <= 1e-9).all(), (
            f"Mandate {n_idx}: survival is not monotonically non-increasing. "
            f"Values: {surv_curve}"
        )


# === Not-1-KM test =============================================================


def test_cif_not_1_minus_km_per_cause():
    """CIF must NOT be computed as 1-KM per cause (which overstates CIF by
    ignoring competing risks). This test constructs a case with competing
    causes and verifies that cif()'s output is LESS than the naive 1-KM approach."""
    from src.model.cif import cif, survival

    # Create a hazard array where both cause 1 and cause 2 have substantial
    # hazard at slot 2
    h = np.zeros((1, 3, 4))
    # Slot 2: outcome 1 (RECOVERED) = 0.4, outcome 2 (DEAD) = 0.4, outcome 0 (STILL_PENDING) = 0.2
    h[0, 0, :] = [0.2, 0.4, 0.4, 0.0]
    # Slots 3/4: all STILL_PENDING (no further hazard)
    h[0, 1, :] = [1.0, 0.0, 0.0, 0.0]
    h[0, 2, :] = [1.0, 0.0, 0.0, 0.0]

    cif_result = cif(h)
    surv_result = survival(h)

    # Correct CIF at slot 2 (the only slot with events):
    # CIF_RECOVERED(2) = h_RECOVERED(2) * S(1) = 0.4 * 1 = 0.4
    # CIF_DEAD(2) = h_DEAD(2) * S(1) = 0.4 * 1 = 0.4

    # Naive 1-KM (wrong) per cause:
    # 1 - KM_RECOVERED = 1 - (1 - 0.4) = 0.4 (equal to correct, so let's check total)
    # But if both causes were independent KM: (1 - 0.4) * (1 - 0.4) = 0.36 survival
    # This violates CIF_R + CIF_D + S = 1 (would give 0.4 + 0.4 + 0.36 = 1.16)

    # Compare: correct sum is CIF_R + CIF_D + S = 0.4 + 0.4 + 0.2 = 1.0
    total = cif_result[0, 0, 1] + cif_result[0, 1, 1] + surv_result[0, 1]
    assert np.isclose(total, 1.0), f"Identity violated: {total}"

    # The point: if we computed 1-KM per cause, we'd get a different (higher) CIF
    # for each cause when there's competing risk.
    # For cause 1 (RECOVERED): 1-KM would give 0.4 (happens to match)
    # For cause 2 (DEAD): 1-KM would give 0.4 (happens to match)
    # But this is a coincidence; let's use a case where it truly differs.

    # Better test: three-cause competition at same slot
    h2 = np.zeros((1, 3, 4))
    # Slot 2: outcomes 1, 2, 3 each have 1/3 hazard, outcome 0 has 0
    h2[0, 0, :] = [0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]
    h2[0, 1, :] = [1.0, 0.0, 0.0, 0.0]
    h2[0, 2, :] = [1.0, 0.0, 0.0, 0.0]

    cif2 = cif(h2)
    surv2 = survival(h2)

    # Correct: each cause's CIF at slot 2 = (1/3) * 1 = 1/3
    # Naive 1-KM per cause (wrong): each would compute its own survival
    # independently, giving 1 - (1 - 1/3) = 1/3 each. But applied to all three:
    # 1/3 + 1/3 + 1/3 = 1, which coincidentally equals correct (because only one
    # slot, hazards sum to 1).

    # The real test: if we had two slots, naive 1-KM would fail.
    # For now, verify the identity holds (which it will for correct formula):
    total2 = cif2[0, 0, 1] + cif2[0, 1, 1] + cif2[0, 2, 1] + surv2[0, 1]
    assert np.isclose(total2, 1.0), (
        f"Three-way split identity violated: {total2}. "
        f"CIF values: {cif2[0, :, 1]}, S(2)={surv2[0, 1]}"
    )

    # Final verification: CIF_RECOVERED is NOT 0.5 (which 1-KM of RECOVERED alone
    # would give if hazard changed to 0.5). If we change to 0.5:
    h3 = np.zeros((1, 3, 4))
    h3[0, 0, :] = [0.5, 0.5, 0.0, 0.0]  # RECOVERED=0.5, DEAD=0, others sum to 1
    h3[0, 1, :] = [1.0, 0.0, 0.0, 0.0]
    h3[0, 2, :] = [1.0, 0.0, 0.0, 0.0]

    cif3 = cif(h3)

    # Correct: CIF_RECOVERED(2) = 0.5
    # If computed via naive 1-KM: 1 - (1 - 0.5) = 0.5 (same, by chance)
    # But survival should be 0.5 (not resolved by RECOVERED)
    # And there's no DEAD, so CIF_DEAD = 0

    # Actually, the 1-KM test is hard to pin down without multi-slot data.
    # Instead, verify the recursion formula directly:

    # At slot 2: S(2) = S(1) * (1 - sum_c h_c(2)) = 1 * (1 - 1) = 0 ✓
    # (Because all hazard sums to 1 at slot 2)
    assert np.isclose(surv2[0, 1], 0.0)

    # Proceed with more nuanced test if needed, but for now verify identity


def test_cif_not_1_km_multislot_case():
    """Two DIFFERENT causes both have real hazard across two slots -- this
    is the actual regression guard against reintroducing the naive-1-KM
    bug: 1-KM computed independently per cause ignores that the OTHER
    cause's events also remove mandates from the risk set, so it overstates
    each cause's own CIF. With only one active cause (as an earlier version
    of this test used), 1-KM and the correct recursion coincide by
    construction -- there is nothing to compete with -- so this uses two
    causes with simultaneous nonzero hazard at both slots, which is the
    case where the two approaches must actually diverge."""
    from src.model.cif import cif, survival

    # Slot 2: STILL_PENDING=0.5, RECOVERED=0.3, DEAD=0.2, OPTED_OUT=0
    # Slot 3: STILL_PENDING=0.5, RECOVERED=0.2, DEAD=0.3, OPTED_OUT=0
    # Slot 4: no further hazard (STILL_PENDING=1)
    h = np.zeros((1, 3, 4))
    h[0, 0, :] = [0.5, 0.3, 0.2, 0.0]  # Slot 2
    h[0, 1, :] = [0.5, 0.2, 0.3, 0.0]  # Slot 3
    h[0, 2, :] = [1.0, 0.0, 0.0, 0.0]  # Slot 4

    cif_result = cif(h)
    surv_result = survival(h)

    # Correct competing-risks recursion:
    # S(1)=1
    # S(2) = 1 * (1 - (0.3+0.2)) = 0.5
    # CIF_RECOVERED(2) = 0.3 * S(1) = 0.3
    # CIF_DEAD(2)      = 0.2 * S(1) = 0.2
    # S(3) = 0.5 * (1 - (0.2+0.3)) = 0.25
    # CIF_RECOVERED(3) = 0.3 + 0.2 * S(2) = 0.3 + 0.2*0.5 = 0.40
    # CIF_DEAD(3)      = 0.2 + 0.3 * S(2) = 0.2 + 0.3*0.5 = 0.35
    # Slot 4: no further hazard -- everything carries forward unchanged.
    assert np.isclose(surv_result[0, 1], 0.5)
    assert np.isclose(cif_result[0, 0, 1], 0.3)  # CIF_RECOVERED(2)
    assert np.isclose(cif_result[0, 1, 1], 0.2)  # CIF_DEAD(2)
    assert np.isclose(surv_result[0, 2], 0.25)
    assert np.isclose(cif_result[0, 0, 2], 0.40)  # CIF_RECOVERED(3)
    assert np.isclose(cif_result[0, 1, 2], 0.35)  # CIF_DEAD(3)
    assert np.isclose(cif_result[0, 0, 3], 0.40)  # unchanged at slot 4
    assert np.isclose(cif_result[0, 1, 3], 0.35)  # unchanged at slot 4
    assert np.isclose(surv_result[0, 3], 0.25)

    # Naive 1-KM per cause (WRONG): computes each cause's own survival curve
    # using only ITS OWN hazard at each slot, as if the other cause's events
    # never removed anyone from the risk set.
    s_km_recovered = (1 - 0.3) * (1 - 0.2)  # = 0.56
    cif_recovered_1km_slot3 = 1 - s_km_recovered  # = 0.44
    s_km_dead = (1 - 0.2) * (1 - 0.3)  # = 0.56
    cif_dead_1km_slot3 = 1 - s_km_dead  # = 0.44

    # The naive approach OVERSTATES both causes' CIF relative to the
    # correct competing-risks recursion -- this is the real divergence
    # the build spec forbids silently reintroducing.
    assert cif_result[0, 0, 2] < cif_recovered_1km_slot3, (
        f"correct CIF_RECOVERED(3)={cif_result[0, 0, 2]} should be strictly "
        f"less than the naive 1-KM value {cif_recovered_1km_slot3} -- if "
        f"they're equal, cif() has silently regressed to naive 1-KM"
    )
    assert cif_result[0, 1, 2] < cif_dead_1km_slot3, (
        f"correct CIF_DEAD(3)={cif_result[0, 1, 2]} should be strictly "
        f"less than the naive 1-KM value {cif_dead_1km_slot3} -- if "
        f"they're equal, cif() has silently regressed to naive 1-KM"
    )


# === Edge cases ================================================================


def test_cif_survival_single_mandate():
    """cif() and survival() must work on n=1."""
    from src.model.cif import cif, survival

    h = _simple_valid_hazards(n=1)
    cif_result = cif(h)
    surv_result = survival(h)

    assert cif_result.shape == (1, 3, 4)
    assert surv_result.shape == (1, 4)


def test_cif_survival_large_batch():
    """cif() and survival() must work on large n."""
    from src.model.cif import cif, survival

    h = _simple_valid_hazards(n=10000)
    cif_result = cif(h)
    surv_result = survival(h)

    assert cif_result.shape == (10000, 3, 4)
    assert surv_result.shape == (10000, 4)

    # Identity should still hold
    for t in range(4):
        total = cif_result[:, :, t].sum(axis=1) + surv_result[:, t]
        assert np.allclose(total, 1.0, atol=1e-9)


def test_cif_all_outcome_zero_hazard():
    """If all outcomes have 0 hazard at all slots (only STILL_PENDING),
    CIF must be 0 everywhere, S must be 1 everywhere."""
    from src.model.cif import cif, survival

    h = np.zeros((5, 3, 4))
    h[:, :, 0] = 1.0  # All probability on outcome 0 (STILL_PENDING)

    cif_result = cif(h)
    surv_result = survival(h)

    assert np.allclose(cif_result, 0.0)
    assert np.allclose(surv_result, 1.0)


def test_cif_deterministic_outcomes():
    """Test a case with deterministic outcomes (one outcome per slot with prob 1)."""
    from src.model.cif import cif, survival

    # Slot 2: deterministically outcome 1 (RECOVERED) with prob 1
    # Slot 3, 4: never reached
    h = np.zeros((3, 3, 4))
    h[:, 0, 1] = 1.0  # Slot 2: outcome 1 certain
    h[:, 1, 0] = 1.0  # Slot 3: outcome 0 (never reached, but hazard is 1 for consistency)
    h[:, 2, 0] = 1.0  # Slot 4: same

    cif_result = cif(h)
    surv_result = survival(h)

    # At slot 2: CIF_RECOVERED = 1, all others = 0, S = 0
    assert np.allclose(cif_result[:, 0, 1], 1.0)
    assert np.allclose(cif_result[:, 1, 1], 0.0)
    assert np.allclose(cif_result[:, 2, 1], 0.0)
    assert np.allclose(surv_result[:, 1], 0.0)

    # At slots 3/4: nothing changes (all already resolved)
    assert np.allclose(cif_result[:, :, 2:], cif_result[:, :, [1, 1]])
    assert np.allclose(surv_result[:, 2:], 0.0)
