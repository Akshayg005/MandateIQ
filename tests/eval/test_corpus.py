"""
eval/corpus.py -- generate behaviour-policy exploring corpus for model training.

Design decision this file pins: every mandate_id is namespaced by seed to prevent
collision with the frozen evaluation batch (which uses TRAIN_SEEDS-disjoint seed);
the corpus filters AFA-cliff mandates out entirely rather than around or past them;
schedule draws are pre-committed before any attempt, giving the corpus real variance
on timing signals and avoiding the zero-variance ladder pathology. assert_legal()
is both checked at generation time and asserted in tests to machine-check the
feasibility constraints the allocator must respect.
"""
from __future__ import annotations

import pytest

from src.core.types import CensorReason, Outcome
from src.policy.constraints import afa_free_limit_paise, ELEVATED_AFA_CATEGORIES
from eval.corpus import (
    Episode,
    LegalityError,
    MIN_CELL_COUNT,
    TRAIN_SEEDS,
    assert_legal,
    generate,
    cell_counts,
    thin_cells,
    _above_afa_cliff,
    load_config,
)
from eval.frozen.simulator import SimMandate, Simulator, AttemptResult


# === Seed Namespace Tests ======================================================


def test_namespaced_mandate_ids_across_seeds():
    """Every mandate_id returned by generate() is namespaced f"s{seed}:{M_base}".
    For at least two different seeds, confirm the namespace prefix differs and
    that the same underlying M000x suffix appears under both prefixes."""
    # check_coverage=False: this test is about id namespacing, not corpus-
    # wide (cause x slot x bucket) coverage -- a single seed can easily miss
    # a thin cell (e.g. slot-4-in-window) by chance, which is expected here
    # and not a defect this test is checking for.
    episodes_seed1 = generate(seeds=(90001,), rng_seed=7, check_coverage=False)
    episodes_seed2 = generate(seeds=(90002,), rng_seed=7, check_coverage=False)  # Same rng_seed, different corpus seed

    # Collect mandate_id prefixes and suffixes
    ids_seed1 = {ep.mandate.mandate_id for ep in episodes_seed1}
    ids_seed2 = {ep.mandate.mandate_id for ep in episodes_seed2}

    # All IDs from seed 90001 should start with "s90001:"
    assert all(mid.startswith("s90001:") for mid in ids_seed1)
    assert all(mid.startswith("s90002:") for mid in ids_seed2)

    # Extract base mandate_ids (after the colon)
    base_ids_seed1 = {mid.split(":", 1)[1] for mid in ids_seed1}
    base_ids_seed2 = {mid.split(":", 1)[1] for mid in ids_seed2}

    # The base IDs should overlap (same underlying Simulator mandates, regenerated per seed)
    assert len(base_ids_seed1 & base_ids_seed2) > 0, "No overlap in base mandate IDs"


def test_load_config_seed_not_in_train_seeds():
    """Explicit check that load_config()["seed"] is not in TRAIN_SEEDS.
    This should pass as a proper test, not just silently at import time."""
    frozen_seed = load_config()["seed"]
    assert frozen_seed not in TRAIN_SEEDS, (
        f"Collision: frozen seed {frozen_seed} is in TRAIN_SEEDS {TRAIN_SEEDS}"
    )


# === Reproducibility ==========================================================


def test_generate_reproducible_given_seeds_and_rng_seed():
    """Calling generate() with the same seeds and rng_seed twice produces identical episodes."""
    # check_coverage=False: this test is about reproducibility, not
    # coverage -- a 2-seed sample can legitimately miss a thin cell by
    # chance, and doing so should not make this test flaky.
    episodes1 = generate(seeds=(90001, 90002), rng_seed=42, check_coverage=False)
    episodes2 = generate(seeds=(90001, 90002), rng_seed=42, check_coverage=False)

    assert len(episodes1) == len(episodes2)

    # Compare structurally (not object identity)
    for ep1, ep2 in zip(episodes1, episodes2):
        assert ep1.mandate.mandate_id == ep2.mandate.mandate_id
        assert ep1.mandate.amount_paise == ep2.mandate.amount_paise
        assert ep1.mandate.ceiling_paise == ep2.mandate.ceiling_paise
        assert len(ep1.attempts) == len(ep2.attempts)

        for att1, att2 in zip(ep1.attempts, ep2.attempts):
            assert att1.slot == att2.slot
            assert att1.on_day == att2.on_day
            assert att1.outcome == att2.outcome

        assert ep1.censor_reason == ep2.censor_reason


# === Censoring Diversity =======================================================


def test_both_censor_reasons_appear_in_default_generate():
    """The default generate() call (TRAIN_SEEDS) should include both
    WINDOW_CLOSED and BUDGET_EXHAUSTED censored episodes somewhere."""
    episodes = generate(seeds=TRAIN_SEEDS, rng_seed=1)

    censor_reasons = {ep.censor_reason for ep in episodes if ep.censor_reason != CensorReason.NONE}

    assert CensorReason.WINDOW_CLOSED in censor_reasons
    assert CensorReason.BUDGET_EXHAUSTED in censor_reasons


# === assert_legal Tests ========================================================


def test_assert_legal_passes_on_all_generated_episodes():
    """Every episode from generate() must pass assert_legal() without raising."""
    # check_coverage=False: this test is about per-episode legality, not
    # corpus-wide coverage -- unrelated properties, and a 2-seed sample can
    # legitimately miss a thin cell by chance.
    episodes = generate(seeds=(90001, 90002), rng_seed=5, check_coverage=False)

    for ep in episodes:
        # Should not raise
        assert_legal(ep)


def test_assert_legal_rejects_ceiling_below_amount():
    """assert_legal must raise LegalityError if ceiling_paise < amount_paise."""
    from src.core.types import Cause

    mandate = SimMandate(
        mandate_id="M_bad_ceiling",
        cycle_id=1,
        amount_paise=100_000,
        ceiling_paise=50_000,  # Violation: ceiling < amount
        category="subscription",
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )
    attempt = AttemptResult(
        mandate_id="M_bad_ceiling", slot=2, on_day=3, outcome=Outcome.RECOVERED
    )
    episode = Episode(mandate=mandate, attempts=(attempt,), censor_reason=CensorReason.NONE)

    with pytest.raises(LegalityError):
        assert_legal(episode)


def test_assert_legal_rejects_above_afa_cliff():
    """assert_legal must raise LegalityError if amount > AFA-free limit for category."""
    from src.core.types import Cause

    # Pick a category with a known limit
    category = "subscription"  # Uses base AFA_FREE_LIMIT_PAISE
    limit = afa_free_limit_paise(category)

    mandate = SimMandate(
        mandate_id="M_above_cliff",
        cycle_id=1,
        amount_paise=limit + 1,  # Just above the limit
        ceiling_paise=limit + 100_000,
        category=category,
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )
    attempt = AttemptResult(
        mandate_id="M_above_cliff", slot=2, on_day=3, outcome=Outcome.RECOVERED
    )
    episode = Episode(mandate=mandate, attempts=(attempt,), censor_reason=CensorReason.NONE)

    with pytest.raises(LegalityError):
        assert_legal(episode)


def test_assert_legal_rejects_non_increasing_on_day():
    """assert_legal must raise LegalityError if on_day is not strictly increasing."""
    from src.core.types import Cause

    mandate = SimMandate(
        mandate_id="M_bad_day_order",
        cycle_id=1,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )
    attempts = (
        AttemptResult(mandate_id="M_bad_day_order", slot=2, on_day=5, outcome=Outcome.STILL_PENDING),
        AttemptResult(mandate_id="M_bad_day_order", slot=3, on_day=4, outcome=Outcome.RECOVERED),  # on_day decreased
    )
    episode = Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.NONE)

    with pytest.raises(LegalityError):
        assert_legal(episode)


def test_assert_legal_rejects_first_on_day_below_1():
    """assert_legal must raise LegalityError if the first attempt's on_day < 1."""
    from src.core.types import Cause

    mandate = SimMandate(
        mandate_id="M_bad_first_day",
        cycle_id=1,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )
    attempts = (
        AttemptResult(mandate_id="M_bad_first_day", slot=2, on_day=0, outcome=Outcome.STILL_PENDING),  # Violation
    )
    episode = Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.NONE)

    with pytest.raises(LegalityError):
        assert_legal(episode)


# === AFA Filtering =============================================================


def test_no_above_afa_cliff_mandates_in_generate():
    """Every mandate in generate()'s output must have amount <= AFA-free limit for its category."""
    episodes = generate(seeds=TRAIN_SEEDS, rng_seed=3)

    for ep in episodes:
        limit = afa_free_limit_paise(ep.mandate.category)
        assert ep.mandate.amount_paise <= limit, (
            f"Mandate {ep.mandate.mandate_id} exceeds AFA limit: "
            f"amount={ep.mandate.amount_paise}, limit={limit}, category={ep.mandate.category}"
        )


# === Timing Variance ===========================================================


def test_timing_signals_have_variance_across_episodes():
    """The corpus must show variance in both in_salary_window status and
    days_since_last_attempt gaps. This is the regression test for the ladder's
    zero-variance pathology that led to this file's existence."""
    episodes = generate(seeds=TRAIN_SEEDS, rng_seed=2)

    # Collect all (in_salary_window, gap_days) from the corpus
    in_salary_window_vals = []
    gap_days_vals = []

    for ep in episodes:
        prev_day = 0
        for attempt in ep.attempts:
            # in_salary_window
            in_window = 1 <= attempt.on_day <= 5
            in_salary_window_vals.append(in_window)

            # days_since_last_attempt (gap)
            gap = attempt.on_day - prev_day
            gap_days_vals.append(gap)
            prev_day = attempt.on_day

    # Both should have more than one distinct value
    assert len(set(in_salary_window_vals)) > 1, (
        "in_salary_window has no variance (all True or all False)"
    )
    assert len(set(gap_days_vals)) > 1, (
        "days_since_last_attempt gaps are constant"
    )


# === cell_counts Tests =========================================================


def test_cell_counts_structure():
    """cell_counts() returns a dict keyed (cause_str, slot_int, day_bucket_int)
    where day_bucket is 0 (in_salary_window) or 1 (out), and sum of all counts
    equals total attempts across episodes."""
    # check_coverage=False: this test is about cell_counts()'s structure,
    # not corpus-wide coverage -- a 2-seed sample can legitimately miss a
    # thin cell by chance, and doing so should not make this test flaky.
    episodes = generate(seeds=(90003, 90004), rng_seed=1, check_coverage=False)
    counts = cell_counts(episodes)

    # Check structure
    assert isinstance(counts, dict)
    total_attempts = sum(counts.values())

    # Count total attempts manually
    expected_total = sum(len(ep.attempts) for ep in episodes)
    assert total_attempts == expected_total

    # Check that all keys have valid structure. NOTE: count is NOT asserted
    # > 0 here -- a cell can legitimately be 0 for a small input, and that
    # is exactly the state cell_counts() must be able to represent (see
    # test_cell_counts_full_grid_reported_even_when_a_cell_is_zero below).
    for (cause_str, slot_int, day_bucket_int), count in counts.items():
        assert isinstance(cause_str, str)
        assert isinstance(slot_int, int)
        assert isinstance(day_bucket_int, int)
        assert day_bucket_int in (0, 1)
        assert 2 <= slot_int <= 4
        assert count >= 0


def test_cell_counts_full_grid_reported_even_when_a_cell_is_zero():
    """cell_counts() must report all 18 (cause x slot x bucket) cells,
    including any that are genuinely 0 -- this is the exact bug
    stats-reviewer's B4 finding 3 caught: the old dict.get()-based
    accumulation only created a key for a cell that occurred at least once,
    so a truly empty cell was invisible to thin_cells(), which only filters
    counts.items(). A single episode with a single slot-2 attempt cannot
    possibly populate the other 17 cells, so this is a deterministic,
    non-flaky way to pin the full-grid guarantee."""
    from src.core.types import Cause

    mandate = SimMandate(
        mandate_id="M_single", cycle_id=1, amount_paise=50_000, ceiling_paise=100_000,
        category="subscription", household_id=None, initial_cause=Cause.CANT_PAY_NOW,
    )
    attempt = AttemptResult(mandate_id="M_single", slot=2, on_day=3, outcome=Outcome.RECOVERED)
    episode = Episode(mandate=mandate, attempts=(attempt,), censor_reason=CensorReason.NONE)

    counts = cell_counts([episode])

    assert len(counts) == 18  # 3 causes x 3 slots x 2 buckets, always
    assert counts[("CANT_PAY_NOW", 2, 0)] == 1
    zero_cells = [k for k, v in counts.items() if v == 0]
    assert len(zero_cells) == 17
    # And thin_cells() at threshold=1 must actually be able to see them --
    # this is what generate()'s hard refusal relies on.
    assert set(thin_cells(counts, threshold=1)) == set(zero_cells)


def test_cell_counts_day_bucket_only_0_or_1():
    """day_bucket in cell_counts must only ever be 0 (in_salary_window) or 1 (out)."""
    # check_coverage=False: this test is about the bucket VALUES cell_counts()
    # produces, not about whether every cell is populated -- a single seed
    # can legitimately miss a thin cell by chance.
    episodes = generate(seeds=(90005,), rng_seed=1, check_coverage=False)
    counts = cell_counts(episodes)

    for (cause_str, slot_int, day_bucket_int), count in counts.items():
        assert day_bucket_int in (0, 1), (
            f"Invalid day_bucket {day_bucket_int}; must be 0 or 1"
        )


# === generate()'s hard refusal on an empty cell ================================


def test_generate_raises_on_empty_cell(monkeypatch):
    """generate() must refuse (raise ValueError) rather than silently return
    a corpus with a zero-count cell. Forced deterministically by disabling
    the compressed-schedule component (COMPRESSED_FRAC=0.0) and using only
    2 seeds -- under wide-only draws, slot 3/4 landing in the salary window
    is vanishingly rare (this is literally the bug stats-reviewer found:
    0/365 slot-4-in-window rows under the pre-fix schedule), so this
    reliably reproduces an empty cell without depending on luck."""
    import eval.corpus as corpus_module

    monkeypatch.setattr(corpus_module, "COMPRESSED_FRAC", 0.0)
    with pytest.raises(ValueError, match="zero-count cell"):
        corpus_module.generate(seeds=(90001, 90002), rng_seed=1)


def test_generate_default_corpus_has_no_empty_cell():
    """Positive counterpart to the above: the real, shipped generate()
    defaults (COMPRESSED_FRAC=0.30 and all of TRAIN_SEEDS) must not raise,
    and cell_counts() over its output must show every one of the 18 cells
    with at least one attempt -- the actual fix, not just the guard."""
    episodes = generate(seeds=TRAIN_SEEDS, rng_seed=1)
    counts = cell_counts(episodes)
    assert len(counts) == 18
    assert min(counts.values()) > 0, (
        f"empty cell(s) found: {[k for k, v in counts.items() if v == 0]}"
    )


def test_generate_default_corpus_has_no_thin_cells():
    """At the original TRAIN_SEEDS size (10 seeds), slot-4-in-window was
    real but thin (7/14/11 observations per cause, below MIN_CELL_COUNT=20)
    -- disclosed in DECISIONS.md, 2026-08-28, B4, as a structural
    consequence of on_day being strictly increasing against a one-time,
    non-recurring salary window, not a remaining bug.

    TRAIN_SEEDS was widened 10 -> 40 on 2026-08-28 (DECISIONS.md, B5
    stats-reviewer entry, finding 2 -- unrelated motivation: a held-out
    log-loss claim needed more statistical power). That widening resolved
    this disclosed limitation as a side effect: all three cells now clear
    MIN_CELL_COUNT (22/52/38), and the corpus has zero thin cells anywhere.
    Pinned here, updated from the original (now-stale) assertion, so a
    future change to _draw_schedule or TRAIN_SEEDS that silently
    reintroduces thin cells is caught."""
    episodes = generate(seeds=TRAIN_SEEDS, rng_seed=1)
    counts = cell_counts(episodes)
    thin = set(thin_cells(counts))
    assert thin == set(), f"unexpected thin cell(s): {sorted(thin)}"
    slot4_in_window = {("CANT_PAY_EVER", 4, 0), ("CANT_PAY_NOW", 4, 0), ("WONT_PAY", 4, 0)}
    for key in slot4_in_window:
        assert counts[key] >= MIN_CELL_COUNT, (
            f"{key} has {counts[key]} observations, below MIN_CELL_COUNT={MIN_CELL_COUNT}"
        )


# === thin_cells Tests ==========================================================


def test_thin_cells_on_hand_built_dict():
    """thin_cells() returns cells below the threshold. Test on a hand-constructed
    counts dict with known thin and non-thin cells."""
    # Build a manual dict with some thin and some non-thin cells
    hand_built_counts = {
        ("CANT_PAY_NOW", 2, 0): 10,   # thin
        ("CANT_PAY_NOW", 2, 1): 30,   # not thin
        ("CANT_PAY_EVER", 3, 0): 50,  # not thin
        ("CANT_PAY_EVER", 3, 1): 5,   # thin
        ("WONT_PAY", 4, 0): 25,       # not thin
        ("WONT_PAY", 4, 1): 15,       # thin (less than MIN_CELL_COUNT=20)
    }

    thin = thin_cells(hand_built_counts, threshold=20)

    # Should include cells with count < 20
    thin_set = set(thin)
    assert ("CANT_PAY_NOW", 2, 0) in thin_set  # 10 < 20
    assert ("CANT_PAY_EVER", 3, 1) in thin_set  # 5 < 20
    assert ("WONT_PAY", 4, 1) in thin_set  # 15 < 20

    # Should NOT include cells with count >= 20
    assert ("CANT_PAY_NOW", 2, 1) not in thin_set  # 30 >= 20
    assert ("CANT_PAY_EVER", 3, 0) not in thin_set  # 50 >= 20
    assert ("WONT_PAY", 4, 0) not in thin_set  # 25 >= 20


def test_thin_cells_threshold_parameter():
    """thin_cells() respects the threshold parameter."""
    hand_built = {
        ("CANT_PAY_NOW", 2, 0): 25,
        ("CANT_PAY_NOW", 3, 1): 15,
    }

    thin_strict = thin_cells(hand_built, threshold=20)
    thin_loose = thin_cells(hand_built, threshold=10)

    # At threshold 20, only the 15-count cell is thin
    assert ("CANT_PAY_NOW", 3, 1) in thin_strict
    assert ("CANT_PAY_NOW", 2, 0) not in thin_strict

    # At threshold 10, both are acceptable
    assert ("CANT_PAY_NOW", 3, 1) not in thin_loose


# === Edge Cases ================================================================


def test_generate_with_empty_seeds_returns_empty():
    """Calling generate with seeds=() should return an empty list."""
    episodes = generate(seeds=(), rng_seed=99)
    assert isinstance(episodes, list)
    assert len(episodes) == 0


def test_assert_legal_on_zero_attempt_episode():
    """assert_legal() should not raise on an episode with no attempts."""
    from src.core.types import Cause

    mandate = SimMandate(
        mandate_id="M_zero_attempts",
        cycle_id=1,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )
    episode = Episode(mandate=mandate, attempts=(), censor_reason=CensorReason.WINDOW_CLOSED)

    # Should not raise
    assert_legal(episode)
