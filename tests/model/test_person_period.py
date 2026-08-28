"""
src/model/person_period.py -- reshape episodes into one row per (mandate, cycle, slot).

Design decision this file pins: slot 1 is synthesized (not read from episode.attempts),
every at-risk slot gets exactly one row, and terminal/censoring status is encoded
precisely per PLAN_DETAIL.md section 2 to enable correct likelihood for a right-censored
competing-risks model. The anti-pattern `y = (df.outcome == RECOVERED).astype(int)` is
guarded against at the frame level, not silently passed through to a later fit().
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.ids import row_id
from src.core.types import CensorReason, Outcome
from eval.corpus import Episode
from eval.frozen.simulator import AttemptResult, SimMandate
from src.model.person_period import EMITTED_COLUMNS, FrameError, build, validate


def _mandate(
    mandate_id: str = "M_0412",
    cycle_id: int = 7,
    amount_paise: int = 50_000,
    ceiling_paise: int = 100_000,
    category: str = "subscription",
) -> SimMandate:
    """Helper to build a SimMandate for test use."""
    from src.core.types import Cause

    return SimMandate(
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        amount_paise=amount_paise,
        ceiling_paise=ceiling_paise,
        category=category,
        household_id=None,
        initial_cause=Cause.CANT_PAY_NOW,
    )


def _attempt(
    mandate_id: str,
    slot: int,
    on_day: int,
    outcome: Outcome = Outcome.STILL_PENDING,
) -> AttemptResult:
    """Helper to build an AttemptResult for test use."""
    return AttemptResult(
        mandate_id=mandate_id,
        slot=slot,
        on_day=on_day,
        outcome=outcome,
        iatrogenic_insufficient_funds=False,
    )


# === Worked Examples ===========================================================


def test_worked_example_a_recovered_at_slot_3_with_3_rows():
    """Mandate M_0412, cycle 7: slot 2 STILL_PENDING, slot 3 RECOVERED.
    Expected 3 rows total (slot 1 synthesized + 2 from attempts).
    Slot 3 is terminal and not censored (resolved episode)."""
    mandate = _mandate("M_0412", cycle_id=7)
    attempts = (
        _attempt("M_0412", slot=2, on_day=3, outcome=Outcome.STILL_PENDING),
        _attempt("M_0412", slot=3, on_day=9, outcome=Outcome.RECOVERED),
    )
    episode = Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.NONE)

    df = build([episode])
    assert len(df) == 3
    assert set(df["slot"].unique()) == {1, 2, 3}

    # Slot 1: synthesized
    # bool(...) before `is` because a homogeneous bool-dtype pandas column
    # yields numpy.bool_ scalars on extraction, not the Python True/False
    # singletons -- bool() always normalizes to one of the two singletons,
    # value comparisons like `== True` would work too but `is` is stricter
    # and this keeps that strictness rather than loosening the assertion.
    row1 = df[df["slot"] == 1].iloc[0]
    assert row1["mandate_id"] == "M_0412"
    assert row1["cycle_id"] == 7
    assert row1["outcome"] == Outcome.STILL_PENDING
    assert row1["event_code"] == 0
    assert bool(row1["at_risk"]) is True
    assert bool(row1["censored"]) is False
    assert row1["censor_reason"] == CensorReason.NONE
    assert bool(row1["is_terminal"]) is False
    assert bool(row1["estimable"]) is False
    assert row1["row_id"] == row_id("M_0412", 7, 1)

    # Slot 2: from attempts[0]
    row2 = df[df["slot"] == 2].iloc[0]
    assert row2["outcome"] == Outcome.STILL_PENDING
    assert bool(row2["is_terminal"]) is False
    assert bool(row2["censored"]) is False
    assert bool(row2["estimable"]) is True
    assert row2["on_day"] == 3

    # Slot 3: terminal, recovered, not censored
    row3 = df[df["slot"] == 3].iloc[0]
    assert row3["outcome"] == Outcome.RECOVERED
    assert row3["event_code"] == 1
    assert bool(row3["is_terminal"]) is True
    assert bool(row3["censored"]) is False
    assert bool(row3["estimable"]) is True
    assert row3["censor_reason"] == CensorReason.NONE
    assert row3["on_day"] == 9


def test_worked_example_b_exhausted_4_slots_with_censoring():
    """Mandate M_0887, cycle 3: slot 2-4 all STILL_PENDING (never resolved).
    Expected 4 rows total. Slot 4 is terminal, censored, with BUDGET_EXHAUSTED reason."""
    mandate = _mandate("M_0887", cycle_id=3)
    attempts = (
        _attempt("M_0887", slot=2, on_day=2, outcome=Outcome.STILL_PENDING),
        _attempt("M_0887", slot=3, on_day=7, outcome=Outcome.STILL_PENDING),
        _attempt("M_0887", slot=4, on_day=15, outcome=Outcome.STILL_PENDING),
    )
    episode = Episode(
        mandate=mandate, attempts=attempts, censor_reason=CensorReason.BUDGET_EXHAUSTED
    )

    df = build([episode])
    assert len(df) == 4
    assert set(df["slot"].unique()) == {1, 2, 3, 4}

    # Slot 4: terminal, censored, BUDGET_EXHAUSTED
    row4 = df[df["slot"] == 4].iloc[0]
    assert row4["outcome"] == Outcome.STILL_PENDING
    assert row4["event_code"] == 0
    assert bool(row4["is_terminal"]) is True
    assert bool(row4["censored"]) is True
    assert bool(row4["estimable"]) is True  # slot >= 2: censored, but still a real hazard row
    assert row4["censor_reason"] == CensorReason.BUDGET_EXHAUSTED

    # Slot 1 is never estimable, regardless of how the episode ends
    row1 = df[df["slot"] == 1].iloc[0]
    assert bool(row1["estimable"]) is False


def test_zero_attempt_episode_edge_case():
    """Episode with no attempts (schedule was WINDOW_CLOSED before slot 2).
    Expected exactly 1 row at slot 1, terminal, censored, WINDOW_CLOSED."""
    mandate = _mandate("M_test_zero", cycle_id=1)
    episode = Episode(mandate=mandate, attempts=(), censor_reason=CensorReason.WINDOW_CLOSED)

    df = build([episode])
    assert len(df) == 1

    row = df.iloc[0]
    assert row["slot"] == 1
    assert row["outcome"] == Outcome.STILL_PENDING
    assert bool(row["at_risk"]) is True
    assert bool(row["is_terminal"]) is True
    assert bool(row["censored"]) is True
    assert bool(row["estimable"]) is False
    assert row["censor_reason"] == CensorReason.WINDOW_CLOSED


# === Validate() Rejection Cases ================================================


def test_validate_rejects_row_with_at_risk_false():
    """Any row with at_risk == False must raise FrameError."""
    df = pd.DataFrame({
        "mandate_id": ["M_A"],
        "cycle_id": [1],
        "slot": [1],
        "row_id": [row_id("M_A", 1, 1)],
        "outcome": [Outcome.STILL_PENDING],
        "event_code": [0],
        "at_risk": [False],  # Violation
        "censored": [False],
        "censor_reason": [CensorReason.NONE],
        "is_terminal": [False],
        "estimable": [False],
        "amount_paise": [50_000],
        "ceiling_paise": [100_000],
        "category": ["subscription"],
        "on_day": [1],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_non_contiguous_slots():
    """(mandate, cycle) group must have contiguous 1..K slots."""
    df = pd.DataFrame({
        "mandate_id": ["M_B", "M_B", "M_B"],
        "cycle_id": [1, 1, 1],
        "slot": [1, 2, 4],  # Gap: slot 3 missing
        "row_id": [row_id("M_B", 1, 1), row_id("M_B", 1, 2), row_id("M_B", 1, 4)],
        "outcome": [Outcome.STILL_PENDING, Outcome.STILL_PENDING, Outcome.RECOVERED],
        "event_code": [0, 0, 1],
        "at_risk": [True, True, True],
        "censored": [False, False, False],
        "censor_reason": [CensorReason.NONE, CensorReason.NONE, CensorReason.NONE],
        "is_terminal": [False, False, True],
        "estimable": [False, True, True],
        "amount_paise": [50_000, 50_000, 50_000],
        "ceiling_paise": [100_000, 100_000, 100_000],
        "category": ["subscription", "subscription", "subscription"],
        "on_day": [1, 3, 9],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_terminal_row_followed_by_another():
    """If is_terminal == True, that row must be the last in its group."""
    df = pd.DataFrame({
        "mandate_id": ["M_C", "M_C"],
        "cycle_id": [1, 1],
        "slot": [1, 2],
        "row_id": [row_id("M_C", 1, 1), row_id("M_C", 1, 2)],
        "outcome": [Outcome.RECOVERED, Outcome.STILL_PENDING],
        "event_code": [1, 0],
        "at_risk": [True, True],
        "censored": [False, False],
        "censor_reason": [CensorReason.NONE, CensorReason.NONE],
        "is_terminal": [True, False],  # Violation: terminal at slot 1, but slot 2 exists
        "estimable": [False, True],
        "amount_paise": [50_000, 50_000],
        "ceiling_paise": [100_000, 100_000],
        "category": ["subscription", "subscription"],
        "on_day": [1, 3],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_censored_non_final_row():
    """censored == True must only appear on the last row of a group."""
    df = pd.DataFrame({
        "mandate_id": ["M_D", "M_D"],
        "cycle_id": [1, 1],
        "slot": [1, 2],
        "row_id": [row_id("M_D", 1, 1), row_id("M_D", 1, 2)],
        "outcome": [Outcome.STILL_PENDING, Outcome.STILL_PENDING],
        "event_code": [0, 0],
        "at_risk": [True, True],
        "censored": [True, False],  # Violation: censored at slot 1 but slot 2 follows
        "censor_reason": [CensorReason.WINDOW_CLOSED, CensorReason.NONE],
        "is_terminal": [False, True],
        "estimable": [False, True],
        "amount_paise": [50_000, 50_000],
        "ceiling_paise": [100_000, 100_000],
        "category": ["subscription", "subscription"],
        "on_day": [1, 3],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_censored_non_still_pending():
    """censored == True rows must have outcome == STILL_PENDING."""
    df = pd.DataFrame({
        "mandate_id": ["M_E"],
        "cycle_id": [1],
        "slot": [1],
        "row_id": [row_id("M_E", 1, 1)],
        "outcome": [Outcome.RECOVERED],
        "event_code": [1],
        "at_risk": [True],
        "censored": [True],  # Violation: censored but outcome != STILL_PENDING
        "censor_reason": [CensorReason.BUDGET_EXHAUSTED],
        "is_terminal": [True],
        "estimable": [False],
        "amount_paise": [50_000],
        "ceiling_paise": [100_000],
        "category": ["subscription"],
        "on_day": [1],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_missing_required_column():
    """Every column in EMITTED_COLUMNS must be present -- validate() must
    not silently accept a frame that is missing one."""
    df = pd.DataFrame({
        "mandate_id": ["M_G"],
        "cycle_id": [1],
        "slot": [1],
        "row_id": [row_id("M_G", 1, 1)],
        "outcome": [Outcome.STILL_PENDING],
        "event_code": [0],
        "at_risk": [True],
        "censored": [False],
        "censor_reason": [CensorReason.NONE],
        "is_terminal": [True],
        "estimable": [False],
        "amount_paise": [50_000],
        "ceiling_paise": [100_000],
        "category": ["subscription"],
        # "on_day" deliberately omitted
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_group_whose_last_row_is_not_terminal():
    """A group's last row (by slot) must be marked is_terminal -- distinct
    from "a terminal row followed by another row": here NO row in the group
    is terminal at all, which that other check would not catch."""
    df = pd.DataFrame({
        "mandate_id": ["M_H", "M_H"],
        "cycle_id": [1, 1],
        "slot": [1, 2],
        "row_id": [row_id("M_H", 1, 1), row_id("M_H", 1, 2)],
        "outcome": [Outcome.STILL_PENDING, Outcome.STILL_PENDING],
        "event_code": [0, 0],
        "at_risk": [True, True],
        "censored": [False, False],
        "censor_reason": [CensorReason.NONE, CensorReason.NONE],
        "is_terminal": [False, False],  # Violation: no row in the group is terminal
        "estimable": [False, True],
        "amount_paise": [50_000, 50_000],
        "ceiling_paise": [100_000, 100_000],
        "category": ["subscription", "subscription"],
        "on_day": [1, 3],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_duplicate_row_ids():
    """row_id must be globally unique across the frame."""
    df = pd.DataFrame({
        "mandate_id": ["M_F", "M_F"],
        "cycle_id": [1, 1],
        "slot": [1, 1],  # Same slot -> same row_id
        "row_id": [row_id("M_F", 1, 1), row_id("M_F", 1, 1)],
        "outcome": [Outcome.STILL_PENDING, Outcome.RECOVERED],
        "event_code": [0, 1],
        "at_risk": [True, True],
        "censored": [False, False],
        "censor_reason": [CensorReason.NONE, CensorReason.NONE],
        "is_terminal": [False, True],
        "estimable": [False, False],  # both rows are slot 1 in this contrived duplicate
        "amount_paise": [50_000, 50_000],
        "ceiling_paise": [100_000, 100_000],
        "category": ["subscription", "subscription"],
        "on_day": [1, 1],
    })
    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_estimable_slot_mismatch():
    """estimable must equal (slot >= 2) -- a slot-2 row marked non-estimable
    (or a slot-1 row marked estimable) must raise, since B5 filters on this
    flag directly and a drift here would silently reopen the slot-1
    contamination it exists to prevent (stats-reviewer, B4)."""
    df = pd.DataFrame({
        "mandate_id": ["M_I", "M_I"],
        "cycle_id": [1, 1],
        "slot": [1, 2],
        "row_id": [row_id("M_I", 1, 1), row_id("M_I", 1, 2)],
        "outcome": [Outcome.STILL_PENDING, Outcome.RECOVERED],
        "event_code": [0, 1],
        "at_risk": [True, True],
        "censored": [False, False],
        "censor_reason": [CensorReason.NONE, CensorReason.NONE],
        "is_terminal": [False, True],
        "estimable": [True, True],  # Violation: slot 1 marked estimable
        "amount_paise": [50_000, 50_000],
        "ceiling_paise": [100_000, 100_000],
        "category": ["subscription", "subscription"],
        "on_day": [0, 3],
    })
    with pytest.raises(FrameError):
        validate(df)


# === Anti-pattern Guard ========================================================


def test_censored_rows_are_not_encoded_as_failures():
    """This test documents the hazard PLAN_DETAIL.md:685-689 warns against:
    naive y = (df.outcome == RECOVERED).astype(int) would turn every censored
    STILL_PENDING row into a hard negative, biasing all downstream estimates.

    We build a frame with both resolved and unresolved episodes and assert
    that treating censored rows naively as failures would be wrong."""
    # One resolved episode
    mandate_a = _mandate("M_resolved", cycle_id=1)
    episode_a = Episode(
        mandate=mandate_a,
        attempts=(_attempt("M_resolved", 2, 2, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )

    # One unresolved episode (censored)
    mandate_b = _mandate("M_unresolved", cycle_id=1)
    episode_b = Episode(
        mandate=mandate_b,
        attempts=(
            _attempt("M_unresolved", 2, 2, Outcome.STILL_PENDING),
            _attempt("M_unresolved", 3, 8, Outcome.STILL_PENDING),
            _attempt("M_unresolved", 4, 14, Outcome.STILL_PENDING),
        ),
        censor_reason=CensorReason.BUDGET_EXHAUSTED,
    )

    df = build([episode_a, episode_b])

    # The resolved episode contributes 2 rows: one STILL_PENDING (slot 1) and one RECOVERED (slot 2)
    # The unresolved episode contributes 4 rows: all STILL_PENDING but the last is censored
    assert len(df) == 6

    # Naive approach (WRONG):
    # y = (df.outcome == Outcome.RECOVERED).astype(int)
    # This gives: [0, 1] (for resolved) + [0, 0, 0, 0] (for unresolved) = 1 total positive out of 6
    # But the BUDGET_EXHAUSTED rows are not negatives; they're censored evidence of survival.

    recovered_rows = (df["outcome"] == Outcome.RECOVERED).sum()
    censored_rows = (df["censored"] == True).sum()

    # The actual recovered count is 1, but censored count is 1 (the last row of unresolved episode)
    # They're not interchangeable, and the test ensures they're kept distinct.
    assert recovered_rows == 1
    assert censored_rows == 1
    # This assertion would fail if the frame was built incorrectly, collapsing these concepts.
    assert df[df["censored"] == True]["outcome"].unique()[0] == Outcome.STILL_PENDING


# === Output Quality =============================================================


def test_build_output_passes_its_own_validate():
    """build() must never return a frame it would itself reject."""
    mandate = _mandate("M_test", cycle_id=5)
    attempts = (
        _attempt("M_test", 2, 3, Outcome.STILL_PENDING),
        _attempt("M_test", 3, 9, Outcome.DEAD),
    )
    episode = Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.NONE)

    df = build([episode])
    # This should not raise
    validate(df)


def test_build_emits_all_required_columns():
    """Every column in EMITTED_COLUMNS must be present in build()'s output."""
    mandate = _mandate("M_test", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_test", 2, 2, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )

    df = build([episode])

    for col in EMITTED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_build_multiple_episodes_same_mandate_different_cycles():
    """A mandate can contribute multiple cycles; each cycle gets its own group."""
    mandate1 = _mandate("M_multi", cycle_id=1)
    episode1 = Episode(
        mandate=mandate1,
        attempts=(_attempt("M_multi", 2, 2, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )

    mandate2 = _mandate("M_multi", cycle_id=2)
    episode2 = Episode(
        mandate=mandate2,
        attempts=(
            _attempt("M_multi", 2, 2, Outcome.STILL_PENDING),
            _attempt("M_multi", 3, 8, Outcome.STILL_PENDING),
        ),
        censor_reason=CensorReason.WINDOW_CLOSED,
    )

    df = build([episode1, episode2])

    # Cycle 1: 2 rows (slot 1 synthesized, slot 2 from attempt)
    # Cycle 2: 3 rows (slot 1 synthesized, slots 2-3 from attempts)
    assert len(df) == 5

    cycle1_rows = df[df["cycle_id"] == 1]
    assert len(cycle1_rows) == 2
    assert set(cycle1_rows["slot"]) == {1, 2}

    cycle2_rows = df[df["cycle_id"] == 2]
    assert len(cycle2_rows) == 3
    assert set(cycle2_rows["slot"]) == {1, 2, 3}


# === Household ID (coupled arm) ================================================


def _mandate_with_household(
    mandate_id: str,
    household_id: str | None,
    cycle_id: int = 7,
    amount_paise: int = 50_000,
    ceiling_paise: int = 100_000,
    category: str = "subscription",
) -> SimMandate:
    """Helper to build a SimMandate with an explicit household_id --
    dataclasses.replace() over _mandate()'s pattern, since SimMandate is a
    frozen dataclass and household_id is the one field _mandate() always
    hard-codes to None (matching every mandate on the nominal/misspecified
    arms; only the coupled arm ever sets a real value -- see
    eval/frozen/simulator.py's SimMandate docstring)."""
    import dataclasses

    return dataclasses.replace(
        _mandate(
            mandate_id=mandate_id,
            cycle_id=cycle_id,
            amount_paise=amount_paise,
            ceiling_paise=ceiling_paise,
            category=category,
        ),
        household_id=household_id,
    )


def test_household_id_column_present_and_null_when_all_mandates_have_none():
    """Every mandate on nominal/misspecified has household_id=None (see
    eval/frozen/simulator.py's SimMandate docstring) -- build() must still
    emit a household_id column, with every value null, and EMITTED_COLUMNS
    must list it. This is the status-quo shape every already-reported
    model number was computed against."""
    assert "household_id" in EMITTED_COLUMNS

    mandate = _mandate("M_nohh", cycle_id=1)
    episode = Episode(
        mandate=mandate,
        attempts=(_attempt("M_nohh", 2, 2, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )
    df = build([episode])

    assert "household_id" in df.columns
    assert df["household_id"].isna().all()


def test_household_id_matches_source_mandate_real_and_none():
    """Each row's household_id must match its source episode's
    mandate.household_id exactly -- a real string for a coupled-style
    mandate, null for a mandate with none. Two mandates in the same
    household both carry that household's id."""
    mandate_a = _mandate_with_household("M_hh_a", "H0", cycle_id=1)
    episode_a = Episode(
        mandate=mandate_a,
        attempts=(_attempt("M_hh_a", 2, 2, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.WINDOW_CLOSED,
    )

    mandate_b = _mandate_with_household("M_hh_b", "H0", cycle_id=1)
    episode_b = Episode(
        mandate=mandate_b,
        attempts=(_attempt("M_hh_b", 2, 5, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )

    mandate_c = _mandate_with_household("M_hh_c", "H1", cycle_id=1)
    episode_c = Episode(
        mandate=mandate_c,
        attempts=(_attempt("M_hh_c", 2, 3, Outcome.DEAD),),
        censor_reason=CensorReason.NONE,
    )

    mandate_none = _mandate_with_household("M_hh_none", None, cycle_id=1)
    episode_none = Episode(
        mandate=mandate_none, attempts=(), censor_reason=CensorReason.WINDOW_CLOSED
    )

    df = build([episode_a, episode_b, episode_c, episode_none])

    assert set(df[df["mandate_id"] == "M_hh_a"]["household_id"]) == {"H0"}
    assert set(df[df["mandate_id"] == "M_hh_b"]["household_id"]) == {"H0"}
    assert set(df[df["mandate_id"] == "M_hh_c"]["household_id"]) == {"H1"}
    assert df[df["mandate_id"] == "M_hh_none"]["household_id"].isna().all()


def test_household_id_constant_within_mandate_cycle_group():
    """household_id must be the same value on every row of one
    (mandate_id, cycle_id) group -- build a multi-attempt episode and
    check all its rows agree."""
    mandate = _mandate_with_household("M_hh_multi", "H7", cycle_id=2)
    attempts = (
        _attempt("M_hh_multi", 2, 2, Outcome.STILL_PENDING),
        _attempt("M_hh_multi", 3, 8, Outcome.STILL_PENDING),
        _attempt("M_hh_multi", 4, 15, Outcome.STILL_PENDING),
    )
    episode = Episode(
        mandate=mandate, attempts=attempts, censor_reason=CensorReason.BUDGET_EXHAUSTED
    )

    df = build([episode])
    assert len(df) == 4
    assert df["household_id"].nunique(dropna=False) == 1
    assert set(df["household_id"]) == {"H7"}


def test_validate_accepts_consistent_household_id_per_group():
    """validate() must accept a frame where every (mandate_id, cycle_id)
    group's household_id is internally consistent, including a mix of
    real households and no household across different mandates."""
    mandate_a = _mandate_with_household("M_ok_a", "H0", cycle_id=1)
    episode_a = Episode(
        mandate=mandate_a,
        attempts=(_attempt("M_ok_a", 2, 3, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )
    mandate_b = _mandate_with_household("M_ok_b", None, cycle_id=1)
    episode_b = Episode(
        mandate=mandate_b,
        attempts=(_attempt("M_ok_b", 2, 3, Outcome.STILL_PENDING),),
        censor_reason=CensorReason.WINDOW_CLOSED,
    )

    df = build([episode_a, episode_b])
    # Non-vacuous precondition: household_id must actually be a real,
    # populated column here, or the validate() call below would trivially
    # pass on today's pre-implementation build() (which has no
    # household_id column at all yet) without exercising the new
    # consistency check in the slightest.
    assert "household_id" in df.columns
    assert df[df["mandate_id"] == "M_ok_a"]["household_id"].iloc[0] == "H0"

    # build() already calls validate() internally (it would have raised
    # above if this were wrong) -- call it again directly so the intent,
    # that validate() itself accepts this shape, is explicit rather than
    # merely inferred from build() not raising.
    validate(df)


def test_validate_rejects_inconsistent_household_id_within_group():
    """validate() must raise FrameError if two rows of the SAME
    (mandate_id, cycle_id) group carry DIFFERENT household_id values --
    hand-corrupt a build() output post-hoc, since build()'s own
    construction (household_id copied once per episode, constant across
    every row it emits) cannot produce this state naturally."""
    mandate = _mandate_with_household("M_corrupt", "H0", cycle_id=1)
    attempts = (
        _attempt("M_corrupt", 2, 3, Outcome.STILL_PENDING),
        _attempt("M_corrupt", 3, 9, Outcome.RECOVERED),
    )
    episode = Episode(mandate=mandate, attempts=attempts, censor_reason=CensorReason.NONE)
    df = build([episode])

    # Directly corrupt: the last row (slot 3) gets a different household_id
    # than the rest of its own (mandate_id, cycle_id) group.
    df.loc[df["slot"] == 3, "household_id"] = "H999"

    with pytest.raises(FrameError):
        validate(df)


def test_validate_rejects_same_mandate_different_household_across_cycles():
    """A mandate cannot belong to two different households across its
    cycles -- build() must reject this directly (it calls validate()
    internally). Constructed from two ordinary episodes that happen to
    share a mandate_id but disagree on household_id and cycle_id: nothing
    about build()'s own construction rules out this input shape (each
    episode independently carries its own mandate.household_id), so this
    is the natural construction, not a hand-corrupted post-hoc frame --
    build()'s own machinery produces the invalid state, validate() is what
    must catch it."""
    mandate_cycle1 = _mandate_with_household("M_two_hh", "H0", cycle_id=1)
    episode_cycle1 = Episode(
        mandate=mandate_cycle1,
        attempts=(_attempt("M_two_hh", 2, 3, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )
    mandate_cycle2 = _mandate_with_household("M_two_hh", "H1", cycle_id=2)
    episode_cycle2 = Episode(
        mandate=mandate_cycle2,
        attempts=(_attempt("M_two_hh", 2, 4, Outcome.RECOVERED),),
        censor_reason=CensorReason.NONE,
    )

    with pytest.raises(FrameError):
        build([episode_cycle1, episode_cycle2])


# === Household Worked Example ==================================================


def test_worked_example_household_two_mandates_share_a_household():
    """Coupled-arm-style worked example, mirroring
    test_worked_example_a_recovered_at_slot_3_with_3_rows's row-by-row
    assertion depth, but the axis under test is household_id: M_h001 and
    M_h002 share household H0, M_h003 has no household. build() makes no
    arm assumption, so mixing a household mandate with a None-household
    mandate in one call is a legitimate input shape even though a single
    real Simulator arm would never itself produce that mix."""
    mandate_h1 = _mandate_with_household("M_h001", "H0", cycle_id=7)
    attempts_h1 = (
        _attempt("M_h001", slot=2, on_day=3, outcome=Outcome.STILL_PENDING),
        _attempt("M_h001", slot=3, on_day=9, outcome=Outcome.RECOVERED),
    )
    episode_h1 = Episode(
        mandate=mandate_h1, attempts=attempts_h1, censor_reason=CensorReason.NONE
    )

    mandate_h2 = _mandate_with_household("M_h002", "H0", cycle_id=7)
    attempts_h2 = (_attempt("M_h002", slot=2, on_day=4, outcome=Outcome.STILL_PENDING),)
    episode_h2 = Episode(
        mandate=mandate_h2, attempts=attempts_h2, censor_reason=CensorReason.WINDOW_CLOSED
    )

    mandate_h3 = _mandate_with_household("M_h003", None, cycle_id=7)
    episode_h3 = Episode(
        mandate=mandate_h3, attempts=(), censor_reason=CensorReason.WINDOW_CLOSED
    )

    df = build([episode_h1, episode_h2, episode_h3])
    # h1: slot 1 (synthesized) + slot 2 + slot 3 = 3 rows
    # h2: slot 1 (synthesized) + slot 2 = 2 rows
    # h3: slot 1 (synthesized) only, zero-attempt episode = 1 row
    assert len(df) == 6

    h1_rows = df[df["mandate_id"] == "M_h001"]
    assert len(h1_rows) == 3
    assert set(h1_rows["household_id"]) == {"H0"}

    h2_rows = df[df["mandate_id"] == "M_h002"]
    assert len(h2_rows) == 2
    assert set(h2_rows["household_id"]) == {"H0"}

    h3_rows = df[df["mandate_id"] == "M_h003"]
    assert len(h3_rows) == 1
    assert h3_rows["household_id"].isna().all()

    # Row-by-row, mirroring worked example A's level of detail: slot 3 of
    # M_h001 is terminal/recovered AND carries the household id.
    row3 = h1_rows[h1_rows["slot"] == 3].iloc[0]
    assert row3["outcome"] == Outcome.RECOVERED
    assert row3["event_code"] == 1
    assert bool(row3["is_terminal"]) is True
    assert bool(row3["censored"]) is False
    assert row3["household_id"] == "H0"
    assert row3["row_id"] == row_id("M_h001", 7, 3)

    # Slot 1 of M_h002 (synthesized) also carries H0, even though it is
    # neither terminal nor the row that gave the household its name.
    row1_h2 = h2_rows[h2_rows["slot"] == 1].iloc[0]
    assert bool(row1_h2["is_terminal"]) is False
    assert row1_h2["household_id"] == "H0"
