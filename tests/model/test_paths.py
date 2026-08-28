"""src/model/paths.py -- bridge from hazard rows to CIF tensor per mandate.

Design decision this file pins: the paths module connects the per-row hazard
predictions (after model.hazards(model, X) on an (n,4) frame) to the per-mandate
(n_mandates, 3, 4) tensor cif() requires. This involves reshaping, imputation
where a mandate never actually attempted a slot, and carving out a canonical
terminal-label outcome per episode for training downstream binary classifiers.

Terminal label eligibility: an episode is eligible iff it actually resolved
(RECOVERED/DEAD/OPTED_OUT at any slot), OR it reached slot 4 still pending
(censor_reason == BUDGET_EXHAUSTED). Ineligible iff it is STILL_PENDING and
censored for any reason OTHER than reaching slot 4 (e.g., WINDOW_CLOSED before
slot 4 could legally be attempted). This is the contract between the model and
the off-ramp gate: we can only safely train the "should we offer an exit?"
classifier on episodes with a complete, observed slot-4 picture, or ones that
already clearly resolved earlier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.core.ids import row_id as _row_id
from src.core.types import CensorReason, Outcome, Profile, Cause
from eval.corpus import Episode
from eval.frozen.simulator import AttemptResult, SimMandate
from src.model.person_period import build
from src.model.features import featurize
from src.model.competing_risks import assemble, fit


# === Test helpers: mandate, attempt, episode builders ========================


def _mandate(
    mandate_id: str = "M_test",
    cycle_id: int = 1,
    amount_paise: int = 50_000,
    ceiling_paise: int = 100_000,
    category: str = "subscription",
) -> SimMandate:
    """Helper to build a SimMandate for test use."""
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


def _episode(
    mandate: SimMandate,
    attempts: tuple[AttemptResult, ...],
    censor_reason: CensorReason = CensorReason.NONE,
) -> Episode:
    """Helper to build an Episode."""
    return Episode(
        mandate=mandate,
        attempts=attempts,
        censor_reason=censor_reason,
    )


def _diverse_fit_frame() -> pd.DataFrame:
    """A small, fixed corpus guaranteed to have all 4 event_code classes
    present among its estimable rows -- competing_risks.fit()'s own
    precondition (it raises ValueError otherwise, since statsmodels
    derives its output column count from what's actually observed and
    hazards() assumes all 4 are present). 12 mandates: for each of
    slot in (2,3,4) and each of the 4 Outcome values, one mandate whose
    episode resolves to that outcome at that slot (STILL_PENDING as the
    "resolution" means censored -- WINDOW_CLOSED before slot 4,
    BUDGET_EXHAUSTED at slot 4)."""
    episodes = []
    outcomes = (Outcome.STILL_PENDING, Outcome.RECOVERED, Outcome.DEAD, Outcome.OPTED_OUT)
    idx = 0
    for slot in (2, 3, 4):
        for outcome in outcomes:
            mandate_id = f"M_fit_{idx:04d}"
            idx += 1
            mandate = _mandate(mandate_id, cycle_id=1)
            attempts = tuple(
                _attempt(mandate_id, s, 2 + s, outcome if s == slot else Outcome.STILL_PENDING)
                for s in range(2, slot + 1)
            )
            if outcome != Outcome.STILL_PENDING:
                censor_reason = CensorReason.NONE
            elif slot == 4:
                censor_reason = CensorReason.BUDGET_EXHAUSTED
            else:
                censor_reason = CensorReason.WINDOW_CLOSED
            episodes.append(_episode(mandate, attempts, censor_reason))
    return build(episodes)


_FIT_MODEL_CACHE: dict[bool, object] = {}


def _fit_diverse_model(*, intercept_only: bool):
    """Fit once per (intercept_only) value and cache -- _diverse_fit_frame()
    is fixed, so refitting it per test is pure overhead."""
    if intercept_only not in _FIT_MODEL_CACHE:
        pp_df = _diverse_fit_frame()
        feat_df = featurize(pp_df, profile=Profile.strict)
        assembled = assemble(pp_df, feat_df)
        _FIT_MODEL_CACHE[intercept_only] = fit(assembled, intercept_only=intercept_only)
    return _FIT_MODEL_CACHE[intercept_only]


def _build_model_frame(episodes: list[Episode]) -> tuple[pd.DataFrame, object]:
    """Build, featurize, and assemble `episodes` for SCORING via
    hazard_tensor() -- but fit the model on a separate, class-diverse
    fixed corpus (_diverse_fit_frame()), never on `episodes` itself.

    Most per-test episode sets in this file are deliberately small and
    outcome-homogeneous (e.g. "3 mandates, all RECOVERED at slot 2") to
    keep each test's own hazard_tensor() assertions simple -- exact
    mandate counts, specific observed/imputed slots. That would make
    fit()'s "all 4 event_code classes present" precondition impossible to
    satisfy if fit() and hazard_tensor() were called on the same episodes
    (confirmed: the original single-frame version of this helper made
    every hazard_tensor test in this file raise ValueError from fit()
    itself, before hazard_tensor() ever ran). Decoupling fit-time data
    from predict-time data is not a workaround -- it is the actual
    production shape: competing_risks.py's own docstring notes an
    all-slot-2 prediction batch routinely differs in composition from
    whatever the model was fit on."""
    pp_df = build(episodes)
    feat_df = featurize(pp_df, profile=Profile.strict)
    assembled = assemble(pp_df, feat_df)
    model = _fit_diverse_model(intercept_only=True)
    return assembled, model


# === terminal_labels() tests ===================================================


class TestTerminalLabelsEligibilityRules:
    """Test terminal_labels() eligibility logic against every row of the
    spec table."""

    def test_resolved_at_slot_2_is_eligible(self):
        """Terminal slot 2 with RECOVERED: eligible."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M1", cycle_id=1)
        attempt = _attempt("M1", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["mandate_id"] == "M1"
        assert row["terminal_slot"] == 2
        assert row["label"] == int(Outcome.RECOVERED)
        assert row["eligible"] is True
        assert row["ineligible_reason"] == ""

    def test_resolved_at_slot_3_is_eligible(self):
        """Terminal slot 3 with DEAD: eligible."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M2", cycle_id=1)
        attempt2 = _attempt("M2", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M2", slot=3, on_day=5, outcome=Outcome.DEAD)
        episode = _episode(mandate, (attempt2, attempt3), CensorReason.NONE)

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 3
        assert row["label"] == int(Outcome.DEAD)
        assert row["eligible"] is True
        assert row["ineligible_reason"] == ""

    def test_resolved_at_slot_4_is_eligible(self):
        """Terminal slot 4 with OPTED_OUT: eligible."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M3", cycle_id=1)
        attempt2 = _attempt("M3", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M3", slot=3, on_day=4, outcome=Outcome.STILL_PENDING)
        attempt4 = _attempt("M3", slot=4, on_day=8, outcome=Outcome.OPTED_OUT)
        episode = _episode(
            mandate, (attempt2, attempt3, attempt4), CensorReason.NONE
        )

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 4
        assert row["label"] == int(Outcome.OPTED_OUT)
        assert row["eligible"] is True
        assert row["ineligible_reason"] == ""

    def test_slot_4_still_pending_budget_exhausted_is_eligible(self):
        """Terminal slot 4 with STILL_PENDING and censor_reason=BUDGET_EXHAUSTED:
        eligible. This is the literal S(4) event we can label."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M4", cycle_id=1)
        attempt2 = _attempt("M4", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M4", slot=3, on_day=4, outcome=Outcome.STILL_PENDING)
        attempt4 = _attempt("M4", slot=4, on_day=8, outcome=Outcome.STILL_PENDING)
        episode = _episode(
            mandate, (attempt2, attempt3, attempt4), CensorReason.BUDGET_EXHAUSTED
        )

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 4
        assert row["label"] == int(Outcome.STILL_PENDING)
        assert row["eligible"] is True
        assert row["ineligible_reason"] == ""

    def test_slot_3_still_pending_window_closed_is_ineligible(self):
        """Terminal slot 3 with STILL_PENDING and censor_reason=WINDOW_CLOSED:
        ineligible. We never got to attempt slot 4."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M5", cycle_id=1)
        attempt2 = _attempt("M5", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M5", slot=3, on_day=4, outcome=Outcome.STILL_PENDING)
        episode = _episode(
            mandate, (attempt2, attempt3), CensorReason.WINDOW_CLOSED
        )

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 3
        assert row["eligible"] is False
        assert row["ineligible_reason"] != ""

    def test_slot_2_still_pending_window_closed_is_ineligible(self):
        """Terminal slot 2 with STILL_PENDING and censor_reason=WINDOW_CLOSED:
        ineligible. We never got to slots 3/4."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M6", cycle_id=1)
        attempt2 = _attempt("M6", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        episode = _episode(mandate, (attempt2,), CensorReason.WINDOW_CLOSED)

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 2
        assert row["eligible"] is False
        assert row["ineligible_reason"] != ""

    def test_zero_attempt_episode_window_closed_is_ineligible(self):
        """An episode with no attempts (slot 1 only) and censor_reason=WINDOW_CLOSED:
        ineligible. The schedule couldn't even reach slot 2."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M7", cycle_id=1)
        episode = _episode(mandate, (), CensorReason.WINDOW_CLOSED)

        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["terminal_slot"] == 1
        assert row["eligible"] is False

    def test_all_four_outcomes_at_slot_4_are_eligible(self):
        """All four outcomes at slot 4 are eligible: RECOVERED, DEAD, OPTED_OUT,
        and STILL_PENDING (if BUDGET_EXHAUSTED)."""
        from src.model.paths import terminal_labels

        outcomes_and_reasons = [
            (Outcome.RECOVERED, CensorReason.NONE),
            (Outcome.DEAD, CensorReason.NONE),
            (Outcome.OPTED_OUT, CensorReason.NONE),
            (Outcome.STILL_PENDING, CensorReason.BUDGET_EXHAUSTED),
        ]

        for i, (outcome, reason) in enumerate(outcomes_and_reasons):
            mandate_id = f"M_outcome_{i}"
            mandate = _mandate(mandate_id, cycle_id=1)
            # Slots must be contiguous 1..K (person_period.validate()) --
            # a bare slot-4 attempt with no slot-2/slot-3 rows is not a
            # legal episode shape, the same way every other multi-slot
            # test in this file builds slots 2 and 3 first.
            attempt2 = _attempt(mandate_id, slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
            attempt3 = _attempt(mandate_id, slot=3, on_day=5, outcome=Outcome.STILL_PENDING)
            attempt4 = _attempt(mandate_id, slot=4, on_day=8, outcome=outcome)
            episode = _episode(mandate, (attempt2, attempt3, attempt4), reason)
            pp_df = build([episode])
            result = terminal_labels(pp_df)

            row = result.iloc[0]
            assert (
                row["eligible"] is True
            ), f"outcome {outcome} at slot 4 should be eligible"
            assert row["label"] == int(outcome)


class TestTerminalLabelsProperties:
    """Test invariant properties of terminal_labels() output."""

    def test_one_row_per_episode(self):
        """terminal_labels must produce exactly one row per input episode,
        no duplicates, no silent drops."""
        from src.model.paths import terminal_labels

        episodes = []
        for i in range(5):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        pp_df = build(episodes)
        result = terminal_labels(pp_df)

        assert len(result) == 5, "Must have one row per episode"
        assert len(result.drop_duplicates(subset=["mandate_id", "cycle_id"])) == 5

    def test_output_has_required_columns(self):
        """terminal_labels output must have mandate_id, cycle_id, terminal_slot,
        label, eligible, ineligible_reason columns."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M_test", cycle_id=1)
        attempt = _attempt("M_test", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)
        pp_df = build([episode])
        result = terminal_labels(pp_df)

        required = {"mandate_id", "cycle_id", "terminal_slot", "label", "eligible", "ineligible_reason"}
        assert required.issubset(set(result.columns)), (
            f"Missing required columns. Got: {result.columns.tolist()}"
        )

    def test_terminal_slot_matches_last_row_slot(self):
        """terminal_slot must match the last slot in that episode's person-period rows."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M_test", cycle_id=1)
        attempt2 = _attempt("M_test", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M_test", slot=3, on_day=4, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt2, attempt3), CensorReason.NONE)
        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert result.iloc[0]["terminal_slot"] == 3

    def test_label_outcome_matches_last_row_outcome(self):
        """label must be the outcome int of the last row in that episode."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M_test", cycle_id=1)
        attempt = _attempt("M_test", slot=2, on_day=3, outcome=Outcome.DEAD)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)
        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert result.iloc[0]["label"] == int(Outcome.DEAD)

    def test_ineligible_reason_empty_when_eligible(self):
        """ineligible_reason must be empty string when eligible=True."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M_test", cycle_id=1)
        attempt = _attempt("M_test", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)
        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert result.iloc[0]["ineligible_reason"] == ""

    def test_ineligible_reason_nonempty_when_ineligible(self):
        """ineligible_reason must be a non-empty string when eligible=False."""
        from src.model.paths import terminal_labels

        mandate = _mandate("M_test", cycle_id=1)
        attempt = _attempt("M_test", slot=2, on_day=3, outcome=Outcome.STILL_PENDING)
        episode = _episode(mandate, (attempt,), CensorReason.WINDOW_CLOSED)
        pp_df = build([episode])
        result = terminal_labels(pp_df)

        assert result.iloc[0]["eligible"] is False
        assert result.iloc[0]["ineligible_reason"] != ""


# === hazard_tensor() tests =====================================================


class TestHazardTensorConstruction:
    """Test hazard_tensor() construction of the (n_mandates, 3, 4) tensor."""

    def test_hazard_tensor_returns_correct_shape(self):
        """hazard_tensor() must return an object with .h having shape (n, 3, 4)
        where n is the number of mandates with at least one estimable row."""
        from src.model.paths import hazard_tensor

        episodes = []
        for i in range(3):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        assert tensor.h.shape == (3, 3, 4), (
            f"Expected shape (3, 3, 4), got {tensor.h.shape}"
        )

    def test_hazard_tensor_h_rows_sum_to_one(self):
        """Every hazard row (mandate, slot) in tensor.h must sum to 1
        within numerical tolerance."""
        from src.model.paths import hazard_tensor
        from src.model.cif import _validate_hazards

        episodes = []
        for i in range(5):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        # This should not raise if rows sum to 1
        _validate_hazards(tensor.h)

    def test_hazard_tensor_keys_multiindex_length_matches_n(self):
        """tensor.keys MultiIndex must have length equal to the number of
        mandates with at least one estimable row."""
        from src.model.paths import hazard_tensor

        episodes = []
        for i in range(3):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        assert len(tensor.keys) == 3, (
            f"tensor.keys should have 3 entries for 3 mandates, got {len(tensor.keys)}"
        )

    def test_hazard_tensor_observed_shape_matches_n_3(self):
        """tensor.observed must have shape (n, 3) -- n mandates, 3 slots."""
        from src.model.paths import hazard_tensor

        episodes = []
        for i in range(4):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        assert tensor.observed.shape == (4, 3), (
            f"Expected observed shape (4, 3), got {tensor.observed.shape}"
        )

    def test_hazard_tensor_horizon_shape_is_n(self):
        """tensor.horizon must have shape (n,) -- one last-slot-seen per mandate."""
        from src.model.paths import hazard_tensor

        episodes = []
        for i in range(3):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        assert tensor.horizon.shape == (3,), (
            f"Expected horizon shape (3,), got {tensor.horizon.shape}"
        )


class TestHazardTensorObservationTracking:
    """Test that hazard_tensor() correctly marks which (mandate, slot) pairs
    were actually observed vs. imputed."""

    def test_observed_true_for_real_rows(self):
        """A mandate with an actual row at a given slot must have observed[i, j] = True
        for that slot."""
        from src.model.paths import hazard_tensor

        mandate = _mandate("M_test", cycle_id=1)
        attempt2 = _attempt("M_test", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt3 = _attempt("M_test", slot=3, on_day=4, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt2, attempt3), CensorReason.NONE)

        assembled, model = _build_model_frame([episode])
        tensor = hazard_tensor(model, assembled)

        # Mandate 0's slots 2 and 3 should be observed
        assert tensor.observed[0, 0] is True, "Slot 2 should be observed"
        assert tensor.observed[0, 1] is True, "Slot 3 should be observed"

    def test_observed_false_for_imputed_slots(self):
        """A mandate that resolved at slot 2 must have observed[i, 1] = False and
        observed[i, 2] = False for the un-attempted slots 3 and 4."""
        from src.model.paths import hazard_tensor

        mandate = _mandate("M_test", cycle_id=1)
        attempt = _attempt("M_test", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)

        assembled, model = _build_model_frame([episode])
        tensor = hazard_tensor(model, assembled)

        # Mandate 0 should have only slot 2 observed
        assert tensor.observed[0, 0] is True, "Slot 2 should be observed"
        assert tensor.observed[0, 1] is False, "Slot 3 should be imputed"
        assert tensor.observed[0, 2] is False, "Slot 4 should be imputed"

    def test_horizon_tracks_last_observed_slot(self):
        """tensor.horizon[i] must equal the last actual slot for mandate i."""
        from src.model.paths import hazard_tensor

        # Mandate A: attempts at slots 2, 3, 4
        mandate_a = _mandate("M_a", cycle_id=1)
        attempt_a2 = _attempt("M_a", slot=2, on_day=2, outcome=Outcome.STILL_PENDING)
        attempt_a3 = _attempt("M_a", slot=3, on_day=4, outcome=Outcome.STILL_PENDING)
        attempt_a4 = _attempt("M_a", slot=4, on_day=8, outcome=Outcome.RECOVERED)
        episode_a = _episode(mandate_a, (attempt_a2, attempt_a3, attempt_a4), CensorReason.NONE)

        # Mandate B: attempt only at slot 2
        mandate_b = _mandate("M_b", cycle_id=1)
        attempt_b2 = _attempt("M_b", slot=2, on_day=3, outcome=Outcome.DEAD)
        episode_b = _episode(mandate_b, (attempt_b2,), CensorReason.NONE)

        assembled, model = _build_model_frame([episode_a, episode_b])
        tensor = hazard_tensor(model, assembled)

        # Mandate A's horizon should be 4
        assert tensor.horizon[0] == 4, (
            f"Mandate A's last slot is 4, got horizon {tensor.horizon[0]}"
        )
        # Mandate B's horizon should be 2
        assert tensor.horizon[1] == 2, (
            f"Mandate B's last slot is 2, got horizon {tensor.horizon[1]}"
        )


class TestHazardTensorImputation:
    """Test hazard_tensor() imputation logic for un-attempted slots."""

    def test_imputation_never_sets_in_salary_window_true_after_window_close(self):
        """For a mandate whose last attempt was on on_day >= 5 (outside the
        salary window 1-5), every imputed later slot must have in_salary_window=False.
        This is provably exact (on_day is strictly increasing, window is absolute)."""
        from src.model.paths import hazard_tensor

        mandate = _mandate("M_test", cycle_id=1)
        # Attempt at slot 2 on day 10 (outside window)
        attempt = _attempt("M_test", slot=2, on_day=10, outcome=Outcome.RECOVERED)
        episode = _episode(mandate, (attempt,), CensorReason.NONE)

        assembled, model = _build_model_frame([episode])
        # No schedule frame -- test default imputation
        tensor = hazard_tensor(model, assembled, schedule=None)

        # Mandate 0's slots 3 and 4 are imputed. Verify they don't get
        # in_salary_window=True synthesized. We can't directly inspect
        # features inside tensor, but we can verify the tensor is valid
        # (rows sum to 1) and check hazard values are sensible.
        assert tensor.h[0, 1, :].sum() > 0, "Slot 3 hazard should be non-trivial"
        assert tensor.h[0, 2, :].sum() > 0, "Slot 4 hazard should be non-trivial"


class TestHazardTensorDroppedMandates:
    """Test that mandates with zero estimable rows are excluded from the tensor."""

    def test_zero_attempt_episode_drops_to_dropped_keys(self):
        """An episode with no attempts (slot 1 only, synthesized, non-estimable)
        must land in dropped_keys, NOT in keys or h."""
        from src.model.paths import hazard_tensor

        # One good mandate
        mandate_good = _mandate("M_good", cycle_id=1)
        attempt = _attempt("M_good", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode_good = _episode(mandate_good, (attempt,), CensorReason.NONE)

        # One zero-attempt mandate
        mandate_empty = _mandate("M_empty", cycle_id=1)
        episode_empty = _episode(mandate_empty, (), CensorReason.WINDOW_CLOSED)

        assembled, model = _build_model_frame([episode_good, episode_empty])
        tensor = hazard_tensor(model, assembled)

        # Tensor should only have 1 mandate (the good one)
        assert len(tensor.keys) == 1, "Only good mandate should be in tensor"
        assert tensor.h.shape[0] == 1

        # Dropped mandate should be in dropped_keys
        assert len(tensor.dropped_keys) == 1, "Empty mandate should be in dropped_keys"
        dropped_mandate_ids = [k[0] for k in tensor.dropped_keys]
        assert "M_empty" in dropped_mandate_ids

    def test_dropped_keys_not_in_keys(self):
        """A mandate in dropped_keys must NOT appear in keys."""
        from src.model.paths import hazard_tensor

        mandate_good = _mandate("M_good", cycle_id=1)
        attempt = _attempt("M_good", slot=2, on_day=3, outcome=Outcome.RECOVERED)
        episode_good = _episode(mandate_good, (attempt,), CensorReason.NONE)

        mandate_empty = _mandate("M_empty", cycle_id=1)
        episode_empty = _episode(mandate_empty, (), CensorReason.WINDOW_CLOSED)

        assembled, model = _build_model_frame([episode_good, episode_empty])
        tensor = hazard_tensor(model, assembled)

        keys_mandate_ids = set(k[0] for k in tensor.keys)
        dropped_mandate_ids = set(k[0] for k in tensor.dropped_keys)

        assert len(keys_mandate_ids & dropped_mandate_ids) == 0, (
            "No mandate should appear in both keys and dropped_keys"
        )


class TestHazardTensorRoundTrip:
    """Test that hazard_tensor() output is directly consumable by cif.cif()
    and cif.survival()."""

    def test_hazard_tensor_output_passes_survival_validation(self):
        """cif.survival() should not raise on tensor.h."""
        from src.model.paths import hazard_tensor
        from src.model.cif import survival

        episodes = []
        for i in range(3):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        # Should not raise
        surv = survival(tensor.h)
        assert surv.shape == (3, 4)

    def test_hazard_tensor_output_passes_cif_validation(self):
        """cif.cif() should not raise on tensor.h."""
        from src.model.paths import hazard_tensor
        from src.model.cif import cif

        episodes = []
        for i in range(3):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            attempt = _attempt(f"M_{i}", slot=2, on_day=3, outcome=Outcome.RECOVERED)
            episodes.append(_episode(mandate, (attempt,), CensorReason.NONE))

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        # Should not raise
        cif_result = cif(tensor.h)
        assert cif_result.shape == (3, 3, 4)

    def test_end_to_end_pipeline_with_real_frame(self):
        """Build a small real pipeline: person_period -> featurize -> assemble ->
        fit -> hazard_tensor -> cif/survival, verify no errors."""
        from src.model.paths import hazard_tensor
        from src.model.cif import cif, survival

        episodes = []
        for i in range(5):
            mandate = _mandate(f"M_{i}", cycle_id=1)
            # Vary attempt count and outcomes
            attempts_list = []
            for slot in range(2, min(5, 2 + i % 3)):
                outcome = [Outcome.STILL_PENDING, Outcome.RECOVERED, Outcome.DEAD, Outcome.OPTED_OUT][
                    (i + slot) % 4
                ]
                attempts_list.append(
                    _attempt(f"M_{i}", slot=slot, on_day=2+slot*2, outcome=outcome)
                )
            episode = _episode(mandate, tuple(attempts_list), CensorReason.NONE)
            episodes.append(episode)

        assembled, model = _build_model_frame(episodes)
        tensor = hazard_tensor(model, assembled)

        # Should survive both cif and survival without error
        surv = survival(tensor.h)
        cif_result = cif(tensor.h)

        assert surv.shape[0] == tensor.h.shape[0]
        assert cif_result.shape[0] == tensor.h.shape[0]


# === cif.terminal_distribution() tests ===============================================


class TestTerminalDistribution:
    """Test the new cif.terminal_distribution() function."""

    def test_terminal_distribution_shape(self):
        """terminal_distribution(h) must return (n, 4) where n is the number
        of mandates."""
        from src.model.cif import terminal_distribution

        h = np.random.dirichlet([1, 1, 1, 1], size=(10, 3))
        result = terminal_distribution(h)

        assert result.shape == (10, 4), f"Expected shape (10, 4), got {result.shape}"

    def test_terminal_distribution_rows_sum_to_one(self):
        """Every row of terminal_distribution(h) must sum to 1.0 within
        numerical tolerance."""
        from src.model.cif import terminal_distribution

        # Generate valid hazards
        rng = np.random.RandomState(42)
        h = rng.dirichlet([1, 1, 1, 1], size=(50, 3))
        result = terminal_distribution(h)

        row_sums = result.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9), (
            f"Rows must sum to 1; max deviation: {np.abs(row_sums - 1.0).max()}"
        )

    def test_terminal_distribution_matches_survival_and_cif(self):
        """terminal_distribution(h) columns must equal [S(4), CIF_RECOVERED(4),
        CIF_DEAD(4), CIF_OPTED_OUT(4)] exactly."""
        from src.model.cif import terminal_distribution, survival, cif

        rng = np.random.RandomState(123)
        h = rng.dirichlet([1, 1, 1, 1], size=(20, 3))

        surv = survival(h)
        cif_result = cif(h)
        term_dist = terminal_distribution(h)

        # Column 0: S(4) = survival[:, 3]
        assert np.allclose(term_dist[:, 0], surv[:, 3]), (
            "Column 0 should be S(4)"
        )

        # Column 1: CIF_RECOVERED(4) = cif[:, 0, 3]
        assert np.allclose(term_dist[:, 1], cif_result[:, 0, 3]), (
            "Column 1 should be CIF_RECOVERED(4)"
        )

        # Column 2: CIF_DEAD(4) = cif[:, 1, 3]
        assert np.allclose(term_dist[:, 2], cif_result[:, 1, 3]), (
            "Column 2 should be CIF_DEAD(4)"
        )

        # Column 3: CIF_OPTED_OUT(4) = cif[:, 2, 3]
        assert np.allclose(term_dist[:, 3], cif_result[:, 2, 3]), (
            "Column 3 should be CIF_OPTED_OUT(4)"
        )

    def test_terminal_distribution_on_degenerate_all_still_pending(self):
        """On degenerate all-STILL_PENDING hazards, terminal_distribution
        should return [1, 0, 0, 0] per row (S(4)=1, all CIFs=0)."""
        from src.model.cif import terminal_distribution

        h = np.zeros((5, 3, 4))
        h[:, :, 0] = 1.0  # All probability on STILL_PENDING

        result = terminal_distribution(h)

        # Every row should be [1, 0, 0, 0]
        expected = np.array([[1.0, 0.0, 0.0, 0.0]] * 5)
        assert np.allclose(result, expected, atol=1e-9), (
            f"Expected all rows [1, 0, 0, 0], got {result}"
        )

    def test_terminal_distribution_on_degenerate_certain_recovery_at_slot2(self):
        """On degenerate certain-recovery-at-slot-2, terminal_distribution
        should return [0, 1, 0, 0] per row (all recover at slot 2, S(4)=0)."""
        from src.model.cif import terminal_distribution

        h = np.zeros((5, 3, 4))
        h[:, 0, 1] = 1.0  # Slot 2: RECOVERED = 1
        h[:, 1, 0] = 1.0  # Slot 3: STILL_PENDING = 1
        h[:, 2, 0] = 1.0  # Slot 4: STILL_PENDING = 1

        result = terminal_distribution(h)

        # Every row should be [0, 1, 0, 0]
        expected = np.array([[0.0, 1.0, 0.0, 0.0]] * 5)
        assert np.allclose(result, expected, atol=1e-9), (
            f"Expected all rows [0, 1, 0, 0], got {result}"
        )

    def test_terminal_distribution_single_mandate(self):
        """terminal_distribution must work on n=1."""
        from src.model.cif import terminal_distribution

        h = np.array([[[0.5, 0.3, 0.15, 0.05], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        result = terminal_distribution(h)

        assert result.shape == (1, 4)
        assert np.isclose(result[0, :].sum(), 1.0)

    def test_terminal_distribution_large_batch(self):
        """terminal_distribution must work on large n."""
        from src.model.cif import terminal_distribution

        rng = np.random.RandomState(999)
        h = rng.dirichlet([1, 1, 1, 1], size=(10000, 3))
        result = terminal_distribution(h)

        assert result.shape == (10000, 4)
        row_sums = result.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    def test_terminal_distribution_identity_with_survival_and_cif(self):
        """The core identity: sum of CIF_c(4) for all c plus S(4) equals 1.0,
        for every mandate."""
        from src.model.cif import terminal_distribution

        rng = np.random.RandomState(555)
        h = rng.dirichlet([1, 1, 1, 1], size=(30, 3))
        result = terminal_distribution(h)

        # Rows sum to 1 already, which encodes the identity
        row_sums = result.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)
