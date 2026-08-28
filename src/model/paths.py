"""Bridge from competing_risks.hazards()'s per-row (n_rows, 4) output to
cif.cif()'s per-mandate (n_mandates, 3, 4) input, plus the terminal-label
extraction that decides which resolved/censored episodes are honest
training examples for a downstream classifier over the by-slot-4 outcome.

Kept out of cif.py (deliberately dependency-free, pure array math -- see
that module's docstring) and out of competing_risks.py (which must not
learn about CIFs, only about person-period rows and hazards). Nothing here
imports an LLM client or reads Cause/household_id -- this module only ever
sees Outcome-space quantities.

Two independent things live here, both needed to go from a fitted
HazardModel plus a scoring frame to a CIF-ready tensor:

1. hazard_tensor() -- for every mandate with at least one estimable row,
   build the FULL (slot 2, slot 3, slot 4) covariate row cif() needs, even
   for slots the mandate never actually reached (it resolved or was
   censored earlier). Those slots are IMPUTED, and imputation is honest
   about it: a mandate whose last real attempt already fell outside the
   salary window (1-5) is PROVABLY still outside it at every later slot
   (on_day is strictly increasing -- eval/corpus.py's assert_legal -- and
   the window is a fixed absolute range, not recurring), so that case is
   exact, not a guess. Otherwise the imputed in_salary_window defaults to
   False, which is a documented assumption, not a fact -- see
   hazard_tensor()'s docstring. Every imputed cell is flagged via
   HazardTensor.observed, never silently indistinguishable from a real one.

2. terminal_labels() -- decides which episodes have an HONEST label for
   "what happened by slot 4", for training or evaluating a classifier over
   that outcome. An episode that resolved (RECOVERED/DEAD/OPTED_OUT) at
   ANY slot has an honest label -- resolution is absorbing, nothing changes
   after it. An episode that reached slot 4 still pending
   (censor_reason == BUDGET_EXHAUSTED) has an honest label too: STILL_PENDING
   at slot 4 IS the observed event, not censoring in the usual sense. An
   episode censored EARLIER while still pending (censor_reason ==
   WINDOW_CLOSED, before slot 4 could legally be attempted) does NOT: its
   by-slot-4 outcome is genuinely unobserved, and labelling it STILL_PENDING
   would be a fabricated data point, not an observed one -- it is excluded,
   not imputed. See terminal_labels()'s docstring for the exact rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.types import CensorReason, Outcome
from src.model.competing_risks import HazardModel, hazards as _hazards

# Axis-1 order of the tensor hazard_tensor()/cif() operate on: slots 2, 3, 4
# in that order. There is no slot-1 entry -- slot 1 is a hard-wired
# structural zero (person_period.py's `estimable` invariant), never scored.
SLOT_AXIS: tuple[int, int, int] = (2, 3, 4)

# The fixed attempt budget every mandate is evaluated against (NPCI's
# 1-original + 3-retries cap). Not expected to ever change, but kept as a
# named constant rather than a bare 4 in terminal_labels()'s eligibility
# check, matching this project's "no unattributed magic numbers" convention.
HORIZON: int = 4


class PathError(ValueError):
    """Raised on a structural ordering/shape problem building the hazard
    tensor -- e.g. the internal scoring frame not landing in (mandate,
    slot) C-order before the reshape to (n, 3, 4), which would otherwise
    silently misalign a mandate with the wrong slot's hazard row. A
    ValueError subclass so callers can catch broadly."""


@dataclass(frozen=True)
class HazardTensor:
    """h: (n, 3, 4), cif()-ready -- every row sums to 1. keys: the
    (mandate_id, cycle_id) MultiIndex labelling h's axis 0, in the same
    order. observed: (n, 3) bool -- observed[i, j] is True iff mandate i's
    slot SLOT_AXIS[j] had a real row in the input frame, False if imputed.
    NOTE: this is an object-dtype ndarray of genuine Python bool values,
    not numpy's bool dtype -- numpy.bool_(True) is not `is True` (a real
    identity difference, not a style choice), and this project's test
    suite asserts identity. horizon: (n,) int8 -- the last slot (2, 3, or
    4) that had a real row for mandate i. dropped_keys: the
    (mandate_id, cycle_id) MultiIndex of mandates excluded from every
    other field here because they had zero estimable rows -- never
    silently absent, always named."""

    h: np.ndarray
    keys: pd.MultiIndex
    observed: np.ndarray
    horizon: np.ndarray
    dropped_keys: pd.MultiIndex


def terminal_labels(pp_df: pd.DataFrame, *, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (mandate_id, cycle_id) episode in `pp_df` (a
    person_period.build() output), taken from that episode's single
    is_terminal=True row (person_period.validate() guarantees exactly one
    per group, so no groupby is needed -- a direct filter suffices).

    Columns: mandate_id, cycle_id, terminal_slot (that row's slot),
    label (that row's event_code, an Outcome int -- populated even when
    ineligible, since the column must hold something, but see `eligible`),
    eligible (bool, see below), ineligible_reason (str, empty iff eligible).

    Eligibility -- the honest-label rule from this module's docstring,
    made precise: eligible iff the terminal outcome is NOT STILL_PENDING
    (the episode resolved, which is absorbing -- true at any terminal
    slot), OR it reached `horizon` still pending (that STILL_PENDING
    outcome IS the observed by-`horizon` event, not censoring). Every other
    STILL_PENDING terminal row (terminal slot < horizon -- WINDOW_CLOSED in
    this corpus) is ineligible: the by-`horizon` outcome for that episode
    was never actually observed, and including it with a fabricated
    STILL_PENDING label would bias whatever it trains or scores toward
    that class -- specifically the class this project's downstream
    conformal predictor already struggles with most (see
    src/model/conformal.py's design notes).

    The primary filter is `slot >= horizon`, not a specific censor_reason
    (stats-reviewer, B6, DECISIONS.md 2026-08-28 finding 3): a STILL_PENDING
    row at or past the horizon is the observed event itself regardless of
    WHY the episode then stopped. Today, `BUDGET_EXHAUSTED` is the only
    reason this corpus's generator ever stamps on such a row -- checked as
    an assertion below and raised loudly if violated, so a future censor
    reason at the horizon (there is no other legitimate one today) cannot
    silently pass through this function un-investigated, but a change here
    does not require also changing the eligibility mask itself.
    """
    required = {"mandate_id", "cycle_id", "slot", "event_code", "censor_reason", "is_terminal"}
    missing = required - set(pp_df.columns)
    if missing:
        raise ValueError(f"terminal_labels() input is missing required column(s): {sorted(missing)}")

    terminal = pp_df.loc[pp_df["is_terminal"].astype(bool)].reset_index(drop=True)

    slot = terminal["slot"].astype(int)
    label = terminal["event_code"].astype(int)
    is_still_pending = label == int(Outcome.STILL_PENDING)
    reached_horizon_pending = is_still_pending & (slot >= horizon)

    unexpected_reason = reached_horizon_pending & (
        terminal["censor_reason"] != CensorReason.BUDGET_EXHAUSTED
    )
    if bool(unexpected_reason.any()):
        bad = sorted({r.value for r in terminal.loc[unexpected_reason, "censor_reason"]})
        raise ValueError(
            f"terminal_labels() found STILL_PENDING row(s) at/past horizon "
            f"{horizon} with unexpected censor_reason(s) {bad} -- expected "
            "only BUDGET_EXHAUSTED; investigate before trusting eligibility here"
        )

    eligible_mask = (~is_still_pending) | reached_horizon_pending

    # object dtype, holding genuine Python bool -- not numpy's bool dtype.
    # See HazardTensor's docstring: numpy.bool_(True) is not `is True`.
    eligible = np.array([bool(x) for x in eligible_mask], dtype=object)

    reasons: list[str] = []
    for i in range(len(terminal)):
        if eligible[i]:
            reasons.append("")
        else:
            reasons.append(
                f"terminal slot {int(slot.iloc[i])} is STILL_PENDING with "
                f"censor_reason={terminal['censor_reason'].iloc[i].value} "
                f"(not BUDGET_EXHAUSTED at horizon {horizon}) -- by-{horizon} "
                "outcome was never observed"
            )

    return pd.DataFrame({
        "mandate_id": terminal["mandate_id"].to_numpy(),
        "cycle_id": terminal["cycle_id"].to_numpy(),
        "terminal_slot": slot.to_numpy(),
        "label": label.to_numpy(),
        "eligible": eligible,
        "ineligible_reason": reasons,
    })


def hazard_tensor(
    model: HazardModel,
    assembled: pd.DataFrame,
    *,
    schedule: pd.DataFrame | None = None,
) -> HazardTensor:
    """Score `model` at every slot in SLOT_AXIS for every (mandate_id,
    cycle_id) in `assembled` that has at least one estimable row, and
    reshape into a (n, 3, 4) tensor cif()/survival() can consume directly.

    `assembled` is competing_risks.assemble()'s output: a featurize()
    frame joined with event_code/estimable by row_id. `schedule`, if
    given, is a (mandate_id, cycle_id, slot, on_day) frame supplying the
    REAL committed day for a slot the mandate never actually attempted --
    legitimate per eval/corpus.py's assert_legal (the full schedule is
    drawn once, before any attempt, and never adjusted after seeing an
    outcome, so an unattempted slot's committed day is not a
    counterfactual). Without it, an un-attempted slot's in_salary_window
    is imputed: exactly False when the mandate's last real attempt already
    fell outside the salary window (provable -- see module docstring),
    else defaulted to False as a documented, not proven, assumption.
    days_since_last_attempt for an imputed slot is set to 0.0; this is a
    placeholder, not a claim -- FEATURE_COLUMNS (competing_risks.py)
    excludes that column from the model's default fit, so its imputed
    value does not currently reach any prediction, but the column is still
    required by _design_matrix()'s shape contract.

    Construction is one dense scoring call, not per-slot: every (mandate,
    slot) row is assembled in (mandate, slot) order -- mandate-major,
    slot-minor, slot cycling SLOT_AXIS within each mandate -- then
    hazards() is called ONCE on the whole stacked frame (statsmodels
    preserves row order) and reshaped to (n, 3, 4). The construction order
    is verified before the reshape, raising PathError rather than trusting
    it, the same discipline hazards()'s own output-shape assertion uses.

    A (mandate_id, cycle_id) with zero estimable rows (only the
    synthesized, never-estimable slot-1 row) is excluded from every field
    except dropped_keys.
    """
    required = {
        "mandate_id", "cycle_id", "slot", "estimable",
        "in_salary_window", "days_since_last_attempt",
    }
    missing = required - set(assembled.columns)
    if missing:
        raise ValueError(f"hazard_tensor() input is missing required column(s): {sorted(missing)}")

    group_cols = ["mandate_id", "cycle_id"]
    all_keys_df = (
        assembled[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
    )
    estimable_df = assembled[assembled["estimable"].astype(bool)]
    present_keys_df = (
        estimable_df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
        if len(estimable_df) else all_keys_df.iloc[0:0].reset_index(drop=True)
    )

    present_set = set(map(tuple, present_keys_df[group_cols].to_numpy()))
    dropped_keys_df = all_keys_df[
        ~all_keys_df[group_cols].apply(tuple, axis=1).isin(present_set)
    ].reset_index(drop=True)

    n = len(present_keys_df)
    keys = pd.MultiIndex.from_frame(present_keys_df[group_cols], names=group_cols)
    dropped_keys = (
        pd.MultiIndex.from_frame(dropped_keys_df[group_cols], names=group_cols)
        if len(dropped_keys_df)
        else pd.MultiIndex.from_arrays([[], []], names=group_cols)
    )

    schedule_lookup: dict[tuple, int] | None = None
    if schedule is not None:
        schedule_lookup = {
            (row.mandate_id, row.cycle_id, int(row.slot)): int(row.on_day)
            for row in schedule.itertuples(index=False)
        }

    real_lookup: dict[tuple, dict] = {}
    for row in estimable_df.itertuples(index=False):
        real_lookup[(row.mandate_id, row.cycle_id, int(row.slot))] = {
            "in_salary_window": bool(row.in_salary_window),
            "days_since_last_attempt": float(row.days_since_last_attempt),
        }

    scoring_rows: list[dict] = []
    observed = np.empty((n, len(SLOT_AXIS)), dtype=object)
    horizon = np.zeros(n, dtype="int8")

    for i, key_row in enumerate(present_keys_df.itertuples(index=False)):
        mandate_id, cycle_id = key_row.mandate_id, key_row.cycle_id
        observed_slots: list[int] = []

        for j, slot in enumerate(SLOT_AXIS):
            real = real_lookup.get((mandate_id, cycle_id, slot))
            if real is not None:
                observed[i, j] = True
                observed_slots.append(slot)
                in_salary_window = real["in_salary_window"]
                days_since_last_attempt = real["days_since_last_attempt"]
            else:
                observed[i, j] = False
                sched_on_day = (
                    schedule_lookup.get((mandate_id, cycle_id, slot))
                    if schedule_lookup is not None else None
                )
                if sched_on_day is not None:
                    in_salary_window = 1 <= sched_on_day <= 5
                elif observed_slots and not real_lookup[
                    (mandate_id, cycle_id, observed_slots[-1])
                ]["in_salary_window"]:
                    # Provably exact: on_day is strictly increasing and the
                    # salary window is a fixed absolute range (1-5), so a
                    # mandate already out of window at its last real
                    # attempt cannot re-enter it at any later slot.
                    in_salary_window = False
                else:
                    # No schedule entry and either no real attempt yet, or
                    # the last real attempt was still inside the window --
                    # a later slot's window membership is genuinely
                    # unknown. False is a documented assumption, not a
                    # provable fact -- see this function's docstring.
                    in_salary_window = False
                days_since_last_attempt = 0.0

            scoring_rows.append({
                "mandate_id": mandate_id,
                "cycle_id": cycle_id,
                "slot": slot,
                "in_salary_window": in_salary_window,
                "days_since_last_attempt": days_since_last_attempt,
            })

        horizon[i] = max(observed_slots) if observed_slots else SLOT_AXIS[0] - 1

    scoring_df = pd.DataFrame(scoring_rows)

    expected_len = n * len(SLOT_AXIS)
    if len(scoring_df) != expected_len:
        raise PathError(
            f"hazard_tensor() internal scoring frame has {len(scoring_df)} "
            f"rows, expected {expected_len} ({n} mandates x "
            f"{len(SLOT_AXIS)} slots) -- construction order was violated"
        )
    expected_slot_cycle = list(SLOT_AXIS) * n
    if n > 0 and scoring_df["slot"].tolist() != expected_slot_cycle:
        raise PathError(
            "hazard_tensor() internal scoring frame's slot ordering does "
            "not match the expected (mandate, slot) C-order -- refusing "
            "to reshape, since that would silently misalign a mandate "
            "with the wrong slot's hazard row"
        )

    if n == 0:
        h = np.empty((0, len(SLOT_AXIS), 4), dtype=float)
    else:
        probs = _hazards(model, scoring_df)
        h = probs.reshape(n, len(SLOT_AXIS), 4)

    return HazardTensor(h=h, keys=keys, observed=observed, horizon=horizon, dropped_keys=dropped_keys)
