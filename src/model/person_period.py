"""Reshape mandate episodes into one row per (mandate, cycle, slot) --
one row for each slot the mandate was actually at risk for, never a row for
a slot it was not attempted, never a row after a terminal outcome. Schema,
censoring semantics, and both worked examples are normative at
PLAN_DETAIL.md section 2; this module implements that section exactly,
against the corpus eval/corpus.py generates.

Column ownership, so this file and src/model/features.py never duplicate
or silently drop a column between them: this module emits IDENTITY
(mandate_id, cycle_id, slot, row_id) and OUTCOME/CENSORING (outcome,
event_code, at_risk, censored, censor_reason, is_terminal, estimable)
columns, plus the raw substrate features.featurize() needs (amount_paise,
ceiling_paise, category, on_day) -- `on_day` is intentionally NOT part of
PLAN_DETAIL.md section 2's schema and must not survive into featurize()'s
output; it exists only so featurize() can derive in_salary_window,
days_since_last_attempt, and committed_day_of_month without recomputing
anything from the source episodes. Every other column in section 2's
"Features" table (including prior_failures_this_cycle, which is only
`slot - 1` but is listed there, not here) is features.py's to add.

See src/model/CLAUDE.md rule 1 before touching anything below: a censored
episode is not a missing value, and the worked-example B round-trip test
this module must pass exists specifically to catch the
`y = (df.outcome == "RECOVERED")` anti-pattern (PLAN_DETAIL.md:685-689) one
level up, at the frame itself, before any model ever sees it.

`estimable` (added post-freeze-of-this-block, per stats-reviewer's B4
finding -- DECISIONS.md, 2026-08-28): False on every slot-1 row, True
everywhere else. Slot 1 is not a hazard observation -- every episode enters
this system BECAUSE slot 1 already failed, so P(outcome=STILL_PENDING at
slot 1) = 1 by construction, for every mandate, unconditionally. That is a
structural zero, not a parameter to estimate: h_c(1) is identically 0 for
every cause c, and CIF_c(1) = 0 / S(1) = 1 are the correct initial
conditions for the CIF recursion at PLAN_DETAIL.md:700, starting cleanly at
k=2. Fitting slot 1 into the same likelihood as slot 2-4 lets the MLE
"explain" a deterministic outcome using whichever covariates happen to
separate slot-1 rows from the rest -- measured concretely: a model fit with
slot-1 rows included invented a +0.10 logit/day `days_since_last_attempt`
effect on RECOVERED where the true value is 0.0, and a +1.4 logit
`in_salary_window` effect on OPTED_OUT where the true value is 0.0. B5 MUST
filter on `estimable` before fitting anything -- `df[df.estimable]` -- and
must NOT reconstruct this filter as `df.slot >= 2` independently, since
that duplicates the invariant in two places where it can drift apart.

Slot 1's `on_day = 0` is NOT a fabricated value despite being paired with a
non-estimable row: it is the mandate's true cycle-start anchor (the day the
original attempt was made and failed), and features.featurize()'s
days_since_last_attempt computation for slot 2 depends on it being a real,
correct number -- switching it to a null would silently corrupt slot 2's
(estimable=True) gap to a wrong constant instead. `estimable=False` is what
prevents slot 1's row from contaminating a fit; hiding its `on_day` value
would not add safety and would break a real, needed computation for a
different, valid row. Do not "fix" this by nulling on_day.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.core.ids import row_id as _row_id
from src.core.types import CensorReason, Outcome


def _as_outcome(x: object) -> Outcome:
    return x if isinstance(x, Outcome) else Outcome(x)


def _as_censor_reason(x: object) -> CensorReason:
    return x if isinstance(x, CensorReason) else CensorReason(x)

# Identity/index + outcome/censoring columns this module emits, in the
# order PLAN_DETAIL.md section 2 lists them, plus the on_day carry-through
# documented above. features.py's SPEC_COLUMNS/UNSOURCED bookkeeping is
# deliberately a disjoint list from this one -- see that module's docstring.
EMITTED_COLUMNS: tuple[str, ...] = (
    "mandate_id", "cycle_id", "slot", "row_id", "household_id",
    "outcome", "event_code", "at_risk", "censored", "censor_reason", "is_terminal",
    "estimable",
    "amount_paise", "ceiling_paise", "category", "on_day",
)


class FrameError(ValueError):
    """Raised by validate() -- the frame is not a valid person-period shape.
    Every malformed shape PLAN_DETAIL.md:692-695 names raises this, never a
    silent coercion or a dropped row."""


def build(episodes: Sequence[object]) -> pd.DataFrame:
    """One row per (mandate, cycle, slot) actually at risk, across every
    episode in `episodes` (eval.corpus.Episode instances, or anything with
    the same `.mandate` / `.attempts` / `.censor_reason` shape).

    Slot 1 is synthesized, not read from `episode.attempts` -- the frozen
    Simulator never simulates slot 1 (it is the given, already-failed
    original attempt; simulator.py:213), but PLAN_DETAIL.md's worked
    example A shows it as a real STILL_PENDING row, and every downstream
    hazard slot needs slot 1 in the at-risk set to condition on. It is
    terminal/censored only for the zero-attempt episode (an episode whose
    schedule was WINDOW_CLOSED before slot 2 could even be attempted).

    For each subsequent recorded attempt, `is_terminal` is True only on the
    episode's last row; `censored` is True only when that last row's
    outcome is STILL_PENDING (a resolved episode -- RECOVERED, DEAD, or
    OPTED_OUT -- is never censored); `censor_reason` is `episode.censor_reason`
    on that one row and CensorReason.NONE everywhere else. `at_risk` is
    True on every emitted row, by construction -- validate() asserts it
    rather than trusting it, since a future caller could pass a
    pre-filtered frame that violates it.

    Calls validate() on its own output before returning -- build() must
    never hand back a frame it would itself reject.
    """
    rows: list[dict] = []
    for ep in episodes:
        mandate = ep.mandate
        mandate_id = mandate.mandate_id
        cycle_id = mandate.cycle_id
        n = len(ep.attempts)

        # Slot 1 is synthesized -- the frozen simulator never simulates it
        # (it is the given, already-failed original attempt, per
        # eval/frozen/simulator.py:213), but every downstream hazard slot
        # needs it in the at-risk set to condition on. It is
        # terminal/censored only for the zero-attempt episode.
        slot1_terminal = n == 0
        rows.append({
            "mandate_id": mandate_id,
            "cycle_id": cycle_id,
            "slot": 1,
            "row_id": _row_id(mandate_id, cycle_id, 1),
            "household_id": mandate.household_id,
            "outcome": Outcome.STILL_PENDING,
            "event_code": int(Outcome.STILL_PENDING),
            "at_risk": True,
            "censored": slot1_terminal,
            "censor_reason": ep.censor_reason if slot1_terminal else CensorReason.NONE,
            "is_terminal": slot1_terminal,
            "estimable": False,
            "amount_paise": mandate.amount_paise,
            "ceiling_paise": mandate.ceiling_paise,
            "category": mandate.category,
            "on_day": 0,
        })

        for i, a in enumerate(ep.attempts):
            is_last = i == n - 1
            row_censored = is_last and a.outcome == Outcome.STILL_PENDING
            rows.append({
                "mandate_id": mandate_id,
                "cycle_id": cycle_id,
                "slot": a.slot,
                "row_id": _row_id(mandate_id, cycle_id, a.slot),
                "household_id": mandate.household_id,
                "outcome": a.outcome,
                "event_code": int(a.outcome),
                "at_risk": True,
                "censored": row_censored,
                "censor_reason": ep.censor_reason if row_censored else CensorReason.NONE,
                "is_terminal": is_last,
                "estimable": True,
                "amount_paise": mandate.amount_paise,
                "ceiling_paise": mandate.ceiling_paise,
                "category": mandate.category,
                "on_day": a.on_day,
            })

    df = pd.DataFrame(rows, columns=list(EMITTED_COLUMNS))
    df = _apply_dtypes(df)
    validate(df)
    return df


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce build()'s freshly-assembled columns to the dtypes
    PLAN_DETAIL.md section 2 specifies.

    `censor_reason` (a `str, Enum`) keeps its enum MEMBERS as the category
    values, so `df.censor_reason == CensorReason.BUDGET_EXHAUSTED` compares
    the real enum object. `outcome` does NOT, despite the parallel
    `.map(_as_outcome)` call below: `Outcome` is an `IntEnum`, and pandas/
    numpy silently unbox an IntEnum member to a plain `numpy.int64` when
    building the categorical's backing array (verified:
    `pd.Categorical([Outcome.RECOVERED]).categories.dtype` is `int64`, not
    `object`) -- there is no member object left to preserve.
    `df.outcome == Outcome.RECOVERED` still works, but only because
    `Outcome.RECOVERED == 1` holds as an int comparison, not because the
    stored value "is" the enum member -- `df.outcome.iloc[0].name` will
    raise `AttributeError`. Documented here so a future session doesn't
    "fix" `censor_reason` to match `outcome`'s behaviour, or rely on
    `.name` on an extracted `outcome` cell.

    Both categoricals declare their FULL enum as the category set, not just
    whatever values happen to appear in this particular batch -- a batch
    with no OPTED_OUT row must still be assignment-compatible with one that
    has some, and two build() outputs must be pd.concat-able without a
    "new category" error. Cheap to get right here; expensive to debug later
    as an intermittent pandas TypeError in whichever B5/B8 code first
    concatenates two batches that happened to have different observed
    outcome sets."""
    df = df.copy()
    df["mandate_id"] = df["mandate_id"].astype("string")
    df["cycle_id"] = df["cycle_id"].astype("int16")
    df["slot"] = df["slot"].astype("int8")
    df["row_id"] = df["row_id"].astype("string")
    df["household_id"] = df["household_id"].astype("string")
    outcome_dtype = pd.CategoricalDtype(categories=list(Outcome))
    df["outcome"] = df["outcome"].map(_as_outcome).astype(outcome_dtype)
    df["event_code"] = df["event_code"].astype("int8")
    df["at_risk"] = df["at_risk"].astype(bool)
    df["censored"] = df["censored"].astype(bool)
    censor_dtype = pd.CategoricalDtype(categories=list(CensorReason))
    df["censor_reason"] = df["censor_reason"].map(_as_censor_reason).astype(censor_dtype)
    df["is_terminal"] = df["is_terminal"].astype(bool)
    df["estimable"] = df["estimable"].astype(bool)
    df["amount_paise"] = df["amount_paise"].astype("int64")
    df["ceiling_paise"] = df["ceiling_paise"].astype("int64")
    df["category"] = df["category"].astype("category")
    df["on_day"] = df["on_day"].astype("int16")
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise FrameError on any of the shapes PLAN_DETAIL.md:692-695 names:

    - any row with at_risk == False
    - any (mandate_id, cycle_id) group whose slots are not exactly a
      contiguous 1..K (no gap, no repeat, starts at 1)
    - a terminal row (is_terminal == True) followed by another row in the
      same group
    - a censored == True row that is not that group's last row (by slot)
    - a censored row whose outcome != STILL_PENDING

    Plus, not explicitly named in section 2 but required for the frame to
    be usable at all: row_id values are globally unique, every column in
    EMITTED_COLUMNS is present, and `estimable == (slot >= 2)` on every row
    -- slot 1 is never estimable and no other slot is ever non-estimable.
    This is checked directly rather than trusted, since `estimable` is
    exactly the flag B5 must filter on before fitting anything (see this
    module's docstring) and a drift between it and `slot` would silently
    reopen the contamination it exists to prevent.

    Plus, added at B6 for src/model/splits.py's household-grouping (see
    that module): `household_id` must be constant within every
    (mandate_id, cycle_id) group, AND constant across every cycle of a
    given mandate_id -- a mandate cannot belong to two households. This is
    what lets splits.split() group on household_id (falling back to
    mandate_id where null) and still inherit build()'s own "a mandate's
    rows never straddle a split" guarantee as a corollary rather than a
    separate assumption. build() cannot itself construct within-group
    disagreement (household_id is copied once per episode, constant across
    every row it emits), but a caller could still hand validate() a
    corrupted frame directly, and two ordinary episodes that happen to
    share a mandate_id across cycles with different household_id values
    build() *can* construct -- both are checked here, not assumed away.

    Raises FrameError with a message naming the offending mandate_id/
    cycle_id and which check failed -- never returns a bool, never logs and
    continues.
    """
    missing = [c for c in EMITTED_COLUMNS if c not in df.columns]
    if missing:
        raise FrameError(f"missing required column(s): {missing}")

    dupes = df.loc[df["row_id"].duplicated(), "row_id"].tolist()
    if dupes:
        raise FrameError(f"row_id is not unique: {dupes}")

    if not bool(df["at_risk"].astype(bool).all()):
        bad = df.loc[~df["at_risk"].astype(bool), "row_id"].tolist()
        raise FrameError(f"at_risk == False on row(s): {bad}")

    expected_estimable = df["slot"].astype(int) >= 2
    actual_estimable = df["estimable"].astype(bool)
    if not bool((expected_estimable == actual_estimable).all()):
        bad = df.loc[expected_estimable != actual_estimable, "row_id"].tolist()
        raise FrameError(
            f"estimable does not match (slot >= 2) on row(s): {bad}"
        )

    for (mandate_id, cycle_id), group in df.groupby(
        ["mandate_id", "cycle_id"], sort=False, observed=True
    ):
        group = group.sort_values("slot")
        slots = [int(s) for s in group["slot"].tolist()]
        expected = list(range(1, len(slots) + 1))
        if slots != expected:
            raise FrameError(
                f"{mandate_id}:{cycle_id}: slots {slots} are not a "
                f"contiguous 1..K sequence starting at 1"
            )

        terminal_flags = [bool(t) for t in group["is_terminal"].tolist()]
        if any(terminal_flags[:-1]):
            raise FrameError(
                f"{mandate_id}:{cycle_id}: a terminal row is followed by "
                "another row"
            )
        if not terminal_flags[-1]:
            raise FrameError(
                f"{mandate_id}:{cycle_id}: the group's last row (slot "
                f"{slots[-1]}) is not marked is_terminal"
            )

        censored_flags = [bool(c) for c in group["censored"].tolist()]
        if any(censored_flags[:-1]):
            raise FrameError(
                f"{mandate_id}:{cycle_id}: a censored row is not the "
                "group's last row"
            )
        if censored_flags[-1]:
            last_outcome = _as_outcome(group["outcome"].iloc[-1])
            if last_outcome != Outcome.STILL_PENDING:
                raise FrameError(
                    f"{mandate_id}:{cycle_id}: censored row has outcome "
                    f"{last_outcome!r}, expected STILL_PENDING"
                )

        if int(group["household_id"].nunique(dropna=False)) != 1:
            raise FrameError(
                f"{mandate_id}:{cycle_id}: household_id is not constant "
                f"within this group: "
                f"{sorted(group['household_id'].dropna().unique().tolist())}"
            )

    for mandate_id, group in df.groupby("mandate_id", sort=False, observed=True):
        if int(group["household_id"].nunique(dropna=False)) != 1:
            raise FrameError(
                f"{mandate_id}: household_id is not constant across this "
                f"mandate's cycles: "
                f"{sorted(group['household_id'].dropna().unique().tolist())}"
            )
