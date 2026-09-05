"""Add the model-input feature columns the build spec section 2's
"Features" table specifies, on top of a frame src/model/person_period.py's
build() has already produced and validated. See that module's docstring
for the exact column split -- this file owns everything in section 2's
"Features" table and nothing from "Identity/index" or "Outcome and
censoring".

Every column featurize() adds is knowable >=24h before the slot it
describes (RBI clause 6(a)) by construction: each is either a static
mandate/category attribute, or derived only from attempts at strictly
earlier slots within the same (mandate_id, cycle_id) group. See
src/model/DESIGN.md rule 2 -- this is the file that rule is about.

Not every column section 2 specifies has a source in the frozen simulator.
UNSOURCED below lists each one with why, and is asserted (by
tests/model/test_features.py) to be exactly SPEC_COLUMNS minus what this
module actually emits -- so a later session cannot silently drop a
sourced feature or silently invent an unsourced one. This follows the
user's explicit choice (2026-08-27, B4): omit and declare, never emit as
all-null and never synthesize a plausible-looking value.
"""
from __future__ import annotations

import pandas as pd

from src.core.types import CensorReason, Outcome, Profile

# The full column-name vocabulary the build spec section 2's "Features"
# table specifies, verbatim.
SPEC_COLUMNS: frozenset[str] = frozenset({
    "amount_paise", "ceiling_paise", "afa_limit_paise",
    "above_afa_cliff",
    "category",
    "prior_failures_this_cycle",
    "last_decline_class", "decline_class_slot1", "decline_class_slot2", "decline_class_slot3",
    "days_since_last_attempt",
    "committed_day_of_month",
    "in_salary_window",
    "mandate_age_days",
    "prior_cycles_ok", "prior_cycles_failed",
    "issuer_id", "instrument_type",
    "notification_lead_hours",
    "profile",
})

# name -> why this SPEC_COLUMNS member has no source in the current
# pipeline and is therefore never emitted by featurize().
UNSOURCED: dict[str, str] = {
    "afa_limit_paise": (
        "constant across the corpus by construction -- eval/corpus.py "
        "excludes every mandate above its category's AFA-free limit, since "
        "the frozen simulator has no re-auth path for one (DECISIONS.md, "
        "2026-08-27, B4). This is not a training gap to backfill: a "
        "compliant above-cliff mandate should never reach this hazard "
        "model at all -- clause 8(a)/8(b) routes it to Action.REAUTH "
        "before any retry-timing decision. B8's allocator must apply this "
        "SAME afa_free_limit_paise() filter before consulting the model, "
        "not just this corpus (DECISIONS.md, 2026-08-28, B4 the statistics review "
        "finding 4)"
    ),
    "above_afa_cliff": (
        "constant-False for the same reason as afa_limit_paise -- the "
        "corpus contains no above-cliff mandate to be True for"
    ),
    "last_decline_class": (
        "the frozen simulator emits Outcome only, never a DeclineClass -- "
        "no bridge from Outcome to DeclineClass exists anywhere in the repo"
    ),
    "decline_class_slot1": "same as last_decline_class",
    "decline_class_slot2": "same as last_decline_class",
    "decline_class_slot3": "same as last_decline_class",
    "mandate_age_days": (
        "SimMandate.cycle_id is hard-coded to 1 for every generated "
        "mandate (simulator.py:186) -- multi-cycle mandate history does "
        "not exist to compute an age from"
    ),
    "prior_cycles_ok": "same as mandate_age_days -- no multi-cycle history exists",
    "prior_cycles_failed": "same as mandate_age_days -- no multi-cycle history exists",
    "issuer_id": "never generated anywhere in eval/frozen/simulator.py",
    "instrument_type": "never generated anywhere in eval/frozen/simulator.py",
    "notification_lead_hours": (
        "a policy output (B7/B8's committed-schedule lead time), not an "
        "input this corpus's generative process produces"
    ),
}

# Outcome-derived and ground-truth columns that must never reach a design
# matrix. Checked by featurize() against its own output before returning.
FORBIDDEN: frozenset[str] = frozenset({
    "outcome", "event_code", "censored", "censor_reason", "is_terminal",
    "initial_cause", "effective_cause", "household_id",
    "iatrogenic_insufficient_funds",
})


def featurize(df: pd.DataFrame, *, profile: Profile = Profile.strict) -> pd.DataFrame:
    """Add every SPEC_COLUMNS member this pipeline can actually source
    (SPEC_COLUMNS - UNSOURCED) to `df` -- the output of
    person_period.build() -- and drop `on_day`, which is build()'s internal
    carry-through column, not part of the modelled feature set.

    `profile` is stamped as a constant column across every row of this
    call, not derived from anything in `df` -- the two RBI compliance
    interpretations are evaluated as separate runs (protocol.md, "both
    profiles produce numbers"), never mixed within one frame. This is a
    small, additive extension of the build spec's stated
    `featurize(df) -> pd.DataFrame` signature (a keyword-only parameter
    with a default), not a divergence from it -- see DECISIONS.md,
    2026-08-27, B4. Note for B5: being constant within any one call makes
    `profile` perfectly collinear with the intercept once dummy-encoded --
    it is meant to travel with the frame for bookkeeping/labelling, not to
    be fit as a covariate against a single-profile batch. Drop it from the
    design matrix, or only include it when a batch genuinely mixes both
    profiles' rows.

    Derivations, all computed only from slot <= k within a (mandate_id,
    cycle_id) group -- never an aggregate over slot >= k (this module's one
    MUST NOT):

    - prior_failures_this_cycle = slot - 1
    - in_salary_window = 1 <= on_day <= 5
    - days_since_last_attempt = on_day - (previous row's on_day within the
      same group), 0 on each group's first row
    - committed_day_of_month = on_day. The frozen simulator has no
      calendar-month concept (on_day is a day-offset from cycle start with
      no month rollover) -- this column is the honest raw proxy for it,
      not a fabricated modulo-30 reconstruction of month-boundary structure
      the generative process does not have.

    Asserts, before returning: no column in FORBIDDEN is present; every
    emitted column is in SPEC_COLUMNS; `on_day` is absent.
    """
    required = {"mandate_id", "cycle_id", "slot", "on_day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"featurize() input is missing required column(s): {sorted(missing)}")

    out = df.sort_values(["mandate_id", "cycle_id", "slot"]).reset_index(drop=True).copy()

    out["prior_failures_this_cycle"] = (out["slot"].astype("int64") - 1).astype("int8")
    out["in_salary_window"] = out["on_day"].between(1, 5)
    # committed_day_of_month is the honest on_day proxy -- see this module's
    # docstring on why a synthetic modulo-30 reconstruction would fabricate
    # month-boundary structure the frozen simulator does not have.
    out["committed_day_of_month"] = out["on_day"].astype("int8")

    prev_on_day = out.groupby(["mandate_id", "cycle_id"], sort=False, observed=True)["on_day"].shift(1)
    out["days_since_last_attempt"] = (out["on_day"] - prev_on_day).fillna(0).astype("int16")

    # Enum MEMBERS as the categorical values, not `.value` strings -- same
    # convention person_period.py uses for outcome/censor_reason, so
    # `df.profile == Profile.strict` works directly.
    out["profile"] = pd.Categorical([profile] * len(out), categories=list(Profile))

    # Physically drop on_day (build()'s internal carry-through) plus every
    # outcome/censoring column build() emits, INCLUDING estimable. Not just
    # "unused as a feature" -- ABSENT, so B5's fit() cannot re-derive a
    # target from this frame even by accident. The target (event_code) is
    # read from person_period.build()'s own output instead, joined back by
    # row_id -- this is what makes the y = (df.outcome == "RECOVERED")
    # anti-pattern structurally harder to commit: the column simply isn't
    # here to leak. `estimable` (marks slot-1's structural-zero rows unfit
    # for hazard fitting -- see person_period.py) is dropped the same way:
    # B5 must filter using build()'s frame's estimable/row_id, exactly how
    # it already must join back event_code -- one mechanism, not two.
    outcome_columns = {
        "outcome", "event_code", "at_risk", "censored", "censor_reason", "is_terminal",
        "estimable",
    }
    # household_id is dropped the same way as on_day -- an identity column
    # person_period.build() emits that this module's feature vocabulary
    # never includes. Physically absent, not merely unused: household_id
    # is latent ground truth (eval/frozen/simulator.py's SimMandate
    # docstring) a policy under test must never read, and it is also
    # listed in FORBIDDEN below as a belt-and-suspenders check -- dropping
    # it here is what keeps that check from ever actually firing on this
    # function's own output.
    out = out.drop(
        columns=["on_day", "household_id"] + sorted(outcome_columns & set(out.columns))
    )

    forbidden_present = FORBIDDEN & set(out.columns)
    if forbidden_present:
        raise ValueError(
            f"featurize() output retained forbidden column(s): {sorted(forbidden_present)}"
        )

    identity_columns = {"mandate_id", "cycle_id", "slot", "row_id"}
    emitted_features = set(out.columns) - identity_columns
    unexpected = emitted_features - SPEC_COLUMNS
    if unexpected:
        raise ValueError(
            f"featurize() emitted undeclared column(s) not in SPEC_COLUMNS: {sorted(unexpected)}"
        )

    return out
