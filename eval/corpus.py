"""Exploring behaviour-policy corpus for model training (B4). NOT part of
the frozen evaluation protocol and does not modify anything under
eval/frozen/ -- eval/frozen/protocol.md:40-47 pre-commits only the split
MECHANISM (mandate-level grouping, never row-level); this file is how the
training data that mechanism will split is generated.

Why this exists rather than training on eval/baseline_ladder.py's own
episodes: the ladder commits every attempt on a fixed T+1/T+2/T+3 cadence
from a cycle start of day 0 (sim_config.yaml:26-33), which sits entirely
inside the salary window (simulator.py:230, `1 <= on_day <= 5`) and holds
`days_since_last_attempt` at exactly 1 on every attempt. A model trained on
ladder-generated episodes would see zero variance on two of the three real
hazard signals the simulator's own hazard draw depends on (slot,
in_salary_window, days_since_last -- see _draw_outcome, simulator.py:267)
-- it would fit cleanly, validate() would pass, and the model would have
simply never observed a counterfactual timing. B8's allocator would then be
choosing on_day by extrapolating outside the support of its own training
data. See DECISIONS.md, 2026-08-27, B4, for the full reasoning and the
user's explicit direction to build this file.

`nominal` arm only, by design. `misspecified` exists to test whether a
model fitted to nominal's logit world degrades under a genuinely different
link function (protocol.md:62-67); `coupled` exists to test independence.
Training on either voids its own purpose as a held-out arm.

The reported evaluation batch (the frozen seed's 200 mandates, all three
arms) is NEVER part of this corpus and is not split by src/model/splits.py
-- it is read directly by eval/baseline_ladder.py and, from B8 onward, by
the allocator, exactly as protocol.md's own "B2's baseline-ladder run is
not subject to a split at all" paragraph already establishes for the ladder.

Above-cliff mandates are excluded from this corpus, not routed anywhere
within it (see _above_afa_cliff, assert_legal). This is not a training gap
to backfill later: a compliant above-cliff mandate should never reach the
hazard model's retry-timing decision at all -- clause 8(a)/8(b) requires
re-authorisation instead, a structurally different action
(src.core.types.Action.REAUTH). **B8's allocator must apply this exact
same afa_free_limit_paise() filter before ever consulting the hazard
model**, not just this corpus, or the model will be asked to score mandates
it was deliberately never shown (stats-reviewer, B4, finding 4 --
DECISIONS.md, 2026-08-28). eval/baseline_ladder.py does NOT apply this
filter, and that is faithful to the incumbent it models: the real,
documented Razorpay ladder has no AFA-aware routing either.

Least-confident assumption in this design, stated plainly rather than
buried: excluding above-cliff mandates does not bias the `nominal` arm's
hazards, because _draw_outcome (simulator.py:267) never reads amount_paise
or category. That independence is a property of the frozen simulator
today, not something this file enforces, and it does NOT hold for every
arm -- `coupled` makes recovery depend on household_balance versus
mandate.amount_paise directly (simulator.py:351-355), so training on
`coupled` would make this exclusion a real selection-on-outcome bias, not
just a sample-size cost. Training on `nominal` only (below) is what
currently keeps this safe.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src.core.types import Cause, CensorReason, Outcome
from src.policy.constraints import afa_free_limit_paise
from eval.frozen.simulator import AttemptResult, SimMandate, Simulator, load_config

# The full (cause, slot, day_bucket) grid cell_counts() must always report
# a key for, whether or not that cell was ever actually observed -- see
# cell_counts()'s docstring for why enumerating only observed cells was
# itself the bug that let a complete-separation case go undetected.
_ALL_CAUSES: tuple[str, ...] = tuple(c.value for c in Cause)
_ALL_SLOTS: tuple[int, ...] = (2, 3, 4)
_ALL_DAY_BUCKETS: tuple[int, ...] = (0, 1)

# Seeds this corpus is generated from. Deliberately disjoint from
# sim_config.yaml's frozen seed (20260826) -- asserted below, not just
# documented, so the disjointness the user asked to have confirmed is
# actually machine-checked on every import of this module.
# Widened from 10 to 40 seeds, 2026-08-28 (DECISIONS.md, B5 stats-reviewer
# finding 2): at 10 seeds (1,769 mandates, 3,016 estimable rows) the
# held-out log-loss "full model beats intercept-only null" claim was
# statistically indistinguishable from noise (paired t=-0.84, p=0.40) --
# the corpus is simulated and free, and widening it to 40 seeds (~12,300
# estimable rows) is what turns a real, DGP-matching effect into one the
# held-out split can actually detect (t=-6.32, p<1e-9, verified by
# stats-reviewer). Still disjoint from eval/frozen/sim_config.yaml's frozen
# seed, asserted below.
TRAIN_SEEDS: tuple[int, ...] = tuple(range(90001, 90041))

_FROZEN_SEED = load_config()["seed"]
assert _FROZEN_SEED not in TRAIN_SEEDS, (
    "a TRAIN_SEEDS entry collided with eval/frozen/sim_config.yaml's frozen "
    "seed -- this would silently let a reported-batch mandate id also "
    "appear in training data"
)

# Outer bound (days from cycle start) on how long a mandate's retry window
# stays open in this corpus. A schedule that would require an attempt past
# this day is stopped early with censor_reason=WINDOW_CLOSED rather than
# extended -- this is what gives the corpus its WINDOW_CLOSED examples,
# alongside BUDGET_EXHAUSTED ones (slot 4 reached, still unresolved).
MAX_DAY = 40

# cell_counts() below this count is reported by thin_cells() as a sizing
# risk rather than silently accepted.
MIN_CELL_COUNT = 20

# Fraction of episodes whose whole schedule is drawn "compressed" -- see
# _draw_schedule(). Needed because on_day is strictly increasing and
# in_salary_window (simulator.py:230) is a ONE-TIME, absolute, cycle-start
# range (1-5), never a recurring monthly window: with day2/day3/day4 drawn
# as a growing sum of independent gaps, day4 <= 5 requires day2, gap_2_3,
# AND gap_3_4 to all be tiny simultaneously, which is vanishingly rare
# (measured: 0/365 slot-4 rows landed in-window before this fix). A
# dedicated tight-sequence component is the only way to give slot 3/4 real,
# non-zero support inside the window without fabricating a recurring
# "salary window" the simulator's own hazard logic does not have.
COMPRESSED_FRAC = 0.30
_COMPRESSED_MAX_DAY = 7  # 1..7: three distinct days chosen from this range


@dataclass(frozen=True)
class Episode:
    """One mandate's full simulated retry episode.

    `mandate` carries this corpus's OWN (seed-namespaced) mandate_id --
    never the Simulator-internal one. `attempts` is whatever AttemptResult
    objects the Simulator actually returned (zero to three of them, since
    slot 1 is given and never re-simulated); each AttemptResult's own
    `.mandate_id` field is the Simulator-internal, un-namespaced id and
    MUST be ignored by every caller -- `episode.mandate.mandate_id` is the
    only authoritative identity for this episode, all the way through
    src/model/person_period.py's row construction.

    `schedule` is the FULL (day2, day3, day4) committed schedule
    _draw_schedule() drew for this mandate, regardless of how many slots
    were actually attempted before the episode resolved or was censored --
    legitimate per assert_legal's own guarantee that the schedule is drawn
    once, before any attempt() call, and never adjusted after seeing an
    outcome (clause 6(a): committed ahead, not reactive). Optional
    (default None) so every existing direct `Episode(...)` construction in
    this repo's test suites -- which predate this field and have no reason
    to care about it -- keeps working unchanged.

    Added at B6 (stats-reviewer finding 1, DECISIONS.md 2026-08-28): without
    it, src/model/paths.hazard_tensor()'s schedule=None fallback imputes an
    un-attempted slot's in_salary_window from whether the episode SURVIVED
    to that slot -- which is a deterministic function of the very outcome
    being predicted (measured: ~100% of STILL_PENDING episodes have a real
    slot-3 row, ~36-43% of RECOVERED ones do). That is exactly
    src/model/CLAUDE.md rule 2 ("no feature may encode the future"), and it
    made the reported conformal coverage a number for a predictor that
    cannot exist at commit time. Carrying the real, pre-registered schedule
    here and threading it into hazard_tensor(schedule=...) removes the
    leak: every mandate's slot-3/4 covariates come from the one source
    (the committed schedule) regardless of what happened at earlier slots.
    """
    mandate: SimMandate
    attempts: tuple[AttemptResult, ...]
    censor_reason: CensorReason  # NONE if the episode resolved
    schedule: tuple[int, int, int] | None = None


class LegalityError(ValueError):
    """Raised by assert_legal() when an episode contains an attempt pattern
    the real B7/B8 allocator could never legally have produced. A model
    fitted on an infeasible region has meaningless coefficients there --
    this is checked at generation time so such an episode never quietly
    enters the training corpus."""


def _above_afa_cliff(mandate: SimMandate) -> bool:
    """True if `mandate`'s amount is above the AFA-free limit for its
    category (8(a) base limit, or 8(b)'s elevated limit for the three named
    categories) -- i.e. a real e-mandate would require re-authorisation,
    which the frozen simulator has no path for."""
    return mandate.amount_paise > afa_free_limit_paise(mandate.category)


def assert_legal(episode: "Episode") -> None:
    """Raise LegalityError if `episode` is not a schedule the real
    allocator could legally have committed.

    Checked: mandate ceiling (clause 4(c), ceiling_paise >= amount_paise);
    the AFA cliff (clause 8(a)/8(b) -- an above-cliff episode should never
    have been generated in the first place; this is a defensive re-check);
    strictly increasing on_day across attempts; first attempt's on_day >= 1
    as this corpus's day-granularity proxy for the >=24h commitment lag
    (clause 6(a)) -- the schedule is drawn once, before any attempt() call,
    and never adjusted after seeing an outcome, so "committed ahead" holds
    by construction and this assertion is what makes that machine-checked
    rather than merely true-by-code-review.

    THIS IS A TRAINING-DATA ARTIFACT, NOT CLAUSE 6(a) ENFORCEMENT. Flagged
    explicitly by compliance-auditor's B4 review (DECISIONS.md, 2026-08-27)
    because it is easy to mistake for the real thing once B9 exists: this
    corpus has no intraday clock, so "on_day >= 1" is the finest check it
    CAN express, and it says nothing about hours. The real >=24h lead must
    be enforced by B9's executor at the hour level, via
    src.core.clock.now() against committed_schedule.committed_at -- a day-0
    schedule committed at 23:59 and attempted at 00:01 the next day would
    pass this check while violating the actual clause. Do not port this
    assertion into B9 as if it were sufficient.

    NOT checked, deliberately: the NPCI 4-attempt cap, because
    Simulator.attempt() enforces it structurally (raises before this
    function would ever see a violation) -- re-checking it here would only
    ever be dead code. Quiet hours, because on_day is a day index with no
    hour component (stopping_rules.py is B7) -- no generated episode can
    violate or encode a quiet-hours rule, so none is asserted.
    """
    m = episode.mandate
    if m.ceiling_paise < m.amount_paise:
        raise LegalityError(
            f"{m.mandate_id}: ceiling_paise {m.ceiling_paise} < amount_paise "
            f"{m.amount_paise} -- violates clause 4(c)"
        )
    if _above_afa_cliff(m):
        raise LegalityError(
            f"{m.mandate_id}: amount_paise {m.amount_paise} is above the "
            f"AFA-free limit for category {m.category!r} -- requires "
            "re-authorisation, not a silent retry (clause 8(a)/8(b))"
        )
    prev_day = 0
    for a in episode.attempts:
        if a.on_day <= prev_day:
            raise LegalityError(
                f"{m.mandate_id}: slot {a.slot} on_day={a.on_day} is not "
                f"strictly after the previous attempt's day ({prev_day})"
            )
        prev_day = a.on_day
    if episode.attempts and episode.attempts[0].on_day < 1:
        raise LegalityError(
            f"{m.mandate_id}: first retry committed on day "
            f"{episode.attempts[0].on_day} -- violates the >=24h lead proxy "
            "(clause 6(a))"
        )


def _draw_schedule(rng: np.random.Generator) -> tuple[int, int, int]:
    """Draw (day2, day3, day4), strictly increasing, from this corpus's own
    Generator -- never `sim._rng`, so the Simulator's own outcome stream
    stays exactly reproducible regardless of how these day draws are
    implemented.

    Two-component mixture, per stats-reviewer's B4 finding 3 (DECISIONS.md,
    2026-08-28): with probability COMPRESSED_FRAC, the whole three-attempt
    sequence is drawn as three distinct days within the first
    _COMPRESSED_MAX_DAY days -- the only way slot 3 or slot 4 can land
    inside the salary window (1-5) at all, since on_day only increases and
    the window never recurs. Otherwise (the majority case), gaps are drawn
    from a wide independent range, which is what gives days_since_last_attempt
    its broad variety and lets WINDOW_CLOSED/BUDGET_EXHAUSTED both occur
    naturally against `max_day`."""
    if rng.random() < COMPRESSED_FRAC:
        days = rng.choice(
            np.arange(1, _COMPRESSED_MAX_DAY + 1), size=3, replace=False
        )
        days.sort()
        return int(days[0]), int(days[1]), int(days[2])

    day2 = int(rng.integers(1, 21))       # 1..20: inside and outside the
                                           # salary window (1-5)
    gap_2_3 = int(rng.integers(1, 21))    # 1..20
    gap_3_4 = int(rng.integers(1, 21))    # 1..20
    day3 = day2 + gap_2_3
    day4 = day3 + gap_3_4
    return day2, day3, day4


def generate(
    seeds: tuple[int, ...] = TRAIN_SEEDS,
    *,
    rng_seed: int = 1,
    arm: str = "nominal",
    max_day: int = MAX_DAY,
    check_coverage: bool = True,
) -> list[Episode]:
    """Drive the frozen Simulator, once per seed in `seeds`, under an
    exploring behaviour policy: each mandate's full (day2, day3, day4)
    schedule is committed up front by _draw_schedule() before any attempt()
    call, then executed in mandate order (fixed, since a Simulator
    instance's own `_rng` is shared across mandates -- reordering calls
    would change every draw) and in slot order (enforced by attempt()
    itself). Stops a mandate's episode early, uncommitted for the remaining
    slots, if the next scheduled day exceeds `max_day`.

    Mandates above their category's AFA-free limit are excluded outright
    (see _above_afa_cliff) -- not sampled around, not clipped, simply not
    in the corpus, because the frozen simulator has no re-auth path and so
    cannot legally represent what should happen to them. Every included
    episode is additionally checked by assert_legal() before being
    returned.

    Every mandate_id is namespaced `f"s{seed}:{mandate_id}"`: the Simulator
    regenerates M0000..M0199 for every seed, so leaving ids unnamespaced
    would put the identical key in what becomes both the train and calib
    splits later and silently defeat the mandate-level grouping
    src/model/splits.py exists to guarantee.

    Refuses to return (raises ValueError) if any (cause, slot, day_bucket)
    cell is completely empty -- per stats-reviewer's B4 finding 3, an empty
    cell means complete separation for whatever interaction term touches
    it, and that must be caught here, not discovered three blocks later as
    a non-estimable coefficient in B5. A merely-thin (non-zero but small)
    cell is not fatal here; call thin_cells() on the result to inspect
    those.

    `check_coverage=False` opts out of that guard -- for a small,
    deliberately partial `seeds` subset used to test something OTHER than
    corpus-wide coverage (namespacing, assert_legal, day-bucket structure),
    where an empty cell is expected and not a defect. The real
    corpus-building call (this function's own defaults, and anything B5
    actually fits on) must never pass this.
    """
    day_rng = np.random.default_rng(rng_seed)
    episodes: list[Episode] = []
    for seed in seeds:
        sim = Simulator(arm, seed=seed)
        for mandate in sim.mandates:
            if _above_afa_cliff(mandate):
                continue
            namespaced = SimMandate(
                mandate_id=f"s{seed}:{mandate.mandate_id}",
                cycle_id=mandate.cycle_id,
                amount_paise=mandate.amount_paise,
                ceiling_paise=mandate.ceiling_paise,
                category=mandate.category,
                household_id=mandate.household_id,
                initial_cause=mandate.initial_cause,
            )
            day2, day3, day4 = _draw_schedule(day_rng)
            days = {2: day2, 3: day3, 4: day4}

            attempts: list[AttemptResult] = []
            for slot in (2, 3, 4):
                if days[slot] > max_day:
                    break
                result = sim.attempt(mandate.mandate_id, slot, days[slot])
                attempts.append(result)
                if result.outcome != Outcome.STILL_PENDING:
                    break

            if attempts and attempts[-1].outcome != Outcome.STILL_PENDING:
                censor_reason = CensorReason.NONE
            elif attempts and attempts[-1].slot == 4:
                censor_reason = CensorReason.BUDGET_EXHAUSTED
            else:
                # Either zero attempts (day2 already past max_day) or the
                # loop broke on the max_day guard before reaching slot 4.
                censor_reason = CensorReason.WINDOW_CLOSED

            episode = Episode(
                mandate=namespaced, attempts=tuple(attempts), censor_reason=censor_reason,
                schedule=(day2, day3, day4),
            )
            assert_legal(episode)
            episodes.append(episode)

    # Skip the coverage guard for a deliberately empty request (seeds=())
    # -- every cell is vacuously 0 with no episodes to have a coverage gap
    # over at all, which is a different thing from a real-sized run that
    # happens to miss a cell. Also skip when the caller opted out via
    # check_coverage=False (see docstring).
    empty_cells = (
        thin_cells(cell_counts(episodes), threshold=1)
        if episodes and check_coverage else []
    )
    if empty_cells:
        raise ValueError(
            f"generate() produced a corpus with zero-count cell(s): "
            f"{empty_cells} -- widen `seeds`, raise `max_day`, or adjust "
            f"_draw_schedule's COMPRESSED_FRAC"
        )
    return episodes


def cell_counts(episodes: list["Episode"]) -> dict[tuple[str, int, int], int]:
    """(initial_cause, slot, day_bucket) -> attempt count, where day_bucket
    is 0 for in_salary_window (on_day in 1..5) and 1 otherwise. Diagnostic
    only, to confirm the corpus's sizing actually gives every cell support
    before it is handed to B5 -- see thin_cells().

    Initializes ALL 18 cells (3 causes x 3 slots x 2 buckets) to 0 before
    counting, rather than only creating a dict entry for a cell that
    actually occurs. This was a real bug (stats-reviewer, B4 finding 3,
    DECISIONS.md, 2026-08-28): a cell with a true zero count never became a
    key under the old dict.get()-based accumulation, so thin_cells() --
    which only filters counts.items() -- could never report the single
    most dangerous case (complete absence of data) precisely because it
    never appeared as an item to filter. Verified: slot-4/in-window was
    silently 0 across the whole corpus before this fix, invisible to the
    diagnostic built to catch exactly that.

    Reads SimMandate.initial_cause, which the simulator's own docstring
    scopes to "inspection/testing only" (simulator.py:52-62, :200-205).
    That is what this function is: an aggregate count for a diagnostic
    table, in eval/, never a per-row label reaching a design matrix.
    features.FORBIDDEN keeps cause out of the model input regardless of
    what this function does -- do not read this docstring as licence to
    wire initial_cause through as a feature anywhere else.
    """
    counts: dict[tuple[str, int, int], int] = {
        key: 0 for key in itertools.product(_ALL_CAUSES, _ALL_SLOTS, _ALL_DAY_BUCKETS)
    }
    for ep in episodes:
        cause = ep.mandate.initial_cause.value
        for a in ep.attempts:
            bucket = 0 if 1 <= a.on_day <= 5 else 1
            counts[(cause, a.slot, bucket)] += 1
    return counts


def thin_cells(
    counts: dict[tuple[str, int, int], int], threshold: int = MIN_CELL_COUNT
) -> list[tuple[str, int, int]]:
    """Cells from cell_counts() below `threshold` -- sizing risks to report,
    not to silently accept."""
    return sorted(key for key, n in counts.items() if n < threshold)
