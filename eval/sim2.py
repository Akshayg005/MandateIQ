"""A second, non-frozen simulator (R1 Phase B, DECISIONS.md 2026-09-04, R0).

`eval/frozen/simulator.py` never generates `issuer_id`, `instrument_type`, or
real multi-cycle mandate history (`cycle_id` is hard-coded to `1` there,
`simulator.py:186`) -- src/model/features.py's `UNSOURCED` dict documents
exactly this gap. That file cannot be touched (B2's freeze commit), and
re-freezing was explicitly rejected (DECISIONS.md, R0): it would invalidate
every published number in `reports/` to answer a defensibility question a
side study can answer without doing that.

This module is that side study: a DIFFERENT, independent data-generating
process whose hazards actually vary with issuer, instrument type, and
mandate age, built ONLY to let `reports/model_defensibility.md`'s Phase B
report fit-and-measure per-cause coefficients for those three covariates.
It feeds nothing else -- `scripts/guard_invariants.py` mechanically asserts
`eval/run.py` never imports it, so it can never reach the three-bar
headline in `reports/regimes.md`.

Deliberately independent of `eval.frozen.simulator`'s hazard MECHANISM (the
`_softmax`/`_logits_from_base_rates`/`_weighted_choice` helpers below are
reimplemented locally, not imported) -- the whole point of a second
simulator is that its generative process is its own, auditable standalone,
not a variant bolted onto the frozen one. `AttemptResult` is reused as-is
from `eval.frozen.simulator`: a plain data container, not simulator logic,
so importing it carries no independence concern.

SCOPE DECISION, stated up front rather than discovered later:
`mandate_age_days` is a STATIC per-mandate generated covariate (drawn once,
the same way `amount_paise` already is), not real multi-cycle history.
Building genuine multi-cycle simulation -- repeated cycles per mandate,
real aging across cycles -- is a materially bigger DGP than the gate asks
for. The gate wants a hazard that actually *varies* with mandate age,
reported with a coefficient and a confidence interval; a static, generated
age covariate does that honestly. `cycle_id` therefore stays `1` for every
mandate here too, same as the frozen simulator, and this module does not
claim otherwise.

Every hazard number below is, like `eval/frozen/sim_config.yaml`'s own
header says of itself, an ILLUSTRATIVE synthetic parameter chosen to make
the covariate's effect big enough to measure -- not a statistic from real
issuer or instrument failure rates.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.clock import now as clock_now
from src.core.types import Cause, CensorReason, Outcome
from src.model.competing_risks import (
    INSTRUMENT_LEVELS,
    ISSUER_LEVELS,
    SIM2_FEATURE_COLUMNS,
    HazardModel,
    assemble,
    fit,
)
from src.model.features import featurize
from src.model.person_period import build as build_person_period
from eval.frozen.simulator import AttemptResult

N_MANDATES = 200

# Independently authored, illustrative-only -- see module docstring. Chosen
# to match eval/frozen/sim_config.yaml's own shape for narrative
# consistency (same 3 causes, comparable magnitudes), not derived from it.
_CAUSE_MIX: dict[str, float] = {
    Cause.CANT_PAY_NOW.value: 0.50,
    Cause.CANT_PAY_EVER.value: 0.20,
    Cause.WONT_PAY.value: 0.30,
}
_CATEGORY_MIX: dict[str, float] = {
    "subscription": 0.70,
    "insurance_premium": 0.15,
    "mutual_fund": 0.10,
    "credit_card_bill": 0.05,
}
_AMOUNT_RANGE_PAISE = (50_000, 1_400_000)          # Rs 500 -- Rs 14,000
_CEILING_MULTIPLIER_RANGE = (1.0, 1.5)
_MANDATE_AGE_RANGE_DAYS = (0, 720)

# instrument_type mix -- upi_autopay is India's dominant recurring-debit
# rail, so it gets the largest share, same illustrative framing as
# category_mix above.
_INSTRUMENT_MIX: dict[str, float] = {
    "upi_autopay": 0.60,
    "debit_card": 0.25,
    "credit_card": 0.15,
}

_HAZARDS: dict[str, dict[str, float]] = {
    Cause.CANT_PAY_NOW.value: {
        "base_recovery": 0.35, "base_dead": 0.02, "base_optout": 0.03,
        "salary_window_bonus_logit": 0.9,
        "age_recovery_bonus_logit_per_year": 0.6,
    },
    Cause.CANT_PAY_EVER.value: {"base_recovery": 0.03, "base_dead": 0.55, "base_optout": 0.02},
    Cause.WONT_PAY.value: {"base_recovery": 0.08, "base_dead": 0.02, "base_optout": 0.35},
}

# Additive dead-hazard logit bonuses. issuer_gamma and upi_autopay are
# DELIBERATELY elevated -- the resulting aggregate dead-rate gap (marginal
# over cause_mix and the mandate_age_days distribution, both causes/salary-
# window held as the test itself holds them) is ~6.7-7.8pp, verified both
# analytically and by direct simulation (DECISIONS.md, 2026-09-04, "R1b
# review pass"). tests/eval/test_sim2.py's two hazard-difference tests use a
# 150-seed window specifically because a 20-seed window's sampling SD
# (~1.9pp issuer, ~1.4pp instrument) leaves the required >=5pp assertion too
# close to the true gap for real safety margin -- see that file's own
# docstrings and the DECISIONS.md entry for the derivation.
_ISSUER_DEAD_BONUS_LOGIT: dict[str, float] = {
    "issuer_alpha": 0.0, "issuer_beta": 0.15, "issuer_gamma": 1.1, "issuer_delta": 0.05,
}
_INSTRUMENT_DEAD_BONUS_LOGIT: dict[str, float] = {
    "upi_autopay": 1.0, "debit_card": 0.0, "credit_card": -0.1,
}

assert set(_ISSUER_DEAD_BONUS_LOGIT) == set(ISSUER_LEVELS), (
    "_ISSUER_DEAD_BONUS_LOGIT must name exactly ISSUER_LEVELS -- a level "
    "with no bonus entry would KeyError silently deep inside a simulation run"
)
assert set(_INSTRUMENT_DEAD_BONUS_LOGIT) == set(INSTRUMENT_LEVELS), (
    "_INSTRUMENT_DEAD_BONUS_LOGIT must name exactly INSTRUMENT_LEVELS"
)

# Seeds this module's own corpus is generated from -- disjoint from the
# frozen simulator's seed and from eval/corpus.py's TRAIN_SEEDS, asserted
# below rather than merely documented, same discipline eval/corpus.py uses
# for its own seed range.
SIM2_SEEDS: tuple[int, ...] = tuple(range(80001, 80041))

MIN_LEVEL_COUNT = 20  # coverage floor per issuer/instrument level, per cause


@dataclass(frozen=True)
class Sim2Mandate:
    """Static attributes of one sim2 mandate. Carries the same core fields
    `src.model.person_period.build()` reads by duck-typing (`mandate_id`,
    `cycle_id`, `household_id`, `amount_paise`, `ceiling_paise`, `category`)
    plus the three covariates this module exists to vary: `issuer_id`,
    `instrument_type`, `mandate_age_days`. `initial_cause` is ground truth,
    for aggregation/diagnostics only -- never a design-matrix input (same
    convention as `eval.frozen.simulator.SimMandate`)."""

    mandate_id: str
    cycle_id: int
    amount_paise: int
    ceiling_paise: int
    category: str
    household_id: str | None
    initial_cause: Cause
    issuer_id: str
    instrument_type: str
    mandate_age_days: int


@dataclass(frozen=True)
class Sim2Episode:
    """Mirrors eval.corpus.Episode's shape (`.mandate`/`.attempts`/
    `.censor_reason`) -- a separate, local dataclass rather than a reused
    import, so this module's corpus stays legibly self-contained (R0's
    "new, honestly-labelled surface" framing) even though the shape is
    intentionally the same one `person_period.build()` duck-types against.

    Deliberately does NOT carry `eval.corpus.Episode`'s `schedule` field.
    That field exists to stop `src/model/paths.hazard_tensor()`'s
    `schedule=None` fallback from imputing an un-attempted slot's
    `in_salary_window` from whether the episode survived to it -- a
    src/model/DESIGN.md rule-2 leak (see `eval/corpus.py`'s own docstring).
    Nothing in `eval/sim2.py` calls `hazard_tensor()` today, so this is
    latent, not live -- but a `Sim2Episode` must never be passed there
    without adding `schedule` back first (the statistics review, 2026-09-04,
    DECISIONS.md "R1b review pass")."""

    mandate: Sim2Mandate
    attempts: tuple[AttemptResult, ...]
    censor_reason: CensorReason


@dataclass
class _MandateState2:
    last_attempt_day: int = 0
    last_slot_seen: int = 1


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = vals - vals.max()
    exp = np.exp(vals)
    probs = exp / exp.sum()
    return dict(zip(keys, probs.tolist()))


def _logits_from_base_rates(base_recovery: float, base_dead: float, base_optout: float) -> dict[str, float]:
    p_survive = 1.0 - base_recovery - base_dead - base_optout
    return {
        "recover": float(np.log(base_recovery / p_survive)),
        "dead": float(np.log(base_dead / p_survive)),
        "optout": float(np.log(base_optout / p_survive)),
        "survive": 0.0,
    }


def _weighted_choice(rng: np.random.Generator, options: tuple, weights: dict) -> object:
    probs = [weights[o] for o in options]
    return options[rng.choice(len(options), p=probs)]


class Simulator2:
    """Drives one batch of sim2 mandates. Stateful per-mandate slot/day
    tracking, same shape as `eval.frozen.simulator.Simulator` -- no
    household coupling, no arms: sim2 is a single, deliberately simple
    generative story (logit link only), since its purpose is measuring
    three covariate effects cleanly, not modelling misspecification or
    contention."""

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._mandates = self._generate_mandates()
        self._state: dict[str, _MandateState2] = {
            m.mandate_id: _MandateState2() for m in self._mandates
        }

    def _generate_mandates(self) -> tuple[Sim2Mandate, ...]:
        rng = self._rng
        n = N_MANDATES

        cause_names = list(_CAUSE_MIX)
        cause_probs = [_CAUSE_MIX[c] for c in cause_names]
        causes = rng.choice(cause_names, size=n, p=cause_probs)

        cat_names = list(_CATEGORY_MIX)
        cat_probs = [_CATEGORY_MIX[c] for c in cat_names]
        categories = rng.choice(cat_names, size=n, p=cat_probs)

        issuer_probs = [1.0 / len(ISSUER_LEVELS)] * len(ISSUER_LEVELS)
        issuers = rng.choice(list(ISSUER_LEVELS), size=n, p=issuer_probs)

        instrument_names = list(_INSTRUMENT_MIX)
        instrument_probs = [_INSTRUMENT_MIX[i] for i in instrument_names]
        instruments = rng.choice(instrument_names, size=n, p=instrument_probs)

        # `ceil_mult` is a unitless ratio, not money -- the same shape
        # `eval/frozen/simulator.py:172-174` uses. `amounts`/`ceilings`
        # themselves are always rounded to whole paise (`.astype(int)`)
        # before becoming Sim2Mandate's amount_paise and ceiling_paise; no
        # fractional paise value is ever stored or compared.
        lo, hi = _AMOUNT_RANGE_PAISE
        amounts = rng.integers(lo, hi + 1, size=n)
        ceil_lo, ceil_hi = _CEILING_MULTIPLIER_RANGE
        ceil_mult = rng.uniform(ceil_lo, ceil_hi, size=n)
        ceilings = np.round(amounts * ceil_mult).astype(int)

        age_lo, age_hi = _MANDATE_AGE_RANGE_DAYS
        ages = rng.integers(age_lo, age_hi + 1, size=n)

        mandates = []
        for i in range(n):
            mandates.append(
                Sim2Mandate(
                    mandate_id=f"M{i:04d}",
                    cycle_id=1,
                    amount_paise=int(amounts[i]),
                    ceiling_paise=int(ceilings[i]),
                    category=str(categories[i]),
                    household_id=None,
                    initial_cause=Cause(str(causes[i])),
                    issuer_id=str(issuers[i]),
                    instrument_type=str(instruments[i]),
                    mandate_age_days=int(ages[i]),
                )
            )
        return tuple(mandates)

    @property
    def mandates(self) -> tuple[Sim2Mandate, ...]:
        return self._mandates

    def _by_id(self, mandate_id: str) -> Sim2Mandate:
        for m in self._mandates:
            if m.mandate_id == mandate_id:
                return m
        raise KeyError(mandate_id)

    def attempt(self, mandate_id: str, slot: int, on_day: int) -> AttemptResult:
        if slot not in (2, 3, 4):
            raise ValueError(f"slot must be 2, 3, or 4 (slot 1 is given); got {slot}")
        state = self._state[mandate_id]
        expected_next = state.last_slot_seen + 1
        if slot != expected_next:
            raise ValueError(
                f"{mandate_id}: attempted slot {slot} out of order "
                f"(expected slot {expected_next})"
            )
        if on_day <= state.last_attempt_day:
            raise ValueError(
                f"{mandate_id}: slot {slot} on_day={on_day} is not after "
                f"the previous attempt's day ({state.last_attempt_day})"
            )
        mandate = self._by_id(mandate_id)
        in_salary_window = 1 <= on_day <= 5

        outcome = self._draw_outcome(mandate=mandate, in_salary_window=in_salary_window)

        state.last_attempt_day = on_day
        state.last_slot_seen = slot

        return AttemptResult(mandate_id=mandate_id, slot=slot, on_day=on_day, outcome=outcome)

    def _draw_outcome(self, *, mandate: Sim2Mandate, in_salary_window: bool) -> Outcome:
        cause = mandate.initial_cause
        h = _HAZARDS[cause.value]
        logits = _logits_from_base_rates(h["base_recovery"], h["base_dead"], h["base_optout"])

        if cause == Cause.CANT_PAY_NOW and in_salary_window:
            logits["recover"] += h.get("salary_window_bonus_logit", 0.0)
        if cause == Cause.CANT_PAY_NOW:
            age_years = mandate.mandate_age_days / 365.0
            logits["recover"] += h.get("age_recovery_bonus_logit_per_year", 0.0) * age_years

        logits["dead"] += _ISSUER_DEAD_BONUS_LOGIT[mandate.issuer_id]
        logits["dead"] += _INSTRUMENT_DEAD_BONUS_LOGIT[mandate.instrument_type]

        probs = _softmax(logits)
        options = ("recover", "dead", "optout", "survive")
        draw = _weighted_choice(self._rng, options, probs)

        outcome_map = {
            "recover": Outcome.RECOVERED, "dead": Outcome.DEAD,
            "optout": Outcome.OPTED_OUT, "survive": Outcome.STILL_PENDING,
        }
        return outcome_map[draw]


def _draw_schedule(rng: np.random.Generator) -> tuple[int, int, int]:
    """(day2, day3, day4), strictly increasing. A local copy of
    `eval.corpus._draw_schedule`'s compressed/wide mixture -- see that
    function's docstring for why the compressed component exists (slot 3/4
    otherwise almost never land inside the 1-5 salary window). Duplicated
    rather than imported, matching this module's own stated independence
    from every other eval/ generative surface."""
    if rng.random() < 0.30:
        days = rng.choice(np.arange(1, 8), size=3, replace=False)
        days.sort()
        return int(days[0]), int(days[1]), int(days[2])

    day2 = int(rng.integers(1, 21))
    gap_2_3 = int(rng.integers(1, 21))
    gap_3_4 = int(rng.integers(1, 21))
    day3 = day2 + gap_2_3
    day4 = day3 + gap_3_4
    return day2, day3, day4


def _cell_counts(episodes: list[Sim2Episode]) -> dict[tuple[str, str], int]:
    """(dimension, level) -> attempt count, for cause/issuer/instrument.
    Diagnostic used by generate_corpus()'s coverage guard only."""
    counts: dict[tuple[str, str], int] = {
        (dim, level): 0
        for dim, levels in (
            ("cause", tuple(c.value for c in Cause)),
            ("issuer", ISSUER_LEVELS),
            ("instrument", INSTRUMENT_LEVELS),
        )
        for level in levels
    }
    for ep in episodes:
        n = len(ep.attempts)
        counts[("cause", ep.mandate.initial_cause.value)] += n
        counts[("issuer", ep.mandate.issuer_id)] += n
        counts[("instrument", ep.mandate.instrument_type)] += n
    return counts


def generate_corpus(
    seeds: tuple[int, ...] = SIM2_SEEDS,
    *,
    rng_seed: int = 1,
    max_day: int = 40,
    check_coverage: bool = True,
) -> list[Sim2Episode]:
    """Drive `Simulator2`, once per seed, under an exploring behaviour
    policy -- same shape as `eval.corpus.generate()`: commit a full
    (day2, day3, day4) schedule before any attempt() call, execute in
    mandate order, namespace every mandate_id `f"s{seed}:{mandate_id}"` so
    two seeds never collide.

    `check_coverage=True` (default) raises ValueError if any (cause,
    issuer, or instrument) level has fewer than MIN_LEVEL_COUNT attempts --
    the continuous `mandate_age_days` dimension has no discrete cells to
    check this way, so age coverage is not part of this guard."""
    day_rng = np.random.default_rng(rng_seed)
    episodes: list[Sim2Episode] = []
    for seed in seeds:
        sim = Simulator2(seed=seed)
        for mandate in sim.mandates:
            namespaced = Sim2Mandate(
                mandate_id=f"s{seed}:{mandate.mandate_id}",
                cycle_id=mandate.cycle_id,
                amount_paise=mandate.amount_paise,
                ceiling_paise=mandate.ceiling_paise,
                category=mandate.category,
                household_id=mandate.household_id,
                initial_cause=mandate.initial_cause,
                issuer_id=mandate.issuer_id,
                instrument_type=mandate.instrument_type,
                mandate_age_days=mandate.mandate_age_days,
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
                censor_reason = CensorReason.WINDOW_CLOSED

            episodes.append(Sim2Episode(
                mandate=namespaced, attempts=tuple(attempts), censor_reason=censor_reason,
            ))

    if episodes and check_coverage:
        thin = sorted(key for key, n in _cell_counts(episodes).items() if n < MIN_LEVEL_COUNT)
        if thin:
            raise ValueError(
                f"generate_corpus() produced a corpus with under-covered "
                f"cell(s) (< {MIN_LEVEL_COUNT} attempts): {thin} -- widen `seeds`"
            )
    return episodes


def build_sim2_features(pp_df: pd.DataFrame, *, corpus: list[Sim2Episode]) -> pd.DataFrame:
    """One row per `pp_df` row (same order), carrying `row_id`, `issuer_id`,
    `instrument_type`, `mandate_age_days` -- static per mandate, looked up
    from `corpus` by `pp_df`'s own `mandate_id` column (person_period.build()
    already carries mandate_id through; no row_id parsing needed).

    NOT `src.model.features.featurize()`'s job: that module's UNSOURCED
    dict correctly documents these three columns as absent from the REAL
    pipeline, and stays that way -- this function is sim2's own, separate
    source for them, merged in by `eval.sim2`'s own fit pipeline alongside
    the standard `featurize()` output (which still supplies slot/
    in_salary_window/days_since_last_attempt)."""
    lookup: dict[str, Sim2Mandate] = {ep.mandate.mandate_id: ep.mandate for ep in corpus}
    missing = set(pp_df["mandate_id"]) - set(lookup)
    if missing:
        raise ValueError(
            f"build_sim2_features(): {len(missing)} mandate_id(s) in pp_df have no "
            f"matching mandate in `corpus` -- e.g. {sorted(missing)[:5]}"
        )
    return pd.DataFrame({
        "row_id": pp_df["row_id"].to_numpy(),
        "issuer_id": [lookup[mid].issuer_id for mid in pp_df["mandate_id"]],
        "instrument_type": [lookup[mid].instrument_type for mid in pp_df["mandate_id"]],
        "mandate_age_days": [lookup[mid].mandate_age_days for mid in pp_df["mandate_id"]],
    })


def assembled_sim2_frame(seeds: tuple[int, ...] = SIM2_SEEDS) -> pd.DataFrame:
    """The full pipeline: generate_corpus() -> person_period.build() ->
    features.featurize() (unmodified -- gives slot/in_salary_window/
    days_since_last_attempt) merged with build_sim2_features() (gives
    issuer_id/instrument_type/mandate_age_days) -> assemble() (attaches
    event_code/estimable). Output is fit()-ready for `feature_columns=
    SIM2_FEATURE_COLUMNS`."""
    corpus = generate_corpus(seeds)
    pp_df = build_person_period(corpus)
    feat_df = featurize(pp_df)
    sim2_feat_df = build_sim2_features(pp_df, corpus=corpus)
    combined = feat_df.merge(sim2_feat_df, on="row_id", how="inner", validate="one_to_one")
    return assemble(pp_df, combined)


def fit_sim2_model(df: pd.DataFrame) -> HazardModel:
    """Fit the competing-risks model against SIM2_FEATURE_COLUMNS on an
    already-assembled sim2 frame (see assembled_sim2_frame())."""
    return fit(df, feature_columns=SIM2_FEATURE_COLUMNS)


# --- report generation -------------------------------------------------

_NEW_COLUMNS: tuple[str, ...] = tuple(
    c for c in SIM2_FEATURE_COLUMNS if c not in ("const", "slot_3", "slot_4", "in_salary_window")
)
_OUTCOME_EQUATIONS: tuple[str, ...] = ("RECOVERED", "DEAD", "OPTED_OUT")

_OUT_MD = pathlib.Path(__file__).resolve().parent.parent / "reports" / "model_defensibility.md"
_SECTION_BEGIN = "<!-- PHASE_B:BEGIN -->"
_SECTION_END = "<!-- PHASE_B:END -->"

# Each (issuer/instrument/age) column is coded into the DGP as a direct
# effect on exactly ONE outcome equation -- issuer/instrument dummies bump
# the DEAD logit only (_draw_outcome), mandate_age_years bumps RECOVERED
# only, and only for CANT_PAY_NOW. Any OTHER outcome equation showing a
# nonzero fitted coefficient for that column -- significant or not -- is a
# cause-marginal composition effect (fit() has no `cause` covariate at all),
# not something the DGP coded directly. Found by the statistics review, 2026-09-04
# (DECISIONS.md, "R1b review pass"): the first version of this report
# explained ONE such artifact (mandate_age_years on DEAD/OPTED_OUT) with the
# WRONG mechanism (claimed a share-decrease that would predict a NEGATIVE
# coefficient; the fitted and true marginal coefficients are both positive),
# and missed a SECOND artifact entirely (issuer_gamma on RECOVERED).
_DIRECT_TARGET_OUTCOME: dict[str, str] = {
    **{f"issuer_{level}": "DEAD" for level in ISSUER_LEVELS[1:]},
    **{f"instrument_{level}": "DEAD" for level in INSTRUMENT_LEVELS[1:]},
    "mandate_age_years": "RECOVERED",
}

# The DGP's own coded additive dead-hazard logit, relative to each column's
# reference level -- i.e. what a per-cause (not cause-marginal) fit would be
# expected to recover exactly, for the DIRECT (column, outcome) pairs above
# only. Used to show the fitted CI against the value actually generated,
# not just its own significance.
_DGP_CODED_LOGIT: dict[str, float] = {
    **{
        f"issuer_{level}": _ISSUER_DEAD_BONUS_LOGIT[level] - _ISSUER_DEAD_BONUS_LOGIT[ISSUER_LEVELS[0]]
        for level in ISSUER_LEVELS[1:]
    },
    **{
        f"instrument_{level}": (
            _INSTRUMENT_DEAD_BONUS_LOGIT[level] - _INSTRUMENT_DEAD_BONUS_LOGIT[INSTRUMENT_LEVELS[0]]
        )
        for level in INSTRUMENT_LEVELS[1:]
    },
    "mandate_age_years": _HAZARDS[Cause.CANT_PAY_NOW.value]["age_recovery_bonus_logit_per_year"],
}


def _coefficient_table(model: HazardModel) -> pd.DataFrame:
    """One row per (new column, outcome equation) -- same statsmodels
    indexing convention `eval.design_matrix_comparison._coefficient_table`
    documents and this module reuses verbatim: `.params`/`.bse`/`.pvalues`
    integer-position indexed (0/1/2), `.conf_int()` string-indexed
    ("1"/"2"/"3")."""
    result = model.result
    params, bse, pvalues = result.params, result.bse, result.pvalues
    conf_int = result.conf_int()

    rows = []
    for eq_pos, eq_name in enumerate(_OUTCOME_EQUATIONS):
        ci_label = str(eq_pos + 1)
        for col in _NEW_COLUMNS:
            coef = float(params.loc[col, eq_pos])
            se = float(bse.loc[col, eq_pos])
            p = float(pvalues.loc[col, eq_pos])
            ci_lo = float(conf_int.loc[(ci_label, col), "lower"])
            ci_hi = float(conf_int.loc[(ci_label, col), "upper"])
            rows.append({
                "outcome": eq_name, "column": col, "coef": coef, "se": se,
                "z": coef / se if se > 0 else float("nan"), "p": p,
                "ci_low": ci_lo, "ci_high": ci_hi,
                "excludes_zero": not (ci_lo <= 0.0 <= ci_hi),
            })
    return pd.DataFrame(rows)


def _write_report(n_mandates: int, n_estimable: int, coef_df: pd.DataFrame, n_excludes_zero: int) -> None:
    lines: list[str] = [_SECTION_BEGIN, "## Phase B: issuer, instrument type and mandate age on eval/sim2.py", ""]
    lines.append(
        f"_Generated {clock_now().strftime('%Y-%m-%d %H:%M UTC')} by "
        f"`python -m eval.sim2`. Corpus: {len(SIM2_SEEDS)} seeds, {n_mandates} "
        f"mandates, {n_estimable} estimable person-period rows, from a "
        f"SECOND, non-frozen simulator (`eval/sim2.py`) whose data-generating "
        f"process actually varies dead-hazard by `issuer_id`/`instrument_type` "
        f"and CANT_PAY_NOW's recovery hazard by `mandate_age_days` -- unlike "
        f"`eval/frozen/simulator.py`, which never generates any of the three "
        f"(`src/model/features.py`'s `UNSOURCED`). Guarded: "
        f"`scripts/guard_invariants.py` denies `eval/run.py` importing this "
        f"module, so nothing here can reach `reports/regimes.md`'s headline._"
    )
    lines.append("")
    lines.append(
        "**Scope decision, disclosed rather than discovered later**: "
        "`mandate_age_days` is a STATIC per-mandate generated covariate "
        "(drawn once, like `amount_paise` already is everywhere in this "
        "codebase), not real multi-cycle mandate history -- `cycle_id` stays "
        "`1` here too. Building genuine multi-cycle simulation is a "
        "materially bigger DGP than this gate asks for; a static, generated "
        "age covariate that the hazard genuinely depends on answers the "
        "gate's actual question (does a fitted, honest coefficient with a CI "
        "exist for mandate age) without overbuilding."
    )
    lines.append("")
    lines.append(f"`SIM2_FEATURE_COLUMNS` (`src/model/competing_risks.py`): {', '.join(SIM2_FEATURE_COLUMNS)}.")
    lines.append("")
    lines.append("### Fitted coefficients, the six new columns (full-corpus fit, 95% CI)")
    lines.append("")
    lines.append("| Outcome | Column | Coef | SE | z | p | 95% CI | Excludes 0? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in coef_df.iterrows():
        lines.append(
            f"| {row['outcome']} | `{row['column']}` | {row['coef']:+.4f} | "
            f"{row['se']:.4f} | {row['z']:+.2f} | {row['p']:.3f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{'yes' if row['excludes_zero'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        f"{n_excludes_zero}/{len(coef_df)} of these {len(coef_df)} coefficients have a "
        f"95% CI excluding zero -- the opposite of Phase A's result, as expected: "
        f"this DGP was built specifically to make issuer/instrument/age carry "
        f"real signal (see module docstring, `eval/sim2.py`), unlike the frozen "
        f"corpus's amount and category, which the DGP never reads at all."
    )
    lines.append("")
    lines.append(
        "**This is an in-sample, full-corpus descriptive fit, not a held-out "
        "evaluation** -- no train/test split. Appropriate for reading off a "
        "coefficient and its CI; not a generalisation claim, and not "
        "comparable to Phase A's held-out log-loss test."
    )
    lines.append("")

    coef_df = coef_df.copy()
    coef_df["is_direct"] = [
        _DIRECT_TARGET_OUTCOME.get(col) == outcome
        for col, outcome in zip(coef_df["column"], coef_df["outcome"])
    ]
    direct_sig = coef_df[coef_df["is_direct"] & coef_df["excludes_zero"]]
    artifact_sig = coef_df[~coef_df["is_direct"] & coef_df["excludes_zero"]]

    lines.append("### Direct effects vs cause-marginal artifacts")
    lines.append(
        "Each issuer/instrument column is coded into the DGP as a direct "
        "additive dead-hazard bonus (`_draw_outcome`, `eval/sim2.py`) -- i.e. "
        "a direct effect on the **DEAD** equation only. `mandate_age_years` is "
        "coded as a direct bonus to CANT_PAY_NOW's recovery hazard only -- a "
        "direct effect on the **RECOVERED** equation only. `fit()` pools every "
        "row into ONE multinomial logit with no `cause` covariate at all "
        "(production has no true-cause label, ever -- the same reason "
        "`reports/gates.md`'s B7 entry adds a `CauseConditionedHazard` Protocol "
        "instead of fitting per-cause models), so a column can show a nonzero "
        "coefficient in an outcome equation the DGP never coded it into -- a "
        "cause-marginal composition effect, not a directly-coded one."
    )
    lines.append("")
    lines.append(
        f"Of the {n_excludes_zero} significant coefficients: "
        f"**{len(direct_sig)} are direct** DGP effects "
        f"({', '.join(f'{r.column}→{r.outcome}' for r in direct_sig.itertuples())}); "
        f"**{len(artifact_sig)} are cause-marginal artifacts** "
        f"({', '.join(f'{r.column}→{r.outcome}' for r in artifact_sig.itertuples())}). "
        f"Both artifacts were checked, not assumed: pooled cause-marginal "
        f"log-odds computed analytically from the DGP's own cause_mix and age "
        f"distribution move in the SAME direction as the fitted coefficients "
        f"(mandate_age_years on DEAD/OPTED_OUT: analytic pooled slope "
        f"+0.13/year vs fitted +0.18/+0.17; issuer_gamma on RECOVERED: "
        f"analytic pooled log-odds shift +0.12 vs fitted +0.18) -- both are "
        f"real, understood, cause-marginal-fitting artifacts, not fit noise "
        f"or a DGP bug."
    )
    lines.append("")
    lines.append("### The fitted CIs do not cover the DGP's own coded values -- disclosed, not hidden")
    lines.append(
        "For the DIRECT (column, outcome) pairs only, `eval/sim2.py`'s own "
        "coded additive dead-hazard logit is a natural reference point -- and "
        "the fitted, cause-marginal coefficient consistently falls short of "
        "it:"
    )
    lines.append("")
    lines.append("| Column | Coded DGP logit | Fitted coef | 95% CI | Covers coded value? |")
    lines.append("|---|---|---|---|---|")
    for col, target_outcome in _DIRECT_TARGET_OUTCOME.items():
        row = coef_df[(coef_df["column"] == col) & (coef_df["outcome"] == target_outcome)].iloc[0]
        coded = _DGP_CODED_LOGIT[col]
        covers = row["ci_low"] <= coded <= row["ci_high"]
        lines.append(
            f"| `{col}` (→{target_outcome}) | {coded:+.2f} | {row['coef']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {'yes' if covers else '**no**'} |"
        )
    lines.append("")
    lines.append(
        "This is expected, not a fitting error: a cause-marginal coefficient "
        "answers a different question than the per-cause parameter coded "
        "into the DGP (the same distinction `src/model/competing_risks.py`'s "
        "own module docstring already draws for `slot3_x_in_salary_window`), "
        "and pooling across the cause mixture attenuates the coded effect "
        "rather than recovering it exactly. The table above is evidence that "
        "`_design_matrix()`'s issuer/instrument/age machinery detects the "
        "right SIGN and roughly the right ORDER OF MAGNITUDE, not that it "
        "recovers the generating parameter."
    )
    lines.append("")
    lines.append("### What this does and does not license concluding")
    lines.append(
        "Every hazard number `eval/sim2.py` uses is an ILLUSTRATIVE synthetic "
        "parameter, chosen to make each covariate's effect measurable -- not a "
        "statistic from real issuer or instrument failure rates (same framing "
        "`eval/frozen/sim_config.yaml`'s own header states of itself). This "
        "table is evidence that `src/model/competing_risks.py`'s design-matrix "
        "machinery CAN fit and report a defensible, CI-bearing coefficient for "
        "these three covariate types when a corpus actually contains their "
        "effect -- not a claim about what issuer, instrument or age effects "
        "look like in real Razorpay data."
    )
    lines.append("")
    lines.append(
        "**The least comfortable assumption in this corpus, stated rather than "
        "buried**: `initial_cause` is drawn independently of issuer, "
        "instrument and age here, so every artifact and every attenuation "
        "above comes from marginal-fitting alone, with zero confounding. Real "
        "issuer data would not have that independence (a bank whose customers "
        "are poorer plausibly has both more dead instruments AND more "
        "CANT_PAY_NOW mandates) -- under that correlation, attenuation like "
        "the table above does not merely shrink coefficients, it can flip "
        "their sign. This report demonstrates the fitting machinery works "
        "under the EASY case (independent covariates); it does not demonstrate "
        "it is safe under real-world confounding."
    )
    lines.append(_SECTION_END)

    new_section = "\n".join(lines) + "\n"

    existing = _OUT_MD.read_text(encoding="utf-8") if _OUT_MD.exists() else ""
    if _SECTION_BEGIN in existing and _SECTION_END in existing:
        pre = existing[: existing.index(_SECTION_BEGIN)]
        post = existing[existing.index(_SECTION_END) + len(_SECTION_END):]
        _OUT_MD.write_text(pre + new_section + post.lstrip("\n"), encoding="utf-8", newline="\n")
    elif existing:
        _OUT_MD.write_text(existing.rstrip("\n") + "\n\n" + new_section, encoding="utf-8", newline="\n")
    else:
        header = (
            "# Model defensibility\n\n"
            "Answers a reviewer's question the frozen three-bar headline "
            "doesn't: does the hazard model actually use the mandate's own "
            "covariates, and what happens when it does? Two phases -- Phase A "
            "on the frozen corpus (amount, category); Phase B on "
            "`eval/sim2.py` (issuer, instrument type, mandate age). Neither "
            "phase feeds the three-bar headline in `reports/regimes.md`.\n\n"
        )
        _OUT_MD.write_text(header + new_section, encoding="utf-8", newline="\n")


def main() -> None:
    df = assembled_sim2_frame()
    n_mandates = df["mandate_id"].nunique()
    estimable = df[df["estimable"]]
    print(f"sim2 corpus: {len(SIM2_SEEDS)} seeds -> {n_mandates} mandates, "
          f"{len(estimable)} estimable person-period rows")

    model = fit_sim2_model(df)
    coef_df = _coefficient_table(model)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(coef_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    n_excludes_zero = int(coef_df["excludes_zero"].sum())
    print(f"\n{n_excludes_zero}/{len(coef_df)} of the {len(coef_df)} new coefficients have a 95% CI excluding zero")

    _write_report(n_mandates, len(estimable), coef_df, n_excludes_zero)
    print(f"\nwrote {_OUT_MD}")


if __name__ == "__main__":
    main()
