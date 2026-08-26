"""Clairvoyant upper-bound policy. NOT a candidate for src/policy/ to imitate
and NOT part of the frozen protocol -- it exists to answer one question:
how much headroom exists between the incumbent ladder and the best ANY
policy could conceivably achieve, given perfect knowledge of the generative
process. If a real, imperfectly-informed policy (B5-B8) beats the ladder by
anywhere close to this gap, that would be remarkable; if the gap itself is
tiny on some arm, no real policy can be expected to show a meaningful win
there, and that arm's contribution to "beats the ladder on all three arms"
is not falsifiable. Better to know that before B3 than at B13.

Deliberately given information no real system could ever have: the true
(or current effective) cause, and -- for the coupled arm -- the household's
live balance. This is standard "oracle baseline" methodology (see e.g.
bandit/RL literature): a ceiling constructed from privileged information,
never a deployable policy.

Algorithm, per mandate, at each decision point (slot 2, then 3 if still
pending, then 4):
  1. Read the mandate's CURRENT true effective_cause (ground truth a real
     policy can never observe -- see Simulator/SimMandate's own "must never
     read" warnings, which this file is the one deliberate exception to).
  2. Solve, via exact backward induction over a dense day grid and the
     remaining attempt budget, the single next day that maximizes EXPECTED
     recovered value -- using simulator.py's own pure probability helpers
     (imported directly, not re-derived), so this can never silently drift
     from what the real simulator actually computes.
  3. Attempt on that day against the REAL simulator -- real randomness.
     This oracle does not peek at unrealized dice rolls, only at the true
     generative PARAMETERS. It is the ceiling a perfectly-informed MODEL
     could reach, not the ceiling of hindsight.

The per-mandate solve treats the observed cause as fixed for its own
lookahead; an actual mid-episode switch (misspecified arm only) is not
modelled within one solve; but because the solve is re-run (from a fresh
cache) whenever the live cause has changed since the last check, an
actual switch is fully picked up at the next decision point.

Coupled arm: mandates within one household are attempted in descending
order of their own STANDALONE optimal-schedule EXPECTED VALUE (probability
of recovering times amount -- not probability alone), ignoring coupling in
that per-mandate calculation since order is exactly what is being decided.
Household balances here average ~9% of household demand (see
protocol.md's Known limitations), so in the typical case only the FIRST
successfully-attempted member can be funded at all; under that scarcity,
maximizing expected value of whichever member goes first dominates the
objective, which is why value (not probability) is the sort key -- a
low-probability, high-amount member can be worth more than a
high-probability, low-amount one. This is a defensible greedy heuristic for
severe scarcity, not a proven joint optimum for every balance regime; true
joint scheduling under shared liquidity is a harder combinatorial problem
this sanity check does not need to solve exactly to answer the headroom
question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import Action, Cause, Outcome, Profile
from eval.frozen.scoring import BatchResult, aggregate, score_mandate
from eval.frozen.simulator import AttemptResult, Simulator, SimMandate, _logits_from_base_rates, _softmax

# Deliberately dense and NOT restricted to B8's eventual ~4 structural
# candidates -- this file answers "what is the true ceiling," not "what B8
# will actually search." Using B8's own restricted grid here would make the
# ceiling artificially low and understate real headroom.
CANDIDATE_DAYS = tuple(range(1, 41))


def _hazard_probs(sim: Simulator, cause: Cause, day: int, retries_so_far: int, last_day: int) -> dict[str, float]:
    """Mirrors Simulator._draw_outcome's probability computation exactly,
    without drawing randomness -- reuses simulator.py's own helpers so this
    can never silently diverge from what the real simulator does."""
    h = sim.config["hazards"][cause.value]
    logits = _logits_from_base_rates(h["base_recovery"], h["base_dead"], h["base_optout"])
    in_salary_window = 1 <= day <= 5
    if cause == Cause.CANT_PAY_NOW and in_salary_window:
        logits["recover"] += h.get("salary_window_bonus_logit", 0.0)
    if cause == Cause.WONT_PAY:
        logits["optout"] += h.get("optout_escalation_logit_per_attempt", 0.0) * retries_so_far

    link = sim.config["arms"][sim.arm]["link"]
    if link == "logit":
        return _softmax(logits)
    elif link == "cloglog":
        return sim._cloglog_probs(logits, cause, day - last_day)
    else:
        raise ValueError(f"unknown link {link!r}")


def _solve(
    sim: Simulator, cause: Cause, amount_paise: int, attempts_left: int,
    last_day: int, memo: dict,
) -> tuple[float, int | None]:
    """Exact backward induction over CANDIDATE_DAYS, memoized on
    (attempts_left, last_day) for a fixed (sim, cause, amount_paise). Returns
    (expected recovered paise under the optimal remaining schedule, best
    next day, or None if attempts_left == 0)."""
    if attempts_left == 0:
        return 0.0, None
    key = (attempts_left, last_day)
    if key in memo:
        return memo[key]

    retries_so_far = 3 - attempts_left
    best_value, best_day = -1.0, None
    for day in CANDIDATE_DAYS:
        if day <= last_day:
            continue
        probs = _hazard_probs(sim, cause, day, retries_so_far, last_day)
        future_value, _ = _solve(sim, cause, amount_paise, attempts_left - 1, day, memo)
        value = probs["recover"] * amount_paise + probs["survive"] * future_value
        if value > best_value:
            best_value, best_day = value, day

    memo[key] = (best_value, best_day)
    return best_value, best_day


def _execution_order(sim: Simulator) -> list[SimMandate]:
    """Natural generation order for nominal/misspecified (no cross-mandate
    interaction, so order is irrelevant there). For coupled, households are
    processed in a fixed order (sorted by id, for determinism) and members
    within a household are attempted in descending order of their own
    standalone optimal recovery probability."""
    if sim.arm != "coupled":
        return list(sim.mandates)

    priorities: dict[str, float] = {}
    for m in sim.mandates:
        cause = sim.effective_cause(m.mandate_id)
        value, _ = _solve(sim, cause, m.amount_paise, 3, 0, {})
        priorities[m.mandate_id] = value  # raw expected paise, not probability -- see module docstring

    households: dict[str, list[SimMandate]] = {}
    for m in sim.mandates:
        households.setdefault(m.household_id, []).append(m)

    order: list[SimMandate] = []
    for household_id in sorted(households):
        order.extend(sorted(households[household_id], key=lambda m: -priorities[m.mandate_id]))
    return order


def run(sim: Simulator, profile: Profile) -> BatchResult:
    order = _execution_order(sim)
    results = []
    for mandate in order:
        attempts = []
        last_day = 0
        attempts_left = 3
        memo: dict = {}
        current_cause = sim.effective_cause(mandate.mandate_id)
        while attempts_left > 0:
            cause = sim.effective_cause(mandate.mandate_id)
            if cause != current_cause:
                memo = {}
                current_cause = cause
            _, day = _solve(sim, cause, mandate.amount_paise, attempts_left, last_day, memo)
            slot = 2 + (3 - attempts_left)
            attempt = sim.attempt(mandate.mandate_id, slot, day)
            attempts.append(attempt)
            last_day = day
            attempts_left -= 1
            if attempt.outcome != Outcome.STILL_PENDING:
                break
        results.append(score_mandate(mandate, attempts))
    return aggregate(results, arm=sim.arm, profile=profile.value)


@dataclass(frozen=True)
class CauseAwareMandateResult:
    mandate_id: str
    action: Action
    attempts: tuple[AttemptResult, ...]
    iatrogenic_failures: int


@dataclass(frozen=True)
class CauseAwareBatchResult:
    """`run()` above answers "how much is there to gain from perfect
    timing," holding the ATTEMPT-every-mandate policy fixed -- that is why
    it can go through the frozen `score_mandate`/`aggregate`, which require
    at least one attempt per mandate (protocol.md, Known limitations, last
    bullet). This dataclass exists because `run_cause_aware` below answers a
    different question -- "how much attempt-budget and iatrogenic
    contention does perfect cause-targeting avoid" -- and deliberately
    produces mandates with ZERO attempts (STOP, REAUTH-bound; OFFER, an
    exit offered), which the frozen scorer rejects by design. Reporting
    attempts_spent and iatrogenic_failures only (not recovered_paise or
    preserved) sidesteps needing an offer-acceptance model this diagnostic
    has no business inventing."""

    arm: str
    profile: str
    n_mandates: int
    n_stopped_reauth: int
    n_offered_exit: int
    n_attempted: int
    total_attempts_spent: int
    total_iatrogenic_failures: int
    per_mandate: tuple[CauseAwareMandateResult, ...] = field(repr=False)

    def summary(self) -> str:
        return (
            f"[{self.arm}/{self.profile}] n={self.n_mandates} "
            f"attempted={self.n_attempted} stopped_reauth={self.n_stopped_reauth} "
            f"offered_exit={self.n_offered_exit} "
            f"attempts_spent={self.total_attempts_spent} "
            f"iatrogenic_failures={self.total_iatrogenic_failures}"
        )


def run_cause_aware(sim: Simulator, profile: Profile) -> CauseAwareBatchResult:
    """Same privileged timing logic as `run()`, plus acting on the true
    cause: `CANT_PAY_EVER` is never attempted (real action: stop, request
    re-authorisation) and `WONT_PAY` is never attempted (real action: offer
    an exit). Only `CANT_PAY_NOW` consumes retry attempts and -- under
    `coupled` -- household balance. The effective cause is re-checked before
    every decision point, so a mid-episode switch (misspecified arm) is
    caught at the next slot, exactly like `run()`'s timing re-solve.

    This is the lever `run()` cannot show: `run()` attempts every mandate
    regardless of cause, so it can only ever demonstrate timing headroom.
    Under `coupled`, skipping `CANT_PAY_EVER`/`WONT_PAY` members means they
    never draw the shared household balance at all (simulator.py's
    `attempt()` is the only place balance is debited), which is exactly the
    contention protocol.md's `coupled` section says a batch-blind policy
    fails to avoid.
    """
    order = _execution_order(sim)
    results = []
    for mandate in order:
        attempts: list[AttemptResult] = []
        last_day = 0
        attempts_left = 3
        memo: dict = {}
        current_cause = sim.effective_cause(mandate.mandate_id)
        action = Action.ATTEMPT
        while attempts_left > 0:
            cause = sim.effective_cause(mandate.mandate_id)
            if cause == Cause.CANT_PAY_EVER:
                action = Action.REAUTH
                break
            if cause == Cause.WONT_PAY:
                action = Action.OFFER
                break
            if cause != current_cause:
                memo = {}
                current_cause = cause
            _, day = _solve(sim, cause, mandate.amount_paise, attempts_left, last_day, memo)
            slot = 2 + (3 - attempts_left)
            attempt = sim.attempt(mandate.mandate_id, slot, day)
            attempts.append(attempt)
            last_day = day
            attempts_left -= 1
            if attempt.outcome != Outcome.STILL_PENDING:
                break
        iatrogenic = sum(1 for a in attempts if a.iatrogenic_insufficient_funds)
        results.append(
            CauseAwareMandateResult(
                mandate_id=mandate.mandate_id, action=action,
                attempts=tuple(attempts), iatrogenic_failures=iatrogenic,
            )
        )
    return CauseAwareBatchResult(
        arm=sim.arm,
        profile=profile.value,
        n_mandates=len(results),
        n_stopped_reauth=sum(1 for r in results if r.action == Action.REAUTH),
        n_offered_exit=sum(1 for r in results if r.action == Action.OFFER),
        n_attempted=sum(1 for r in results if r.action == Action.ATTEMPT),
        total_attempts_spent=sum(len(r.attempts) for r in results),
        total_iatrogenic_failures=sum(r.iatrogenic_failures for r in results),
        per_mandate=tuple(results),
    )
