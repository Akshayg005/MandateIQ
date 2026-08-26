"""Pre-registered synthetic mandate simulator. Frozen alongside sim_config.yaml
and protocol.md: the generative mechanism is part of what is pre-registered,
not just the numbers in the config -- a session could otherwise "improve" a
policy's score by quietly changing hazard-curve logic here after seeing a
result, without touching a single frozen number.

Every mandate entering this simulator has already had its slot-1 (original)
attempt fail -- that failure is what puts a mandate into a recovery system in
the first place. Only slots 2/3/4 (up to 3 retries, NPCI's cap) are
simulated as decisions; slot 1 is given.

Three arms (see sim_config.yaml, PLAN_DETAIL.md section 8.1 decision 1):
  nominal       -- full multinomial-logit hazard. The "easy" arm: a
                   correctly-specified competing-risks model should do well.
  misspecified  -- cloglog link (not logit), heavier-tailed CANT_PAY_NOW
                   replenishment, and per-attempt cause-switching. A model
                   assuming the nominal generative story should degrade here.
  coupled       -- mandates share a household balance; recovering one
                   mandate can starve a sibling mandate's recovery later the
                   same cycle, purely through liquidity contention. This is
                   the arm that can reproduce a policy misreading its own
                   debit storm as customer illiquidity (PLAN_DETAIL.md
                   section 8.2 finding 1) -- the only one of the three that
                   varies independence rather than functional form.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

from src.core.types import Cause, Outcome

CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "sim_config.yaml"

ARMS = ("nominal", "misspecified", "coupled")
CAUSES = (Cause.CANT_PAY_NOW, Cause.CANT_PAY_EVER, Cause.WONT_PAY)


def load_config(path: pathlib.Path | None = None) -> dict:
    """Read and parse sim_config.yaml. `path` is overridable only for tests
    that need to exercise a deliberately-broken config -- production callers
    always get the one frozen file."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class SimMandate:
    """Static attributes of one simulated mandate-cycle. `initial_cause` is
    the cause it was generated with; under the misspecified arm the
    effective cause used for hazard draws can later diverge from this (see
    Simulator.effective_cause) -- callers scoring the simulation should use
    initial_cause as ground truth for what kind of mandate this really is.

    `initial_cause` AND `household_id` are unobservable ground truth. A real
    payment aggregator has no way to know which mandates share a bank
    account, any more than it knows a mandate's true latent cause -- a
    policy under test must never read either field. (Evaluation/scoring
    code reading them is fine; that is what ground truth is for.)"""

    mandate_id: str
    cycle_id: int
    amount_paise: int
    ceiling_paise: int
    category: str
    household_id: str | None
    initial_cause: Cause


@dataclass(frozen=True)
class AttemptResult:
    """`iatrogenic_insufficient_funds` is an evaluation-only diagnostic --
    ground truth about WHY this attempt failed that a real issuer decline
    string never carries. A policy under test must never read it; it exists
    so scoring code and tests can measure the coupled arm's storm effect
    directly instead of inferring it after the fact."""

    mandate_id: str
    slot: int
    on_day: int
    outcome: Outcome
    iatrogenic_insufficient_funds: bool = False


@dataclass
class _MandateState:
    effective_cause: Cause
    last_attempt_day: int = 0
    last_slot_seen: int = 1


def _weighted_choice(rng: np.random.Generator, options: tuple, weights: dict) -> object:
    probs = [weights[o] for o in options]
    return options[rng.choice(len(options), p=probs)]


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = vals - vals.max()
    exp = np.exp(vals)
    probs = exp / exp.sum()
    return dict(zip(keys, probs.tolist()))


def _logits_from_base_rates(base_recovery: float, base_dead: float, base_optout: float) -> dict[str, float]:
    """Inverse-softmax: the logit-space scores (vs. survive=0) that
    reproduce these exact base rates when no context adjustment is applied."""
    p_survive = 1.0 - base_recovery - base_dead - base_optout
    return {
        "recover": float(np.log(base_recovery / p_survive)),
        "dead": float(np.log(base_dead / p_survive)),
        "optout": float(np.log(base_optout / p_survive)),
        "survive": 0.0,
    }


class Simulator:
    """Drives one batch of mandates through one arm. Stateful: household
    balances (coupled arm) and per-mandate cause/timing state persist across
    calls to attempt() within one Simulator instance, in whatever order the
    caller schedules them -- that ordering is what lets a policy's own
    scheduling choices produce (or avoid) a debit storm under the coupled
    arm."""

    def __init__(self, arm: str, config: dict | None = None, seed: int | None = None):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, must be one of {ARMS}")
        self.arm = arm
        self.config = config if config is not None else load_config()
        self._rng = np.random.default_rng(
            seed if seed is not None else self.config["seed"]
        )
        self._mandates = self._generate_mandates()
        self._state: dict[str, _MandateState] = {
            m.mandate_id: _MandateState(effective_cause=m.initial_cause) for m in self._mandates
        }
        self._household_balance: dict[str, int] = {}
        if arm == "coupled":
            lo, hi = self.config["arms"]["coupled"]["household_balance_range"]
            for m in self._mandates:
                if m.household_id not in self._household_balance:
                    self._household_balance[m.household_id] = int(self._rng.integers(lo, hi + 1))

    # -- generation -----------------------------------------------------

    def _generate_mandates(self) -> tuple[SimMandate, ...]:
        cfg = self.config
        rng = self._rng
        n = cfg["n_mandates"]

        cause_names = list(cfg["cause_mix"])
        cause_probs = [cfg["cause_mix"][c] for c in cause_names]
        causes = rng.choice(cause_names, size=n, p=cause_probs)

        cat_names = list(cfg["category_mix"])
        cat_probs = [cfg["category_mix"][c] for c in cat_names]
        categories = rng.choice(cat_names, size=n, p=cat_probs)

        amt_cfg = cfg["amount_paise"]
        below_lo, below_hi = amt_cfg["below_afa_range"]
        above_lo, above_hi = amt_cfg["above_afa_range"]
        is_below = rng.random(n) < amt_cfg["below_afa_frac"]
        amounts = np.where(
            is_below,
            rng.integers(below_lo, below_hi + 1, size=n),
            rng.integers(above_lo, above_hi + 1, size=n),
        )
        ceil_lo, ceil_hi = amt_cfg["ceiling_multiplier_range"]
        ceil_mult = rng.uniform(ceil_lo, ceil_hi, size=n)
        ceilings = np.round(amounts * ceil_mult).astype(int)

        household_size = None
        if self.arm == "coupled":
            household_size = cfg["arms"]["coupled"]["household_size"]

        mandates = []
        for i in range(n):
            household_id = f"H{i // household_size}" if household_size else None
            mandates.append(
                SimMandate(
                    mandate_id=f"M{i:04d}",
                    cycle_id=1,
                    amount_paise=int(amounts[i]),
                    ceiling_paise=int(ceilings[i]),
                    category=str(categories[i]),
                    household_id=household_id,
                    initial_cause=Cause(str(causes[i])),
                )
            )
        return tuple(mandates)

    @property
    def mandates(self) -> tuple[SimMandate, ...]:
        return self._mandates

    def effective_cause(self, mandate_id: str) -> Cause:
        """The cause currently driving this mandate's hazard draws -- equal
        to initial_cause except under the misspecified arm's cause-switching,
        after at least one switch has occurred. For inspection/testing only;
        a policy under test must never call this."""
        return self._state[mandate_id].effective_cause

    def household_balance(self, household_id: str) -> int:
        return self._household_balance[household_id]

    # -- the one behavioral entry point ----------------------------------

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

        days_since_last = on_day - state.last_attempt_day
        in_salary_window = 1 <= on_day <= 5
        retries_so_far = slot - 2  # 0 at slot 2, 1 at slot 3, 2 at slot 4

        if self.arm == "misspecified":
            switch_prob = self.config["arms"]["misspecified"]["cause_switch_prob"]
            if self._rng.random() < switch_prob:
                cause_names = list(self.config["cause_mix"])
                cause_probs = [self.config["cause_mix"][c] for c in cause_names]
                state.effective_cause = Cause(
                    str(self._rng.choice(cause_names, p=cause_probs))
                )
        cause = state.effective_cause

        outcome, iatrogenic = self._draw_outcome(
            mandate=mandate,
            cause=cause,
            in_salary_window=in_salary_window,
            retries_so_far=retries_so_far,
            days_since_last=days_since_last,
        )

        state.last_attempt_day = on_day
        state.last_slot_seen = slot

        return AttemptResult(
            mandate_id=mandate_id, slot=slot, on_day=on_day,
            outcome=outcome, iatrogenic_insufficient_funds=iatrogenic,
        )

    def _by_id(self, mandate_id: str) -> SimMandate:
        for m in self._mandates:
            if m.mandate_id == mandate_id:
                return m
        raise KeyError(mandate_id)

    # -- hazard mechanics -------------------------------------------------

    def _draw_outcome(
        self, *, mandate: SimMandate, cause: Cause, in_salary_window: bool,
        retries_so_far: int, days_since_last: int,
    ) -> tuple[Outcome, bool]:
        h = self.config["hazards"][cause.value]
        base_recovery, base_dead, base_optout = h["base_recovery"], h["base_dead"], h["base_optout"]
        logits = _logits_from_base_rates(base_recovery, base_dead, base_optout)

        if cause == Cause.CANT_PAY_NOW and in_salary_window:
            logits["recover"] += h.get("salary_window_bonus_logit", 0.0)
        if cause == Cause.WONT_PAY:
            logits["optout"] += h.get("optout_escalation_logit_per_attempt", 0.0) * retries_so_far

        link = self.config["arms"][self.arm]["link"]
        if link == "logit":
            probs = _softmax(logits)
        elif link == "cloglog":
            probs = self._cloglog_probs(logits, cause, days_since_last)
        else:
            raise ValueError(f"unknown link {link!r}")

        options = ("recover", "dead", "optout", "survive")
        draw = _weighted_choice(self._rng, options, probs)

        outcome_map = {
            "recover": Outcome.RECOVERED, "dead": Outcome.DEAD,
            "optout": Outcome.OPTED_OUT, "survive": Outcome.STILL_PENDING,
        }
        outcome = outcome_map[draw]
        iatrogenic = False

        if self.arm == "coupled" and outcome == Outcome.RECOVERED and mandate.household_id:
            outcome, iatrogenic = self._apply_household_coupling(mandate)

        return outcome, iatrogenic

    def _cloglog_probs(self, logits: dict[str, float], cause: Cause, days_since_last: int) -> dict[str, float]:
        """Two-stage: cloglog decides P(any terminal event this attempt),
        applied directly to the combined linear score (not round-tripped
        through the logit link's own probability -- doing that would make
        cloglog(logit_inverse(p)) mathematically equal to p again, silently
        collapsing this arm back onto nominal). The original relative
        shares among {recover, dead, optout} decide which one, with
        CANT_PAY_NOW's recovery share boosted by a heavy-tailed function of
        days_since_last (misspecified arm only)."""
        non_ref = np.array([logits["recover"], logits["dead"], logits["optout"]])
        s_combined = float(np.logaddexp.reduce(non_ref))  # log(sum(exp(.)))
        p_terminal = 1 - np.exp(-np.exp(s_combined))
        p_terminal = min(max(p_terminal, 0.0), 1.0 - 1e-9)

        exp_non_ref = np.exp(non_ref - non_ref.max())
        shares = exp_non_ref / exp_non_ref.sum()
        share = {"recover": float(shares[0]), "dead": float(shares[1]), "optout": float(shares[2])}
        if cause == Cause.CANT_PAY_NOW:
            exponent = self.config["arms"]["misspecified"]["replenishment_exponent"]
            boost = max(days_since_last, 1) ** exponent
            share["recover"] *= boost
            total = sum(share.values())
            share = {k: v / total for k, v in share.items()}

        return {
            "recover": p_terminal * share["recover"],
            "dead": p_terminal * share["dead"],
            "optout": p_terminal * share["optout"],
            "survive": 1 - p_terminal,
        }

    def _apply_household_coupling(self, mandate: SimMandate) -> tuple[Outcome, bool]:
        """This attempt's own hazard draw said RECOVERED. Whether it
        actually can, given the household's shared balance, is a separate
        question -- and a failure here is iatrogenic: caused by our own
        prior debit(s) against this household, not by this mandate's own
        true state.

        A debit either succeeds in full or not at all -- UPI AutoPay has no
        partial-debit semantics, so there is no "probably recovers anyway"
        branch here. That matters beyond realism: an earlier version of
        this method gave a below-balance attempt a chance to succeed
        weighted by balance/amount, but on success still credited the FULL
        mandate amount while only debiting the household down to zero --
        fabricating money (verified: a full batch run recovered 1.7x the
        total liquidity that existed across every household). A household
        can never pay out more than it demonstrably has.
        """
        balance = self._household_balance[mandate.household_id]
        if balance >= mandate.amount_paise:
            self._household_balance[mandate.household_id] = balance - mandate.amount_paise
            return Outcome.RECOVERED, False
        return Outcome.STILL_PENDING, True
