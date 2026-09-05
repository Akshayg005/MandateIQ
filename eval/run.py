"""B13 batch driver: every arm x every regime x both compliance profiles.

One command produces every number in the report:

    python -m eval.run --all-regimes --both-profiles

It writes a single machine-readable artifact, `reports/regimes.json`, which
`eval/report.py` then renders. Nothing in the report is computed anywhere
else -- if a number appears in `reports/` it came out of this file, so
"reproducible by one command" is a property of the pipeline rather than a
claim in prose.

Per PLAN_DETAIL.md's B13 row this module must NOT print per-mandate logs to
stdout. It prints one progress line per cell and a final summary; the
per-mandate detail goes to the JSON artifact.

Three things here are deliberate and easy to get wrong:

1. **The hazard model is fit on the NOMINAL corpus, once, and reused under
   every regime.** A regime is a shift the deployed system did not see
   coming; refitting per regime would measure a model that already knew
   about the shift, which is not the question. The engine is therefore
   MISSPECIFIED under every regime except baseline, on purpose.

2. **The conformal gate is calibrated once, on baseline**, from a separate
   calibration draw with its own seed, and the SAME fitted gate is used
   under every regime. Split conformal's 95% coverage guarantee holds under
   exchangeability; a regime breaks exchangeability by construction. Coverage
   is therefore MEASURED per regime rather than assumed, and degradation is a
   result, not a bug. `reports/regimes.json` records which gate was live for
   every cell so report.py can print the coverage claim only where the real
   gate actually ran.

3. **Error costs are per-mandate counterfactuals under common random
   numbers -- NOT the exact realisation.** When the engine stops early we
   deepcopy the simulator at that moment and keep grinding on the copy. An
   earlier version of this docstring claimed the copy replays "the same
   random draws the real run would have seen"; that is false and worth
   stating plainly, because one `np.random.Generator` serves the whole
   batch, so the draws the counterfactual consumes are the ones the real
   run gives to LATER mandates. There is no "the draws this mandate would
   have seen" in a shared-stream simulator. As variance reduction this is
   sound and better than re-seeding; as an exactness claim it was wrong.

   Two further biases, disclosed rather than fixed (fixing them means
   changing the frozen simulator, which is not permitted):
   * Coupled arm: households are four consecutive mandate indices and the
     batch is iterated in index order, so a deepcopy taken on the first
     mandate of a household captures the shared balance BEFORE its
     siblings have drawn on it. `missed_recovery` is therefore biased by
     position-within-household in exactly the arm built to model
     contention.
   * The counterfactual grinds on consecutive days from where we stopped,
     which always lands inside the days-1-5 salary window and so always
     collects `salary_window_bonus_logit` -- a bonus under baseline, a
     penalty under `delayed_salary`. It is close to the most favourable
     counterfactual available, which makes `missed_recovery` an upper
     bound on what we gave up, not a point estimate.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from eval import regimes as regimes_mod
from eval.allocator_sweep import (
    INTENT_CHANNEL_SOURCE_VERSION,
    PROXY_SOURCE_VERSION,
    WONTPAY_CHANNEL_SOURCE_VERSION,
    WontPayChannel,
    apply_intent_channel,
    channel_decline_class,
    fit_nominal_hazard_model,
    hazard_from_fit,
    initial_belief,
)
from eval.frozen.scoring import MandateResult, aggregate, score_mandate
from eval.frozen.simulator import Simulator, load_config
from src.core.types import Action, Cause, DeclineClass, MandateState, Outcome, Profile
from src.model import conformal
from src.policy import belief as belief_mod
from src.policy.allocator import AllocationContext, AllocatorError, Plan, solve
from src.policy.constraints import MAX_ATTEMPTS, afa_free_limit_paise, requires_afa
from src.policy.costs import PolicyCosts, load as load_costs
from src.policy.gate import ConformalCauseGate, FullSetGate

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "regimes.json"

ALL_ARMS: tuple[str, ...] = ("nominal", "misspecified", "coupled")
ALL_PROFILES: tuple[Profile, ...] = (Profile.strict, Profile.permissive)

# Distinct RNG offsets, so adding a stream can never perturb an existing one.
# 500_000 is allocator_sweep's slot-1 stream; these must not collide with it.
_SLOT1_OFFSET = 700_000
_CALIB_SLOT1_OFFSET = 900_000
# R5: the synthetic WONT_PAY channel draws from its OWN stream. Sharing
# _SLOT1_OFFSET would shift every slot-1 decline draw the moment the
# channel was switched on, silently changing every number this project has
# already published for reasons unrelated to the channel.
_CHANNEL_OFFSET = 1_100_000
_CALIB_CHANNEL_OFFSET = 1_300_000

# The channel-quality operating point the PUBLISHED grid runs at.
# PRE-REGISTERED in eval/offramp_channel.py's module docstring before the
# first sweep ran (reports/gates.md, R5; DECISIONS.md, 2026-09-05) and
# imported from there rather than restated, so the two cannot drift apart.
# The "decline" kind, not "intent": folding a fabricated support-ticket
# signal into the headline grid is a bigger fabrication than a decline
# string (DECISIONS.md, 2026-09-04, R0's own words). The intent channel is
# measured in eval/offramp_channel.py's sweep instead.
DEFAULT_CHANNEL_KIND = "decline"
# The calibration draw's own seed. Disjoint from any reported seed, and its
# mandate ids are namespaced (see _calib_group_id) so a real disjointness
# check WOULD have something meaningful to compare against.
#
# CORRECTED, R5 review pass, 2026-09-05 (stats-reviewer): this comment
# previously claimed conformal.assert_disjoint() "can actually check the
# split" here, as though it does. It is never called on this path --
# verified: `assert_disjoint` appears only in eval/model_fit_report.py, a
# different, unrelated report. Disjointness holds by CONSTRUCTION alone
# (CALIB_SEED sits outside the reported 0-7 range, and calib ids carry the
# `calib424242:` prefix _calib_group_id adds, a different namespace
# entirely from the plain/slot-qualified ids the reported cells use) --
# not by a runtime assertion. Not wired up here: a real check would compare
# sets living in genuinely different id formats, which would pass
# trivially regardless of whether the SEEDS actually overlapped, so it
# would not catch the failure mode assert_disjoint() exists to catch.
CALIB_SEED = 424_242

CAUSE_ORDER: tuple[Cause, ...] = tuple(belief_mod.CAUSE_ORDER)

# R4, 2026-09-04 (reports/gates.md, "Post-B16 remediation gates"): relocated
# to src/policy/belief.py as TERMINAL_OBSERVED_CAUSE_PROBS /
# TERMINAL_OBSERVATION_SOURCE_VERSION -- src/execute/cycle.py (R4) is the
# first PRODUCTION caller of observe_terminal(), and src/ must never import
# eval/, so a value only this module defined would be unreachable from
# there. Values, derivation and the measured 0.8991/0.9040 numbers are
# unchanged; see belief.py's own docstring for the full writeup (the
# ~10%-wrong-cause reasoning, the degenerate-1.0 correction this project
# made and reversed the same day, and why RECOVERED is deliberately absent
# -- the cycle succeeded, there is no cause left to decide, and
# _run_engine_mandate must not call belief_mod.observe_terminal or
# ctx.with_terminal for it). Aliased under their original names here so
# every existing call site below needs no further change.
_TERMINAL_OBSERVED_CAUSE_PROBS = belief_mod.TERMINAL_OBSERVED_CAUSE_PROBS
TERMINAL_OBSERVATION_SOURCE_VERSION = belief_mod.TERMINAL_OBSERVATION_SOURCE_VERSION


# --- results -----------------------------------------------------------------


@dataclass
class CellResult:
    """One (regime, arm, profile, policy) cell -- the three bars plus
    everything needed to say what the policy actually did and what it cost."""

    regime: str
    arm: str
    profile: str
    policy: str
    seed: int
    gate_kind: str

    n_mandates: int = 0
    # Total value at stake in this cell -- the denominator a recovery
    # PERCENTAGE needs. Without it the report can only quote absolute paise,
    # and "recovered_pct" in reports/results.json was rendering as "?".
    billable_paise: int = 0
    # the three bars
    recovered_paise: int = 0
    attempts_spent: int = 0
    mandates_preserved: int = 0
    # outcome breakdown
    recovered: int = 0
    dead: int = 0
    opted_out: int = 0
    censored: int = 0
    iatrogenic_failures: int = 0
    # what the policy chose
    n_attempt: int = 0
    n_offer: int = 0
    n_reauth: int = 0
    n_stop: int = 0
    n_above_afa: int = 0
    # Post-terminal re-solves that returned ATTEMPT -- i.e. the allocator
    # wanting to retry an instrument the issuer just confirmed dead. Counted
    # because it was previously falling through unrecorded.
    n_attempt_after_terminal: int = 0
    # the two error costs (protocol.md: reported alongside, never folded in)
    missed_recovery_count: int = 0
    missed_recovery_paise: int = 0
    # PRE-REGISTERED Day-1 metric, MEANING UNCHANGED BYTE FOR BYTE (R2b's
    # lesson: never redefine a pre-registered metric after seeing it). It
    # counts an OFFER to a mandate the exact counterfactual says WOULD have
    # paid -- i.e. it is computed inside the `would_pay` branch below.
    false_offramp_count: int = 0
    false_offramp_paise: int = 0
    # R5, ADDED BESIDE IT, never folded into it. false_offramp_count had no
    # denominator: an OFFER to a mandate that would NOT have paid was
    # counted nowhere, so root CLAUDE.md's "report BOTH error costs" was one
    # real number and one structural zero. These three make the pair two
    # measurements.
    #   offramp_scored_count -- OFFERs that reached the counterfactual at
    #     all, i.e. the exact denominator of false_offramp_count. Normally
    #     equal to n_offer; it can be smaller, because an OFFER returned by
    #     the POST-TERMINAL re-solve lands on an already-resolved mandate
    #     that the counterfactual branch skips. Reported rather than assumed
    #     so `false_offramp_count / n_offer` can never quietly become a rate
    #     with the wrong denominator.
    #   true_offramp_* -- the counterpart: an OFFER to a mandate that would
    #     NOT have recovered. Not "correct" (the counterfactual cannot see
    #     intent), only "cost us no recovery".
    offramp_scored_count: int = 0
    true_offramp_count: int = 0
    true_offramp_paise: int = 0
    # issuer_outage's own pre-registered falsification criterion: REAUTH
    # issued on a mandate whose true cause is NOT CANT_PAY_EVER. PRE-
    # REGISTERED, NEVER REDEFINED (DECISIONS.md, 2026-09-04, R0) -- the
    # four fields below are ADDED alongside it, R2b, to separate two things
    # this one criterion conflates: REAUTH has a COMPLIANCE route (above
    # the AFA cliff, clause 8(a)/8(b), legally mandatory regardless of
    # cause) and an INFERENCE route (belief-driven); this criterion scores
    # both the same way.
    false_reauth_count: int = 0
    false_reauth_paise: int = 0
    # REAUTH issued via the compliance route (requires_afa() was true) --
    # says nothing about whether the belief was right.
    compliance_reauth_count: int = 0
    # false_reauth_count, restricted to the INFERENCE route only (excludes
    # every compliance-route REAUTH, which is correct by law regardless of
    # cause). The genuinely-interesting "did the belief mislead us" count.
    false_reauth_inference_count: int = 0
    # false_reauth_count / false_reauth_inference_count, but scored against
    # sim.effective_cause() instead of m.initial_cause -- the misspecified
    # arm's cause-switching means a mandate can become CANT_PAY_EVER after
    # starting as something else, and initial_cause then scores a REAUTH
    # that turned out to be correct as if it were wrong.
    false_reauth_count_effective: int = 0
    false_reauth_inference_count_effective: int = 0
    # gate evidence, engine only. coverage_marginal/singleton_rate/
    # singleton_wont_pay_rate/mean_set_size/coverage_per_class are computed
    # over LIVE queries only (R2, 2026-09-04) -- coverage_n is that same
    # live-query count, and coverage_n_retrospective (excluded from all of
    # the above) is reported alongside for transparency, never folded in.
    coverage_marginal: float | None = None
    coverage_n: int = 0
    coverage_n_retrospective: int = 0
    singleton_wont_pay_rate: float | None = None
    singleton_rate: float | None = None
    mean_set_size: float | None = None
    coverage_per_class: dict[str, float] = field(default_factory=dict)
    # R5: the synthetic WONT_PAY channel this cell ran under, and the ROC
    # it ACTUALLY realised on this cell's own draws -- never a restatement
    # of the configured (tpr, fpr). "off" is the pre-R5 configuration every
    # previously published number came from.
    channel_kind: str = "off"
    channel_tpr: float | None = None
    channel_fpr: float | None = None
    channel_n_wont_pay: int = 0
    channel_positive_on_wont_pay: int = 0
    channel_n_other: int = 0
    channel_positive_on_other: int = 0
    violations: list[str] = field(default_factory=list)
    # Wall clock. Measured, deliberately NOT serialised -- see
    # UNSERIALISED_CELL_FIELDS.
    seconds: float = 0.0


# Fields excluded from reports/regimes.json. B13's gate is "every number
# reproducible by one command"; that holds for every value that means
# anything, but a per-cell wall-clock timing made the artifact differ
# byte-for-byte between two runs of the same seed, so anyone checking the
# claim by hashing rather than by reading got a mismatch and no way to tell
# jitter from a real divergence. Nothing reads `seconds`. Named explicitly,
# rather than filtered by a rule, so adding a metric cannot accidentally
# drop it.
UNSERIALISED_CELL_FIELDS = frozenset({"seconds"})


def _serialise_cell(cell: CellResult) -> dict:
    """A cell as it appears in the artifact: every field except the
    non-reproducible ones."""
    return {k: v for k, v in asdict(cell).items() if k not in UNSERIALISED_CELL_FIELDS}


# --- the engine policy -------------------------------------------------------


def _bind(gate, key: str):
    """Bind a per-decision smoothing key if the gate takes one. FullSetGate
    is keyless by construction (it ignores the belief entirely), so this is
    a no-op there and the driver stays gate-agnostic."""
    bind = getattr(gate, "bind", None)
    return bind(key) if bind is not None else gate


class _RecordingGate:
    """Delegates to the real gate and records every query.

    Coverage was previously measured by replaying only the 200 slot-1
    beliefs. The gate is actually consulted ~4,900 times per cell -- the
    allocator re-solves after every attempt with an updated belief -- and
    the unmeasured queries are exactly the concentrated post-update ones
    where the gate emits singletons and where a conformal error becomes a
    wrong ACTION rather than an abstention. Replaying slot 1 also ignored
    `arm` and `profile` entirely, so six distinct numbers were being printed
    as thirty-two. Recording what the gate was actually asked fixes both.
    (stats-reviewer, 2026-08-31.)

    R2, 2026-09-04 (payments-domain review): each recorded query now also
    carries whether its belief was a LIVE, ordinarily-inferred one or a
    RETROSPECTIVE one from belief.observe_terminal() (tagged
    `;observed=terminal` in `Belief.provenance`). Reason: `calib_conf` (the
    conformal predictor's own calibration pool, see fit_gate()) is drawn
    entirely from slot-1 LIVE beliefs -- an observe_terminal() belief is
    never exchangeable with that pool (it is a hand-constructed, ~90%-on-
    one-cause distribution that never occurs during ordinary inference), so
    mixing it into a coverage/singleton-rate MEASUREMENT would silently
    answer a different question ("how often does a synthetic belief we
    built land in its own predicted set", which is close to tautological)
    while looking like the same "is the live gate well-calibrated" number.
    The Plan object's own `conformal_set` audit field is UNCHANGED by this
    -- _build_plan() still queries the gate unconditionally for every
    solve() call, live or retrospective; only the AGGREGATE coverage
    statistics computed from this log (_score_recorded_queries) now
    separate the two.
    """

    def __init__(self, inner, mandate_id: str | None = None) -> None:
        self._inner = inner
        self._mandate_id = mandate_id
        self.queries: list[tuple[str, frozenset, bool]] = []

    def bind(self, key: str) -> "_RecordingGate":
        g = _RecordingGate(_bind(self._inner, key), key.split(":")[0])
        g.queries = self.queries          # one shared log per cell
        return g

    def pred_set(self, b: Belief) -> frozenset[Cause]:
        s = self._inner.pred_set(b)
        is_retrospective = ";observed=terminal" in b.provenance
        self.queries.append((self._mandate_id or "", s, is_retrospective))
        return s


def _initial_context(m, profile: Profile, costs: PolicyCosts) -> AllocationContext:
    return AllocationContext(
        mandate_id=m.mandate_id,
        cycle_id=m.cycle_id,
        profile=profile,
        amount_paise=m.amount_paise,
        ceiling_paise=m.ceiling_paise,
        category=m.category,
        plan_day=1,
        attempts_used=1,
        committed_days=(1,),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def _counterfactual_recovers(sim: Simulator, mandate_id: str, from_slot: int,
                             last_day: int) -> bool:
    """Would this mandate have recovered if we had kept grinding?

    Called with `sim` ALREADY deepcopied by the caller, at the exact moment
    the engine stopped -- so the RNG state, the household balance, and the
    effective cause are the ones the real run would have carried forward.
    The counterfactual policy is the incumbent's: spend every remaining slot,
    on the tightest legal cadence (the day after the last attempt), until a
    terminal outcome or the NPCI budget runs out.

    Ladder day-offsets are NOT reused here: they are absolute days (1/2/3)
    and the engine may already have attempted past them, which the frozen
    simulator rejects as out-of-order. Consecutive days from where we stopped
    is the same question -- "keep trying" -- expressed legally.
    """
    day = last_day
    for slot in range(from_slot, MAX_ATTEMPTS + 1):
        day += 1
        result = sim.attempt(mandate_id, slot=slot, on_day=day)
        if result.outcome == Outcome.RECOVERED:
            return True
        if result.outcome in (Outcome.DEAD, Outcome.OPTED_OUT):
            return False
    return False


@dataclass
class DecisionTrace:
    """One solve() call and what came back from it -- the Plan the allocator
    actually produced, plus the simulated outcome if that Plan was executed.

    B14's drill-down (belief, chosen slot, binding constraint, conformal set)
    is satisfiable only from a Plan, and this loop otherwise reduces every
    Plan to a counter and discards it. Recording is strictly additive: it
    draws no randomness and takes no branch, so a traced run and an untraced
    run are the same run. tests/eval/test_export_mandates.py compares the
    whole CellResult across both to keep that true.

    `outcome` is None for a decision that spent no slot -- a REAUTH, an
    OFFER, a STOP, or the post-terminal re-solve -- which is exactly the
    distinction the dashboard needs to show an action that cost nothing.
    """

    plan: "Plan"
    outcome: str | None = None


def _run_engine_mandate(m, sim: Simulator, profile: Profile, hazard,
                        costs: PolicyCosts, gate, b, cell: CellResult,
                        trace: list[DecisionTrace] | None = None,
                        channel=None):
    """Drive one mandate through the allocator. Returns the ordered attempts
    actually made; mutates `cell`'s action counters and error costs. When
    `trace` is given, appends one DecisionTrace per solve() call.

    Adapted from eval/allocator_sweep.py's _run_one_mandate, which answers a
    different question (B8's gate criteria: did we attempt, how often) and so
    throws away the AttemptResults the three bars are computed from.
    """
    ctx = _initial_context(m, profile, costs)
    attempts = []
    stopped_action: Action | None = None
    last_day = 1

    while ctx.attempts_used < MAX_ATTEMPTS:
        # Bind the conformal gate to THIS decision point. The smoothing key
        # must be a per-row id, never a function of the belief -- see
        # ConformalCauseGate's docstring for the bug that motivated this.
        g = _bind(gate, f"{m.mandate_id}:{m.cycle_id}:s{ctx.attempts_used + 1}")
        try:
            plan = solve(b, ctx, hazard=hazard, costs=costs, gate=g)
        except AllocatorError as exc:
            cell.violations.append(f"{m.mandate_id}: AllocatorError: {exc}")
            stopped_action = Action.STOP
            break

        if plan.chosen_action != Action.ATTEMPT:
            stopped_action = plan.chosen_action
            if trace is not None:
                trace.append(DecisionTrace(plan))
            break

        committed = plan.committed[0]
        if committed.amount_paise > ctx.ceiling_paise:
            cell.violations.append(
                f"{m.mandate_id}: committed {committed.amount_paise} over "
                f"ceiling {ctx.ceiling_paise}"
            )
        result = sim.attempt(m.mandate_id, slot=committed.slot, on_day=committed.on_day)
        if trace is not None:
            trace.append(DecisionTrace(plan, outcome=result.outcome.name))
        attempts.append(result)
        last_day = committed.on_day
        ctx = ctx.with_attempt(committed.on_day)
        cell.n_attempt += 1

        # R5: with a "decline" channel live this may be CUSTOMER_DECLINED
        # instead of the plain proxy -- see eval/allocator_sweep.py's
        # channel section.
        #
        # CORRECTED, R5 review pass, 2026-09-05 (stats-reviewer): this
        # comment previously claimed the channel "consumes exactly one draw
        # per decision point regardless of which branch is taken". Verified
        # FALSE for two of the four outcomes: `channel_decline_class()`
        # returns the proxy immediately, without calling `channel.fires()`,
        # whenever `_proxy_decline_class(outcome) is None` -- true for
        # RECOVERED and OPTED_OUT (`_OUTCOME_TO_DECLINE_CLASS` maps both to
        # None). So no draw is consumed on those two outcomes; one IS
        # consumed on DEAD, but `dc` is then never read below (the terminal
        # branch conditions belief on the OBSERVED outcome via
        # `observe_terminal()`, never on `dc`) -- that draw still enters
        # `channel.log` (and so the published ROC/repeat-rate) without ever
        # reaching a belief. Verified benign for the ROC estimate: `fires()`
        # draws from a stream independent of `sim.attempt()`'s own, so the
        # outcome-dependent selection of WHICH mandates get a DEAD draw is
        # independent of that draw's own result given cause -- tpr/fpr stay
        # unbiased, only the sample size differs from a naive count of
        # decision points.
        dc = channel_decline_class(result.outcome, cause=m.initial_cause, channel=channel)
        if result.outcome != Outcome.STILL_PENDING:
            # Terminal. The ATTEMPT sequence is over, but the DECISION
            # sequence is not: a dead instrument is exactly when REAUTH is
            # the right next action (CLAUDE.md's own cause->action table).
            #
            # R2, 2026-09-04: DEAD and OPTED_OUT are OBSERVED facts, not
            # decline-string evidence to Bayes-update on -- belief_mod.
            # observe_terminal() replaces belief with a MEASURED posterior
            # (see _TERMINAL_OBSERVED_CAUSE_PROBS above -- ~90% confident,
            # not the degenerate 1.0 an earlier same-day version assumed)
            # regardless of what came before, and ctx.with_terminal() marks
            # the context so permitted() denies ATTEMPT (DEAD) or everything
            # but STOP (OPTED_OUT, via the existing opted_out field) at the
            # re-solve below -- that denial is a HARD rule, unaffected by
            # the exact belief value. Before this existed: an ordinary
            # update(..., CARD_EXPIRED) often could not move b.dominant() to
            # CANT_PAY_EVER after a couple of INSUFFICIENT_FUNDS-shaped
            # updates, the context was never marked at all, and the
            # re-solve could (and did) return ATTEMPT on an instrument the
            # issuer had just confirmed dead (reports/gates.md, R2a,
            # measured: 4,032 such events across 256 engine cells). For
            # OPTED_OUT specifically, _proxy_decline_class() returns None,
            # so the OLD `if dc is not None:` gate skipped this branch
            # ENTIRELY -- no decision was ever recorded for what happens
            # after an opt-out. `dc`/`_proxy_decline_class` play no further
            # role for a terminal outcome; the branch below checks
            # `result.outcome` directly instead.
            observed_probs = _TERMINAL_OBSERVED_CAUSE_PROBS.get(result.outcome)
            if observed_probs is not None:
                b = belief_mod.observe_terminal(
                    observed_probs, source_version=TERMINAL_OBSERVATION_SOURCE_VERSION,
                )
                ctx = ctx.with_terminal(result.outcome)
                gf = _bind(gate, f"{m.mandate_id}:{m.cycle_id}:final")
                try:
                    final = solve(b, ctx, hazard=hazard, costs=costs, gate=gf)
                    stopped_action = final.chosen_action
                    if trace is not None:
                        # The post-terminal re-solve spends no slot, so it
                        # carries no outcome -- but it is the decision that
                        # produces a REAUTH (or STOP, for OPTED_OUT), and
                        # the drill-down would be missing the engine's
                        # answer to "the instrument is dead / the customer
                        # left, now what?" without it.
                        trace.append(DecisionTrace(final))
                except AllocatorError as exc:
                    cell.violations.append(f"{m.mandate_id}: final AllocatorError: {exc}")
            # RECOVERED (observed_probs is None): the cycle succeeded,
            # nothing left to decide -- no belief/context change, no
            # re-solve. Unchanged from before this fix.
            break

        if dc is not None:
            source = (
                WONTPAY_CHANNEL_SOURCE_VERSION
                if dc == DeclineClass.CUSTOMER_DECLINED
                else PROXY_SOURCE_VERSION
            )
            b = belief_mod.update(b, dc, source_version=source)
        b = apply_intent_channel(b, m.initial_cause, channel)

    if stopped_action == Action.OFFER:
        cell.n_offer += 1
    elif stopped_action == Action.REAUTH:
        cell.n_reauth += 1
        # A pre-registered falsification criterion for issuer_outage: does
        # the engine over-issue REAUTH on instruments that are in fact
        # alive? Ground truth, read to SCORE and never to decide -- the same
        # privileged read eval/gate_criteria.py already makes.
        if m.initial_cause != Cause.CANT_PAY_EVER:
            cell.false_reauth_count += 1
            cell.false_reauth_paise += m.amount_paise

        # R2b, 2026-09-04: split the pre-registered count above by WHICH of
        # allocator.py's two REAUTH routes actually fired. amount_paise and
        # category never change across a mandate's cycle (with_terminal()
        # touches only instrument_dead/opted_out), so ctx's current values
        # are the mandate's real ones regardless of which ctx snapshot this
        # is. sim.effective_cause() is the SAME privileged, score-only read
        # m.initial_cause already is (never used to decide) -- see
        # eval/frozen/simulator.py's own docstring on that method.
        via_compliance = requires_afa(ctx.amount_paise, ctx.category)
        if via_compliance:
            cell.compliance_reauth_count += 1
        elif m.initial_cause != Cause.CANT_PAY_EVER:
            cell.false_reauth_inference_count += 1
        effective_cause = sim.effective_cause(m.mandate_id)
        if effective_cause != Cause.CANT_PAY_EVER:
            cell.false_reauth_count_effective += 1
            if not via_compliance:
                cell.false_reauth_inference_count_effective += 1
    elif stopped_action == Action.STOP:
        cell.n_stop += 1
    elif stopped_action == Action.ATTEMPT:
        # The post-terminal re-solve asked "now what?" and the allocator said
        # ATTEMPT -- on a mandate whose instrument the issuer just confirmed
        # DEAD, or which just OPTED_OUT. The first version of this function
        # counted only OFFER/REAUTH/STOP, so this case fell through every
        # branch and was silently discarded; payments-domain found it by
        # re-running the same solve path and getting 16 ATTEMPTs and 0
        # REAUTHs on 19 observed-DEAD mandates. Counting it is what makes
        # that visible in the report instead of invisible in the code.
        cell.n_attempt_after_terminal += 1

    # -- error costs, by exact counterfactual --------------------------------
    resolved = bool(attempts) and attempts[-1].outcome != Outcome.STILL_PENDING
    slots_left = MAX_ATTEMPTS - (1 + len(attempts))
    if not resolved and slots_left > 0:
        shadow = copy.deepcopy(sim)
        would_pay = _counterfactual_recovers(
            shadow, m.mandate_id, from_slot=2 + len(attempts), last_day=last_day
        )
        if stopped_action == Action.OFFER:
            # R5: the denominator false_offramp_count never had. Counted
            # OUTSIDE the would_pay branch, so an OFFER to a mandate that
            # would not have paid lands somewhere instead of nowhere.
            cell.offramp_scored_count += 1
        if would_pay:
            cell.missed_recovery_count += 1
            cell.missed_recovery_paise += m.amount_paise
            if stopped_action == Action.OFFER:
                cell.false_offramp_count += 1
                cell.false_offramp_paise += m.amount_paise
        elif stopped_action == Action.OFFER:
            cell.true_offramp_count += 1
            cell.true_offramp_paise += m.amount_paise

    return attempts


def _result_for(m, attempts) -> MandateResult:
    """score_mandate() raises on a zero-attempt mandate -- reasonably, since
    the ladder can never produce one. This engine can: REAUTH or OFFER at the
    first decision point spends no slot at all, which is the entire point of
    having those actions. Such a mandate is STILL_PENDING and PRESERVED under
    protocol.md's own definitions (budget unspent, right-censored, still an
    active mandate next cycle), so it is scored here rather than by editing
    the frozen scorer.
    """
    if attempts:
        return score_mandate(m, list(attempts))
    return MandateResult(
        mandate_id=m.mandate_id,
        attempts=(),
        final_outcome=Outcome.STILL_PENDING,
        amount_recovered_paise=0,
        preserved=True,
        iatrogenic_failures=0,
    )


# --- the conformal gate ------------------------------------------------------


def _calib_group_id(mandate_id: str) -> str:
    """Simulator mandate ids (M0000...) repeat across seeds, so a bare id
    cannot prove the calibration and report sets are disjoint by inspection.
    Namespacing by the calibration seed gives a disjointness check (if one
    were run against this stream) something real to compare -- the same
    convention eval/corpus.py already uses. No such check is called on this
    path today (see CALIB_SEED's own comment); disjointness holds by
    construction, not by assertion."""
    return f"calib{CALIB_SEED}:{mandate_id}"


def make_channel(spec, seed: int, *, offset: int = _CHANNEL_OFFSET):
    """One WontPayChannel for one cell, on its own RNG stream.

    `spec` is (kind, tpr, fpr), (kind, tpr, fpr, habitual_fraction), or
    None. The 4-tuple form is additive (R5 review pass, 2026-09-05): every
    EXISTING caller passes the 3-tuple, which defaults habitual_fraction to
    1.0 -- WontPayChannel's own exactly-iid default -- so nothing already
    published changes.

    A fresh instance per cell: the channel accumulates its own
    realised-ROC counters, and sharing one across cells would pool them
    into a number no single cell could be checked against.
    """
    if spec is None:
        return None
    if len(spec) == 4:
        kind, tpr, fpr, habitual_fraction = spec
    else:
        kind, tpr, fpr = spec
        habitual_fraction = 1.0
    return WontPayChannel(kind=kind, tpr=tpr, fpr=fpr,
                          habitual_fraction=habitual_fraction,
                          rng=random.Random(seed + offset))


def fit_gate(base_cfg: dict, *, alpha: float = 0.05, channel_spec=None):
    """Calibrate the off-ramp gate ONCE, on the baseline regime, from its own
    simulator draw. Returns (gate, kind, diagnostics).

    The predictor is over CAUSES, not terminal Outcomes: the off-ramp asks
    why the mandate is failing, and allocator.py fires only on the singleton
    {WONT_PAY}. Scores are LAC over the belief the system actually holds
    after the slot-1 decline -- i.e. the gate is calibrated on exactly the
    object it will be asked about in production, not on a proxy.

    Falls back to FullSetGate (never offers) if calibration is underpowered,
    and says so. That is the safe direction and B8's documented default.
    """
    sim = Simulator("nominal", seed=CALIB_SEED, config=base_cfg)
    rng = random.Random(CALIB_SEED + _CALIB_SLOT1_OFFSET)
    # R5: RE-CALIBRATION IS MANDATORY, not optional. The channel changes the
    # belief distribution, so it changes this calibration pool -- calibrating
    # on the pre-R5 pool and querying a post-R5 belief would be exactly the
    # exchangeability break split conformal's coverage guarantee rests on.
    # Its own stream (_CALIB_CHANNEL_OFFSET), disjoint from every reported
    # seed's, for the same reason the slot-1 stream already is.
    channel = make_channel(channel_spec, CALIB_SEED, offset=_CALIB_CHANNEL_OFFSET)
    scores, y, ids = [], [], []
    for m in sim.mandates:
        if channel is not None:
            channel.for_mandate(m.mandate_id)
        b = initial_belief(m.initial_cause, base_cfg, rng, channel=channel)
        scores.append(list(b.probs))
        y.append(CAUSE_ORDER.index(m.initial_cause))
        ids.append(_calib_group_id(m.mandate_id))

    score_rows = conformal.lac_scores(np.asarray(scores, dtype=float))
    try:
        predictor = conformal.calibrate(
            scores=score_rows,
            y=np.asarray(y, dtype=int),
            labels=CAUSE_ORDER,
            row_group_ids=ids,
            provenance="calib_conf",
            alpha=alpha,
        )
    except conformal.ConformalUnderpowered as exc:
        return FullSetGate(), "full_set", {"reason": f"underpowered: {exc}"}

    diag = {"alpha": alpha, "n_calib": len(y), "calib_seed": CALIB_SEED}
    if channel is not None:
        # The Mondrian floor is ceil(1/alpha) - 1 per class (19 at
        # alpha=0.05); calibrate() raises ConformalUnderpowered above if any
        # class is below it, so reaching here proves the floor holds. The
        # per-class counts are recorded anyway: "it did not raise" is a
        # weaker artifact than the numbers themselves, and R5 has to
        # re-report calibration after changing the belief distribution.
        diag["channel"] = channel.describe()
        diag["channel_realised"] = channel.realised()
        diag["calib_per_class"] = {
            c.value: int(sum(1 for yy in y if CAUSE_ORDER[yy] == c)) for c in CAUSE_ORDER
        }
        diag["mondrian_floor"] = -(-1 // alpha) - 1 if alpha else None
    return (ConformalCauseGate(predictor), "conformal", diag)


def _score_recorded_queries(recorder: "_RecordingGate", truth: dict[str, Cause],
                            cell: CellResult) -> None:
    """Empirical behaviour of the live gate over the queries it ACTUALLY
    received in this cell -- every (mandate, slot) decision point, not just
    slot 1.

    Uses SimMandate.initial_cause, privileged ground truth the policy itself
    must never read (simulator.py's own warning). Read here to SCORE, not to
    decide -- the same read eval/gate_criteria.py already makes.

    Per-class coverage is reported alongside the marginal because Mondrian
    conformal's entire purpose is class-conditional coverage, and a marginal
    number can sit at target while one class is badly under-covered.

    R2, 2026-09-04 (payments-domain review): filters to LIVE queries only
    (see _RecordingGate's docstring for why a retrospective, observe_
    terminal()-built belief is not exchangeable with calib_conf and would
    contaminate this measurement). cell.coverage_n_retrospective records
    the excluded count so the exclusion itself is visible, never silent.
    """
    all_qs = recorder.queries
    qs = [(mid, s) for mid, s, retro in all_qs if not retro]
    cell.coverage_n_retrospective = sum(1 for _, _, retro in all_qs if retro)
    cell.coverage_n = len(qs)
    if not qs:
        return
    sizes = [len(s) for _, s in qs]
    covered = sum(1 for mid, s in qs if truth.get(mid) in s)
    cell.coverage_marginal = covered / len(qs)
    cell.mean_set_size = sum(sizes) / len(qs)
    cell.singleton_wont_pay_rate = sum(
        1 for _, s in qs if s == frozenset({Cause.WONT_PAY})
    ) / len(qs)
    cell.singleton_rate = sum(1 for s in sizes if s == 1) / len(qs)
    per_class: dict[str, float] = {}
    for c in CAUSE_ORDER:
        rows = [(mid, s) for mid, s in qs if truth.get(mid) == c]
        if rows:
            per_class[c.value] = sum(1 for _, s in rows if c in s) / len(rows)
    cell.coverage_per_class = per_class


# --- cells -------------------------------------------------------------------


def _fill_bars(cell: CellResult, batch, mandates=()) -> None:
    cell.billable_paise = sum(m.amount_paise for m in mandates)
    cell.n_mandates = batch.n_mandates
    cell.recovered_paise = batch.total_recovered_paise
    cell.attempts_spent = batch.total_attempts_spent
    cell.mandates_preserved = batch.mandates_preserved
    cell.recovered = batch.mandates_recovered
    cell.dead = batch.mandates_dead
    cell.opted_out = batch.mandates_opted_out
    cell.censored = batch.mandates_censored
    cell.iatrogenic_failures = batch.total_iatrogenic_failures


def run_ladder_cell(regime: str, arm: str, profile: Profile, cfg: dict,
                    seed: int) -> CellResult:
    from eval import baseline_ladder

    t0 = time.perf_counter()
    cell = CellResult(regime=regime, arm=arm, profile=profile.value,
                      policy="ladder", seed=seed, gate_kind="n/a")
    sim = Simulator(arm, seed=seed, config=cfg)
    batch = baseline_ladder.run(sim, profile)
    _fill_bars(cell, batch, sim.mandates)
    cell.n_attempt = batch.total_attempts_spent
    cell.n_above_afa = sum(
        1 for m in sim.mandates if m.amount_paise > afa_free_limit_paise(m.category)
    )
    cell.seconds = time.perf_counter() - t0
    return cell


def run_engine_cell(regime: str, arm: str, profile: Profile, cfg: dict, seed: int,
                    hazard, costs: PolicyCosts, gate, gate_kind: str,
                    traces: dict[str, list[DecisionTrace]] | None = None,
                    channel=None) -> CellResult:
    """`channel` is a PREPARED WontPayChannel (see make_channel), not a
    spec -- the caller keeps the instance so it can read the emission log
    back afterwards, which is what eval/offramp_channel.py computes the
    channel's own ROC and its cluster-bootstrap CI from."""
    t0 = time.perf_counter()
    cell = CellResult(regime=regime, arm=arm, profile=profile.value,
                      policy="engine", seed=seed, gate_kind=gate_kind)
    sim = Simulator(arm, seed=seed, config=cfg)
    slot1_rng = random.Random(seed + _SLOT1_OFFSET)
    recorder = _RecordingGate(gate)

    results = []
    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            cell.n_above_afa += 1
        if channel is not None:
            channel.for_mandate(m.mandate_id)
        b0 = initial_belief(m.initial_cause, cfg, slot1_rng, channel=channel)
        trace = [] if traces is not None else None
        attempts = _run_engine_mandate(m, sim, profile, hazard, costs, recorder, b0,
                                       cell, trace=trace, channel=channel)
        if traces is not None:
            traces[m.mandate_id] = trace
        results.append(_result_for(m, attempts))

    if channel is not None:
        cell.channel_kind = channel.kind
        cell.channel_tpr = channel.tpr
        cell.channel_fpr = channel.fpr
        cell.channel_n_wont_pay = channel.n_wont_pay
        cell.channel_positive_on_wont_pay = channel.n_positive_on_wont_pay
        cell.channel_n_other = channel.n_other
        cell.channel_positive_on_other = channel.n_positive_on_other
    _fill_bars(cell, aggregate(results, arm=arm, profile=profile.value), sim.mandates)
    if gate_kind == "conformal":
        _score_recorded_queries(
            recorder, {m.mandate_id: m.initial_cause for m in sim.mandates}, cell
        )
    cell.seconds = time.perf_counter() - t0
    return cell


def run_null_cell(regime: str, arm: str, profile: Profile, cfg: dict, seed: int,
                  policy: str) -> CellResult:
    """Two cause-blind, model-free reference policies. Neither is a candidate
    for anything; both exist to make the engine's headline falsifiable.

    `null`     -- never attempt. Spends nothing, recovers nothing, and
                  PRESERVES EVERY MANDATE, because DEAD and OPTED_OUT are
                  reachable only through attempt(). It is the upper bound of
                  the mandates-preserved bar and it needs no model at all.
    `one_shot` -- exactly one attempt per mandate, on day 2, no belief, no
                  hazard, no gate.

    B5 already measured that a do-nothing policy clears mandates-preserved on
    all three arms with no model (gates.md, the B5 note). B13's first draft
    reported "the engine preserves more in 16 of 16 cells" without carrying
    that column forward, which made the headline unfalsifiable: every metric
    here is monotonically decreasing in attempt count by construction, so
    "preserves more" follows from "attempts less" and says nothing about
    knowing WHY a payment failed. payments-domain measured one_shot beating
    the engine on that bar in 14 of 16 cells. Printing both columns is what
    stops the report overstating the system.
    """
    t0 = time.perf_counter()
    cell = CellResult(regime=regime, arm=arm, profile=profile.value,
                      policy=policy, seed=seed, gate_kind="n/a")
    sim = Simulator(arm, seed=seed, config=cfg)
    results = []
    for m in sim.mandates:
        if m.amount_paise > afa_free_limit_paise(m.category):
            cell.n_above_afa += 1
        if policy == "null":
            results.append(_result_for(m, []))
            continue
        r = sim.attempt(m.mandate_id, slot=2, on_day=2)
        cell.n_attempt += 1
        results.append(_result_for(m, [r]))
    _fill_bars(cell, aggregate(results, arm=arm, profile=profile.value), sim.mandates)
    cell.seconds = time.perf_counter() - t0
    return cell


# --- driver ------------------------------------------------------------------


def run_all(*, regime_names: Sequence[str], arms: Sequence[str],
            profiles: Sequence[Profile], seed: int,
            verbose: bool = True,
            config_path: pathlib.Path | None = None,
            seeds: Sequence[int] | None = None,
            channel_spec=None) -> dict[str, Any]:
    """`seeds` runs the whole grid once per seed and concatenates the cells.

    Everything in B13's first report was a single draw with no error bar,
    which is the weakest possible footing for its central comparison: the
    engine against a model-free one-attempt policy. A gap of a few mandates
    on one seed is not a result. The report aggregates across whatever seeds
    are present and reports the spread, so a claim can be checked against
    its own noise.
    """
    seed_list = list(seeds) if seeds else [seed]
    base_cfg = load_config(config_path)
    costs = load_costs()

    if verbose:
        print("fitting hazard model on the nominal corpus (once, reused everywhere)...",
              file=sys.stderr)
    hazard = hazard_from_fit(fit_nominal_hazard_model())

    if verbose:
        print("calibrating the conformal off-ramp gate on baseline...", file=sys.stderr)
    gate, gate_kind, gate_diag = fit_gate(base_cfg, channel_spec=channel_spec)
    if verbose:
        print(f"  gate: {gate_kind} {gate_diag}", file=sys.stderr)
        if channel_spec is not None:
            print(f"  SYNTHETIC WONT_PAY channel live: {channel_spec} "
                  f"-- see eval/allocator_sweep.py's channel section",
                  file=sys.stderr)

    cells: list[CellResult] = []
    for sd in seed_list:
      if verbose and len(seed_list) > 1:
          print(f"-- seed {sd}", file=sys.stderr)
      for regime in regime_names:
        cfg = regimes_mod.config_for(regime, base_cfg)
        for arm in regimes_mod.arms_for(regime, tuple(arms)):
            for profile in profiles:
                cells.append(run_ladder_cell(regime, arm, profile, cfg, sd))
                cells.append(run_engine_cell(regime, arm, profile, cfg, sd,
                                             hazard, costs, gate, gate_kind,
                                             channel=make_channel(channel_spec, sd)))
                cells.append(run_null_cell(regime, arm, profile, cfg, sd, "null"))
                cells.append(run_null_cell(regime, arm, profile, cfg, sd, "one_shot"))
                if verbose:
                    lad, eng, nul, one = cells[-4], cells[-3], cells[-2], cells[-1]
                    print(
                        f"  {regime:16s} {arm:13s} {profile.value:11s} "
                        f"ladder[{lad.recovered_paise:>9d}/{lad.attempts_spent:>3d}/{lad.mandates_preserved:>3d}] "
                        f"engine[{eng.recovered_paise:>9d}/{eng.attempts_spent:>3d}/{eng.mandates_preserved:>3d}] "
                        f"one_shot[{one.recovered_paise:>9d}/{one.attempts_spent:>3d}/{one.mandates_preserved:>3d}] "
                        f"null[pres={nul.mandates_preserved:>3d}]   (rec/att/pres)",
                        file=sys.stderr,
                    )

    return {
        # schema 3 (R5, 2026-09-05): CellResult gained the off-ramp
        # denominator/true-positive fields and the synthetic-channel ROC
        # fields, and the payload gained `wontpay_channel`. Additive --
        # every schema-2 key still means exactly what it meant.
        "schema": 3,
        "seed": seed_list[0],
        "seeds": seed_list,
        "gate_kind": gate_kind,
        "gate_diagnostics": gate_diag,
        "wontpay_channel": (
            None if channel_spec is None
            else {"kind": channel_spec[0], "tpr": channel_spec[1], "fpr": channel_spec[2]}
        ),
        "arms": list(arms),
        "profiles": [p.value for p in profiles],
        "regimes": {
            name: {
                "story": spec.story,
                "hypothesis": spec.hypothesis,
                "approximation": spec.approximation,
                "overlay": spec.overlay,
            }
            for name, spec in regimes_mod.REGIMES.items()
            if name in regime_names
        },
        "cells": [_serialise_cell(c) for c in cells],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all-regimes", action="store_true",
                    help="every regime in eval/regimes.py (the reported configuration)")
    ap.add_argument("--regime", action="append", dest="regime_names", default=None)
    ap.add_argument("--both-profiles", action="store_true",
                    help="strict and permissive (the reported configuration)")
    ap.add_argument("--arm", action="append", dest="arms", default=None)
    ap.add_argument("--profile", action="append", dest="profile_names", default=None,
                    choices=[p.value for p in ALL_PROFILES])
    ap.add_argument("--config", type=pathlib.Path, default=None,
                    help="the frozen sim config to overlay regimes onto; "
                         "defaults to eval/frozen/sim_config.yaml. Accepted so "
                         "the run command names its own input explicitly.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None, metavar="N",
                    help="run seeds 0..N-1 and report the spread. A single "
                         "seed carries no error bar, which is the weakest "
                         "footing for the engine-vs-one_shot comparison.")
    ap.add_argument("--out", type=pathlib.Path, default=ARTIFACT)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--channel-kind", default=DEFAULT_CHANNEL_KIND,
                    choices=("decline", "intent", "off"),
                    help="R5's SYNTHETIC WONT_PAY evidence channel. It reads "
                         "the simulator's privileged true cause and feeds it "
                         "into the decision path -- see eval/allocator_sweep.py's "
                         "channel section. `off` reproduces the pre-R5 "
                         "configuration, in which the off-ramp can never fire.")
    ap.add_argument("--channel-tpr", type=float, default=None,
                    help="channel sensitivity; defaults to the operating point "
                         "pre-registered in eval/offramp_channel.py")
    ap.add_argument("--channel-fpr", type=float, default=None,
                    help="channel false-positive rate; defaults as above")
    return ap.parse_args(argv)


def channel_spec_from_args(args) -> tuple[str, float, float] | None:
    """(kind, tpr, fpr), or None for `off`. The default rates come from
    eval/offramp_channel.py's PRE-REGISTERED operating point -- imported,
    never restated, so the published grid and the sweep that justified its
    operating point cannot silently disagree."""
    if args.channel_kind == "off":
        return None
    from eval.offramp_channel import OPERATING_POINT

    tpr = OPERATING_POINT[0] if args.channel_tpr is None else args.channel_tpr
    fpr = OPERATING_POINT[1] if args.channel_fpr is None else args.channel_fpr
    return (args.channel_kind, tpr, fpr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    regime_names = (
        list(regimes_mod.REGIMES) if args.all_regimes
        else (args.regime_names or ["baseline"])
    )
    arms = tuple(args.arms or ALL_ARMS)
    if args.both_profiles:
        profiles = ALL_PROFILES
    elif args.profile_names:
        profiles = tuple(Profile(p) for p in args.profile_names)
    else:
        profiles = (Profile.strict,)

    payload = run_all(regime_names=regime_names, arms=arms, profiles=profiles,
                      seed=args.seed, verbose=not args.quiet,
                      config_path=args.config,
                      seeds=list(range(args.seeds)) if args.seeds else None,
                      channel_spec=channel_spec_from_args(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    try:
        shown = args.out.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        shown = args.out
    print(f"wrote {shown} "
          f"({len(payload['cells'])} cells, gate={payload['gate_kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
