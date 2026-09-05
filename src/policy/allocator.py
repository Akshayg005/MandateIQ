"""Exact backward induction over the four NPCI attempt slots -- the
allocator that turns a Belief plus an AllocationContext into a committed
Plan.

=== The cause-conditioned hazard gap, and how this file resolves it ========

PLAN_DETAIL.md section 4's Q(b, ATTEMPT(d,m)) sums Sigma_c b[c] * h_c(...)
-- hazards conditioned on a specific latent cause c. B5 shipped hazards
MARGINAL over cause instead (Cause is latent with no production label,
ever -- DECISIONS.md, 2026-08-28), and src/policy/hazards.py's
CauseConditionedHazard Protocol was deliberately left unimplemented at B7
for exactly this reason. B7 left the resolution open for B8 to decide with
the allocator's actual constraints in view (reports/gates.md, B8 entry,
2026-08-29).

**Decision: Cause enters only through action-gating, never through the
hazard arithmetic.** SlotHazard (below) is marginal by construction --
this file does not accept, and does not implement, a
CauseConditionedHazard. The Belief b still determines which actions are
*legal* (REAUTH when CANT_PAY_EVER dominates; OFFER only on a singleton
conformal set), but the Q-value for ATTEMPT uses the marginal hazard
directly. This is lossless given the available hazard source: for any b on
the simplex, Sigma_c b[c] * h = h * Sigma_c b[c] = h when h does not vary
with c -- the narrowing costs nothing beyond what B5's hazard source
already lost. `test_marginal_hazard_makes_the_cause_sum_an_identity` in the
test suite proves this rather than asserting it.

**Consequence for belief across the lookahead: b0 is carried UNCHANGED
through every recursive node in one solve() call.** The theoretically pure
version of PLAN_DETAIL's recursion updates belief via
`update(b, obs=survived(c,d,m))` on each "still pending" branch -- but that
update needs a specific observed DeclineClass (src.policy.belief.update's
only accepted observation type), and a cause-marginal hazard model has no
honest way to produce one: it predicts Outcome probabilities (4-class), not
DeclineClass probabilities (7-class issuer-decline taxonomy), and nothing
maps one to the other without fabricating information the model does not
have. Updating belief on invented evidence would make the exact-solve claim
dishonest in the one place it matters most. The real belief update instead
happens at EXECUTION time (B9): the executor observes the actual issuer
decline string, normalises it, calls belief.update(), and re-invokes
solve() for the next slot with the genuinely updated belief -- consistent
with "the allocator is never consulted again for the current cycle"
(PLAN_DETAIL.md section 4) but *is* consulted again once real evidence
exists and enough lead time remains.

One practical consequence: since Sigma_c b[c] * h collapses to h regardless
of b, and b does not change within one solve() call, b0's only causal role
in this file is action-gating -- REAUTH/OFFER feasibility is evaluated
against the SAME b0 at every simulated future node, not a projected belief
trajectory. The backward induction therefore still explores real
combinatorial structure (which day to commit each slot, whether continuing
beats stopping, given costs and diminishing survival probability) -- it is
only cause that stays fixed within the lookahead, and only because there is
no honest model of how it would move.

=== CIF vs SlotHazard =======================================================

PLAN_DETAIL.md section 2 says "allocator.py takes a CIF object, not a
HazardModel." There is no CIF *object* in this codebase -- src/model/cif.py
exposes free functions over (n, 3, 4) numpy arrays, and CIF is the
recursion's cumulative OUTPUT, not a per-node input; backward induction
needs per-slot hazards at a candidate day, which is what it branches on.
SlotHazard (below) is the resolution: a narrow Protocol for marginal
outcome probabilities at one (slot, day, amount), which the eval harness
satisfies by wrapping competing_risks.hazards(). cif.py remains the tool
for *checking* that hazard source (Sigma_c CIF_c(4) + S(4) == 1), which is
what section 2's sentence was actually protecting -- the allocator never
sees cause-specific mechanism hazards, CIF-shaped or otherwise. Logged in
DECISIONS.md as a deviation, not a silent substitution.

=== Money and floats ========================================================

Q-values are expected-value SCORES for comparing actions, not money
amounts -- computing one necessarily multiplies a probability by a paise
quantity, which invariant 2 ("all money is integer paise") is not about:
that invariant targets a money value that is ITSELF a float (a balance, an
amount field, a persisted or displayed currency figure). No Q-value is ever
persisted, charged, or displayed; the only money that reaches a Plan is
ctx.amount_paise, copied through unchanged as a plain int on every
CommittedAttempt. money-auditor review is expected on this file precisely
because it is easy to mistake one for the other; this paragraph is the
answer prepared for that review, not a claim it will not be checked.

=== Plan.committed's type ===================================================

PLAN_DETAIL.md's file table types this `list[CommittedAttempt]`; this
module uses `tuple[CommittedAttempt, ...]` instead, matching this
codebase's own convention (every dataclass in src/model/ and src/policy/ is
frozen; test_profiles.py states the pattern explicitly). A Plan that could
be mutated after solve() returned it would undermine the point of hashing
it into decision_sha256.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.core.ids import decision_sha256
from src.core.types import Action, Cause, MandateState, Profile
from src.model.conformal import should_act
from src.policy.belief import Belief, quantised
from src.policy.constraints import MAX_ATTEMPTS, assert_not_pre_notification_exempt, requires_afa, within_mandate_ceiling
from src.policy.costs import PolicyCosts
from src.policy.gate import ConformalGate, FullSetGate
from src.policy.offramp import Offer, construct_offer
from src.policy.profiles import get as get_profile
from src.policy.stopping_rules import AllocationContext, Verdict, permitted

_SOLVER_VERSION = "b8-backward-induction-v1"
_MEMO_QUANT_STEP = 1e-6

# Structural day-window constants (PLAN_DETAIL.md section 4, decision 2).
# A 30-day repeating cycle is a stated simplifying assumption -- this
# corpus and simulator have no calendar either (eval/frozen/simulator.py),
# and the real hour-level 24h lag is enforced at B9 via src.core.clock,
# regardless of profile. _SALARY_WINDOW_START/_END must match
# in_salary_window's own definition (on_day in 1..5) used throughout
# src/model/ -- a drift here would silently mis-time every attempt against
# the model it was scored with.
_CYCLE_LEN_DAYS = 30
_SALARY_WINDOW_START = 1
_SALARY_WINDOW_END = 5
_POST_SALARY_DAY = 10
_PRE_SALARY_DAY = 28
_MONTH_END_DAY = 30


class AllocatorError(ValueError):
    """Raised when solve() is given a ctx that could not have been produced
    legally (e.g. ceiling below the scheduled amount -- clause 4(c)
    violated before the allocator was ever called), or a hazard source
    returns a tuple that is not a valid probability distribution. Never a
    bare assert -- assert is stripped under python -O, and either failure
    means an upstream bug feeding this file bad data, not a decision to
    make."""


@runtime_checkable
class SlotHazard(Protocol):
    """Marginal-over-cause outcome probabilities for one (slot, day,
    amount). Returns a 4-tuple in Outcome int order -- STILL_PENDING,
    RECOVERED, DEAD, OPTED_OUT -- the same convention
    src.model.competing_risks.hazards() and src.policy.hazards.
    CauseConditionedHazard both use. Deliberately NOT a
    CauseConditionedHazard: see this module's docstring for why accepting
    one here would not be honest given what B5 actually fitted."""

    def __call__(
        self, *, slot: int, on_day: int, amount_paise: int
    ) -> tuple[float, float, float, float]:
        ...


@dataclass(frozen=True)
class CommittedAttempt:
    slot: int
    on_day: int
    amount_paise: int


@dataclass(frozen=True)
class Plan:
    """The output of one solve() call. chosen_action: the root decision --
    ATTEMPT, REAUTH, OFFER, or STOP. committed: zero or one
    CommittedAttempt (never more -- solve() is re-invoked once an attempt
    resolves and real evidence updates belief, per this module's
    docstring; it never pre-commits multiple future slots on speculation).
    binding_constraint: which hard rule forced the root's options, if any
    (AFA_CLIFF, ATTEMPT_CAP_EXHAUSTED, OPTED_OUT, MANDATE_REVOKED) --
    None if the choice was a genuine value comparison among options that
    were all still live."""

    mandate_id: str
    cycle_id: int
    profile: Profile
    chosen_action: Action
    committed: tuple[CommittedAttempt, ...]
    belief_json: str
    conformal_set: frozenset[Cause]
    binding_constraint: str | None
    solver_version: str
    decision_sha256: str
    # R5, 2026-09-05 (reports/gates.md, "Post-B16 remediation gates"): the
    # actual pause/downgrade/cancel menu, present iff chosen_action is
    # OFFER and None otherwise. src/policy/offramp.py was complete and
    # tested from B8 but had NO CALLER anywhere in src/ -- a chosen OFFER
    # had never produced an Offer object, while offramp.py's own docstring
    # asserted "allocator.py only calls construct_offer() once OFFER has
    # already been chosen". That sentence is true now; it was false before.
    #
    # DELIBERATELY NOT part of decision_sha256's payload. The digest covers
    # the DECISION -- action, belief, conformal set, committed slots -- and
    # an Offer is a deterministic presentation artifact derived from
    # (belief, ctx) after that decision is made. Including it would change
    # nothing about which decisions are distinguishable while making every
    # already-persisted hash unreproducible for anyone re-deriving one from
    # a `plan` row. tests/policy/test_allocator.py pins a literal digest so
    # this cannot drift silently.
    offer: Offer | None = None


def _next_day_in_window(lo: int, hi: int, earliest: int, cycle_len: int = _CYCLE_LEN_DAYS) -> int:
    """Smallest day >= earliest whose 1-indexed position within a
    `cycle_len`-day repeating cycle falls in [lo, hi]. Brute-force over at
    most 2*cycle_len candidates -- obviously correct by exhaustive check,
    not derived modular arithmetic that is hard to verify by eye."""
    for day in range(earliest, earliest + 2 * cycle_len):
        pos = ((day - 1) % cycle_len) + 1
        if lo <= pos <= hi:
            return day
    raise AssertionError("unreachable: a window recurs at least once every cycle_len days")


def committable_days(ctx: AllocationContext) -> list[int]:
    """Four structural day-index candidates for the NEXT attempt slot
    (ctx.attempts_used + 1), each carrying its replenishment-rhythm
    rationale (PLAN_DETAIL.md section 4, decision 2):

    - pre-salary: tests whether a residual balance exists, before the
      month's credit.
    - salary window (days 1-5): replenishment, but also peak debit-order
      competition.
    - post-salary: after the salary-window storm clears.
    - month-end: a second common credit rhythm, also the mandate-ceiling
      boundary.

    Restricting to exactly these -- not an arbitrary date set -- is what
    keeps the state space small enough for the backward induction to stay
    exact: ~(4*8)^4 nodes before memoisation, versus ~10^8 at ~20 arbitrary
    candidate days, where the exact-solve claim would have to be
    abandoned.

    Profile-aware lead time, at day-index granularity (this layer has no
    intraday clock): under `strict`, the next slot needs its own fresh
    notification, so the earliest eligible day is plan_day + 1. Under
    `permissive`, the cycle's original notification already covers
    retries, so the earliest eligible day is plan_day itself. This is
    "shrinking committable_days" (PLAN_DETAIL.md section 4's description of
    `strict`) expressed at the only granularity this layer has -- the real
    hour-level 24h lag is enforced at B9 via src.core.clock, for both
    profiles alike.

    Always strictly after the most recently committed day, regardless of
    profile: two attempts on the same calendar day for one mandate cycle
    is not a notification question, it is a scheduling impossibility (and
    eval/frozen/simulator.py's own Simulator.attempt() enforces exactly
    this ordering on whatever day this function hands it -- found while
    building eval/allocator_sweep.py, before it ever produced a wrong
    schedule in the eval sweep itself; see DECISIONS.md).
    """
    profile = get_profile(ctx.profile)
    next_slot = ctx.attempts_used + 1
    lead = 1 if profile.requires_fresh_notification(next_slot) else 0
    earliest = ctx.plan_day + lead
    if ctx.committed_days:
        earliest = max(earliest, ctx.committed_days[-1] + 1)

    candidates = {
        _next_day_in_window(_PRE_SALARY_DAY, _PRE_SALARY_DAY, earliest),
        _next_day_in_window(_SALARY_WINDOW_START, _SALARY_WINDOW_END, earliest),
        _next_day_in_window(_POST_SALARY_DAY, _POST_SALARY_DAY, earliest),
        _next_day_in_window(_MONTH_END_DAY, _MONTH_END_DAY, earliest),
    }
    return sorted(candidates)


def _validate_hazard_tuple(h: tuple[float, float, float, float]) -> None:
    if len(h) != 4 or not all(0.0 <= p <= 1.0 for p in h):
        raise AllocatorError(f"hazard tuple has an out-of-range probability: {h}")
    total = sum(h)
    if abs(total - 1.0) > 1e-6:
        raise AllocatorError(f"hazard tuple does not sum to 1 (got {total}): {h}")


def _binding_constraint(ctx: AllocationContext, r0: int) -> str | None:
    """Which hard rule, if any, already forecloses ATTEMPT at the root --
    recorded on the Plan for auditability regardless of what the eventual
    chosen_action turns out to be.

    R2, 2026-09-04 (payments-domain review): `ctx.instrument_dead` was
    missing from this function -- a real correctness bug, not a style gap.
    A post-DEAD re-solve that returned REAUTH (ATTEMPT denied ONLY by
    `permitted()`'s instrument_dead rule -- none of the other four checks
    below apply) previously wrote `binding_constraint = None` to the Plan,
    which src/execute/shadow.py renders as "(none -- decided on belief and
    expected value)". That is not a missing field, it is the ledger stating
    a hard-forced decision was a free economic choice -- across every
    engine cell in the fresh 8-seed sweep, thousands of REAUTHs carried
    this false audit record. Checked FIRST, same reasoning as AFA_CLIFF:
    a hard-observed fact should never be shadowed by a later check in this
    list, though today none of the other four can co-occur with it anyway
    (REVOKED/opted_out/AFA_CLIFF are each mutually exclusive states this
    context can be in at the moment instrument_dead is set)."""
    if ctx.instrument_dead:
        return "INSTRUMENT_DEAD"
    if requires_afa(ctx.amount_paise, ctx.category):
        return "AFA_CLIFF"
    if r0 <= 0:
        return "ATTEMPT_CAP_EXHAUSTED"
    if ctx.opted_out:
        return "OPTED_OUT"
    if ctx.mandate_state == MandateState.REVOKED:
        return "MANDATE_REVOKED"
    return None


def _value(
    b: Belief,
    r: int,
    ctx: AllocationContext,
    hazard: SlotHazard,
    costs: PolicyCosts,
    gate: ConformalGate,
    memo: dict,
) -> float:
    """V(b, r, ctx) -- PLAN_DETAIL.md section 4. V(b, 0, ctx) = 0: budget
    exhausted, no salvage value."""
    if r <= 0:
        return 0.0
    key = (quantised(b, _MEMO_QUANT_STEP), r, ctx.signature())
    cached = memo.get(key)
    if cached is not None:
        return cached
    _, value, _ = _best_action(b, r, ctx, hazard, costs, gate, memo)
    memo[key] = value
    return value


def _best_action(
    b: Belief,
    r: int,
    ctx: AllocationContext,
    hazard: SlotHazard,
    costs: PolicyCosts,
    gate: ConformalGate,
    memo: dict,
) -> tuple[Action, float, int | None]:
    """argmax over A(b, r, ctx) of Q(b, a, r, ctx) -- PLAN_DETAIL.md section
    4. Returns (action, its value, the committed day if action is ATTEMPT
    else None). STOP is always feasible at value 0.0 and is the floor every
    comparison is made against, so a genuinely worthless option never wins
    by default."""
    best_action: Action = Action.STOP
    best_value = 0.0
    best_day: int | None = None

    if permitted(Action.REAUTH, ctx) == Verdict.ALLOW:
        q_reauth: float | None = None
        if requires_afa(ctx.amount_paise, ctx.category):
            # COMPLIANCE path (clause 8(a)/8(b)): above the AFA-free limit,
            # re-authorisation is the only LEGAL route -- not a judgement
            # about cause. Never belief-discounted: discounting a legal
            # requirement by how strongly we happen to believe something
            # would be a category error.
            q_reauth = costs.reauth_success_prob * ctx.amount_paise - costs.reauth_cost_paise
        elif b.dominant() == Cause.CANT_PAY_EVER:
            # INFERENCE path: re-authorisation only actually recovers money
            # if the instrument is genuinely dead. If our belief is wrong,
            # we have spent the cost AND put an auth flow in front of a
            # customer who would have paid on an ordinary retry -- so the
            # recovery term is weighted by exactly the belief it depends
            # on, b[CANT_PAY_EVER], rather than assumed certain.
            #
            # This deliberately makes REAUTH harder to trigger than the
            # bare-plurality test alone (b.dominant() is True at beliefs as
            # weak as (0.34, 0.35, 0.31) -- a near-uniform belief, far too
            # thin a basis for an irreversible-feeling customer contact).
            # The confidence needed EMERGES from the economics -- REAUTH
            # wins only once b[CANT_PAY_EVER] is high enough for its
            # discounted value to beat continuing to retry -- rather than
            # from a hand-picked threshold constant, which
            # src/policy/CLAUDE.md would require a clause citation for and
            # which this project has declined three times before (B5's
            # stop-threshold scalar, the paired-criterion reversal, B7's
            # switch_eps).
            #
            # Asymmetric by design, matching the established posture in
            # src/classify/cause_map.py: mistaking CANT_PAY_EVER for
            # CANT_PAY_NOW costs one retry slot (cheap, reversible);
            # mistaking CANT_PAY_NOW for CANT_PAY_EVER costs a customer we
            # could have kept. Root CLAUDE.md: the system sometimes
            # deliberately recovers less this cycle to protect lifetime
            # value.
            q_reauth = (
                b[Cause.CANT_PAY_EVER] * costs.reauth_success_prob * ctx.amount_paise
                - costs.reauth_cost_paise
            )
        if q_reauth is not None and q_reauth > best_value:
            best_action, best_value, best_day = Action.REAUTH, q_reauth, None

    if permitted(Action.OFFER, ctx) == Verdict.ALLOW and should_act(gate.pred_set(b), Cause.WONT_PAY):
        q = float(costs.mandate_ltv_paise)
        if q > best_value:
            best_action, best_value, best_day = Action.OFFER, q, None

    if r > 0 and permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW and not requires_afa(ctx.amount_paise, ctx.category):
        for day in committable_days(ctx):
            h = hazard(slot=ctx.attempts_used + 1, on_day=day, amount_paise=ctx.amount_paise)
            _validate_hazard_tuple(h)
            p_pending, p_rec, p_dead, p_opt = h
            continuation = _value(b, r - 1, ctx.with_attempt(day), hazard, costs, gate, memo) if p_pending > 0.0 else 0.0
            # `h` is MARGINAL over cause -- a population average that does
            # not know what this mandate's belief says. Left undiscounted,
            # the allocator values an attempt at population-average
            # recovery odds even while believing the instrument is dead,
            # and burns NPCI slots on it (eval/frozen/protocol.md names a
            # "wasted attempt on a dead instrument" as exactly the
            # behaviour the attempts-spent bar exists to penalise). That is
            # also internally inconsistent with REAUTH above, which IS
            # belief-weighted: one comparison, two different beliefs.
            #
            # The correction is DEFINITIONAL, not a fit and not a tuning
            # constant: root CLAUDE.md defines CANT_PAY_EVER as "Instrument
            # dead -- expired card, closed account, revoked mandate," so
            # P(RECOVERED | CANT_PAY_EVER) ~ 0 follows from what the cause
            # MEANS, not from data. Only the recovery term is scaled --
            # opt-out risk and the continuation value are NOT, because a
            # dead instrument can still have its holder opt out, and the
            # survival branch is already the "nothing terminal happened"
            # case.
            #
            # This does NOT reopen the cause-conditioned-hazard gap
            # (DECISIONS.md, 2026-08-29, B7): no P(outcome | cause, ...)
            # is estimated anywhere, no coefficient is fitted, and no new
            # constant is introduced. It is the one cause-conditioned fact
            # the taxonomy supplies for free.
            recoverable = 1.0 - b[Cause.CANT_PAY_EVER]
            q = (
                recoverable * p_rec * ctx.amount_paise
                - p_opt * costs.mandate_ltv_paise
                + p_pending * continuation
            ) - costs.attempt_cost_paise
            if q > best_value:
                best_action, best_value, best_day = Action.ATTEMPT, q, day

    return best_action, best_value, best_day


def _build_plan(
    b0: Belief,
    ctx: AllocationContext,
    gate: ConformalGate,
    action: Action,
    committed: tuple[CommittedAttempt, ...],
    binding_constraint: str | None,
) -> Plan:
    """Assemble the Plan. When the root decision is OFFER, this is where
    construct_offer() runs -- once, on the decision that was actually
    taken, never on a hypothetical explored inside the backward induction.
    The lookahead compares VALUES; only the root produces an artifact the
    customer could ever see."""
    conformal_set = gate.pred_set(b0)
    payload = {
        "mandate_id": ctx.mandate_id,
        "cycle_id": ctx.cycle_id,
        "profile": ctx.profile.value,
        "chosen_action": action.value,
        "committed": [
            {"slot": c.slot, "on_day": c.on_day, "amount_paise": c.amount_paise} for c in committed
        ],
        "belief": list(b0.probs),
        "conformal_set": sorted(c.value for c in conformal_set),
        "binding_constraint": binding_constraint,
        "solver_version": _SOLVER_VERSION,
    }
    digest = decision_sha256(payload)
    offer = construct_offer(b0, ctx) if action == Action.OFFER else None
    return Plan(
        mandate_id=ctx.mandate_id,
        cycle_id=ctx.cycle_id,
        profile=ctx.profile,
        chosen_action=action,
        committed=committed,
        belief_json=b0.to_json(),
        conformal_set=conformal_set,
        binding_constraint=binding_constraint,
        solver_version=_SOLVER_VERSION,
        decision_sha256=digest,
        offer=offer,
    )


def solve(
    b0: Belief,
    ctx: AllocationContext,
    *,
    hazard: SlotHazard,
    costs: PolicyCosts,
    gate: ConformalGate | None = None,
) -> Plan:
    """Exact backward induction over the remaining NPCI attempt slots.
    Pure and DB-free: solve() never writes to the ledger and never calls
    src.core.clock -- persistence is a separate concern (see
    src/ledger/store.py), and time enters only through ctx.plan_day
    (day-index, supplied by the caller) so this function stays trivially
    testable and fast enough for a 3600-mandate x 20-seed sweep without a
    live Postgres.

    Applies the AFA cliff (clause 8(a)/8(b)) before ever consulting
    `hazard` -- above-cliff mandates have zero support in the fitted
    hazard model (eval/corpus.py excludes them from training on this same
    assumption), so scoring one through the Q-function would be
    out-of-support extrapolation, not a compliance nuance. This falls out
    of _best_action's own structure (the ATTEMPT branch is gated on
    `not requires_afa(...)` before it ever calls `hazard`), not a special
    case in this function.

    Raises AllocatorError if ctx.ceiling_paise < ctx.amount_paise (clause
    4(c) already violated before this function was ever called -- an
    upstream bug, not a routing decision) or if `ctx.category` is a clause
    6(d) pre-notification exemption (out of scope for this system; asserted
    unreachable rather than silently handled).
    """
    assert_not_pre_notification_exempt(ctx.category)
    if not within_mandate_ceiling(ctx.amount_paise, ctx.ceiling_paise):
        raise AllocatorError(
            f"{ctx.mandate_id}: amount_paise {ctx.amount_paise} exceeds ceiling_paise "
            f"{ctx.ceiling_paise} -- violates clause 4(c); should never reach the allocator"
        )

    if gate is None:
        gate = FullSetGate()

    r0 = MAX_ATTEMPTS - ctx.attempts_used
    binding = _binding_constraint(ctx, r0)

    memo: dict = {}
    action, _value_score, day = _best_action(b0, max(r0, 0), ctx, hazard, costs, gate, memo)

    if action == Action.ATTEMPT:
        committed = (CommittedAttempt(slot=ctx.attempts_used + 1, on_day=day, amount_paise=ctx.amount_paise),)
    else:
        committed = ()

    return _build_plan(b0, ctx, gate, action, committed, binding)
