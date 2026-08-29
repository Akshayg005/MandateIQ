"""src/policy/allocator.py -- exact backward induction over the NPCI attempt
slots.

Design spec: solve(b0, ctx, hazard=..., costs=...) applies the AFA cliff
before ever consulting `hazard`; the chosen root action and (if ATTEMPT) its
committed day come from an exact max over Q(b, a, r, ctx); STOP is the
value-0.0 floor every option is compared against. Belief is carried
UNCHANGED across the "still pending" continuation (see allocator.py's own
module docstring for why) -- cause enters only through which actions are
feasible (REAUTH when CANT_PAY_EVER dominates; OFFER only on a singleton
conformal set), never through the Q-value arithmetic.

The gate-required 2-slot brute-force equivalence test lives here:
`_bf_value`/`_bf_solve` are an independent, unmemoized reimplementation of
the same Bellman recursion, written from scratch in this file rather than
calling allocator.py's private `_value`/`_best_action` -- so a bug in
either implementation's arithmetic, or in the memo's cache-key logic,
shows up as a disagreement rather than hiding behind a shared bug.
"""
from __future__ import annotations

import pytest

from src.core.types import Action, Cause, MandateState, Profile
from src.model.conformal import should_act
from src.policy import belief as belief_mod
from src.policy.allocator import (
    AllocatorError,
    CommittedAttempt,
    committable_days,
    solve,
    _best_action,
    _value,
)
from src.policy.constraints import MAX_ATTEMPTS, requires_afa
from src.policy.costs import PolicyCosts
from src.policy.gate import ConformalGate, FullSetGate
from src.policy.stopping_rules import AllocationContext, Verdict, permitted

_COSTS = PolicyCosts(
    attempt_cost_paise=50,
    mandate_ltv_paise=180_000,
    reauth_cost_paise=200,
    reauth_success_prob=0.35,
    quiet_hours_start=21,
    quiet_hours_end=8,
    max_contacts_per_cycle=4,
)


def _ctx(**overrides) -> AllocationContext:
    base = dict(
        mandate_id="M-1",
        cycle_id=1,
        profile=Profile.strict,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        plan_day=1,
        attempts_used=1,
        committed_days=(1,),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=4,
        quiet_hours_start=21,
        quiet_hours_end=8,
    )
    base.update(overrides)
    return AllocationContext(**base)


def _belief(**probs) -> belief_mod.Belief:
    filled = {c: probs.get(c.value, probs.get(c, 0.0)) for c in Cause}
    total = sum(filled.values())
    return belief_mod.init({c: p / total for c, p in filled.items()})


def _uniform_belief() -> belief_mod.Belief:
    return belief_mod.init({c: 1.0 / 3.0 for c in Cause})


def _flat_hazard(p_pending, p_rec, p_dead, p_opt):
    def h(*, slot: int, on_day: int, amount_paise: int) -> tuple[float, float, float, float]:
        return (p_pending, p_rec, p_dead, p_opt)
    return h


# === SlotHazard protocol conformance ========================================

def test_a_conforming_callable_satisfies_slot_hazard():
    from src.policy.allocator import SlotHazard

    def stub(*, slot: int, on_day: int, amount_paise: int) -> tuple[float, float, float, float]:
        return (0.5, 0.3, 0.1, 0.1)

    assert isinstance(stub, SlotHazard)


# === committable_days =======================================================

def test_committable_days_returns_four_structural_candidates():
    days = committable_days(_ctx(profile=Profile.strict, plan_day=0, attempts_used=1))
    assert len(days) == 4, f"expected 4 structural candidates, got {days}"


def test_committable_days_are_strictly_after_the_lead_time_under_strict():
    ctx = _ctx(profile=Profile.strict, plan_day=5, attempts_used=1)
    days = committable_days(ctx)
    assert all(d >= 6 for d in days), f"strict must require >= plan_day+1, got {days}"


def test_strict_and_permissive_committable_days_differ():
    """The 'shrinking committable_days' PLAN_DETAIL.md describes for strict
    must be an observable difference, not a vacuous one -- both profiles
    must not silently produce the same set."""
    strict_days = committable_days(_ctx(profile=Profile.strict, plan_day=4, attempts_used=1))
    permissive_days = committable_days(_ctx(profile=Profile.permissive, plan_day=4, attempts_used=1))
    assert strict_days != permissive_days, \
        f"strict {strict_days} and permissive {permissive_days} committable_days must differ"
    assert min(permissive_days) < min(strict_days), \
        "permissive should reach an eligible day at least as early as strict, and here strictly earlier"


def test_committable_days_never_repeats_an_already_committed_day():
    """Regression: under `permissive` (no fresh-notification lead
    required), the earliest eligible day used to equal plan_day exactly --
    which, if plan_day is itself the day just committed, reproduced that
    same day as a candidate. eval/frozen/simulator.py's Simulator.attempt()
    enforces strictly-increasing on_day per mandate and would raise on a
    repeat. Found building eval/allocator_sweep.py; see DECISIONS.md."""
    ctx = _ctx(profile=Profile.permissive, plan_day=1, committed_days=(1,), attempts_used=1)
    days = committable_days(ctx)
    assert 1 not in days, f"day 1 was already committed but reappears in {days}"
    assert all(d > 1 for d in days)


def test_committable_days_recur_within_a_reasonable_horizon():
    """Every structural window must recur within one cycle length of the
    earliest eligible day -- a candidate should never require waiting an
    unreasonable number of days."""
    for plan_day in (0, 10, 20, 29, 45, 100):
        days = committable_days(_ctx(profile=Profile.strict, plan_day=plan_day, attempts_used=1))
        assert all(d - plan_day <= 60 for d in days), f"plan_day={plan_day} produced {days}"
        assert all(d >= plan_day + 1 for d in days)


# === AFA cliff: hazard must never be consulted ==============================

def test_afa_cliff_routes_to_reauth_without_calling_hazard():
    def exploding(*, slot, on_day, amount_paise):
        raise AssertionError("hazard was called on an above-cliff mandate")

    ctx = _ctx(amount_paise=2_000_000, ceiling_paise=3_000_000, category="subscription")
    assert requires_afa(ctx.amount_paise, ctx.category)
    plan = solve(_uniform_belief(), ctx, hazard=exploding, costs=_COSTS)
    assert plan.chosen_action == Action.REAUTH
    assert plan.binding_constraint == "AFA_CLIFF"
    assert plan.committed == ()


def test_elevated_category_afa_cliff_also_never_calls_hazard():
    def exploding(*, slot, on_day, amount_paise):
        raise AssertionError("hazard was called on an above-cliff mandate")

    # Above the base 8(a) limit but within 8(b)'s elevated limit for an
    # elevated category -- must NOT be treated as above-cliff.
    ctx = _ctx(amount_paise=5_000_000, ceiling_paise=6_000_000, category="insurance_premium")
    assert not requires_afa(ctx.amount_paise, ctx.category)
    plan = solve(_uniform_belief(), ctx, hazard=_flat_hazard(0.5, 0.3, 0.1, 0.1), costs=_COSTS)
    # Should not raise, and should not be forced to AFA_CLIFF.
    assert plan.binding_constraint != "AFA_CLIFF"


# === hard data errors ========================================================

def test_ceiling_below_amount_raises():
    with pytest.raises(AllocatorError):
        solve(
            _uniform_belief(),
            _ctx(amount_paise=200_000, ceiling_paise=100_000),
            hazard=_flat_hazard(0.5, 0.3, 0.1, 0.1),
            costs=_COSTS,
        )


def test_pre_notification_exempt_category_raises():
    with pytest.raises(ValueError):
        solve(
            _uniform_belief(),
            _ctx(category="fastag"),
            hazard=_flat_hazard(0.5, 0.3, 0.1, 0.1),
            costs=_COSTS,
        )


def test_invalid_hazard_tuple_raises():
    with pytest.raises(AllocatorError):
        solve(
            _uniform_belief(),
            _ctx(),
            hazard=_flat_hazard(0.5, 0.5, 0.5, 0.5),  # sums to 2.0
            costs=_COSTS,
        )


# === economically sensible decisions ========================================

def test_dead_belief_with_poor_hazard_routes_to_reauth():
    b = _belief(CANT_PAY_EVER=0.9, CANT_PAY_NOW=0.05, WONT_PAY=0.05)
    hazard = _flat_hazard(0.05, 0.02, 0.90, 0.03)  # near-certain dead
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action == Action.REAUTH


def test_attempt_value_is_discounted_by_belief_the_instrument_is_dead():
    """The symmetric counterpart of REAUTH's belief weighting: a marginal
    hazard is a population average that does not know this mandate's
    belief. Left undiscounted, the allocator would value an attempt at
    population-average recovery odds while believing the instrument is
    dead, and burn NPCI slots on it. Scaling ONLY the recovery term by
    (1 - b[CANT_PAY_EVER]) follows from root CLAUDE.md's own definition of
    CANT_PAY_EVER ("instrument dead"), not from any fit.

    Checked as a strict ordering rather than a pinned constant: an
    otherwise-identical belief with more mass on CANT_PAY_EVER must
    produce a strictly lower ATTEMPT value.

    BOTH beliefs deliberately keep CANT_PAY_NOW dominant, so REAUTH stays
    infeasible in each and `_best_action` is comparing ATTEMPT against
    ATTEMPT. Comparing a CANT_PAY_EVER-dominant belief here instead would
    silently compare REAUTH's value against ATTEMPT's and prove nothing
    about the discount."""
    hazard = _flat_hazard(0.4, 0.45, 0.1, 0.05)
    ctx = _ctx()
    gate = FullSetGate()

    b_low = _belief(CANT_PAY_NOW=0.8, CANT_PAY_EVER=0.1, WONT_PAY=0.1)
    b_high = _belief(CANT_PAY_NOW=0.5, CANT_PAY_EVER=0.4, WONT_PAY=0.1)
    assert b_low.dominant() == Cause.CANT_PAY_NOW
    assert b_high.dominant() == Cause.CANT_PAY_NOW, "test premise: REAUTH must stay infeasible in both"

    a_low, v_low, _ = _best_action(b_low, 1, ctx, hazard, _COSTS, gate, {})
    a_high, v_high, _ = _best_action(b_high, 1, ctx, hazard, _COSTS, gate, {})
    assert a_low == Action.ATTEMPT and a_high == Action.ATTEMPT, "test premise: both compare ATTEMPT"

    assert v_high < v_low, \
        f"more CANT_PAY_EVER mass must lower an attempt's value ({v_high} !< {v_low})"


def test_dead_belief_prefers_reauth_over_burning_a_retry_slot():
    """The behavioural consequence, end to end: at a belief consistent
    with a slot-1 CARD_EXPIRED observation (cause_map's own 0.75 on
    CANT_PAY_EVER), the allocator must re-authorise rather than spend a
    retry on an instrument it believes is dead -- even when the marginal
    hazard is optimistic, which it is, because it averages over a
    population that is mostly recoverable."""
    b = _belief(CANT_PAY_NOW=0.15, CANT_PAY_EVER=0.75, WONT_PAY=0.10)
    hazard = _flat_hazard(0.4568, 0.3017, 0.124, 0.1175)  # the real fitted slot-2 values, to 4dp
    plan = solve(b, _ctx(amount_paise=888_980, ceiling_paise=1_200_000), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action == Action.REAUTH


def test_weak_plurality_cant_pay_ever_belief_does_not_trigger_reauth():
    """Safety property: b.dominant() == CANT_PAY_EVER is True at beliefs as
    weak as (0.34, 0.35, 0.31) -- a near-uniform belief. REAUTH's recovery
    term is weighted by b[CANT_PAY_EVER] precisely so such a thin
    plurality cannot put an auth flow in front of a customer who would
    have paid on an ordinary retry. Mistaking CANT_PAY_NOW for
    CANT_PAY_EVER costs a customer; the reverse costs one retry slot."""
    b = _belief(CANT_PAY_NOW=0.34, CANT_PAY_EVER=0.35, WONT_PAY=0.31)
    assert b.dominant() == Cause.CANT_PAY_EVER, "test premise: bare plurality on CANT_PAY_EVER"
    hazard = _flat_hazard(0.4, 0.45, 0.1, 0.05)
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action != Action.REAUTH, \
        "a 0.35 plurality belief must not be enough to trigger REAUTH"


def test_strong_cant_pay_ever_belief_still_triggers_reauth():
    """The complement: belief-weighting must not make REAUTH unreachable.
    A confident CANT_PAY_EVER belief against a poor hazard must still
    route to re-authorisation -- otherwise the safety weighting would have
    silently disabled the whole CANT_PAY_EVER lane."""
    b = _belief(CANT_PAY_EVER=0.9, CANT_PAY_NOW=0.05, WONT_PAY=0.05)
    hazard = _flat_hazard(0.05, 0.02, 0.90, 0.03)
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action == Action.REAUTH


def test_afa_cliff_reauth_is_not_belief_discounted():
    """Above the AFA-free limit, re-authorisation is the only LEGAL route
    (clause 8(a)/8(b)) -- a compliance requirement, not an inference about
    cause. It must fire even at a uniform belief, where the inference path
    would not."""
    def exploding(*, slot, on_day, amount_paise):
        raise AssertionError("hazard was called on an above-cliff mandate")

    ctx = _ctx(amount_paise=2_000_000, ceiling_paise=3_000_000, category="subscription")
    plan = solve(_uniform_belief(), ctx, hazard=exploding, costs=_COSTS)
    assert plan.chosen_action == Action.REAUTH
    assert plan.binding_constraint == "AFA_CLIFF"


def test_now_belief_with_decent_hazard_attempts():
    b = _belief(CANT_PAY_NOW=0.8, CANT_PAY_EVER=0.1, WONT_PAY=0.1)
    hazard = _flat_hazard(0.4, 0.45, 0.1, 0.05)
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action == Action.ATTEMPT
    assert len(plan.committed) == 1
    assert plan.committed[0].amount_paise == 50_000
    assert plan.committed[0].slot == 2


def test_terrible_hazard_and_non_dominant_belief_stops():
    b = _uniform_belief()
    hazard = _flat_hazard(0.1, 0.01, 0.1, 0.79)  # mostly opts out
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action == Action.STOP
    assert plan.committed == ()


def test_full_set_gate_never_produces_offer():
    """Under the B8 default gate, OFFER must be structurally unreachable
    regardless of belief or hazard."""
    b = _belief(WONT_PAY=0.98, CANT_PAY_NOW=0.01, CANT_PAY_EVER=0.01)
    hazard = _flat_hazard(0.1, 0.01, 0.1, 0.79)
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan.chosen_action != Action.OFFER


def test_a_singleton_wont_pay_gate_can_produce_offer():
    """With a real gate wired in (not the FullSetGate stub), OFFER must
    become reachable when the conformal set is the singleton {WONT_PAY} --
    proves OFFER is not dead code, only gated behind B6."""

    class _SingletonWontPayGate:
        def pred_set(self, b) -> frozenset[Cause]:
            return frozenset({Cause.WONT_PAY})

    b = _belief(WONT_PAY=0.98, CANT_PAY_NOW=0.01, CANT_PAY_EVER=0.01)
    hazard = _flat_hazard(0.1, 0.01, 0.1, 0.79)
    plan = solve(b, _ctx(), hazard=hazard, costs=_COSTS, gate=_SingletonWontPayGate())
    assert plan.chosen_action == Action.OFFER


# === attempt cap / opted-out / revoked =======================================

def test_attempt_cap_exhausted_forces_non_attempt():
    ctx = _ctx(attempts_used=MAX_ATTEMPTS)
    plan = solve(_uniform_belief(), ctx, hazard=_flat_hazard(0.4, 0.45, 0.1, 0.05), costs=_COSTS)
    assert plan.chosen_action != Action.ATTEMPT
    assert plan.binding_constraint == "ATTEMPT_CAP_EXHAUSTED"


def test_opted_out_forces_stop():
    def exploding(*, slot, on_day, amount_paise):
        raise AssertionError("hazard was called for an opted-out mandate")

    ctx = _ctx(opted_out=True)
    plan = solve(_uniform_belief(), ctx, hazard=exploding, costs=_COSTS)
    assert plan.chosen_action == Action.STOP
    assert plan.binding_constraint == "OPTED_OUT"


def test_revoked_denies_attempt_but_allows_reauth():
    b = _belief(CANT_PAY_EVER=0.9, CANT_PAY_NOW=0.05, WONT_PAY=0.05)
    ctx = _ctx(mandate_state=MandateState.REVOKED)
    plan = solve(b, ctx, hazard=_flat_hazard(0.9, 0.05, 0.03, 0.02), costs=_COSTS)
    assert plan.chosen_action != Action.ATTEMPT
    assert plan.binding_constraint == "MANDATE_REVOKED"


# === decision_sha256 determinism =============================================

def test_decision_sha256_is_deterministic():
    b = _uniform_belief()
    hazard = _flat_hazard(0.4, 0.45, 0.1, 0.05)
    plan1 = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    plan2 = solve(b, _ctx(), hazard=hazard, costs=_COSTS)
    assert plan1.decision_sha256 == plan2.decision_sha256


def test_decision_sha256_changes_with_amount():
    b = _uniform_belief()
    hazard = _flat_hazard(0.4, 0.45, 0.1, 0.05)
    plan1 = solve(b, _ctx(amount_paise=50_000), hazard=hazard, costs=_COSTS)
    plan2 = solve(b, _ctx(amount_paise=60_000), hazard=hazard, costs=_COSTS)
    assert plan1.decision_sha256 != plan2.decision_sha256


# === marginal hazard makes the cause-sum an identity ========================

def test_marginal_hazard_makes_the_cause_sum_an_identity():
    """Sigma_c b[c] * h == h when h does not vary with c, for any belief on
    the simplex -- the identity that makes narrowing PLAN_DETAIL.md's
    Sigma_c b[c] * h_c(...) down to a single marginal h lossless given the
    available (cause-marginal) hazard source. Checked directly, not just
    asserted in a docstring."""
    h = (0.4, 0.3, 0.2, 0.1)
    for b in (
        _uniform_belief(),
        _belief(CANT_PAY_NOW=0.9, CANT_PAY_EVER=0.05, WONT_PAY=0.05),
        _belief(CANT_PAY_EVER=0.98, CANT_PAY_NOW=0.01, WONT_PAY=0.01),
    ):
        weighted = tuple(sum(b[c] * h[i] for c in Cause) for i in range(4))
        assert weighted == pytest.approx(h), f"identity broke for belief {b.probs}: {weighted} != {h}"


# === belief is carried unchanged across the lookahead ========================

def test_belief_is_unchanged_across_the_still_pending_branch():
    """The recursion's design decision, checked directly: REAUTH/OFFER
    feasibility at a depth-2 lookahead node must be evaluated against the
    SAME b0 passed to solve(), not a belief that has silently drifted."""
    b = _belief(CANT_PAY_EVER=0.9, CANT_PAY_NOW=0.05, WONT_PAY=0.05)
    ctx = _ctx(attempts_used=0, committed_days=())
    hazard = _flat_hazard(0.6, 0.1, 0.1, 0.2)  # mostly "still pending"
    memo: dict = {}
    # At r=2, ATTEMPT's continuation calls _value(b, 1, ctx', ...) with the
    # SAME b object -- if belief were (wrongly) updated per-branch, this
    # call would need a different b, which the recursion's signature does
    # not even provide a mechanism for. This test pins that absence.
    action, value, day = _best_action(b, 2, ctx, hazard, _COSTS, FullSetGate(), memo)
    assert action == Action.REAUTH, \
        "with CANT_PAY_EVER dominant and a mediocre hazard, REAUTH should still be live and considered"


# === 2-slot brute-force equivalence (gate requirement) ======================

def _bf_value(b, r, ctx, hazard, costs, gate) -> float:
    """Independent, unmemoized reimplementation of V(b, r, ctx) -- written
    from scratch, calling none of allocator.py's private helpers."""
    if r <= 0:
        return 0.0
    best = 0.0
    if permitted(Action.REAUTH, ctx) == Verdict.ALLOW:
        if requires_afa(ctx.amount_paise, ctx.category):
            best = max(best, costs.reauth_success_prob * ctx.amount_paise - costs.reauth_cost_paise)
        elif b.dominant() == Cause.CANT_PAY_EVER:
            best = max(
                best,
                b[Cause.CANT_PAY_EVER] * costs.reauth_success_prob * ctx.amount_paise
                - costs.reauth_cost_paise,
            )
    if permitted(Action.OFFER, ctx) == Verdict.ALLOW and should_act(gate.pred_set(b), Cause.WONT_PAY):
        best = max(best, float(costs.mandate_ltv_paise))
    if permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW and not requires_afa(ctx.amount_paise, ctx.category):
        for day in committable_days(ctx):
            p_pending, p_rec, p_dead, p_opt = hazard(
                slot=ctx.attempts_used + 1, on_day=day, amount_paise=ctx.amount_paise
            )
            cont = _bf_value(b, r - 1, ctx.with_attempt(day), hazard, costs, gate) if p_pending > 0 else 0.0
            recoverable = 1.0 - b[Cause.CANT_PAY_EVER]
            q = (
                recoverable * p_rec * ctx.amount_paise
                - p_opt * costs.mandate_ltv_paise
                + p_pending * cont
                - costs.attempt_cost_paise
            )
            best = max(best, q)
    return best


def _bf_solve(b, r, ctx, hazard, costs, gate) -> tuple[Action, float, int | None]:
    best_action, best_value, best_day = Action.STOP, 0.0, None
    if permitted(Action.REAUTH, ctx) == Verdict.ALLOW:
        q = None
        if requires_afa(ctx.amount_paise, ctx.category):
            q = costs.reauth_success_prob * ctx.amount_paise - costs.reauth_cost_paise
        elif b.dominant() == Cause.CANT_PAY_EVER:
            q = (
                b[Cause.CANT_PAY_EVER] * costs.reauth_success_prob * ctx.amount_paise
                - costs.reauth_cost_paise
            )
        if q is not None and q > best_value:
            best_action, best_value, best_day = Action.REAUTH, q, None
    if permitted(Action.OFFER, ctx) == Verdict.ALLOW and should_act(gate.pred_set(b), Cause.WONT_PAY):
        q = float(costs.mandate_ltv_paise)
        if q > best_value:
            best_action, best_value, best_day = Action.OFFER, q, None
    if permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW and not requires_afa(ctx.amount_paise, ctx.category):
        for day in committable_days(ctx):
            p_pending, p_rec, p_dead, p_opt = hazard(
                slot=ctx.attempts_used + 1, on_day=day, amount_paise=ctx.amount_paise
            )
            cont = _bf_value(b, r - 1, ctx.with_attempt(day), hazard, costs, gate) if p_pending > 0 else 0.0
            recoverable = 1.0 - b[Cause.CANT_PAY_EVER]
            q = (
                recoverable * p_rec * ctx.amount_paise
                - p_opt * costs.mandate_ltv_paise
                + p_pending * cont
                - costs.attempt_cost_paise
            )
            if q > best_value:
                best_action, best_value, best_day = Action.ATTEMPT, q, day
    return best_action, best_value, best_day


_EQUIVALENCE_SCENARIOS = [
    dict(belief=dict(CANT_PAY_NOW=0.8, CANT_PAY_EVER=0.1, WONT_PAY=0.1), hazard=(0.4, 0.45, 0.1, 0.05)),
    dict(belief=dict(CANT_PAY_NOW=0.34, CANT_PAY_EVER=0.33, WONT_PAY=0.33), hazard=(0.5, 0.3, 0.1, 0.1)),
    dict(belief=dict(CANT_PAY_EVER=0.9, CANT_PAY_NOW=0.05, WONT_PAY=0.05), hazard=(0.05, 0.02, 0.9, 0.03)),
    dict(belief=dict(CANT_PAY_NOW=0.5, CANT_PAY_EVER=0.3, WONT_PAY=0.2), hazard=(0.3, 0.6, 0.05, 0.05)),
    dict(belief=dict(CANT_PAY_NOW=0.1, CANT_PAY_EVER=0.1, WONT_PAY=0.8), hazard=(0.1, 0.01, 0.1, 0.79)),
    # near-colliding at the 1e-6 quantisation grid, in case a future change
    # reintroduces per-branch belief updates -- the memo key must still be
    # sound even when two beliefs are extremely close.
    dict(belief=dict(CANT_PAY_NOW=0.500000, CANT_PAY_EVER=0.300000, WONT_PAY=0.200000), hazard=(0.4, 0.4, 0.1, 0.1)),
    dict(belief=dict(CANT_PAY_NOW=0.5000001, CANT_PAY_EVER=0.2999999, WONT_PAY=0.2), hazard=(0.4, 0.4, 0.1, 0.1)),
]


@pytest.mark.parametrize("scenario", _EQUIVALENCE_SCENARIOS)
def test_two_slot_brute_force_equivalence(scenario):
    b = _belief(**scenario["belief"])
    hazard = _flat_hazard(*scenario["hazard"])
    ctx = _ctx(attempts_used=2, committed_days=(1, 5))  # r0 = MAX_ATTEMPTS(4) - 2 = 2
    gate = FullSetGate()

    bf_action, bf_value, bf_day = _bf_solve(b, 2, ctx, hazard, _COSTS, gate)

    memo: dict = {}
    solver_action, solver_value, solver_day = _best_action(b, 2, ctx, hazard, _COSTS, gate, memo)

    assert solver_action == bf_action, f"action mismatch: solver={solver_action} bf={bf_action}"
    assert solver_value == pytest.approx(bf_value, abs=1e-9), f"value mismatch: {solver_value} != {bf_value}"
    assert solver_day == bf_day

    # Public, black-box confirmation via solve() itself.
    plan = solve(b, ctx, hazard=hazard, costs=_COSTS, gate=gate)
    assert plan.chosen_action == bf_action
    if bf_action == Action.ATTEMPT:
        assert plan.committed[0].on_day == bf_day
    else:
        assert plan.committed == ()


def test_memo_is_sound_same_key_same_value():
    """Calling _value twice with an identical (b, r, ctx) must return the
    identical value -- the memo must never return a stale or mismatched
    result for a repeated key."""
    b = _belief(CANT_PAY_NOW=0.6, CANT_PAY_EVER=0.25, WONT_PAY=0.15)
    ctx = _ctx(attempts_used=1, committed_days=(1,))
    hazard = _flat_hazard(0.4, 0.4, 0.1, 0.1)
    memo: dict = {}
    v1 = _value(b, 3, ctx, hazard, _COSTS, FullSetGate(), memo)
    v2 = _value(b, 3, ctx, hazard, _COSTS, FullSetGate(), memo)
    assert v1 == v2
    fresh_memo: dict = {}
    v3 = _value(b, 3, ctx, hazard, _COSTS, FullSetGate(), fresh_memo)
    assert v1 == pytest.approx(v3, abs=1e-9), "a cached value must match a freshly-computed one"


def test_belief_is_the_sole_constant_key_component_within_one_solve_call():
    """Documents and pins the consequence of 'belief unchanged across the
    lookahead': within a single solve() call, quantised(b0) never varies,
    so no two distinct beliefs can ever collide in the memo -- the
    collision risk that would exist under a per-branch belief update
    simply does not arise in this design. Regression guard: if a future
    change reintroduces per-branch updates, this test's premise (a single
    quantised value used throughout) should be revisited alongside it."""
    from src.policy.belief import quantised

    b = _belief(CANT_PAY_NOW=0.5, CANT_PAY_EVER=0.3, WONT_PAY=0.2)
    ctx = _ctx(attempts_used=0, committed_days=())
    hazard = _flat_hazard(0.5, 0.3, 0.1, 0.1)
    memo: dict = {}
    _value(b, 4, ctx, hazard, _COSTS, FullSetGate(), memo)
    quantised_keys = {k[0] for k in memo}
    assert quantised_keys == {quantised(b, 1e-6)}, \
        f"expected exactly one distinct quantised belief across the whole memo, got {quantised_keys}"
