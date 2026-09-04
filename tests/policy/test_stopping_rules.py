"""src/policy/stopping_rules.py -- hard, ledger-observed stopping rules.

Design spec: permitted(action, ctx) returns Verdict.DENY for: a permanently
opted-out mandate (any action but STOP), a revoked mandate attempting to be
charged, the NPCI attempt cap, the contact-frequency cap, and -- only when a
real scheduled timestamp is supplied -- quiet hours. A DENY is final; this
file never reads a Belief (that is allocator.py's soft, belief-based
routing, kept deliberately separate).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.types import Action, MandateState, Profile
from src.policy.stopping_rules import AllocationContext, Verdict, permitted


def _ctx(**overrides) -> AllocationContext:
    base = dict(
        mandate_id="M-1",
        cycle_id=1,
        profile=Profile.strict,
        amount_paise=50_000,
        ceiling_paise=100_000,
        category="subscription",
        plan_day=0,
        attempts_used=1,
        committed_days=(),
        contacts_sent=1,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=4,
        quiet_hours_start=21,
        quiet_hours_end=8,
    )
    base.update(overrides)
    return AllocationContext(**base)


# === baseline: an ordinary active mandate allows everything relevant =======

def test_ordinary_context_allows_attempt():
    assert permitted(Action.ATTEMPT, _ctx()) == Verdict.ALLOW


def test_stop_is_always_allowed():
    """STOP must never be denied -- it is the universal escape hatch every
    other action can fall back to."""
    ctx = _ctx(opted_out=True, mandate_state=MandateState.REVOKED, attempts_used=4, contacts_sent=4)
    assert permitted(Action.STOP, ctx) == Verdict.ALLOW


# === permanent opt-out: terminal, per clause 6(c) ===========================

def test_opted_out_denies_attempt():
    assert permitted(Action.ATTEMPT, _ctx(opted_out=True)) == Verdict.DENY


def test_opted_out_denies_offer_and_reauth_too():
    """6(c) is terminal -- no further contact of any kind, not just no
    further debit attempts."""
    ctx = _ctx(opted_out=True)
    assert permitted(Action.OFFER, ctx) == Verdict.DENY
    assert permitted(Action.REAUTH, ctx) == Verdict.DENY


def test_opted_out_only_stop_survives():
    ctx = _ctx(opted_out=True)
    assert permitted(Action.STOP, ctx) == Verdict.ALLOW


# === revoked-never-retried ===================================================

def test_revoked_denies_attempt():
    assert permitted(Action.ATTEMPT, _ctx(mandate_state=MandateState.REVOKED)) == Verdict.DENY


def test_revoked_allows_reauth():
    """A revoked mandate has no instrument to charge, but re-authorisation
    is exactly the recovery path for a dead instrument -- REAUTH must stay
    available."""
    ctx = _ctx(mandate_state=MandateState.REVOKED)
    assert permitted(Action.REAUTH, ctx) == Verdict.ALLOW


# === NPCI attempt cap =========================================================

def test_attempt_cap_denies_the_fifth_attempt():
    from src.policy.constraints import MAX_ATTEMPTS

    ctx = _ctx(attempts_used=MAX_ATTEMPTS)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.DENY


def test_attempt_cap_allows_the_fourth_attempt():
    from src.policy.constraints import MAX_ATTEMPTS

    ctx = _ctx(attempts_used=MAX_ATTEMPTS - 1)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW


# === contact-frequency cap ====================================================

def test_contact_cap_denies_attempt_when_exhausted():
    ctx = _ctx(contacts_sent=4, max_contacts_per_cycle=4)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.DENY


def test_contact_cap_denies_offer_and_reauth_too():
    ctx = _ctx(contacts_sent=4, max_contacts_per_cycle=4)
    assert permitted(Action.OFFER, ctx) == Verdict.DENY
    assert permitted(Action.REAUTH, ctx) == Verdict.DENY


def test_contact_cap_does_not_gate_stop():
    ctx = _ctx(contacts_sent=4, max_contacts_per_cycle=4)
    assert permitted(Action.STOP, ctx) == Verdict.ALLOW


def test_contact_cap_allows_below_the_limit():
    ctx = _ctx(contacts_sent=3, max_contacts_per_cycle=4)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW


# === quiet hours: no-op without `at`, real with it ==========================

def test_quiet_hours_is_a_noop_without_a_timestamp():
    """B8's own planning-time call sites work in day-index space and have
    no real scheduled moment yet -- permitted() must not deny anything on
    quiet-hours grounds when `at` is omitted."""
    ctx = _ctx(quiet_hours_start=21, quiet_hours_end=8)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.ALLOW


def test_quiet_hours_denies_when_at_falls_inside_the_window():
    ctx = _ctx(quiet_hours_start=21, quiet_hours_end=8)
    at = datetime(2026, 8, 29, 23, 0, tzinfo=timezone.utc)  # 23:00, inside 21-8
    assert permitted(Action.ATTEMPT, ctx, at=at) == Verdict.DENY


def test_quiet_hours_denies_past_midnight_too():
    """The window wraps past midnight -- 03:00 is inside 21:00-08:00."""
    ctx = _ctx(quiet_hours_start=21, quiet_hours_end=8)
    at = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)
    assert permitted(Action.ATTEMPT, ctx, at=at) == Verdict.DENY


def test_quiet_hours_allows_daytime():
    ctx = _ctx(quiet_hours_start=21, quiet_hours_end=8)
    at = datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc)  # 14:00, outside 21-8
    assert permitted(Action.ATTEMPT, ctx, at=at) == Verdict.ALLOW


def test_quiet_hours_does_not_gate_stop():
    ctx = _ctx(quiet_hours_start=21, quiet_hours_end=8)
    at = datetime(2026, 8, 29, 23, 0, tzinfo=timezone.utc)
    assert permitted(Action.STOP, ctx, at=at) == Verdict.ALLOW


# === AllocationContext helpers ===============================================

def test_context_is_frozen():
    from dataclasses import FrozenInstanceError

    ctx = _ctx()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ctx.attempts_used = 99  # type: ignore


def test_with_attempt_advances_state():
    ctx = _ctx(attempts_used=1, committed_days=(3,), contacts_sent=1, plan_day=0)
    nxt = ctx.with_attempt(on_day=10)
    assert nxt.attempts_used == 2
    assert nxt.committed_days == (3, 10)
    assert nxt.contacts_sent == 2
    assert nxt.plan_day == 10
    # original untouched
    assert ctx.attempts_used == 1
    assert ctx.committed_days == (3,)


def test_with_contact_only_bumps_contacts_sent():
    ctx = _ctx(attempts_used=1, contacts_sent=1)
    nxt = ctx.with_contact()
    assert nxt.contacts_sent == 2
    assert nxt.attempts_used == 1


def test_signature_is_hashable_and_reflects_state():
    ctx1 = _ctx(attempts_used=1)
    ctx2 = _ctx(attempts_used=2)
    hash(ctx1.signature())  # must not raise
    assert ctx1.signature() != ctx2.signature()


def test_signature_equal_for_equal_contexts():
    assert _ctx().signature() == _ctx().signature()


# === instrument_dead field and rules (R2, 2026-09-04) =======================

def test_instrument_dead_field_defaults_to_false():
    """The instrument_dead field must default to False, so every existing
    construction site (none of which mentions this field) continues to work."""
    ctx = _ctx()
    assert ctx.instrument_dead is False


def test_instrument_dead_can_be_set_to_true():
    """The instrument_dead field must be settable at construction time via
    the AllocationContext constructor or via with_terminal()."""
    ctx = _ctx(instrument_dead=True)
    assert ctx.instrument_dead is True


def test_signature_includes_instrument_dead():
    """Two contexts differing ONLY in instrument_dead must produce DIFFERENT
    signature() tuples. This prevents memo collisions where a cached Q-value
    from one context would be reused incorrectly in another."""
    ctx_alive = _ctx(instrument_dead=False)
    ctx_dead = _ctx(instrument_dead=True)

    sig_alive = ctx_alive.signature()
    sig_dead = ctx_dead.signature()

    assert sig_alive != sig_dead, \
        "signature() must differ when only instrument_dead differs"


def test_instrument_dead_is_last_element_of_signature():
    """The signature tuple must include instrument_dead as its final element,
    after quiet_hours_end."""
    ctx = _ctx(instrument_dead=True)
    sig = ctx.signature()

    # signature() is a tuple with 16 elements; instrument_dead must be the last
    assert isinstance(sig, tuple), f"signature() is {type(sig).__name__}, not tuple"
    assert len(sig) == 16, f"signature() has {len(sig)} elements, expected 16"
    assert sig[-1] == True, f"Last element of signature should be True (instrument_dead), got {sig[-1]}"


def test_instrument_dead_denies_attempt():
    """When ctx.instrument_dead is True, permitted(Action.ATTEMPT, ctx) must
    return Verdict.DENY, regardless of all other context fields."""
    ctx = _ctx(
        instrument_dead=True,
        attempts_used=1,  # well below MAX_ATTEMPTS
        contacts_sent=1,  # well below max_contacts_per_cycle
        opted_out=False,  # not opted out
        mandate_state=MandateState.ACTIVE,  # active mandate
    )

    assert permitted(Action.ATTEMPT, ctx) == Verdict.DENY


def test_instrument_dead_allows_reauth():
    """When ctx.instrument_dead is True, permitted(Action.REAUTH, ctx) must
    return Verdict.ALLOW. REAUTH is exactly the recovery path for a dead
    instrument."""
    ctx = _ctx(instrument_dead=True)
    assert permitted(Action.REAUTH, ctx) == Verdict.ALLOW


def test_instrument_dead_allows_offer():
    """When ctx.instrument_dead is True, permitted(Action.OFFER, ctx) must
    return Verdict.ALLOW. OFFER remains available."""
    ctx = _ctx(instrument_dead=True)
    assert permitted(Action.OFFER, ctx) == Verdict.ALLOW


def test_instrument_dead_allows_stop():
    """When ctx.instrument_dead is True, permitted(Action.STOP, ctx) must
    return Verdict.ALLOW. STOP is the universal escape hatch."""
    ctx = _ctx(instrument_dead=True)
    assert permitted(Action.STOP, ctx) == Verdict.ALLOW


def test_instrument_dead_rule_is_checked_before_attempt_cap():
    """If a mandate has attempts_used == MAX_ATTEMPTS AND instrument_dead==True,
    it is the instrument_dead rule that denies ATTEMPT, not the attempt cap.
    Both rules apply, but order should not matter for the outcome (DENY either way)."""
    from src.policy.constraints import MAX_ATTEMPTS

    ctx = _ctx(instrument_dead=True, attempts_used=MAX_ATTEMPTS)
    assert permitted(Action.ATTEMPT, ctx) == Verdict.DENY


# === with_terminal() method (R2, 2026-09-04) ================================

def test_with_terminal_dead_sets_instrument_dead():
    """with_terminal(Outcome.DEAD) must return a new context with
    instrument_dead=True."""
    from src.core.types import Outcome

    ctx = _ctx(instrument_dead=False)
    ctx_dead = ctx.with_terminal(Outcome.DEAD)

    assert ctx_dead.instrument_dead is True
    # Original must be unchanged
    assert ctx.instrument_dead is False


def test_with_terminal_dead_preserves_other_fields():
    """with_terminal(Outcome.DEAD) must preserve all other fields unchanged."""
    from src.core.types import Outcome

    ctx = _ctx(
        instrument_dead=False,
        opted_out=False,
        attempts_used=2,
        contacts_sent=1,
        plan_day=5,
        mandate_id="M-123",
    )
    ctx_dead = ctx.with_terminal(Outcome.DEAD)

    # instrument_dead should be set to True
    assert ctx_dead.instrument_dead is True
    # Everything else should be identical
    assert ctx_dead.opted_out == False
    assert ctx_dead.attempts_used == 2
    assert ctx_dead.contacts_sent == 1
    assert ctx_dead.plan_day == 5
    assert ctx_dead.mandate_id == "M-123"


def test_with_terminal_opted_out_sets_opted_out():
    """with_terminal(Outcome.OPTED_OUT) must return a new context with
    opted_out=True."""
    from src.core.types import Outcome

    ctx = _ctx(opted_out=False)
    ctx_opted = ctx.with_terminal(Outcome.OPTED_OUT)

    assert ctx_opted.opted_out is True
    # Original must be unchanged
    assert ctx.opted_out is False


def test_with_terminal_opted_out_preserves_instrument_dead():
    """with_terminal(Outcome.OPTED_OUT) must NOT modify instrument_dead,
    only opted_out."""
    from src.core.types import Outcome

    ctx = _ctx(opted_out=False, instrument_dead=False)
    ctx_opted = ctx.with_terminal(Outcome.OPTED_OUT)

    assert ctx_opted.opted_out is True
    # instrument_dead should remain False
    assert ctx_opted.instrument_dead is False


def test_with_terminal_opted_out_preserves_other_fields():
    """with_terminal(Outcome.OPTED_OUT) must preserve all other fields
    unchanged except opted_out."""
    from src.core.types import Outcome

    ctx = _ctx(
        opted_out=False,
        instrument_dead=True,
        attempts_used=1,
        contacts_sent=2,
        plan_day=3,
        mandate_id="M-456",
    )
    ctx_opted = ctx.with_terminal(Outcome.OPTED_OUT)

    # opted_out should be set to True
    assert ctx_opted.opted_out is True
    # Everything else should be identical
    assert ctx_opted.instrument_dead is True
    assert ctx_opted.attempts_used == 1
    assert ctx_opted.contacts_sent == 2
    assert ctx_opted.plan_day == 3
    assert ctx_opted.mandate_id == "M-456"


def test_with_terminal_rejects_recovered():
    """with_terminal(Outcome.RECOVERED) must raise ValueError because
    RECOVERED is not terminal-for-this-purpose (the cycle succeeded, nothing
    left to re-solve for)."""
    from src.core.types import Outcome

    ctx = _ctx()

    with pytest.raises(ValueError):
        ctx.with_terminal(Outcome.RECOVERED)


def test_with_terminal_rejects_still_pending():
    """with_terminal(Outcome.STILL_PENDING) must raise ValueError because
    STILL_PENDING is not terminal at all."""
    from src.core.types import Outcome

    ctx = _ctx()

    with pytest.raises(ValueError):
        ctx.with_terminal(Outcome.STILL_PENDING)


def test_with_terminal_original_context_is_immutable():
    """Calling with_terminal() on a context must not modify the original
    context, even though with_terminal() returns a new one."""
    from src.core.types import Outcome

    ctx = _ctx(opted_out=False, instrument_dead=False)
    ctx_copy = _ctx(opted_out=False, instrument_dead=False)

    # Call with_terminal on a copy
    _ = ctx.with_terminal(Outcome.OPTED_OUT)

    # Original must be unchanged
    assert ctx.opted_out == ctx_copy.opted_out == False
    assert ctx.instrument_dead == ctx_copy.instrument_dead == False
