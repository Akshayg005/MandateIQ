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
