"""Attempt cap, quiet hours, contact frequency, permanent opt-out, and
revoked-never-retried -- the hard stopping rules that gate every action the
allocator considers. A DENY here is final: `allocator.py` must exclude a
denied action from A(b, r, ctx) entirely, never merely warn and proceed
(src/policy/CLAUDE.md; the B8 file table's own "Must NOT: be advisory").

Also defines AllocationContext, the state PLAN_DETAIL.md section 4 calls
`ctx` -- belief lives separately (src/policy/belief.py), but everything else
the Q-function and the stopping rules need (amount, ceiling, category,
attempts already spent, days already committed, contacts already sent,
observed lifecycle state) lives here. Defined in this file rather than
allocator.py so gate.py, offramp.py, and this module can all depend on it
without a circular import back to allocator.py, which is the one file that
depends on everything else in this block.

Two kinds of gating this project keeps distinct:
- HARD, ledger-observed facts (opted out, revoked, attempt cap exhausted,
  contact cap exhausted) -- enforced here, unconditionally, regardless of
  what the allocator's belief says.
- SOFT, belief-based routing (which cause the allocator's current belief
  favours) -- enforced in allocator.py's own action-set construction, never
  here. This file never reads a Belief.

Quiet hours is checked only when a real scheduled timestamp is supplied
(`at`); every B8 call site plans in day-index space and has no such
timestamp yet, so quiet hours is mechanically implemented and tested here
but not exercised end-to-end until B9's executor has a real
`committed_schedule.scheduled_for` to check it against. Stated plainly
rather than left to look like more than it is -- the same disclosure this
project made of belief.py's overconfidence (DECISIONS.md, 2026-08-29, B7).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from src.core.types import Action, MandateState, Outcome, Profile


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass(frozen=True)
class AllocationContext:
    """mandate_id and cycle_id: identity, carried for Plan-writing, not read
    by any decision logic. profile: which compliance interpretation governs
    this cycle. amount_paise, ceiling_paise, and category: the mandate's
    own terms (clause 4(c)'s ceiling, the AFA category). plan_day: the
    day-index this context is being planned as of -- day-index, not a real
    timestamp, matching eval/corpus.py's own day-granularity proxy for the
    6(a) lead time (see that module's assert_legal docstring: "this corpus
    has no intraday clock, so 'on_day >= 1' is the finest check it CAN
    express"). attempts_used: total attempts already spent this cycle
    (including slot 1, the given original). committed_days: on_day values
    already committed this cycle, strictly increasing. contacts_sent: how
    many customer-facing contacts (ATTEMPT, OFFER, or REAUTH) have already
    gone out this cycle. mandate_state, opted_out, and instrument_dead: hard,
    ledger-observed facts -- never inferred from belief. max_contacts_per_cycle,
    quiet_hours_start, and quiet_hours_end: denormalised from PolicyCosts at
    construction time, so stopping_rules.py needs no import of
    src/policy/costs.py.

    instrument_dead -- added R2, 2026-09-04 (reports/gates.md, "Post-B16
    remediation gates"): True once a terminal DEAD outcome has actually been
    observed for this mandate. Defaults False so every pre-existing
    construction site (none of which mentions this field) is unaffected.
    Distinct from `opted_out`, which already existed and already denies
    everything but STOP -- this field exists because a dead instrument has
    NO equivalent representation before R2: `permitted()` had no rule at all
    for "the issuer just confirmed this instrument does not work", so
    ATTEMPT stayed legal after a DEAD outcome purely by omission. See
    with_terminal() below for how a caller sets this (or opted_out, for the
    other terminal outcome) from an observed Outcome."""

    mandate_id: str
    cycle_id: int
    profile: Profile
    amount_paise: int
    ceiling_paise: int
    category: str
    plan_day: int
    attempts_used: int
    committed_days: tuple[int, ...]
    contacts_sent: int
    mandate_state: MandateState
    opted_out: bool
    max_contacts_per_cycle: int
    quiet_hours_start: int
    quiet_hours_end: int
    instrument_dead: bool = False

    def signature(self) -> tuple:
        """Hashable projection used as the memo key's ctx component
        (PLAN_DETAIL.md:1022, `(quantised(b, 1e-6), r, ctx.signature())`).
        Every field that can vary the feasible action set or a Q-value must
        appear here, or two distinct contexts would collide in the memo and
        silently share a cached value that does not apply to both. --
        instrument_dead included (R2): it changes permitted(ATTEMPT)'s
        verdict, exactly the kind of field this docstring warns about."""
        return (
            self.mandate_id,
            self.cycle_id,
            self.profile,
            self.amount_paise,
            self.ceiling_paise,
            self.category,
            self.plan_day,
            self.attempts_used,
            self.committed_days,
            self.contacts_sent,
            self.mandate_state,
            self.opted_out,
            self.max_contacts_per_cycle,
            self.quiet_hours_start,
            self.quiet_hours_end,
            self.instrument_dead,
        )

    def with_attempt(self, on_day: int) -> "AllocationContext":
        """A new context reflecting one more committed attempt on `on_day`
        -- attempts_used +1, on_day appended to committed_days,
        contacts_sent +1, plan_day advanced to on_day (the next decision is
        made no earlier than the attempt just committed)."""
        return replace(
            self,
            attempts_used=self.attempts_used + 1,
            committed_days=self.committed_days + (on_day,),
            contacts_sent=self.contacts_sent + 1,
            plan_day=on_day,
        )

    def with_contact(self) -> "AllocationContext":
        """A new context reflecting one more customer-facing contact that
        is not an ATTEMPT (OFFER or REAUTH) -- contacts_sent +1 only."""
        return replace(self, contacts_sent=self.contacts_sent + 1)

    def with_terminal(self, outcome: Outcome) -> "AllocationContext":
        """A new context reflecting an OBSERVED terminal outcome -- R2,
        2026-09-04. `Outcome.DEAD` sets instrument_dead (the issuer
        confirmed the instrument does not work: CANT_PAY_EVER by
        definition); `Outcome.OPTED_OUT` sets the existing opted_out (the
        customer said so: WONT_PAY by definition, and 6(c) makes it
        terminal). Both are hard, ledger-observed facts, not an inference
        from belief -- the same category as every other field this class
        already treats that way. Raises ValueError for any other Outcome:
        RECOVERED needs no context transition (the cycle succeeded, nothing
        left to re-solve for) and STILL_PENDING is not terminal at all --
        calling this with either is an upstream bug, not a case to handle
        silently."""
        if outcome == Outcome.DEAD:
            return replace(self, instrument_dead=True)
        if outcome == Outcome.OPTED_OUT:
            return replace(self, opted_out=True)
        raise ValueError(
            f"with_terminal() called with non-terminal-for-this-purpose "
            f"outcome {outcome!r} -- only DEAD and OPTED_OUT need a context "
            f"transition"
        )


_CONTACTING_ACTIONS = (Action.ATTEMPT, Action.OFFER, Action.REAUTH)


def _in_quiet_hours(at: datetime, start_hour: int, end_hour: int) -> bool:
    """True if `at`'s local hour falls in the [start_hour, end_hour) window,
    wrapping past midnight (e.g. 21 -> 8 means 21:00-23:59 and 00:00-07:59
    are both in-window)."""
    hour = at.hour
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def permitted(
    action: Action,
    ctx: AllocationContext,
    *,
    at: datetime | None = None,
) -> Verdict:
    """Whether `action` is permitted under `ctx`'s hard, ledger-observed
    state. A DENY is final -- the caller must treat it as excluding the
    action, never as advice to weigh against a Q-value.

    `at`, if given, is the real wall-clock moment the action would actually
    reach the customer (e.g. `committed_schedule.scheduled_for` at B9); only
    then is the quiet-hours rule checked. B8's own planning-time call sites
    never have such a timestamp (they work in day-index space), so this
    parameter defaults to None and quiet hours is a no-op there --
    mechanically correct and tested, but not exercised live until B9.
    """
    # 6(c): opt-out is terminal. No further contact of any kind, ever,
    # except STOP.
    if ctx.opted_out and action != Action.STOP:
        return Verdict.DENY

    # Revoked-never-retried: a revoked mandate has no instrument to charge.
    # ATTEMPT is never legal; REAUTH/OFFER/STOP remain available.
    if ctx.mandate_state == MandateState.REVOKED and action == Action.ATTEMPT:
        return Verdict.DENY

    # R2, 2026-09-04: a CONFIRMED-dead instrument (an observed DEAD outcome,
    # not a belief about one) has nothing left to charge. Same shape as the
    # REVOKED rule above -- ATTEMPT denied, REAUTH/OFFER/STOP remain
    # available -- but a distinct ledger fact: REVOKED is a mandate-lifecycle
    # state: never had a live instrument; instrument_dead is "had one, the
    # issuer just confirmed it stopped working." Before this rule existed,
    # ATTEMPT stayed legal after an observed DEAD outcome purely because
    # nothing denied it (reports/gates.md, R2a).
    if ctx.instrument_dead and action == Action.ATTEMPT:
        return Verdict.DENY

    # NPCI attempt cap: 1 original + 3 retries = 4, ever.
    if action == Action.ATTEMPT:
        from src.policy.constraints import MAX_ATTEMPTS

        if ctx.attempts_used >= MAX_ATTEMPTS:
            return Verdict.DENY

    # Contact-frequency cap: ATTEMPT/OFFER/REAUTH all reach the customer.
    if action in _CONTACTING_ACTIONS and ctx.contacts_sent >= ctx.max_contacts_per_cycle:
        return Verdict.DENY

    # Quiet hours: only checked when a real scheduled moment is supplied.
    if at is not None and action in _CONTACTING_ACTIONS:
        if _in_quiet_hours(at, ctx.quiet_hours_start, ctx.quiet_hours_end):
            return Verdict.DENY

    return Verdict.ALLOW
