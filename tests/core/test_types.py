"""
src/core/types.py -- shared enums, no behavior. Membership and (for
Outcome) exact ordering are the only things worth testing here.

Design decision this test file pins (documented since the module doesn't
exist yet to document it itself): Outcome is an IntEnum with explicit
values 0/1/2/3, because a downstream MNLogit survival model needs
STILL_PENDING as event_code 0 (the reference category) -- see
src/model/CLAUDE.md. Profile's member NAMES are lowercase (`strict`,
`permissive`) to mirror COMPLIANCE_PROFILE=strict|permissive in .env and
the exact spelling CLAUDE.md uses throughout, rather than the usual
upper-case enum convention.
"""
from __future__ import annotations

import pathlib
import re
from enum import IntEnum

from src.core.types import (
    Action,
    Cause,
    CensorReason,
    DeclineClass,
    LedgerState,
    MandateState,
    Outcome,
    Profile,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TYPES_SRC = ROOT / "src" / "core" / "types.py"


def test_cause_has_exactly_the_three_latent_causes():
    names = {m.name for m in Cause}
    assert names == {"CANT_PAY_NOW", "CANT_PAY_EVER", "WONT_PAY"}


def test_outcome_is_int_enum_with_mnlogit_reference_category_ordering():
    """STILL_PENDING must be event_code 0 (the MNLogit reference category);
    RECOVERED, DEAD, OPTED_OUT are 1, 2, 3 in that exact order."""
    assert issubclass(Outcome, IntEnum)
    assert Outcome.STILL_PENDING == 0
    assert Outcome.RECOVERED == 1
    assert Outcome.DEAD == 2
    assert Outcome.OPTED_OUT == 3


def test_outcome_has_exactly_four_members():
    names = {m.name for m in Outcome}
    assert names == {"STILL_PENDING", "RECOVERED", "DEAD", "OPTED_OUT"}


def test_decline_class_keeps_insufficient_funds_and_mandate_revoked_distinct():
    """The whole point of this taxonomy: a later module must never collapse
    a transient liquidity gap into a dead instrument."""
    assert DeclineClass.INSUFFICIENT_FUNDS != DeclineClass.MANDATE_REVOKED


def test_decline_class_has_required_members():
    required = {"INSUFFICIENT_FUNDS", "MANDATE_REVOKED", "UNKNOWN", "CARD_EXPIRED", "ACCOUNT_CLOSED"}
    names = {m.name for m in DeclineClass}
    assert required.issubset(names)


def test_decline_class_unknown_is_an_explicit_member_not_a_silent_default():
    assert hasattr(DeclineClass, "UNKNOWN")
    assert DeclineClass.UNKNOWN in list(DeclineClass)


def test_action_has_exactly_the_four_actions():
    names = {m.name for m in Action}
    assert names == {"ATTEMPT", "OFFER", "REAUTH", "STOP"}


def test_profile_has_both_compliance_interpretations():
    names = {m.name for m in Profile}
    assert names == {"strict", "permissive"}
    assert Profile.strict != Profile.permissive


def test_ledger_state_has_exactly_the_four_states():
    """SENT is a real member even though nothing writes it until a later
    block -- it must exist in the type now."""
    names = {m.name for m in LedgerState}
    assert names == {"INTENT", "SENT", "RESULT", "FAILED"}


def test_mandate_state_has_exactly_the_six_lifecycle_states():
    names = {m.name for m in MandateState}
    assert names == {"CREATED", "ACTIVE", "PAUSED", "REVOKED", "EXPIRED", "COMPLETED"}


def test_censor_reason_has_exactly_the_four_reasons():
    names = {m.name for m in CensorReason}
    assert names == {"NONE", "BUDGET_EXHAUSTED", "WINDOW_CLOSED", "MANDATE_EXPIRED"}


def test_types_module_does_not_import_model_or_policy():
    """types.py is a shared leaf module used by every layer. It must not
    create a dependency from src/core back up into src/model or
    src/policy -- that would be a layering violation, not just untidy."""
    text = TYPES_SRC.read_text(encoding="utf-8")
    forbidden = [
        r"from\s+src\.model\b",
        r"from\s+src\.policy\b",
        r"import\s+src\.model\b",
        r"import\s+src\.policy\b",
    ]
    for pattern in forbidden:
        match = re.search(pattern, text)
        assert match is None, f"forbidden import found in types.py: {match.group(0)!r}"
