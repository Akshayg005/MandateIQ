"""Enums shared across every layer.

Nothing here may import from src/model/ or src/policy/ -- this is a leaf
module every layer depends on, and the dependency edge only ever points
one way (see src/core/CLAUDE.md).
"""
from __future__ import annotations

from enum import Enum, IntEnum


class Cause(str, Enum):
    """The three latent causes a failed debit can have."""

    CANT_PAY_NOW = "CANT_PAY_NOW"
    CANT_PAY_EVER = "CANT_PAY_EVER"
    WONT_PAY = "WONT_PAY"


class Outcome(IntEnum):
    """Person-period outcome. STILL_PENDING is 0, the MNLogit reference
    category; RECOVERED/DEAD/OPTED_OUT are 1/2/3 in that order."""

    STILL_PENDING = 0
    RECOVERED = 1
    DEAD = 2
    OPTED_OUT = 3


class DeclineClass(str, Enum):
    """Issuer decline string, normalised into a fixed taxonomy.

    INSUFFICIENT_FUNDS and MANDATE_REVOKED must never collapse into one
    class -- one is a transient liquidity gap, the other a dead instrument.
    UNKNOWN is explicit, never a silent default.
    """

    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    MANDATE_REVOKED = "MANDATE_REVOKED"
    CARD_EXPIRED = "CARD_EXPIRED"
    ACCOUNT_CLOSED = "ACCOUNT_CLOSED"
    ISSUER_DECLINE = "ISSUER_DECLINE"
    BANK_TIMEOUT = "BANK_TIMEOUT"
    UNKNOWN = "UNKNOWN"


class Action(str, Enum):
    ATTEMPT = "ATTEMPT"
    OFFER = "OFFER"
    REAUTH = "REAUTH"
    STOP = "STOP"


class Profile(str, Enum):
    """The two RBI compliance interpretations. Lowercase to mirror
    COMPLIANCE_PROFILE=strict|permissive in .env and CLAUDE.md."""

    strict = "strict"
    permissive = "permissive"


class LedgerState(str, Enum):
    """A ledger row's stage. SENT is forensic only -- see PLAN_DETAIL §3."""

    INTENT = "INTENT"
    SENT = "SENT"
    RESULT = "RESULT"
    FAILED = "FAILED"


class MandateState(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    COMPLETED = "COMPLETED"


class CensorReason(str, Enum):
    NONE = "NONE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    WINDOW_CLOSED = "WINDOW_CLOSED"
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
