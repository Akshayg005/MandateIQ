"""Idempotency key derivation for this layer, plus ScheduledAttempt -- the
in-process and committed_schedule-row shape everything in src/execute/
passes around.

key_for() is a thin, deterministic adapter over src.core.ids.idempotency_key
-- it adds no entropy and no derivation logic of its own. It exists as its
own file, rather than every module importing src.core.ids directly, so:

  1. src/execute/ has exactly one import site for key derivation, and this
     file's source is what test_keys.py's no-clock/uuid/pid guard reads --
     mirroring src/core/ids.py's own guard test (tests/core/test_ids.py),
     scoped to this layer.
  2. amount_paise and generation are always read from an already-committed
     committed_schedule row (via ScheduledAttempt) or from the explicit
     arguments building one -- never from a policy object in flight. B8's
     allocator.CommittedAttempt (slot, on_day, amount_paise) is a PLANNING
     object with no mandate_id, cycle_id, or generation; it cannot key a
     row and must never be passed here. src/execute/commit.py is what
     turns a CommittedAttempt into the first ScheduledAttempt, at
     generation=0.

Must never import time, uuid, os, or random -- src.core.ids.idempotency_key
already doesn't, and this file must not reintroduce non-determinism one
layer up. Enforced by this module's own source-guard test, the same
standard CLAUDE.md sets for src/core/ids.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.ids import idempotency_key


@dataclass(frozen=True)
class ScheduledAttempt:
    """The committed_schedule row shape -- what commit.py writes, and what
    the executor reads back, possibly in a different process, any amount of
    time later. Deliberately mirrors the table's own columns 1:1 rather than
    hiding any of them behind a narrower view: the executor's pre-call late
    read (PLAN_DETAIL.md section 1, B9's late-read principle) and void.py's
    generation bump both need the full row, not a projection of it.

    action, profile: stored as the enum's own .value string, matching how
    the ledger and committed_schedule tables store them -- this dataclass
    is a row shape, not a richer domain object.
    """

    idempotency_key: str
    mandate_id: str
    cycle_id: int
    attempt_index: int
    generation: int
    action: str
    amount_paise: int
    profile: str
    decision_sha256: str
    scheduled_for: datetime
    committed_at: datetime
    voided_at: datetime | None = None
    void_reason: str | None = None


def key_for(
    *,
    mandate_id: str,
    cycle_id: int,
    attempt_index: int,
    generation: int,
    action: str,
    amount_paise: int,
) -> str:
    """The idempotency key for one committed attempt. Pure passthrough to
    src.core.ids.idempotency_key -- see that function's docstring for why
    each field is included (generation makes a void-and-reissue derive a
    distinct key without colliding on ledger_intent_once; amount_paise
    stops a repriced attempt from silently reusing a key)."""
    return idempotency_key(
        mandate_id=mandate_id,
        cycle_id=cycle_id,
        attempt_index=attempt_index,
        generation=generation,
        action=action,
        amount_paise=amount_paise,
    )
