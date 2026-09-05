"""Append-only writes and replay on top of schema.sql. Every public function
takes an explicit connection as its first argument -- there is no
module-level connection to manage, and no attempt to hide the DB.

This file must never UPDATE or DELETE a ledger row. Anything that needs to
change (voiding a schedule, recording a lifecycle transition) lives in a
different table, and a different module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.types import MandateState


@dataclass(frozen=True)
class LedgerEntry:
    """What a caller supplies to append(). Covers every NOT NULL column on
    `ledger` except ledger_id (BIGSERIAL) and created_at (DB DEFAULT now())."""

    idempotency_key: str
    mandate_id: str
    cycle_id: int
    attempt_index: int
    action: str
    state: str
    amount_paise: int
    profile: str
    payload_sha256: str
    decision_sha256: str
    provider_ref: str | None = None
    outcome: str | None = None
    decline_class: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LedgerRow:
    """A row read back from `ledger`."""

    ledger_id: int
    idempotency_key: str
    mandate_id: str
    cycle_id: int
    attempt_index: int
    action: str
    state: str
    amount_paise: int
    provider_ref: str | None
    outcome: str | None
    decline_class: str | None
    reason: str | None
    profile: str
    payload_sha256: str
    decision_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class PlanRow:
    """A row read back from `plan`, mirroring the table 1:1 -- the same
    convention LedgerRow above follows for `ledger`.

    R6, 2026-09-05 (reports/gates.md, "Post-B16 remediation gates"). Before
    this, NOTHING in src/ read the `plan` table: src/execute/commit.py's
    `_insert_plan_row` was the only code that touched it, and it only ever
    wrote. Both /plan/{mandate_id} and /decision/{sha} are net-new SQL.

    NOTE what is NOT here, because the table does not have it:
    `chosen_action`. A plan's action is recoverable only by outer-joining
    `committed_schedule` on `decision_sha256`, since commit()'s own gate
    writes such a row for ATTEMPT and for nothing else. That derivation
    belongs to the caller that needs it (src/api/read.py), not to this
    row type, which mirrors the table honestly rather than inventing a
    column. tests/execute/test_cycle.py::_non_attempt_plan_rows already
    documented that workaround at R4; this reuses its reasoning.

    `conformal_set` is returned VERBATIM -- the sorted comma-joined string
    commit.py writes. Splitting it is a presentation decision (and `""`
    must become `[]`, not `[""]`), made where the JSON is shaped.
    """

    decision_sha256: str
    mandate_id: str
    cycle_id: int
    profile: str
    belief_json: str
    conformal_set: str
    binding_constraint: str | None
    solver_version: str
    created_at: datetime


_PLAN_COLUMNS = (
    "decision_sha256", "mandate_id", "cycle_id", "profile", "belief_json",
    "conformal_set", "binding_constraint", "solver_version", "created_at",
)


def _row_to_plan(row) -> PlanRow:
    return PlanRow(**dict(zip(_PLAN_COLUMNS, row)))


def find_plan(conn, decision_sha256: str) -> PlanRow | None:
    """The plan a decision hash names, or None. `decision_sha256` is the
    table's PRIMARY KEY, so this is exact -- there is no "most recent" to
    disambiguate, unlike find_by_key() above."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_PLAN_COLUMNS)} FROM plan WHERE decision_sha256 = %s",
            (decision_sha256,),
        )
        row = cur.fetchone()
    return _row_to_plan(row) if row else None


def plans_for_mandate(conn, mandate_id: str) -> list[PlanRow]:
    """Every plan written for `mandate_id`, oldest first.

    Ordered by `created_at`, then by `decision_sha256` as a deterministic
    tie-break: `plan.created_at` is a DB-clock `DEFAULT now()` with no
    serial ordinal, and two rows written inside one transaction share an
    identical timestamp -- exactly the ordering ambiguity R4's own
    `_is_eligible()` docstring records declining to build on. A read
    endpoint cannot decline, so it breaks the tie on a stable value
    instead of returning an order that varies between identical calls.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_PLAN_COLUMNS)} FROM plan WHERE mandate_id = %s "
            "ORDER BY created_at ASC, decision_sha256 ASC",
            (mandate_id,),
        )
        rows = cur.fetchall()
    return [_row_to_plan(row) for row in rows]


def committed_for_decision(conn, decision_sha256: str) -> dict | None:
    """The `committed_schedule` row citing this decision, if any.

    This is what makes a plan's `chosen_action` recoverable at all: only
    ATTEMPT ever gets such a row (src/execute/commit.py's own gate), so
    its presence IS the action. Returns a plain dict rather than a new
    frozen dataclass -- the caller needs a handful of fields for a JSON
    surface, and a full CommittedScheduleRow type with no other consumer
    would be a convention invented for one call site.

    **VOIDED ROWS ARE DELIBERATELY NOT FILTERED OUT.** `money-auditor`
    (2026-09-05) proposed adding `AND voided_at IS NULL`, on the reasoning
    that `committed_one_live_per_slot` -- the schema's own unique index --
    establishes non-voided as the "live" convention. The convention is real
    and the fix would be wrong here, which is why this paragraph exists
    rather than the filter.

    The question this function answers is "what did the allocator DECIDE",
    not "what is currently scheduled". `commit()` writes a
    committed_schedule row ONLY for ATTEMPT, and voiding is a LATER event
    (src/execute/void.py, an overtaken-by-events reissue path) that cannot
    retroactively change what was chosen. Filtering voided rows would
    return None for a decision that provably WAS an ATTEMPT, and
    src/api/read.py's `_derive_action()` would then report it as
    NOT_ATTEMPT with candidates [REAUTH, STOP] -- a strictly false answer
    where the current one is a true-but-incomplete one.

    The real defect the review found is narrower and IS fixed: nothing said
    the cited row was dead. The caller now surfaces `is_live` and says so
    in its derivation message.

    ORDER BY generation DESC picks the LIVE row when one exists:
    `void.reissue()` inserts a replacement at generation+1 carrying the
    SAME decision_sha256 (void.py's own INSERT), so a voided generation 0
    and a live generation 1 can both cite one decision.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT idempotency_key, attempt_index, amount_paise, scheduled_for, "
            "committed_at, voided_at, void_reason "
            "FROM committed_schedule WHERE decision_sha256 = %s "
            "ORDER BY generation DESC, committed_at DESC LIMIT 1",
            (decision_sha256,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    keys = ("idempotency_key", "attempt_index", "amount_paise", "scheduled_for",
            "committed_at", "voided_at", "void_reason")
    return dict(zip(keys, row))


def ledger_for_decision(conn, decision_sha256: str) -> list[LedgerRow]:
    """Every ledger row citing this decision, in insertion order. Empty for
    a STOP/REAUTH/OFFER plan, which writes a `plan` row and no ledger row
    -- absence of execution, not absence of the decision."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_LEDGER_COLUMNS)} FROM ledger "
            "WHERE decision_sha256 = %s ORDER BY ledger_id ASC",
            (decision_sha256,),
        )
        rows = cur.fetchall()
    return [_row_to_entry(row) for row in rows]


_LEDGER_COLUMNS = (
    "ledger_id", "idempotency_key", "mandate_id", "cycle_id", "attempt_index",
    "action", "state", "amount_paise", "provider_ref", "outcome",
    "decline_class", "reason", "profile", "payload_sha256", "decision_sha256",
    "created_at",
)


def _row_to_entry(row) -> LedgerRow:
    return LedgerRow(**dict(zip(_LEDGER_COLUMNS, row)))


def append(conn, entry: LedgerEntry) -> int | None:
    """Insert one ledger row. Returns the new ledger_id, or None if a row
    with the same idempotency_key and state='INTENT' already exists -- the
    "0 rows -> this attempt already exists" step of the write-ordering
    protocol. A retried append of the same INTENT must never create a
    second row: that is exactly what would let a retry double-charge."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ledger (
                idempotency_key, mandate_id, cycle_id, attempt_index, action,
                state, amount_paise, provider_ref, outcome, decline_class,
                reason, profile, payload_sha256, decision_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) WHERE state = 'INTENT' DO NOTHING
            RETURNING ledger_id
            """,
            (
                entry.idempotency_key, entry.mandate_id, entry.cycle_id,
                entry.attempt_index, entry.action, entry.state,
                entry.amount_paise, entry.provider_ref, entry.outcome,
                entry.decline_class, entry.reason, entry.profile,
                entry.payload_sha256, entry.decision_sha256,
            ),
        )
        row = cur.fetchone()
    return row[0] if row else None


def replay(conn, mandate_id: str) -> list[LedgerRow]:
    """Every ledger row for `mandate_id`, in insertion order."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_LEDGER_COLUMNS)} FROM ledger "
            "WHERE mandate_id = %s ORDER BY ledger_id ASC",
            (mandate_id,),
        )
        rows = cur.fetchall()
    return [_row_to_entry(row) for row in rows]


def find_by_key(conn, idempotency_key: str) -> LedgerRow | None:
    """The most recent row for `idempotency_key` -- an attempt can have an
    INTENT, then a SENT, then a RESULT row sharing one key, and this
    returns the latest stage, not the first."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(_LEDGER_COLUMNS)} FROM ledger "
            "WHERE idempotency_key = %s ORDER BY ledger_id DESC LIMIT 1",
            (idempotency_key,),
        )
        row = cur.fetchone()
    return _row_to_entry(row) if row else None


def latest_state(conn, mandate_id: str) -> MandateState:
    """The mandate's current lifecycle state -- the latest mandate_lifecycle
    row by effective_at, not by insertion order. Raises LookupError if the
    mandate has no lifecycle rows at all; every real mandate has at least a
    CREATED row, so silently returning an "unknown" sentinel would hide a
    real bug rather than surface one."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT state FROM mandate_lifecycle WHERE mandate_id = %s "
            "ORDER BY effective_at DESC LIMIT 1",
            (mandate_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"no mandate_lifecycle rows for mandate_id={mandate_id!r}")
    return MandateState(row[0])


def record_lifecycle_event(
    conn, *, event_id: str, mandate_id: str, state: str, source: str,
    effective_at: datetime,
) -> MandateState:
    """Record one mandate_lifecycle transition, keyed by the provider's own
    event_id (also this table's dedupe key). ON CONFLICT DO NOTHING: a
    retried webhook delivery of an event_id we already recorded is not an
    error, it's the normal case. Returns the MandateState the row
    represents -- either the state just inserted, or, on a duplicate
    event_id, the state recorded the FIRST time (re-read from the existing
    row), never the caller's second-call arguments."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO mandate_lifecycle (event_id, mandate_id, state, source, effective_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            RETURNING state
            """,
            (event_id, mandate_id, state, source, effective_at),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT state FROM mandate_lifecycle WHERE event_id = %s",
                (event_id,),
            )
            row = cur.fetchone()
    return MandateState(row[0])


def record_ingested_event(
    conn, *, event_id: str, event_type: str, raw_payload_sha256: str,
    mandate_id: str | None = None, provider_ref: str | None = None,
    decline_code: str | None = None, decline_text: str | None = None,
    decline_class: str | None = None, cause_prior_json: str | None = None,
    taxonomy_version: str | None = None, prior_version: str | None = None,
    amount_paise: int | None = None,
) -> None:
    """Record one classified ingest event into `ingested_event`. Fire and
    forget: no return value. ON CONFLICT DO NOTHING, mirroring append()'s
    dedup discipline -- a retried webhook delivery of the same event_id is
    silently a no-op, never a second row and never an exception.

    taxonomy_version / prior_version record which decline_taxonomy.py /
    cause_map.py ruleset produced decline_class / cause_prior_json on THIS
    row -- see schema.sql's comment on these columns."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_event (
                event_id, event_type, mandate_id, provider_ref, decline_code,
                decline_text, decline_class, cause_prior, taxonomy_version,
                prior_version, amount_paise, raw_payload_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (
                event_id, event_type, mandate_id, provider_ref, decline_code,
                decline_text, decline_class, cause_prior_json, taxonomy_version,
                prior_version, amount_paise, raw_payload_sha256,
            ),
        )


@dataclass(frozen=True)
class NormalizedDeclineRow:
    """A row read back from `normalized_decline`."""

    event_id: str
    value: str
    confidence: float
    normalizer_version: str
    model_id: str
    raw_sha256: str
    created_at: datetime


def record_normalized_decline(
    conn, *, event_id: str, value: str, confidence: float, normalizer_version: str,
    model_id: str, raw_sha256: str,
) -> None:
    """Append one normaliser verdict into `normalized_decline`. ON CONFLICT
    (event_id, normalizer_version) DO NOTHING, mirroring
    record_ingested_event's dedup discipline: a retried write under the
    SAME prompt version is silently a no-op. A write under a NEW
    normalizer_version is a different row entirely (see schema.sql's
    comment on this table) -- never an overwrite of the old one.

    confidence is required, not optional: a verdict written without it
    cannot later be disputed ("why did the model say this?"), which is the
    same auditability gap this whole table exists to close for the
    verdict itself -- see schema.sql's comment on the column."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO normalized_decline (
                event_id, value, confidence, normalizer_version, model_id, raw_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id, normalizer_version) DO NOTHING
            """,
            (event_id, value, confidence, normalizer_version, model_id, raw_sha256),
        )


def find_normalized_decline(
    conn, event_id: str, normalizer_version: str,
) -> NormalizedDeclineRow | None:
    """Exact (event_id, normalizer_version) lookup -- the read-back path
    src/policy/belief.update()'s required source_version must come from:
    never trust an in-memory NormalizedDecline directly, always round-trip
    it through the ledger first (PLAN_DETAIL.md B11 gate clause 3). None if
    no row exists for this event under this normaliser version."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT event_id, value, confidence, normalizer_version, model_id,
                   raw_sha256, created_at
            FROM normalized_decline
            WHERE event_id = %s AND normalizer_version = %s
            """,
            (event_id, normalizer_version),
        )
        row = cur.fetchone()
    return NormalizedDeclineRow(*row) if row is not None else None


def append_shadow(
    conn, *, run_id: str, mandate_id: str, cycle_id: int, profile: str,
    ladder_action: str, ladder_slot: int, ladder_day: int,
    ladder_committed_attempts: int,
    our_action: str, our_slot: int | None, our_day: int | None,
    binding_constraint: str | None, conformal_set: str, belief_json: str,
    decision_sha256: str, divergence: str, agrees: bool,
) -> None:
    """Append one shadow-mode decision into `shadow_ledger` (B12).

    A SEPARATE function writing a SEPARATE table rather than a mode flag on
    append(): shadow decisions never executed, and the one thing that must
    stay impossible is a query counting them as money. Same precedent as
    record_normalized_decline above -- its own table, its own writer, rather
    than generalising append() into something that can address two tables.

    Deliberately NOT idempotent-by-conflict on rerun: the PK is
    (run_id, mandate_id, cycle_id), and a new run means a new run_id, so a
    repeated run records a new comparable observation instead of silently
    overwriting the previous one. A genuine duplicate WITHIN one run is a
    bug in the caller's batch, so it raises rather than being swallowed --
    unlike the ledger, where a retried write under the same idempotency key
    is a legitimate crash-recovery path.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO shadow_ledger (
                run_id, mandate_id, cycle_id, profile,
                ladder_action, ladder_slot, ladder_day, ladder_committed_attempts,
                our_action, our_slot, our_day,
                binding_constraint, conformal_set, belief_json,
                decision_sha256, divergence, agrees
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id, mandate_id, cycle_id, profile,
                ladder_action, ladder_slot, ladder_day, ladder_committed_attempts,
                our_action, our_slot, our_day,
                binding_constraint, conformal_set, belief_json,
                decision_sha256, divergence, agrees,
            ),
        )
