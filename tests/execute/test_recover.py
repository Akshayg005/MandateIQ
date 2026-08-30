"""src/execute/recover.py -- dangling-INTENT reconciliation and the
UNCONFIRMED -> UNRESOLVED_FINAL backoff walk. The B9 gate's third clause:
UNCONFIRMED must have a resolution path that is ACTUALLY REACHABLE -- these
tests drive a key all the way to UNRESOLVED_FINAL, not merely assert the
constant exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import clock
from src.core.types import Action, MandateState, Profile
from src.execute.commit import commit
from src.execute.lease import claim as lease_claim
from src.execute.recover import UNCONFIRMED, UNRESOLVED_FINAL, reconcile
from src.ledger.store import LedgerEntry, append, record_lifecycle_event, replay
from src.policy.allocator import CommittedAttempt, Plan

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


class _FakeClient:
    """Records every find_by_receipt call; raises if anything else is
    called -- recover.py must NEVER charge, create_order, or pause."""

    def __init__(self, found_by_key: dict | None = None):
        self.calls: list[str] = []
        self._found_by_key = found_by_key or {}

    def find_by_receipt(self, receipt):
        self.calls.append(receipt)
        return self._found_by_key.get(receipt)

    def create_order(self, **kwargs):
        raise AssertionError("recover.py must never call create_order()")

    def charge(self, **kwargs):
        raise AssertionError("recover.py must never call charge()")

    def pause_subscription(self, *args, **kwargs):
        raise AssertionError("recover.py must never call pause_subscription()")


def _plan(*, decision_sha256, mandate_id="M-RECOVER-1", on_day=2, amount_paise=50_000):
    return Plan(
        mandate_id=mandate_id, cycle_id=1, profile=Profile.strict,
        chosen_action=Action.ATTEMPT,
        committed=(CommittedAttempt(slot=1, on_day=on_day, amount_paise=amount_paise),),
        belief_json="{}", conformal_set=frozenset(), binding_constraint=None,
        solver_version="test-solver-v0", decision_sha256=decision_sha256,
    )


def _committed(pg_schema, *, decision_sha256, mandate_id="M-RECOVER-1", committed_at):
    clock.set_frozen(committed_at)
    record_lifecycle_event(
        pg_schema.conn, event_id=f"evt-created-{mandate_id}-{decision_sha256}",
        mandate_id=mandate_id, state=MandateState.ACTIVE.value, source="INTERNAL",
        effective_at=committed_at - timedelta(days=1),
    )
    plan = _plan(decision_sha256=decision_sha256, mandate_id=mandate_id)
    return commit(pg_schema.conn, plan, cycle_start=CYCLE_START)


def _make_dangling(pg_schema, attempt, *, state="INTENT", ttl_seconds=60):
    """Simulates a worker that wrote INTENT (optionally SENT), claimed a
    lease, then crashed before ever reaching a terminal row. Advances the
    frozen clock past the lease TTL so it becomes discoverable."""
    append(pg_schema.conn, LedgerEntry(
        idempotency_key=attempt.idempotency_key, mandate_id=attempt.mandate_id,
        cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
        action=attempt.action, state="INTENT", amount_paise=attempt.amount_paise,
        profile=attempt.profile, payload_sha256="0" * 64,
        decision_sha256=attempt.decision_sha256,
    ))
    if state == "SENT":
        append(pg_schema.conn, LedgerEntry(
            idempotency_key=attempt.idempotency_key, mandate_id=attempt.mandate_id,
            cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
            action=attempt.action, state="SENT", amount_paise=attempt.amount_paise,
            profile=attempt.profile, payload_sha256="0" * 64,
            decision_sha256=attempt.decision_sha256,
        ))
    assert lease_claim(pg_schema.conn, attempt.idempotency_key, owner="crashed-worker", ttl_seconds=ttl_seconds)
    clock.set_frozen(clock.now() + timedelta(seconds=ttl_seconds + 1))


# === dangling: found by the provider =======================================

def test_reconcile_resolves_a_dangling_intent_when_the_provider_confirms_it(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-recover-found", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={attempt.idempotency_key: {"id": "pay_found", "status": "captured"}})
    results = reconcile(pg_schema.conn, client)

    assert len(results) == 1
    assert results[0].state == "RESULT"
    assert results[0].outcome == "RECOVERED"
    assert results[0].provider_ref == "pay_found"
    assert client.calls == [attempt.idempotency_key]


def test_reconcile_resolves_a_dangling_sent_row_the_same_way(pg_schema):
    """A crash after SENT (the actual crash window PLAN_DETAIL.md section 3
    names) must be discoverable exactly like a crash after INTENT alone."""
    attempt = _committed(pg_schema, decision_sha256="d-recover-sent", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt, state="SENT")

    client = _FakeClient(found_by_key={attempt.idempotency_key: {"id": "pay_sent", "status": "captured"}})
    results = reconcile(pg_schema.conn, client)

    assert len(results) == 1
    assert results[0].state == "RESULT"


def test_reconcile_records_pending_when_found_but_status_unrecognised(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-recover-pending", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={attempt.idempotency_key: {"id": "pay_p", "status": "created"}})
    results = reconcile(pg_schema.conn, client)

    assert results[0].state == "RESULT"
    assert results[0].outcome is None
    assert results[0].reason == "PENDING_WEBHOOK_CONFIRMATION"


# === dangling: NOT found -> UNCONFIRMED, never a blind resend ==============

def test_reconcile_marks_unconfirmed_when_the_provider_has_no_record(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-recover-notfound", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={})  # provider has no record at all
    results = reconcile(pg_schema.conn, client)

    assert results[0].state == "FAILED"
    assert results[0].reason == UNCONFIRMED
    # The slot stays consumed -- proven by the row existing, never deleted
    # or rewritten; attempt_index is untouched (recover.py never mutates it).
    rows = replay(pg_schema.conn, attempt.mandate_id)
    assert rows[-1].reason == UNCONFIRMED


def test_reconcile_never_blind_resends(pg_schema):
    """The B3 spike's own conclusion: recovery is by ASKING, never by
    resending. _FakeClient raises if charge()/create_order() is ever
    called -- this test just confirms reconcile() completes without
    tripping that guard."""
    attempt = _committed(pg_schema, decision_sha256="d-recover-noresend", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)
    client = _FakeClient(found_by_key={})
    reconcile(pg_schema.conn, client)  # must not raise


# === a live lease is left alone =============================================

def test_reconcile_ignores_a_key_whose_lease_has_not_expired(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-recover-live-lease", committed_at=CYCLE_START)
    append(pg_schema.conn, LedgerEntry(
        idempotency_key=attempt.idempotency_key, mandate_id=attempt.mandate_id,
        cycle_id=attempt.cycle_id, attempt_index=attempt.attempt_index,
        action=attempt.action, state="INTENT", amount_paise=attempt.amount_paise,
        profile=attempt.profile, payload_sha256="0" * 64,
        decision_sha256=attempt.decision_sha256,
    ))
    assert lease_claim(pg_schema.conn, attempt.idempotency_key, owner="still-working", ttl_seconds=3600)
    # Clock NOT advanced -- the lease is still live.

    client = _FakeClient()
    results = reconcile(pg_schema.conn, client)

    assert results == []
    assert client.calls == []


# === the backoff walk: proven reachable, not merely asserted ==============

def test_unconfirmed_reaches_unresolved_final_after_max_passes(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-backoff-walk", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={})  # never found, every pass

    # Pass 1: dangling -> first UNCONFIRMED.
    r1 = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)
    assert r1[0].reason == UNCONFIRMED

    # Passes 2 and 3: now discovered via the STUCK path, still under the cap.
    r2 = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)
    assert r2[0].reason == UNCONFIRMED
    r3 = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)
    assert r3[0].reason == UNCONFIRMED

    # Pass 4: the cap (3 prior UNCONFIRMED rows) is reached -> UNRESOLVED_FINAL.
    r4 = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)
    assert r4[0].state == "FAILED"
    assert r4[0].reason == UNRESOLVED_FINAL


def test_unresolved_final_is_terminal_and_never_reprocessed(pg_schema):
    attempt = _committed(pg_schema, decision_sha256="d-terminal", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)
    client = _FakeClient(found_by_key={})

    for _ in range(4):
        reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)

    rows = replay(pg_schema.conn, attempt.mandate_id)
    assert rows[-1].reason == UNRESOLVED_FINAL

    calls_before = len(client.calls)
    again = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)
    assert again == [], "a key already at UNRESOLVED_FINAL must never be reprocessed"
    assert len(client.calls) == calls_before, "the provider must not be queried again"


def test_unresolved_final_is_reported_never_silently_dropped(pg_schema):
    """The gate's exact words: UNRESOLVED_FINAL is a reported metric. This
    pins that reconcile() RETURNS it in its result list, rather than
    swallowing it -- a caller building a report has something to count."""
    attempt = _committed(pg_schema, decision_sha256="d-reported", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)
    client = _FakeClient(found_by_key={})

    results = []
    for _ in range(4):
        results = reconcile(pg_schema.conn, client, max_unconfirmed_passes=3)

    assert any(r.reason == UNRESOLVED_FINAL for r in results)


def test_a_late_confirmation_during_the_backoff_walk_still_resolves(pg_schema):
    """The provider can start answering mid-walk -- recover.py must notice
    on the very next pass, not only on the first or the last."""
    attempt = _committed(pg_schema, decision_sha256="d-late-confirm", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={})
    reconcile(pg_schema.conn, client, max_unconfirmed_passes=5)  # pass 1: still nothing

    client._found_by_key[attempt.idempotency_key] = {"id": "pay_late", "status": "captured"}
    results = reconcile(pg_schema.conn, client, max_unconfirmed_passes=5)  # pass 2: found

    assert results[0].state == "RESULT"
    assert results[0].outcome == "RECOVERED"


# === the same-call double-processing fix ===================================

def test_a_freshly_dangling_key_is_not_double_processed_in_one_call(pg_schema):
    """Regression guard: a key resolved to FAILED/UNCONFIRMED by the
    dangling pass must not ALSO be picked up by the stuck pass within the
    very same reconcile() call (writes are autocommit, so it would
    otherwise already match the stuck query) -- that would double-count
    one backoff pass as two and query the provider twice in one call."""
    attempt = _committed(pg_schema, decision_sha256="d-no-double-process", committed_at=CYCLE_START)
    _make_dangling(pg_schema, attempt)

    client = _FakeClient(found_by_key={})
    results = reconcile(pg_schema.conn, client)

    assert len(results) == 1
    assert client.calls == [attempt.idempotency_key]  # exactly one ask, not two

    rows = replay(pg_schema.conn, attempt.mandate_id)
    unconfirmed_rows = [r for r in rows if r.reason == UNCONFIRMED]
    assert len(unconfirmed_rows) == 1
