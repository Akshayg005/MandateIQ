"""src/execute/lease.py -- crash-safe lease claiming over attempt_lease.

Every test freezes src.core.clock so lease expiry is proven by advancing a
frozen clock, never by a real sleep() -- DESIGN.md's clock discipline
applies here exactly as it does everywhere else in the codebase.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import clock
from src.execute.lease import claim, expired, release

FROZEN = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


def test_claim_succeeds_on_a_fresh_key(pg_schema):
    clock.set_frozen(FROZEN)
    assert claim(pg_schema.conn, "key-1", owner="worker-a", ttl_seconds=300) is True


def test_second_owner_cannot_claim_a_live_lease(pg_schema):
    clock.set_frozen(FROZEN)
    assert claim(pg_schema.conn, "key-2", owner="worker-a", ttl_seconds=300) is True
    assert claim(pg_schema.conn, "key-2", owner="worker-b", ttl_seconds=300) is False


def test_second_owner_can_claim_after_expiry(pg_schema):
    """The core crash-safety property: a worker that died mid-attempt must
    not permanently block that key. Advancing the frozen clock past the
    TTL is what proves expiry, not a real sleep."""
    clock.set_frozen(FROZEN)
    assert claim(pg_schema.conn, "key-3", owner="worker-a", ttl_seconds=60) is True

    clock.set_frozen(FROZEN + timedelta(seconds=61))
    assert claim(pg_schema.conn, "key-3", owner="worker-b", ttl_seconds=300) is True


def test_second_owner_cannot_claim_one_second_before_expiry(pg_schema):
    clock.set_frozen(FROZEN)
    assert claim(pg_schema.conn, "key-4", owner="worker-a", ttl_seconds=60) is True

    clock.set_frozen(FROZEN + timedelta(seconds=59))
    assert claim(pg_schema.conn, "key-4", owner="worker-b", ttl_seconds=300) is False


def test_release_frees_the_key_immediately_regardless_of_ttl(pg_schema):
    clock.set_frozen(FROZEN)
    assert claim(pg_schema.conn, "key-5", owner="worker-a", ttl_seconds=3600) is True

    release(pg_schema.conn, "key-5")

    # Still well inside the original TTL window -- only release(), not
    # expiry, is what frees this.
    clock.set_frozen(FROZEN + timedelta(seconds=1))
    assert claim(pg_schema.conn, "key-5", owner="worker-b", ttl_seconds=300) is True


def test_release_on_a_never_claimed_key_is_a_silent_no_op(pg_schema):
    release(pg_schema.conn, "key-never-claimed")  # must not raise


def test_expired_lists_only_leases_past_their_ttl(pg_schema):
    clock.set_frozen(FROZEN)
    claim(pg_schema.conn, "key-short", owner="worker-a", ttl_seconds=10)
    claim(pg_schema.conn, "key-long", owner="worker-a", ttl_seconds=10_000)

    clock.set_frozen(FROZEN + timedelta(seconds=20))
    result = expired(pg_schema.conn)

    assert "key-short" in result
    assert "key-long" not in result


def test_expired_is_empty_when_nothing_has_been_claimed(pg_schema):
    clock.set_frozen(FROZEN)
    assert expired(pg_schema.conn) == []
