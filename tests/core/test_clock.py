"""
src/core/clock.py -- the only source of "now" in the codebase. Must be
freezable, because the 24-hour commitment lag (RBI clause 6(a)) is
untestable against a live clock.

The isolation fixture is kept INLINE (not in tests/core/conftest.py) on
purpose: a shared conftest importing src.core.clock would fail collection
for every file in this directory the moment clock.py is missing, which
would hide test_money.py / test_ids.py / test_types.py behind the wrong
error. Each file should fail for its own reason.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from freezegun import freeze_time

from src.core.clock import commit_deadline, now, set_frozen

FROZEN = datetime(2026, 3, 15, 9, 30, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    """A clock left frozen by one test must never leak into the next."""
    yield
    set_frozen(None)


# --- freezing --------------------------------------------------------------

def test_frozen_now_returns_exact_value_repeatedly():
    set_frozen(FROZEN)
    assert now() == FROZEN
    assert now() == FROZEN
    assert now() == FROZEN


def test_frozen_now_does_not_drift_with_real_wall_clock():
    set_frozen(FROZEN)
    first = now()
    time.sleep(0.05)
    second = now()
    assert first == second == FROZEN


def test_now_is_tzaware_when_frozen():
    set_frozen(FROZEN)
    assert now().tzinfo is not None


# --- unfreezing --------------------------------------------------------------

def test_set_frozen_none_unfreezes_back_to_real_time():
    set_frozen(datetime(2020, 1, 1, tzinfo=timezone.utc))
    set_frozen(None)

    got = now()
    real = datetime.now(timezone.utc)

    assert got != datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert abs((got - real).total_seconds()) < 5


def test_unfrozen_now_tracks_real_time_across_successive_calls():
    a = now()
    time.sleep(0.05)
    b = now()
    assert b >= a
    assert abs((b - datetime.now(timezone.utc)).total_seconds()) < 5


def test_now_is_tzaware_when_unfrozen():
    assert now().tzinfo is not None


def test_unfrozen_now_reflects_freezegun_intercepted_system_clock():
    """now() must delegate to the real datetime.now() under the hood when
    nothing has been frozen via set_frozen -- freezegun intercepts
    datetime.now(), so this also rules out clock.py reading time via
    time.time() or some other source freezegun would not touch."""
    with freeze_time("2030-01-01 00:00:00+00:00"):
        got = now()
    assert (got.year, got.month, got.day) == (2030, 1, 1)


# --- commit_deadline: RBI clause 6(a), >=24h lead time ----------------------

def test_commit_deadline_default_is_exactly_24_hours_before_target():
    target = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert commit_deadline(target) == target - timedelta(hours=24)


def test_commit_deadline_respects_custom_lead_hours():
    target = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert commit_deadline(target, lead_hours=48) == target - timedelta(hours=48)
    assert commit_deadline(target, lead_hours=1) == target - timedelta(hours=1)


def test_commit_deadline_is_a_pure_function_of_its_arguments():
    target = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert commit_deadline(target) == commit_deadline(target)
    set_frozen(datetime(2099, 1, 1, tzinfo=timezone.utc))
    assert commit_deadline(target) == target - timedelta(hours=24)
