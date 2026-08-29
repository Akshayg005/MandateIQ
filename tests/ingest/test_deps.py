"""src/ingest/deps.py -- FastAPI DB dependency.

One test, guarding the exact gap that let POSTMORTEM.md Incident 2 ship:
every other test in tests/ingest/ overrides get_conn() via
app.dependency_overrides (see test_webhook.py's `client` fixture), pointed
at pg_schema.conn -- which is opened with autocommit=True directly in
tests/conftest.py, for an unrelated reason (so a test's own verification
queries see its writes immediately). That override is correct and
necessary for test isolation, but as a side effect nothing automated ever
ran the REAL get_conn() generator -- so when it opened a connection
without autocommit=True, every write in production silently vanished on
connection close (psycopg3 defaults to autocommit=False; nothing else in
this module ever calls .commit()), while returning HTTP 200. Caught only
by a manual end-to-end run against a live server and the real database.

This test exercises the real generator, deliberately not overridden,
which is the only way to actually guard against a repeat.
"""
from __future__ import annotations

import pytest

from src.ingest.deps import get_conn


def test_get_conn_yields_an_autocommit_connection(_pg_reachable):
    """autocommit=True is required, not optional: every write reachable
    through this dependency (dedupe.mark, record_ingested_event,
    record_lifecycle_event) is already a single, atomic
    INSERT ... ON CONFLICT DO NOTHING -- there is no multi-statement
    transaction for a manual commit to wrap, and without autocommit,
    closing the connection at the end of a request discards the write
    instead of persisting it.

    Does not use the pg_schema fixture (deliberately -- see module
    docstring), so it consults the session-cached _pg_reachable directly
    to skip fast rather than paying its own ~6s connect attempt against an
    already-known-down Postgres (see conftest.py, DECISIONS.md 2026-08-29)."""
    reachable, reason = _pg_reachable
    if not reachable:
        pytest.skip(f"Postgres unavailable: {reason}")

    gen = get_conn()
    try:
        conn = next(gen)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")

    try:
        assert conn.autocommit is True
    finally:
        try:
            next(gen)
        except StopIteration:
            pass  # expected: the generator closes conn and returns
