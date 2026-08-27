"""FastAPI dependencies for src/ingest/. Kept in its own module, separate
from both app.py and webhook.py, so neither has to import the other to
reach it -- app.py mounts the router, webhook.py's handler depends on
get_conn, and both need this file rather than each other.
"""
from __future__ import annotations

from src.core.db import connect


def get_conn():
    """One connection per request, closed when the request finishes.
    autocommit=True: every write in this module is already a single,
    already-atomic INSERT ... ON CONFLICT DO NOTHING (see dedupe.py,
    lifecycle_route.py, store.py's record_* functions) -- there is no
    multi-statement transaction to wrap, and matches how the pg_schema
    test fixture itself opens its connection. Without this, a write
    commits nothing: psycopg3 defaults to autocommit=False, so closing
    the connection at the end of the request would silently discard
    everything the request just wrote.

    Tests override this dependency (`app.dependency_overrides[get_conn]`)
    to hand back an already-open, schema-scoped connection instead --
    see tests/ingest/test_webhook.py's `client` fixture."""
    conn = connect(autocommit=True)
    try:
        yield conn
    finally:
        conn.close()
