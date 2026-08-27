"""The single place a Postgres DSN is resolved and a raw connection opened.
Every other module -- application code and test fixtures alike -- goes
through this file rather than assembling its own connection string, so
there is exactly one definition of "how do we reach the database."

This module does not load .env itself. Loading environment files is an
entrypoint's job (see tests/conftest.py, src/ingest/app.py, and run.ps1's
`verify` probe, each of which already does this) -- a library module
reaching into the process environment as an import-time side effect is the
kind of implicit behaviour this project avoids elsewhere.
"""
from __future__ import annotations

import os

import psycopg

DEFAULT_DSN = "postgresql://postgres:dev@localhost:5432/mandate_recovery"


def dsn() -> str:
    """The DSN to connect with: DATABASE_URL from the environment if set,
    otherwise DEFAULT_DSN (the docker container `mrdb`'s default)."""
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def connect(**kwargs) -> psycopg.Connection:
    """Open a new connection to dsn(). Callers own the connection's
    lifecycle -- closing it, autocommit, isolation level -- this only
    resolves which database to talk to."""
    return psycopg.connect(dsn(), **kwargs)
