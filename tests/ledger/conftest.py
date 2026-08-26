"""
Shared fixtures for tests/ledger/*.

Every test in this package needs a live Postgres to apply
src/ledger/schema.sql into an isolated scratch schema, and to tear that
schema down afterward -- never touching `public`, never assuming a clean
database.

Two distinct failure modes matter here, and they must NOT be confused:

- Postgres itself is unreachable (Docker not running, wrong DATABASE_URL,
  etc.) -> pytest.skip. This is an environment problem, not a code problem.
- schema.sql is missing or broken -> let it raise. That is the thing under
  test. Right now, before B1 is implemented, every test that uses the
  pg_schema fixture fails with FileNotFoundError -- that is the correct,
  expected failure, not a skip.
"""
from __future__ import annotations

import os
import pathlib
import uuid
from dataclasses import dataclass

import pytest

try:
    import psycopg
except ImportError:  # pragma: no cover - environment guard
    psycopg = None

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover - environment guard
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT / "src" / "ledger" / "schema.sql"

DEFAULT_DSN = "postgresql://postgres:dev@localhost:5432/mandate_recovery"


def _dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


@dataclass
class PgScratchSchema:
    conn: "psycopg.Connection"
    schema: str


@pytest.fixture
def pg_schema():
    """
    Creates a throwaway schema, applies schema.sql into it with search_path
    pointed there, yields a PgScratchSchema(conn, schema_name), and drops
    the schema afterward regardless of test outcome.
    """
    if psycopg is None:
        pytest.skip("psycopg is not installed")

    try:
        conn = psycopg.connect(_dsn(), autocommit=True, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")

    schema_name = f"test_b1_{uuid.uuid4().hex[:16]}"
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema_name}"')
            cur.execute(f'SET search_path TO "{schema_name}"')

        # Deliberately NOT wrapped in try/except: a missing or broken
        # schema.sql must fail this fixture loudly, not skip.
        sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql_text)

        yield PgScratchSchema(conn=conn, schema=schema_name)
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        finally:
            conn.close()


@pytest.fixture
def seed_plan(pg_schema):
    """
    Factory fixture: inserts a minimal valid `plan` row (satisfying every
    NOT NULL column) and returns its decision_sha256, so ledger-row tests
    can satisfy ledger.decision_sha256's FK without depending on the real
    planner/allocator, which doesn't exist yet either.
    """
    conn = pg_schema.conn

    def _make(
        decision_sha256: str,
        mandate_id: str = "M-TEST",
        cycle_id: int = 1,
        profile: str = "strict",
    ) -> str:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO plan (
                    decision_sha256, mandate_id, cycle_id, profile,
                    belief_json, conformal_set, binding_constraint, solver_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (decision_sha256, mandate_id, cycle_id, profile, "{}", "{}", None, "test-solver-v0"),
            )
        return decision_sha256

    return _make
