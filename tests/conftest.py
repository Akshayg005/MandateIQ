"""
Root conftest.

`python -m pytest` (what `.\\run.ps1 test` runs) already puts the repo root
on sys.path because of how `-m` works, so `from src.core.money import ...`
resolves without this. This is here defensively, for the plain `pytest`
console-script invocation, which does not always do that -- so imports keep
working regardless of which entry point is used to run the suite.

Also hosts the pg_schema / seed_plan fixtures, promoted here from
tests/ledger/conftest.py (B3) so tests/ingest/ and tests/classify/ pick them
up through normal pytest fixture resolution -- a conftest.py in tests/ledger/
is invisible to sibling test packages. Behaviour is unchanged from the
original: same scratch-schema naming, same skip-vs-raise split (see the
pg_schema docstring below).
"""
from __future__ import annotations

import pathlib
import sys
import uuid
from dataclasses import dataclass

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# src.core.db resolves *which* database to talk to; it must be imported only
# after the sys.path fix-up above has actually run.
from src.core.db import dsn

try:
    import psycopg
except ImportError:  # pragma: no cover - environment guard
    psycopg = None

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover - environment guard
    pass

SCHEMA_PATH = _ROOT / "src" / "ledger" / "schema.sql"


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

    Two distinct failure modes, not to be confused:
    - Postgres itself is unreachable (Docker not running, wrong
      DATABASE_URL) -> pytest.skip. An environment problem, not a code one.
    - schema.sql is missing or broken -> let it raise. That is the thing
      under test.
    """
    if psycopg is None:
        pytest.skip("psycopg is not installed")

    try:
        conn = psycopg.connect(dsn(), autocommit=True, connect_timeout=3)
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
    planner/allocator (B8), which doesn't exist yet either.
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
