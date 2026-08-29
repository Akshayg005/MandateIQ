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

Also registers the `chaos` and `slow` markers `run.ps1`'s $TestFastFilter
excludes from the default (`test-fast`, `ci`) path -- registered here rather
than left implicit so an unmarked typo raises `--strict-markers`-style
attention instead of silently matching nothing. `slow`: a real simulation of
meaningful size runs inside the test (not a fixture query, not a unit
computation) -- see DECISIONS.md, 2026-08-29, for which tests earned it and
why. `chaos`: reserved for B10's induced-kill tests, none exist yet.
"""
from __future__ import annotations

import pathlib
import sys
import uuid
from dataclasses import dataclass

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: real simulation of meaningful size, excluded from test-fast/ci"
    )
    config.addinivalue_line(
        "markers", "chaos: induced-kill fault injection (B10), excluded from test-fast/ci"
    )

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


@pytest.fixture(scope="session")
def _pg_reachable() -> tuple[bool, str]:
    """Whether Postgres is reachable at all, checked ONCE per test session
    -- pytest caches a session-scoped fixture's return value automatically,
    so no manual cache is needed here.

    Why this exists: `pg_schema` below is function-scoped and used by 61
    test items across 6 files; each independently made its own fresh
    `psycopg.connect()` attempt, and on this machine `localhost` resolves
    to both `::1` and `127.0.0.1` -- psycopg/libpq tries each address in
    turn, each with the full `connect_timeout`, so a single "Postgres is
    down" attempt costs ~2x connect_timeout (measured: 6.03-6.06s), summing
    to ~370s across all 61 items (measured directly, DECISIONS.md,
    2026-08-29). This fixture pays that cost at most once per session.

    Deliberately narrow: only the negative "is anything even listening"
    case is cached. `pg_schema` still opens its OWN connection when
    Postgres IS reachable -- every test needs an isolated schema-scoped
    connection for real work, and this fixture does not attempt to share
    or pool live connections across tests. That would be the fixture-
    architecture change this one-fixture addition is deliberately not.
    """
    if psycopg is None:
        return False, "psycopg is not installed"
    try:
        probe = psycopg.connect(dsn(), connect_timeout=3)
        probe.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


@pytest.fixture
def pg_schema(_pg_reachable):
    """
    Creates a throwaway schema, applies schema.sql into it with search_path
    pointed there, yields a PgScratchSchema(conn, schema_name), and drops
    the schema afterward regardless of test outcome.

    Two distinct failure modes, not to be confused:
    - Postgres itself is unreachable (Docker not running, wrong
      DATABASE_URL) -> pytest.skip. An environment problem, not a code one.
      Checked via the session-cached `_pg_reachable` first, so 60 of 61
      Postgres-dependent tests skip immediately rather than each re-probing
      an already-known-down Postgres.
    - schema.sql is missing or broken -> let it raise. That is the thing
      under test.
    """
    reachable, reason = _pg_reachable
    if not reachable:
        pytest.skip(f"Postgres unavailable: {reason}")

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
