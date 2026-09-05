"""
The suite must not report success while the money-critical surface was
never run.

With Docker down, every Postgres-backed test skipped: ledger, executor,
lease, void, recover, commit, webhook, dedupe, chaos -- the entire
idempotency and crash-recovery surface -- and `pytest` still exited 0.
CLAUDE.md's definition-of-done step 3 ("`.\run.ps1 test` passes before any
commit") was therefore satisfiable without exercising any of it. That is the
same class of defect as the Invoke-Step bug in POSTMORTEM: a check that
passes by not checking.

So the default is now to FAIL, loudly, per test. Skipping is still
available, but only as a deliberate, named act: set the opt-out env var.
These tests pin that policy, since the policy is the whole point.
"""
from __future__ import annotations

import pytest

from conftest import PG_SKIP_OPT_OUT, pg_skip_allowed, require_pg


class TestOptOutParsing:
    def test_unset_means_not_allowed(self) -> None:
        assert pg_skip_allowed({}) is False

    def test_empty_means_not_allowed(self) -> None:
        """Set-but-empty is how a shell hands over an unset variable. It must
        not read as consent."""
        assert pg_skip_allowed({PG_SKIP_OPT_OUT: ""}) is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " 1 "])
    def test_truthy_values_allow_the_skip(self, value: str) -> None:
        assert pg_skip_allowed({PG_SKIP_OPT_OUT: value}) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "maybe"])
    def test_falsy_values_do_not(self, value: str) -> None:
        """Anything that is not affirmative is refused, rather than being
        treated as "set, therefore true". `FOO=0` must never enable an
        opt-out."""
        assert pg_skip_allowed({PG_SKIP_OPT_OUT: value}) is False

    def test_reads_the_real_environment_when_no_mapping_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(PG_SKIP_OPT_OUT, raising=False)
        assert pg_skip_allowed() is False
        monkeypatch.setenv(PG_SKIP_OPT_OUT, "1")
        assert pg_skip_allowed() is True


class TestRequirePg:
    def test_reachable_postgres_is_a_no_op(self) -> None:
        require_pg(True, "", env={})

    def test_unreachable_postgres_fails_by_default(self) -> None:
        with pytest.raises(pytest.fail.Exception) as exc:
            require_pg(False, "connection timeout expired", env={})
        assert "connection timeout expired" in str(exc.value)

    def test_the_failure_says_how_to_fix_it(self) -> None:
        """A red suite that does not name the cure just gets the guard
        deleted by whoever hits it next."""
        with pytest.raises(pytest.fail.Exception) as exc:
            require_pg(False, "boom", env={})
        message = str(exc.value)
        # R7: BOTH runners, because a Linux reviewer told to run
        # `.\run.ps1 up` has been handed a translation task, which is
        # exactly what R7's gate forbids. Asserting both is what stops the
        # POSIX half being dropped later without the suite noticing.
        assert "run.ps1 up" in message
        assert "run.sh db-up" in message
        assert "docker start mrdb" in message
        assert PG_SKIP_OPT_OUT in message

    def test_unreachable_postgres_skips_under_the_opt_out(self) -> None:
        with pytest.raises(pytest.skip.Exception) as exc:
            require_pg(False, "boom", env={PG_SKIP_OPT_OUT: "1"})
        assert "boom" in str(exc.value)

    def test_the_skip_names_the_opt_out_that_produced_it(self) -> None:
        """`pytest -rs` output must say the skip was chosen, not inherent --
        otherwise the opt-out becomes invisible in CI logs."""
        with pytest.raises(pytest.skip.Exception) as exc:
            require_pg(False, "boom", env={PG_SKIP_OPT_OUT: "yes"})
        assert PG_SKIP_OPT_OUT in str(exc.value)


def test_no_postgres_test_still_calls_pytest_skip_directly() -> None:
    """Every Postgres-availability skip must route through require_pg.

    A second, hand-rolled `pytest.skip("Postgres unavailable: ...")` is
    exactly how this hole reopens: the guard gets added in one place and the
    other site keeps quietly skipping.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    this_file = pathlib.Path(__file__).resolve()
    offenders: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        if path.resolve() == this_file:
            continue  # this file quotes the banned pattern in order to ban it
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "pytest.skip(" in line and "Postgres" in line:
                offenders.append(f"{path.relative_to(root)}:{lineno}")
    assert offenders == [], (
        "these sites skip on Postgres availability without going through "
        f"require_pg(): {offenders}"
    )
