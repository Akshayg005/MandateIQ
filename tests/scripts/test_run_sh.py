"""run.sh -- the POSIX task runner (R7).

The one property that matters here is the one run.ps1 had to learn the hard
way: **a task must exit NON-ZERO when the thing it ran failed.** run.ps1's
`Invoke-Step` exists because a bare call as a switch branch's last statement
returned 0 on a red test suite, which made DESIGN.md's own definition-of-done
step 3 ("`.\\run.ps1 test` passes before any commit") unfalsifiable for
several blocks (run.ps1:157-165; reports/gates.md:597-600). A second runner
that quietly reproduced that bug would make the same claim unfalsifiable on
Linux, where CI is the only thing checking it.

So this file does not test that `step()` is spelled a particular way. It
drives run.sh with a STUB interpreter (via the MANDATEIQ_PY override run.sh
documents) that exits 1, and asserts the runner exits non-zero -- and with a
stub that exits 0, and asserts it exits zero, so the first assertion cannot
pass merely because everything fails.

Skipped where bash is unavailable. That is a real gap on a bare Windows box
without Git Bash, and it is why R7's actual evidence is a green CI run on
ubuntu-latest, not this file alone.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_SH = ROOT / "run.sh"


def _find_bash() -> str | None:
    """Resolve a bash that actually inherits this process's environment.

    CORRECTED, 2026-09-05: the first version of this used bare
    `shutil.which("bash")`, on the theory that it "finds Git Bash's msys
    binary" -- true from a shell whose own PATH happens to put Git's bin
    directories first, false otherwise. Caught empirically, not by
    reasoning about it in advance: this file passed running from this
    session's own Bash tool (a git-bash shell, PATH order favours Git) and
    FAILED running via `.\\run.ps1 test-fast` (a PowerShell child process,
    where `shutil.which` on this machine resolves to
    `C:\\Windows\\System32\\bash.exe` -- the WSL LAUNCHER, a different
    machine with a different filesystem that does not inherit the Win32
    environment; only WSLENV-listed variables cross). Every test here
    failed there with "venv not found at .venv/bin/python", because
    MANDATEIQ_PY and the repo path both failed to cross into WSL.

    So: check Git Bash's own standard install locations FIRST, in a fixed
    order that does not depend on which shell launched this test process.
    Fall back to `shutil.which()` only if none exist, and explicitly
    reject a System32/SysWOW64 hit even then -- that directory is exactly
    where the WSL launcher lives, never a bash that shares this process's
    environment. On Linux/macOS there is only one bash and this reduces to
    `shutil.which("bash")` on the first candidate check.
    """
    candidates: list[pathlib.Path] = []
    for env_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(pathlib.Path(base) / "Git" / "bin" / "bash.exe")
            candidates.append(pathlib.Path(base) / "Git" / "usr" / "bin" / "bash.exe")
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which("bash")
    if found and "system32" not in found.lower() and "syswow64" not in found.lower():
        return found
    return None


BASH = _find_bash()

pytestmark = pytest.mark.skipif(
    BASH is None, reason="no bash found that inherits this process's environment"
)


def _stub(tmp_path: pathlib.Path, exit_code: int) -> pathlib.Path:
    """A fake `python` that ignores its arguments and exits `exit_code`."""
    p = tmp_path / "fakepy"
    p.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8", newline="\n")
    p.chmod(0o755)
    return p


def _posix(p: pathlib.Path) -> str:
    r"""Git Bash on Windows treats a backslash as an escape, so a native
    `C:\dev\...` path arrives as `C:dev...` and every invocation fails with
    exit 127. Forward slashes work on both platforms."""
    return str(p).replace("\\", "/")


def _run(task: str, py: pathlib.Path, *args: str):
    env = dict(os.environ, MANDATEIQ_PY=_posix(py))
    return subprocess.run(
        # Relative, with cwd=ROOT: run.sh cd's to its own directory anyway,
        # and a relative path sidesteps the escaping trap above entirely.
        [BASH, "./run.sh", task, *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_the_shell_scripts_have_lf_line_endings():
    """A CRLF `run.sh` is not a style nit: bash reads the trailing CR as
    part of the option name and dies with "set: pipefail<CR>: invalid
    option name" before running anything. This file hit exactly that while
    being written, on a Windows checkout, which is why the guard is a test
    and not a comment. `.gitattributes` pins the STORED form; this pins the
    working-tree form, which is what actually executes."""
    crlf = bytes((13, 10))
    for rel in ("run.sh", "setup.sh"):
        raw = (ROOT / rel).read_bytes()
        assert crlf not in raw, (
            f"{rel} has CRLF line endings; bash will not run it"
        )


def test_run_sh_exists_and_is_syntactically_valid():
    r = subprocess.run([BASH, "-n", "./run.sh"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(os.name != "nt", reason="the WSL-launcher collision is Windows-only")
def test_find_bash_never_returns_the_wsl_launcher():
    """Pins the exact failure this file hit: every test here passed when
    run via this session's own Bash tool (a git-bash shell, whose own PATH
    happens to favour Git's bin directories) and FAILED when run via
    `.\\run.ps1 test-fast` (a PowerShell child process, where bare
    `shutil.which("bash")` on this machine resolved to
    `C:\\Windows\\System32\\bash.exe` -- the WSL launcher, which does not
    inherit MANDATEIQ_PY or anything else from the Win32 environment). A
    fix that merely happened to work in whichever shell it was tested from
    would silently reintroduce this the next time someone runs the suite
    from a different one."""
    resolved = _find_bash()
    assert resolved is not None
    assert "system32" not in resolved.lower()
    assert "syswow64" not in resolved.lower()


def test_test_task_exits_non_zero_on_a_failing_suite(tmp_path):
    """THE test. A green `./run.sh test` must mean the suite was green."""
    r = _run("test", _stub(tmp_path, 1))
    assert r.returncode != 0
    assert "FAILED (tests)" in r.stdout + r.stderr


def test_test_task_exits_zero_on_a_passing_suite(tmp_path):
    """The control: without this, the assertion above would pass even if
    run.sh failed unconditionally."""
    assert _run("test", _stub(tmp_path, 0)).returncode == 0


def test_a_multi_step_task_stops_at_the_first_failure(tmp_path):
    """`eval` is sweep-then-report. A failed sweep must not be followed by a
    report rendered from a stale artifact and an exit code of 0."""
    r = _run("eval", _stub(tmp_path, 1))
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "FAILED (sweep)" in out
    assert "== report" not in out, "the second step ran after the first failed"


def test_ci_task_propagates_a_failure(tmp_path):
    r = _run("ci", _stub(tmp_path, 1))
    assert r.returncode != 0


def test_help_works_without_a_venv(tmp_path):
    """`help` is the one command a reviewer runs BEFORE ./setup.sh. Failing
    there tells them nothing about what to do next."""
    missing = tmp_path / "does-not-exist"
    r = _run("help", missing)
    assert r.returncode == 0
    assert "./run.sh eval" in r.stdout


def test_an_unknown_task_fails_loudly(tmp_path):
    r = _run("definitely-not-a-task", _stub(tmp_path, 0))
    assert r.returncode != 0
    assert "unknown task" in r.stdout + r.stderr


def test_every_task_run_ps1_offers_is_either_mirrored_or_declined():
    """R7's scope decision was `run.sh` MIRRORING `run.ps1`, not a subset
    chosen by whoever wrote it. Any run.ps1 action that is neither
    implemented here nor named in the "NOT ported" list is an omission, not
    a decision -- and this fails rather than letting it pass as one."""
    import re

    ps1 = (ROOT / "run.ps1").read_text(encoding="utf-8", errors="ignore")
    sh = RUN_SH.read_text(encoding="utf-8")

    ps_tasks = set(re.findall(r'^\s{4}"([a-z][a-z-]*)"\s*[{]', ps1, re.MULTILINE))
    assert len(ps_tasks) > 15, f"run.ps1 task scrape looks broken: {ps_tasks}"

    declined = {"up", "down", "verify", "freeze"}
    for task in sorted(ps_tasks - declined):
        assert f"\n  {task})" in sh or f"| {task})" in sh or f"  {task}|" in sh \
            or re.search(rf'^\s+{re.escape(task)}\)', sh, re.MULTILINE), (
            f"run.ps1 offers '{task}' and run.sh neither implements it nor "
            f"declines it in the NOT-ported list"
        )
    for task in sorted(declined):
        assert task in sh, f"'{task}' is not ported and run.sh does not say so"
