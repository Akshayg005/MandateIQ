"""scripts/guard_invariants.py -- the empty-input path specifically.

Design spec this file pins: when no file paths can be resolved (argv, env,
stdin, and git's fallback all empty -- rare in practice, since hookio's git
fallback scans staged/modified/untracked files and "never empty-passes" by
its own docstring, but not impossible on a genuinely clean working tree),
main() must exit 2, not 0. Found during the 2026-08-29 vacuous-checks audit
(DECISIONS.md): the function's own comment already said "never silently
pass... say so loudly rather than exiting 0," but the code returned 0
anyway -- the intent was right, the return value was not. Confirmed
empirically the same day that a genuine violation's exit-2 stderr surfaces
fully as a PostToolUse hook message; exit 0 has no such proof and the Stop
hook's own documented contract (exit 0 -> debug log only) is the same
family of risk. This test is the regression guard for that fix, not a
docstring pinning a pre-existing behaviour.

The rest of guard_invariants.py (LLM-import / float-money / live-key /
hard-cancel detection) is exercised indirectly all session via the
PostToolUse hook itself, not duplicated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_main_exits_2_when_no_paths_resolve_bare_mode(monkeypatch):
    """Bare invocation (no --all): resolve_paths() returning ([], "none")
    must make main() return 2, not 0."""
    import scripts.guard_invariants as gi

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([], "none"))

    assert gi.main() == 2, \
        "main() returned something other than 2 when resolve_paths() found nothing"


def test_main_exits_2_when_no_paths_resolve_all_mode(monkeypatch):
    """--all invocation: all_tracked() returning [] must also make main()
    return 2 -- the empty-input guard is shared code, not just the bare path."""
    import scripts.guard_invariants as gi

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py", "--all"])
    monkeypatch.setattr(gi, "all_tracked", lambda: [])

    assert gi.main() == 2, \
        "main() --all returned something other than 2 when all_tracked() found nothing"


def test_main_still_exits_0_on_a_real_clean_file(monkeypatch, tmp_path):
    """Regression guard against overcorrecting: a real, clean, resolvable
    file must still exit 0 -- only the "nothing resolved" case changed."""
    import scripts.guard_invariants as gi

    clean_file = tmp_path / "clean.py"
    clean_file.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([clean_file], "test"))

    assert gi.main() == 0, \
        "main() no longer exits 0 for a real file with no violations"
