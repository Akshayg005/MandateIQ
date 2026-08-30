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


# --- B11: protected dirs must not import src.llm ---


def test_protects_src_classify_from_llm_import_from_form(tmp_path, monkeypatch):
    """A file under src/classify/ importing from src.llm -> violation caught."""
    import scripts.guard_invariants as gi

    classify_dir = tmp_path / "src" / "classify"
    classify_dir.mkdir(parents=True)
    bad_file = classify_dir / "cause_map.py"
    bad_file.write_text("from src.llm.normalizer import normalize\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([bad_file], "test"))

    result = gi.main()
    assert result != 0, "Should have flagged src.llm import in protected dir"


def test_protects_src_policy_from_llm_import_from_form(tmp_path, monkeypatch):
    """A file under src/policy/ importing from src.llm -> violation caught."""
    import scripts.guard_invariants as gi

    policy_dir = tmp_path / "src" / "policy"
    policy_dir.mkdir(parents=True)
    bad_file = policy_dir / "gate.py"
    bad_file.write_text("from src.llm.intent import intent_score\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([bad_file], "test"))

    result = gi.main()
    assert result != 0, "Should have flagged src.llm import in protected dir"


def test_protects_src_model_from_llm_import_from_form(tmp_path, monkeypatch):
    """A file under src/model/ importing from src.llm -> violation caught."""
    import scripts.guard_invariants as gi

    model_dir = tmp_path / "src" / "model"
    model_dir.mkdir(parents=True)
    bad_file = model_dir / "scorer.py"
    bad_file.write_text("from src.llm.normalizer import NormalizedDecline\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([bad_file], "test"))

    result = gi.main()
    assert result != 0, "Should have flagged src.llm import in protected dir"


def test_protects_src_core_from_llm_import_from_form(tmp_path, monkeypatch):
    """A file under src/core/ importing from src.llm -> violation caught."""
    import scripts.guard_invariants as gi

    core_dir = tmp_path / "src" / "core"
    core_dir.mkdir(parents=True)
    bad_file = core_dir / "shared.py"
    bad_file.write_text("from src.llm.narrator import narrate\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([bad_file], "test"))

    result = gi.main()
    assert result != 0, "Should have flagged src.llm import in protected dir"


def test_protects_src_classify_from_llm_import_statement_form(tmp_path, monkeypatch):
    """A file under src/classify/ using 'import src.llm.intent' -> violation caught."""
    import scripts.guard_invariants as gi

    classify_dir = tmp_path / "src" / "classify"
    classify_dir.mkdir(parents=True)
    bad_file = classify_dir / "classifier.py"
    bad_file.write_text("import src.llm.intent\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([bad_file], "test"))

    result = gi.main()
    assert result != 0, "Should have flagged src.llm import in protected dir"


def test_allows_src_execute_to_import_llm(tmp_path, monkeypatch):
    """A file under src/execute/ (NOT a protected dir) can import src.llm."""
    import scripts.guard_invariants as gi

    execute_dir = tmp_path / "src" / "execute"
    execute_dir.mkdir(parents=True)
    clean_file = execute_dir / "runner.py"
    clean_file.write_text("from src.llm.normalizer import normalize\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([clean_file], "test"))

    result = gi.main()
    assert result == 0, "src/execute/ should be allowed to import src.llm"


def test_allows_protected_dir_unrelated_imports(tmp_path, monkeypatch):
    """A file under a protected dir with normal imports (no src.llm) is clean."""
    import scripts.guard_invariants as gi

    classify_dir = tmp_path / "src" / "classify"
    classify_dir.mkdir(parents=True)
    clean_file = classify_dir / "cause_map.py"
    clean_file.write_text("from src.core.types import DeclineClass\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([clean_file], "test"))

    result = gi.main()
    assert result == 0, "Protected dir file with unrelated imports should be clean"


def test_substring_match_googlemaps_not_flagged(tmp_path, monkeypatch):
    """A file under protected dir with 'llm' inside an unrelated word is clean."""
    import scripts.guard_invariants as gi

    policy_dir = tmp_path / "src" / "policy"
    policy_dir.mkdir(parents=True)
    clean_file = policy_dir / "allocator.py"
    # 'willm' and 'algorithm' contain 'llm' as substring but are not imports
    clean_file.write_text(
        "# This is a brilliant algorithm for allocation\n"
        "# Not the llm module import at all\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["guard_invariants.py"])
    monkeypatch.setattr(gi, "resolve_paths", lambda: ([clean_file], "test"))

    result = gi.main()
    assert result == 0, "Substring 'llm' in comments/strings should not be flagged"
