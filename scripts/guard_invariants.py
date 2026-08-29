#!/usr/bin/env python3
"""Architectural invariant guard. Cross-platform.

Runs as a Claude Code PostToolUse hook on every Edit/Write. Exits non-zero
with a readable reason when an edit violates a project invariant.

This file is a deliverable. It is the mechanical proof that the "no LLM in
the decision core" claim in the README is an enforced property rather than a
paragraph of intent.

Invariants enforced (see CLAUDE.md):
  1. src/model/, src/policy/, src/core/ may not import an LLM client
  2. money is integer paise, never float
  5. no live Razorpay keys anywhere in the repo
  6. the system never executes a cancellation

Usage:
  python scripts/guard_invariants.py [FILE ...]
  python scripts/guard_invariants.py --all      # scan every tracked .py
File paths come from argv, CLAUDE_FILE_PATHS, stdin JSON, or git -- in that
order -- so it works identically under bash, PowerShell and cmd.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hookio import ROOT, resolve_paths  # noqa: E402

# --- invariant 1: no generative models in the deterministic core ------------
PROTECTED_DIRS = ("src/model/", "src/policy/", "src/core/", "src/classify/")
LLM_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(anthropic|openai|cohere|google\.generativeai|litellm)\b",
    re.MULTILINE,
)

# --- invariant 2: money is integer paise -----------------------------------
FLOAT_MONEY = re.compile(
    r"\b(amount|amt|paise|balance|ltv|revenue|cost|price|fee|ceiling)\w*"
    r"\s*(?::\s*float\b|=\s*float\()",
    re.IGNORECASE,
)
MONEY_DIVISION = re.compile(r"\b(amount|paise|balance|ltv)\w*\s*/\s*(?!/)")

# --- invariant 5: test mode only -------------------------------------------
# Match an actual key (prefix + id characters), not the bare prefix. Matching
# the bare prefix makes this file -- and every doc that names the pattern --
# flag itself, so the check can never pass.
LIVE_KEY = re.compile(r"rzp_live_[A-Za-z0-9]{10,}")

# --- invariant 6: never execute a cancellation -----------------------------
HARD_CANCEL = re.compile(r"\.(cancel_subscription|cancel)\s*\(", re.IGNORECASE)
OFFRAMP_OK = "src/policy/offramp.py"

SKIP = (".venv", "node_modules", ".git", "site\\node_modules", "site/node_modules",
        "dashboard/node_modules", "dashboard\\node_modules", "__pycache__")


def norm(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def check(path: pathlib.Path) -> list[str]:
    rel = norm(path)
    if any(s.replace("\\", "/") in rel for s in SKIP):
        return []
    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    problems: list[str] = []

    if LIVE_KEY.search(text):
        problems.append(
            "LIVE RAZORPAY KEY detected (invariant 5). This project is test mode "
            "only. Remove it and rotate the key."
        )

    if any(d in rel for d in PROTECTED_DIRS):
        m = LLM_IMPORT.search(text)
        if m:
            problems.append(
                f"LLM import '{m.group(1)}' in the deterministic core (invariant 1). "
                "Move this work to src/llm/ and pass the result in as a plain value."
            )
        m = FLOAT_MONEY.search(text)
        if m:
            problems.append(
                f"float money: '{m.group(0).strip()}' (invariant 2). Use integer paise."
            )
        m = MONEY_DIVISION.search(text)
        if m:
            problems.append(
                f"division on a money value: '{m.group(0).strip()}' (invariant 2). "
                "Use // with an explicit rounding decision, documented."
            )

    if rel.endswith(".py") and OFFRAMP_OK not in rel and "test" not in rel:
        if HARD_CANCEL.search(text):
            problems.append(
                "direct cancellation call (invariant 6). The system OFFERS an "
                "off-ramp; the customer decides. Route through src/policy/offramp.py."
            )

    return problems


def all_tracked() -> list[pathlib.Path]:
    try:
        r = subprocess.run(
            ["git", "ls-files", "*.py"], cwd=ROOT,
            capture_output=True, text=True, timeout=15,
        )
        return [ROOT / l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return list(ROOT.rglob("*.py"))


def main() -> int:
    if "--all" in sys.argv:
        paths, source = [p for p in all_tracked() if p.is_file()], "git(all)"
    else:
        paths, source = resolve_paths()

    if not paths:
        # Never silently pass. Previously returned 0 here despite this
        # comment already saying not to -- the intent was right, the
        # return value contradicted it (found in the 2026-08-29 vacuous-
        # checks audit, DECISIONS.md). Confirmed empirically the same day:
        # a genuine violation's exit-2 stderr surfaces fully as a
        # PostToolUse:Write hook message; there is no equivalent proof
        # exit 0 would have been seen at all, and the Stop hook's own
        # documented contract (exit 0 -> debug log only, never shown) is
        # the same family of risk. Exit 2 here now, so "nothing was
        # checked" gets the same loud treatment as "a violation was found"
        # rather than being the one path that stays quiet.
        print(
            "guard_invariants: no files resolved (argv/env/stdin/git all empty). "
            "Nothing was checked. If this repeats, the hook is not wired correctly "
            "-- run: python scripts/guard_invariants.py --all",
            file=sys.stderr,
        )
        return 2

    problems = [f"{norm(p)}: {msg}" for p in paths for msg in check(p)]

    if problems:
        print("\n[X] INVARIANT VIOLATION\n", file=sys.stderr)
        for x in problems:
            print(f"  - {x}", file=sys.stderr)
        print("\nSee CLAUDE.md > Non-negotiable invariants.\n", file=sys.stderr)
        return 2

    if "--all" in sys.argv or "-v" in sys.argv:
        print(f"guard_invariants: {len(paths)} file(s) checked via {source} -- clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
