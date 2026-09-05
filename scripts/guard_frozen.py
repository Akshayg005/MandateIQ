#!/usr/bin/env python3
"""Pre-registration guard. Cross-platform.

Runs as an editor/pre-commit write guard. Denies any write under eval/frozen/.

Why this exists: the most common way an ML demo fools itself is tuning the
evaluation until the policy wins. We commit the simulator config and the
evaluation protocol on Day 1, BEFORE any policy code is written, and record
the commit hash in reports/FREEZE_HASH. This hook turns that promise into
something mechanically enforced.

If a change to the frozen eval is genuinely necessary, that is a decision
logged in DECISIONS.md and made by a human editing the file outside the
session -- not something an agent does mid-task on day eight.

Usage:  python scripts/guard_frozen.py [FILE ...]
Paths come from argv, MANDATEIQ_FILE_PATHS, or stdin JSON -- shell-independent.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hookio import ROOT, _from_argv, _from_env, _from_stdin  # noqa: E402

FROZEN = "eval/frozen/"


def candidates() -> list[str]:
    # Deliberately does NOT fall back to git here: a git fallback would list
    # every changed file and could deny an unrelated edit. For a *deny* hook,
    # a false positive is worse than a miss, and the write-guard plus
    # the Day-1 hash check catch anything that slips through.
    for fn in (_from_argv, _from_env, _from_stdin):
        got = fn()
        if got:
            return got
    return []


def main() -> int:
    for raw in candidates():
        p = pathlib.Path(raw)
        try:
            rel = str(p.relative_to(ROOT)) if p.is_absolute() else str(p)
        except ValueError:
            rel = str(p)
        rel = rel.replace("\\", "/")

        if FROZEN in rel:
            print(
                f"{rel} is inside eval/frozen/, pre-registered before "
                "any policy code was written (see reports/FREEZE_HASH). "
                "Editing it invalidates every number in the report. "
                "If this change is truly required, log it in DECISIONS.md "
                "and make it manually, outside this guard.",
                file=sys.stderr,
            )
            print(f"DENIED: {rel} is frozen.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
