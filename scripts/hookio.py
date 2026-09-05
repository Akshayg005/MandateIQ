#!/usr/bin/env python3
"""Resolve which files a hook is about, without depending on the shell.

The problem this solves: hook commands are strings run by whatever shell the
platform uses. `$MANDATEIQ_FILE_PATHS` expands on bash and does nothing on
PowerShell or cmd, where the syntax is `$env:MANDATEIQ_FILE_PATHS` or
`%MANDATEIQ_FILE_PATHS%`. Get it wrong and the guard runs with zero arguments,
finds nothing, exits 0, and you believe your invariants are enforced when
they are not.

That is the worst possible failure mode here -- silent, and discovered on
day nine when an LLM import has been sitting in the decision core for a week.

So the guards do not rely on shell expansion at all. They try four sources
in order and use the first that yields file paths:

  1. command-line arguments        (works everywhere if the shell expanded)
  2. MANDATEIQ_FILE_PATHS env var     (read directly, no shell involved)
  3. JSON on stdin                 (editor hook payloads arrive this way)
  4. git: staged + modified files  (last-resort fallback, never empty-passes)

Source 4 matters most: if the first three come up empty, scanning changed
files means the guard still catches the violation. A guard that cannot see
its input should over-scan, never under-scan.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _from_argv() -> list[str]:
    return [a for a in sys.argv[1:] if a and not a.startswith("-")]


def _from_env() -> list[str]:
    raw = os.environ.get("MANDATEIQ_FILE_PATHS", "").strip()
    if not raw:
        return []
    # may be whitespace-, comma-, or semicolon-separated depending on platform
    for sep in (";", ","):
        raw = raw.replace(sep, " ")
    return [p for p in raw.split() if p]


def _from_stdin() -> list[str]:
    if sys.stdin is None or sys.stdin.isatty():
        return []
    try:
        data = sys.stdin.read()
    except Exception:
        return []
    if not data.strip():
        return []
    try:
        payload = json.loads(data)
    except Exception:
        return []

    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("file_path", "filePath", "path", "notebook_path") and isinstance(v, str):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _from_git() -> list[str]:
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        files = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        r2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        files += [l.strip() for l in r2.stdout.splitlines() if l.strip()]
        return files
    except Exception:
        return []


def resolve_paths() -> tuple[list[pathlib.Path], str]:
    """Return (existing files, which source produced them)."""
    for name, fn in (
        ("argv", _from_argv),
        ("env", _from_env),
        ("stdin", _from_stdin),
        ("git", _from_git),
    ):
        raw = fn()
        if raw:
            paths = []
            for r in raw:
                # Normalise separators: a Windows-style path may arrive on a
                # POSIX runtime or vice versa. Try the literal form first,
                # then the flipped one, before giving up on this entry.
                for variant in (r, r.replace("\\", "/"), r.replace("/", "\\")):
                    p = pathlib.Path(variant)
                    if not p.is_absolute():
                        p = ROOT / p
                    if p.is_file():
                        paths.append(p)
                        break
            if paths:
                return paths, name
    return [], "none"
