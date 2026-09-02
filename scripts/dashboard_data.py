"""Stage the report artifacts where a Vite build can read them.

    python scripts\\dashboard_data.py            # B14 dashboard
    python scripts\\dashboard_data.py site       # B15 landing page

B14's rule is that the dashboard RENDERS the report and recomputes nothing,
and B15's gate says the landing page's counters are wired to real report
output rather than hard-coded. This script is what makes both checkable: the
SPAs read only files under their own `public/data/`, and those files arrive
there by copy, never by transformation. If a number appears on screen it came
out of `reports/`, and `manifest.json` records which commit and which freeze
hash produced it, so a screenshot can be traced to a run.

Deliberately a copy and not a Vite alias into `../reports`: `vite build`
must produce a self-contained bundle that still works when Postgres is down
and the eval has not been run, which is the state a reviewer clones into.

One staging implementation for both targets on purpose. Two would drift, and
the claim being defended -- that every published number came from `reports/`
-- is only as good as the number of places that can quietly stop being true.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import shutil
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = _ROOT / "reports"

# target -> (destination, [(artifact, required)]).
#
# A missing OPTIONAL artifact is reported, not fatal: mandates.json needs
# Postgres to regenerate, and a reviewer without Docker running must still be
# able to build the aggregate views.
#
# The site takes results.json alone, and that is a correctness constraint
# rather than a size one. results.json's headline figures are means over the
# 8 seeds in its `seeds` field; mandates.json is the seed-0 batch only (see
# eval/export_mandates.py). Mixing them on one page would put a seed-0 count
# beside an 8-seed mean under a single label -- the landing page therefore
# gets the aggregate artifact and nothing else.
TARGETS: dict[str, tuple[pathlib.Path, list[tuple[str, bool]]]] = {
    "dashboard": (
        _ROOT / "dashboard" / "public" / "data",
        [("results.json", True), ("regimes.json", True), ("mandates.json", False)],
    ),
    "site": (
        _ROOT / "site" / "public" / "data",
        [("results.json", True)],
    ),
}


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    target = args[0] if args else "dashboard"
    if target not in TARGETS:
        print(f"ERROR: unknown target {target!r}; expected one of "
              f"{', '.join(sorted(TARGETS))}", file=sys.stderr)
        return 2
    dest, artifacts = TARGETS[target]

    dest.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    missing: list[str] = []

    for name, required in artifacts:
        src = REPORTS / name
        if not src.exists():
            if required:
                print(f"ERROR: {src} is missing -- run .\\run.ps1 eval first",
                      file=sys.stderr)
                return 1
            missing.append(name)
            print(f"  skipped {name} (absent; run python -m eval.export_mandates)")
            continue
        shutil.copyfile(src, dest / name)
        staged.append(name)
        print(f"  staged  {name} ({src.stat().st_size:,} bytes)")

    freeze_hash = (REPORTS / "FREEZE_HASH")
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "freeze_hash": freeze_hash.read_text(encoding="utf-8").strip()
                       if freeze_hash.exists() else "unknown",
        "staged": staged,
        "missing": missing,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    print(f"  wrote   manifest.json (git {manifest['git_sha'][:12]}) -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
