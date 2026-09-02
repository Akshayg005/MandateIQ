"""Stage the report artifacts where the dashboard's Vite build can read them.

    python scripts\\dashboard_data.py

B14's rule is that the dashboard RENDERS the report and recomputes nothing.
This script is what makes that checkable: the SPA reads only files under
`dashboard/public/data/`, and those files arrive here by copy, never by
transformation. If a number appears on screen it came out of `reports/`,
and `manifest.json` records which commit and which freeze hash produced it,
so a screenshot can be traced to a run.

Deliberately a copy and not a Vite alias into `../reports`: `vite build`
must produce a self-contained bundle that still works when Postgres is down
and the eval has not been run, which is the state a reviewer clones into.
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
DEST = _ROOT / "dashboard" / "public" / "data"

# (source, required). A missing optional artifact is reported, not fatal:
# mandates.json needs Postgres to regenerate, and a reviewer without Docker
# running must still be able to build the aggregate views.
ARTIFACTS = [
    ("results.json", True),
    ("regimes.json", True),
    ("mandates.json", False),
]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    missing: list[str] = []

    for name, required in ARTIFACTS:
        src = REPORTS / name
        if not src.exists():
            if required:
                print(f"ERROR: {src} is missing -- run .\\run.ps1 eval first",
                      file=sys.stderr)
                return 1
            missing.append(name)
            print(f"  skipped {name} (absent; run python -m eval.export_mandates)")
            continue
        shutil.copyfile(src, DEST / name)
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
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    print(f"  wrote   manifest.json (git {manifest['git_sha'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
