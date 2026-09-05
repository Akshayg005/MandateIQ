#!/usr/bin/env bash
# One-time setup for the Mandate Recovery Engine on Linux and macOS.
# The POSIX mirror of setup.ps1.
#
# R7 (reports/gates.md, "Post-B16 remediation gates"): before this file, the
# ONLY documented install path in this repo was `.\setup.ps1`. There was no
# venv instruction, no `pip install -r`, no docker line outside setup.ps1's
# own body, and no `.env` creation -- so "a reviewer on Linux can install"
# was not merely untested, it was undocumented.
#
# One deliberate difference from setup.ps1, and it matters:
#
#   setup.ps1 pip-installs a HAND-LISTED set of ~24 unpinned packages and
#   THEN overwrites requirements.txt from `pip freeze`. That makes
#   `pip install -r requirements.txt` an install path nothing in this repo
#   documented or exercised. This script installs FROM requirements.txt --
#   the pinned set, the same bytes CI installs on ubuntu-latest. If a pin is
#   wrong for this platform, that is a finding and it surfaces here rather
#   than in a reviewer's confusion.
set -euo pipefail
cd "$(dirname "$0")"

PY_TARGET="${PY_TARGET:-python3.13}"
say()  { printf '\n== %s\n' "$1"; }
ok()   { printf '   ok -- %s\n' "$1"; }
warn() { printf '   warning -- %s\n' "$1" >&2; }
die()  { printf '   FAILED -- %s\n' "$1" >&2; exit 1; }

# --------------------------------------------------------------- python --
say "locating python"
if command -v "$PY_TARGET" >/dev/null 2>&1; then
  REAL_PY="$(command -v "$PY_TARGET")"
elif command -v python3 >/dev/null 2>&1; then
  REAL_PY="$(command -v python3)"
  warn "$PY_TARGET not found; using $($REAL_PY -V). The project targets 3.13."
else
  die "no python3 on PATH. Install Python 3.13, or set PY_TARGET."
fi
ok "$($REAL_PY -V) at $REAL_PY"

say "creating .venv"
if [ ! -d .venv ]; then
  "$REAL_PY" -m venv .venv || die "venv creation failed"
fi
PY=".venv/bin/python"
[ -x "$PY" ] || die "venv python missing at $PY"
ok ".venv ready"

say "installing dependencies from requirements.txt (2-4 minutes)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt || die \
  "pip install -r requirements.txt failed. Read the error above for which
   package. If it is a missing wheel, that package has no build for this
   Python yet: pin an older version, or set PY_TARGET=python3.12, delete
   .venv, and re-run."
ok "dependencies installed"

# ------------------------------------------------------------- postgres --
# Lifted out of setup.ps1:103-114, which was the only place in the repo this
# line existed -- `run.ps1 up` only does `docker start mrdb`, which fails if
# the container was never created.
#
# Port: 5432 here, matching .env.example. This machine's own .env uses 15432
# because Windows reserves a dynamic port range covering 5432; that is a
# Windows problem and not one a Linux reviewer should inherit. If 5432 is
# taken, change BOTH the -p flag below and DATABASE_URL in .env.
say "starting postgres"
if ! command -v docker >/dev/null 2>&1; then
  warn "docker not on PATH -- skipping. The test suite FAILS (it does not
   skip) without a database; see README. Set MANDATEIQ_ALLOW_PG_SKIP=1 to
   restore skipping, deliberately."
elif [ -n "$(docker ps -a --filter name=^mrdb$ --format '{{.Names}}')" ]; then
  docker start mrdb >/dev/null
  ok "existing mrdb container started"
else
  docker run -d --name mrdb \
    -e POSTGRES_PASSWORD=dev \
    -e POSTGRES_DB=mandate_recovery \
    -p 5432:5432 postgres:16 >/dev/null || die "postgres container failed to start"
  ok "mrdb container created on 5432"
fi
if command -v docker >/dev/null 2>&1; then
  sleep 3
  if docker exec mrdb pg_isready >/dev/null 2>&1; then
    ok "postgres accepting connections"
  else
    warn "postgres not ready yet -- give it a few seconds"
  fi
fi

# ------------------------------------------------------------------ env --
say "environment"
if [ ! -f .env ]; then
  cp .env.example .env
  ok ".env created from .env.example -- edit it if your Postgres port differs"
else
  ok ".env already exists, left untouched"
fi

# ------------------------------------------------------------- frontend --
# npm install only. setup.ps1 SCAFFOLDS the two workspaces with
# `npm create vite@latest` if src/ is missing -- that was a Day-1 bootstrap
# and both workspaces are tracked in git now, so re-scaffolding a clone
# would be destructive rather than helpful.
if [ "${SKIP_FRONTEND:-}" = "1" ]; then
  say "frontend skipped (SKIP_FRONTEND=1)"
elif ! command -v npm >/dev/null 2>&1; then
  warn "npm not on PATH -- skipping. Only the dashboard and landing page
   need it; every python task above works without it."
else
  for w in dashboard site; do
    say "npm install ($w)"
    (cd "$w" && npm install --silent) || die "npm install failed in $w"
    ok "$w ready"
  done
fi

cat <<'EOF'

== next

  ./run.sh test          the suite (needs the database above)
  ./run.sh eval          the full sweep, ~15 minutes
  ./run.sh report        re-render reports/regimes.md from the last sweep
  ./run.sh help          everything else

A FRESH CLONE HAS NO reports/*.json (.gitignore excludes them), so
`report`, `dashboard` and `site` have nothing to read until `eval` has run.
EOF
