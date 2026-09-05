#!/usr/bin/env bash
# Task runner for the Mandate Recovery Engine on Linux and macOS.
# The POSIX mirror of run.ps1 -- `.\run.ps1 test` becomes `./run.sh test`.
#
# R7 (reports/gates.md, "Post-B16 remediation gates"): "a reviewer on Linux
# or macOS can install, test, run the eval and read the report using commands
# printed in the README, without translating anything." A README that told a
# reviewer to translate six PowerShell lines into `python -m` invocations
# would BE the translation the gate forbids, so this file exists rather than
# a paragraph of raw commands (DECISIONS.md, 2026-09-05, scope decision 1).
#
# === The exit-code discipline, carried over rather than re-earned ==========
#
# run.ps1's Invoke-Step exists because a bare call as a switch branch's last
# statement returned 0 on a RED test suite, which made CLAUDE.md's own
# definition-of-done step 3 ("`run.ps1 test` passes before any commit")
# unfalsifiable for several blocks (run.ps1:157-165; reports/gates.md:597).
# `step()` below is the same guard: it prints a labelled banner, runs the
# command, and propagates a non-zero exit immediately. `set -euo pipefail`
# would catch most of it, but not a failure inside a pipeline's left side
# and not a multi-command branch, so the wrapper is explicit.
#
# tests/scripts/test_run_sh.py proves `./run.sh test` exits non-zero on a
# failing suite -- the property, not the code shape.
#
# === What is deliberately NOT ported ======================================
#
# `up` / `down`: run.ps1 gives every server its own console window
# (Start-Pane wraps `powershell.exe -NoExit`) and `down` walks Win32_Process
# parentage before `taskkill /T /F`. There is no honest POSIX translation of
# either, and a worse version pretending to be the same command is more
# confusing than its absence. `db-up` / `db-down` below do the one piece
# that does translate: starting and stopping the Postgres container.
#
# `verify`: its live Razorpay probe is a Windows-desktop pre-flight, not
# something a reviewer runs.
#
# `freeze`: BLOCK B2 ONLY, and already executed. Re-running it on any
# platform would be a mistake, so it is not offered here.
set -euo pipefail

cd "$(dirname "$0")"

# The project venv, so nothing here depends on activation state -- the same
# reason run.ps1 resolves .venv\Scripts\python.exe explicitly.
#
# MANDATEIQ_PY overrides it. Two real uses, neither of them a testing hook
# bolted onto production code: a reviewer whose venv lives elsewhere, and
# tests/scripts/test_run_sh.py, which points it at a stub interpreter to
# prove `./run.sh test` exits NON-ZERO on a red suite. That property is the
# whole reason step() exists (see the header), and it cannot be checked
# without being able to make the suite fail on demand.
PY="${MANDATEIQ_PY:-.venv/bin/python}"

require_py() {
  if [ ! -x "$PY" ]; then
    echo "venv not found at $PY" >&2
    echo "Run ./setup.sh first." >&2
    exit 1
  fi
}

# Single source of truth for the fast-path test filter, so `test-fast` and
# `ci` cannot drift into two definitions of "fast" -- the same reason
# run.ps1 keeps $TestFastFilter in one place.
TEST_FAST_FILTER="not chaos and not slow"

step() {
  local label="$1"; shift
  printf '\n== %s\n' "$label"
  # `|| rc=$?`, NOT a bare `"$@"` followed by `rc=$?`. Under `set -e` a bare
  # call that fails aborts the script AT THAT LINE, so the check below would
  # never run and the FAILED banner would never print -- the exit code would
  # still be right, but a reviewer would be left staring at a bare "== sweep"
  # and nothing else. Putting the call in a condition context suppresses the
  # abort so the label can be reported first. Caught by
  # tests/scripts/test_run_sh.py, which asserts the BANNER as well as the
  # code: run.ps1's Invoke-Step prints one, and a mirror that silently
  # dropped it is a worse mirror.
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '   FAILED (%s)\n' "$label" >&2
    exit "$rc"
  fi
}

usage() {
  cat <<'EOF'

Mandate Recovery Engine -- tasks (POSIX; the mirror of run.ps1)

  ./run.sh test              full test suite with coverage
                              NEEDS POSTGRES -- the ledger, executor and
                              crash-recovery tests FAIL, not skip, without
                              it. Start it with ./run.sh db-up.
                              MANDATEIQ_ALLOW_PG_SKIP=1 restores skipping.
  ./run.sh test-fast         unit tests only, skips chaos and slow/simulation
  ./run.sh lint              invariant guards + live-key scan
  ./run.sh ci                test-fast + lint

  ./run.sh eval              full eval, all regimes, both profiles (~15 min)
  ./run.sh eval-quick        baseline regime, nominal arm, strict profile
                              NOTE: writes the SAME reports/regimes.json
                              a full `eval` does, so it overwrites one.
                              Pass --out elsewhere to keep both.
  ./run.sh offramp-channel   R5's off-ramp channel-quality sweep
  ./run.sh ltv               R3's LTV sensitivity sweep
  ./run.sh report            re-render tables+figures from the last run
  ./run.sh bench             LLM vs statistical core benchmark (needs a key)
  ./run.sh golden            golden-set regression on the LLM layer (cached)
  ./run.sh shadow            decide without executing; delta vs the ladder
  ./run.sh chaos [N]         induced process kills (default 50)
  ./run.sh coverage          decline-taxonomy coverage report

  ./run.sh serve             the HTTP API -- webhook ingest + the three
                              read endpoints (uvicorn, port 8000)
  ./run.sh dashboard         export + stage + serve the reviewer dashboard
  ./run.sh dashboard-build   stage, lint, build, render-check
  ./run.sh site              stage + serve the landing page
  ./run.sh site-build        stage, lint, build, render-check

  ./run.sh db-up             start the mrdb Postgres container
  ./run.sh db-down           stop it
  ./run.sh state             print the session-start orientation block
  ./run.sh checkpoint B4     end of session -- regenerate STATE.md
  ./run.sh clean             remove __pycache__, .pytest_cache, .coverage
  ./run.sh help              this list

NOT ported from run.ps1, deliberately (see this file's header):
  up / down   -- per-server console windows and Win32_Process teardown have
                 no honest POSIX equivalent. Use db-up/db-down plus serve,
                 dashboard and site in separate terminals.
  verify      -- a Windows-desktop pre-flight with a live Razorpay probe.
  freeze      -- BLOCK B2 ONLY, already executed. Re-running it is a bug.

A FRESH CLONE HAS NO reports/*.json: .gitignore excludes them, so the
tracked reports/*.md are present but `dashboard`, `site` and `report` have
nothing to read until `./run.sh eval` has run (~15 minutes). That is
documented rather than worked around.
EOF
}

TASK="${1:-help}"
shift || true

# `help` must work in a tree that has never been set up -- it is the one
# command a reviewer runs BEFORE ./setup.sh, and failing there tells them
# nothing about what to do next.
case "$TASK" in
  help|-h|--help) usage; exit 0 ;;
esac
require_py

case "$TASK" in

  test)      step "tests"      "$PY" -m pytest -q --cov=src --cov-report=term-missing ;;
  test-fast) step "tests-fast" "$PY" -m pytest -q -m "$TEST_FAST_FILTER" ;;

  lint)
    step "invariant guards" "$PY" scripts/guard_invariants.py --all
    printf '\n== live-key scan\n'
    # The same vacuous-check trap run.ps1's lint branch documents (the
    # 2026-08-29 audit): a scan that examined zero files used to print OK.
    # Assert a non-zero file count before trusting a clean result.
    n_files=$(git ls-files | wc -l)
    if [ "$n_files" -eq 0 ]; then
      echo "   FAILED -- zero files found to scan (wrong directory?)" >&2
      exit 1
    fi
    if git grep -nE 'rzp_live_[A-Za-z0-9]{10,}' -- . ; then
      echo "LIVE KEY FOUND" >&2
      exit 1
    fi
    echo "   no live keys: OK ($n_files tracked files scanned)"
    ;;

  ci)
    step "tests"  "$PY" -m pytest -q -m "$TEST_FAST_FILTER"
    step "guards" "$PY" scripts/guard_invariants.py --all
    ;;

  eval)
    step "sweep"  "$PY" -m eval.run --config eval/frozen/sim_config.yaml \
                        --all-regimes --both-profiles --seeds 8
    step "report" "$PY" -m eval.report --figures
    ;;
  eval-quick)
    step "eval-quick" "$PY" -m eval.run --config eval/frozen/sim_config.yaml \
                            --regime baseline --arm nominal --profile strict --quiet
    ;;
  offramp-channel) step "offramp-channel" "$PY" -m eval.offramp_channel ;;
  ltv)             step "ltv"             "$PY" -m eval.ltv_sensitivity ;;
  report)          step "report"          "$PY" -m eval.report --figures ;;
  bench)           step "bench"           "$PY" bench/llm_vs_stats.py --n 140 --repeats 5 --variance-n 30 ;;
  golden)
    if [ "${1:-}" = "--no-cache" ]; then
      step "golden (live)"   "$PY" eval/golden_check.py --no-cache
    else
      step "golden (cached)" "$PY" eval/golden_check.py
    fi
    ;;
  shadow)   step "shadow"   "$PY" -m src.execute.shadow ;;
  chaos)    step "chaos"    "$PY" -m eval.chaos --kills="${1:-50}" ;;
  coverage) step "coverage" "$PY" scripts/decline_coverage.py ;;

  serve)    exec "$PY" -m uvicorn src.ingest.app:app --reload --port 8000 ;;

  dashboard)
    step "export" "$PY" -m eval.export_mandates
    step "stage"  "$PY" scripts/dashboard_data.py
    cd dashboard && exec npm run dev
    ;;
  dashboard-build)
    step "stage" "$PY" scripts/dashboard_data.py
    cd dashboard
    step "lint"   npm run lint
    step "build"  npm run build
    step "render" npm run render-check
    ;;
  site)
    step "stage" "$PY" scripts/dashboard_data.py site
    cd site && exec npm run dev
    ;;
  site-build)
    step "stage" "$PY" scripts/dashboard_data.py site
    cd site
    step "lint"   npm run lint
    step "build"  npm run build
    step "render" npm run render-check
    ;;

  db-up)   step "postgres" docker start mrdb ;;
  db-down) step "postgres" docker stop mrdb ;;

  state)      step "state"      "$PY" scripts/show_state.py ;;
  # $TEST_FAST_FILTER threaded through as argv[2], exactly as run.ps1 does
  # -- checkpoint.py's own test count previously hard-coded "not chaos" and
  # silently diverged from what test-fast/ci actually run (DECISIONS.md,
  # 2026-08-29). Two runners passing two different filters would reopen it.
  checkpoint) step "checkpoint" "$PY" scripts/checkpoint.py "${1:-}" "$TEST_FAST_FILTER" ;;

  clean)
    find . -type d -name __pycache__ -not -path './.venv/*' -not -path './node_modules/*' \
      -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .coverage
    echo "cleaned"
    ;;

  *)
    echo "unknown task: $TASK" >&2
    usage >&2
    exit 1
    ;;
esac
