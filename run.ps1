<#
.SYNOPSIS
  Task runner for the Mandate Recovery Engine on Windows.
  Replaces the Makefile -- `make test` becomes `.\run.ps1 test`.

.EXAMPLE
  .\run.ps1 help
  .\run.ps1 test
  .\run.ps1 checkpoint -Day B4
  .\run.ps1 chaos -Kills 50
#>
param(
    [Parameter(Position = 0)]
    [string]$Task = "help",
    # Block id, e.g. B4. Typed as a string, not an int: work is keyed to
    # dependency blocks (reports/gates.md), not to calendar days.
    # -Block is the preferred spelling; -Day stays valid so existing docs
    # and habits keep working.
    [Alias("Block")]
    [string]$Day = "",
    [int]$Kills = 50,
    # golden: bypass eval/golden/.cache/ and force a live call for every
    # row. The gate should be ticked against a -NoCache run, not a cached
    # one -- see DECISIONS.md, 2026-08-30.
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# --- resolve the venv python, so you never depend on activation state ------
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "venv not found at .venv\Scripts\python.exe" -ForegroundColor Yellow
    Write-Host "Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Single source of truth for the fast-path test filter, so `test-fast` and
# `ci` cannot drift apart into two different definitions of "fast." `chaos`
# (B10) and `slow` (real-simulation tests, see DECISIONS.md 2026-08-29) are
# both excluded from the default frequent-run path.
$TestFastFilter = "not chaos and not slow"

function Invoke-Step([string]$Label, [scriptblock]$Block) {
    Write-Host "`n== $Label" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   FAILED ($Label)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Task.ToLower()) {

    "help" {
        Write-Host @"

Mandate Recovery Engine -- tasks

  .\run.ps1 test              full test suite with coverage
  .\run.ps1 test-fast         unit tests only, skips chaos and slow/simulation
  .\run.ps1 lint              invariant guards across all tracked python
  .\run.ps1 ci                test-fast + lint. Does NOT run eval -- eval.run
                               is B13's file (PLAN_DETAIL.md:518) and has
                               never existed; see DECISIONS.md, 2026-08-29

  .\run.ps1 eval              full eval, all regimes, both compliance profiles
  .\run.ps1 eval-quick        nominal regime only, strict profile
  .\run.ps1 golden            golden-set regression on the LLM layer (cached;
                               -NoCache forces a live call on every row)
  .\run.ps1 bench             LLM vs statistical core benchmark
  .\run.ps1 shadow            decide without executing; delta vs the fixed ladder
  .\run.ps1 chaos -Kills 50   induced process kills
  .\run.ps1 report            regenerate all figures and tables

  .\run.ps1 freeze            BLOCK B2 ONLY -- commit and record the eval hash
  .\run.ps1 checkpoint -Day B4  end of session -- regenerate STATE.md
  .\run.ps1 state             print the session-start orientation block
  .\run.ps1 verify            full pre-flight: guards, keys, docker, hooks
  .\run.ps1 serve             run the webhook ingest API (uvicorn, port 8000)
  .\run.ps1 coverage          decline_class / UNKNOWN-rate breakdown from ingested_event
  .\run.ps1 clean             remove caches

"@
    }

    "test"      { & $Py -m pytest -q --cov=src --cov-report=term-missing }
    "test-fast" { & $Py -m pytest -q -m $TestFastFilter }

    "lint" {
        Invoke-Step "invariant guards" { & $Py scripts\guard_invariants.py --all }
        Write-Host "== live-key scan" -ForegroundColor Cyan
        $scanFiles = Get-ChildItem -Recurse -File -Exclude *.zip |
            Where-Object { $_.FullName -notmatch '\\(\.venv|node_modules|\.git)\\' }
        # Found in the 2026-08-29 vacuous-checks audit (DECISIONS.md): an
        # empty $scanFiles (wrong cwd, permissions) made Select-String
        # match nothing and this printed "OK" having examined nothing --
        # the security check that should be hardest to fool was the
        # easiest. Assert a non-zero file count before trusting a clean
        # $hits.
        if (-not $scanFiles) {
            Write-Host "   FAILED -- zero files found to scan (wrong directory?)" -ForegroundColor Red
            exit 1
        }
        $hits = Select-String -Path $scanFiles.FullName -Pattern "rzp_live_[A-Za-z0-9]{10,}" -ErrorAction SilentlyContinue
        if ($hits) {
            Write-Host "LIVE KEY FOUND:" -ForegroundColor Red
            $hits | ForEach-Object { Write-Host "  $($_.Path):$($_.LineNumber)" }
            exit 1
        }
        Write-Host "   no live keys: OK ($($scanFiles.Count) files scanned)" -ForegroundColor Green
    }

    "ci" {
        # No eval step. eval-quick (below) calls eval.run, which is B13's
        # file (PLAN_DETAIL.md:518) and has never existed -- not since B4,
        # since the scaffold commit, 2026-08-25. Putting it in `ci` meant
        # this target could not pass from day one. Removed 2026-08-29;
        # DECISIONS.md has the full reasoning. Re-add once B13 lands.
        Invoke-Step "tests"  { & $Py -m pytest -q -m $TestFastFilter }
        Invoke-Step "guards" { & $Py scripts\guard_invariants.py --all }

        # Advisory, not Invoke-Step: a stale golden-set cache should warn,
        # never block session end. No live API calls -- a file-existence
        # check against the current prompt-content-hash version, so it's
        # cheap enough to run every time. This is B11's actual answer to
        # PLAN_DETAIL.md's "wired into the Stop hook": a full live run would
        # cost ~5.5 minutes on the first checkpoint after any prompt edit,
        # which is what caching was built to avoid (DECISIONS.md, 2026-08-30
        # and 2026-08-31).
        Write-Host "`n== golden-set cache freshness (advisory)" -ForegroundColor Cyan
        & $Py eval\golden_check.py --check-freshness
    }

    "eval" {
        & $Py -m eval.run --config eval/frozen/sim_config.yaml --all-regimes --both-profiles
        & $Py -m eval.report
    }
    "eval-quick" { & $Py -m eval.run --config eval/frozen/sim_config.yaml --regime nominal --profile strict --quiet }
    # NOTE: no "golden" case here. PowerShell's switch runs EVERY matching
    # branch unless a branch breaks, and a second "golden" case (the one that
    # honours -NoCache) lives further down. Having both meant `.\run.ps1
    # golden` ran the whole set twice -- once unconditionally cached, once
    # respecting the switch -- doubling a multi-minute run and, on -NoCache,
    # doubling the live API spend. Found while planning B12, 2026-08-31.
    # --n 200 --repeats 5 (the original pinned args) plans 600 live calls per
    # model against a 500/model/day free-tier cap, and cannot complete --
    # POSTMORTEM.md incident 8, where it died at call 400. 140 + 5*30*2 = 440
    # fits, and bench refuses to start anything that does not.
    "bench"      { & $Py bench\llm_vs_stats.py --n 140 --repeats 5 --variance-n 30 }
    "shadow"     { & $Py -m src.execute.shadow }
    "chaos"      { & $Py -m eval.chaos --kills=$Kills }
    "report"     { & $Py -m eval.report --figures --update-readme }

    "freeze" {
        git add eval/frozen
        git commit -m "FREEZE: evaluation protocol, pre-registered before any policy code"
        $hash = (git rev-parse HEAD).Trim()
        New-Item -ItemType Directory -Force -Path reports | Out-Null
        Set-Content -Path reports\FREEZE_HASH -Value $hash -NoNewline
        Write-Host "frozen at $hash" -ForegroundColor Green
    }

    "checkpoint" {
        # $TestFastFilter threaded through explicitly so checkpoint.py's
        # own test count cannot silently diverge from what test-fast/ci
        # actually run -- it previously hard-coded "not chaos" only,
        # missing "and not slow" (DECISIONS.md, 2026-08-29).
        if ($Day) { & $Py scripts\checkpoint.py $Day $TestFastFilter } else { & $Py scripts\checkpoint.py "" $TestFastFilter }
    }

    "state" { & $Py scripts\show_state.py }

    "verify" {
        # PS 5.1 turns native-exe stderr into a terminating NativeCommandError
        # under ErrorActionPreference=Stop. The guards below write to stderr by
        # design, so relax it for this block only.
        $ErrorActionPreference = "Continue"
        $fail = 0

        Write-Host "`n== 1. guard fires on a deliberate violation" -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path src\model | Out-Null
        # Probe the LIVE provider first, then a retired one. Testing only a
        # client we no longer use would be a green check that proves nothing
        # about the import that can actually reach the core today -- the same
        # vacuous shape audited out of the gates on 2026-08-29.
        foreach ($probe in @("from google import genai", "import anthropic")) {
            Set-Content -Path src\model\_check.py -Value $probe
            & $Py scripts\guard_invariants.py src\model\_check.py 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 2) {
                Write-Host "   PASS -- '$probe'" -ForegroundColor Green
            } else {
                Write-Host "   FAIL -- guard did not fire on '$probe'" -ForegroundColor Red
                $fail = 1
            }
            Remove-Item src\model\_check.py -ErrorAction SilentlyContinue
        }

        Write-Host "== 2. frozen guard denies" -ForegroundColor Cyan
        & $Py scripts\guard_frozen.py eval\frozen\sim_config.yaml 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 2) { Write-Host "   PASS" -ForegroundColor Green }
        else { Write-Host "   FAIL" -ForegroundColor Red; $fail = 1 }

        Write-Host "== 3. no live keys" -ForegroundColor Cyan
        $scanFiles3 = Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.yaml,*.yml,*.ps1,*.env* -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -notmatch '\\(\.venv|node_modules|\.git)\\' }
        # Same vacuous shape as lint's scan (DECISIONS.md, 2026-08-29):
        # zero files found must FAIL, not print PASS having checked nothing.
        if (-not $scanFiles3) {
            Write-Host "   FAIL -- zero files found to scan" -ForegroundColor Red; $fail = 1
        } else {
            $lk = $scanFiles3 | Select-String -Pattern "rzp_live_[A-Za-z0-9]{10,}" -ErrorAction SilentlyContinue
            if ($lk) { Write-Host "   FAIL -- live key found" -ForegroundColor Red; $fail = 1 }
            else { Write-Host "   PASS ($($scanFiles3.Count) files scanned)" -ForegroundColor Green }
        }

        Write-Host "== 4. postgres reachable" -ForegroundColor Cyan
        docker exec mrdb pg_isready 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "   PASS" -ForegroundColor Green }
        else { Write-Host "   FAIL -- is the mrdb container running?" -ForegroundColor Red; $fail = 1 }

        Write-Host "== 5. Razorpay test keys work" -ForegroundColor Cyan
        $probe = @"
import os, razorpay
from dotenv import load_dotenv, find_dotenv
# This probe is written to TEMP, and find_dotenv() walks up from the script's
# own directory -- not cwd -- so it would never see the repo .env.
load_dotenv(find_dotenv(usecwd=True))
kid = os.environ.get('RAZORPAY_KEY_ID','')
assert kid.startswith('rzp_test_'), 'KEY_ID is not a test key: ' + kid[:12]
c = razorpay.Client(auth=(kid, os.environ['RAZORPAY_KEY_SECRET']))
print(c.order.create({'amount':100,'currency':'INR'})['id'])
"@
        $tmp = Join-Path $env:TEMP "mr_probe.py"
        Set-Content -Path $tmp -Value $probe -Encoding UTF8
        & $Py $tmp
        Remove-Item $tmp -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -eq 0) { Write-Host "   PASS" -ForegroundColor Green }
        else { Write-Host "   FAIL -- check .env" -ForegroundColor Red; $fail = 1 }

        Write-Host ""
        if ($fail -eq 0) { Write-Host "ALL CHECKS PASSED" -ForegroundColor Green }
        else { Write-Host "SOMETHING FAILED -- fix before building" -ForegroundColor Red; exit 1 }
    }

    "serve" {
        & $Py -m uvicorn src.ingest.app:app --reload --port 8000
    }

    "coverage" { & $Py scripts\decline_coverage.py }

    "golden" {
        if ($NoCache) { & $Py eval\golden_check.py --no-cache }
        else { & $Py eval\golden_check.py }
    }

    "clean" {
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force .pytest_cache, .coverage -ErrorAction SilentlyContinue
        Write-Host "cleaned" -ForegroundColor Green
    }

    default {
        Write-Host "unknown task: $Task" -ForegroundColor Red
        Write-Host "run: .\run.ps1 help"
        exit 1
    }
}
