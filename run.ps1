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
    [int]$Kills = 50
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
  .\run.ps1 test-fast         unit tests only, skips chaos
  .\run.ps1 lint              invariant guards across all tracked python
  .\run.ps1 ci                what the Stop hook runs (test-fast + lint + eval-quick)

  .\run.ps1 eval              full eval, all regimes, both compliance profiles
  .\run.ps1 eval-quick        nominal regime only, strict profile
  .\run.ps1 golden            golden-set regression on the LLM layer
  .\run.ps1 bench             LLM vs statistical core benchmark
  .\run.ps1 chaos -Kills 50   induced process kills
  .\run.ps1 report            regenerate all figures and tables

  .\run.ps1 freeze            BLOCK B2 ONLY -- commit and record the eval hash
  .\run.ps1 checkpoint -Day B4  end of session -- regenerate STATE.md
  .\run.ps1 state             print the session-start orientation block
  .\run.ps1 verify            full pre-flight: guards, keys, docker, hooks
  .\run.ps1 clean             remove caches

"@
    }

    "test"      { & $Py -m pytest -q --cov=src --cov-report=term-missing }
    "test-fast" { & $Py -m pytest -q -m "not chaos" }

    "lint" {
        Invoke-Step "invariant guards" { & $Py scripts\guard_invariants.py --all }
        Write-Host "== live-key scan" -ForegroundColor Cyan
        $hits = Select-String -Path (Get-ChildItem -Recurse -File -Exclude *.zip |
                Where-Object { $_.FullName -notmatch '\\(\.venv|node_modules|\.git)\\' }
            ).FullName -Pattern "rzp_live_[A-Za-z0-9]{10,}" -ErrorAction SilentlyContinue
        if ($hits) {
            Write-Host "LIVE KEY FOUND:" -ForegroundColor Red
            $hits | ForEach-Object { Write-Host "  $($_.Path):$($_.LineNumber)" }
            exit 1
        }
        Write-Host "   no live keys: OK" -ForegroundColor Green
    }

    "ci" {
        Invoke-Step "tests"  { & $Py -m pytest -q -m "not chaos" }
        Invoke-Step "guards" { & $Py scripts\guard_invariants.py --all }
        Invoke-Step "eval"   { & $Py -m eval.run --config eval/frozen/sim_config.yaml --regime nominal --profile strict --quiet }
    }

    "eval" {
        & $Py -m eval.run --config eval/frozen/sim_config.yaml --all-regimes --both-profiles
        & $Py -m eval.report
    }
    "eval-quick" { & $Py -m eval.run --config eval/frozen/sim_config.yaml --regime nominal --profile strict --quiet }
    "golden"     { & $Py -m eval.golden_check }
    "bench"      { & $Py bench\llm_vs_stats.py --n 200 --repeats 5 }
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
        if ($Day) { & $Py scripts\checkpoint.py $Day } else { & $Py scripts\checkpoint.py }
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
        Set-Content -Path src\model\_check.py -Value "import anthropic"
        & $Py scripts\guard_invariants.py src\model\_check.py 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 2) { Write-Host "   PASS" -ForegroundColor Green }
        else { Write-Host "   FAIL -- guard did not fire" -ForegroundColor Red; $fail = 1 }
        Remove-Item src\model\_check.py -ErrorAction SilentlyContinue

        Write-Host "== 2. frozen guard denies" -ForegroundColor Cyan
        & $Py scripts\guard_frozen.py eval\frozen\sim_config.yaml 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 2) { Write-Host "   PASS" -ForegroundColor Green }
        else { Write-Host "   FAIL" -ForegroundColor Red; $fail = 1 }

        Write-Host "== 3. no live keys" -ForegroundColor Cyan
        $lk = Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.yaml,*.yml,*.ps1,*.env* -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -notmatch '\\(\.venv|node_modules|\.git)\\' } |
              Select-String -Pattern "rzp_live_[A-Za-z0-9]{10,}" -ErrorAction SilentlyContinue
        if ($lk) { Write-Host "   FAIL -- live key found" -ForegroundColor Red; $fail = 1 }
        else { Write-Host "   PASS" -ForegroundColor Green }

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
