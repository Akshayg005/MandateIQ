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

# --- helpers for `up` / `down` ---------------------------------------------
# Each long-running server gets its OWN console window rather than a
# background job. A judge watching a demo needs to see uvicorn's reload log
# and vite's port line, and needs to be able to close one thing without
# killing the rest; a PowerShell job hides both.
#
# Returns the wrapper process so `up` can record its id. Window TITLES are not
# a usable handle here: a process started from another session reports an
# empty MainWindowTitle, so matching on it finds nothing and silently does
# not clean up. The pid file is deterministic.
function Start-Pane([string]$Title, [string]$WorkDir, [string]$Command) {
    $inner = "`$host.UI.RawUI.WindowTitle = '$Title'; Set-Location '$WorkDir'; $Command"
    return Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $inner
    ) -WindowStyle Normal -PassThru
}

# Where `up` records the console windows it opened, so `down` can close them.
$PaneFile = Join-Path $PSScriptRoot ".run-panes.json"

# Poll rather than sleep a fixed amount: a cold `npm run dev` on this repo is
# ~1s and a cold uvicorn ~4s, but a first run that has to compile is far
# slower, and a fixed sleep either wastes time or opens the browser at a
# connection-refused page.
function Wait-Url([string]$Url, [int]$TimeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        } catch {
            # A server that is up but returns 4xx/5xx still counts as
            # listening -- only a connection failure means "not yet".
            if ($_.Exception.Response) { return $true }
            Start-Sleep -Milliseconds 400
        }
    }
    return $false
}

function Test-Command([string]$Name) {
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

# npm install is slow and almost never needed; only run it when the folder is
# genuinely absent, so a normal `up` is a couple of seconds.
function Confirm-NodeModules([string]$Dir) {
    if (Test-Path (Join-Path $Dir "node_modules")) { return $true }
    Write-Host "   installing npm dependencies in $Dir (first run only)..." -ForegroundColor Yellow
    Push-Location $Dir
    try {
        if (Test-Path "package-lock.json") { npm ci --no-audit --no-fund }
        else { npm install --no-audit --no-fund }
    } finally { Pop-Location }
    return (Test-Path (Join-Path $Dir "node_modules"))
}

switch ($Task.ToLower()) {

    "help" {
        Write-Host @"

Mandate Recovery Engine -- tasks

  .\run.ps1 test              full test suite with coverage
                               NEEDS POSTGRES -- 132 tests cover the ledger,
                               executor and crash-recovery path and FAIL, not
                               skip, without it. Start it with .\run.ps1 up.
                               MANDATEIQ_ALLOW_PG_SKIP=1 restores skipping.
  .\run.ps1 test-fast         unit tests only, skips chaos and slow/simulation
  .\run.ps1 lint              invariant guards across all tracked python
  .\run.ps1 ci                test-fast + lint. Does NOT run eval -- eval.run
                               is B13's file (PLAN_DETAIL.md:518) and has
                               never existed; see DECISIONS.md, 2026-08-29

  .\run.ps1 eval              full eval, all regimes, both compliance profiles
  .\run.ps1 eval-quick        baseline regime, nominal arm, strict profile
  .\run.ps1 golden            golden-set regression on the LLM layer (cached;
                               -NoCache forces a live call on every row)
  .\run.ps1 bench             LLM vs statistical core benchmark
  .\run.ps1 shadow            decide without executing; delta vs the fixed ladder
  .\run.ps1 chaos -Kills 50   induced process kills
  .\run.ps1 report            re-render tables+figures from the last run

  .\run.ps1 freeze            BLOCK B2 ONLY -- commit and record the eval hash
  .\run.ps1 checkpoint -Day B4  end of session -- regenerate STATE.md
  .\run.ps1 up                START EVERYTHING -- db, api, dashboard, site
  .\run.ps1 down              stop everything `up` started
  .\run.ps1 state             print the session-start orientation block
  .\run.ps1 verify            full pre-flight: guards, keys, docker, hooks
  .\run.ps1 serve             run the webhook ingest API (uvicorn, port 8000)
  .\run.ps1 dashboard         B14 -- export per-mandate artifact, stage, serve
  .\run.ps1 dashboard-build   stage + lint + production build of the dashboard
  .\run.ps1 site              B15 -- stage results.json, serve the landing page
  .\run.ps1 site-build        stage + lint + production build of the landing page
  .\run.ps1 coverage          decline_class / UNKNOWN-rate breakdown from ingested_event
  .\run.ps1 clean             remove caches

"@
    }

    # Invoke-Step, NOT a bare call. A bare `& $Py ...` as the last statement
    # of a switch branch does NOT set this script's exit code: PowerShell
    # returns 0 and the failure vanishes. `.\run.ps1 test` therefore reported
    # success on a RED suite, which made CLAUDE.md's own definition-of-done
    # step 3 ("`.\run.ps1 test` passes before any commit") unfalsifiable.
    # Found 2026-08-31 in the B13 end-of-project pass and proven with a
    # minimal repro; same class as the 2026-08-29 vacuous-checks audit.
    # Invoke-Step existed and was correct -- it was simply never called from
    # any branch except ci and lint.
    "test"      { Invoke-Step "tests"      { & $Py -m pytest -q --cov=src --cov-report=term-missing } }
    "test-fast" { Invoke-Step "tests-fast" { & $Py -m pytest -q -m $TestFastFilter } }

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

    # B13's gate: "every number reproducible by one command". THIS is that
    # command -- it re-runs the whole sweep and re-renders every table and
    # figure from the artifact it just wrote, so reports/regimes.md can never
    # drift from reports/regimes.json.
    # --seeds 8 is the PUBLISHED configuration, not a nicety. Everything in
    # B13's first draft was seed 0 with no error bar, and its central
    # comparison (the engine against a model-free one-attempt policy) turns
    # on a handful of mandates. The report's headline is a per-seed sign test
    # over 256 paired comparisons; it cannot be produced from one seed.
    # Costs ~15 minutes wall clock. Use eval-quick for a fast smoke check.
    "eval" {
        Invoke-Step "sweep"  { & $Py -m eval.run --config eval/frozen/sim_config.yaml --all-regimes --both-profiles --seeds 8 }
        Invoke-Step "report" { & $Py -m eval.report --figures }
    }
    # "nominal" is an ARM, not a regime -- the regimes are baseline,
    # issuer_outage, delayed_salary, stacking_spike, festival_season,
    # retry_storm (eval/regimes.py). This line predated that file.
    "eval-quick" { Invoke-Step "eval-quick" { & $Py -m eval.run --config eval/frozen/sim_config.yaml --regime baseline --arm nominal --profile strict --quiet } }
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
    "bench"      { Invoke-Step "bench" { & $Py bench\llm_vs_stats.py --n 140 --repeats 5 --variance-n 30 } }
    "shadow"     { Invoke-Step "shadow" { & $Py -m src.execute.shadow } }
    "chaos"      { Invoke-Step "chaos" { & $Py -m eval.chaos --kills=$Kills } }
    # Re-renders from the EXISTING artifact without re-running the sweep.
    # Use .\run.ps1 eval for the full reproduce-from-scratch path.
    "report"     { Invoke-Step "report" { & $Py -m eval.report --figures } }

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

    # B14. Regenerates the per-mandate artifact, stages all three reports into
    # dashboard/public/data, then serves. The export needs Postgres (it builds
    # its own throwaway schema and drops it again); the dashboard itself does
    # not, which is why staging is a copy and not a live query -- `npm run
    # build` must produce a bundle that still works on a clone with Docker
    # down.
    "dashboard" {
        Invoke-Step "export" { & $Py -m eval.export_mandates }
        Invoke-Step "stage"  { & $Py scripts\dashboard_data.py }
        Push-Location dashboard
        try { npm run dev } finally { Pop-Location }
    }

    "dashboard-build" {
        Invoke-Step "stage" { & $Py scripts\dashboard_data.py }
        Push-Location dashboard
        try {
            Invoke-Step "lint"   { npm run lint }
            Invoke-Step "build"  { npm run build }
            # tsc proves it compiles; this proves it RENDERS, against the
            # artifacts the sweep actually wrote, and that B14's five
            # drill-down fields reach the output.
            Invoke-Step "render" { npm run render-check }
        } finally { Pop-Location }
    }

    # B15. The landing page stages results.json only -- its headline figures
    # are means over 8 seeds, and mandates.json is the seed-0 batch, so the
    # two must not share a page. Same staging script as the dashboard, so
    # there is one answer to "where did that number come from".
    "site" {
        Invoke-Step "stage" { & $Py scripts\dashboard_data.py site }
        Push-Location site
        try { npm run dev } finally { Pop-Location }
    }

    "site-build" {
        Invoke-Step "stage" { & $Py scripts\dashboard_data.py site }
        Push-Location site
        try {
            Invoke-Step "lint"  { npm run lint }
            Invoke-Step "build" { npm run build }
            # Proves the gate: real figures reach the rendered output, and
            # PLAN.md's storyboard placeholders do not.
            Invoke-Step "render" { npm run render-check }
        } finally { Pop-Location }
    }

    # ------------------------------------------------------------------
    # Start the whole project: Postgres, the ingest API, the reviewer
    # dashboard and the landing page, each in its own window.
    #
    # Degrades on purpose rather than refusing to start. Docker down means no
    # API, but both front-ends read STAGED JSON and still come up -- that is
    # the state a reviewer clones into, and the demo they most need to see is
    # the one that does not require a database. Anything skipped is reported
    # at the end with the reason, so nothing fails silently.
    # ------------------------------------------------------------------
    "up" {
        Write-Host "`n  MANDATE RECOVERY ENGINE -- starting everything" -ForegroundColor Cyan
        Write-Host "  ---------------------------------------------`n"

        $skipped = @()
        $started = @()
        $panes = @()

        # 1. Postgres -----------------------------------------------------
        $pgReady = $false
        Write-Host "== postgres" -ForegroundColor Cyan
        if (-not (Test-Command "docker")) {
            Write-Host "   SKIP -- docker not on PATH" -ForegroundColor Yellow
            $skipped += "postgres (no docker)"
        } else {
            docker start mrdb 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "   SKIP -- container 'mrdb' would not start (is Docker Desktop running?)" -ForegroundColor Yellow
                $skipped += "postgres (container did not start)"
            } else {
                for ($i = 0; $i -lt 30; $i++) {
                    docker exec mrdb pg_isready 2>&1 | Out-Null
                    if ($LASTEXITCODE -eq 0) { $pgReady = $true; break }
                    Start-Sleep -Milliseconds 500
                }
                if ($pgReady) {
                    Write-Host "   ready on localhost:15432" -ForegroundColor Green
                    $started += "postgres  localhost:15432"
                } else {
                    Write-Host "   SKIP -- container started but never accepted connections" -ForegroundColor Yellow
                    $skipped += "postgres (never became ready)"
                }
            }
        }

        # 2. Report artifacts --------------------------------------------
        # Both front-ends render reports/ and recompute nothing, so if the
        # eval has never been run there is nothing to show. Say so precisely;
        # the page's own error state says the same thing.
        Write-Host "`n== report artifacts" -ForegroundColor Cyan
        if (-not (Test-Path "reports\results.json")) {
            Write-Host "   MISSING reports\results.json -- run .\run.ps1 eval first." -ForegroundColor Yellow
            Write-Host "   The UIs will start and show their 'could not load results' state." -ForegroundColor Yellow
            $skipped += "staged data (no reports\results.json)"
        } else {
            & $Py scripts\dashboard_data.py      | Out-Null
            & $Py scripts\dashboard_data.py site | Out-Null
            Write-Host "   staged into dashboard\public\data and site\public\data" -ForegroundColor Green
        }

        # 3. Node dependencies -------------------------------------------
        Write-Host "`n== node dependencies" -ForegroundColor Cyan
        $haveNode = Test-Command "npm"
        if (-not $haveNode) {
            Write-Host "   SKIP -- npm not on PATH; no front-end will start" -ForegroundColor Yellow
            $skipped += "dashboard and site (no npm)"
        } else {
            $dashOk = Confirm-NodeModules (Join-Path $PSScriptRoot "dashboard")
            $siteOk = Confirm-NodeModules (Join-Path $PSScriptRoot "site")
            Write-Host "   ok" -ForegroundColor Green
        }

        # 4. The servers --------------------------------------------------
        Write-Host "`n== launching" -ForegroundColor Cyan

        if ($pgReady) {
            # No --reload here, unlike `serve`. The reloader is a supervisor
            # whose multiprocessing worker INHERITS the listening socket, so
            # when the supervisor dies the port stays served by an orphan that
            # Get-NetTCPConnection still attributes to the dead parent -- which
            # made `down` unable to free port 8000. Nobody is editing source
            # during a demo, so the reloader buys nothing and costs that.
            $panes += (Start-Pane "MandateIQ api" $PSScriptRoot "& '$Py' -m uvicorn src.ingest.app:app --port 8000").Id
            Write-Host "   api        http://localhost:8000/docs" -ForegroundColor Green
            $started += "api       http://localhost:8000/docs"
        } else {
            Write-Host "   api        SKIPPED -- needs postgres" -ForegroundColor Yellow
            $skipped += "api (needs postgres)"
        }

        if ($haveNode -and $dashOk) {
            $panes += (Start-Pane "MandateIQ dashboard" (Join-Path $PSScriptRoot "dashboard") "npm run dev").Id
            Write-Host "   dashboard  http://localhost:4317" -ForegroundColor Green
            $started += "dashboard http://localhost:4317"
        }
        if ($haveNode -and $siteOk) {
            $panes += (Start-Pane "MandateIQ site" (Join-Path $PSScriptRoot "site") "npm run dev").Id
            Write-Host "   site       http://localhost:4318" -ForegroundColor Green
            $started += "site      http://localhost:4318"
        }
        if ($panes.Count) { $panes | ConvertTo-Json -Compress | Set-Content $PaneFile -Encoding UTF8 }

        # 5. Wait, then open ONE tab --------------------------------------
        # The landing page only. It links through to the dashboard ("Open the
        # data") and the dashboard links back ("Overview"), so opening both
        # would just hand the reader two tabs and no idea which to look at
        # first. The dashboard is still waited on, because a link to a server
        # that is not up yet is worse than a slower start.
        Write-Host "`n== waiting for servers" -ForegroundColor Cyan
        $landing = $null
        if ($haveNode -and $dashOk) {
            if (Wait-Url "http://localhost:4317" 90) {
                Write-Host "   dashboard is up" -ForegroundColor Green
            } else { Write-Host "   dashboard did not answer in 90s -- check its window" -ForegroundColor Yellow }
        }
        if ($haveNode -and $siteOk) {
            if (Wait-Url "http://localhost:4318" 90) {
                Write-Host "   site is up" -ForegroundColor Green
                $landing = "http://localhost:4318"
            } else { Write-Host "   site did not answer in 90s -- check its window" -ForegroundColor Yellow }
        }
        # Fall back to the dashboard if the landing page is the thing that
        # failed, so `up` still lands the reader somewhere useful.
        if (-not $landing -and $haveNode -and $dashOk) { $landing = "http://localhost:4317" }
        if ($landing) { Start-Process $landing | Out-Null }

        # 6. Summary ------------------------------------------------------
        Write-Host "`n  ---------------------------------------------" -ForegroundColor Cyan
        Write-Host "  RUNNING" -ForegroundColor Green
        foreach ($s in $started) { Write-Host "    $s" }
        if ($skipped.Count) {
            Write-Host "`n  NOT RUNNING" -ForegroundColor Yellow
            foreach ($s in $skipped) { Write-Host "    $s" -ForegroundColor Yellow }
        }
        Write-Host "`n  Each server has its own window. Close them, or run" -ForegroundColor DarkGray
        Write-Host "  .\run.ps1 down  to stop everything at once.`n" -ForegroundColor DarkGray
    }

    # Stops what `up` started. Deliberately targets THIS project's ports
    # rather than killing every node/python on the machine.
    "down" {
        Write-Host "`n== stopping servers" -ForegroundColor Cyan
        foreach ($port in 8000, 4317, 4318) {
            # Loop, and kill the whole tree.
            #
            # `uvicorn --reload` is a supervisor whose child worker inherits
            # the listening socket, so Stop-Process on the pid the socket
            # reports leaves the worker alive and the port still served --
            # measured, not theorised: `down` said "stopped pid 14596" and
            # http://localhost:8000/docs kept answering. taskkill /T takes the
            # children with it, and re-querying catches whatever survives a
            # pass.
            $killedAny = $false
            for ($pass = 0; $pass -lt 5; $pass++) {
                $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
                if (-not $conns) { break }
                foreach ($pid_ in ($conns.OwningProcess | Sort-Object -Unique)) {
                    if (Get-Process -Id $pid_ -ErrorAction SilentlyContinue) {
                        taskkill /PID $pid_ /T /F 2>&1 | Out-Null
                        Write-Host "   :$port  stopped pid $pid_ (and children)" -ForegroundColor Green
                        $killedAny = $true
                        continue
                    }
                    # The owner is already dead but the port is still served:
                    # a child inherited the socket and Windows still reports
                    # the socket against the parent. Kill the orphans by
                    # parentage -- there is nothing else left to match on.
                    $orphans = Get-CimInstance Win32_Process -Filter "ParentProcessId=$pid_" -ErrorAction SilentlyContinue
                    foreach ($o in $orphans) {
                        taskkill /PID $($o.ProcessId) /T /F 2>&1 | Out-Null
                        Write-Host "   :$port  stopped orphan pid $($o.ProcessId) (parent $pid_ already gone)" -ForegroundColor Green
                        $killedAny = $true
                    }
                }
                Start-Sleep -Milliseconds 400
            }
            $left = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
            if ($left) {
                Write-Host "   :$port  STILL LISTENING -- stop it by hand" -ForegroundColor Red
            } elseif (-not $killedAny) {
                Write-Host "   :$port  nothing listening"
            }
        }
        # Killing the listener leaves the -NoExit wrapper window sitting there
        # empty, so close the windows `up` recorded. Only ids from that file
        # are touched, and only if they are still powershell -- a pid gets
        # reused fast on Windows, and a stale file must never take out
        # whatever happens to hold that number now.
        Write-Host "`n== closing server windows" -ForegroundColor Cyan
        if (-not (Test-Path $PaneFile)) {
            Write-Host "   no record of any (.run-panes.json absent)"
        } else {
            # [int[]] on purpose: in Windows PowerShell 5.1 ConvertFrom-Json
            # emits the whole array as ONE pipeline object, so a bare @(...)
            # yields a single element holding an Object[] and the loop body
            # runs once with every id at once.
            $ids = [int[]](Get-Content $PaneFile -Raw | ConvertFrom-Json)
            $closed = 0
            foreach ($paneId in $ids) {
                $proc = Get-Process -Id $paneId -ErrorAction SilentlyContinue
                if (-not $proc) { continue }
                if ($proc.ProcessName -ne "powershell") {
                    Write-Host "   skipped pid $paneId -- now '$($proc.ProcessName)', not ours" -ForegroundColor Yellow
                    continue
                }
                try {
                    Stop-Process -Id $paneId -Force -ErrorAction Stop
                    Write-Host "   closed window pid $paneId" -ForegroundColor Green
                    $closed++
                } catch {
                    Write-Host "   could not close pid $paneId -- $($_.Exception.Message)" -ForegroundColor Yellow
                }
            }
            if ($closed -eq 0) { Write-Host "   none still open" }
            Remove-Item $PaneFile -ErrorAction SilentlyContinue
        }

        Write-Host "`n== stopping postgres" -ForegroundColor Cyan
        if (Test-Command "docker") {
            docker stop mrdb 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Host "   mrdb stopped" -ForegroundColor Green }
            else { Write-Host "   mrdb was not running" }
        } else { Write-Host "   docker not on PATH" }
        Write-Host ""
    }

    "coverage" { Invoke-Step "coverage" { & $Py scripts\decline_coverage.py } }

    "golden" {
        if ($NoCache) { Invoke-Step "golden (live)"   { & $Py eval\golden_check.py --no-cache } }
        else          { Invoke-Step "golden (cached)" { & $Py eval\golden_check.py } }
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
