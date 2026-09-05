<#
.SYNOPSIS
  Day 0 environment bootstrap for Windows. Run from the repo root.

.DESCRIPTION
  Creates the venv, installs dependencies, starts Postgres in Docker,
  scaffolds both Vite workspaces, copies .env.example to .env, and
  self-tests the invariant guards.

  Assumes already installed: Python 3.13, Node 20+, Docker Desktop, git,
  and the Claude Code CLI.

.EXAMPLE
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\setup.ps1
#>
param([switch]$SkipFrontend)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg, $colour = "Cyan") { Write-Host "`n==> $msg" -ForegroundColor $colour }
function Ok($msg)  { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg){ Write-Host "    $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "    $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- prereqs --
Say "checking prerequisites"

$PyTarget = "3.13"
$pyExe = $null
foreach ($cand in @("py -3.13", "python3.13", "python")) {
    try {
        # Split into exe + args explicitly. Indexing $cand as an array yields
        # characters, and $parts[1..0] on a single-word candidate reverses the
        # range into a bogus argument -- both silently pick the wrong python.
        $parts = @($cand.Split(" "))
        if ($parts.Count -gt 1) { $v = & $parts[0] $parts[1] --version 2>&1 }
        else                    { $v = & $parts[0] --version 2>&1 }
        if ("$v" -match ([regex]::Escape($PyTarget))) { $pyExe = $cand; break }
    } catch { }
}
if (-not $pyExe) {
    Die "Python $PyTarget not found. Install from python.org and tick 'Add to PATH'. Then reopen PowerShell."
}
Ok "python: $pyExe"

foreach ($tool in @("node", "docker", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Die "$tool not found on PATH." }
}
Ok "node, docker, git present"

docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Die "Docker CLI is present but the engine is not running. Open Docker Desktop from the Start menu, wait for the whale icon in the system tray to stop animating, confirm ``docker info`` shows a Server section, then re-run this script."
}
Ok "docker daemon running"

# ------------------------------------------------------------------- venv --
Say "creating virtual environment"

# An existing .venv built on a different Python is the failure that wastes an
# afternoon: it is reused silently and every version-specific wheel is wrong.
# Check what is in there and rebuild if it does not match $PyTarget.
if (Test-Path ".venv\Scripts\python.exe") {
    $have = & ".venv\Scripts\python.exe" -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($have -ne $PyTarget) {
        Warn "existing .venv is Python $have, target is $PyTarget -- rebuilding"
        Remove-Item .venv -Recurse -Force
    }
}

if (-not (Test-Path ".venv")) {
    $pyParts = @($pyExe.Split(" "))
    if ($pyParts.Count -gt 1) {
        $realPy = & $pyParts[0] $pyParts[1] -c "import sys; print(sys.executable)"
    } else {
        $realPy = & $pyParts[0] -c "import sys; print(sys.executable)"
    }
    & $realPy -m venv .venv
    if ($LASTEXITCODE -ne 0) { Die "venv creation failed" }
}
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Die "venv python missing at $Py" }
Ok ".venv ready"

Say "installing dependencies (2-4 minutes)"
& $Py -m pip install --quiet --upgrade pip
& $Py -m pip install --quiet `
    fastapi "uvicorn[standard]" "psycopg[binary]" sqlalchemy alembic `
    statsmodels lifelines scikit-learn pandas numpy scipy mapie matplotlib `
    google-genai razorpay apscheduler python-dotenv pydantic-settings `
    pytest pytest-cov httpx freezegun pyyaml
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Die "pip install failed on Python $PyTarget. Read the error above for which package. If it is a missing wheel, that package has no Python $PyTarget build yet: pin an older version, or set `$PyTarget to 3.12 at the top of this script, delete the .venv folder, and re-run."
}
# -Encoding utf8 in PS 5.1 ALWAYS writes a BOM, and Out-File writes CRLF.
# `pip install -r` tolerates both, but nothing else does reliably, and R7
# needs this file to be a clean, platform-neutral install input for CI on
# ubuntu-latest. Written through .NET's UTF8Encoding($false) -- no BOM --
# with LF joins, so setup.ps1 and setup.sh produce the same bytes.
$frozen = (& $Py -m pip freeze) -join "`n"
[System.IO.File]::WriteAllText(
    (Join-Path $PSScriptRoot "requirements.txt"),
    $frozen + "`n",
    (New-Object System.Text.UTF8Encoding $false))
Ok "dependencies installed"

# --------------------------------------------------------------- postgres --
Say "starting postgres"
$existing = docker ps -a --filter "name=mrdb" --format "{{.Names}}"
if ($existing -eq "mrdb") {
    docker start mrdb 2>&1 | Out-Null
    Ok "existing mrdb container started"
} else {
    $inUse = docker ps --filter "publish=5432" --format "{{.Names}}"
    if ($inUse) { Warn "port 5432 already used by container '$inUse' -- stop it or change the port in .env" }
    docker run -d --name mrdb -e POSTGRES_PASSWORD=dev `
        -e POSTGRES_DB=mandate_recovery -p 5432:5432 postgres:16 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "postgres container failed to start" }
    Ok "mrdb container created on 5432"
}
Start-Sleep -Seconds 3
docker exec mrdb pg_isready 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Ok "postgres accepting connections" } else { Warn "postgres not ready yet -- give it a few seconds" }

# --------------------------------------------------------------- frontend --
if (-not $SkipFrontend) {
    Say "scaffolding frontend workspaces"
    if (-not (Test-Path "dashboard\src")) { npm create vite@latest dashboard -- --template react-ts }
    if (-not (Test-Path "site\src"))      { npm create vite@latest site -- --template react-ts }
    Push-Location site
    npm install --silent three "@react-three/fiber" "@react-three/drei" `
        "@react-three/postprocessing" gsap lenis
    Pop-Location
    Ok "dashboard/ and site/ ready"
}

# -------------------------------------------------------------------- env --
Say "environment file"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Ok ".env created -- add your TEST-mode keys"
} else {
    Ok ".env already exists, left alone"
}

# ------------------------------------------------------------ guard check --
Say "verifying the invariant guards actually fire"
New-Item -ItemType Directory -Force -Path src\model | Out-Null
Set-Content -Path src\model\_guardcheck.py -Value "import anthropic"
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Py scripts\guard_invariants.py src\model\_guardcheck.py 2>&1 | Out-Null
$guardExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Remove-Item src\model\_guardcheck.py -ErrorAction SilentlyContinue

if ($guardExit -ne 2) {
    Write-Host ""
    Die "GUARD DID NOT FIRE. Do not build on a repo where invariants are not enforced -- by day eight an LLM import will have crept into the decision core and you will not know when. Check that scripts\guard_invariants.py and scripts\hookio.py both exist."
}
Ok "guards OK"

# ------------------------------------------------------------------- next --
Write-Host @"

==> NEXT STEPS

  1. Put TEST-mode Razorpay keys in .env  (rzp_test_... only)
     Generate a webhook secret with:
       `$b = New-Object byte[] 32
       [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes(`$b)
       (`$b | ForEach-Object { `$_.ToString("x2") }) -join ""

  2. Wire the Razorpay MCP server (PowerShell):
       `$kid = "rzp_test_xxxxx"
       `$sec = "xxxxx"
       `$auth = "Basic " + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("`$kid`:`$sec"))
       `$cfg = @{ command = "npx"
                 args = @("mcp-remote","https://mcp.razorpay.com/mcp","--header","Authorization:`$auth") } | ConvertTo-Json -Compress
       claude mcp add-json razorpay `$cfg
       claude mcp list

  3. Create the parallel worktree for the landing page:
       git worktree add ..\mr-site site

  4. Verify everything:
       .\run.ps1 verify

  5. Open Claude Code, press shift+tab twice for plan mode, and paste the
     prompt from OPUS_PROMPT.md

  Then follow PLAN.md, Day 1.
  NOTE: on Windows use  .\run.ps1 <task>  wherever PLAN.md says  make <task>

"@ -ForegroundColor White
