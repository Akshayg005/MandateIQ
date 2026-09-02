@echo off
REM ==========================================================================
REM  Mandate Recovery Engine -- double-click this file to start everything.
REM
REM  Brings up Postgres, the ingest API, the reviewer dashboard and the
REM  landing page, then opens the browser. Each server gets its own window.
REM
REM  Not named start.cmd on purpose: a file called start.cmd in this folder
REM  would shadow cmd's built-in `start` for anyone whose shell is sitting
REM  here, which is a confusing thing to inflict on a repo you have just
REM  cloned.
REM
REM  Equivalent to:  .\run.ps1 up      (and .\run.ps1 down to stop)
REM ==========================================================================

REM %~dp0 is this file's folder, so double-clicking works no matter what the
REM shell's current directory happens to be.
pushd "%~dp0"

REM -ExecutionPolicy Bypass because a freshly cloned repo on a default
REM Windows install cannot run an unsigned .ps1 otherwise, and telling a
REM reviewer to change a machine-wide policy to see a demo is not reasonable.
REM -NoProfile so a slow or broken user profile cannot affect startup.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" up
set "EXITCODE=%ERRORLEVEL%"

popd

if not "%EXITCODE%"=="0" (
    echo.
    echo Startup reported a problem ^(exit %EXITCODE%^).
    echo.
)

REM Keep this window open either way: when double-clicked from Explorer it
REM would otherwise vanish, taking the summary of what started -- and what
REM did not, and why -- with it.
echo.
echo Press any key to close this window. The server windows stay open.
pause >nul
exit /b %EXITCODE%
