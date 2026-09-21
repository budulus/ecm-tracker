@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Resolve everything from this file, including when launched from a shortcut.
if not exist "%~dp0.venv\Scripts\pythonw.exe" goto missing_environment

start "" /d "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" -m app.main
if errorlevel 1 goto launch_failed
exit /b 0

:missing_environment
echo ECM Tracker's Python environment is missing.
echo Open a terminal in this application folder and run:
echo.
echo   uv sync
echo.
echo Then double-click this launcher again.
pause
exit /b 1

:launch_failed
echo ECM Tracker could not be started.
echo Open a terminal in this application folder and run:
echo.
echo   uv run tracker
echo.
echo This will display any startup errors.
pause
exit /b 1
