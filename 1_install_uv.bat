@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Avoid reinstalling uv if it is already available.
uv.exe --version >nul 2>&1
if not errorlevel 1 goto already_installed
"%USERPROFILE%\.local\bin\uv.exe" --version >nul 2>&1
if not errorlevel 1 goto already_installed

echo Installing uv using the official Astral installer...
echo An internet connection is required.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression"
if errorlevel 1 goto install_failed

echo.
echo uv installation completed.
goto next_steps

:already_installed
echo uv is already installed.

:next_steps
echo.
echo Close and reopen any terminals so they pick up uv on PATH.
echo Open a new terminal in this application folder and run:
echo.
echo   uv sync
echo.
echo Then double-click "Start ECM Tracker.bat".
pause
exit /b 0

:install_failed
echo.
echo uv installation failed. Check the error above and your internet connection.
echo You can retry this file or follow the manual installation instructions:
echo https://docs.astral.sh/uv/getting-started/installation/
pause
exit /b 1
