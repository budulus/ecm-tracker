<#
.SYNOPSIS
  Assemble a self-contained, portable ECM Tracker bundle (Phase 5 packaging).

.DESCRIPTION
  Builds the release exe (via this machine's cargoenv.ps1) and lays out a folder that runs WITHOUT
  cargoenv:

      ecm-tracker/
        ecm-tracker.exe
        python312.dll  python3.dll  vcruntime140.dll  vcruntime140_1.dll   (interpreter runtime)
        opencv_world4110.dll  [+ opencv_videoio_*4110_64.dll]               (OpenCV runtime)
        python/      consolidated embedded interpreter — the python-build-standalone base (stdlib in
                     Lib/, extension modules in DLLs/) with the venv's site-packages overlaid
                     (numpy / scipy / matplotlib / PyQt5)
        plugins/     the rust/plugins tree (the 3 ported examples + scaffolds)

  At startup the exe calls `ecm_pyhost::bootstrap_embedded_env()`, which (when it finds a sibling
  `python/`) sets PYTHONHOME / ECM_PY_SITE and prepends the exe dir to PATH — so the folder is
  relocatable and needs no cargoenv. Plugins are found via the exe-adjacent `plugins/` (no env var).

  The output is large (hundreds of MB) and lands OUTSIDE the repo by default, so it is neither
  Dropbox-synced nor committed.

.PARAMETER Dist
  Output folder (default: %LOCALAPPDATA%\ecm-tracker\dist\ecm-tracker).

.PARAMETER SkipBuild
  Reuse the existing release exe instead of rebuilding.

.EXAMPLE
  pwsh -NoProfile .\package.ps1
#>
param(
    [string]$Dist = "$env:LOCALAPPDATA\ecm-tracker\dist\ecm-tracker",
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path            # ...\rust
$manifest = Join-Path $here 'Cargo.toml'
$CE = "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1"

# Source locations (mirror cargoenv.ps1's paths).
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "$env:LOCALAPPDATA\ecm-tracker\target" }
$exe = Join-Path $target 'release\ecm-tracker.exe'
$pybase = "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none"
$pyenv = "$env:LOCALAPPDATA\ecm-tracker\pyenv"
$ocvBin = "$env:LOCALAPPDATA\ecm-tracker\opencv\opencv\build\x64\vc16\bin"
$plugins = Join-Path $here 'plugins'

function Invoke-Robocopy($src, $dst) {
    robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy '$src' -> '$dst' failed ($LASTEXITCODE)" }
}

if (-not $SkipBuild) {
    Write-Host "[package] building release exe via cargoenv..."
    & pwsh -NoProfile $CE build --release -p ecm-tracker --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw "release build failed ($LASTEXITCODE)" }
}

foreach ($p in @($exe, $pybase, $pyenv, $ocvBin, $plugins)) {
    if (-not (Test-Path $p)) { throw "missing source: $p" }
}

Write-Host "[package] clean $Dist"
if (Test-Path $Dist) { Remove-Item $Dist -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Dist | Out-Null

Write-Host "[package] exe + runtime DLLs"
Copy-Item $exe (Join-Path $Dist 'ecm-tracker.exe')
foreach ($dll in 'python312.dll', 'python3.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $pybase $dll) $Dist
}
Copy-Item (Join-Path $ocvBin 'opencv_world4110.dll') $Dist
foreach ($dll in 'opencv_videoio_ffmpeg4110_64.dll', 'opencv_videoio_msmf4110_64.dll') {
    $src = Join-Path $ocvBin $dll
    if (Test-Path $src) { Copy-Item $src $Dist }
}

Write-Host "[package] embedded python base (stdlib) -> python\"
$pydst = Join-Path $Dist 'python'
Invoke-Robocopy $pybase $pydst
Write-Host "[package] overlay venv site-packages -> python\Lib\site-packages\"
Invoke-Robocopy "$pyenv\Lib\site-packages" "$pydst\Lib\site-packages"

Write-Host "[package] plugins -> plugins\"
Invoke-Robocopy $plugins (Join-Path $Dist 'plugins')

$size = (Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("[package] DONE: {0}  ({1:N0} MB)" -f $Dist, $size)
Write-Host "[package] verify:  `$env:ECM_SMOKE=1; `$env:ECM_SMOKE_DIR='<frames>'; & '$Dist\ecm-tracker.exe'"
