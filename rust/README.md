# ECM Tracker — Rust rewrite (`rust/`)

The compiled, cross-platform rewrite of ECM Tracker. **Rust host + egui GUI + in-process
embedded CPython (PyO3) for Python plugins.** See the full migration plan in
`../.claude/plans/please-review-this-codebase-scalable-starlight.md`.

> Status: **Phase 0 (scaffold + toolchain de-risk) complete.** The Python app in `../app/` is
> still the working application; this tree is the in-progress port.

## Workspace layout

```
crates/
  core/    # Qt-free CV pipeline + models — ports ../app/core + ../app/models. Depends on the
           # `opencv` crate (calcOpticalFlowPyrLK, goodFeaturesToTrack, cvtColor, …).
  gui/     # egui host app (binary `ecm-tracker`) — ports ../app/gui.
  pyhost/  # embedded CPython + PluginContext bridge — ports ../app/plugins SDK.
```

## Prerequisites

- **Rust** (MSVC toolchain on Windows) + cargo.
- **Visual Studio Build Tools** with the *Desktop development with C++* workload (provides the
  MSVC STL, `cl.exe`, and the Windows SDK). Rust's linker needs it; clang needs its headers.
- **uv** (manages the bundled standalone CPython).
- **OpenCV** prebuilt libs + **LLVM/Clang** toolchain — see one-time setup below.

`target/` must live **outside** the Dropbox tree (Dropbox/AV file-locking breaks Cargo writes).
Copy `.cargo/config.toml.example` → `.cargo/config.toml` and set an absolute `target-dir`
(both are gitignored / machine-specific).

## One-time external deps (Windows reference)

These were installed out-of-Dropbox under `%LOCALAPPDATA%\ecm-tracker`:

| Dep | How | Location used |
|---|---|---|
| Standalone CPython 3.12 | `uv python install 3.12`; `uv venv --python 3.12 <pyenv>`; `uv pip install --python <pyenv> numpy` | venv at `…\ecm-tracker\pyenv` |
| OpenCV 4.11.0 (prebuilt) | download `opencv-4.11.0-windows.exe`, silent SFX extract (`-o<dir> -y`) | `…\ecm-tracker\opencv\opencv\build` |
| LLVM/Clang 18.1.8 | download `clang+llvm-18.1.8-x86_64-pc-windows-msvc.tar.xz`, `tar -xf` (no admin) | `…\ecm-tracker\llvm\clang+llvm-18.1.8-…\bin` |

The `opencv` crate is trimmed to the modules the port needs and uses **runtime libclang
loading** (so a build-time `libclang.lib` isn't required):

```toml
opencv = { version = "0.95.1", default-features = false,
           features = ["imgproc", "imgcodecs", "video", "clang-runtime"] }
```

## Build / test environment

The `opencv` crate's binding generator runs **clang** to parse OpenCV's C++ headers, which needs
the MSVC environment and a couple of overrides. Set these before `cargo build`/`cargo test`
(see the working template at `%LOCALAPPDATA%\ecm-tracker\verify_all.ps1`):

```powershell
# MSVC INCLUDE/LIB (so clang finds <memory> etc.):
cmd /c "`"<VS>\VC\Auxiliary\Build\vcvars64.bat`" && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "Env:\$($matches[1])" $matches[2] } }

$env:OPENCV_INCLUDE_PATHS = "<opencv>\build\include"
$env:OPENCV_LINK_PATHS    = "<opencv>\build\x64\vc16\lib"
$env:OPENCV_LINK_LIBS     = "opencv_world4110"
$env:LIBCLANG_PATH        = "<llvm>\bin"
# MSVC STL 14.50 gates on Clang >=19 (STL1000); 18.1.8 needs this bypass.
# (Cleaner long-term fix: use a Clang >=19 toolchain and drop this.)
$env:OPENCV_CLANG_ARGS    = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

# Embedded Python: PYO3_PYTHON at build; PYTHONHOME + dll dir + site-packages at runtime.
$env:PYO3_PYTHON = "<pyenv>\Scripts\python.exe"
$env:PYTHONHOME  = "<cpython-base>"          # the uv-managed standalone install
$env:ECM_PY_SITE = "<pyenv>\Lib\site-packages"
$env:PATH        = "<llvm>\bin;<opencv>\build\x64\vc16\bin;<cpython-base>;$env:PATH"

cargo test --workspace
```

Runtime DLLs that must be on `PATH` (and later bundled by the installer):
`opencv_world4110.dll`, `python312.dll`, and `libclang.dll` is build-time only.

## macOS (later)

Homebrew supplies both deps cleanly (`brew install opencv llvm`); `pkg-config` lets the `opencv`
crate auto-detect, and Apple Clang/Homebrew LLVM avoids the MSVC-STL version dance. uv provides
the same standalone CPython. To be filled in when the Mac port starts.
