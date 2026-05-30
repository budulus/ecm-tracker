# ECM Tracker — Rust Rewrite: Progress & Continuation Guide

**Link this file when resuming in a fresh session.** It captures current state, the (non-obvious)
build environment, and the next steps. Companion docs:
- **Strategy / roadmap:** `../.claude/plans/please-review-this-codebase-scalable-starlight.md`
- **Build environment recipe:** `./README.md`

## TL;DR for resuming

- This is the in-progress **Rust + egui + embedded-Python** rewrite of the PyQt5 ECM Tracker.
  The Python app under `../app/` is the working reference; `rust/` is the port.
- **Done:** Phase 0 (scaffold + toolchain), Phase 1 (full `core`+`models` port, parity-tested
  bit-identical), Phase 2 slices 1–2 (canvas + ROI/detection).
- Building needs a special environment (OpenCV + LLVM clang + MSVC vcvars + embedded Python).
  Use the helper: `pwsh "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1" <cargo args>`.
- **Next:** Phase 2 slice 3 — Run Tracking (with progress) + tracked-point/trail overlays.

## Status

| Phase | State | Commit |
|---|---|---|
| 0 — scaffold + toolchain de-risk | ✅ done, pushed | `b0a3c26` |
| 1 — core + models port (parity-tested) | ✅ done, pushed | `3b25e05` |
| 2 slice 1 — canvas (open/display/zoom/pan/scrub) | ✅ done | `ca0f1ab` (⚠ unpushed) |
| 2 slice 2 — ROI rect + corner detection + overlays | ✅ done | latest commit (⚠ unpushed) |
| 2 slice 3+ — tracking, cleanup, dialogs, theme/icons | ⬜ next | — |

⚠ **`origin/main` is behind local `main`** — the Forgejo push fails on Git Credential Manager
auth (interactive prompt this tool can't answer). Run `git push origin main` yourself; if it
rejects, clear the `placksiserver.tail87cfa8.ts.net` entry in Windows Credential Manager.

## Build environment (critical — read before building)

External deps live **outside Dropbox** under `%LOCALAPPDATA%\ecm-tracker\` (per-machine, like
`.venv`):
- `pyenv\` — uv-managed standalone CPython 3.12 + numpy (for embedded Python / `pyhost`)
- `opencv\` — OpenCV 4.11.0 prebuilt (the `opencv` crate links `opencv_world4110`)
- `llvm\` — clang+llvm 18.1.8 (the `opencv` crate's binding generator needs `clang.exe` + libclang)
- helper scripts: **`cargoenv.ps1`** (run any cargo command with the full env), `build_opencv.ps1`,
  `verify_all.ps1`

`target/` is redirected out of Dropbox via `rust/.cargo/config.toml` (gitignored — copy from
`.cargo/config.toml.example` per machine; Dropbox file-locking breaks in-tree builds).

Non-obvious build requirements (all encoded in `cargoenv.ps1`; full story in `README.md`):
- `opencv` crate uses the **`clang-runtime`** feature (loads libclang.dll dynamically).
- clang must run inside **`vcvars64`** so it finds the MSVC STL (`INCLUDE`).
- `OPENCV_CLANG_ARGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` (clang 18 vs MSVC STL 14.50's
  Clang-≥19 gate; cleaner long-term fix is a Clang ≥19 toolchain).

## Commands (run from `rust/`)

```powershell
$CE = "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1"
pwsh $CE build --workspace --manifest-path .\Cargo.toml     # build everything
pwsh $CE test  --workspace --manifest-path .\Cargo.toml     # 11 tests incl. parity
pwsh $CE run -p ecm-tracker --manifest-path .\Cargo.toml    # launch the GUI window
```
Headless GUI smoke check (loads fixtures, detects corners, exits): set `ECM_SMOKE=1` and
`ECM_SMOKE_DIR=<rust>\crates\core\tests\fixtures\frames`, run the built exe with the opencv `bin`
on `PATH`.

Regenerate the parity fixture (uses the app's `.venv`, which has the real `cv2`), from repo root:
```
uv run python rust/crates/core/tests/fixtures/gen_fixture.py
```

## Architecture / what's ported

```
rust/crates/
  core/    Qt-free port of app/core + app/models. Modules: image_sequence, roi,
           feature_detection, tracking, cleanup, export, settings, result
           (TrackerResult/LkParams), project_state (dual global/cut index model).
           GUI helpers: ImageSequence::load_rgba, feature_detection::detect_corners.
  gui/     egui app. src/main.rs = app shell (ProjectState, toolbar, frame slider,
           Open Folder, Pan/RoiRect tools, Detect Corners). src/canvas.rs = image↔screen
           Transform (fit→zoom→pan), texture draw, draw_roi/draw_points overlays.
  pyhost/  embedded CPython (pyo3 0.27) — only a numpy-import smoke test so far (Phase 3).
```
- **Tests: 11**, including **bit-identical** parity vs Python: `tests/tracking_parity.rs`,
  `tests/cleanup_parity.rs` (both 0.000000 diff), plus `project_state.rs`, `settings_roundtrip.rs`,
  and unit tests. Fixtures (`tests/fixtures/*.npy` + `frames/*.png`) are committed (a fixtures-local
  `.gitignore` re-includes the `.npy` past the repo-root `*.npy` rule).
- **Key decisions:** faithful 1:1 port validated by parity; `ndarray` pinned **0.16** (ndarray-npy
  compat); `serde` `#[serde(default)]` reproduces Python's `{**DEFAULT, **section}` param merge;
  `opencv` 0.95 trimmed to `imgproc,imgcodecs,video`.

## Next steps (priority order)

1. **Phase 2 slice 3 — Run Tracking + overlays.** Wire `core::tracking::track` (it takes an
   optional `progress` callback → drive a progress bar / cancel). On completion set
   `state.result` and a default all-true `active_mask`. Add tracked-point markers + motion-trail
   overlays (green = kept) in `canvas.rs`, drawn from `coords_fw` at the current cut index.
2. **Slice 4 — Cleanup panel.** A side panel binding `core::cleanup::Thresholds` band filters;
   `compute_metrics` once per result; live green/red preview mask; apply (`active_mask &= keep`) +
   undo (snapshot stack). Mirror `app/gui/cleanup_dialog.py`.
3. **Slice 5 — Dialogs + extras.** Parameter dialogs (corner/grid/tracker/display) with
   Save-as-defaults via `settings::update_section`; Circle + N-Gon ROI tools; grid detection;
   Export UI (`core::export`).
4. **Slice 6 — Theme + icons.** egui light style; render `app/gui/icons/*.svg` via `resvg`+
   `tiny-skia` → recolored egui textures.
5. **Phase 3 — Plugin host (`pyhost`).** `PluginContext` `#[pyclass]`, zero-copy NumPy via the
   `numpy` crate, host-rendered `OverlayPainter` draw-commands, event-bus signals, declared UI
   panels, `apply_keep_mask`. Bundle numpy/scipy/opencv-python/matplotlib into the plugin env.
6. **Phase 4** — port the 3 example plugins. **Phase 5** — packaging/installer (`cargo-packager`).

## Verification approach

Parity-first: numerical modules are validated **bit-identically** against the live Python
implementation via committed fixtures. GUI slices are verified by a headless `ECM_SMOKE` run
(asserts frames load, texture uploads, detection runs) plus manual `cargo run`.
