# ECM Tracker — Rust Rewrite: Progress & Continuation Guide

**Link this file when resuming in a fresh session.** It captures current state, the (non-obvious)
build environment, and the next steps. Companion docs:
- **Strategy / roadmap:** `../.claude/plans/please-review-this-codebase-scalable-starlight.md`
- **Build environment recipe:** `./README.md`

## TL;DR for resuming

- This is the in-progress **Rust + egui + embedded-Python** rewrite of the PyQt5 ECM Tracker.
  The Python app under `../app/` is the working reference; `rust/` is the port.
- **Done:** Phase 0 (scaffold + toolchain), Phase 1 (full `core`+`models` port, parity-tested
  bit-identical), Phase 2 slices 1–4 (canvas + ROI/detection + Run Tracking & overlays + Cleanup
  panel).
- Building needs a special environment (OpenCV + LLVM clang + MSVC vcvars + embedded Python).
  Use the helper: `pwsh "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1" <cargo args>`.
- **Next:** Phase 2 slice 5 — parameter dialogs (corner/grid/tracker/display), grid detection,
  Circle + N-Gon ROI tools, Export UI.

## Status

| Phase | State | Commit |
|---|---|---|
| 0 — scaffold + toolchain de-risk | ✅ done, pushed | `b0a3c26` |
| 1 — core + models port (parity-tested) | ✅ done, pushed | `3b25e05` |
| 2 slice 1 — canvas (open/display/zoom/pan/scrub) | ✅ done, pushed | `ca0f1ab` |
| 2 slice 2 — ROI rect + corner detection + overlays | ✅ done, pushed | `95f61a0` |
| 2 slice 3 — Run Tracking (bg thread + progress/cancel) + track overlays | ✅ done | `b468af7` (⚠ unpushed) |
| 2 slice 4 — Cleanup panel (band filters + live green/red preview + apply/undo) | ✅ done | ⚠ uncommitted |
| 2 slice 5+ — dialogs, grid detect, Circle/N-Gon ROI, Export UI, theme/icons | ⬜ next | — |

⚠ **`origin/main` is behind: slice 3 (`b468af7`) is committed but the push failed on Git
Credential Manager auth (`401 — credentials expired`), which this tool can't answer
interactively; slice 4 is verified locally but not yet committed.** Run `git push origin main`
yourself (it will carry both once slice 4 is committed); if it rejects, clear the
`placksiserver.tail87cfa8.ts.net` entry in Windows Credential Manager and retry.

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

> ✅ **Slice 3 — Run Tracking + overlays (done).** `core::tracking::track` runs on a **background
> thread** (egui can't pump events mid-call like the Qt `QProgressDialog`); progress + a cancel
> flag flow back over an `mpsc` channel to a centered progress Window (egui 0.30 has no `Modal`).
> On completion `state.result` + an all-true `active_mask` are set. `canvas::draw_tracks` draws
> the tracked points at the current cut index (green = kept, skips masked/non-finite points).
> Re-detecting or changing the ROI invalidates a stale result (`invalidate_tracking`).
> **Deviation from the original note:** motion-*trail* overlays were omitted — the Python
> reference (`canvas_view._draw_tracked`) draws only per-frame markers + an optional 1-frame
> trail/window-box gated on `display_params`; trails/box belong with the Display dialog (slice 5),
> so they were deferred to keep the port faithful. Add them when `DisplayParams` gets a UI.

1. **Slice 4 — Cleanup panel.** A side panel binding `core::cleanup::Thresholds` band filters;
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
