# ECM Tracker — Rust Rewrite: Progress & Continuation Guide

**Link this file when resuming in a fresh session.** It captures current state, the (non-obvious)
build environment, and the next steps. Companion docs:
- **Strategy / roadmap:** `../.claude/plans/please-review-this-codebase-scalable-starlight.md`
- **Build environment recipe:** `./README.md`

## TL;DR for resuming

- This is the in-progress **Rust + egui + embedded-Python** rewrite of the PyQt5 ECM Tracker.
  The Python app under `../app/` is the working reference; `rust/` is the port.
- **Done:** Phase 0 (scaffold + toolchain), Phase 1 (full `core`+`models` port, parity-tested
  bit-identical), Phase 2 slices 1–4 + **5a** (canvas + ROI/detection + Run Tracking & overlays +
  Cleanup panel + parameter dialogs, grid detection, Export UI, Display wiring).
- Building needs a special environment (OpenCV + LLVM clang + MSVC vcvars + embedded Python).
  Use the helper: `pwsh "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1" <cargo args>`.
- **Next:** finish Phase 2 slice 5 — **Circle + N-Gon ROI tools** (canvas interaction) and the
  optional window-box overlay; then slice 6 (theme + icons).

## Status

| Phase | State | Commit |
|---|---|---|
| 0 — scaffold + toolchain de-risk | ✅ done, pushed | `b0a3c26` |
| 1 — core + models port (parity-tested) | ✅ done, pushed | `3b25e05` |
| 2 slice 1 — canvas (open/display/zoom/pan/scrub) | ✅ done, pushed | `ca0f1ab` |
| 2 slice 2 — ROI rect + corner detection + overlays | ✅ done, pushed | `95f61a0` |
| 2 slice 3 — Run Tracking (bg thread + progress/cancel) + track overlays | ✅ done, pushed | `b468af7` |
| 2 slice 4 — Cleanup panel (band filters + live green/red preview + apply/undo) | ✅ done, pushed | `7f35c8d` |
| 2 slice 5a — param dialogs + Save-defaults, grid detect, Export UI, Display wiring | ✅ done, pushed | `8ffe412` |
| 2 slice 5b — Circle + N-Gon ROI tools, window-box overlay | ⬜ next | — |
| 2 slice 6 — theme + icons | ⬜ later | — |

> Note: Forgejo pushes go over Tailscale + Git Credential Manager and can intermittently fail with
> `401 — credentials expired` (GCM needs an interactive prompt this tool can't answer). If a push
> rejects, run `git push origin main` yourself; if it still rejects, clear the
> `placksiserver.tail87cfa8.ts.net` entry in Windows Credential Manager and retry.

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
pwsh $CE test  --workspace --manifest-path .\Cargo.toml     # 12 tests incl. parity
pwsh $CE run -p ecm-tracker --manifest-path .\Cargo.toml    # launch the GUI window
```
`cargoenv.ps1` changes the working dir, so when invoking it from elsewhere pass an **absolute**
`--manifest-path` (…\rust\Cargo.toml). The `'vswhere.exe' is not recognized` line it prints is a
**benign warning** — vcvars still sets up the env and the build succeeds. Invoke with
`pwsh -NoProfile`. If interactive output capture looks empty/garbled, run the command as a
background task that tees to a log file and read the log — that has been reliable here.

Headless GUI smoke check (loads fixtures → detects corners → tracks → reports counts → exits):
set `ECM_SMOKE=1` and `ECM_SMOKE_DIR=<rust>\crates\core\tests\fixtures\frames`, then run the built
exe with the opencv `bin` (`%LOCALAPPDATA%\ecm-tracker\opencv\opencv\build\x64\vc16\bin`) on `PATH`.
A pass prints `[smoke] frames=12 … tracked_points=267 …`.

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
           GUI helpers: ImageSequence::load_rgba, feature_detection::detect_corners,
           settings::save_section<T: Serialize>.
  gui/     egui app. src/main.rs = app shell: ProjectState, toolbar, frame slider, Open Folder,
           Pan/RoiRect tools, Detect Corners + Detect Grid, Run Tracking (bg thread + progress
           Window + cancel), Clear Tracking, Cleanup side-panel, ⚙ Params menu (4 dialogs +
           Save-as-defaults), Export menu (.npy/.csv via rfd). src/canvas.rs = image↔screen
           Transform (fit→zoom→pan), texture draw, and draw_roi / draw_points / draw_tracks
           (alpha + green-kept / red-preview-drop) overlays.
  pyhost/  embedded CPython (pyo3 0.27) — only a numpy-import smoke test so far (Phase 3).
```
- **Tests: 12**, including **bit-identical** parity vs Python: `tests/tracking_parity.rs`,
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

> ✅ **Slice 4 — Cleanup panel (done).** Right `SidePanel` bound to `core::cleanup::Thresholds`:
> `compute_metrics` once per result (with ROI + image size), then per band an enable checkbox + a
> single max-threshold `DragValue` (7 bands, "keep iff metric ≤ hi", clamped to the finite data
> ceiling) plus the two boolean filters (left-image / left-ROI). Live green/red preview is
> recomputed each frame via `build_mask` and drawn by `canvas::draw_tracks`; Apply
> (`active_mask &= keep`, undoably) / Undo / Close. Mirrors `app/gui/cleanup_dialog.py`.
> **Deferred:** cleanup-threshold settings persistence (`thresholds_to/from_value`) — lands with
> the slice-5 Save-as-defaults work.
> ✅ **Slice 5a — Dialogs + extras (done).** Four parameter dialogs (corner/grid/tracker/display)
> as egui `Window`s editing the live `ProjectState` params, each with **Save as defaults** via
> `settings::save_section`. **Grid detection** (`regular_grid`, clipped to a complete ROI) on a
> "Detect Grid" button. **Export** menu (`.npy` via `export::export`, `.csv` via `export_csv`)
> behind a native `rfd` save dialog, enabled only when active points remain. The **Display**
> dialog now feeds the overlays live (show-markers / show-ROI / marker size / opacity); the LK
> window-box overlay is deferred to slice 5b.

1. **Slice 5b — Circle & N-Gon ROI tools (+ window box).** The Rect ROI is a plain primary-drag
   in `main.rs::handle_roi_draw`; Circle/N-Gon need a small **canvas interaction** path. Mirror
   `app/gui/roi_tools.py` (RectangleTool / CircleTool / NGonTool): Circle = centre press + drag
   radius → emit a many-sided polygon (Python uses ~64 pts) via `Roi::new(corners)`; N-Gon =
   left-click `Roi::add_corner`, right-click `Roi::close`, Esc cancels. Add `Tool` variants +
   toolbar buttons and route image-space mouse events from `canvas.rs` (it returns the click
   `Response`; `Transform::screen_to_image` converts). Then the optional LK window-box overlay
   gated on `display_params.show_window_box`: draw `result.win_size`-sized rects in image space
   (scaled by zoom) around each active tracked point.
2. **Slice 6 — Theme + icons.** egui light style; render `app/gui/icons/*.svg` via `resvg` +
   `tiny-skia` → recolored egui textures.
3. **Phase 3 — Plugin host (`pyhost`).** `PluginContext` `#[pyclass]`, zero-copy NumPy via the
   `numpy` crate, host-rendered `OverlayPainter` draw-commands, event-bus signals, declared UI
   panels, `apply_keep_mask`. Bundle numpy/scipy/opencv-python/matplotlib into the plugin env.
4. **Phase 4** — port the 3 example plugins. **Phase 5** — packaging/installer (`cargo-packager`).

## Gotchas learned (egui 0.30 + core API + build)

- **egui is pinned to 0.30** (`crates/gui/Cargo.toml`, `eframe = "0.30.0"`). There is **no
  `egui::Modal`** — use a centered `Window`. `DragValue::clamp_range` is deprecated → use
  `.range(lo..=hi)` (present in 0.30). `menu_button` + `ui.close_menu()`, `Window::open(&mut bool)`,
  and `Color32::from_rgba_unmultiplied` are all available.
- **Verify against the real `core` API before coding** (these are easy to misremember and cost two
  build round-trips this session): `cleanup::compute_metrics(result, Option<&Roi>, (h,w)) ->
  Result<Metrics>`; `cleanup::Thresholds` has *named* `BandFilter` fields (each a single max `hi`,
  keep iff `metric <= hi`) + bools `drop_left_image` / `drop_left_roi`; `cleanup::build_mask(
  &Metrics, &Thresholds) -> Vec<bool>`; `feature_detection::regular_grid(&Roi, sx: f64, sy: f64)
  -> Result`; `ShiTomasiParams.use_harris_detector`, `GridParams { spacing_x, spacing_y }`;
  `export::export(...) -> NpyExport { shape: (N,P,2) }`, `export::export_csv(...) -> (PathBuf,
  (N,P,2))`; `settings::update_section(name, Value)` / `settings::save_section<T: Serialize>`.
- **Build invocation:** see the Commands note above (absolute manifest path; benign vswhere
  warning; background-task-to-log if interactive output capture misbehaves).

## Verification approach

Parity-first: numerical modules are validated **bit-identically** against the live Python
implementation via committed fixtures. GUI slices are verified by a headless `ECM_SMOKE` run
(asserts frames load, texture uploads, detection runs) plus manual `cargo run`.
