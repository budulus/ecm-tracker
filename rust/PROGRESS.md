# ECM Tracker — Rust Rewrite: Progress & Continuation Guide

**Link this file when resuming in a fresh session.** It captures current state, the (non-obvious)
build environment, and the next steps. Companion docs:
- **Strategy / roadmap:** `../.claude/plans/please-review-this-codebase-scalable-starlight.md`
- **Build environment recipe:** `./README.md`

## TL;DR for resuming

- This is the in-progress **Rust + egui + embedded-Python** rewrite of the PyQt5 ECM Tracker.
  The Python app under `../app/` is the working reference; `rust/` is the port.
- **Done:** Phase 0 (scaffold + toolchain), Phase 1 (full `core`+`models` port, parity-tested
  bit-identical), **Phase 2 complete (slices 1–6)** — canvas + ROI/detection + Run Tracking &
  overlays + Cleanup panel + parameter dialogs + grid detection + Export UI + Display wiring +
  Circle/N-Gon ROI tools + LK window-box overlay + **light theme & SVG toolbar icons**.
  **Phase 3 (plugin host) STARTED:** slices **3a** (`#[pyclass] PluginContext` over an immutable
  state snapshot, scalar/list read API) + **3b** (rust-numpy arrays — `coords()` / `track_status()`
  / `active_mask` / `point_indices()`), both driven from embedded Python in tests.
- Building needs a special environment (OpenCV + LLVM clang + MSVC vcvars + embedded Python).
  Use the helper: `pwsh "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1" <cargo args>`.
- **Next:** Phase 3 slice **3c** — plugin discovery + loading (port `manager.py`) + a Rust-side
  Python SDK (`TrackerPlugin` base, `PluginContext` `#[pymodule]`) + a Plugins menu in the GUI;
  then 3d overlays, 3e events + `apply_keep_mask` write-back + panels.

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
| 2 slice 5b — Circle + N-Gon ROI tools, window-box overlay | ✅ done | `7e8f8bb` |
| 2 slice 6 — light theme + SVG toolbar icons | ✅ done | `8edfe0f` |
| 3a — plugin host: `PluginContext` pyclass (read-bridge, snapshot) | ✅ done | `f912a82` |
| 3b — plugin host: rust-numpy arrays (coords/status/mask/indices) | ✅ done | `44f7cdc` |
| 3c — plugin discovery + loading + Plugins menu + Python SDK base | ⬜ next | — |
| 3d — overlays as host-rendered draw-commands + canvas integration | ⬜ later | — |
| 3e — event hub + `apply_keep_mask` write-back + panels + settings | ⬜ later | — |

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
pwsh $CE test  --workspace --manifest-path .\Cargo.toml     # 16 tests incl. parity
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
           Pan + Rect/Circle/N-Gon ROI tools (rubber-band drag for Rect/Circle; N-Gon = click-add
           / right-click-close / Esc-cancel), Detect Corners + Detect Grid, Run Tracking (bg thread
           + progress Window + cancel), Clear Tracking, Cleanup side-panel, ⚙ Params menu (4
           dialogs + Save-as-defaults), Export menu (.npy/.csv via rfd). src/canvas.rs =
           image↔screen Transform (fit→zoom→pan), texture draw, and draw_roi / draw_points /
           draw_tracks (alpha + green-kept / red-preview-drop) / draw_window_boxes (LK search
           window, zoom-scaled) overlays. src/theme.rs = light egui Visuals (LIGHT_QSS palette
           port, applied to the context at startup). src/icons.rs = Lucide SVGs embedded from
           app/gui/icons via include_str! → resvg/tiny-skia render → alpha-recolored, cached egui
           textures (toolbar glyphs via a tool_button helper + the OS window icon).
  pyhost/  embedded CPython (pyo3 0.27) + rust-numpy 0.27. src/context.rs = `#[pyclass]
           PluginContext` over an immutable `ContextSnapshot` (plain data, no core types) — the
           Rust port of `app/plugins/api.py:PluginContext`. Scalar/list reads (counts, frame
           indices, global↔cut conversions, image_size, roi_corners) + NumPy arrays
           (`coords(active_only)` / `track_status(active_only)` / `active_mask` / `point_indices()`,
           cut-indexed, active_only selects kept columns). Plus the Phase-0 numpy-import smoke.
```
- **Tests: 16**, including **bit-identical** parity vs Python: `tests/tracking_parity.rs`,
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

> ✅ **Slice 5b — Circle & N-Gon ROI tools + window box (done).** `main.rs::handle_roi_interaction`
> routes canvas mouse input to the active tool: Rect/Circle share `handle_roi_drag` (press fixes the
> anchor, drag previews live via `drag_corners`, release commits — or discards a sub-`MIN_ROI_SIZE`
> drag); Circle emits a 64-gon via `circle_corners` + `Roi::new`. N-Gon (`handle_ngon`) appends a
> vertex per left-click (`Roi::add_corner`, starting a fresh polygon when the last was closed),
> right-click closes once it has ≥ `Roi::MIN_CORNERS`, Esc abandons; leaving the tool drops an
> unclosed polygon. New `Tool::RoiCircle`/`RoiNgon` + toolbar buttons (`◯`/`△`, Geometric-Shapes
> glyphs so font coverage matches the existing `▭`). The optional LK window-box overlay
> (`canvas::draw_window_boxes`, gated on `display_params.show_window_box` — now a Display-dialog
> checkbox) draws a green, zoom-scaled `result.win_size` square around each active point, under the
> markers (matching `canvas_view._draw_tracked`). Faithful to `app/gui/roi_tools.py`. Build +
> 12 tests + `ECM_SMOKE` all green. **Deferred (as before):** per-frame motion *trails* — they
> belong with the trail/Display work, not this slice.

> ✅ **Slice 6 — Light theme + SVG toolbar icons (done).** `theme::apply` sets egui `Visuals` from
> the `LIGHT_QSS` palette (window #f4f5f7 / surface #fff / border #d6dade / text #2b2f33 / accent
> #2563eb; buttons flat until hover; accent selection) once on the context at startup — egui has no
> stylesheet, so it's a palette port, not a 1:1 QSS reproduction. `icons` embeds the Lucide SVGs
> (`include_str!` from `app/gui/icons`), renders them with **resvg 0.47 + tiny-skia**, recolors by
> alpha (the Qt `SourceIn` recolor: keep coverage-alpha, replace RGB with the tint), and caches the
> egui textures by (name, color); the full-color `app_icon.svg` becomes the OS window icon. Toolbar
> buttons use a `tool_button` helper (icon-or-text + `Button::selected` toggle + accent-filled Run);
> Circle/N-Gon keep geometric-shape glyphs (no bundled icon). Build + 12 tests + `ECM_SMOKE` green.

> ✅ **Slice 3a — `PluginContext` read-bridge (done).** `pyhost::context` defines a `#[pyclass]
> PluginContext` wrapping an immutable `ContextSnapshot` (plain data — no `ecm-core`/opencv types,
> so the GUI layer assembles it from `ProjectState`). Exposes the scalar/list read surface of
> `app/plugins/api.py`: `has_sequence`/`has_result`/`n_total_images`, `image_size()`,
> `reference_index`/`last_index`/`current_index`/`current_cut`, `global_to_cut`/`cut_to_global`,
> `frame_count`/`point_count`/`n_active`, `roi_corners`. **Design choice:** plugins read from a
> *snapshot*, not a live `&mut EcmApp` (pyo3 `#[pyclass]` is `'static`; in-process embedded Python
> can't safely hold a live borrow). The single write (`apply_keep_mask`) becomes a host-applied
> command in 3e. Two tests drive the pyclass through Python's `getattr`/`call_method` (proving the
> bridge is exposed to embedded Python) + assert the index math. No new deps (pyo3 only). 14 tests.

> ✅ **Slice 3b — rust-numpy arrays (done).** Added `numpy = "0.27"` + `ndarray = "0.16"` to
> `pyhost` (verified: numpy 0.27.1 needs `pyo3 ^0.27` + `ndarray >=0.15,<=0.17`, so both unify with
> the existing pins; 0.28 is held back by the workspace's rustc-1.80 pin). `ContextSnapshot` gained
> `coords_fw` (Array3) / `status_fw` (Array2) / `active_mask` (Vec<bool>), and `PluginContext` now
> serves `coords(active_only)` → `(frames,P,2)` f32, `track_status(active_only)` → `(frames,P)` u8,
> `active_mask` → `(P,)` bool, and `point_indices()` → kept column indices. `active_only=True`
> (default) selects the kept columns via `ndarray.select(Axis(1), …)` (an owned array →
> `into_pyarray`); the full case copies via `to_pyarray`. Cut-indexed on axis 0. Two tests drive
> all four from Python, asserting shapes + the [0,2,3] column selection + values; a no-result
> snapshot returns None for every array. 16 tests.

1. **Slice 3c — discovery + menu + SDK.** Port `manager.py` (scan `plugins/`, import
   `plugins.<name>`, find `PLUGIN`/`TrackerPlugin` subclass, instantiate, isolate errors); define a
   Rust-side Python SDK module (`TrackerPlugin` base + the `PluginContext` registered as a
   `#[pymodule]`); add a **Plugins** menu to the GUI; ship a Rust-compatible example/test plugin.
2. **Slice 3d — overlays.** Host-rendered draw-commands (plugins return shapes in image space; the
   Rust canvas renders them via the existing overlay hook) replacing the Qt `OverlayFn`.
3. **Slice 3e — events + mutation + panels.** Event hub (replaces `PluginSignals`), the undoable
   `apply_keep_mask` write-back command, declared egui control panels, plugin settings persistence.
4. **Phase 4** — port the 3 example plugins to the new API. **Phase 5** — packaging (`cargo-packager`).

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
- **Theming/icons (egui 0.30 + resvg 0.47, verified before coding):** egui has no stylesheet —
  theme via `Visuals` (`Visuals::light()` then override `panel_fill`/`window_fill`/`extreme_bg_color`
  /`selection`/`widgets.{noninteractive,inactive,hovered,active,open}`). `WidgetVisuals` fields are
  `bg_fill / weak_bg_fill / bg_stroke / rounding / fg_stroke / expansion`; `Rounding::same(f32)`
  (NOT egui 0.31's `corner_radius`/u8). `Button` has `image_and_text` / `selected(bool)` / `fill`.
  resvg API: `usvg::Tree::from_str(&str, &usvg::Options)`, `tree.size().width()` (a **method**),
  `resvg::render(&tree, Transform, &mut pixmap.as_mut())` (transform **by value**); resvg re-exports
  `usvg` + `tiny_skia` (`tiny_skia::Pixmap::pixels()` → `[PremultipliedColorU8]`, `.alpha()` /
  `.demultiply()`). The window icon wants **unmultiplied** RGBA (`IconData`); recolor keeps alpha
  and swaps RGB so premultiplied-vs-not doesn't matter there.
- **Build invocation:** see the Commands note above (absolute manifest path; benign vswhere
  warning; background-task-to-log if interactive output capture misbehaves).

## Verification approach

Parity-first: numerical modules are validated **bit-identically** against the live Python
implementation via committed fixtures. GUI slices are verified by a headless `ECM_SMOKE` run
(asserts frames load, texture uploads, detection runs) plus manual `cargo run`.
