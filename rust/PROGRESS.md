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
  / `active_mask` / `point_indices()`) + **3c** (the `ecm_host` SDK module: `PluginContext` +
  a `TrackerPlugin` base, importable from embedded Python) + **3d** (plugin **discovery + loading** —
  `discover`/`launch`, the headless port of `manager.py`) + **3e** (**Plugins menu + GUI wiring** —
  the gui crate now drives the host: `ProjectState`→`ContextSnapshot`, a Plugins menu, an example
  plugin) + **3f** (**canvas overlays** — `OverlayPainter` draw-commands a plugin builds in
  `overlay(self, painter)`; the canvas renders them; an example overlay plugin) + **3g-a**
  (**reactive event hub** — five `on_*` plugin hooks dispatched at state transitions, replacing the
  3f overlay state-signature poll) + **3g-b** (**`apply_keep_mask` write-back** — the one plugin
  mutation: `ctx.apply_keep_mask` records a keep-mask the host applies through its undoable path).
  3a–3d/3g-a/3g-b driven from embedded Python in tests; 3e/3f verified end-to-end via `ECM_SMOKE`.
- Building needs a special environment (OpenCV + LLVM clang + MSVC vcvars + embedded Python).
  Use the helper: `pwsh "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1" <cargo args>`.
- **Next:** Phase 3 slice **3g-c** — declared egui control panels (a serializable widget spec the
  host renders as egui — sliders/checkboxes/buttons — with value-changes wired back to the plugin).
  3g-a (event hub) and 3g-b (`apply_keep_mask` write-back) are **done**. Then 3g-d (plugin settings
  persistence), Phase 4 (port the 3 example plugins), Phase 5 (packaging).

## Working method — use subagents, keep the main context lean

This is a long, slice-by-slice port; the main context fills fast from reading reference files,
grepping crate sources, and fetching docs. **Delegate context-heavy work to subagents (the `Agent`
tool) and keep only their conclusions.** Each slice, before/while coding:

1. **Contract reading → subagent.** To learn what a slice must port (e.g. `app/plugins/manager.py`,
   `app/gui/*.py`), spawn a subagent (`Explore` or `general-purpose`) to read it and return a
   **compact contract** — method signatures, behaviors, edge cases — not the file text.
2. **API verification → subagent (do this every slice — it's why builds rarely round-trip here).**
   Spawn a subagent to confirm exact crate APIs before coding: grep the local registry source
   (`~/.cargo/registry/src/index.crates.io-*/<crate>-<ver>/src/`) and/or WebFetch docs.rs, and have
   it return **only** the verified signatures + a tiny usage snippet + gotchas. Fold those into the
   Gotchas section. When checking several crates/files, launch the subagents **in parallel** (one
   message, multiple `Agent` calls).
3. **Build/test → background + Grep, never full-log reads.** Run `cargoenv.ps1` test/build as a
   background task teeing to a log; pull results with **Grep** (`test result`, `warning:`,
   `error\[`), don't `Read` the whole log. Hand a genuinely failing log to a subagent for root-cause
   and get back the diagnosis only.
4. **Optional — delegate a whole well-specified slice.** When a slice is fully spelled out below,
   you may hand the entire implement→build→test loop to one subagent (it edits, runs the cargoenv
   tests, returns the diff + results); the main loop then reviews, updates this file, and commits.
5. **Stays in the main loop:** slice planning + design decisions, the commit/push, and this file's
   updates — so the durable record and your judgment stay in context.

Rule of thumb: if a step means reading more than ~100 lines you won't edit, delegate it and keep the
conclusion. (See `MEMORY.md` → the matching feedback note.)

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
| 3c — plugin host: `ecm_host` SDK module (PluginContext + TrackerPlugin base) | ✅ done | `dfa860d` |
| 3d — plugin discovery + loading (port `manager.py`) | ✅ done | `cd23163` |
| 3e — Plugins menu in GUI + `ProjectState`→`ContextSnapshot` wiring | ✅ done | `85915fe` |
| 3f — overlays as host-rendered draw-commands + canvas integration | ✅ done | `818e837` |
| 3g-a — reactive event hub (`on_*` hooks, replaces 3f overlay poll) | ✅ done | `db8a3f8` |
| 3g-b — `apply_keep_mask` write-back (undoable plugin mutation) | ✅ done | `c62ed50` |
| 3g-c — declared egui control panels | ⬜ next | — |
| 3g-d — plugin settings persistence | ⬜ | — |

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
pwsh $CE test  --workspace --manifest-path .\Cargo.toml     # 22 tests incl. parity
pwsh $CE run -p ecm-tracker --manifest-path .\Cargo.toml    # launch the GUI window
```
`cargoenv.ps1` changes the working dir, so when invoking it from elsewhere pass an **absolute**
`--manifest-path` (…\rust\Cargo.toml). The `'vswhere.exe' is not recognized` line it prints is a
**benign warning** — vcvars still sets up the env and the build succeeds. Invoke with
`pwsh -NoProfile`. If interactive output capture looks empty/garbled, run the command as a
background task that tees to a log file and read the log — that has been reliable here.

Headless GUI smoke check (loads fixtures → detects corners → tracks → discovers/launches plugins →
refreshes overlays → reports counts → exits): set `ECM_SMOKE=1`, `ECM_SMOKE_DIR=<rust>\crates\core\
tests\fixtures\frames`, and `ECM_PLUGINS_DIR=<rust>\plugins` (exercises the plugin+overlay path).
**Easiest:** run via `pwsh $CE run -p ecm-tracker --manifest-path <abs>\Cargo.toml` (cargoenv provides
the opencv + embedded-Python env); otherwise run the built exe with the opencv `bin`
(`%LOCALAPPDATA%\ecm-tracker\opencv\opencv\build\x64\vc16\bin`) + the Python env on `PATH`. A pass
prints `[smoke] frames=12 … tracked_points=267 …`, `[smoke] plugins_discovered=2 loaded=[…]`, and
`[smoke] overlay_plugins=1 draw_commands=267`.

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
           dialogs + Save-as-defaults), Export menu (.npy/.csv via rfd), Plugins menu (lazy
           discover + launch via pyhost). src/plugins.rs = GUI↔pyhost glue: ProjectState→
           ContextSnapshot map, plugins_dir resolution, discover()/launch()/refresh_overlays()/
           dispatch_events() (+ the `PluginEvent` enum, slice 3g-a) under Python::attach
           (+ ensure_embedded_site); `launch()` now also returns `LaunchOutcome.keep_mask` (slice
           3g-b, applied by the GUI's undoable `apply_keep_mask`). Example plugins in rust/plugins/. src/
           canvas.rs = image↔screen Transform (fit→zoom→pan), texture draw, and draw_roi /
           draw_points / draw_tracks (alpha + green-kept / red-preview-drop) / draw_window_boxes (LK
           search window, zoom-scaled) / draw_overlay_commands (plugin DrawCommands) overlays.
           src/theme.rs = light egui Visuals (LIGHT_QSS palette
           port, applied to the context at startup). src/icons.rs = Lucide SVGs embedded from
           app/gui/icons via include_str! → resvg/tiny-skia render → alpha-recolored, cached egui
           textures (toolbar glyphs via a tool_button helper + the OS window icon).
  pyhost/  embedded CPython (pyo3 0.27) + rust-numpy 0.27. src/context.rs = `#[pyclass]
           PluginContext` over an immutable `ContextSnapshot` (plain data, no core types) — the
           Rust port of `app/plugins/api.py:PluginContext`. Scalar/list reads (counts, frame
           indices, global↔cut conversions, image_size, roi_corners) + NumPy arrays
           (`coords(active_only)` / `track_status(active_only)` / `active_mask` / `point_indices()`,
           cut-indexed, active_only selects kept columns) + the one mutation `apply_keep_mask(keep)`
           (slice 3g-b: records a full-length keep-mask into `pending_keep` — length P or n_active,
           ValueError otherwise; the host takes + applies it, so the context stays read-only). src/sdk.rs = `register_sdk(py)` injects
           the `ecm_host` module into sys.modules (PluginContext class + a Python `TrackerPlugin`
           base) so embedded plugins can `import ecm_host`. src/host.rs = `discover(py, dir)` +
           `launch(record, snapshot)` + `PluginRecord` — the headless port of `manager.py`'s
           discovery/loading (scan, import, resolve `PLUGIN`/`TrackerPlugin` subclass, isolate
           errors) + instantiate/has_overlay/overlay_commands (slice 3f) + dispatch_event (slice
           3g-a: refresh ctx, call a plugin's `on_<event>` hook) + take_keep_mask (slice 3g-b: take
           the `pending_keep` a plugin recorded via `ctx.apply_keep_mask`). src/overlay.rs =
           DrawCommand enum + `#[pyclass] OverlayPainter` builder (the canvas overlay API). Plus the
           Phase-0 numpy-import smoke. `ensure_embedded_site(py)` (pub, make `ECM_PY_SITE` numpy
           importable) + test-only `interp_test_lock()` (serialize the shared interpreter) in lib.rs.
```
- **Tests: 22**, including **bit-identical** parity vs Python: `tests/tracking_parity.rs`,
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

> ✅ **Slice 3c — `ecm_host` SDK module (done).** `pyhost::sdk::register_sdk(py)` injects an
> `ecm_host` module into `sys.modules` (idempotent) so embedded plugins can `import ecm_host`. It
> exposes the Rust `PluginContext` (`m.add_class`) and a Python-defined `TrackerPlugin` base
> (NAME/DESCRIPTION, `__init__(ctx)`, `launch`, `on_unload`) `py.run` into the module dict from a
> raw C-string (`cr#"..."#`). **Design choice:** sys.modules injection (not `append_to_inittab!`)
> keeps pyo3's `auto-initialize` — the inittab route panics if the interpreter is already running
> and needs manual `Python::initialize()` first. `TrackerPlugin` is pure Python (not a
> `#[pyclass(subclass)]`) so Python subclasses get a normal `__dict__`/super() without pyclass
> subclassing/`dict`-flag friction. Two tests: `import ecm_host` exposes both symbols; and the full
> contract — a plugin imports the SDK, subclasses `TrackerPlugin`, receives a `PluginContext`, reads
> `point_count` in `launch()`. 18 tests.

> ✅ **Slice 3d — discovery + loading (done, `cd23163`).** `pyhost::host` ports the headless half of
> `app/plugins/manager.py`. `discover(py, dir)` registers the `ecm_host` SDK, makes `dir` importable
> (**appended** to `sys.path`, guarded against duplicates — append not insert-at-0 so a plugin folder
> can't shadow a stdlib/site-packages module), scans sorted subdirs (skip `.`/`_`, require
> `__init__.py`), imports each as a top-level package, and resolves the class (a module-level
> `PLUGIN`, else the first in-package `TrackerPlugin` subclass — via `Bound::cast::<PyType>` +
> `PyTypeMethods::is_subclass`, excluding the base, requiring `__module__` in-package). Reads
> `NAME`(or id)/`DESCRIPTION` into `PluginRecord { id, name, description, cls: Option<Py<PyAny>>,
> error: Option<String> }`, isolating per-plugin import/resolution errors so one broken plugin
> doesn't abort the rest. `launch(record, snapshot)` = `cls(PluginContext(snapshot))` then
> `.launch()`. **Free functions, not a `PluginHost` struct** — the slice is stateless; the GUI (3e)
> owns the records (re-discovery/caching is 3g). One test drives good / lone-subclass / no-subclass /
> raises-at-import plugins + skip-cases + a launch round-trip, and is **hermetic** (restores
> `sys.path`/`sys.modules`). 19 workspace tests.
>
> **Test-infra fix (shared embedded interpreter) — important for all future pyhost tests.** The
> pyhost tests share one process-global CPython and the documented command runs them **in parallel**.
> Added `interp_test_lock` (a `#[cfg(test)]` `Mutex` in `lib.rs`) so every interpreter-touching test
> holds it across `Python::attach` and they run serially — this closes a `register_sdk` TOCTOU where
> two harness threads could each build a separate `ecm_host`, so a plugin's `issubclass(TrackerPlugin)`
> compared against a different base and discovery silently dropped the class (the one-off
> `good.cls`-missing failure). Serializing then **exposed a latent ordering bug** in the slice-3b
> array test: `tracked_arrays_from_python` needs numpy importable, but the embedded interpreter only
> gets numpy on `sys.path` via `ECM_PY_SITE`, which **only `imports_numpy_and_runs` had added** —
> under parallel it won that race, under serialization it sometimes ran second and rust-numpy's lazy
> array-API init panicked (`Failed to access NumPy array API capsule: ModuleNotFoundError: numpy`).
> Fixed with `ensure_embedded_site(py)` (idempotent `ECM_PY_SITE` insert) so each numpy-using test
> stands alone. **Rule for new pyhost tests: take `interp_test_lock()` and, if you touch numpy arrays,
> call `ensure_embedded_site(py)` first.** Verified 0 failures over 18 parallel + 12 serial repeats.

> ✅ **Slice 3e — Plugins menu + GUI wiring (done, `85915fe`).** The gui crate now drives the
> embedded-Python plugin host (first GUI↔pyhost wiring; added `ecm-pyhost` + `pyo3 0.27` as gui deps).
> New `gui/src/plugins.rs`: `snapshot(&ProjectState) -> ContextSnapshot` (the field-for-field map),
> `plugins_dir()` (resolves `ECM_PLUGINS_DIR` → else a `plugins/` folder beside the exe → else
> `./plugins`, which under `cargoenv` = `rust/plugins/`), and `discover()`/`launch()` that wrap
> `pyhost::{discover, launch}` under `Python::attach` (calling `ensure_embedded_site` first so plugins
> can use the bundled scientific stack). `main.rs` gains a **Plugins** menu (in the top toolbar after
> ⚙ Params) that discovers **lazily on first open** (cached in `EcmApp.plugins`), lists each plugin
> (loaded = clickable; failed = greyed `add_enabled(false, …)` with the load error as an
> `on_hover_text` tooltip), and launches by index via `launch_plugin(i)` — a `str` returned from
> `launch()` is shown in the status bar. **`pyhost::ensure_embedded_site` promoted test-only→`pub`**
> (shared by GUI + tests). Ships `rust/plugins/session_summary/` — a numpy-free `TrackerPlugin`
> (reads ctx scalars/ROI, returns a one-line summary; `PLUGIN = SessionSummary`) as the SDK
> starting point. **Verified end-to-end via an extended `ECM_SMOKE`** (discovers + launches it,
> printing `Session Summary: 12 frames, 320×240 px, 267 points (267 active) over 12 frames.`). Build
> clean, 19 workspace tests green. **Deviations:** free functions (no `PluginHost` struct); launch
> runs on the UI thread (fine for quick plugins; long/own-window plugins land with 3f/3g). Running
> the GUI outside `cargoenv` needs the Python env (`ECM_PY_SITE`/`PYTHONHOME`) until Phase-5 packaging.

> ✅ **Slice 3f — canvas overlays (done, `818e837`).** Plugins draw on the canvas via host-rendered
> draw-commands, replacing the Qt `OverlayFn`. New `pyhost::overlay`: a `DrawCommand` enum
> (`Circle`/`Polyline`/`Polygon`/`Text`, geometry in **image** coords) + a `#[pyclass] OverlayPainter`
> builder (`circle`/`line`/`polyline`/`polygon`/`text`; colors are `(r,g,b[,a])` sequences; widths/
> radii/text sizes are **screen** px, constant under zoom), exposed in the `ecm_host` SDK. A plugin
> optionally defines `overlay(self, painter)`; new host fns: `instantiate()` (retains the instance),
> `has_overlay()`, `overlay_commands()` (sets a fresh `ctx` on the instance so the overlay reflects
> current state, calls `overlay()`, returns the accumulated commands). `launch()` now reuses
> `instantiate()`. GUI: `canvas::draw_overlay_commands` renders the commands (image→screen positions,
> screen-px sizes) on **top** of the built-in markers; `main.rs` retains overlay-providing instances
> and re-invokes them **only when the state signature `(current_index, has_result, n_active)`
> changes** (cached otherwise, so the per-frame render is pure Rust — no per-frame GIL), plus a
> **Clear overlays** menu item; `plugins::launch` now returns a `LaunchOutcome { message, overlay }`
> (+ `refresh_overlays`/`unload`). Ships `rust/plugins/point_overlay/` — draws the ROI outline +
> active tracked points at the current frame (reads `ctx.coords`, exercising the NumPy array API).
> **Verified:** a pyhost overlay round-trip unit test (**20 workspace tests**, 0/8 parallel-flake
> repeats) + extended `ECM_SMOKE` (`Point Overlay` → 267 draw-commands). **egui note:** *circle*
> strokes take `Stroke`, *line/polygon* strokes take `PathStroke` — use `egui::Stroke::new(w, color)`
> uniformly (auto-converts). **Deviations:** overlay refresh is a state-signature poll, not yet
> event-driven (the reactive event hub is 3g — until then a mid-session change refreshes on the next
> frame scrub); polygon fill assumes convex (egui), stroke is always exact.

> ✅ **Slice 3g-a — reactive event hub (done).** Replaced the 3f overlay state-signature poll with a
> real event queue dispatched to plugins' `on_*` hooks (the port of Qt `PluginSignals`). Five events:
> `sequence_changed`, `frame_changed(global_index)`, `result_changed`, `mask_changed`, `roi_changed`.
> `pyhost::host::dispatch_event(instance, event, frame_index, snapshot)` refreshes the instance's
> `ctx` (so a hook reads current state via `self.ctx`) then calls `on_<event>`; the five hooks are
> optional no-ops on the `TrackerPlugin` base (`sdk.rs`). GUI side: a `plugins::PluginEvent` enum +
> `dispatch_events(instances, events, state)` (one `Python::attach`, errors swallowed per hook), and
> `EcmApp` now holds an `events: Vec<PluginEvent>` queue (**replacing the `overlay_sig` field**)
> drained once per frame by `process_plugin_events()` — which dispatches the queued events **then**
> re-invokes `refresh_overlays`. Emit sites in `main.rs`: `SequenceChanged` (open_dir success),
> `ResultChanged` (track done; and guarded in `invalidate_tracking` so clear/detect/ROI-discard fire
> it once), `MaskChanged` (`apply_keep_mask` + Cleanup undo), `FrameChanged` (slider commit),
> `RoiChanged` (rect/circle drag commit, n-gon close, Clear-ROI button). **Design choice (deviation
> from the Python `ctx.signals.X.connect`):** the Rust `PluginContext` is an immutable snapshot the
> host swaps per call, so a connect-on-ctx model would lose callbacks each refresh — the event
> surface is **method-override hooks** on the retained instance instead, matching the existing
> `overlay()` pattern. **Bonus fix:** the old `(current_index, has_result, n_active)` signature never
> refreshed on **ROI** changes, so `point_overlay` (which draws the ROI outline) silently missed
> edits; `roi_changed` closes that gap, and mid-session changes now refresh immediately, not on the
> next scrub. **Verified:** new pyhost unit test `dispatch_event_delivers_to_plugin_hooks`
> (**21 workspace tests**) + `ECM_SMOKE` still green (`overlay_plugins=1 draw_commands=267`).
> **Note:** an unclosed mid-N-Gon polygon and the tool-switch ROI-abandon do **not** emit
> `roi_changed` (no committed ROI existed); intentional.

> ✅ **Slice 3g-b — `apply_keep_mask` write-back (done).** The one plugin mutation: a plugin shrinks
> the active point set undoably. Since `PluginContext` is an immutable snapshot, `ctx.apply_keep_mask`
> is **collect-then-apply** (same shape as 3f overlay commands): the pymethod only *records* a mask
> into a new `pending_keep: Option<Vec<bool>>` field; the host takes it after the call and applies it
> through the **existing undoable path** (3g-a's `EcmApp::apply_keep_mask` → snapshot undo → `active_mask
> &= keep` → push `MaskChanged`). Pieces: `context.rs` — `apply_keep_mask(&mut self, py, keep)` coerces
> the arg with `np.asarray(keep).astype(bool).ravel()`, accepts length **P or n_active** (expands the
> n_active form at the kept columns, mirroring `api.py`), raises `ValueError` otherwise, no-op without
> a result; `pub(crate) take_pending_keep()`. `host.rs` — `take_keep_mask(py, instance)` casts the
> instance's `ctx` back to `Bound<PluginContext>` (`cast::<PluginContext>()`) and takes the field.
> `plugins.rs` — `LaunchOutcome` gains `keep_mask`, collected right after `launch()` (before the
> instance is moved into the overlay slot). `main.rs::launch_plugin` applies it via the shared
> undoable path. **Verified:** new unit test `apply_keep_mask_records_and_validates` (n_active-expand /
> P-passthrough / wrong-length ValueError / no-result no-op → **22 workspace tests**) + extended
> `ECM_SMOKE` (`[smoke] apply_keep_mask: n_active 267 -> 134`). Ships `rust/plugins/decimate_points/`
> — drops every other active point via a plain-list `keep` (numpy-free), the apply_keep_mask example.
> **Design choice:** apply happens **after `launch()` returns** (one undo entry), not live mid-call —
> keeps the ctx a pure snapshot. **Deferred:** apply on *event hooks* too (not just launch) lands when
> a plugin needs it (probably with 3g-c panels — a button that filters). **Repo note:** set
> `core.fileMode false` here — Dropbox/Windows flips 644↔755 and produced spurious mode-only diffs.

1. **Slice 3g-c..d — panels + settings (next).** Port the rest of the reactive layer (3g-a event
   hub + 3g-b `apply_keep_mask` write-back are done — see above). **First, delegate a contract-read**
   (per the Working method) of `app/plugins/api.py` (panel/settings surface) to a subagent. The two
   remaining parts:
   - **(c) Declared egui control panels** — plugins declare controls (a serializable widget spec the
     host renders as egui — e.g. sliders/checkboxes/buttons), replacing the embedded-Qt panels; wire
     control-value changes back to the plugin (a callback or a re-read each change). Keep the first
     cut small (a few widget kinds) — the example plugins (Phase 4) will show what's actually needed.
   - **(d) Plugin settings persistence** — per-plugin JSON via `core::settings` (same dir/merge model
     as the param dialogs); expose load/save on `PluginContext`.
   Verify with a pyhost unit test (event delivery / apply_keep_mask command) + extend `ECM_SMOKE`
   (e.g. a plugin that calls `apply_keep_mask` and asserts `n_active` dropped). Update the
   `point_overlay`/`session_summary` examples or add one exercising (b)/(c)/(d).
2. **Phase 4** — port the 3 example plugins to the new API. **Phase 5** — packaging (`cargo-packager`).

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
- **Plugin host / pyo3 0.27 (verified from the local registry source before coding):** `pyhost`
  keeps the `auto-initialize` feature. Expose the SDK by **injecting a module into `sys.modules`**
  (`PyModule::new(py, name)` → `m.add_class::<PluginContext>()` → `py.run(SRC, Some(&m.dict()),
  None)` to define Python classes into it → `sys.modules.set_item(name, &m)`), NOT
  `append_to_inittab!` (that panics if the interpreter is already initialized and needs
  `auto-initialize` **off** + a manual `Python::initialize()` before any `Python::attach`). API
  shapes: `Python::run(self, code: &CStr, globals: Option<&Bound<PyDict>>, locals:
  Option<&Bound<PyDict>>)`; raw C-string literals `cr#"...python..."#` are `&'static CStr` (avoid
  the sequence `"#` inside); `PyModuleMethods`/`PyDictMethods`/`PyAnyMethods` come from
  `pyo3::prelude::*`; `sys.modules` supports `.contains`/`.get_item`/`.set_item` via the mapping
  protocol (no downcast to `PyDict` needed). Stored class objects are `Py<PyAny>`; call them with
  `cls.call1(py, (ctx,))` then `instance.call_method0(py, "launch")`. **rust-numpy 0.27** pairs with
  pyo3 0.27 + ndarray 0.16 (numpy 0.28 needs rustc 1.83 > the 1.80 pin); `to_pyarray(&self, py)`
  copies, `into_pyarray(self, py)` moves, both → `Bound<'py, PyArrayN>`; read back via
  `obj.extract::<PyReadonlyArrayN<T>>()` then `.as_array()`. Discovery (3d): import a top-level
  package with `py.import(name)` (`&str`); `is_subclass` lives on **`PyTypeMethods` only** → first
  `value.cast::<PyType>()` (the 0.27 replacement for the now-deprecated `downcast`; doubles as the
  "is it a class?" guard) then `cls.is_subclass(&base)? && !value.is(&base)`; iterate a module's
  members via `module.dict().iter()` (returns `Bound<PyDict>`, no `?`); attribute/method **names are
  `&str`** (NOT `&CStr`). `getattr_opt(name) -> Option<Bound>` for optional attrs.
- **Shared embedded interpreter in tests (bit us in 3d — read before adding a pyhost test):** all
  `#[cfg(test)]` tests run against **one** process-global CPython, and the documented `test` command
  runs them **in parallel**. Two hazards: (1) concurrent `register_sdk` can build duplicate `ecm_host`
  modules → `issubclass` identity mismatch; (2) numpy is only importable after `ECM_PY_SITE` is on
  `sys.path`, and CPython releases the GIL mid-bytecode so "another test imported it first" is racy.
  **Convention:** every interpreter-touching test takes `let _g = crate::interp_test_lock();` across
  its `Python::attach` (serializes them), and any test that creates numpy arrays calls
  `crate::ensure_embedded_site(py)` first (idempotent `ECM_PY_SITE` insert). Tests that mutate
  `sys.path`/`sys.modules` (e.g. discovery) should restore them so they stay hermetic. Symptom if you
  forget: intermittent `Failed to access NumPy array API capsule: ModuleNotFoundError: numpy`,
  surfacing only under parallel/serialized ordering — debug by looping the suite, not single runs.
- **Overlay draw-commands (3f, verified before coding):** egui *circle* methods (`circle_filled`/
  `circle_stroke`) take `impl Into<Stroke>`; *line/path/polygon* (`Shape::line`/`convex_polygon`/
  `closed_line`, `Painter::line_segment`) take `impl Into<PathStroke>` — build `egui::Stroke::new(w,
  color)` **everywhere** and rely on `From<Stroke> for PathStroke` (don't name `PathStroke`). All
  `Painter` draw methods return `ShapeIdx` (ignore). pyo3: to read a Rust-built `#[pyclass]`'s field
  back after a Python call, create it with **`Bound::new(py, T)`** and use `painter.borrow().field`
  — **not** `Py::new` + `Py::borrow(py)` (the latter ties the `PyRef` to the local's lifetime vs the
  GIL's → E0597). Pass it into the call as `(&bound,)`. `&mut self` pymethods are auto-`PyRefMut`
  (don't hold another borrow across the call). `#[pyo3(signature = (a, b=4.0, c=None))]` with
  `Option<Vec<u8>>` for `c=None`.
- **Build invocation:** see the Commands note above (absolute manifest path; benign vswhere
  warning; background-task-to-log if interactive output capture misbehaves).

## Verification approach

Parity-first: numerical modules are validated **bit-identically** against the live Python
implementation via committed fixtures. GUI slices are verified by a headless `ECM_SMOKE` run
(asserts frames load, texture uploads, detection runs) plus manual `cargo run`.
