# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: a `CLAUDE.md` is inherited from `C:\Users\raulh\Dropbox\CLAUDE.md`. That file describes an unrelated "LLM Council" project and does **not** apply here — this is a PyQt5 desktop feature-tracking app. Ignore the inherited file's architecture notes (e.g. "use relative imports", "run as `python -m backend.main`").

## What this is

A desktop GUI tool for tracking image features through a sequence of frames. Workflow: load an image sequence → define a 4-corner ROI on a reference frame → seed feature points (Shi-Tomasi corners or a regular grid) → track them forward and backward with pyramidal Lucas-Kanade optical flow → interactively filter out low-quality tracks → export the surviving coordinates as a `.npy` array.

## Commands

The project uses **uv** with a dedicated, self-contained `.venv` and a uv-managed Python
interpreter (pinned to 3.12 in `.python-version` — `numpy<2` only ships Windows wheels up to
3.12), so it is independent of any system Python. `pyproject.toml` (with `uv.lock`) is the
single source of truth for deps.

```powershell
uv sync                                   # create .venv (managed Python 3.12) + install deps

uv run python -m app.main                 # launch the GUI (run from project root)

# Headless regression tests — runs every test_* function as a script:
$env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_pipeline
$env:QT_QPA_PLATFORM = "offscreen"; uv run python -m tests.test_plugins
```

- **Always run from the project root.** Imports are absolute and rooted at the `app` package (`from app.core...`, `from app.gui...`); there is no installed package (`tool.uv.package = false`), so the root must be on `sys.path`.
- Tests live in `tests/test_pipeline.py` (core + GUI pipeline) and `tests/test_plugins.py` (plugin SDK/manager/context/canvas hooks) and run as plain scripts (the `__main__` block calls each `test_*` in turn). They are also pytest-compatible. `tests/synthetic.py` generates a synthetic sequence with known per-frame translation, giving ground-truth motion to validate tracking against.
- `QT_QPA_PLATFORM=offscreen` is needed for the GUI test (`test_gui_pipeline`) and for all of `test_plugins` (it builds a real `MainWindow`); the core pipeline tests are Qt-free. Note: a headless `QApplication` must be kept referenced (an unreferenced one is GC'd, after which constructing any `QWidget` aborts) — see `_app()` in `test_plugins.py`.

### Dropbox / multi-machine note (important)

This project folder is synced via Dropbox across machines (a Windows PC and a Mac). **`.venv` must never be synced** — a virtualenv is not portable across OSes/machines (its `pyvenv.cfg` hard-codes an absolute interpreter path, and it has `bin/` on Unix vs `Scripts/`+`Lib/` on Windows). Syncing it produces a corrupt hybrid venv and, on macOS, a fatal `ModuleNotFoundError: No module named 'encodings'` at startup. Always recreate `.venv` per-machine with `uv sync`, and mark it Dropbox-ignored on **every** machine:

```bash
# macOS / Linux (after `uv sync`):
xattr -w com.dropbox.ignored 1 .venv      # macOS
attr  -s com.dropbox.ignored -V 1 .venv   # Linux
```
```powershell
# Windows: right-click .venv → Dropbox → Ignore, or:
Set-Content -Path '.venv:com.dropbox.ignored' -Value 1
```

If you ever hit the `encodings` crash, the venv has been re-synced: `rm -rf .venv && uv sync`, then re-apply the ignore marker. `pyproject.toml`, `uv.lock`, and `.python-version` are platform-independent and are fine to sync.

## Architecture

Three layers under `app/`, with a strict dependency direction `gui → models → core` and a hard rule: **`app/core/` and `app/core/settings.py` are Qt-free** so the whole pipeline can be exercised headlessly. Do not import PyQt5 into `core`.

- **`app/core/`** — pure logic: `image_sequence` (path discovery + on-demand decode with an LRU cache, normalizes everything to BGR uint8), `roi` (4-corner polygon, masks, point-in-polygon), `feature_detection` (Shi-Tomasi corners, regular grid), `tracking` (the LK pipeline), `cleanup` (quality metrics + filtering), `export`, `settings` (JSON persistence).
- **`app/models/`** — `tracker_result.TrackerResult` (immutable tracking output) and `project_state.ProjectState` (mutable per-session state: loaded sequence, indices, ROI, params, result, active mask, undo stack).
- **`app/gui/`** — PyQt5 widgets: `main_window` (orchestrates everything, owns `ProjectState`), `canvas_view` (frame display + overlays), `dialogs` (parameter editors), `cleanup_dialog` (filter UI). Presentation-only helpers: `theme` (app-wide light stylesheet), `icon_loader` (tinted SVG → `QIcon`), and the bundled `icons/` SVGs.

### The dual index system (most important concept)

Every per-frame array and frame reference is in one of two coordinate systems — confusing them is the easiest way to introduce bugs:

- **Global index**: position in the full loaded folder (`0 .. total_images-1`).
- **Cut index**: position within the active `reference_index .. last_index` range, where **cut 0 = the reference frame**.

`ProjectState.global_to_cut` / `cut_to_global` convert between them. All `TrackerResult` arrays (`coords_fw`, `status_fw`, etc.) are indexed by **cut** index, so `coords_fw[t]` and `coords_bw[t]` describe the same frame and can be compared directly. The invariant `0 <= reference <= last < total` is enforced both by `ProjectState`'s setters and at the slider level in `MainWindow._sync_range_constraints`.

### Tracking (`core/tracking.py`)

`track()` runs LK **forward** (reference→last), then **backward** seeded from the forward pass's last-frame positions (last→reference). The backward arrays are reversed back into cut order so they align with the forward arrays. The per-point **forward-backward (FB) error** (distance between where a point started and where the round-trip returns it) is the primary quality signal. Points that never track validly get `+inf` error so filters can drop them. A `progress_cb(done, total) -> bool` is polled for cancellation; returning `True` aborts and `track()` returns `None`.

### Result immutability + active mask + undo (cleanup flow)

`TrackerResult` is **never mutated** after creation. Filtering instead operates on a separate boolean `active_mask` (shape `(P,)`) held by `ProjectState`, with snapshots pushed onto `ProjectState.undo_stack`. This is what makes cleanup non-destructive and undoable.

`core/cleanup.py`: `compute_metrics()` derives per-point metrics once from a `TrackerResult`. `Thresholds` is a set of `BandFilter`s (one per metric) plus two boolean filters. `build_mask()` produces the keep mask with these semantics: **a disabled band is ignored entirely** (so `+inf` points survive), and an enabled band keeps a point iff `lo <= metric <= hi`. `MainWindow` connects the dialog's signals to preview (`set_preview_mask` → green = kept, red = will drop) and applies the mask by `active_mask &= preview_keep`.

### Canvas coordinate model (`gui/canvas_view.py`)

All overlay geometry is stored in **image coordinates**. A single image→screen `QTransform` (fit-to-widget scale, then user zoom/pan) is rebuilt on every `paintEvent` so it tracks resizing. The frame image is drawn under that transform; overlays (ROI, feature points, tracks) are drawn in **screen space** after resetting the transform, so marker sizes and line widths stay constant regardless of zoom.

### Theming & toolbar icons (`gui/theme.py`, `gui/icon_loader.py`, `gui/icons/`)

The app ships a light visual theme. `theme.apply_theme(app)` is called once in `app/main.py` right after the `QApplication` is created — it sets the `"Fusion"` base style and a single `LIGHT_QSS` stylesheet on the application, so styling reaches the main window **and** every dialog. Palette/accent (`#2563eb`) live at the top of `LIGHT_QSS`; the icon glyph colors in `icon_loader` (`NORMAL`/`ACCENT`/`DISABLED`) are kept visually in sync with it. **Sliders are intentionally left unstyled** (native/Fusion look) — don't re-add `QSlider` QSS.

Toolbar icons come from MIT-licensed Lucide SVGs in `gui/icons/` (plain XML, safe to Dropbox-sync, one file per action). `icon_loader.load_icon(name, color, size)` renders an SVG via `QSvgRenderer` and recolors it with a `SourceIn` composite, returning a `QIcon` that carries an auto-faded Disabled variant; results are memoized and rendered at the device pixel ratio for crisp HiDPI. `QtSvg` ships with the PyQt5 wheel, so this adds no dependency.

The toolbar (`MainWindow._build_toolbar`) is grouped into captioned clusters (ROI · DETECT · TRACK · VIEW) built by `_toolbar_group(title, actions, primary=...)`. Each cluster hosts `QToolButton`s whose `setDefaultAction` proxies the **existing** `QAction`s — so all enable/disable/checked logic in `_update_tool_states` is unchanged; the buttons just follow their actions. The `primary` action (Run Tracking) gets `objectName("primaryAction")` for the accent QSS rule.

### Plugin system (`app/plugins/`, root `plugins/`)

The app is extensible via plugins for **post-processing / export / visualization**. There are
two locations, deliberately separate:

- **`app/plugins/` — the SDK** (GUI layer; may import PyQt5). `api.py` is the canonical, fully
  docstring'd reference a plugin author reads; `manager.py` discovers/loads plugins and owns the
  menu + window lifecycle. The dependency direction stays `gui → plugins → models → core`; the
  Qt-free rule for `app/core/` is untouched.
- **`plugins/` (repo root) — installed plugins**, one package per folder, imported as
  `plugins.<name>` (root is already on `sys.path`). Plain Python, safe to Dropbox-sync.
  `plugins/README.md` is the author's guide; the three bundled examples (`custom_exporter`,
  `displacement_overlay`, `affine_zones`) are the copy-paste scaffolds and cover all three
  capabilities (data access, canvas overlay, mouse capture).

**The façade (`PluginContext`)** is the whole point: a plugin only ever learns this one object
(handed to it as `self.ctx`). It wraps `MainWindow`/`ProjectState`/`CanvasView` and **hides the
dual global/cut index system** — all `ctx` indices are global, while `ctx.coords()` is
cut-indexed (`coords[0]` = reference), with `ctx.global_to_cut`/`cut_to_global` to convert.
Plugins get read access to coords/images/ROI/mask/metrics, plus overlays, mouse capture, settings,
and exactly one mutation: `ctx.apply_keep_mask()`.

Key integration points in the core (all small + additive):
- **Canvas hooks** (`canvas_view.py`): `add_overlay/remove_overlay` (painters called in screen
  space each `paintEvent`, each wrapped in try/except so a buggy overlay self-removes) and
  `set_interaction/clear_interaction` (a `CanvasInteraction` that receives image-space mouse
  events, taking priority over the ROI click; middle/right-drag still pans). The canvas does
  **not** import the SDK — overlays are plain callables and the interaction handler is duck-typed.
- **Signal hub** (`PluginSignals`, owned by `MainWindow` as `self.signals`): `sequence_changed`,
  `frame_changed(int)`, `result_changed`, `mask_changed`, `roi_changed`, emitted at the existing
  state transitions so plugins refresh reactively instead of polling.
- **`MainWindow.apply_keep_mask(keep)`**: the single undoable mask-mutation path, shared by the
  Cleanup dialog (`_cleanup_apply`) and `PluginContext.apply_keep_mask` — snapshots onto the undo
  stack, ANDs in `keep` (points only ever leave the active set), refreshes, emits `mask_changed`.

`scipy` is a dependency because the bundled example plugins use it (`Delaunay`, `savemat`); the
core pipeline itself does not.

Starting point for a plugin author: read `app/plugins/api.py` + `plugins/README.md`, then copy
the closest example.

### Settings persistence (`core/settings.py`)

Parameter-dialog defaults persist to a JSON file. The config directory resolves in order: `TRACKER_CONFIG_DIR` (tests point this at a temp dir) → `XDG_CONFIG_HOME/feature_tracker` → `~/.config/feature_tracker`. On load, `ProjectState` merges built-in defaults (`DEFAULT_SHI_TOMASI`, `DEFAULT_GRID`, `DEFAULT_LK`) with any persisted section, so partial/missing keys are tolerated.

## Conventions

- LK parameters are stored as a flat dict (`DEFAULT_LK`) and assembled into OpenCV's nested `winSize`/`criteria` kwargs by `tracking._cv_lk_kwargs`. Edit the flat dict; don't pass OpenCV-shaped kwargs around.
- Sliders/spin boxes use `blockSignals` extensively to avoid reentrant update loops between interdependent controls (`current`/`reference`/`last`). When adding linked widgets, follow the same pattern.
- Changing the reference frame or ROI invalidates downstream state — `MainWindow` clears features/ROI and prompts before discarding an existing tracking result (`_confirm_discard_tracking`).
