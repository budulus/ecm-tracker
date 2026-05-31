# ECM Tracker — Rust Rewrite: Continuation PLAN

**Condensed, action-oriented spec for finishing the port. Point `/loop` at this file after a context
reset and execute it slice-by-slice until done.** Full history/rationale: `./PROGRESS.md`. The
working Python reference is always at `../app/` and `../plugins/`. Strategy: `../.claude/plans/please-review-this-codebase-scalable-starlight.md`.

## State (2026-05-31)

- **Phases 0–3 COMPLETE**, pushed to `origin/main @ 43e2945`. Full Qt-free `core`+`models` (bit-identical
  parity), the egui GUI (canvas, ROI tools, detection, tracking, cleanup, dialogs, export, theme/icons),
  and the embedded-Python **plugin host** (read-bridge, numpy arrays, `ecm_host` SDK, discovery/loading,
  Plugins menu, canvas overlays, reactive `on_*` event hub, `apply_keep_mask`, declared egui control
  panels, per-plugin settings). **24 workspace tests green.**
- **REMAINING: Phase 4** (port the 3 example plugins) **→ Phase 5** (packaging). That's the whole job.

## The execution loop (do this for every slice, no approval gates between slices)

1. **Delegate heavy reads to FRESH subagents, keep only conclusions** (this keeps the main context lean —
   it is the core working method). Per slice: spawn `Explore`/`general-purpose` subagents to (a) read the
   Python original (`../plugins/<name>/`) for exact algorithms/formats, and (b) verify exact crate/pyo3
   APIs before coding. Launch independent subagents in parallel (one message). Never read >~100 lines you
   won't edit into the main context — delegate it.
2. **Implement** the slice (edit Rust and/or the `rust/plugins/<name>/` Python).
3. **Build + test** via the cargoenv as a **background task teeing to a log**, then **Grep** the log for
   `test result` / `error\[` / `warning:` (don't Read whole logs). Run `ECM_SMOKE` for GUI/plugin slices.
4. **Commit + push** (use the `commit-push` skill; co-author footer required; Forgejo-over-Tailscale can
   401 — just retry `git push origin main`).
5. **Update status** (the table below + a one-line note in `PROGRESS.md`).
6. **Next slice.** Keep going phase→slice→phase until Phase 5 is done.

## Build / run (from `rust/`)

```powershell
$CE = "$env:LOCALAPPDATA\ecm-tracker\cargoenv.ps1"
pwsh -NoProfile $CE build --workspace --manifest-path <abs>\rust\Cargo.toml
pwsh -NoProfile $CE test  --workspace --manifest-path <abs>\rust\Cargo.toml
pwsh -NoProfile $CE run   -p ecm-tracker --manifest-path <abs>\rust\Cargo.toml   # GUI / ECM_SMOKE
```
Pass an **absolute** `--manifest-path` (cargoenv changes cwd). The `'vswhere.exe' is not recognized`
line is a **benign** warning. If interactive capture looks empty, run as a background task → log → Grep.
**ECM_SMOKE** (headless E2E): set `ECM_SMOKE=1`, `ECM_SMOKE_DIR=<rust>\crates\core\tests\fixtures\frames`,
`ECM_PLUGINS_DIR=<rust>\plugins`, run via cargoenv `run`. Build env details: `./PROGRESS.md` + `./README.md`.

## Embedded Python — provisioned this session (per-machine, like `.venv`)

The embedded interpreter (`%LOCALAPPDATA%\ecm-tracker\pyenv`, a uv venv over standalone CPython 3.12.13)
was **numpy-only**; it now also has **matplotlib 3.10.9, scipy 1.17.1, PyQt5 5.15.11** (numpy left at
2.4.6 — do NOT let it change; rust-numpy 0.27 depends on it). **tkinter 8.6 is bundled.** Reprovision on
any machine (Mac too) with:
```
uv pip install --python "<...>/ecm-tracker/pyenv/Scripts/python.exe" matplotlib scipy PyQt5
```
Record this in `README.md` as part of the per-machine setup. **No cv2** in the embedded env (OpenCV is
only linked into the Rust layer) — reimplement the few `cv2` calls in numpy (see 4b/4c).

## Phase 4 architecture (DECIDED — don't re-litigate)

Two plugin styles, chosen per plugin:

- **Canvas-integrated** (in-process): draws on the tracker image via the existing egui **overlay** API
  (`overlay(self, painter)`) and uses the existing egui **declarative panel** (`panel`/`on_control`).
  Live, integrated, no extra process. → **displacement_overlay**.
- **Native window** (child process): `launch()` spawns a detached **PyQt5/matplotlib** process that *is*
  the whole plugin UI (its own event loop — egui/winit owns the main thread, and macOS forbids GUI off
  the main thread, so in-process Qt/Tk is out). The host passes a data **snapshot** (coords/status/
  point_indices/ref+last/frame_globals + image paths + `ctx.get_settings()`) via a temp `.npz`/`.json`.
  → **custom_exporter, affine_zones**.

**Child → host channel (pure Python, no host change):** the child appends JSON-line commands
(`{"type":"save_settings","data":{…}}`, `{"type":"apply_keep_mask","keep":[…]}`) to an **outbox** file;
the retained plugin instance **drains the outbox on each reactive `on_*` event** and dispatches to
`ctx.save_settings` / `ctx.apply_keep_mask`. (Eventual-consistent; fine — events fire constantly.)

**Spawn launcher:** under the embedded interpreter `sys.executable` may be the host exe, so resolve the
child python as `sys.prefix + (Scripts/pythonw.exe | bin/python)`, fallback `sys.executable`, then PATH.
Inherit `os.environ` (carries `ECM_PY_SITE` etc. so the child sees site-packages).

## Slices

### 4-prep — host: expose image paths  (Rust: `pyhost` + `gui`)
Native windows need the frame image to draw on. Add to `ContextSnapshot` a `image_paths: Vec<String>`
and `PluginContext` methods `frame_path(global_i) -> Option<String>` + `sequence_paths() -> Vec<String>`
(the lightweight, paths-only equivalent of the old Qt `api.py` `frame_rgb`). Fill `image_paths` in
`gui/src/plugins.rs::snapshot` from the `ImageSequence` (add a Qt-free paths getter to
`core::image_sequence` if one doesn't exist). Add a `pyhost` unit test (take `interp_test_lock()`).
Build + test + commit.

### 4a — custom_exporter  (native PyQt window)
`launch()` writes a temp `.npz` of `coords` (`active_only` per saved setting) + `point_ids`
(`point_indices()` or `arange(point_count)`) + `reference_index`/`last_index`/frame-global indices +
`ctx.get_settings()`, then spawns `rust/plugins/custom_exporter/window.py`. The window: format combo
(csv/mat/npz) + "active only" checkbox + count label + Export → `QFileDialog`. Writes (exact formats in
`../plugins/custom_exporter/exporter.py`): **CSV** header `[frame_global, frame_cut, point_id, x, y]`,
row per (frame,point); **.mat** via `scipy.io.savemat` (`coords` f64, `point_ids` int64 col,
`frame_global_indices` int64 col, `reference_index`, `last_index`); **.npz** (`coords` f32 + ids +
globals + ref/last) **plus `*_meta.json`**. Persist `format`/`active_only`/`last_dir` via outbox →
`ctx.save_settings`. Smoke + commit.

### 4b — displacement_overlay  (egui overlay + egui panel, in-process)
`overlay(self, painter)`: `simplices = scipy.spatial.Delaunay(ref_active_pts).simplices`; per triangle
`dX,dx`=edge vectors (cols), skip `|det dX|<1e-9`, `F = dx @ inv(dX)`, `ε = 0.5*(F+Fᵀ) − I`,
`exx,eyy,exy = ε[0,0],ε[1,1],ε[0,1]`, `vm = √(exx²−exx·eyy+eyy²+3·exy²)`; range = symmetric `±max|finite|`
(von Mises `vmin=0`); color via a **numpy JET LUT** (reimplement `cv2.COLORMAP_JET`), alpha = opacity;
draw filled triangles (`painter.polygon(pts, fill=…)`) + yellow displacement lines `(cur−ref)*scale`
(`painter.line`). Controls via `panel`: component **cycle button + label** (panel API has no dropdown —
deviation), show-field / show-vectors checkboxes, vector-scale slider (0.1–50), opacity slider (0–1),
range label. Refresh on `on_frame_changed`/`on_result_changed`/`on_mask_changed`. Edge cases: <3 pts,
no result, cut 0 (zero displacement). Exact ref: `../plugins/displacement_overlay/overlay.py`. Smoke + commit.

### 4c — affine_zones  (native PyQt + embedded matplotlib window)
`launch()` snapshots coords/status/point_indices + **image paths** + settings → spawns
`rust/plugins/affine_zones/window.py`. Window: load the reference frame image via a passed `frame_path`
(PIL/`matplotlib.image`), draw polygon **zones with Qt mouse events** (no host mouse-capture needed —
deviation: zones live in the plugin window, not the main canvas). Per zone/frame: gather pts-in-polygon
(of `coords[0]`), exclude `status[t]==0`; **affine fit** `lstsq([x y 1] → [x' y'])`, extract 2×2 `F`;
**principal stretches** `C=FᵀF; vals,vecs=eigh(C); λ=√max(vals,0)` sorted desc → `λ1,λ2,v1,v2`. Zone
table (`QTableWidget` + per-zone `QColorDialog`). **Plot Curves**: embedded `FigureCanvasQTAgg`, λ1 solid
/ λ2 dashed vs **global** frame, one color per zone. **RANSAC** dialog (reproj=3.0, maxIters=2000,
conf=0.99): reimplement `cv2.estimateAffine2D(RANSAC)` in numpy (sample 3, fit affine, count inliers
within reproj px, keep best) → build full-`n_active` keep mask (outliers→False) → outbox →
`ctx.apply_keep_mask`; red preview of outliers. **CSV** export header
`[zone, frame_global, n_points, lambda1, lambda2, v1x, v1y, v2x, v2y]`. Exact ref:
`../plugins/affine_zones/zones.py`. Smoke + commit.

### Phase 5 — packaging
`cargo-packager`: bundle the exe + OpenCV runtime DLLs + the embedded `pyenv` (with matplotlib/scipy/
PyQt5) + the `rust/plugins/` tree, so it runs without cargoenv (set `ECM_PY_SITE`/`PYTHONHOME` at
startup). Verify a clean-machine launch + `ECM_SMOKE`. Details in the strategy plan.

## Status

| Slice | State | Commit |
|---|---|---|
| 4-prep — host image paths (`frame_path`/`sequence_paths`) | ✅ done | `5b6bb9b` |
| 4a — custom_exporter (native PyQt window) | ✅ done | `cb5740c` |
| 4b — displacement_overlay (egui overlay + panel) | ✅ done | `377c67a` |
| 4c — affine_zones (PyQt + matplotlib window) | ✅ done | `57c2401` |
| 5 — packaging (cargo-packager) | ⬜ todo | |

## Done-definition

All five rows ✅, `ECM_SMOKE` exercises all three ported plugins, 24+ workspace tests green, everything
committed + pushed to `origin/main`, and the app launches packaged (Phase 5) on a clean machine.
