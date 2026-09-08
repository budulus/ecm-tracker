# ECM Tracker plugin contract — API 1

This document, the example in [_sdk/example](_sdk/example/__init__.py), and the
[API reference](plugin-api.html) are sufficient for a small plugin. Do not read or import
host GUI/model/core internals. This is a trusted Python extension API, not a sandbox.

## Start here

1. Copy the example folder into **Plugins → Open Plugins Folder**, rename it using
   letters/digits/underscores, and edit its `__init__.py`.
2. Import `TrackerPlugin` from `app.plugins`; expose `PLUGIN = YourClass`.
3. Declare `API_VERSION = 1`, a nonempty string `NAME`, string `DESCRIPTION`,
   and integer `ORDER` (default 100). A malformed plugin is disabled independently.
4. Implement `launch()`: create/show and return a PySide6 QWidget, or return None
   for an action without a window. Handle an empty session. Keep launch/paint callbacks fast.
5. Read `ctx.tracks()`; use its validity mask in EVERY displacement/strain calculation.
6. Use `ctx.subscribe(signal, callback)` for lifecycle-managed callbacks. Return/close
   child windows and release your own files, timers and workers in `on_unload()`.
7. Test the pure calculation, then run the host smoke test and try reload/shutdown.

User plugins are stored outside the app's installation/signed bundle. User folders override
bundled folders of the same name after Reload Plugins. Do not edit bundled examples in place.
Legacy source-checkout `plugins/my_plugin` folders still load.

## Data: one coherent snapshot

`ctx.tracks(active_only=True) -> TrackingSnapshot | None`. None means no result.
The snapshot is detached, read-only by contract, and remains a description of the OLD
state after an event. Request a new snapshot after scientific state changes.

| Field | Meaning |
|---|---|
| revision | Monotonic host scientific-state version; use for stale-result checks |
| frame_indices (N,) | Global image indices; first is reference_index |
| point_ids (A,) | Original zero-based point columns, stable within this result |
| coords (N,A,2) | Image pixels, (x,y), x right / y down; origin upper left |
| valid (N,A) | True = measured/trusted; False = lost track with frozen coordinates |
| quality (N,A) | Forward LK quality; ignore the seed row and invalid entries |
| tracking_params | Read-only producing LK parameters (legacy results may have none) |
| error_kind | photometric (lower better), min_eigenvalue (higher better), or unknown |
| reference_index / last_index | Inclusive GLOBAL bounds |

N is tracked frames, A is active points, P is all original points. Arrays are mutually
aligned. An all-filtered result has A=0, not None. Inactive points are omitted, but failed
tracks are retained with valid=False. Point IDs are NOT stable across a new tracking run.
Never infer validity from finite coordinates or a zero displacement.

`ctx.frame_tracks(global_index)` returns `(point_ids, xy, valid)` for one frame, or
None outside the tracking result. Snapshots are cached; repeated paint calls do not copy
the whole trajectory. Frame indices must be integers. Image helpers accept global indices:
`frame_bgr` is read-only uint8 H×W×3 BGR; `frame_rgb` and `frame_gray` are owned copies.
`image_size()` returns (height,width), or None. No sequence raises RuntimeError; an
out-of-range image raises IndexError.

Legacy adapters remain supported: `coords()`, `track_status()`, `point_indices()`,
`active_mask`, `frame_count`, `point_count`, `n_active`, `current_cut`,
`global_to_cut`, `cut_to_global`. Conversions are arithmetic, NOT bounds validation.
`metrics()` contains ALL P points (not A), with read-only arrays; use point_ids to select.
`roi` is a detached copy; changing it does not change the host.
`roi_corners`, `roi_contains`, and `roi_mask` are the usual ROI read interface.

## Mutation and events

All GUI/context calls occur on the Qt main thread. A worker may compute from a detached
snapshot; deliver its answer back to the main thread and reject a changed revision.

`ctx.apply_keep_mask(keep, revision=snapshot.revision)` accepts a one-dimensional mask
of length A or P. It can only remove currently active points, is undoable, and raises
ValueError for wrong shape/length or a stale revision. There is no general state setter.

Workflow methods are also public, but deliberately destructive to dependent state:

| Operation | Effect / signals after coherent state commit |
|---|---|
| load_sequence(paths, source_dir) | Validates first; success replaces session and emits sequence_changed; false retains old session |
| set_current_frame(global) | Clamped display navigation; frame_changed(int); no revision change |
| set_frame_range(ref,last) | Atomic inclusive bounds; changed bounds discard tracks; range_changed, then result_changed if needed |
| set_reference_frame / set_last_frame | Legacy one-bound adapters; same invalidation rules |
| Reference changes | Also clear ROI/seeds and emit roi_changed, seeds_changed |
| Tracking result installed/cleared | result_changed |
| apply_keep_mask / user cleanup undo | mask_changed |
| User seed edits / ROI edits | seeds_changed / roi_changed |
| load_trackers(path) | Validates before install; range_changed, seeds_changed, roi_changed, result_changed, mask_changed, frame_changed |

Events are invalidation notifications, not a transaction log. Re-read context inside the
callback; tolerate repeated events. A sequence_changed callback must forget sequence-owned
zones, imported time mappings and calculation caches. range_changed must invalidate anything
tied to a reference or cut range. Do not mutate the session recursively inside those callbacks.
Tracking initiated before a scientific-state change cannot install its stale result.

`save_trackers(path)` writes a complete atomic NPZ. `load_trackers(path)` requires a loaded
matching image sequence. Current files carry ordered image-content fingerprints and producing
LK metadata. Legacy v1 files have count-only identity; legacy quality is explicitly unknown.
Failures raise ValueError/OSError and do not partially install the candidate.

## Drawing and ownership

`add_overlay(fn)` calls `fn(painter, ctx)` in widget/screen pixels; map image positions with
`image_to_screen(x,y)`. Do not hold a painter after the callback. A failing painter is removed.
`remove_overlay(fn)` is idempotent. `request_redraw()` requests a repaint.

Subclass `CanvasInteraction` for on_press/on_move/on_release/on_cancel, receiving image-space
QPointF values. `begin_canvas_interaction(handler)` replaces the current capture and cancels
conflicting tools. `end_canvas_interaction()` releases only YOUR capture. Cancellation must
discard unfinished geometry. Sequence/range replacement cancels unfinished interactions.

The manager owns the plugin context and returned window. Visible relaunch focuses it;
closed/deleted-window relaunch releases the old instance and creates a fresh one.
Reload/shutdown calls on_unload, disconnects managed subscriptions, removes overlays,
releases owned capture, and schedules window deletion—even if plugin cleanup raises.
Direct `ctx.signals.x.connect(...)` remains a legacy API but requires manual disconnection.
on_unload must stop/join your own workers and timers; host disposal is not thread cancellation.

`ctx.window` is supported ONLY as a Qt parent. `ctx.canvas` and `ctx.result` are legacy
advanced escape hatches, not a stable implementation API. Names starting with _ are private.
New plugins must not use them. These adapters remain in API 1; removing them requires a major
API revision. Do not use them to change scientific state.

## Math helpers and conventions

Import helpers from `app.plugins.analysis`, never `app.core`.
`fit_affine(ref,cur,valid)` returns (eligible_indices,F,b) for cur=ref@F.T+b, or None
for insufficient/poor geometry. At least 3 finite non-collinear correspondences are required;
centered SVD condition >= 10,000 is rejected. This is a numerical guard, not an uncertainty bound.
Coordinates assume square pixels (or calibration applied before fitting), a fixed camera,
and homogeneous in-plane motion over the selected points. Perspective/out-of-plane motion
requires a calibrated model; this SDK does not silently correct it. Generic affine fitting permits reflections; mechanical principal-stretch analysis rejects
det(F)<=0 and condition(F)>1e8. Invalid mechanics is NaN, never plausible zero strain.

`principal_stretches(F)` returns descending stretches and CURRENT-image axes of B=F F.T.
Equal stretches have undefined (NaN) axes. `compute_series(coords,time_ms,status)` returns
engineering principal strains lambda-1, not infinitesimal tensor strain or Green strain.
Reference rows need the same valid geometry as all other rows. Optional loading_axis_deg
defines reference-image axial stretch |F a0| (clockwise from +x). fit_reason and residual_rms diagnose each row; a rejected row is retained.
Without it, axial stretch
assumes the major principal stretch: a legacy tensile-only convention.

Use all valid correspondences including reference validity. Do not relabel the major
principal direction as the loading axis in compression/shear. Incompressible uniaxial
Cauchy conversion uses sigma=lambda_axial*force/A0; predicting transverse stretch
lambda_axial^(-1/2) additionally assumes equal transverse stretches.

RANSAC is deterministic and returned inliers satisfy the RETURNED model's pixel threshold.
The final threshold classification need not equal the least-squares refit support.
ROI is a plugin-owned polygon utility. atomic_open / atomic_save_npy replace ONE file only.

## Testing, dependencies and release use

In a source checkout (Python 3.12):

    uv sync --locked
    uv run python -m unittest discover -s plugins/_sdk/example -p test_*.py
    uv run python run.py --check-plugin plugins/_sdk/example --report plugin-check.json

In an installed build:

    ECMTracker.exe --check-plugin PATH_TO_PLUGIN --report plugin-check.json

On macOS use ECMTracker.app/Contents/MacOS/ECMTracker. The command executes trusted code,
checks supported imports/metadata, and exercises empty-session launch/events/relaunch/unload.
It does NOT prove scientific correctness or test modal workflows. Read the JSON report.
`app.plugins.testing.sample_tracks()` and FakeContext support pure calculation tests with
nonzero global indices and a failed track. Add analytic fixtures for your own quantities.

The host bundles NumPy, OpenCV, PySide6, SciPy and matplotlib; exact release dependency
versions are in the shipped _sdk/uv.lock. Import PySide6, never PyQt. An installed executable
is not a uv project: uv sync/pip does not extend its embedded runtime. Request a rebuilt host
for additional dependencies; do not install packages automatically from a plugin.

## Scientific provenance outside this general SDK

The MTS elastic-energy reference is an intentionally preserved, physically motivated
approximation evaluated by the project authors using tests and finite-element simulations.
It is not asserted to be an exact zero-force equilibrium. Its implementation, fitting,
selection and existing fallback behavior are unchanged by this API cleanup. A returned index
zero from that legacy method is not a solver-success certificate. Users must assess the fit.

MTS resumes validate raw sensor/log hashes. Legacy projects without hashes resume only through
crop, requiring reference selection/tracking again; no raw files are deleted. MTS manifest v2
is authoritative; per-step JSON/CSV files are inspection snapshots. An export_generation
points to a complete immutable export directory; unreferenced generations are recoverable,
not current results. Core .bundle.npz exports atomically bind coordinates, point IDs and
global frame indices; loose .npy/.txt companions exist for legacy consumers only.
MTS clamp displacement is the sum of the two channels; force is A, B, or their average.
This assumes the acquisition channels already use compatible extension/force signs.
No sign is inferred from the data. Unknown declared units and clock resets are rejected;
missing legacy units/positional column fallback and skipped row numbers are recorded.
Pressure CSV exports retain the exact plotted values and a first-row provenance_json field
with the assumed endpoint-mtime mapping, timestamp array, source hash, anchors and reference.
