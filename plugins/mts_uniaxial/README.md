# MTS Uniaxial

Synchronizes high-frequency MTS sensor data (force + displacement, two clamps) with a
lower-frequency image sequence for uniaxial tensile testing, then hands off to the core app's
ROI / tracking / cleanup. It does **not** reimplement any of those — it scopes the experiment in
time and exports the result aligned with the mechanical state.

## Workflow (top to bottom in the window)

1. **Load experiment** — pick the experiment **root** folder. Defaults resolve `veddac/` (frames
   + `VDCCam.log`) and `mts/specimen.dat`; override either with the `…` buttons. Both paths are
   always shown relative to the root, even when picked manually. Images are loaded in
   **acquisition-log order** (the `VDCCam.log` row order), never filename sort. Also set the
   specimen's reference cross-section — **width** (default 10 mm) and **thickness** (default
   0.5 mm), always in mm — and the **Incompressible material** flag (on by default). These are
   *parameters*, not a workflow step: editing them never invalidates tracking; they only feed the
   stress curves and the measures CSV. A₀ = width × thickness is shown live.
2. **Channels & temporal offset** — displacement is always clamp A + clamp B. Force is clamp A,
   clamp B, or their average. The temporal offset (ms) corrects imperfect hardware sync:
   **positive = the sensor recorded later than the images** (added to the sensor timeline).
3. **Crop experiment window** — two sliders set the first/last sensor sample; the force-vs-time
   and force-vs-displacement plots update live, with image frames overlaid (toggle the overlay
   with **Show image frames**).
4. **Find reference** — two sliders set a **search sub-window** inside the crop (shown as a
   force-vs-displacement plot of just that window); a chosen algorithm finds the zero-stress
   reference sample within it. The local index is mapped back to an absolute sensor index and then
   to the nearest image frame, which becomes the reference (cut 0). Displacement and force are
   zeroed at that frame (value shift only — time is never altered). The crop end maps to the last
   frame. The core app's reference/last range is set automatically. **Set preforce** picks the
   first sample whose force exceeds the pre-force threshold (N). An override spinbox nudges the
   reference frame manually. The found reference is marked on all three plots.
5. **Track in the main window** — define the ROI, seed points, run tracking, and filter as usual.
6. **Homogeneous RANSAC filtering** — once a reference is set and points are tracked, the plugin
   fits **one homogeneous in-plane deformation gradient F per frame** across all active points
   (least-squares affine, reference → current) and derives the left Cauchy–Green tensor B = F Fᵀ →
   principal stretches λ₁ ≥ λ₂, principal directions (current config), and linear strains
   εᵢ = λᵢ − 1. This is computed lazily and cached regardless of RANSAC. **Apply RANSAC** fits the
   reference → **last** frame correspondence, drops the outliers through the app's undoable active
   set (so it's reversible in the main window's Cleanup), and refreshes every open plot.
7. **Plotting** — opens standalone, live-refreshing matplotlib windows: **Kinematics** (ε₁/ε₂ over
   time and ε₂ vs ε₁, with the incompressible prediction ε₂ = (1+ε₁)^(−1/2) − 1 overlaid when the
   flag is on), **Piola–Kirchhoff stress** P = force/A₀ vs ε₁, **Cauchy stress** σ = λ₁·P vs ε₁
   (enabled only when incompressible), and a **Direction gauge** — the image sequence with its own
   local frame slider and the principal-strain axes drawn at the points' centroid.
8. **Export aligned data & measures** — *Export aligned data* writes the tracked coordinates with
   the per-frame interpolated, zeroed force & displacement (as before). *Export measures (CSV)*
   writes one row per frame with a checkbox-selected subset of the derived columns (strains,
   stretches, PK/Cauchy stress, force, displacement, principal-direction angle, …).

## Project folder (`<root>/mts_uniaxial_project/`)

Individual, inspectable files, written after each step so work resumes on reopen:

| File | Step | Contents |
|------|------|----------|
| `manifest.json` | all | master index: paths, progress, header/units, offset convention |
| `image_log.csv` | load | filename, time_ms, resolved path (acquisition order) |
| `sensor_raw.csv` | load | time_s, disp_a, force_a, disp_b, force_b |
| `sync_params.json` | channels | force channel + temporal offset |
| `crop.json` | crop | first/last sensor index |
| `reference.json` | reference | algorithm, reference/last frame, zero offsets |
| `tracked_coords.npy` | export | `(frames, points, 2)` float32 (active points) |
| `point_indices.npy` | export | original ids of the exported points |
| `aligned_data.csv` | export | per frame: `frame_global, image_time_ms, displacement, force, in_range` |
| `measures.csv` | export | per frame: selected derived measures (time, strains, stretches, PK/Cauchy stress, …) |

`in_range = 0` flags frames whose timestamp falls outside the sensor's time coverage (the value
is clamped to the nearest sensor endpoint).

## Integrity: downstream wipe

Steps form a strict dependency chain. Editing any step conservatively deletes everything
downstream of it — in memory and on disk — so the project can never hold stale, mismatched data.
Reopening restores up to the reference (the in-app tracking result is not persisted here, so
re-run tracking to export again).

## Reference-finding algorithms

Baked into `reference_algorithms.py` as a registry. Add one by decorating a function with
`@register("Name")` matching the signature `find_reference(displacement, force) -> sensor_index`
(index into the **search sub-window**); it appears in the dropdown automatically. Three ship:
force-onset (baseline-noise threshold), force–displacement knee, and **Set preforce** (first
sample above a user-set pre-force threshold — the one algorithm the UI passes an extra argument to).

## Kinematics module (`kinematics.py`)

Qt-free, unit-tested numpy: `fit_deformation_gradient` (least-squares homogeneous affine fit over
the valid points), `principal_decomposition_B` (stretches + current-config directions from
B = F Fᵀ), `ransac_affine` (deterministic outlier filter, ported from `affine_zones`),
`eps_2_incompressible`, and `compute_series` (the per-frame `KinematicsSeries`). The window caches
its output and invalidates it on `result_changed` / `mask_changed` / a new reference, so the plots
always reflect the current active set. The material parameters (width/thickness/incompressible)
are **not** a `Step` — they persist in the manifest and only redraw stresses, never wiping tracking.

## Extending (future stages)

The project folder and `manifest.json` `stages` block, the `Step` enum, and the
`MtsProjectState` dataclass are designed to grow — further post-processing slots in either as new
steps after EXPORT (reusing the same `invalidate_from` wipe chain) or, like the kinematics, as a
lazily-cached derivation gated on a result.
