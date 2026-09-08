# Review fix implementation — 2026-09-07

The implementation preserves the complete reference_algorithms.py file, not just the
elastic-energy function. SHA-256:
`C3BECE51ED1677E69B4597E7EE71D4BD50EA890C31F472577D043FBD98174D22`.
Its equations, fitting, minimization, selection and legacy fallback are unchanged.
The author guide describes this as the authors' physically motivated, experimentally/FE
evaluated approximate reference, not an exact zero-force equilibrium.

## Changes

| Review items | Implemented resolution |
|---|---|
| R01–R02 | Teardown of old interactions/dialogs; atomic range changes; revision-guarded tracking |
| R03–R04 | MTS cache/view invalidation; raw sensor/log fingerprints; relocated source paths; conservative legacy resume |
| R05–R06 | Independent clamp-unit conversion to s/mm/N; clock-reset rejection; producing LK metadata; distinct lower/upper quality filters |
| R07, R14, R20 | Shared Qt-free kinematics; conditioning and det(F) guards; identical reference validity; undefined isotropic axes; final-model RANSAC membership |
| R08, R10 | Validated plugin metadata/version; ambiguous-class rejection; deleted-window relaunch; host-owned cleanup and shutdown |
| R09 | Single final-range commit, compatible same-reference tracking retained/persisted; external ranges invalidate MTS reference; restore checks expected range |
| R11–R13 | Detached snapshots/ROI, read-only metrics, stable IDs/revisions; seed notifications/selection invalidation; reference-owned zones |
| R15 | Explicit absolute/reference-change force basis and optional loading axis; consistent stress plotting/export; stated constitutive assumptions |
| R16 | Non-energy no-onset/unmet-preforce/degenerate-window failures do not silently select zero; search parameters persist. Energy behavior deliberately preserved |
| R17–R18 | Shifted plotting timeline; reference coverage checks and NaN stress outside coverage; immutable MTS export generations plus authoritative manifest; atomic core bundles |
| R19 | v3 producing metadata; semantic input validation, terminal statuses, matching seeds, nonnegative errors and recomputed FB summaries; v1/v2 compatibility |
| R21–R22 | Standalone contract, generated offline reference, SDK-only example/test fixture/conformance command; public math helpers; separate user/managed plugins; SDK staging checks |
| S01–S04 | Cached trajectory access/exportability/metrics; shared mechanics and core result record; dynamic test discovery/root parsing/temp isolation; pressure provenance and input conventions |

The quality gate runs existing regressions plus new desired-behavior tests. The earlier
reproduce_findings.py is historical diagnostic evidence, NOT the post-fix acceptance gate.
Tests that asserted known incorrect behavior were changed to assert the corrected contract
(e.g. RANSAC's small-sample mask, export range consistency and authoritative manifests).

## Compatibility decisions

- Python support now explicitly targets 3.12; pyproject.toml, .python-version and regenerated
  uv.lock agree. Installed runtime versions did not need changing.
- Tracker NPZ v3 distinguishes producing LK parameters from current editable controls.
  v1/v2 quality is unknown, rather than guessed from possibly edited settings.
- MTS manifest v2 is authoritative. Per-step JSON/CSV files are inspection snapshots.
  Editing/deleting a reference.json snapshot no longer changes the committed project.
- Legacy MTS projects without raw fingerprints resume through crop at most. They require
  reference selection and tracking again; their original input files are preserved.
- Each MTS export commits a fresh export-* directory through one manifest replacement.
  Old/unpublished generations remain recoverable but are not current results. Read the
  manifest or project_io.export_paths(), not former top-level export filenames.
- Core .bundle.npz is the atomic coordinate/range/point-ID artifact. Loose .npy/.txt views
  remain for compatibility but cannot provide a multi-file atomicity guarantee.
- Stress keeps the legacy reference-change/major-tensile-axis convention by default,
  explicitly labeled. Absolute force preserves preload; a specified loading direction
  uses |F a0| instead of sorting a compressive axial stretch into the minor component.
- MTS measure CSVs add force basis, axis choice, coverage, fit reason/residual and axial
  stretch. Pressure CSVs add first-row provenance_json without changing existing columns.
- Source plugins/ and installed legacy plugins/ remain supported. New releases stage
  managed examples under bundled_plugins/, so upgrading does not overwrite legacy edits.
  User plugins outside the installation have highest precedence.

## Validation and release boundary

Run:

    .venv\Scripts\python.exe scripts/check.py
    .venv\Scripts\python.exe scripts/benchmark_sdk.py
    .venv\Scripts\python.exe scripts/build_sdk.py

The suite includes 86 existing checks, 24 new review regressions, two SDK-example tests,
and the host plugin smoke test. It exercises synthetic load/reference/track/cleanup/export/
resume, failure paths, source SDK-only authoring, and simulated packaged discovery/staging.
The energy-reference numerical test and untouched-source hash check both pass.

On this Windows machine, a 400-frame × 1,500-point benchmark took about 6 ms for the initial
snapshot and 7 ms for 2,000 cached frame accesses, with 1,733 bytes peak additional allocation.
These are local measurements, not a cross-machine performance guarantee.

Not claimed as completed validation:

- A full Nuitka binary/installer build and real upgrade test, and a macOS build/signature/
  notarization test. Build scripts now stage the SDK, run checks, use unique scratch folders,
  and sign/verify macOS only after staging; those external release operations were not run.
- A separate weaker-model authoring study or external static type-checker run. The kit was
  exercised with the SDK-only example, not with an independently recruited/model author.
- Moving nonlinear fitting/RANSAC off the UI thread or a broad GUI rewrite. Revision safety
  and caching are implemented; worker scheduling should be a measured follow-up.
- A new solver-success certificate for the energy algorithm. Its existing zero-on-failure
  behavior remains specifically because the user asked to preserve that implementation.

No user experiment files or the pre-existing examples/ directory were changed.
