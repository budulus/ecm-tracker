# Code and scientific review — 2026-09-06

The application has a useful foundation, but it is not yet internally consistent enough to treat all exported quantities as trustworthy or the plugin SDK as a self-contained contract. The highest risks are stale state, silently misinterpreted measurements, insufficient numerical guards, and plugin failures escaping their intended containment.

This review contains **22 actionable findings**, followed by additional design and maintenance concerns. The accompanying [fix plan](FIX_PLAN.md) maps them to incremental changes and acceptance tests. No application implementation was changed.

## Scope and verification

Reviewed the core tracking, geometry, image loading, cleanup, persistence/export, mutable session state, GUI transition and interaction paths, SDK/manager, all three bundled plugins, numerical routines, documentation, regression tests, and release scripts. UI styling/icon assets were inspected as supporting code; they were not subjected to a visual acceptance test.

The existing quality gate passed on Windows with Python 3.12.13: **86 tests** (25 core/pipeline, 23 plugins, 31 MTS, 7 pressure) and **55 Python source files parsed**. These tests do not cover the failures below.

[reproduce_findings.py](reproduce_findings.py) runs **26 diagnostic cases** against the unchanged application. It uses generated temporary experiments, real offscreen Qt objects, analytic coordinates, and controlled fault injection. Cases record current behavior; they are evidence probes, not tests that assert the desired fixed behavior. The reentrancy case injects a plugin mutation through tracking's progress callback; the export case injects an I/O failure. Neither claims to reproduce a particular user's historical incident.

```powershell
.\.venv\Scripts\python.exe -B scripts/check.py
.\.venv\Scripts\python.exe -B review/reproduce_findings.py
```

Release binaries, installer upgrades, macOS behavior, large-experiment performance, and calibration against a physical specimen were not run. Build concerns below are static findings. The existing untracked `examples/` directory was preserved.

Priority: **P1** should be addressed before relying on affected workflows or distributing the SDK more widely; **P2** is a correctness or contract issue for the next hardening pass. Scientific assumptions are identified explicitly rather than counted as universally wrong formulas.

## Findings

### R01 — P1: Sequence replacement retains UI and interactions bound to the old sequence

Locations: [main_window.py:493](C:/Users/raulh/software/ecm_tracker/app/gui/main_window.py:493), [main_window.py:973](C:/Users/raulh/software/ecm_tracker/app/gui/main_window.py:973), [roi_tools.py:34](C:/Users/raulh/software/ecm_tracker/app/gui/roi_tools.py:34).

`_load_paths()` replaces project state without the cleanup-dialog/preview teardown used by tracker replacement. Loading a new sequence while Cleanup is open leaves the old dialog and preview in place. Moving a cleanup control subsequently reaches `None & keep` and raises `TypeError`. A rectangle drag begun on the old sequence can also finish on the new sequence and install its ROI there.

Reproduced by `cleanup_survives_sequence_replacement` and `roi_drag_crosses_sequence`. Centralize sequence-commit teardown: cancel old interactions, close/invalidate result-bound UI, reset preview/undo/selection, install the validated candidate, then emit coherent notifications. A failed candidate must continue to leave the existing session intact.

### R02 — P1: Tracking can install a result for a state that changed while it ran

Locations: [main_window.py:871](C:/Users/raulh/software/ecm_tracker/app/gui/main_window.py:871), [api.py:196](C:/Users/raulh/software/ecm_tracker/app/plugins/api.py:196).

Tracking processes Qt events during computation. SDK mutations have no busy-state or generation check, and completion installs the result unconditionally. A plugin callback changing the reference during progress produced `state.reference_index == 1`, `result.reference_index == 0`, and cleared seed points. This breaks the central cut/global invariant and can make display, analysis, and saving disagree.

Reproduced by `tracking_reentrant_mutation`. Capture a session/range/seed revision and immutable inputs at job start, then reject stale completion. Define which mutations are blocked, queued, or cancel an active job. A worker thread alone would not solve stale result installation.

### R03 — P1: MTS caches survive transitions that invalidate their inputs

Locations: [window.py:557](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:557), [window.py:604](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:604), [window.py:651](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:651), [window.py:902](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:902).

Fresh/adopted loads suppress the sequence handler using `_loading`, but do not consistently perform its cache/window cleanup. `kinematics()` returns an existing cache before checking whether a result still exists. After computing experiment A's kinematics and loading experiment B, it returns A's series even though the core has no result. Channel/crop invalidation also leaves cached analysis and already-open plots intact.

Reproduced by `mts_fresh_load_cache` and `mts_channel_invalidation`. One MTS transition path must clear dependent caches, refresh or clear existing views, reset outlier previews, and recompute gating. Cache keys should identify sequence, result, point selection, and range; `_kin_cache is None` is insufficient provenance.

### R04 — P1: MTS resume combines changed raw data with old reference/tracking state

Location: [project_io.py:204](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/project_io.py:204).

Resume reparses the original sensor/log paths, then restores crop indices, zero offsets, and reference state without comparing raw-input identities. The image fingerprint protects normalized image content/order, but does not protect sensor values or acquisition timestamps. A same-length edited sensor file resumes through TRACK with the old force zero. The probe changed force by +100 N; resume retained `zero_force=0.0095` against a first sample of `100.01`.

Reproduced by `mts_changed_raw_stream_resume`. Persist and verify sensor/log identity and parsing/unit conventions. On changes, require recomputation of the dependent reference/alignment or an explicit new analysis revision. Also define relocation: changing `root` alone still leaves absolute source paths pointing into the previous experiment location.

### R05 — P1: Sensor unit labels are recorded but not validated or converted

Locations: [parsers.py:205](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/parsers.py:205), [sync.py:29](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/sync.py:29), [window.py:1196](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:1196).

The downstream contract assumes seconds, millimetres, and newtons. The parser accepts a file labeled `ms` and `kN`, returns time `[0,1000]` as `time_s` and force `[0,2]` as newtons, and produces no warning. Synchronization and stress can therefore be wrong by factors of 1000. Only one displacement/force unit is retained, so different clamp units are not checked either.

Reproduced by `sensor_units`. Normalize each channel into canonical units at the parsing boundary, with a small explicit conversion table and retained source metadata. Unsupported or ambiguous units should produce a clear load error, not guessed physics. Keep a separately documented legacy convention for genuinely unitless files.

### R06 — P1: Minimum-eigenvalue LK output is filtered in the wrong direction

Locations: [dialogs.py:123](C:/Users/raulh/software/ecm_tracker/app/gui/dialogs.py:123), [tracking.py:22](C:/Users/raulh/software/ecm_tracker/app/core/tracking.py:22), [cleanup.py:70](C:/Users/raulh/software/ecm_tracker/app/core/cleanup.py:70), [tracker_result.py:12](C:/Users/raulh/software/ecm_tracker/app/models/tracker_result.py:12).

The GUI allows `OPTFLOW_LK_GET_MIN_EIGENVALS`. With that flag, the returned values measure local tracking texture/conditioning, and low values are the weak ones. Cleanup still computes maximum/mean “error” and keeps `value <= threshold`. The probe keeps eigenvalue 0.001 and drops 0.1 at threshold 0.01. The result stores window size but not the error kind or full producing parameters; saving uses mutable current settings, which can differ from those used to track.

Reproduced by `lk_eigenvalue_filter_direction`. Separate residual-error and eigenvalue-quality semantics, including the direction of filtering, aggregation, labels, and persisted provenance. Existing results of unknown error kind must not silently acquire a guessed interpretation. OpenCV defines these two outputs distinctly in its [LK reference](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).

### R07 — P1: The mechanics pipeline accepts numerically unstable and physically invalid fits

Locations: [affine.py:20](C:/Users/raulh/software/ecm_tracker/app/core/affine.py:20), [affine.py:58](C:/Users/raulh/software/ecm_tracker/app/core/affine.py:58), [analysis.py:171](C:/Users/raulh/software/ecm_tracker/plugins/pressure_strain/analysis.py:171).

Centering and rank checks are good, but numerical rank is not a useful measurement-quality threshold by itself. A nearly collinear input remains full rank and a 0.1-pixel perturbation produces a principal stretch near `1e9`. No residual, geometric condition number, determinant, or fit-validity reason is exposed downstream.

The mechanics consumers also accept an in-plane reflection (`det(F)=-1`) as essentially zero strain and a collapsed map (`det(F)=0`) as a finite -100% strain. Singular values themselves are correctly computed; treating these maps as valid measurements under the in-plane continuum assumption is the problem. A negative projected determinant could also indicate violation of the imaging assumptions, which still needs to be flagged.

Reproduced by `near_collinear_fit`, `reflection`, and `collapse`. Add measurement-quality checks and an explicit physical-admissibility policy at the mechanics layer. Do not prohibit reflections in a generic affine utility merely because a particular physical analysis disallows them. Return invalid rows with reasons instead of plausible-looking measurements. The distinction between deformation gradient, stretches, and stress is covered in [Bower's mechanics reference](https://solidmechanics.org/Text/Chapter3_5/Chapter3_5.php).

### R08 — P1: One malformed plugin metadata field can abort discovery

Location: [manager.py:81](C:/Users/raulh/software/ecm_tracker/app/plugins/manager.py:81).

Imports are caught per plugin, but metadata sorting is outside that boundary. `ORDER="first"` causes a cross-type comparison `TypeError` and aborts discovery, including startup discovery. A non-string `NAME` can fail at `.lower()` or widget construction. The fallback class search also returns the first match even when multiple classes exist, despite documentation describing a lone-class fallback.

Reproduced by `bad_plugin_metadata`. Validate metadata and the entry class while building each record; record invalid plugins as disabled with actionable errors. Reject ambiguous fallback discovery. This is especially relevant when plugins will be generated by smaller models.

### R09 — P2: Reapplying an MTS reference can leave TRACK state contradictory

Locations: [window.py:744](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:744), [window.py:795](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:795), [window.py:631](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:631).

`_apply_reference()` removes tracker/export artifacts and lowers progress, then relies on individual range setters to clear the core result. If the reference and full last frame are unchanged, those setters are no-ops and the result survives. The probe ends with a core result, progress REFERENCE, and no `trackers.npz`; analysis and export gating then disagree. Conversely, opening the range to the sequence end can discard a result even when the final requested range would be unchanged.

External core range changes are not represented by a range signal, so MTS's saved `ref_image_global`/`last_image_global` can also diverge from the core. Export partly compensates by using the live core reference, while resume/reference plots use stored fields.

Reproduced by `mts_same_reference`; the external-range path is a static finding. Use one range operation and one reference policy: either preserve a proven-compatible result and its artifacts, or invalidate both core and plugin state. Validate range compatibility when restoring trackers.

### R10 — P2: Plugin window and resource ownership is incomplete

Locations: [manager.py:176](C:/Users/raulh/software/ecm_tracker/app/plugins/manager.py:176), [manager.py:210](C:/Users/raulh/software/ecm_tracker/app/plugins/manager.py:210), [main_window.py:545](C:/Users/raulh/software/ecm_tracker/app/gui/main_window.py:545).

A plugin window using `WA_DeleteOnClose` leaves a dead Python wrapper in its record. Relaunch calls `isVisible()` outside the exception guard and raises `RuntimeError`. Qt permits this deletion behavior; see the [QWidget lifecycle documentation](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QWidget.html).

Unload relies on optional plugin cleanup. A plugin that forgets it, or throws partway through it, leaves overlays and mouse capture registered. Closing parented widgets also normally hides them rather than deleting them; repeated reloads can retain old windows/children. There is no manager shutdown call in the main window's close path.

Reproduced by `deleted_plugin_window_relaunch` and `manager_unload_resource_ownership`. Track widget destruction, make context disposal idempotent and manager-owned, disconnect owned subscriptions, and guarantee cleanup in `finally`. Hooks should release plugin-specific work, not be the sole protection for host-owned registrations.

### R11 — P2: The SDK's read-only boundary exposes live mutable objects

Location: [api.py:357](C:/Users/raulh/software/ecm_tracker/app/plugins/api.py:357).

`ctx.roi` returns the live mutable ROI. Calling `reset()` changes application geometry without its invalidation/signals. `ctx.metrics()` returns mutable cached arrays; modifying `max_step` corrupts later reads from the same context. The probe writes -999 into the cached metric successfully. The ROI cache key uses object identity, so an in-place ROI change does not invalidate it. `ctx.window`, `ctx.canvas`, and `ctx.result` also expose implementation types that are not fully specified by the SDK.

Reproduced by `sdk_mutable_objects`. Return immutable/read-only public value objects or defensive copies and key caches by explicit revisions. Keep a QWidget parent handle, but exclude host internals from the supported plugin contract. This is an accidental-mutation/API-design issue, not a claim that trusted in-process Python can be securely sandboxed with array flags.

### R12 — P2: Point Manager becomes stale when seeds change elsewhere

Locations: [main_window.py:1030](C:/Users/raulh/software/ecm_tracker/app/gui/main_window.py:1030), [point_tools.py:45](C:/Users/raulh/software/ecm_tracker/app/gui/point_tools.py:45), [point_manager.py:90](C:/Users/raulh/software/ecm_tracker/app/gui/point_manager.py:90).

The non-modal Point Manager observes result/mask/sequence changes but not seed edits. Open it with three seeds, use the canvas delete tool to remove the third, and its table still contains three rows. Selecting that stale row and deleting through the manager raises `IndexError`. Removing an earlier row can instead change what a retained numeric selection identifies.

Reproduced by `point_manager_seed_edit`. Route seed changes through a single mutation path with a seed-change signal/revision. Rebuild and reconcile selections on that event. A late bounds check is useful defense but does not fix selecting the wrong surviving point.

### R13 — P2: Affine zones lack sequence/reference ownership and a consistent drawing basis

Locations: [zones.py:213](C:/Users/raulh/software/ecm_tracker/plugins/affine_zones/zones.py:213), [zones.py:257](C:/Users/raulh/software/ecm_tracker/plugins/affine_zones/zones.py:257), [zones.py:328](C:/Users/raulh/software/ecm_tracker/plugins/affine_zones/zones.py:328).

The window does not subscribe to `sequence_changed`. Replacing a sequence retains polygons, cached fits, and enabled export controls, even without a result. Reference changes similarly leave zones tied to an old material configuration. Also, drawing is allowed on any displayed frame, but membership is always tested against `coords[0]`. A polygon drawn around a translated cluster on the current image can select the wrong reference points or none.

Reproduced cache/reset behavior in `zones_sequence_reset`; the drawing-basis issue follows directly from `_start_zone()` and `fit_zone_deformation()`. Store sequence/reference ownership with zones and clear or explicitly migrate them. Prefer drawing on the reference frame by default. Zone add/finish should also refresh an already-open stretch plot, as clear/recolor currently do.

### R14 — P2: The three strain consumers disagree on invalid fits and undefined directions

Locations: [kinematics.py:108](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/kinematics.py:108), [zones.py:343](C:/Users/raulh/software/ecm_tracker/plugins/affine_zones/zones.py:343), [gauge.py:143](C:/Users/raulh/software/ecm_tracker/plugins/affine_zones/gauge.py:143), [analysis.py:171](C:/Users/raulh/software/ecm_tracker/plugins/pressure_strain/analysis.py:171).

MTS forces a finite reference row even with too few/collinear points; pressure requires a valid reference fit and returns NaN. MTS suppresses directions for equal stretches after frame zero, but affine zones display/export an arbitrary eigenbasis at those same degenerate states. `n_points` in MTS counts the status mask before the fit filters non-finite inputs, rather than reporting actual fit participation. The public `compute_series()` also lacks the shape/time checks provided by the pressure counterpart.

Reproduced by `reference_fit_consistency`. Share a fit-result contract: eligibility count, fit-used count, validity reason, stretches, and whether directions are defined. Use one explicit reference-frame convention. Eigenvectors describe unoriented axes modulo 180 degrees; deterministic signs do not guarantee a continuous angle series.

### R15 — P2: Stress labels imply stronger physical assumptions than the implementation enforces

Locations: [window.py:951](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:951), [window.py:1167](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:1167), [plots.py:131](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/plots.py:131), [kinematics.py:80](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/kinematics.py:80).

Two assumptions need to become explicit and enforceable:

* **Preload versus sensor tare.** Force is always shifted by its value at the chosen reference, including a preforce reference. That is not automatically the specimen's physical zero force. With reference force 10 N, current force 20 N, area 5 mm² and stretch 1.2, absolute nominal/Cauchy stresses are 4/4.8 MPa, while the app reports 2/2.4 MPa. The latter Cauchy value is not even the change from reference Cauchy stress, which would be 2.8 MPa. This behavior can be a chosen convention, but the output must identify it.
* **Major principal versus loading-axis stretch.** Sorted `lambda_1` is labeled tensile/axial and used in the Cauchy correction and lateral prediction. In incompressible uniaxial compression with axial stretch 0.8, it becomes the lateral stretch 1.118 and reports +11.8% “tensile” strain. Shear, off-axis loading, or out-of-plane motion also invalidate a universal substitution of the largest measured stretch for axial stretch.

The compression case is reproduced by `compression_major_is_not_axial`; the preload example is direct arithmetic from the implementation. Keep absolute/tared/relative force conventions separate, record what area configuration means, and either define a loading axis or explicitly scope/validate the tensile approximation. The underlying scalar formulas are correct within their assumptions; [Bower's stress definitions](https://solidmechanics.org/Text/Chapter2_3/Chapter2_3.php) distinguish nominal and current-area stress.

### R16 — P2: Reference-finding failures look like successful reference selections

Locations: [reference_algorithms.py:85](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/reference_algorithms.py:85), [reference_algorithms.py:99](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/reference_algorithms.py:99), [window.py:711](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:711).

No threshold crossing, insufficient input, and failed elastic fits all return index zero. The caller records that as a detected reference and zeros the channels. The spring-hinge routine uses `sol.x` without checking solver success, fit quality, or parameter identifiability; its energy search is over a fixed extrapolated interval. Existing tests only require an integer in range, so an implementation that always returns zero could pass the energy assertions.

There is also a physical distinction: minimizing the implemented **elastic** energy along a gravity-loaded equilibrium branch does not generally locate zero axial force. Writing `eta=lambda*sin(phi)`, its own model gives `dU/dxi = f_hat + gamma*deta/dxi`. At an elastic-energy minimum, `f_hat` need not vanish. For `gamma=0.1`, `beta=0.05`, the probe finds `xi≈0.923812`, with `f_hat≈0.093402`, not zero. This derivative and example are derived from the repository's model, not a claim about a validated specimen model.

Reproduced by `energy_minimum_is_not_zero_force`; failure-return and solver checks are static/test evidence. Return a structured success/failure result with diagnostics, distinguish a chosen energy-reference convention from a stress-free reference, and preserve the previous valid reference when detection fails. Persist search window, preforce threshold, algorithm version, and fit parameters to make selection reproducible. The knee label “max curvature” should also match its actual normalized chord-distance method.

### R17 — P2: MTS synchronization is calculated on one timeline and partly plotted on another

Locations: [window.py:1342](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:1342), [crop_plot.py:58](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/crop_plot.py:58), [window.py:951](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:951).

Interpolation correctly uses `sensor_time + offset`, but the crop plot receives unshifted sensor time and unshifted image time alongside shifted/interpolated force. With +100 ms offset, the plotted sensor stream still starts at 0 ms instead of 100 ms. Its coverage mask also tests the unshifted interval. This makes the plot used to judge synchronization misleading.

Separately, reference selection accepts clamped out-of-coverage values. Stress plots discard `in_range`, and the measures CSV permits that column to be unchecked. With no sensor coverage at all, the probe still returns a finite all-zero force curve after zeroing. A nearest-frame crop-end mapping may also select a frame beyond the crop endpoint; this boundary policy should be explicit.

Reproduced by `mts_crop_plot_unshifted_sensor_time` and `mts_no_sensor_coverage`. Plot both streams in the same shifted coordinate system. Carry mandatory coverage/validity into analysis outputs and make invalid samples gaps or clearly marked estimates. Do not use an extrapolated/clamped sample as a silently valid reference.

### R18 — P2: Atomic files do not make a multi-file export atomic

Locations: [project_io.py:158](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/project_io.py:158), [export.py:21](C:/Users/raulh/software/ecm_tracker/app/core/export.py:21), [window.py:552](C:/Users/raulh/software/ecm_tracker/plugins/mts_uniaxial/window.py:552).

Each artifact is atomically replaced, but the export group is not. An I/O failure after replacing coordinates leaves new coordinates with old point IDs/aligned data. The injected failure produced coordinates `(3,2,2)` with IDs `(1,)`, while the previous manifest still said EXPORT complete. Restoring the in-memory progress integer does not restore files. Invalidation deletion failures are only transient status messages, so stale artifacts can remain discoverable without a durable invalid marker.

Core exports also use a shared `sequence.txt` regardless of the custom coordinates filename. Exporting different ranges under different names in one directory overwrites the single range descriptor for earlier exports.

Reproduced by `partial_export`; the shared-sidecar issue is static. Commit a versioned export bundle through one authoritative manifest/pointer after staging and validating every artifact. Give each coordinate export its own metadata or a self-contained container. Keep previous committed output recoverable.

### R19 — P2: Version-2 tracker loading checks shapes but accepts impossible tracking semantics

Locations: [tracker_io.py:148](C:/Users/raulh/software/ecm_tracker/app/core/tracker_io.py:148), [tracker_result.py:33](C:/Users/raulh/software/ecm_tracker/app/models/tracker_result.py:33).

The loader accepts a forward status sequence `[1,0,1]`, a negative LK error, and seeds that disagree with `coords_fw[0]`. The probe imports all three through `load_trackers_from()`. Other unverified relationships include backward seed identity/validity, frozen coordinates after terminal failure, and whether stored FB summaries match their source arrays. Scalar index coercions also truncate non-integral numeric inputs instead of validating their type/domain.

Reproduced by `tracker_v2_semantics`. Centralize result validation for construction, load and save; validate post-conversion finite values and meaningful semantic invariants. Recompute derived summaries from authoritative arrays or verify them within an explicit tolerance. Keep legacy migration explicit; do not silently repair a supposedly current-format file.

### R20 — P2: RANSAC's returned inliers need not satisfy its returned model

Location: [affine.py:146](C:/Users/raulh/software/ecm_tracker/app/core/affine.py:146).

The winning hypothesis determines the mask, then its consensus is refit and a different model is returned with the old mask. The deterministic probe obtains an accepted residual 0.82524 for a 0.8-pixel threshold. The description says inliers satisfy the reprojection threshold, which is no longer guaranteed for the returned model.

The documented small-sample fallback also calls every finite point an inlier when `n <= sample_size`, regardless of residual. That may be a deliberate inability-to-clean policy, but it is not evidence that every point fits. Adaptive confidence uses `ratio**sample_size`, an approximation to the probability of drawing an all-inlier subset without replacement; it is particularly optimistic for small populations/large samples and does not account for degenerate samples.

Reproduced by `ransac_final_membership`. Define whether the mask represents the hypothesis consensus or final-model thresholding, and make code, preview, export and docs agree. If promising the latter, use a bounded refinement/reclassification process with a defined stopping rule. Report insufficient redundancy separately from a verified consensus and avoid promising exact statistical confidence.

### R21 — P2: SDK documentation contradicts behavior and does not form a complete standalone contract

Locations: [plugin-api.html:833](C:/Users/raulh/software/ecm_tracker/plugins/plugin-api.html:833), [README.md:1](C:/Users/raulh/software/ecm_tracker/plugins/README.md:1), [api.py:1](C:/Users/raulh/software/ecm_tracker/app/plugins/api.py:1).

Concrete documentation defects:

* The HTML guide says range setters leave a result untouched and misaligned. The implementation discards it. A generated plugin following the guide can dereference a result immediately after discarding it.
* SDK introduction/README say filtering is the only mutation, despite supported sequence loading, tracker loading, and range control. Conversely, the HTML reference/cheat sheet omit the actual load/save tracker contract and a full `load_sequence()` reference.
* `coords()`'s primary docstring does not warn that active points can have failed/frozen positions; the important status requirement is elsewhere. Active-mask filtering and measurement validity are different concepts.
* `metrics()` defaults to all P points while `coords()` defaults to active points; `api.py` sends readers into `cleanup.py` to discover fields. The HTML explains this better, but cannot substitute for a single consistent contract.
* Event ordering, sequence reset versus result notifications, no-result behavior of frame helpers, errors/cancellation, array lifetimes/copy cost, threading, settings schema, and the lifecycle of a hidden versus disposed window are underspecified. There is no range/seed event or API compatibility/version declaration.
* Bundled examples import `app.core.affine`, `app.core.roi`, and `app.core.atomic_io`. The purported SDK does not say whether these imports are stable. Copying examples therefore creates undocumented dependencies on internals.

Static contract audit, reinforced by the reproduced state/lifecycle bugs. Publish a compact normative SDK, generate its expanded reference, and test complete snippets as plugins. Add safe task-oriented data access and manager-owned lifecycle support before telling weaker models that the façade alone is sufficient.

### R22 — P2: Packaged plugin authoring and upgrades are not a complete supported workflow

Locations: [build.py:52](C:/Users/raulh/software/ecm_tracker/build.py:52), [build.py:97](C:/Users/raulh/software/ecm_tracker/build.py:97), [build.py:108](C:/Users/raulh/software/ecm_tracker/build.py:108), [build_macos.py:90](C:/Users/raulh/software/ecm_tracker/build_macos.py:90).

The build explicitly ships loose `plugins/` and compiles `app`; it does not explicitly ship a standalone SDK source/stub reference. The shipped README points authors at `../app/plugins/api.py` and tells them to edit `pyproject.toml`/run `uv sync`, which is a source-checkout workflow rather than a frozen-client dependency workflow. There is no artifact-level SDK check to prove these instructions work for a client without the repository.

The Windows installer recursively installs the whole payload into the same directory where clients are encouraged to edit bundled plugins, without a custom-plugin preservation/versioning policy. On macOS, plugins are copied into the bundle after Nuitka's signing stage, and authoring is encouraged inside the bundle; signing/integrity needs explicit release validation and final signing after staging.

These are static release/design findings, not claims of a tested installer or signature failure. Ship an offline authoring kit and explicit host dependency/version manifest, separate user plugins from installed examples, and validate updates preserve user work. Verify final macOS signatures after all staging.

## What is mathematically consistent

The findings above do not invalidate every formula. Several core choices are coherent:

| Quantity | Implemented meaning and valid interpretation |
|---|---|
| Affine map | `current = reference @ F.T + b`; centering separates translation and improves the solve. |
| Principal stretches | Square roots of eigenvalues of `B = F F.T`, equivalently singular values of F. Correct for principal stretch magnitudes. |
| Directions | B's eigenvectors live in the current configuration; using them on a current-frame gauge is correct when the eigenvalues are distinct. |
| Strain | `lambda - 1` is dimensionless stretch-based/engineering principal strain. It is not Green–Lagrange strain `(lambda²-1)/2`, logarithmic strain `log(lambda)`, or the small-strain tensor for arbitrary finite rotations. “Linear strain” should be defined precisely. |
| Nominal stress | Physical axial force divided by reference area is the uniaxial first-Piola/nominal stress magnitude. N/mm² equals MPa. The force/area reference convention must match the experiment. |
| Cauchy correction | `sigma = lambda_axial * P` requires incompressible uniaxial kinematics and the appropriate axial stretch. It is not a general conversion for arbitrary measured F. |
| Lateral prediction | `lambda_transverse = lambda_axial**(-1/2)` assumes equal stretches in the two transverse directions as well as incompressibility. Incompressibility alone supplies only `lambda_a * lambda_t1 * lambda_t2 = 1`. |
| Synchronization | `interp(image_t, sensor_t + offset, value)` matches the documented positive-offset convention. R17 concerns presentation/coverage, not that interpolation sign. |
| Forward/backward tracking | Backward arrays are restored to cut order. Terminal forward validity prevents revival, and invalid forward endpoints are excluded as backward seeds. |
| Quality summaries | Excluding the artificial seed-frame LK zero and backward seed's identically zero discrepancy avoids dilution. One-frame zero-motion/error handling is reasonable under its validity convention. |

The continuum definitions are consistent with [Bower's hyperelasticity reference](https://solidmechanics.org/Text/Chapter3_5/Chapter3_5.php). The table describes what follows mathematically from the code and its assumptions; it does not validate those assumptions for a particular experiment.

FB mean/max here summarize forward/backward trajectory disagreement over jointly valid frames, excluding the backward seed. They are not solely the final round-trip return distance at the original reference, and a backward track can have finite summaries without reaching that reference. Document this distinction and expose coverage/completion if consumers require a complete round trip. Likewise, status-failure counts count cumulatively invalid frames, not independent LK failure events; max step is pixels per image transition, not velocity.

## Additional design, measurement, and maintenance concerns

### S01 — Repeated work scales poorly in frame-navigation and paint paths

`ctx.coords(active_only=True)` and `track_status()` boolean-index and copy the whole time series on each call. Affine-zone refresh/paint and gauges often need only one frame but copy every frame. `_update_tool_states()` scans all statuses to compute exportability during navigation. Point Manager recomputes all metrics on every mask change. Reference fitting/RANSAC/live plots run synchronously, and event handlers can trigger redundant full replots/persistence writes.

These are static complexity concerns; no large-data benchmark was run. Add frame-oriented reads and cache immutable-result calculations by revision. Profile representative N/P/image sizes before choosing worker scheduling. Share read-only buffers where safe rather than adding another full copy per snapshot.

### S02 — Shared algorithms are only partly shared

Affine fitting is centralized, but fit-series validity/reference rules, stress formulas, synchronization extraction, matplotlib loading, and gauge rendering remain duplicated. Large GUI classes combine state transitions, persistence, scientific decisions, formatting, and rendering. This duplication already explains several disagreements above. Extract small pure calculation/transition services while fixing their behavior, not as an unrelated rewrite.

The declared strict `gui → models → core` direction also disagrees with `core.tracking` and `core.cleanup` importing `TrackerResult` from models. The important Qt-free computational boundary currently holds. Decide and document whether shared data records belong below core or are an explicitly allowed core dependency.

### S03 — The quality gate is narrower than its description and examples suggest

The gate parses only the listed source directories, excluding root `build.py`, `build_macos.py`, and `run.py`; it runs a fixed module list and has no type/doc-example/API-boundary checks. Its energy-reference test checks only range/type, not a known physical solution or successful fit. Tests create many temporary directories without scoped cleanup. `requires-python >=3.9`, the 3.12 pin, and `tomllib` in build scripts should be reconciled with a stated runtime/build support policy. Both release scripts reuse and delete the same fixed temporary build directory, relying on a prose “one build at a time” rule.

Add focused checks for the public contract and observed failures, automatic test/source discovery, declared version/platform checks, and a build lock or unique scratch directory. Avoid chasing unrelated style warnings before the correctness work.

### S04 — Acquisition and spatial assumptions need explicit provenance

Pressure endpoint alignment deliberately stretches logger elapsed time onto relative image modification times. Its first/final anchor formula and cut/global slicing are internally consistent with the README. It does **not** establish that modification time is acquisition time or that the two recordings actually share endpoints. File copying/export can change timestamps without changing image pixels, so the pixel fingerprint cannot prove that pressure alignment is unchanged. Record image timestamps, pressure identity, anchor mapping and time-scale factor; show the assumption and permit measured timestamps when available.

The MTS parser silently skips invalid image-log rows, sorts non-monotonic sensor time, and averages repeated timestamps. Sorting/averaging can combine distinct acquisition runs if the instrument clock resets. Record affected rows/segments and make repair policies explicit. Summing clamp displacement and averaging force assumes compatible signs and matched axes; it cannot be verified from six bundled sensor samples.

Pixel-space strain assumes a calibrated image plane, appropriate pixel aspect ratio, stable camera geometry and limited out-of-plane motion. A constant isotropic pixel-to-mm scale cancels in stretch, but anisotropic calibration and perspective do not. Homogeneous least squares is weighted by detected point density; spatially uneven points or a changing valid set can alter an apparent average strain. RANSAC can reject real heterogeneous deformation, so its output should be described as consistency with a homogeneous model rather than proof of a bad track.

Also validate seed/ROI geometry at public boundaries: canvas clicks can occur outside the image, shape tools can create zero-area/self-intersecting regions, and regular grids inherit their bounds. Image normalization uses fixed scaling for uint16 but switches float scaling based on each frame's value range; a float sequence crossing that convention needs a sequence-level policy to preserve brightness consistency.

These are measurement/domain limitations or hardening opportunities, not evidence that the documented pressure endpoint policy or every nonuniform experiment is inherently wrong. The fix plan keeps them explicit and testable.
