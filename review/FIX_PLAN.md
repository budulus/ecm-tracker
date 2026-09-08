# Incremental correction and plugin-SDK plan

This plan addresses findings R01–R22 and concerns S01–S04 in [CODE_REVIEW.md](CODE_REVIEW.md). It is a proposed implementation sequence, not a record of completed fixes. Application code is unchanged.

The intended result is a consistent tracking/analysis pipeline and an SDK that lets an author implement a plugin from a small, standalone contract, examples, and a test harness. Preserve the working numerical conventions and file compatibility unless a finding specifically requires a correction. A passing gate reduces regression risk; it cannot prove that no future input can fail.

## Implementation sequence

| Change set | Scope | Findings | Completion evidence |
|---|---|---|---|
| 1. Contract corrections and regression cases | Fix immediately false documentation; encode observed failures as desired-behavior tests; state supported scientific conventions. | R21; test gaps in S03 | Existing 86 tests stay green; new cases fail specifically at the known defects before their fixes. |
| 2. Coherent host state transitions | Sequence teardown, revision checks, atomic range updates, seed events and consistent result installation. | R01, R02, R09, R12, R19 | State-transition, stale-job, interaction and tracker-file tests pass. |
| 3. Plugin lifecycle and discovery | Validated metadata, destroyed-window handling, owned resources, reload and shutdown cleanup. | R08, R10 | Broken/ordinary plugins coexist through repeated lifecycle tests. |
| 4. Immutable, typed SDK and authoring kit | Public data snapshots, consistent point/frame mapping, subscriptions, versioning, runnable examples, generated docs. | R11, R21; S01 | Plugins built using only the kit pass the same contract checks against a fake context and the real host. |
| 5. Scientific calculation contracts | LK error kinds, conditioned mechanics fits, consistent strain/direction output, final RANSAC semantics. | R06, R07, R14, R20 | Analytic, degenerate, noisy and round-trip fixtures pass with explicit tolerances. |
| 6. MTS state and measurement conventions | Cache ownership, source identity, units, reference diagnostics, force/stress conventions, shifted plots and coverage. | R03, R04, R05, R09, R15, R16, R17; S04 | Full transition/resume tests and known physical/arithmetic cases pass; unsupported cases are labeled or rejected. |
| 7. Analysis views and durable exports | Zone ownership/drawing basis, shared plot/export series, committed artifact generations and provenance. | R13, R18; remaining S04 | Plot and CSV values agree, stale output cannot be marked current, export faults preserve a valid previous generation. |
| 8. Release and maintenance gates | Offline SDK packaging, custom-plugin preservation, final signing checks, test discovery, targeted performance work. | R22; S01, S02, S03 | Installed-artifact plugin smoke test, update-preservation test and supported-platform checks pass. |

Treat these as separately reviewable changes, splitting large rows further when needed. Fixing R05's unit boundary and R08's metadata validation can land early without waiting for broad refactoring. Public snapshot details should be settled with the transition contract before migrating every bundled plugin.

## 1. Establish the contract and useful regression tests

Correct the HTML statement that range changes preserve a tracking result immediately. Document the current behavior of reset, active masks, frozen coordinates, global/cut indexing and settings before introducing additions. Mark the current undocumented `app.core` imports as legacy example dependencies pending public replacements.

Convert the relevant probes from [reproduce_findings.py](reproduce_findings.py) into focused acceptance tests asserting the desired behavior. Keep production fixes and the regression that demonstrates each fix in the same reviewable change. Use synthetic fixtures with known transforms and isolated settings/temporary directories; do not make tests depend on the user's experiment or local plugins.

Lock down these existing invariants:

* Ordered image identity and global/cut mapping stay explicit. Failed loads do not change the live session.
* Tracking results remain immutable; filtering operates on a separate mask and remains undoable.
* Forward validity is cumulative; failed positions are not measurements. Backward seeds must be valid forward endpoints.
* Filtering cannot resurrect inactive points. Empty active sets remain legal states.
* Existing standard LK mode and pressure endpoint mapping retain their documented meaning.
* The compute/I/O layers remain Qt-free, and plugin algorithms can be tested without creating the entire GUI.

Save small version-1/version-2 tracker fixtures and legacy MTS manifests as compatibility examples. Assert meaning, not byte-for-byte equality of floating-point serialization. Add coverage of every migration policy introduced below.

## 2. Make state transitions indivisible to consumers

Introduce a small host session/controller boundary for sequence installation, range changes, seed changes, result installation, result clearing, and point filtering. Start by extracting existing operations; preserve UI handlers as adapters. Avoid a wholesale replacement of `MainWindow`.

Use monotonically advancing revisions or opaque revision tokens for scientific inputs. A displayed-frame change should not invalidate the entire tracking snapshot. Sequence/range/seeds/result/mask/ROI changes must advance the relevant revision, so work can prove that the data it used is still applicable.

A sequence commit should follow this order:

1. Validate/decode the candidate independently, with cancellation leaving the current session intact.
2. Cancel old canvas interactions while they still refer to the old state. Close/invalidate Cleanup and stale selections/previews.
3. Install the entire new session state and synchronize controls under the existing signal guards.
4. Emit documented notifications only after the state is coherent. No listener should observe half of a transition.

Provide one `set_frame_range(reference, last)` operation. Compute its final target before touching the live state; validate/clamp according to the published policy, invalidate tracking once if required, and emit one coherent range notification. Keep individual legacy setters as adapters.

Tracking captures its sequence, range, seeds and producing parameters. Completion compares the captured revision to the live session before installation. Busy-state behavior must be specified for SDK calls; a request can return a typed busy/stale error or explicitly cancel/restart, but must not mutate beneath an in-flight result silently. Apply the same rule to future background plugin jobs.

Route every seed-edit path through one operation and emit `seeds_changed`. Rebuild Point Manager selections after changes and validate any delayed delete request against the revision it selected from. Define whether changing an ROI preserves manual seeds or clears derived seeds, then apply that policy consistently.

Move semantic result checks into a shared validator used by construction/load/save: exact scalar index domain, shapes, finite converted coordinates, cumulative statuses in their correct pass directions, frozen dead positions, valid backward seed relation, seed/reference agreement and nonnegative error domain. Recompute FB summaries or verify them using the documented algorithm/tolerance. Construct a validated candidate result before changing any GUI state.

Acceptance cases: old Cleanup during sequence replacement; a drag spanning replacement; seed edit while Point Manager is open; changed range during tracking progress; cancelled/failed load; nonzero-reference save/load; current-format revived statuses rejected; legacy migration still works; every emitted signal observes a coherent snapshot.

## 3. Let the manager own plugin resources

Validate `PLUGIN`, `NAME`, `DESCRIPTION`, `ORDER`, API compatibility and identifier rules before storing a record. Use explicit types and useful diagnostics. An ambiguous class fallback should be an error, not whichever class happened to be imported first. Catch a bad plugin's metadata failure without losing valid records.

Track window destruction and clear the corresponding record. Put both refocus and launch inside the plugin error boundary. Support both documented window policies: a cached window hidden on close, and a window deleted on close and recreated on launch.

Add idempotent `PluginContext.dispose()` owned by the manager. It releases the context's overlays, mouse capture and subscriptions even if construction, launch or `on_unload()` fails. A context must not clear a newer context's interaction. Dispose failed-launch resources according to a documented retry policy. Use explicit widget deletion when ending the plugin instance's lifetime; ordinary hide/reopen can retain its state.

Invoke shutdown cleanup when the host closes. Ordinary hiding, final disposal, module reload and shutdown need distinct semantics, particularly because MTS must keep persisting mask changes while its window is hidden. Do not “fix” leaks by disconnecting MTS integrity observation on every ordinary close.

Acceptance cases: invalid metadata; ambiguous entry classes; throwing constructor/launch/unload; deleted Qt window; repeated hide/reopen; multiple reload cycles; self-removing/throwing overlays; interaction ownership replacement; host shutdown. Check object/resource counts remain bounded after event-loop deletion processing.

## 4. Build a small SDK that supports authors using weaker models

The objective should be: **one short normative contract, one matching example, and a conformance command are sufficient to write an ordinary plugin**. More prose alone will not compensate for APIs that expose mismatched axes or require manual host cleanup.

All method/type names in this section are proposed additions and must not be presented in the current documentation as already implemented.

### Data access

Add a typed immutable `TrackingSnapshot`, obtained through a clearly named accessor such as `ctx.tracks()`. It should carry the data needed together:

| Field | Contract |
|---|---|
| `revision` | Opaque token identifying the scientific state/selection used by the snapshot. |
| `frame_indices` | Global indices, shape `(N,)`, mapping every coordinate row explicitly. |
| `point_ids` | Original point IDs, shape `(A,)`, mapping every selected point column explicitly. |
| `coords` | `(N,A,2)`, float32, `(x,y)` image pixels, stable read-only data for the snapshot's lifetime. |
| `valid` | `(N,A)`, boolean validity aligned exactly to coordinates; active does not imply valid. |
| `quality` | Typed per-point metrics aligned to those same A IDs, with units, error kind and coverage defined. |
| `reference_index`, `last_index` | The range that belongs to this snapshot, independent of later UI state. |
| `metadata` | Producing tracking parameters, coordinate convention, sequence identity and any relevant calibration identifier. |

Offer frame-oriented access through the snapshot, taking a global index and returning an explicitly absent result outside its range. This should return views/selected rows efficiently, not copy N frames on every paint. Assemble active-point selection once per revision or use a lazy selection design with a documented cost; do not add full copies unconditionally.

Return public ROI and metric value objects rather than live model instances. NumPy read-only arrays guard ordinary accidental writes; they are not a security boundary against arbitrary Python code. Supply typed ndarray annotations and shape/unit documentation without requiring authors to learn a separate array-schema framework.

Keep legacy `coords()`, `track_status()`, `point_indices()` and full-P metrics available during migration. The new path should be the only path shown in the beginner examples, so authors need not choose among subtly different conventions.

### Mutations and events

Provide revision-aware filtering, for example an additive `revision=` parameter on `apply_keep_mask()`. A delayed filter produced for old IDs or a changed active selection must be rejected with a clear stale-data error; matching array length is not sufficient. Preserve existing no-resurrection and undo semantics. Document existing behavior for callers that omit the revision during a compatibility period.

Add managed signal subscriptions that the context can dispose. Document a small event matrix for sequence/range/seeds/result/mask/ROI changes, including whether reset emits the component events and their order. Retain the existing signals as compatibility adapters. Plugins should refresh from a coherent snapshot and tolerate repeated notifications; they should not have to infer hidden signal timing.

Keep the QWidget parent handle for dialogs, but stop promising supported access to `MainWindow` and `CanvasView` internals. Public overlay/interaction APIs must cover the examples. Add stable public numerical/export helpers only for shared operations the supplied examples actually require. Re-export/migrate those helpers before enforcing a forbidden-internal-import check.

### Documentation and authoring kit

Deliver a checked-in, offline kit containing:

* A concise `PLUGIN_CONTRACT.md`: supported imports/dependencies; folder/entry class; API version; metadata; array shapes/units/validity; mutation effects; events; UI-thread rules; cancellation/errors; persistence and lifecycle.
* A typed public reference/stub and an expanded HTML reference generated from that contract/source. They must describe the same API.
* Three minimal working templates: analysis/export, overlay/point picking, and loader/range control. Keep the advanced MTS workflow as a worked application, not the starter scaffold.
* A small fake context with nonzero reference, filtered points, dead tracks, empty state, and immutable arrays. Use it for pure plugin calculation tests.
* A real-host offscreen conformance runner accepting a plugin directory. It validates discovery, launch, state changes, data shapes, cleanup, repeat lifecycle, and supported imports.
* A short author task sheet listing the contract/template/test command and deliverables. The author must not need `main_window.py`, `ProjectState`, or the entire source tree.

Every complete documentation example should be executed in CI. Check empty active sets before argmax/reductions. Include negative examples for active-vs-valid confusion, global-vs-cut indexing and using stale snapshot masks. Validate compatibility/version metadata before launch and publish a deprecation policy.

For the user's intended weaker-model workflow, run a final acceptance exercise: give a fresh implementation session only the kit and a bounded plugin task, then run its output through the conformance runner. Assess forbidden internal imports, correctness and cleanup. Until this exercise passes, “only this file is needed” remains an unverified claim.

## 5. Define scientific calculations once

Persist an immutable LK error-kind/parameter record with the result. Standard photometric residuals and minimum eigenvalues require separate quality definitions; use maximum/mean residual upper bounds versus an explicitly named minimum-eigenvalue lower bound as appropriate. Make GUI labels and saved thresholds follow their kind. Do not infer a result's producing flag from settings the user edited later. For legacy results whose meaning is unknowable, expose that fact and avoid an unsupported filter.

Create a shared Qt-free mechanics-series result used by pressure, zones and MTS. Retain the centered affine solve and correct B-based current directions. Add:

* Finite/shape/time validation; reference/current joint validity; exact fit-used indices/counts.
* Geometry conditioning and residual diagnostics, with an explicit acceptance policy tied to spatial/noise resolution.
* Physical-map checks for the mechanics consumers, separate from generic affine fitting.
* Consistent invalid rows and reason codes; a defined no-points/degenerate-reference policy.
* One convention for undefined principal directions at equal stretches, including frame zero; axes/angles are modulo 180 degrees.
* Explicit names for stretch-minus-one strain, reference/area conventions, and derived quantities.

Use SVD directly if it improves the stretch calculation and conditioning diagnostics, but preserve the validated B-direction convention. Do not silently replace strain with Green–Lagrange/logarithmic strain. If those measures are added, give them distinct names and metadata.

Define RANSAC's final mask/model semantics. If the API promises a threshold on the final model, reclassify/refit with a bounded stopping policy and verify the postcondition. Give insufficient-redundancy/degenerate outcomes their own status. Explain adaptive confidence as an approximation or implement the actual sampling probability, without promising certainty the algorithm cannot establish.

Acceptance data: identity, translation, proper rotation, shear, known tension/compression, unequal/equal stretches, reflective/collapsed maps, nearly collinear points, planted outliers, nonfinite rows, zero surviving points, changing valid subsets, and reference away from global zero. Compare expected analytic values, with tolerances scaled to precision/conditioning.

## 6. Make MTS's experiment state and physics reproducible

Move workflow transitions into a small controller that owns both in-memory invalidation and artifact state. Each load, resume, channel/offset/crop/reference edit and result/mask change must declare what it invalidates. Clear caches/open plots even when `_loading` suppresses host signals. Key kinematics to its actual snapshot. Avoid computations merely to clear an invalid view.

Make reselecting the same reference intentional: preserve compatible tracking and persist it, or explicitly discard it in both layers. Atomic range changes from change set 2 remove the temporary “open the range fully” workaround. Verify tracker range against MTS reference state on resume; adopt a new external range or invalidate the plugin reference according to one documented policy.

Parse each channel into canonical seconds/mm/N and retain source units. Check displacement/force pairing and clamp sign conventions. Treat clock resets or ambiguous header fallback as explicit repair/selection cases, not silent mixing of runs. Report skipped image/sensor rows with original positions. Give timestamp duplicate handling a documented policy.

Persist sensor/log fingerprints, normalized parsing/unit schema, image identity, alignment offset, crop/search window, selected algorithm/version/parameters, force convention, material geometry and reference decisions. Resume verifies these before restoring dependent state. Support relocating a project through relative-to-root paths plus an explicit fallback for external files; do not accidentally resume from another surviving absolute root.

Replace reference functions' integer-or-zero outcome with a structured result containing success/failure, candidate index, diagnostics and algorithm parameters. An unmet preforce threshold or failed nonlinear fit should not commit frame zero. Check solver status, residual quality and extrapolated minima. Test the spring-hinge model on known generated parameters and distinguish an elastic-energy reference from a zero-force reference.

Choose and expose the physical conventions during implementation, preserving legacy behavior under an explicit legacy label where needed:

* Separate sensor tare from a real mechanical preload and from a display-only force change. Absolute nominal stress, stress change, and Cauchy stress change must have separately correct definitions if offered.
* State which configuration the measured width/thickness belongs to.
* Use an explicitly defined loading-axis stretch, or support only validated uniaxial tension and name that approximation. Incompressible equal-transverse prediction is a model comparison, not an identity every specimen must satisfy.
* Put sensor and image curves on the same shifted timeline and carry coverage into all derived outputs. Reject a reference without real coverage or require an explicit estimated-data mode.

Acceptance cases include every upstream edit while plots are open/hidden; experiment A to B and resume B; repeated reference; external range changes; channel/unit equivalents; unsupported mixed units; changed sensor/log with unchanged image pixels; no overlap/partial overlap; +/− offset plot coordinates; no threshold crossing; failed fit; preload stress arithmetic; compression/axis mismatch; moved project.

## 7. Align visualization, selection and exported artifacts

Attach zones to sequence/reference identity. New zones are drawn on the reference frame, or an explicitly chosen mapping converts them there. Sequence/reference changes clear or deliberately transform zones; there is no implicit reuse. Propagate zone add/remove/color changes and shared mechanics validity into all tables/plots/gauges. Clear a failed image view with a useful error instead of retaining an old image under new labels.

Compute a single immutable analysis result for both plot and CSV. Include mandatory global frame ID, validity/coverage, fit count and provenance sufficient to interpret the selected numerical columns. Record point selection and fit policy: MTS's per-frame kinematics can use partially surviving tracks while coordinate-only export currently keeps only fully surviving tracks. Either export statuses for the analysis selection or explicitly document/store both selections so someone can reproduce the plotted result.

Use versioned/staged output generations: write and validate the complete artifact set, then atomically publish one authoritative manifest/pointer. A failed write leaves the previous generation current. Invalidation changes a durable revision/validity state; inability to delete an old file must not make it current again. File cleanup can occur after the commit without participating in scientific validity.

Give core coordinate exports their own range/point metadata, retaining an explicit legacy `coords.npy`/`sequence.txt` path if existing consumers need it. Never silently change CSV meanings under the same header; version renamed/added columns or provide a compatibility export.

Pressure alignment keeps its existing endpoint model, but stores input timestamps, source fingerprint, anchors, scale factor, reference and pressure-zero flag. Label it as an assumed time mapping; support acquisition timestamps when available without guessing a new default.

Acceptance: table/plot/CSV numerical equality; no directions for undefined axes; zone drawing with nonzero reference and translation; failed fit row retained with reason; output failure at each artifact/manifest boundary; failed cleanup with durable invalidation; two named exports with different ranges; reproducible pressure mapping after files are copied or timestamps change.

## 8. Package the SDK and close maintenance gaps

Ship the contract, typed reference, templates, host dependency manifest, and test command with both source and frozen releases. A client must be able to author/test a simple plugin without opening the original repository. Validate the packaged artifact's files and imports; a source-only unit test cannot prove this.

Keep installed examples and user plugins in separate locations with explicit discovery precedence. Migrate existing custom directories safely and ensure installer upgrades never overwrite edits without preserving them. Validate custom dependencies against the host bundle; document that frozen runtime dependency installation differs from `uv sync` in a checkout. Do not add automatic package installation as a hidden plugin side effect.

For macOS, stage everything first, sign the final bundle, then verify before archiving/notarization. Prefer external user-plugin storage so editing plugins does not mutate the distributed app bundle. Add a build lock or unique temporary scratch directory.

Expand the gate to discover test modules and parse root entry/build scripts. Add targeted type checks for public SDK declarations, contract examples, dependency-boundary checks and packaging smoke tests. Reconcile supported Python/runtime/build versions with actual CI results. Isolate and clean up test fixtures.

Profile representative large experiments before optimizing. The first likely wins are frame-oriented access, cached exportability/metrics per revision, debounced previews, one calculation per change, and reduced duplicate writes. Move expensive work off the UI thread only with the revision/cancellation contract already enforced. Extract duplicated calculation/transition code as part of these fixes, while avoiding unrelated architectural churn.

## Compatibility and release acceptance

Keep established SDK methods working through adapters while migrating all bundled plugins onto the new supported surface. Deprecate escape hatches and undocumented internal imports with a stated version policy. Increase file schema versions only when meanings/metadata require it, preserve supported legacy readers/migrations, and surface ambiguity instead of rewriting it into invented certainty.

Completion requires:

1. All existing regressions plus the targeted new failure cases pass on supported platforms.
2. Every R01–R22 finding has a test/contract decision and an implemented resolution; S01–S04 have their specified documentation, validation or measurement follow-up.
3. A complete synthetic experiment survives load → reference → track → cleanup → export → close → resume, including nonzero references and failure/cancellation paths, without data disagreement.
4. Mathematical fixtures verify the implemented measures and their assumptions; plots and exports consume the same validated series.
5. A plugin authored from only the standalone kit works in the real host, cleans up on reload/shutdown, and uses no unsupported internal APIs.
6. The packaged release includes that same kit and preserves user plugins on update.

No broad rewrite or single large “cleanup” change is necessary. Each change should state the concrete behavior corrected, the compatibility impact, and the test that demonstrates it.
