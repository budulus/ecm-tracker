//! Plugin discovery + loading — the Rust port of `app/plugins/manager.py` (the headless,
//! Qt-free half: scanning, importing, class resolution, instantiation).
//!
//! [`discover`] scans a plugins directory, makes it importable, imports each package, and finds
//! its plugin class (an explicit module-level `PLUGIN`, else the first [`TrackerPlugin`] subclass
//! defined in the package) — recording each result (or its import error) as a [`PluginRecord`].
//! [`launch`] instantiates a record's class with a fresh `PluginContext(snapshot)` and calls
//! `launch()`. Import/resolution errors are isolated per plugin: one broken package never breaks
//! discovery of the rest.
//!
//! The GUI (slice 3e) owns the returned records and builds the `ContextSnapshot`; this module is
//! deliberately stateless (re-discovery / instance caching / window lifecycle stay in the GUI).

use crate::context::{ContextSnapshot, PluginContext};
use crate::overlay::{DrawCommand, OverlayPainter};
use crate::panel::{Control, ControlValue, PanelBuilder};
use crate::sdk::{register_sdk, SDK_MODULE};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::path::Path;

/// One discovered plugin — the headless subset of `manager.py`'s `PluginRecord`.
///
/// Either `cls` is `Some` (loaded, `error` is `None`) or `error` is `Some` (import or
/// class-resolution failure, `cls` is `None`).
pub struct PluginRecord {
    /// Stable slug — the plugin's folder name.
    pub id: String,
    /// Display name (`cls.NAME` if non-empty, else `id`).
    pub name: String,
    /// One-line description (`cls.DESCRIPTION`, else empty).
    pub description: String,
    /// The plugin class object (a `TrackerPlugin` subclass), or `None` on load failure.
    pub cls: Option<Py<PyAny>>,
    /// Per-plugin load error (import error text or "no subclass found"), else `None`.
    pub error: Option<String>,
}

impl PluginRecord {
    fn failed(id: &str, error: String) -> Self {
        PluginRecord {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            cls: None,
            error: Some(error),
        }
    }
}

/// Discover plugins in `dir`: make `dir` importable, then import each candidate package and
/// resolve its plugin class. Returns one [`PluginRecord`] per package, in sorted-by-id order.
/// Import/resolution failures are captured in the record's `error` (never propagated), so a single
/// broken plugin doesn't abort discovery.
///
/// Ensures the `ecm_host` SDK is registered first (idempotent) so plugin code can `import ecm_host`
/// and so the `TrackerPlugin` base is available for the subclass check.
pub fn discover(py: Python<'_>, dir: &Path) -> PyResult<Vec<PluginRecord>> {
    register_sdk(py)?;
    let base = py.import(SDK_MODULE)?.getattr("TrackerPlugin")?;

    // Make `dir` importable as a top-level package root. Append (not insert-at-0) so a plugin
    // folder can never shadow a stdlib/site-packages module, and guard against stacking duplicate
    // `sys.path` entries on repeated discovery (the spec flags sys.path pollution).
    let dir_str = dir.to_string_lossy().into_owned();
    let sys_path = py.import("sys")?.getattr("path")?;
    let present = sys_path
        .call_method1("__contains__", (dir_str.as_str(),))?
        .extract::<bool>()?;
    if !present {
        sys_path.call_method1("append", (dir_str.as_str(),))?;
    }

    Ok(package_dirs(dir)
        .iter()
        .map(|name| load_one(py, &base, name))
        .collect())
}

/// Instantiate a discovered plugin: `cls(PluginContext(snapshot))`. Returns the plugin instance.
/// Errors if the record failed to load. The caller keeps the instance to call `launch()` and
/// (later) `overlay()` on it.
pub fn instantiate(
    py: Python<'_>,
    record: &PluginRecord,
    snapshot: ContextSnapshot,
) -> PyResult<Py<PyAny>> {
    let cls = record.cls.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "plugin '{}' is not loaded: {}",
            record.id,
            record.error.as_deref().unwrap_or("unknown error"),
        ))
    })?;
    let ctx = Py::new(py, PluginContext::new(snapshot))?;
    cls.call1(py, (ctx,))
}

/// Instantiate a plugin and call its `launch()`, returning whatever `launch()` returns. Errors if
/// the record failed to load.
pub fn launch(
    py: Python<'_>,
    record: &PluginRecord,
    snapshot: ContextSnapshot,
) -> PyResult<Py<PyAny>> {
    instantiate(py, record, snapshot)?.call_method0(py, "launch")
}

/// Whether a launched plugin instance provides an `overlay()` method (i.e. draws on the canvas).
pub fn has_overlay(py: Python<'_>, instance: &Py<PyAny>) -> bool {
    instance.bind(py).hasattr("overlay").unwrap_or(false)
}

/// Re-invoke a plugin instance's `overlay(self, painter)` to collect its current draw-commands.
/// Refreshes the instance's `ctx` to `snapshot` first so the overlay reflects current state, then
/// hands it a fresh [`OverlayPainter`] and returns the accumulated commands. Empty if the instance
/// has no `overlay` method.
pub fn overlay_commands(
    py: Python<'_>,
    instance: &Py<PyAny>,
    snapshot: ContextSnapshot,
) -> PyResult<Vec<DrawCommand>> {
    let bound = instance.bind(py);
    if !bound.hasattr("overlay")? {
        return Ok(Vec::new());
    }
    let ctx = Py::new(py, PluginContext::new(snapshot))?;
    bound.setattr("ctx", ctx)?;
    let painter = Bound::new(py, OverlayPainter::default())?;
    bound.call_method1("overlay", (&painter,))?;
    let commands = painter.borrow().commands.clone();
    Ok(commands)
}

/// Take any keep-mask a plugin recorded via `ctx.apply_keep_mask(...)` during the last call
/// (the `pending_keep` on its `PluginContext`). The host applies it through its undoable mask path
/// after the plugin method returns (slice 3g-b). `None` if the instance has no `ctx`, the `ctx`
/// isn't a `PluginContext`, or nothing was recorded.
pub fn take_keep_mask(py: Python<'_>, instance: &Py<PyAny>) -> Option<Vec<bool>> {
    let ctx_obj = instance.bind(py).getattr("ctx").ok()?;
    let ctx = ctx_obj.cast::<PluginContext>().ok()?;
    ctx.borrow_mut().take_pending_keep()
}

/// Take any settings a plugin recorded via `ctx.save_settings(...)` during the last call (the
/// `pending_settings` JSON string on its `PluginContext`). The GUI persists it through
/// `core::settings` after the plugin method returns (slice 3g-d). `None` if the instance has no
/// `ctx`, the `ctx` isn't a `PluginContext`, or nothing was recorded.
pub fn take_settings(py: Python<'_>, instance: &Py<PyAny>) -> Option<String> {
    let ctx_obj = instance.bind(py).getattr("ctx").ok()?;
    let ctx = ctx_obj.cast::<PluginContext>().ok()?;
    ctx.borrow_mut().take_pending_settings()
}

/// Whether a launched plugin instance declares a control panel (i.e. defines a `panel()` method).
pub fn has_panel(py: Python<'_>, instance: &Py<PyAny>) -> bool {
    instance.bind(py).hasattr("panel").unwrap_or(false)
}

/// Collect a plugin's declared controls by invoking its `panel(self, builder)` (slice 3g-c).
/// Refreshes the instance's `ctx` to `snapshot` first (so initial control values can reflect
/// current state), hands it a fresh [`PanelBuilder`], and returns the accumulated [`Control`]s.
/// Empty if the instance has no `panel` method.
pub fn panel_controls(
    py: Python<'_>,
    instance: &Py<PyAny>,
    snapshot: ContextSnapshot,
) -> PyResult<Vec<Control>> {
    let bound = instance.bind(py);
    if !bound.hasattr("panel")? {
        return Ok(Vec::new());
    }
    let ctx = Py::new(py, PluginContext::new(snapshot))?;
    bound.setattr("ctx", ctx)?;
    let builder = Bound::new(py, PanelBuilder::default())?;
    bound.call_method1("panel", (&builder,))?;
    let controls = builder.borrow().controls.clone();
    Ok(controls)
}

/// Deliver a control change to a plugin instance by calling `on_control(self, key, value)` (slice
/// 3g-c). Refreshes the instance's `ctx` to `snapshot` first (so the handler can read current state
/// and call `ctx.apply_keep_mask`), then calls the hook with `value` as a float / bool / `None`. The
/// base hook is a no-op, so this is safe for any instance. An error raised by the hook propagates.
pub fn dispatch_control(
    py: Python<'_>,
    instance: &Py<PyAny>,
    key: &str,
    value: ControlValue,
    snapshot: ContextSnapshot,
) -> PyResult<()> {
    let bound = instance.bind(py);
    let ctx = Py::new(py, PluginContext::new(snapshot))?;
    bound.setattr("ctx", ctx)?;
    match value {
        ControlValue::Float(f) => bound.call_method1("on_control", (key, f)).map(|_| ()),
        ControlValue::Bool(b) => bound.call_method1("on_control", (key, b)).map(|_| ()),
        ControlValue::Click => bound.call_method1("on_control", (key, py.None())).map(|_| ()),
    }
}

/// Deliver a state-change event to a plugin instance by calling its `on_<event>` hook (the port of
/// connecting to Qt's `PluginSignals`). Refreshes the instance's `ctx` to `snapshot` first so the
/// handler sees current state via `self.ctx`, then calls `on_<event>`. `frame_index` is passed only
/// for `frame_changed`. The hooks default to no-ops on `TrackerPlugin`, so this is safe for any
/// instance. An error raised by the hook propagates to the caller.
pub fn dispatch_event(
    py: Python<'_>,
    instance: &Py<PyAny>,
    event: &str,
    frame_index: Option<i64>,
    snapshot: ContextSnapshot,
) -> PyResult<()> {
    let bound = instance.bind(py);
    let ctx = Py::new(py, PluginContext::new(snapshot))?;
    bound.setattr("ctx", ctx)?;
    let method = format!("on_{event}");
    match frame_index {
        Some(i) => bound.call_method1(&method, (i,)).map(|_| ()),
        None => bound.call_method0(&method).map(|_| ()),
    }
}

/// Import one package by name and resolve its plugin class, isolating any failure into the record.
fn load_one(py: Python<'_>, base: &Bound<'_, PyAny>, pkg_name: &str) -> PluginRecord {
    let module = match py.import(pkg_name) {
        Ok(m) => m,
        Err(e) => return PluginRecord::failed(pkg_name, format!("{e}")),
    };
    match find_plugin_class(&module, base, pkg_name) {
        Ok(Some(cls)) => {
            let (name, description) = read_metadata(py, &cls, pkg_name);
            PluginRecord {
                id: pkg_name.to_string(),
                name,
                description,
                cls: Some(cls),
                error: None,
            }
        }
        Ok(None) => PluginRecord::failed(
            pkg_name,
            format!("no TrackerPlugin subclass found in '{pkg_name}'"),
        ),
        Err(e) => PluginRecord::failed(pkg_name, format!("{e}")),
    }
}

/// Resolve the plugin class inside an imported module: an explicit module-level `PLUGIN` (a
/// `TrackerPlugin` subclass), else the first `TrackerPlugin` subclass *defined in this package*
/// (excluding the base itself). Mirrors `manager.py:_find_plugin_class`.
fn find_plugin_class<'py>(
    module: &Bound<'py, PyModule>,
    base: &Bound<'py, PyAny>,
    pkg_name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    // 1. Explicit `PLUGIN = YourClass`.
    if let Some(explicit) = module.getattr_opt("PLUGIN")? {
        let is_plugin = match explicit.cast::<PyType>() {
            Ok(t) => t.is_subclass(base)? && !explicit.is(base),
            Err(_) => false,
        };
        if is_plugin {
            return Ok(Some(explicit.unbind()));
        }
    }
    // 2. Fallback: first TrackerPlugin subclass defined in this package.
    let members = module.dict();
    for (_name, value) in members.iter() {
        let is_plugin = match value.cast::<PyType>() {
            Ok(t) => t.is_subclass(base)? && !value.is(base),
            Err(_) => false,
        };
        if !is_plugin {
            continue;
        }
        // Only classes defined in this package, not ones imported into it.
        let defined_in: String = value.getattr("__module__")?.extract()?;
        if defined_in.starts_with(pkg_name) {
            return Ok(Some(value.unbind()));
        }
    }
    Ok(None)
}

/// Read `NAME` (falling back to `id`) and `DESCRIPTION` (falling back to empty) off the class.
fn read_metadata(py: Python<'_>, cls: &Py<PyAny>, id: &str) -> (String, String) {
    let cls = cls.bind(py);
    let name = cls
        .getattr_opt("NAME")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| id.to_string());
    let description = cls
        .getattr_opt("DESCRIPTION")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default();
    (name, description)
}

/// Candidate plugin package names in `dir`, sorted: subdirectories that aren't hidden/private
/// (don't start with `.` or `_`) and contain an `__init__.py`. Mirrors `manager.py`'s scan.
fn package_dirs(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with('.') || name.starts_with('_') {
            continue;
        }
        if !path.join("__init__.py").exists() {
            continue;
        }
        names.push(name);
    }
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Write `dir/<name>/__init__.py` with `body` (creates the package folder).
    fn write_pkg(root: &Path, name: &str, body: &str) {
        let pkg = root.join(name);
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("__init__.py"), body).unwrap();
    }

    /// A fresh, empty temp dir for this test's plugin tree (cleared if a prior run left it).
    fn fresh_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ecm_host_d3_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discovers_loads_and_launches_plugins() {
        let _g = crate::interp_test_lock();
        let dir = fresh_temp_dir();

        // Good: explicit PLUGIN, custom NAME/DESCRIPTION, reads the context in launch().
        write_pkg(
            &dir,
            "good_plugin_d3",
            "import ecm_host\n\
             class GoodPlugin(ecm_host.TrackerPlugin):\n\
             \x20   NAME = \"Good\"\n\
             \x20   DESCRIPTION = \"works\"\n\
             \x20   def launch(self):\n\
             \x20       return self.ctx.point_count\n\
             PLUGIN = GoodPlugin\n",
        );
        // Lone subclass, no PLUGIN attr (fallback path); inherits the base NAME.
        write_pkg(
            &dir,
            "lone_plugin_d3",
            "import ecm_host\n\
             class LonePlugin(ecm_host.TrackerPlugin):\n\
             \x20   def launch(self):\n\
             \x20       return 1\n",
        );
        // Imports fine but defines no TrackerPlugin subclass.
        write_pkg(&dir, "empty_plugin_d3", "MARKER = 123\n");
        // Raises at import — error must be isolated, not fatal.
        write_pkg(
            &dir,
            "broken_plugin_d3",
            "raise RuntimeError(\"boom at import\")\n",
        );
        // Skip cases: must NOT appear in the records.
        write_pkg(&dir, "_private_d3", "x = 1\n"); // leading underscore
        write_pkg(&dir, ".hidden_d3", "x = 1\n"); // leading dot
        fs::create_dir_all(dir.join("notapkg_d3")).unwrap(); // no __init__.py

        Python::attach(|py| {
            let records = discover(py, &dir).unwrap();

            // Only the four real packages, in sorted order; skip-cases excluded.
            let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(
                ids,
                vec![
                    "broken_plugin_d3",
                    "empty_plugin_d3",
                    "good_plugin_d3",
                    "lone_plugin_d3",
                ]
            );

            let by_id = |id: &str| records.iter().find(|r| r.id == id).unwrap();

            let good = by_id("good_plugin_d3");
            assert!(
                good.cls.is_some() && good.error.is_none(),
                "good plugin failed to load: {:?}",
                good.error
            );
            assert_eq!(good.name, "Good");
            assert_eq!(good.description, "works");

            let lone = by_id("lone_plugin_d3");
            assert!(
                lone.cls.is_some() && lone.error.is_none(),
                "lone plugin failed to load: {:?}",
                lone.error
            );
            assert_eq!(lone.name, "Unnamed Plugin"); // inherited base default

            let empty = by_id("empty_plugin_d3");
            assert!(empty.cls.is_none());
            assert!(empty.error.as_ref().unwrap().contains("no TrackerPlugin subclass"));

            let broken = by_id("broken_plugin_d3");
            assert!(broken.cls.is_none());
            assert!(broken.error.as_ref().unwrap().contains("boom"));

            // Launch the good plugin with a snapshot — launch() returns ctx.point_count.
            let snap = ContextSnapshot {
                point_count: 42,
                ..Default::default()
            };
            let ret = launch(py, good, snap).unwrap();
            let n: usize = ret.bind(py).extract().unwrap();
            assert_eq!(n, 42);

            // Launching a failed record errors instead of panicking.
            assert!(launch(py, broken, ContextSnapshot::default()).is_err());

            // Hermetic: undo this test's interpreter-global pollution before releasing the lock —
            // the temp dir on `sys.path` and the imported plugin packages would otherwise leak into
            // other tests sharing the one embedded interpreter (e.g. break rust-numpy's lazy
            // array-API init in `context::tests`). See the slice-3d note in PROGRESS.md.
            let sys = py.import("sys").unwrap();
            sys.getattr("path")
                .unwrap()
                .call_method1("remove", (dir.to_string_lossy().as_ref(),))
                .ok();
            let modules = sys.getattr("modules").unwrap();
            for pkg in ["good_plugin_d3", "lone_plugin_d3", "empty_plugin_d3", "broken_plugin_d3"] {
                modules.call_method1("pop", (pkg, py.None())).ok();
            }
        });

        let _ = fs::remove_dir_all(&dir);
    }

    /// An overlay-providing plugin: `overlay(self, painter)` draws onto the host's `OverlayPainter`,
    /// and `overlay_commands` collects the resulting draw-commands (after refreshing the ctx).
    #[test]
    fn overlay_commands_collects_draw_commands() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            register_sdk(py).unwrap();
            let src = cr#"
import ecm_host

class P(ecm_host.TrackerPlugin):
    def launch(self):
        return None

    def overlay(self, painter):
        painter.circle((1.0, 2.0), radius=3.0, fill=(0, 200, 255, 255))
        painter.polyline([(0.0, 0.0), (10.0, 10.0)], color=(255, 0, 0), width=2.0)
"#;
            let globals = pyo3::types::PyDict::new(py);
            py.run(src, Some(&globals), None).unwrap();
            let cls = globals.get_item("P").unwrap().unwrap();
            let ctx = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let instance = cls.call1((ctx,)).unwrap().unbind();

            assert!(has_overlay(py, &instance));
            let cmds = overlay_commands(py, &instance, ContextSnapshot::default()).unwrap();
            assert_eq!(cmds.len(), 2);
            match &cmds[0] {
                DrawCommand::Circle {
                    center,
                    radius,
                    fill,
                    ..
                } => {
                    assert_eq!(*center, (1.0, 2.0));
                    assert_eq!(*radius, 3.0);
                    assert!(fill.is_some());
                }
                other => panic!("expected Circle, got {other:?}"),
            }
            match &cmds[1] {
                DrawCommand::Polyline { points, stroke } => {
                    assert_eq!(points.len(), 2);
                    assert_eq!(stroke.width, 2.0);
                }
                other => panic!("expected Polyline, got {other:?}"),
            }
        });
    }

    /// A plugin's `on_*` hooks receive dispatched events, with `self.ctx` refreshed to the event's
    /// snapshot. A non-overridden hook (base no-op) must not error.
    #[test]
    fn dispatch_event_delivers_to_plugin_hooks() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            register_sdk(py).unwrap();
            let src = cr#"
import ecm_host

class P(ecm_host.TrackerPlugin):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.frames = []
        self.masks = 0
    def launch(self):
        return None
    def on_frame_changed(self, global_index):
        self.frames.append(global_index)
    def on_mask_changed(self):
        self.masks += 1
"#;
            let globals = pyo3::types::PyDict::new(py);
            py.run(src, Some(&globals), None).unwrap();
            let cls = globals.get_item("P").unwrap().unwrap();
            let ctx = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let instance = cls.call1((ctx,)).unwrap().unbind();

            dispatch_event(
                py,
                &instance,
                "frame_changed",
                Some(7),
                ContextSnapshot { current_index: 7, ..Default::default() },
            )
            .unwrap();
            dispatch_event(py, &instance, "mask_changed", None, ContextSnapshot::default()).unwrap();
            // A hook the plugin did NOT override (base no-op) must be harmless.
            dispatch_event(py, &instance, "roi_changed", None, ContextSnapshot::default()).unwrap();

            let b = instance.bind(py);
            let frames: Vec<i64> = b.getattr("frames").unwrap().extract().unwrap();
            assert_eq!(frames, vec![7]);
            let masks: i64 = b.getattr("masks").unwrap().extract().unwrap();
            assert_eq!(masks, 1);
            // ctx is refreshed before each dispatch; the last dispatch's snapshot had current_index 0.
            let cur: usize = b
                .getattr("ctx").unwrap()
                .getattr("current_index").unwrap()
                .extract().unwrap();
            assert_eq!(cur, 0);
        });
    }

    /// A panel-declaring plugin: `panel()` builds controls onto the host's `PanelBuilder`
    /// (collected by `panel_controls`), and `dispatch_control` delivers changes to `on_control` —
    /// where a slider/checkbox are recorded and an "apply" button records a keep-mask the host then
    /// takes via `take_keep_mask`.
    #[test]
    fn panel_controls_collects_and_dispatch_control_delivers() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            crate::ensure_embedded_site(py).unwrap(); // apply_keep_mask uses numpy
            register_sdk(py).unwrap();
            let src = cr#"
import ecm_host

class P(ecm_host.TrackerPlugin):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.last = None
    def launch(self):
        return None
    def panel(self, ui):
        ui.label("hint")
        ui.slider("gain", "Gain", 2.0, 0.0, 10.0)
        ui.checkbox("flag", "Flag", True)
        ui.button("apply", "Apply")
    def on_control(self, key, value):
        self.last = (key, value)
        if key == "apply":
            self.ctx.apply_keep_mask([True, False, True, False])
"#;
            let globals = pyo3::types::PyDict::new(py);
            py.run(src, Some(&globals), None).unwrap();
            let cls = globals.get_item("P").unwrap().unwrap();
            let ctx = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let instance = cls.call1((ctx,)).unwrap().unbind();

            // A snapshot with a result, so apply_keep_mask records (P=4, all active).
            let snap = || ContextSnapshot {
                has_result: true,
                point_count: 4,
                n_active: 4,
                active_mask: Some(vec![true; 4]),
                ..Default::default()
            };

            // panel() is detected and its controls collected in declaration order.
            assert!(has_panel(py, &instance));
            let controls = panel_controls(py, &instance, snap()).unwrap();
            assert_eq!(controls.len(), 4);
            assert!(matches!(&controls[0], Control::Label { text } if text == "hint"));
            match &controls[1] {
                Control::Slider { key, value, min, max, .. } => {
                    assert_eq!(key, "gain");
                    assert_eq!((*value, *min, *max), (2.0, 0.0, 10.0));
                }
                other => panic!("expected Slider, got {other:?}"),
            }
            assert!(matches!(&controls[2], Control::Checkbox { key, value, .. } if key == "flag" && *value));
            assert!(matches!(&controls[3], Control::Button { key, .. } if key == "apply"));

            // A slider change is delivered to on_control as a float.
            dispatch_control(py, &instance, "gain", ControlValue::Float(7.5), snap()).unwrap();
            let last: (String, f64) = instance.bind(py).getattr("last").unwrap().extract().unwrap();
            assert_eq!(last, ("gain".to_string(), 7.5));

            // The "apply" button (Click → None) records a keep-mask the host takes.
            dispatch_control(py, &instance, "apply", ControlValue::Click, snap()).unwrap();
            assert_eq!(take_keep_mask(py, &instance), Some(vec![true, false, true, false]));
            assert_eq!(take_keep_mask(py, &instance), None); // taken once, then cleared
        });
    }
}
