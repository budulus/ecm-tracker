//! The embedded-Python plugin SDK module, `ecm_host`.
//!
//! Plugins `import ecm_host` to reach the SDK: the Rust `PluginContext` (`#[pyclass]`, the data
//! façade) and a Python-defined `TrackerPlugin` base class (the lifecycle base they subclass).
//! Together these port `app/plugins/api.py` to the embedded interpreter.
//!
//! The module is created at runtime and injected into `sys.modules` (idempotent) rather than via
//! `append_to_inittab!` — injection keeps pyo3's `auto-initialize` feature (the inittab route
//! requires initializing the interpreter manually, before any `Python::attach`).

use crate::context::PluginContext;
use crate::overlay::OverlayPainter;
use crate::panel::PanelBuilder;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::ffi::CStr;

/// Module name a plugin imports the SDK from (`from ecm_host import TrackerPlugin`).
pub const SDK_MODULE: &str = "ecm_host";

/// Python source for the `TrackerPlugin` base — the Qt-free port of `app/plugins/api.py`'s base.
/// A plugin subclasses this and implements `launch`; the host instantiates it once with a
/// `PluginContext`, then calls `launch()`. Declared into the `ecm_host` module's namespace.
const TRACKER_PLUGIN_SRC: &CStr = cr#"
class TrackerPlugin:
    """Base class for an ECM Tracker plugin. Subclass it and implement launch().

    The host instantiates the class once with a PluginContext (as self.ctx), then calls
    launch() to run the plugin. Declare the plugin by assigning PLUGIN = YourClass in the
    package __init__.py (a lone TrackerPlugin subclass is auto-detected as a fallback).

    Optionally define overlay(self, painter) to draw on the canvas: the host re-invokes it
    when state changes, handing in an ecm_host.OverlayPainter (draw in image coordinates).

    Optionally define panel(self, builder) to declare a control panel: call builder.slider /
    .checkbox / .button / .label (an ecm_host.PanelBuilder) to declare controls. The host renders
    them as a window and reports changes back via on_control(self, key, value).

    Optionally define on_sequence_changed / on_frame_changed(global_index) / on_result_changed /
    on_mask_changed / on_roi_changed to react to state changes (the host calls them with self.ctx
    refreshed to current state).
    """

    NAME = "Unnamed Plugin"
    DESCRIPTION = ""

    def __init__(self, ctx):
        self.ctx = ctx

    def launch(self):
        raise NotImplementedError

    def on_unload(self):
        pass

    # Optional reactive hooks — the host calls these when app state changes (the port of the Qt
    # PluginSignals). Override the ones you care about; they default to no-ops. Reading self.ctx
    # inside a hook reflects the NEW state (the host refreshes it before each call).
    def on_sequence_changed(self):
        pass

    def on_frame_changed(self, global_index):
        pass

    def on_result_changed(self):
        pass

    def on_mask_changed(self):
        pass

    def on_roi_changed(self):
        pass

    # Optional control-panel callback — the host calls this when a control declared in panel()
    # changes: value is a float (slider), a bool (checkbox), or None (button click). Defaults to a
    # no-op. self.ctx is refreshed to current state before the call (so apply_keep_mask works here).
    def on_control(self, key, value):
        pass
"#;

/// Ensure the `ecm_host` SDK module exists in `sys.modules` (idempotent) so plugin code can
/// `import ecm_host`. Exposes the Rust `PluginContext` class and the Python `TrackerPlugin` base.
pub fn register_sdk(py: Python<'_>) -> PyResult<()> {
    let sys_modules = py.import("sys")?.getattr("modules")?;
    if sys_modules.contains(SDK_MODULE)? {
        return Ok(());
    }
    let module = PyModule::new(py, SDK_MODULE)?;
    module.add_class::<PluginContext>()?;
    module.add_class::<OverlayPainter>()?;
    module.add_class::<PanelBuilder>()?;
    // Define TrackerPlugin into the module's namespace (globals = the module dict).
    py.run(TRACKER_PLUGIN_SRC, Some(&module.dict()), None)?;
    sys_modules.set_item(SDK_MODULE, &module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextSnapshot;
    use pyo3::types::PyDict;

    /// After registration, `import ecm_host` succeeds and exposes both SDK symbols; a second
    /// registration is a no-op.
    #[test]
    fn sdk_module_is_importable() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            register_sdk(py).unwrap();
            register_sdk(py).unwrap(); // idempotent — already in sys.modules
            let m = py.import(SDK_MODULE).unwrap();
            assert!(m.getattr("TrackerPlugin").is_ok());
            assert!(m.getattr("PluginContext").is_ok());
        });
    }

    /// The end-to-end plugin contract: a plugin imports the SDK, subclasses `TrackerPlugin`,
    /// receives a `PluginContext`, and reads tracked data from it in `launch()`.
    #[test]
    fn sdk_subclass_receives_and_reads_context() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            register_sdk(py).unwrap();
            // A minimal plugin: launch() returns a value read off the context.
            let plugin_src = cr#"
import ecm_host

class P(ecm_host.TrackerPlugin):
    NAME = "Test Plugin"
    DESCRIPTION = "reads the context in launch"

    def launch(self):
        return self.ctx.point_count
"#;
            let globals = PyDict::new(py);
            py.run(plugin_src, Some(&globals), None).unwrap();
            let cls = globals.get_item("P").unwrap().unwrap();

            // Host side: build a context and instantiate the plugin class with it.
            let snap = ContextSnapshot { point_count: 267, ..Default::default() };
            let ctx = Py::new(py, PluginContext::new(snap)).unwrap();
            let instance = cls.call1((ctx,)).unwrap();

            let n: usize = instance.call_method0("launch").unwrap().extract().unwrap();
            assert_eq!(n, 267);

            // Class metadata used later by discovery is readable.
            let name: String = cls.getattr("NAME").unwrap().extract().unwrap();
            let desc: String = cls.getattr("DESCRIPTION").unwrap().extract().unwrap();
            assert_eq!(name, "Test Plugin");
            assert_eq!(desc, "reads the context in launch");
        });
    }
}
