//! ECM Tracker plugin host (Rust port target).
//!
//! Embeds a bundled CPython (via pyo3) and will expose the `PluginContext`
//! façade to Python plugins — the evolution of `app/plugins/api.py`. Phase 0
//! only proves the embedded interpreter boots and can `import numpy`; the API
//! surface is built out in Phase 3.

use pyo3::prelude::*;

pub mod context;
pub mod host;
pub mod overlay;
pub mod sdk;
pub use context::{ContextSnapshot, PluginContext};
pub use host::{
    discover, dispatch_event, has_overlay, instantiate, launch, overlay_commands, take_keep_mask,
    PluginRecord,
};
pub use overlay::{DrawCommand, OverlayPainter, Rgba, Stroke};
pub use sdk::{register_sdk, SDK_MODULE};

/// Boot the embedded interpreter, make the bundled site-packages importable,
/// and return numpy's version string. Proves the in-process CPython works.
///
/// Phase 0 spike: the site-packages dir comes from the `ECM_PY_SITE` env var,
/// and the interpreter home / DLL come from `PYTHONHOME` + `PATH`. The shipped
/// app will instead derive all paths from its install location and apply them
/// via PyO3's interpreter config before initialization.
pub fn import_numpy_version() -> PyResult<String> {
    Python::attach(|py| {
        if let Ok(site) = std::env::var("ECM_PY_SITE") {
            let sys = py.import("sys")?;
            sys.getattr("path")?.call_method1("insert", (0, site))?;
        }
        let np = py.import("numpy")?;
        np.getattr("__version__")?.extract::<String>()
    })
}

/// Test-only: serialize access to the shared embedded interpreter.
///
/// All pyhost tests run against one process-global CPython, and CPython releases the GIL at its
/// thread-switch interval *while executing Python bytecode* (inside `py.import` / `py.run`), so two
/// `Python::attach` tests on different harness threads can interleave and race on shared global
/// state (`sys.modules` / `sys.path` / the `ecm_host` module). Hold this guard across the whole
/// `Python::attach` block so the interpreter-touching tests run serially.
#[cfg(test)]
pub(crate) fn interp_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Make the bundled interpreter's site-packages importable (idempotent).
///
/// The embedded CPython has no numpy/scipy on its default `sys.path`; during development the path
/// comes from the `ECM_PY_SITE` env var (set by `cargoenv.ps1`), and this is a no-op if it's unset.
/// The shipped app will instead derive the path from its install location (Phase 5 packaging). Call
/// it before importing numpy — directly or via rust-numpy's lazy array-API init — since the
/// embedded interpreter is shared and nothing else guarantees the path is present (the GUI calls it
/// before `discover`/`launch`; tests call it before array ops, as order isn't guaranteed).
pub fn ensure_embedded_site(py: Python<'_>) -> PyResult<()> {
    let Ok(site) = std::env::var("ECM_PY_SITE") else {
        return Ok(());
    };
    let path = py.import("sys")?.getattr("path")?;
    if !path
        .call_method1("__contains__", (site.as_str(),))?
        .extract::<bool>()?
    {
        path.call_method1("insert", (0, site.as_str()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_numpy_and_runs() {
        let _g = crate::interp_test_lock();
        let ver = import_numpy_version().expect("embedded interpreter should import numpy");
        println!("embedded numpy version: {ver}");
        assert!(!ver.is_empty(), "numpy __version__ was empty");

        // Prove numpy actually executes, not just imports.
        Python::attach(|py| {
            let np = py.import("numpy").unwrap();
            let arr = np.call_method1("zeros", (4,)).unwrap();
            let sum: f64 = arr.call_method0("sum").unwrap().extract().unwrap();
            assert_eq!(sum, 0.0, "numpy.zeros(4).sum() should be 0.0");
        });
    }
}
