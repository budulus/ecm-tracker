//! ECM Tracker plugin host (Rust port target).
//!
//! Embeds a bundled CPython (via pyo3) and will expose the `PluginContext`
//! façade to Python plugins — the evolution of `app/plugins/api.py`. Phase 0
//! only proves the embedded interpreter boots and can `import numpy`; the API
//! surface is built out in Phase 3.

use pyo3::prelude::*;

pub mod context;
pub use context::{ContextSnapshot, PluginContext};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_numpy_and_runs() {
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
