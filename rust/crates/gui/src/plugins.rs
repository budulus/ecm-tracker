//! GUI ↔ embedded-Python plugin glue (Phase 3 slice 3e).
//!
//! Bridges the egui app to the `ecm-pyhost` plugin host: builds the read-only [`ContextSnapshot`]
//! from `ProjectState`, discovers plugins from the plugins directory, and launches one. The host
//! owns the dual-index/data details and the `PluginContext` façade; this module only maps
//! `ProjectState` → `ContextSnapshot` and drives `pyhost::{discover, launch}` under
//! `Python::attach`. It is the first (and, for now, only) place the GUI touches embedded Python.

use ecm_core::project_state::ProjectState;
use ecm_pyhost::{discover as host_discover, ensure_embedded_site, launch as host_launch};
use ecm_pyhost::{ContextSnapshot, PluginRecord};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Where the app looks for plugin packages: `ECM_PLUGINS_DIR` if set, else a `plugins/` folder
/// next to the executable (the shipped layout), else `plugins/` relative to the working directory
/// (the dev layout — `cargoenv.ps1` runs from `rust/`, so this resolves to `rust/plugins/`). Each
/// subfolder with an `__init__.py` is a candidate plugin.
pub fn plugins_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("ECM_PLUGINS_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let beside_exe = parent.join("plugins");
            if beside_exe.is_dir() {
                return beside_exe;
            }
        }
    }
    PathBuf::from("plugins")
}

/// Build the immutable read-only snapshot handed to a plugin's `PluginContext`. Mirrors the
/// field-for-field mapping documented in `pyhost::context::ContextSnapshot`.
pub fn snapshot(state: &ProjectState) -> ContextSnapshot {
    ContextSnapshot {
        n_total_images: state.total_images(),
        has_sequence: state.has_sequence(),
        has_result: state.result.is_some(),
        image_size: state
            .image_size()
            .ok()
            .flatten()
            .map(|(h, w)| (h as usize, w as usize)),
        reference_index: state.reference_index,
        last_index: state.last_index,
        current_index: state.current_index,
        frame_count: state.result.as_ref().map_or(0, |r| r.n_frames()),
        point_count: state.result.as_ref().map_or(0, |r| r.n_points()),
        n_active: state
            .active_mask
            .as_ref()
            .map_or(0, |m| m.iter().filter(|&&b| b).count()),
        roi_corners: state
            .roi
            .as_ref()
            .map_or_else(Vec::new, |r| r.corners.clone()),
        coords_fw: state.result.as_ref().map(|r| r.coords_fw.clone()),
        status_fw: state.result.as_ref().map(|r| r.status_fw.clone()),
        active_mask: state.active_mask.clone(),
    }
}

/// Discover plugins from [`plugins_dir`]. Returns one record per package (loaded, or carrying a
/// per-plugin load error); an empty vec if the directory is missing or discovery fails. Ensures
/// the bundled site-packages are importable first so plugins can use the scientific stack.
pub fn discover() -> Vec<PluginRecord> {
    let dir = plugins_dir();
    if !dir.is_dir() {
        return Vec::new();
    }
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        host_discover(py, &dir).unwrap_or_default()
    })
}

/// Launch a discovered plugin with a fresh context built from `snap`. Returns the plugin's
/// `launch()` return value when it is a string (for the status bar), else `Ok(None)`; `Err` with a
/// message when the plugin is unloaded or `launch()` raised.
pub fn launch(record: &PluginRecord, snap: ContextSnapshot) -> Result<Option<String>, String> {
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        match host_launch(py, record, snap) {
            Ok(ret) => Ok(ret.bind(py).extract::<String>().ok()),
            Err(e) => Err(format!("{e}")),
        }
    })
}
