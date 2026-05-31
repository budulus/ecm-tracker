//! GUI ↔ embedded-Python plugin glue (Phase 3 slice 3e).
//!
//! Bridges the egui app to the `ecm-pyhost` plugin host: builds the read-only [`ContextSnapshot`]
//! from `ProjectState`, discovers plugins from the plugins directory, and launches one. The host
//! owns the dual-index/data details and the `PluginContext` façade; this module only maps
//! `ProjectState` → `ContextSnapshot` and drives `pyhost::{discover, launch}` under
//! `Python::attach`. It is the first (and, for now, only) place the GUI touches embedded Python.

use ecm_core::project_state::ProjectState;
use ecm_pyhost::{
    discover as host_discover, dispatch_control as host_dispatch_control, dispatch_event,
    ensure_embedded_site, has_overlay, has_panel, instantiate,
    panel_controls as host_panel_controls, take_keep_mask,
};
use ecm_pyhost::{
    overlay_commands, ContextSnapshot, Control, ControlValue, DrawCommand, PluginRecord,
};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Result of launching a plugin: a message for the status bar, and — if the plugin provides an
/// ongoing surface (an `overlay()` and/or a `panel()`) — its retained instance (the GUI keeps it to
/// re-invoke `overlay()`/`on_control()` and dispatch events).
pub struct LaunchOutcome {
    pub message: Option<String>,
    pub instance: Option<Py<PyAny>>,
    /// A full-length keep-mask the plugin recorded via `ctx.apply_keep_mask` during `launch()`,
    /// for the GUI to apply through its undoable mask path (slice 3g-b). `None` if it recorded none.
    pub keep_mask: Option<Vec<bool>>,
}

/// Plugin-facing state-change events — the reactive hub that replaces the 3f overlay
/// state-signature poll. Mirrors the Python `PluginSignals` (sequence/frame/result/mask/roi);
/// `FrameChanged` carries the new GLOBAL frame index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginEvent {
    SequenceChanged,
    FrameChanged(usize),
    ResultChanged,
    MaskChanged,
    RoiChanged,
}

impl PluginEvent {
    fn name(self) -> &'static str {
        match self {
            PluginEvent::SequenceChanged => "sequence_changed",
            PluginEvent::FrameChanged(_) => "frame_changed",
            PluginEvent::ResultChanged => "result_changed",
            PluginEvent::MaskChanged => "mask_changed",
            PluginEvent::RoiChanged => "roi_changed",
        }
    }
}

/// Deliver `events` to every retained plugin instance (calling their `on_<event>` hooks) under a
/// single `Python::attach`. Each call refreshes the instance's `ctx` to the current `state`. A hook
/// that raises is ignored so one buggy plugin can't break the others.
pub fn dispatch_events(instances: &[Py<PyAny>], events: &[PluginEvent], state: &ProjectState) {
    if instances.is_empty() || events.is_empty() {
        return;
    }
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        for inst in instances {
            for ev in events {
                let idx = match ev {
                    PluginEvent::FrameChanged(i) => Some(*i as i64),
                    _ => None,
                };
                let _ = dispatch_event(py, inst, ev.name(), idx, snapshot(state));
            }
        }
    });
}

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

/// Launch a discovered plugin with a fresh context built from `snap`: instantiate it, call
/// `launch()`, and report a string return (for the status bar). If the plugin provides an ongoing
/// surface (`overlay()` and/or `panel()`), its instance is retained in the outcome so the GUI can
/// re-invoke it and dispatch events/controls. `Err` with a message when the plugin is unloaded or
/// `launch()` raised.
pub fn launch(record: &PluginRecord, snap: ContextSnapshot) -> Result<LaunchOutcome, String> {
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        let instance = instantiate(py, record, snap).map_err(|e| format!("{e}"))?;
        let message = match instance.call_method0(py, "launch") {
            Ok(ret) => ret.bind(py).extract::<String>().ok(),
            Err(e) => return Err(format!("{e}")),
        };
        // Collect any keep-mask the plugin recorded in launch() — before `instance` is moved below.
        let keep_mask = take_keep_mask(py, &instance);
        let retain = has_overlay(py, &instance) || has_panel(py, &instance);
        let instance = retain.then_some(instance);
        Ok(LaunchOutcome { message, instance, keep_mask })
    })
}

/// Collect a plugin instance's declared control panel against the current `state` (slice 3g-c).
/// Empty if the plugin defines no `panel()` (or its `panel()` raised). The GUI renders these as an
/// egui window and owns the live values from there.
pub fn panel_controls(instance: &Py<PyAny>, state: &ProjectState) -> Vec<Control> {
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        host_panel_controls(py, instance, snapshot(state)).unwrap_or_default()
    })
}

/// Deliver a control change to the retained plugin at `idx` (calling its `on_control` hook with the
/// current `state`), and return any keep-mask it recorded via `ctx.apply_keep_mask` so the GUI can
/// apply it through the undoable path. `None` if the index is out of range, the hook raised, or
/// nothing was recorded.
pub fn dispatch_control(
    instances: &[Py<PyAny>],
    idx: usize,
    key: &str,
    value: ControlValue,
    state: &ProjectState,
) -> Option<Vec<bool>> {
    let instance = instances.get(idx)?;
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        host_dispatch_control(py, instance, key, value, snapshot(state)).ok()?;
        take_keep_mask(py, instance)
    })
}

/// Re-invoke every active overlay plugin's `overlay()` against the current `state`, concatenating
/// their draw-commands. A plugin whose `overlay()` raises is skipped (its commands are dropped this
/// refresh) rather than breaking the others.
pub fn refresh_overlays(instances: &[Py<PyAny>], state: &ProjectState) -> Vec<DrawCommand> {
    if instances.is_empty() {
        return Vec::new();
    }
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        let mut cmds = Vec::new();
        for inst in instances {
            if let Ok(mut c) = overlay_commands(py, inst, snapshot(state)) {
                cmds.append(&mut c);
            }
        }
        cmds
    })
}

/// Call `on_unload()` on each plugin instance (best-effort), e.g. when clearing overlays.
pub fn unload(instances: &[Py<PyAny>]) {
    if instances.is_empty() {
        return;
    }
    Python::attach(|py| {
        for inst in instances {
            let _ = inst.call_method0(py, "on_unload");
        }
    });
}
