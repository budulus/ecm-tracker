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
    panel_controls as host_panel_controls, take_keep_mask, take_settings,
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
            // Load this plugin's settings once for the whole drain (they only change via a save,
            // which we persist after the events below).
            let sjson = stashed_id(py, inst).as_deref().and_then(load_settings_json);
            for ev in events {
                let idx = match ev {
                    PluginEvent::FrameChanged(i) => Some(*i as i64),
                    _ => None,
                };
                let mut snap = snapshot(state);
                snap.settings_json = sjson.clone();
                let _ = dispatch_event(py, inst, ev.name(), idx, snap);
            }
            // A hook may have called ctx.save_settings — persist it.
            persist_settings(py, inst);
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
        settings_json: None, // filled per-plugin by `launch` / `snapshot_for` (read side, slice 3g-d)
    }
}

// ---- per-plugin settings persistence (slice 3g-d) -----------------------------------------------
//
// A plugin reads/writes its own persisted settings through `ctx.get_settings()` / `save_settings()`.
// The host can't reach `core::settings` (it stays opencv-free, so it has no `ecm-core` dep), so the
// GUI owns the I/O: it loads the plugin's `plugin:<id>` section into the snapshot's `settings_json`
// (the read side) and, after each call, persists whatever the plugin recorded via the host outbox
// (the save side) — the same collect-then-apply shape as `apply_keep_mask`. The plugin id is stashed
// on the retained instance at launch (`_ecm_plugin_id`) so later dispatches can find its section.

/// The settings.json section key for a plugin (`plugin:<id>`). Mirrors `api.py:_section`.
fn settings_key(id: &str) -> String {
    format!("plugin:{id}")
}

/// Load a plugin's persisted settings section as a compact JSON-object string (`None` if it has
/// saved none), for the snapshot's `settings_json`.
fn load_settings_json(id: &str) -> Option<String> {
    ecm_core::settings::get_section(&settings_key(id)).map(|v| v.to_string())
}

/// Persist a plugin's settings (a JSON-object string from `ctx.save_settings`) into its section.
/// Best-effort: a string that doesn't parse to a JSON object, or a write error, is dropped.
fn save_settings_json(id: &str, json: &str) {
    if let Ok(value @ serde_json::Value::Object(_)) = serde_json::from_str::<serde_json::Value>(json)
    {
        let _ = ecm_core::settings::update_section(&settings_key(id), value);
    }
}

/// The plugin id stashed on a retained instance at launch, used to locate its settings on later
/// dispatches. `None` if absent (it isn't set on a launched instance).
fn stashed_id(py: Python<'_>, instance: &Py<PyAny>) -> Option<String> {
    instance.bind(py).getattr("_ecm_plugin_id").ok()?.extract::<String>().ok()
}

/// Build the per-plugin snapshot for `instance`: the shared state snapshot plus this plugin's
/// persisted settings, so `ctx.get_settings()` reflects what was saved.
fn snapshot_for(py: Python<'_>, instance: &Py<PyAny>, state: &ProjectState) -> ContextSnapshot {
    let mut snap = snapshot(state);
    snap.settings_json = stashed_id(py, instance).as_deref().and_then(load_settings_json);
    snap
}

/// Persist any settings the plugin recorded via `ctx.save_settings` during the last call.
fn persist_settings(py: Python<'_>, instance: &Py<PyAny>) {
    if let Some(json) = take_settings(py, instance) {
        if let Some(id) = stashed_id(py, instance) {
            save_settings_json(&id, &json);
        }
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
pub fn launch(record: &PluginRecord, mut snap: ContextSnapshot) -> Result<LaunchOutcome, String> {
    Python::attach(|py| {
        ensure_embedded_site(py).ok();
        // Load this plugin's persisted settings so launch()/panel() can read them (slice 3g-d).
        snap.settings_json = load_settings_json(&record.id);
        let instance = instantiate(py, record, snap).map_err(|e| format!("{e}"))?;
        // Stash the id on the instance so later dispatches can locate its settings section.
        let _ = instance.bind(py).setattr("_ecm_plugin_id", record.id.as_str());
        let message = match instance.call_method0(py, "launch") {
            Ok(ret) => ret.bind(py).extract::<String>().ok(),
            Err(e) => return Err(format!("{e}")),
        };
        // Persist any settings the plugin saved during launch().
        if let Some(json) = take_settings(py, &instance) {
            save_settings_json(&record.id, &json);
        }
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
        let controls =
            host_panel_controls(py, instance, snapshot_for(py, instance, state)).unwrap_or_default();
        persist_settings(py, instance); // panel() may seed-and-save defaults
        controls
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
        host_dispatch_control(py, instance, key, value, snapshot_for(py, instance, state)).ok()?;
        persist_settings(py, instance); // on_control commonly persists the changed value
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
            if let Ok(mut c) = overlay_commands(py, inst, snapshot_for(py, inst, state)) {
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
