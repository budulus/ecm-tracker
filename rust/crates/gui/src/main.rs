//! ECM Tracker — egui host application (Phase 2).
//!
//! Slice 1: open a folder, display the current frame on a zoom/pan canvas, scrub frames.
//! Slice 2: define a rectangular ROI on the canvas, detect Shi-Tomasi corners inside it, and
//! draw the ROI + seed-feature overlays. Built on the Phase 1 `ecm-core` pipeline.
//!
//! `ECM_SMOKE=1` exits after the first frame (headless build check). `ECM_SMOKE_DIR=<dir>`
//! additionally loads that folder first, so the smoke run exercises the decode→texture path.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console in release

mod canvas;
mod icons;
mod plugins;
mod theme;

use canvas::CanvasView;
use ecm_core::cleanup;
use ecm_core::export;
use ecm_core::feature_detection;
use ecm_core::image_sequence::{discover_dir, ImageSequence};
use ecm_core::project_state::ProjectState;
use ecm_core::result::TrackerResult;
use ecm_core::roi::Roi;
use ecm_core::settings;
use ecm_pyhost::{Control, ControlValue, DrawCommand, PluginRecord};
use eframe::egui;
use pyo3::{Py, PyAny};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

fn main() -> eframe::Result<()> {
    let smoke = std::env::var_os("ECM_SMOKE").is_some();
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1100.0, 720.0])
        .with_min_inner_size([640.0, 480.0])
        .with_title("ECM Tracker");
    if let Some(icon) = icons::app_icon(256) {
        viewport = viewport.with_icon(Arc::new(icon));
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native(
        "ECM Tracker",
        options,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx); // light theme on the whole context (window + dialogs)
            Ok(Box::new(EcmApp::new(smoke)))
        }),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Pan,
    RoiRect,
    RoiCircle,
    RoiNgon,
}

/// Which parameter dialog window is open (Parameters menu).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialog {
    Corner,
    Grid,
    Tracker,
    Display,
}

/// Message from the background tracking thread to the UI.
enum TrackMsg {
    /// Passes completed so far (out of `2 * (n_cut - 1)`).
    Progress(usize),
    /// Tracking finished: `Some` = result, `None` = cancelled. Boxed (large variant).
    Done(Box<Option<TrackerResult>>),
    Error(String),
}

/// A running background tracking job: live progress, a cancel flag, and the result channel.
/// egui can't pump events mid-call (the Python app uses `QProgressDialog` +
/// `processEvents`), so `core::tracking::track` runs on a worker thread and reports back here.
struct TrackJob {
    rx: mpsc::Receiver<TrackMsg>,
    cancel: Arc<AtomicBool>,
    done: usize,
    total: usize,
    _handle: thread::JoinHandle<()>,
}

/// The 7 numeric cleanup bands in panel order. Each maps to a `cleanup::Thresholds` band field
/// and is a single *max* threshold (keep a point iff `metric <= hi`), matching the Python dialog.
const BAND_LABELS: [&str; 7] = [
    "Forward failures (max)",
    "Backward failures (max)",
    "OpenCV error (max)",
    "Mean OpenCV error (max)",
    "FB error — mean (max)",
    "FB error — max (max)",
    "Max step distance (max)",
];

/// Finite maximum of a metric column (ignoring the +inf never-tracked points), or 0.0 if none
/// are finite — the per-band ceiling for the drag clamps, so each slider is scaled to its data.
fn finite_max(values: &[f32]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, f32::max) as f64
}

/// Open cleanup-panel session: metrics computed once from the result, the live thresholds being
/// edited, each band's data ceiling (for the drag clamps), and the latest preview keep-mask.
/// Mirrors the state `MainWindow` holds while `CleanupDialog` is open.
struct CleanupState {
    metrics: cleanup::Metrics,
    thresholds: cleanup::Thresholds,
    caps: [f64; 7],          // per-band data ceiling, in BAND_LABELS order
    preview_keep: Vec<bool>, // build_mask result, recomputed each frame
}

/// A button action collected while rendering the cleanup panel, applied after its borrows end.
#[derive(PartialEq)]
enum CleanupAction {
    None,
    Apply,
    Undo,
    Close,
}

struct EcmApp {
    state: ProjectState,
    canvas: CanvasView,
    /// Lazily-rendered, recolored SVG toolbar-icon textures.
    icons: icons::IconStore,
    tool: Tool,
    /// Image-space start corner while dragging a rectangular ROI.
    roi_draft_start: Option<egui::Pos2>,
    /// (frame index currently uploaded, texture handle).
    tex: Option<(usize, egui::TextureHandle)>,
    /// In-flight background tracking job (progress + cancel), if any.
    track_job: Option<TrackJob>,
    /// Open cleanup panel session (metrics + thresholds + preview), if any.
    cleanup: Option<CleanupState>,
    /// Open parameter dialog window, if any.
    dialog: Option<Dialog>,
    /// Discovered plugins, lazily populated on first Plugins-menu open (`None` until then).
    plugins: Option<Vec<PluginRecord>>,
    /// Retained plugin instances — those providing an ongoing surface (an `overlay()` and/or a
    /// `panel()`). Kept so `overlay()`/`on_control()` can be re-invoked and events dispatched.
    overlay_plugins: Vec<Py<PyAny>>,
    /// Open plugin control panels (slice 3g-c), each indexing into `overlay_plugins`.
    panels: Vec<PluginPanel>,
    /// Cached plugin draw-commands (image space), refreshed when the state signature changes.
    overlay_cmds: Vec<DrawCommand>,
    /// Pending plugin events emitted by state transitions this frame; drained next `update()`
    /// to dispatch to plugins and refresh overlays (the reactive hub, replacing `overlay_sig`).
    events: Vec<plugins::PluginEvent>,
    status: String,
    smoke: bool,
}

/// Image px below which a rubber-band drag is treated as a stray click (ROI discarded).
/// Mirrors `roi_tools.MIN_SIZE`.
const MIN_ROI_SIZE: f32 = 3.0;
/// Polygon segments approximating a Circle ROI. Mirrors `roi_tools.CIRCLE_SEGMENTS`.
const CIRCLE_SEGMENTS: usize = 64;

/// Axis-aligned rectangle corners from two image-space points, or `None` if the drag is too
/// small in either axis. Mirrors `RectangleTool._corners`.
fn rect_corners(a: egui::Pos2, b: egui::Pos2) -> Option<Vec<(f64, f64)>> {
    if (b.x - a.x).abs() < MIN_ROI_SIZE || (b.y - a.y).abs() < MIN_ROI_SIZE {
        return None;
    }
    Some(vec![
        (a.x as f64, a.y as f64),
        (b.x as f64, a.y as f64),
        (b.x as f64, b.y as f64),
        (a.x as f64, b.y as f64),
    ])
}

/// A `CIRCLE_SEGMENTS`-gon approximating the circle centered at `center` with radius reaching
/// `edge`, or `None` if the radius is too small. Mirrors `CircleTool._corners`.
fn circle_corners(center: egui::Pos2, edge: egui::Pos2) -> Option<Vec<(f64, f64)>> {
    let r = (edge - center).length();
    if r < MIN_ROI_SIZE {
        return None;
    }
    let (cx, cy, r) = (center.x as f64, center.y as f64, r as f64);
    Some(
        (0..CIRCLE_SEGMENTS)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * (i as f64) / (CIRCLE_SEGMENTS as f64);
                (cx + r * a.cos(), cy + r * a.sin())
            })
            .collect(),
    )
}

/// A toolbar button with an optional bundled icon (text-only fallback when none is bundled).
/// `selected` gives the checked/toggle look; `accent` paints it as the primary action (blue fill +
/// white glyph). Mirrors the `QToolButton` styling from `theme.py`.
fn tool_button(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    store: &mut icons::IconStore,
    icon: &'static str,
    label: &str,
    selected: bool,
    accent: bool,
    enabled: bool,
) -> egui::Response {
    let glyph = if accent { icons::ACCENT } else { icons::NORMAL };
    let mut btn = match store.image(ctx, icon, glyph, 16.0) {
        Some(img) => egui::Button::image_and_text(img, label),
        None => egui::Button::new(label),
    };
    btn = btn.selected(selected);
    if accent {
        btn = btn.fill(egui::Color32::from_rgb(0x25, 0x63, 0xeb));
    }
    ui.add_enabled(enabled, btn)
}

/// An open plugin control panel (slice 3g-c): the plugin's declared controls with host-owned live
/// values, rendered as an egui window. `instance_idx` indexes [`EcmApp::overlay_plugins`] (the
/// retained instances), so control changes can be dispatched to the right plugin's `on_control`.
struct PluginPanel {
    instance_idx: usize,
    title: String,
    open: bool,
    /// Declared controls; the host owns the live values (egui mutates them in place).
    controls: Vec<Control>,
}

/// Render one declared control as an egui widget, pushing a `(key, new value)` onto `changes` when
/// the user changes it (slice 3g-c). Buttons report a `Click`; labels are static.
fn render_control(ui: &mut egui::Ui, control: &mut Control, changes: &mut Vec<(String, ControlValue)>) {
    match control {
        Control::Slider { key, label, value, min, max } => {
            if ui.add(egui::Slider::new(value, *min..=*max).text(label.as_str())).changed() {
                changes.push((key.clone(), ControlValue::Float(*value)));
            }
        }
        Control::Checkbox { key, label, value } => {
            if ui.checkbox(value, label.as_str()).changed() {
                changes.push((key.clone(), ControlValue::Bool(*value)));
            }
        }
        Control::Button { key, label } => {
            if ui.button(label.as_str()).clicked() {
                changes.push((key.clone(), ControlValue::Click));
            }
        }
        Control::Label { text } => {
            ui.label(text.as_str());
        }
    }
}

impl EcmApp {
    fn new(smoke: bool) -> Self {
        let mut app = Self {
            state: ProjectState::new(),
            canvas: CanvasView::default(),
            icons: icons::IconStore::default(),
            tool: Tool::Pan,
            roi_draft_start: None,
            tex: None,
            track_job: None,
            cleanup: None,
            dialog: None,
            plugins: None,
            overlay_plugins: Vec::new(),
            panels: Vec::new(),
            overlay_cmds: Vec::new(),
            events: Vec::new(),
            status: "Open a folder of images to begin.".into(),
            smoke,
        };
        if let Some(dir) = std::env::var_os("ECM_SMOKE_DIR") {
            app.open_dir(PathBuf::from(dir));
            app.detect_corners(); // exercise the detection path in the smoke check
            if app.smoke {
                app.smoke_track(); // exercise the tracking path in the smoke check
            }
        }
        app
    }

    /// Launch the discovered plugin at index `i`: build a snapshot from the current state, run its
    /// `launch()`, report the outcome in the status bar, and — if it draws an overlay — retain its
    /// instance and force an overlay refresh.
    fn launch_plugin(&mut self, i: usize) {
        let snap = plugins::snapshot(&self.state);
        let (name, result) = {
            let record = &self.plugins.as_ref().expect("plugins discovered")[i];
            (record.name.clone(), plugins::launch(record, snap))
        };
        match result {
            Ok(outcome) => {
                self.status = outcome.message.unwrap_or_else(|| format!("{name} ran."));
                // A plugin may filter the active points via ctx.apply_keep_mask; apply it through
                // the shared undoable path (snapshots undo, ANDs in keep, emits MaskChanged).
                if let Some(keep) = outcome.keep_mask {
                    self.apply_keep_mask(&keep);
                }
                if let Some(instance) = outcome.instance {
                    self.overlay_plugins.push(instance);
                    let idx = self.overlay_plugins.len() - 1;
                    // Collect the plugin's declared control panel, if any (slice 3g-c).
                    let controls = plugins::panel_controls(&self.overlay_plugins[idx], &self.state);
                    if !controls.is_empty() {
                        self.panels.push(PluginPanel {
                            instance_idx: idx,
                            title: name.clone(),
                            open: true,
                            controls,
                        });
                    }
                    // Render the just-launched overlay immediately (don't wait for an event).
                    self.overlay_cmds = plugins::refresh_overlays(&self.overlay_plugins, &self.state);
                }
            }
            Err(e) => self.status = format!("{name}: {e}"),
        }
    }

    /// Drop all retained plugin instances (best-effort `on_unload`), clearing their cached
    /// overlay commands and open control panels.
    fn clear_plugins(&mut self) {
        plugins::unload(&self.overlay_plugins);
        self.overlay_plugins.clear();
        self.panels.clear();
        self.overlay_cmds.clear();
    }

    /// Render the open plugin control panels and process any control changes the user made this
    /// frame (slice 3g-c): each change is delivered to the plugin's `on_control`, then any
    /// `apply_keep_mask` it recorded is applied undoably and the overlays refreshed. Closed windows
    /// are dropped (the instance stays retained — closing a panel doesn't unload the plugin).
    fn show_panels(&mut self, ctx: &egui::Context) {
        if self.panels.is_empty() {
            return;
        }
        // (instance_idx, key, new value) for each control the user changed this frame.
        let mut changes: Vec<(usize, String, ControlValue)> = Vec::new();
        for panel in &mut self.panels {
            let mut open = panel.open;
            egui::Window::new(panel.title.clone())
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| {
                    let mut local: Vec<(String, ControlValue)> = Vec::new();
                    for control in &mut panel.controls {
                        render_control(ui, control, &mut local);
                    }
                    for (key, value) in local {
                        changes.push((panel.instance_idx, key, value));
                    }
                });
            panel.open = open;
        }
        self.panels.retain(|p| p.open);
        for (idx, key, value) in changes {
            self.dispatch_control(idx, &key, value);
        }
    }

    /// Deliver one control change to the retained plugin at `idx`, applying any keep-mask it records
    /// and refreshing overlays in case `on_control` changed overlay-affecting state (slice 3g-c).
    fn dispatch_control(&mut self, idx: usize, key: &str, value: ControlValue) {
        let keep = plugins::dispatch_control(&self.overlay_plugins, idx, key, value, &self.state);
        if let Some(keep) = keep {
            self.apply_keep_mask(&keep);
        }
        self.overlay_cmds = plugins::refresh_overlays(&self.overlay_plugins, &self.state);
    }

    /// Drain the plugin event queue (events emitted by state transitions): dispatch each to the
    /// retained overlay plugins' `on_*` hooks, then re-invoke their `overlay()` so the canvas
    /// reflects the new state. Replaces the 3f state-signature poll — overlays now refresh on real
    /// transitions (including ROI changes, which the old signature missed). No-op when idle.
    fn process_plugin_events(&mut self) {
        if self.events.is_empty() {
            return;
        }
        let events = std::mem::take(&mut self.events);
        if self.overlay_plugins.is_empty() {
            return;
        }
        plugins::dispatch_events(&self.overlay_plugins, &events, &self.state);
        self.overlay_cmds = plugins::refresh_overlays(&self.overlay_plugins, &self.state);
    }

    fn open_dir(&mut self, dir: PathBuf) {
        match discover_dir(&dir) {
            Ok(paths) if !paths.is_empty() => match ImageSequence::new(paths) {
                Ok(seq) => {
                    let n = seq.len();
                    self.state.load_sequence(seq, dir.to_str().map(str::to_string));
                    self.canvas.reset();
                    self.tex = None;
                    self.status = format!("Loaded {n} frames from {}", dir.display());
                    self.events.push(plugins::PluginEvent::SequenceChanged);
                }
                Err(e) => self.status = format!("Load error: {e}"),
            },
            Ok(_) => self.status = "No supported images in that folder.".into(),
            Err(e) => self.status = format!("Open error: {e}"),
        }
    }

    fn detect_corners(&mut self) {
        if !self.state.has_sequence() {
            return;
        }
        let result = {
            let seq = self.state.sequence.as_ref().unwrap();
            feature_detection::detect_corners(
                seq,
                self.state.reference_index,
                self.state.roi.as_ref(),
                &self.state.shi_tomasi_params,
            )
        };
        match result {
            Ok(pts) => {
                self.status = format!("Detected {} corners", pts.len());
                self.invalidate_tracking(); // new seeds → any existing result is stale
                self.state.features = Some(pts);
            }
            Err(e) => self.status = format!("Detect error: {e}"),
        }
    }

    /// Seed a regular grid of points inside the (complete) ROI. Mirrors `_detect_grid`.
    fn detect_grid(&mut self) {
        let result = {
            let Some(roi) = self.state.roi.as_ref() else {
                self.status = "Define an ROI first.".into();
                return;
            };
            if !roi.is_complete() {
                self.status = "ROI is not complete.".into();
                return;
            }
            feature_detection::regular_grid(
                roi,
                self.state.grid_params.spacing_x as f64,
                self.state.grid_params.spacing_y as f64,
            )
        };
        match result {
            Ok(pts) => {
                let n = pts.len();
                self.invalidate_tracking(); // new seeds → any existing result is stale
                self.state.features = Some(pts);
                self.status = format!("Grid: {n} feature points.");
            }
            Err(e) => self.status = format!("Grid error: {e}"),
        }
    }

    /// Export the kept (active) forward coordinates via a native save dialog — `.npy`
    /// (+ a `_sequence.txt` sidecar) or flat `.csv`. Mirrors `_export` / `_export_csv`.
    fn export_coords(&mut self, csv: bool) {
        let Some(result) = self.state.result.as_ref() else {
            return;
        };
        let Some(mask) = self.state.active_mask.as_ref() else {
            return;
        };
        if !mask.iter().any(|&b| b) {
            self.status = "Nothing to export — no active points remain.".into();
            return;
        }
        let default_name = if csv { "coords.csv" } else { "coords.npy" };
        let (filter_name, ext): (&str, &str) =
            if csv { ("CSV file", "csv") } else { ("NumPy array", "npy") };
        let mut dlg = rfd::FileDialog::new()
            .add_filter(filter_name, &[ext])
            .set_file_name(default_name);
        if let Some(dir) = self.state.source_dir.as_ref() {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else {
            return;
        };
        let out_dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(default_name)
            .to_string();

        let outcome: std::io::Result<(usize, usize, usize)> = if csv {
            let frame_names: Vec<String> = match self.state.sequence.as_ref() {
                Some(s) => (result.reference_index..=result.last_index)
                    .map(|g| {
                        s.paths[g]
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect(),
                None => Vec::new(),
            };
            export::export_csv(&result.coords_fw, mask, &frame_names, &out_dir, &filename)
                .map(|(_, shape)| shape)
        } else {
            export::export(
                &result.coords_fw,
                mask,
                result.reference_index,
                result.last_index,
                &out_dir,
                &filename,
            )
            .map(|e| e.shape)
        };
        self.status = match outcome {
            Ok((n, k, _)) => format!("Exported {k} points × {n} frames to {filename}"),
            Err(e) => format!("Export failed: {e}"),
        };
    }

    /// Render the active parameter dialog window. Edits the live params in `ProjectState`;
    /// "Save as defaults" persists that section via `settings::update_section`. Mirrors the
    /// Parameters-menu dialogs in `app/gui/dialogs.py`.
    fn show_dialogs(&mut self, ctx: &egui::Context) {
        let Some(kind) = self.dialog else {
            return;
        };
        let mut open = true;
        let mut save = false;
        match kind {
            Dialog::Corner => {
                egui::Window::new("Corner Detection")
                    .open(&mut open)
                    .resizable(false)
                    .show(ctx, |ui| {
                        let p = &mut self.state.shi_tomasi_params;
                        egui::Grid::new("corner_grid").num_columns(2).show(ui, |ui| {
                            ui.label("Max corners");
                            ui.add(egui::DragValue::new(&mut p.max_corners).range(1..=100_000));
                            ui.end_row();
                            ui.label("Quality level");
                            ui.add(
                                egui::DragValue::new(&mut p.quality_level)
                                    .speed(0.001)
                                    .range(0.0001..=1.0),
                            );
                            ui.end_row();
                            ui.label("Min distance");
                            ui.add(
                                egui::DragValue::new(&mut p.min_distance)
                                    .speed(0.1)
                                    .range(0.0..=500.0),
                            );
                            ui.end_row();
                            ui.label("Block size");
                            ui.add(egui::DragValue::new(&mut p.block_size).range(1..=99));
                            ui.end_row();
                            ui.label("Harris k");
                            ui.add(egui::DragValue::new(&mut p.k).speed(0.001).range(0.0..=1.0));
                            ui.end_row();
                        });
                        ui.checkbox(&mut p.use_harris_detector, "Use Harris detector");
                        ui.separator();
                        save = ui.button("Save as defaults").clicked();
                    });
                if save {
                    let _ = settings::save_section("shi_tomasi", &self.state.shi_tomasi_params);
                    self.status = "Saved corner-detection defaults.".into();
                }
            }
            Dialog::Grid => {
                egui::Window::new("Grid")
                    .open(&mut open)
                    .resizable(false)
                    .show(ctx, |ui| {
                        let p = &mut self.state.grid_params;
                        egui::Grid::new("grid_grid").num_columns(2).show(ui, |ui| {
                            ui.label("Spacing X (px)");
                            ui.add(egui::DragValue::new(&mut p.spacing_x).range(1..=1000));
                            ui.end_row();
                            ui.label("Spacing Y (px)");
                            ui.add(egui::DragValue::new(&mut p.spacing_y).range(1..=1000));
                            ui.end_row();
                        });
                        ui.separator();
                        save = ui.button("Save as defaults").clicked();
                    });
                if save {
                    let _ = settings::save_section("grid", &self.state.grid_params);
                    self.status = "Saved grid defaults.".into();
                }
            }
            Dialog::Tracker => {
                egui::Window::new("Tracker (Lucas–Kanade)")
                    .open(&mut open)
                    .resizable(false)
                    .show(ctx, |ui| {
                        let p = &mut self.state.lk_params;
                        egui::Grid::new("lk_grid").num_columns(2).show(ui, |ui| {
                            ui.label("Window size");
                            ui.add(egui::DragValue::new(&mut p.win_size).range(1..=199));
                            ui.end_row();
                            ui.label("Pyramid levels");
                            ui.add(egui::DragValue::new(&mut p.max_level).range(0..=10));
                            ui.end_row();
                            ui.label("Max iterations");
                            ui.add(egui::DragValue::new(&mut p.max_iter).range(1..=200));
                            ui.end_row();
                            ui.label("Epsilon");
                            ui.add(
                                egui::DragValue::new(&mut p.epsilon)
                                    .speed(0.001)
                                    .range(0.0001..=1.0),
                            );
                            ui.end_row();
                            ui.label("Min eigen threshold");
                            ui.add(
                                egui::DragValue::new(&mut p.min_eig_threshold)
                                    .speed(0.0001)
                                    .range(0.0..=1.0),
                            );
                            ui.end_row();
                        });
                        ui.separator();
                        save = ui.button("Save as defaults").clicked();
                    });
                if save {
                    let _ = settings::save_section("lk", &self.state.lk_params);
                    self.status = "Saved tracker defaults.".into();
                }
            }
            Dialog::Display => {
                egui::Window::new("Display")
                    .open(&mut open)
                    .resizable(false)
                    .show(ctx, |ui| {
                        let p = &mut self.state.display_params;
                        ui.checkbox(&mut p.show_markers, "Show markers");
                        ui.checkbox(&mut p.show_roi, "Show ROI");
                        ui.checkbox(&mut p.show_window_box, "Show LK window box");
                        egui::Grid::new("disp_grid").num_columns(2).show(ui, |ui| {
                            ui.label("Marker size");
                            ui.add(egui::DragValue::new(&mut p.marker_size).range(1..=20));
                            ui.end_row();
                            ui.label("Opacity %");
                            ui.add(egui::DragValue::new(&mut p.marker_opacity).range(0..=100));
                            ui.end_row();
                        });
                        ui.separator();
                        save = ui.button("Save as defaults").clicked();
                    });
                if save {
                    let _ = settings::save_section("display", &self.state.display_params);
                    self.status = "Saved display defaults.".into();
                }
            }
        }
        if !open {
            self.dialog = None;
        }
    }

    /// Discard any tracking result (and its mask/undo). Seeds and ROI are left untouched.
    /// Called whenever the inputs to tracking (seeds/ROI) change so stale tracks aren't shown.
    fn invalidate_tracking(&mut self) {
        let had_result = self.state.result.is_some();
        self.state.result = None;
        self.state.active_mask = None;
        self.state.undo_stack.clear();
        self.cleanup = None; // metrics referenced the old result
        if had_result {
            self.events.push(plugins::PluginEvent::ResultChanged);
        }
    }

    /// Open the cleanup session: compute per-point metrics once and record each band's data
    /// ceiling for the drag clamps. Bands start permissive (`hi` = ceiling) and disabled, so a
    /// freshly enabled band keeps everything until the user tightens it. Mirrors `_open_cleanup`.
    fn open_cleanup(&mut self) {
        let Some(result) = self.state.result.as_ref() else {
            return;
        };
        let image_size = match self.state.image_size() {
            Ok(Some(size)) => size,
            _ => return,
        };
        let metrics = match cleanup::compute_metrics(result, self.state.roi.as_ref(), image_size) {
            Ok(m) => m,
            Err(e) => {
                self.status = format!("Cleanup metrics error: {e}");
                return;
            }
        };
        let caps = [
            metrics.fail_count_fw.iter().copied().max().unwrap_or(0) as f64,
            metrics.fail_count_bw.iter().copied().max().unwrap_or(0) as f64,
            finite_max(&metrics.max_err_fw),
            finite_max(&metrics.mean_err_fw),
            finite_max(&metrics.fb_mean),
            finite_max(&metrics.fb_max),
            finite_max(&metrics.max_step),
        ];
        let mut thresholds = cleanup::default_thresholds();
        for (band, &cap) in [
            &mut thresholds.fw_failures,
            &mut thresholds.bw_failures,
            &mut thresholds.opencv_error,
            &mut thresholds.mean_error,
            &mut thresholds.fb_mean,
            &mut thresholds.fb_max,
            &mut thresholds.distance,
        ]
        .into_iter()
        .zip(caps.iter())
        {
            band.hi = cap;
            band.cap = cap;
        }
        self.cleanup = Some(CleanupState { metrics, thresholds, caps, preview_keep: Vec::new() });
    }

    /// Filter the active point set by a full-length keep mask, undoably: snapshot the current
    /// mask onto the undo stack, then AND in `keep` (points only ever leave the active set).
    /// Mirrors `MainWindow.apply_keep_mask` (the shared mask-mutation path).
    fn apply_keep_mask(&mut self, keep: &[bool]) {
        let Some(active) = self.state.active_mask.clone() else {
            return;
        };
        if active.len() != keep.len() {
            return;
        }
        self.state.undo_stack.push(active.clone());
        let next: Vec<bool> = active.iter().zip(keep).map(|(&a, &k)| a && k).collect();
        self.state.active_mask = Some(next);
        self.events.push(plugins::PluginEvent::MaskChanged);
    }

    fn clear_tracking(&mut self) {
        self.invalidate_tracking();
        self.status = "Tracking cleared.".into();
    }

    /// Synchronous tracking for the headless smoke run only — the interactive path is the
    /// background `run_tracking`, whose thread + modal can't complete in a single `update()`
    /// before the smoke build closes the window. Exercises the track→result→overlay path.
    fn smoke_track(&mut self) {
        let Some(seq) = self.state.sequence.as_ref() else {
            return;
        };
        let Some(feats) = self.state.features.clone() else {
            return;
        };
        if feats.is_empty() {
            return;
        }
        match ecm_core::tracking::track(
            seq,
            self.state.reference_index,
            self.state.last_index,
            &feats,
            &self.state.lk_params,
            None,
        ) {
            Ok(Some(result)) => {
                self.state.active_mask = Some(vec![true; result.n_points()]);
                self.state.result = Some(result);
            }
            Ok(None) => {}
            Err(e) => self.status = format!("smoke track error: {e}"),
        }
    }

    /// Start tracking the current seeds on a background thread (no-op if one is already
    /// running or there are no seeds). Mirrors `MainWindow._run_tracking`.
    fn run_tracking(&mut self) {
        if self.track_job.is_some() {
            return;
        }
        let Some(seq) = self.state.sequence.as_ref() else {
            return;
        };
        let feats = match self.state.features.as_ref() {
            Some(f) if !f.is_empty() => f.clone(),
            _ => return,
        };
        let paths = seq.paths.clone();
        let reference = self.state.reference_index;
        let last = self.state.last_index;
        let lk = self.state.lk_params;
        let total = (2 * self.state.n_cut().saturating_sub(1)).max(1);

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = cancel.clone();
        let (tx, rx) = mpsc::channel::<TrackMsg>();
        let handle = thread::spawn(move || {
            // The worker owns its own ImageSequence (built from cloned paths) so the UI
            // thread keeps decoding frames for display independently.
            let seq = match ImageSequence::new(paths) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(TrackMsg::Error(e));
                    return;
                }
            };
            let progress_tx = tx.clone();
            let mut progress = move |done: usize, _total: usize| -> bool {
                let _ = progress_tx.send(TrackMsg::Progress(done));
                cancel_thread.load(Ordering::Relaxed)
            };
            let outcome =
                ecm_core::tracking::track(&seq, reference, last, &feats, &lk, Some(&mut progress));
            let msg = match outcome {
                Ok(opt) => TrackMsg::Done(Box::new(opt)),
                Err(e) => TrackMsg::Error(e.to_string()),
            };
            let _ = tx.send(msg);
        });

        self.track_job = Some(TrackJob { rx, cancel, done: 0, total, _handle: handle });
        self.status = "Tracking…".into();
    }

    /// Drain progress/result messages from the worker. On completion stores the result with a
    /// default all-true active mask (mirrors `MainWindow._run_tracking`'s success path).
    fn poll_track_job(&mut self, ctx: &egui::Context) {
        if self.track_job.is_none() {
            return;
        }
        enum Outcome {
            Pending,
            Cancelled,
            Done(TrackerResult),
            Error(String),
        }
        let mut outcome = Outcome::Pending;
        {
            let job = self.track_job.as_mut().unwrap();
            loop {
                match job.rx.try_recv() {
                    Ok(TrackMsg::Progress(d)) => job.done = d,
                    Ok(TrackMsg::Done(opt)) => {
                        outcome = match *opt {
                            Some(r) => Outcome::Done(r),
                            None => Outcome::Cancelled,
                        };
                        break;
                    }
                    Ok(TrackMsg::Error(e)) => {
                        outcome = Outcome::Error(e);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        outcome = Outcome::Cancelled;
                        break;
                    }
                }
            }
        }
        match outcome {
            Outcome::Pending => ctx.request_repaint(), // keep polling while the worker runs
            Outcome::Cancelled => {
                self.track_job = None;
                self.status = "Tracking cancelled.".into();
            }
            Outcome::Error(e) => {
                self.track_job = None;
                self.status = format!("Tracking error: {e}");
            }
            Outcome::Done(result) => {
                let (np, nf) = (result.n_points(), result.n_frames());
                self.state.active_mask = Some(vec![true; np]);
                self.state.result = Some(result);
                self.state.undo_stack.clear();
                self.cleanup = None; // a new result invalidates any open cleanup session
                self.track_job = None;
                self.status = format!("Tracked {np} points over {nf} frames.");
                self.events.push(plugins::PluginEvent::ResultChanged);
            }
        }
    }

    /// Upload the current frame to a GPU texture (only when the frame index changes).
    fn ensure_texture(&mut self, ctx: &egui::Context) {
        let Some(seq) = self.state.sequence.as_ref() else {
            self.tex = None;
            return;
        };
        let idx = self.state.current_index;
        if self.tex.as_ref().map(|(i, _)| *i) == Some(idx) {
            return;
        }
        match seq.load_rgba(idx) {
            Ok(f) => {
                let img = egui::ColorImage::from_rgba_unmultiplied([f.width, f.height], &f.pixels);
                let handle = ctx.load_texture(format!("frame{idx}"), img, egui::TextureOptions::LINEAR);
                self.tex = Some((idx, handle));
            }
            Err(e) => {
                self.status = format!("Decode error: {e}");
                self.tex = None;
            }
        }
    }

    /// Route canvas mouse input (image-space) to the active ROI tool for the current frame.
    /// Rect/Circle are rubber-band drags; N-Gon collects clicks. Mirrors the duck-typed
    /// `CanvasInteraction`s wired up by `app/gui/roi_tools.py`.
    fn handle_roi_interaction(&mut self, tf: &canvas::Transform, r: &egui::Response) {
        match self.tool {
            Tool::RoiRect | Tool::RoiCircle => self.handle_roi_drag(tf, r),
            Tool::RoiNgon => self.handle_ngon(tf, r),
            Tool::Pan => {}
        }
    }

    /// Rubber-band drag shared by the Rect/Circle tools: press fixes the anchor, drag previews
    /// the polygon live (via `drag_corners`), release commits it — or discards a too-small drag
    /// (`drag_corners` → `None`). Mirrors the `_DragTool` press/move/release plumbing.
    fn handle_roi_drag(&mut self, tf: &canvas::Transform, r: &egui::Response) {
        if r.drag_started_by(egui::PointerButton::Primary) {
            self.roi_draft_start = r.interact_pointer_pos().map(|p| tf.screen_to_image(p));
        }
        if r.dragged_by(egui::PointerButton::Primary) || r.drag_stopped_by(egui::PointerButton::Primary)
        {
            if let (Some(start), Some(p)) = (self.roi_draft_start, r.interact_pointer_pos()) {
                self.state.roi = self.drag_corners(start, tf.screen_to_image(p)).map(Roi::new);
            }
        }
        if r.drag_stopped_by(egui::PointerButton::Primary) {
            self.roi_draft_start = None;
            self.state.features = None; // ROI changed → seeds are stale
            self.invalidate_tracking(); // …and so is any tracking result
            self.events.push(plugins::PluginEvent::RoiChanged);
        }
    }

    /// Corners for the in-progress drag of the active drag-tool (Rect default, Circle if
    /// selected), or `None` when the drag is too small to be a deliberate ROI.
    fn drag_corners(&self, start: egui::Pos2, cur: egui::Pos2) -> Option<Vec<(f64, f64)>> {
        match self.tool {
            Tool::RoiCircle => circle_corners(start, cur),
            _ => rect_corners(start, cur),
        }
    }

    /// N-Gon tool: each left-click appends a vertex (starting a fresh polygon when the previous
    /// one was closed); a right-click closes it once it has ≥ `Roi::MIN_CORNERS`; Esc abandons an
    /// in-progress polygon. Mirrors `NGonTool`.
    fn handle_ngon(&mut self, tf: &canvas::Transform, r: &egui::Response) {
        if r.clicked_by(egui::PointerButton::Primary) {
            if let Some(p) = r.interact_pointer_pos() {
                let img = tf.screen_to_image(p);
                // No open polygon in progress → start a fresh one (and drop stale seeds/result).
                if self.state.roi.as_ref().is_none_or(|roi| roi.closed) {
                    self.state.roi = Some(Roi::default());
                    self.state.features = None;
                    self.invalidate_tracking();
                }
                if let Some(roi) = self.state.roi.as_mut() {
                    roi.add_corner(img.x as f64, img.y as f64);
                }
            }
        }
        if r.clicked_by(egui::PointerButton::Secondary) {
            enum Close {
                Done(usize),
                TooFew,
            }
            // Decide + mutate the ROI under one borrow, then update status after it ends.
            let act = match self.state.roi.as_mut() {
                Some(roi) if roi.closed => None,
                Some(roi) if roi.corners.len() >= Roi::MIN_CORNERS => {
                    roi.close();
                    Some(Close::Done(roi.corners.len()))
                }
                Some(_) => Some(Close::TooFew),
                None => None,
            };
            match act {
                Some(Close::Done(n)) => {
                    self.state.features = None; // ROI changed → seeds + tracking are stale
                    self.invalidate_tracking();
                    self.status = format!("ROI closed ({n} points).");
                    self.events.push(plugins::PluginEvent::RoiChanged);
                }
                Some(Close::TooFew) => {
                    self.status =
                        format!("Need at least {} points to close the ROI.", Roi::MIN_CORNERS);
                }
                None => {}
            }
        }
        // Esc abandons an in-progress (still-open) polygon.
        if r.ctx.input(|i| i.key_pressed(egui::Key::Escape))
            && self.state.roi.as_ref().is_some_and(|roi| !roi.closed)
        {
            self.state.roi = None;
        }
    }
}

impl eframe::App for EcmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_track_job(ctx);
        self.process_plugin_events();

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open Folder…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.open_dir(dir);
                    }
                }
                if self.state.has_sequence() {
                    ui.separator();
                    // Circle/N-Gon have no bundled icon → text-only (geometric-shape glyph kept).
                    if tool_button(ui, ctx, &mut self.icons, "hand", "Pan", self.tool == Tool::Pan, false, true).clicked() {
                        self.tool = Tool::Pan;
                    }
                    if tool_button(ui, ctx, &mut self.icons, "frame", "Rect ROI", self.tool == Tool::RoiRect, false, true).clicked() {
                        self.tool = Tool::RoiRect;
                    }
                    if tool_button(ui, ctx, &mut self.icons, "circle", "◯ Circle ROI", self.tool == Tool::RoiCircle, false, true).clicked() {
                        self.tool = Tool::RoiCircle;
                    }
                    if tool_button(ui, ctx, &mut self.icons, "ngon", "△ N-Gon ROI", self.tool == Tool::RoiNgon, false, true).clicked() {
                        self.tool = Tool::RoiNgon;
                    }
                    if tool_button(ui, ctx, &mut self.icons, "square-x", "Clear ROI", false, false, true).clicked() {
                        self.state.roi = None;
                        self.state.features = None;
                        self.invalidate_tracking();
                        self.events.push(plugins::PluginEvent::RoiChanged);
                    }
                    ui.separator();
                    if tool_button(ui, ctx, &mut self.icons, "scan", "Corners", false, false, true).clicked() {
                        self.detect_corners();
                    }
                    if tool_button(ui, ctx, &mut self.icons, "grid", "Grid", false, false, true).clicked() {
                        self.detect_grid();
                    }
                    ui.separator();
                    let can_track = self
                        .state
                        .features
                        .as_ref()
                        .is_some_and(|f| !f.is_empty())
                        && self.track_job.is_none();
                    if tool_button(ui, ctx, &mut self.icons, "play", "Run", false, true, can_track).clicked() {
                        self.run_tracking();
                    }
                    if self.state.result.is_some() {
                        let enabled = self.track_job.is_none();
                        if tool_button(ui, ctx, &mut self.icons, "trash", "Clear Tracking", false, false, enabled).clicked() {
                            self.clear_tracking();
                        }
                        let open = self.cleanup.is_some();
                        if tool_button(ui, ctx, &mut self.icons, "sliders", "Cleanup", open, false, true).clicked() {
                            if open {
                                self.cleanup = None;
                            } else {
                                self.open_cleanup();
                            }
                        }
                    }
                    let can_export = self.state.result.is_some()
                        && self
                            .state
                            .active_mask
                            .as_ref()
                            .is_some_and(|m| m.iter().any(|&b| b));
                    if can_export {
                        ui.menu_button("Export", |ui| {
                            if ui.button("Save as .npy…").clicked() {
                                self.export_coords(false);
                                ui.close_menu();
                            }
                            if ui.button("Save as .csv…").clicked() {
                                self.export_coords(true);
                                ui.close_menu();
                            }
                        });
                    }
                    ui.separator();
                    ui.menu_button("⚙ Params", |ui| {
                        if ui.button("Corner Detection…").clicked() {
                            self.dialog = Some(Dialog::Corner);
                            ui.close_menu();
                        }
                        if ui.button("Grid…").clicked() {
                            self.dialog = Some(Dialog::Grid);
                            ui.close_menu();
                        }
                        if ui.button("Tracker…").clicked() {
                            self.dialog = Some(Dialog::Tracker);
                            ui.close_menu();
                        }
                        if ui.button("Display…").clicked() {
                            self.dialog = Some(Dialog::Display);
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Plugins", |ui| {
                        // Discover lazily on first open (boots embedded Python + imports plugins).
                        if self.plugins.is_none() {
                            self.plugins = Some(plugins::discover());
                        }
                        if !self.overlay_plugins.is_empty() {
                            if ui.button("Clear plugins").clicked() {
                                self.clear_plugins();
                                ui.close_menu();
                            }
                            ui.separator();
                        }
                        let mut launch_idx = None;
                        {
                            let records = self.plugins.as_ref().unwrap();
                            if records.is_empty() {
                                ui.add_enabled(false, egui::Button::new("(no plugins found)"));
                            }
                            for (i, record) in records.iter().enumerate() {
                                // Loaded plugins are clickable; failed ones are greyed with the
                                // load error as a tooltip.
                                let resp = ui.add_enabled(
                                    record.cls.is_some(),
                                    egui::Button::new(record.name.as_str()),
                                );
                                if let Some(err) = &record.error {
                                    resp.on_hover_text(format!("Failed to load: {err}"));
                                } else {
                                    let resp = if record.description.is_empty() {
                                        resp
                                    } else {
                                        resp.on_hover_text(record.description.as_str())
                                    };
                                    if resp.clicked() {
                                        launch_idx = Some(i);
                                        ui.close_menu();
                                    }
                                }
                            }
                        }
                        if let Some(i) = launch_idx {
                            self.launch_plugin(i);
                        }
                    });
                    ui.separator();
                    if tool_button(ui, ctx, &mut self.icons, "maximize", "Fit", false, false, true).clicked() {
                        self.canvas.reset();
                    }
                    ui.label(format!(
                        "{:.0}% · frame {}/{}",
                        self.canvas.zoom * 100.0,
                        self.state.current_index + 1,
                        self.state.total_images()
                    ));
                }
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            if self.state.total_images() > 1 {
                let total = self.state.total_images();
                let mut cur = self.state.current_index;
                if ui
                    .add(egui::Slider::new(&mut cur, 0..=total - 1).text("frame"))
                    .changed()
                {
                    self.state.set_current(cur as i64);
                    self.events
                        .push(plugins::PluginEvent::FrameChanged(self.state.current_index));
                }
            }
            ui.label(&self.status);
        });

        // Cleanup side panel: per-metric band filters with a live green/red preview on the canvas.
        // Added before the CentralPanel so the canvas fills the remaining width.
        if self.cleanup.is_some() && self.state.result.is_none() {
            self.cleanup = None; // defensive: the session must never outlive its result
        }
        let mut cleanup_action = CleanupAction::None;
        if self.cleanup.is_some() {
            egui::SidePanel::right("cleanup_panel")
                .default_width(300.0)
                .show(ctx, |ui| {
                    let cs = self.cleanup.as_mut().unwrap();
                    ui.heading("Cleanup — filter tracks");
                    ui.label(
                        "Enable a metric and set its [min, max] keep-band. Points outside any \
                         enabled band are dropped. Preview updates live; Apply commits.",
                    );
                    ui.separator();

                    egui::Grid::new("cleanup_grid")
                        .num_columns(3)
                        .spacing([8.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Metric").strong());
                            ui.label(egui::RichText::new("On").strong());
                            ui.label(egui::RichText::new("Max").strong());
                            ui.end_row();

                            let caps = cs.caps;
                            let band_row = |ui: &mut egui::Ui,
                                            label: &str,
                                            band: &mut cleanup::BandFilter,
                                            cap: f64| {
                                let cap = cap.max(1.0);
                                ui.label(label);
                                ui.checkbox(&mut band.enabled, "");
                                ui.add(
                                    egui::DragValue::new(&mut band.hi)
                                        .speed(cap / 200.0)
                                        .range(0.0..=cap),
                                );
                                ui.end_row();
                            };
                            let t = &mut cs.thresholds;
                            band_row(ui, BAND_LABELS[0], &mut t.fw_failures, caps[0]);
                            band_row(ui, BAND_LABELS[1], &mut t.bw_failures, caps[1]);
                            band_row(ui, BAND_LABELS[2], &mut t.opencv_error, caps[2]);
                            band_row(ui, BAND_LABELS[3], &mut t.mean_error, caps[3]);
                            band_row(ui, BAND_LABELS[4], &mut t.fb_mean, caps[4]);
                            band_row(ui, BAND_LABELS[5], &mut t.fb_max, caps[5]);
                            band_row(ui, BAND_LABELS[6], &mut t.distance, caps[6]);
                        });

                    ui.separator();
                    ui.checkbox(
                        &mut cs.thresholds.drop_left_image,
                        "Drop points that left the image",
                    );
                    ui.checkbox(
                        &mut cs.thresholds.drop_left_roi,
                        "Drop points that left the ROI",
                    );
                    ui.separator();

                    // Recompute the preview keep-mask from the (possibly just-edited) thresholds,
                    // so both the survivor count below and the canvas overlay reflect this frame.
                    cs.preview_keep = cleanup::build_mask(&cs.metrics, &cs.thresholds);

                    let (kept, total) = match self.state.active_mask.as_deref() {
                        Some(active) => {
                            let total = active.iter().filter(|&&a| a).count();
                            let kept = active
                                .iter()
                                .zip(&cs.preview_keep)
                                .filter(|(&a, &k)| a && k)
                                .count();
                            (kept, total)
                        }
                        None => (0, 0),
                    };
                    ui.label(
                        egui::RichText::new(format!("{kept} of {total} active points kept")).strong(),
                    );
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() {
                            cleanup_action = CleanupAction::Apply;
                        }
                        let can_undo = !self.state.undo_stack.is_empty();
                        if ui.add_enabled(can_undo, egui::Button::new("Undo")).clicked() {
                            cleanup_action = CleanupAction::Undo;
                        }
                        if ui.button("Close").clicked() {
                            cleanup_action = CleanupAction::Close;
                        }
                    });
                });
        }
        match cleanup_action {
            CleanupAction::Apply => {
                if let Some(keep) = self.cleanup.as_ref().map(|cs| cs.preview_keep.clone()) {
                    self.apply_keep_mask(&keep);
                }
            }
            CleanupAction::Undo => {
                if let Some(prev) = self.state.undo_stack.pop() {
                    self.state.active_mask = Some(prev);
                    self.events.push(plugins::PluginEvent::MaskChanged);
                }
            }
            CleanupAction::Close => self.cleanup = None,
            CleanupAction::None => {}
        }

        // An unclosed ROI only exists mid-N-Gon; selecting any other tool abandons it
        // (mirrors `_cancel_roi_definition` on tool switch).
        if self.tool != Tool::RoiNgon
            && self.state.roi.as_ref().is_some_and(|roi| !roi.closed)
        {
            self.state.roi = None;
        }

        self.ensure_texture(ctx);
        // Clone the cheap texture handle so the canvas closure doesn't borrow `self.tex`
        // while `self.canvas` is borrowed mutably.
        let tex = self.tex.as_ref().map(|(_, h)| (h.clone(), h.size()));
        let allow_pan = self.tool == Tool::Pan;
        egui::CentralPanel::default().show(ctx, |ui| {
            let t = tex.as_ref().map(|(h, sz)| (h, *sz));
            let out = self.canvas.show(ui, t, allow_pan);
            if let Some(tf) = out.transform {
                self.handle_roi_interaction(&tf, &out.response);

                let painter = ui.painter_at(out.rect);
                let disp = self.state.display_params; // Copy; honors the Display dialog live
                if disp.show_roi {
                    if let Some(roi) = self.state.roi.as_ref() {
                        canvas::draw_roi(&painter, &tf, &roi.corners, roi.closed);
                    }
                }
                let marker_r = disp.marker_size.max(2) as f32;
                let alpha = (255 * disp.marker_opacity.clamp(0, 100) / 100) as u8;
                if disp.show_markers {
                    match self.state.result.as_ref() {
                        // No result yet: show the seed features on the reference frame.
                        None => {
                            if self.state.on_reference_frame() {
                                if let Some(feats) = self.state.features.as_ref() {
                                    canvas::draw_points(
                                        &painter,
                                        &tf,
                                        feats,
                                        egui::Color32::from_rgba_unmultiplied(0, 220, 220, alpha),
                                        marker_r,
                                    );
                                }
                            }
                        }
                        // Result present: tracked points at the current frame (green = kept).
                        Some(result) => {
                            let cut = self.state.global_to_cut(self.state.current_index);
                            if cut >= 0 && (cut as usize) < result.n_frames() {
                                let cut = cut as usize;
                                let pts: Vec<(f32, f32)> = (0..result.n_points())
                                    .map(|j| {
                                        (result.coords_fw[[cut, j, 0]], result.coords_fw[[cut, j, 1]])
                                    })
                                    .collect();
                                let preview =
                                    self.cleanup.as_ref().map(|cs| cs.preview_keep.as_slice());
                                // LK search window around each active point (gated on Display),
                                // drawn under the markers using the win_size recorded on the result.
                                if disp.show_window_box {
                                    canvas::draw_window_boxes(
                                        &painter,
                                        &tf,
                                        &pts,
                                        self.state.active_mask.as_deref(),
                                        result.win_size,
                                        alpha,
                                    );
                                }
                                canvas::draw_tracks(
                                    &painter,
                                    &tf,
                                    &pts,
                                    self.state.active_mask.as_deref(),
                                    preview,
                                    marker_r,
                                    alpha,
                                );
                            }
                        }
                    }
                }
                // Plugin overlays draw on top of the built-in markers (slice 3f).
                canvas::draw_overlay_commands(&painter, &tf, &self.overlay_cmds);
            }
        });

        if let Some(job) = self.track_job.as_ref() {
            let frac = job.done as f32 / job.total.max(1) as f32;
            // egui 0.30 has no `Modal`; a centered, non-collapsible Window is the progress
            // dialog. The Run/Clear buttons are disabled while a job runs (see the toolbar),
            // so this doesn't need to block input behind it.
            egui::Window::new("Tracking…")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_width(240.0);
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    ui.label(format!("{} / {} passes", job.done, job.total));
                    ui.add_space(4.0);
                    if ui.button("Cancel").clicked() {
                        job.cancel.store(true, Ordering::Relaxed);
                    }
                });
        }

        self.show_dialogs(ctx);
        self.show_panels(ctx);

        if self.smoke {
            eprintln!(
                "[smoke] frames={} frame_texture_uploaded={} features={} tracked_points={} status={:?}",
                self.state.total_images(),
                self.tex.is_some(),
                self.state.features.as_ref().map_or(0, Vec::len),
                self.state.result.as_ref().map_or(0, |r| r.n_points()),
                self.status,
            );
            // Isolate plugin settings I/O (slice 3g-d) to a throwaway config dir, so the smoke
            // neither reads nor pollutes the user's real settings.json and starts from empty.
            let smoke_cfg = std::env::temp_dir().join(format!("ecm_smoke_cfg_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&smoke_cfg);
            std::env::set_var("TRACKER_CONFIG_DIR", &smoke_cfg);

            // Exercise the plugin pipeline headlessly: discover, launch each loaded plugin, then
            // refresh any overlays they provide.
            let records = plugins::discover();
            let loaded: Vec<&str> = records
                .iter()
                .filter(|r| r.cls.is_some())
                .map(|r| r.name.as_str())
                .collect();
            eprintln!("[smoke] plugins_discovered={} loaded={loaded:?}", records.len());
            let mut overlays = Vec::new();
            for record in records.iter().filter(|r| r.cls.is_some()) {
                match plugins::launch(record, plugins::snapshot(&self.state)) {
                    Ok(outcome) => {
                        eprintln!("[smoke] launched '{}': {:?}", record.name, outcome.message);
                        if let Some(instance) = outcome.instance {
                            overlays.push(instance);
                        }
                    }
                    Err(e) => eprintln!("[smoke] launch '{}' err: {e}", record.name),
                }
            }
            let cmds = plugins::refresh_overlays(&overlays, &self.state);
            eprintln!(
                "[smoke] overlay_plugins={} draw_commands={}",
                overlays.len(),
                cmds.len()
            );
            // Shared active-point counter, and the full mask so each mutation demo starts fresh.
            let count = |s: &ProjectState| {
                s.active_mask.as_ref().map_or(0, |m| m.iter().filter(|&&b| b).count())
            };
            let full_mask = self.state.active_mask.clone();

            // Exercise the apply_keep_mask mutation (slice 3g-b): the decimate example records a
            // keep-mask in launch(); apply it through the real undoable path and report the drop.
            // (The launch-all loop above ignores keep masks, so the overlay counts stay at full P.)
            if let Some(rec) = records.iter().find(|r| r.id == "decimate_points") {
                let before = count(&self.state);
                if let Ok(outcome) = plugins::launch(rec, plugins::snapshot(&self.state)) {
                    if let Some(keep) = outcome.keep_mask {
                        self.apply_keep_mask(&keep);
                    }
                }
                eprintln!("[smoke] apply_keep_mask: n_active {before} -> {}", count(&self.state));
            }

            // Exercise the control-panel path (slice 3g-c): launch the panel example, collect its
            // declared controls, then simulate clicking its "apply" button — which records a
            // keep-mask via on_control → ctx.apply_keep_mask. Restore the full mask first so the
            // reported drop is clean (independent of the decimate demo above).
            self.state.active_mask = full_mask;
            if let Some(rec) = records.iter().find(|r| r.id == "panel_filter") {
                if let Ok(outcome) = plugins::launch(rec, plugins::snapshot(&self.state)) {
                    if let Some(instance) = outcome.instance {
                        self.overlay_plugins.push(instance);
                        let idx = self.overlay_plugins.len() - 1;
                        let controls =
                            plugins::panel_controls(&self.overlay_plugins[idx], &self.state);
                        let before = count(&self.state);
                        self.dispatch_control(idx, "apply", ControlValue::Click);
                        eprintln!(
                            "[smoke] panel controls={} apply_keep_mask: n_active {before} -> {}",
                            controls.len(),
                            count(&self.state)
                        );
                    }
                }
            }

            // Exercise settings persistence (slice 3g-d): change the panel filter's stride (its
            // on_control saves it via ctx.save_settings), then relaunch a FRESH instance and confirm
            // launch() restored that stride from disk (panel() seeds the slider from it).
            if let Some(rec) = records.iter().find(|r| r.id == "panel_filter") {
                if let Ok(o1) = plugins::launch(rec, plugins::snapshot(&self.state)) {
                    if let Some(i1) = o1.instance {
                        let slice1 = [i1];
                        plugins::dispatch_control(&slice1, 0, "stride", ControlValue::Float(4.0), &self.state);
                    }
                }
                let restored = match plugins::launch(rec, plugins::snapshot(&self.state)) {
                    Ok(o2) => o2.instance.map(|i2| plugins::panel_controls(&i2, &self.state)),
                    Err(_) => None,
                }
                .and_then(|controls| {
                    controls.iter().find_map(|c| match c {
                        Control::Slider { key, value, .. } if key == "stride" => Some(*value),
                        _ => None,
                    })
                });
                eprintln!("[smoke] settings: panel_filter restored stride={restored:?} (saved 4.0)");
            }
            let _ = std::fs::remove_dir_all(&smoke_cfg);
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
