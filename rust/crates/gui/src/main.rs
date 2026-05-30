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

use canvas::CanvasView;
use ecm_core::feature_detection;
use ecm_core::image_sequence::{discover_dir, ImageSequence};
use ecm_core::project_state::ProjectState;
use ecm_core::result::TrackerResult;
use ecm_core::roi::Roi;
use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

fn main() -> eframe::Result<()> {
    let smoke = std::env::var_os("ECM_SMOKE").is_some();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("ECM Tracker"),
        ..Default::default()
    };
    eframe::run_native(
        "ECM Tracker",
        options,
        Box::new(move |_cc| Ok(Box::new(EcmApp::new(smoke)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Pan,
    RoiRect,
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

struct EcmApp {
    state: ProjectState,
    canvas: CanvasView,
    tool: Tool,
    /// Image-space start corner while dragging a rectangular ROI.
    roi_draft_start: Option<egui::Pos2>,
    /// (frame index currently uploaded, texture handle).
    tex: Option<(usize, egui::TextureHandle)>,
    /// In-flight background tracking job (progress + cancel), if any.
    track_job: Option<TrackJob>,
    status: String,
    smoke: bool,
}

/// Axis-aligned rectangle ROI from two image-space corners.
fn rect_roi(a: egui::Pos2, b: egui::Pos2) -> Roi {
    Roi::new(vec![
        (a.x as f64, a.y as f64),
        (b.x as f64, a.y as f64),
        (b.x as f64, b.y as f64),
        (a.x as f64, b.y as f64),
    ])
}

impl EcmApp {
    fn new(smoke: bool) -> Self {
        let mut app = Self {
            state: ProjectState::new(),
            canvas: CanvasView::default(),
            tool: Tool::Pan,
            roi_draft_start: None,
            tex: None,
            track_job: None,
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

    fn open_dir(&mut self, dir: PathBuf) {
        match discover_dir(&dir) {
            Ok(paths) if !paths.is_empty() => match ImageSequence::new(paths) {
                Ok(seq) => {
                    let n = seq.len();
                    self.state.load_sequence(seq, dir.to_str().map(str::to_string));
                    self.canvas.reset();
                    self.tex = None;
                    self.status = format!("Loaded {n} frames from {}", dir.display());
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

    /// Discard any tracking result (and its mask/undo). Seeds and ROI are left untouched.
    /// Called whenever the inputs to tracking (seeds/ROI) change so stale tracks aren't shown.
    fn invalidate_tracking(&mut self) {
        self.state.result = None;
        self.state.active_mask = None;
        self.state.undo_stack.clear();
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
                self.track_job = None;
                self.status = format!("Tracked {np} points over {nf} frames.");
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

    /// Handle ROI rectangle drawing on the canvas (image-space) for the current frame.
    fn handle_roi_draw(&mut self, tf: &canvas::Transform, r: &egui::Response) {
        if self.tool != Tool::RoiRect {
            return;
        }
        if r.drag_started_by(egui::PointerButton::Primary) {
            self.roi_draft_start = r.interact_pointer_pos().map(|p| tf.screen_to_image(p));
        }
        if r.dragged_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(p)) = (self.roi_draft_start, r.interact_pointer_pos()) {
                self.state.roi = Some(rect_roi(start, tf.screen_to_image(p)));
            }
        }
        if r.drag_stopped_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(p)) = (self.roi_draft_start, r.interact_pointer_pos()) {
                let cur = tf.screen_to_image(p);
                if (cur.x - start.x).abs() < 3.0 || (cur.y - start.y).abs() < 3.0 {
                    self.state.roi = None; // reject degenerate drag
                }
            }
            self.roi_draft_start = None;
            self.state.features = None; // ROI changed → seeds are stale
            self.invalidate_tracking(); // …and so is any tracking result
        }
    }
}

impl eframe::App for EcmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_track_job(ctx);

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open Folder…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.open_dir(dir);
                    }
                }
                if self.state.has_sequence() {
                    ui.separator();
                    ui.selectable_value(&mut self.tool, Tool::Pan, "✋ Pan");
                    ui.selectable_value(&mut self.tool, Tool::RoiRect, "▭ Rect ROI");
                    if ui.button("Clear ROI").clicked() {
                        self.state.roi = None;
                        self.state.features = None;
                        self.invalidate_tracking();
                    }
                    ui.separator();
                    if ui.button("Detect Corners").clicked() {
                        self.detect_corners();
                    }
                    ui.separator();
                    let can_track = self
                        .state
                        .features
                        .as_ref()
                        .is_some_and(|f| !f.is_empty())
                        && self.track_job.is_none();
                    if ui
                        .add_enabled(can_track, egui::Button::new("▶ Run Tracking"))
                        .clicked()
                    {
                        self.run_tracking();
                    }
                    if self.state.result.is_some()
                        && ui
                            .add_enabled(
                                self.track_job.is_none(),
                                egui::Button::new("Clear Tracking"),
                            )
                            .clicked()
                    {
                        self.clear_tracking();
                    }
                    ui.separator();
                    if ui.button("Fit").clicked() {
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
                }
            }
            ui.label(&self.status);
        });

        self.ensure_texture(ctx);
        // Clone the cheap texture handle so the canvas closure doesn't borrow `self.tex`
        // while `self.canvas` is borrowed mutably.
        let tex = self.tex.as_ref().map(|(_, h)| (h.clone(), h.size()));
        let allow_pan = self.tool == Tool::Pan;
        egui::CentralPanel::default().show(ctx, |ui| {
            let t = tex.as_ref().map(|(h, sz)| (h, *sz));
            let out = self.canvas.show(ui, t, allow_pan);
            if let Some(tf) = out.transform {
                self.handle_roi_draw(&tf, &out.response);

                let painter = ui.painter_at(out.rect);
                if let Some(roi) = self.state.roi.as_ref() {
                    canvas::draw_roi(&painter, &tf, &roi.corners, roi.closed);
                }
                let marker_r = self.state.display_params.marker_size.max(2) as f32;
                match self.state.result.as_ref() {
                    // No result yet: show the seed features on the reference frame.
                    None => {
                        if self.state.on_reference_frame() {
                            if let Some(feats) = self.state.features.as_ref() {
                                canvas::draw_points(
                                    &painter,
                                    &tf,
                                    feats,
                                    egui::Color32::from_rgb(0, 220, 220), // cyan seeds
                                    marker_r,
                                );
                            }
                        }
                    }
                    // Result present: show the tracked points at the current frame (green = kept).
                    Some(result) => {
                        let cut = self.state.global_to_cut(self.state.current_index);
                        if cut >= 0 && (cut as usize) < result.n_frames() {
                            let cut = cut as usize;
                            let pts: Vec<(f32, f32)> = (0..result.n_points())
                                .map(|j| {
                                    (result.coords_fw[[cut, j, 0]], result.coords_fw[[cut, j, 1]])
                                })
                                .collect();
                            canvas::draw_tracks(
                                &painter,
                                &tf,
                                &pts,
                                self.state.active_mask.as_deref(),
                                marker_r,
                            );
                        }
                    }
                }
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

        if self.smoke {
            eprintln!(
                "[smoke] frames={} frame_texture_uploaded={} features={} tracked_points={} status={:?}",
                self.state.total_images(),
                self.tex.is_some(),
                self.state.features.as_ref().map_or(0, Vec::len),
                self.state.result.as_ref().map_or(0, |r| r.n_points()),
                self.status,
            );
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
