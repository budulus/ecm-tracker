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
use ecm_core::roi::Roi;
use eframe::egui;
use std::path::PathBuf;

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

struct EcmApp {
    state: ProjectState,
    canvas: CanvasView,
    tool: Tool,
    /// Image-space start corner while dragging a rectangular ROI.
    roi_draft_start: Option<egui::Pos2>,
    /// (frame index currently uploaded, texture handle).
    tex: Option<(usize, egui::TextureHandle)>,
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
            status: "Open a folder of images to begin.".into(),
            smoke,
        };
        if let Some(dir) = std::env::var_os("ECM_SMOKE_DIR") {
            app.open_dir(PathBuf::from(dir));
            app.detect_corners(); // exercise the detection path in the smoke check
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
                self.state.features = Some(pts);
            }
            Err(e) => self.status = format!("Detect error: {e}"),
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
        }
    }
}

impl eframe::App for EcmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                    }
                    ui.separator();
                    if ui.button("Detect Corners").clicked() {
                        self.detect_corners();
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
                if self.state.on_reference_frame() {
                    if let Some(feats) = self.state.features.as_ref() {
                        canvas::draw_points(
                            &painter,
                            &tf,
                            feats,
                            egui::Color32::from_rgb(0, 220, 220), // cyan seeds
                            3.0,
                        );
                    }
                }
            }
        });

        if self.smoke {
            eprintln!(
                "[smoke] frames={} frame_texture_uploaded={} features={} status={:?}",
                self.state.total_images(),
                self.tex.is_some(),
                self.state.features.as_ref().map_or(0, Vec::len),
                self.status,
            );
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
