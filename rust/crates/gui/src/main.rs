//! ECM Tracker — egui host application (Phase 2).
//!
//! Slice 1: open a folder of images, display the current frame on a zoom/pan canvas, and
//! scrub frames. Builds on the Phase 1 `ecm-core` pipeline (ProjectState + ImageSequence).
//! Toolbar groups, ROI tools, detection/tracking, overlays, and the cleanup panel follow.
//!
//! `ECM_SMOKE=1` exits after the first frame (headless build check). `ECM_SMOKE_DIR=<dir>`
//! additionally loads that folder first, so the smoke run exercises the decode→texture path.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console in release

mod canvas;

use canvas::CanvasView;
use ecm_core::image_sequence::{discover_dir, ImageSequence};
use ecm_core::project_state::ProjectState;
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

struct EcmApp {
    state: ProjectState,
    canvas: CanvasView,
    /// (frame index currently uploaded, texture handle).
    tex: Option<(usize, egui::TextureHandle)>,
    status: String,
    smoke: bool,
}

impl EcmApp {
    fn new(smoke: bool) -> Self {
        let mut app = Self {
            state: ProjectState::new(),
            canvas: CanvasView::default(),
            tex: None,
            status: "Open a folder of images to begin.".into(),
            smoke,
        };
        if let Some(dir) = std::env::var_os("ECM_SMOKE_DIR") {
            app.open_dir(PathBuf::from(dir));
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
                    ui.label(format!(
                        "Frame {}/{}",
                        self.state.current_index + 1,
                        self.state.total_images()
                    ));
                    ui.separator();
                    if ui.button("Fit").clicked() {
                        self.canvas.reset();
                    }
                    ui.label(format!("{:.0}%", self.canvas.zoom * 100.0));
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
        egui::CentralPanel::default().show(ctx, |ui| {
            let t = tex.as_ref().map(|(h, sz)| (h, *sz));
            self.canvas.show(ui, t);
        });

        if self.smoke {
            eprintln!(
                "[smoke] frames={} frame_texture_uploaded={} status={:?}",
                self.state.total_images(),
                self.tex.is_some(),
                self.status,
            );
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
