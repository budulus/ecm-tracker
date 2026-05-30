//! ECM Tracker — egui host application (Phase 0 skeleton).
//!
//! A minimal eframe window. The real toolbar/canvas/dialogs arrive in Phase 2;
//! for now this proves egui builds and opens a window on Windows + macOS.
//!
//! Setting the `ECM_SMOKE` env var closes the window right after the first
//! rendered frame, so CI / headless checks can confirm the window + GL context
//! initialize without a human closing it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // no console in release

use eframe::egui;

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
        Box::new(move |_cc| Ok(Box::new(EcmApp { smoke }))),
    )
}

struct EcmApp {
    smoke: bool,
}

impl eframe::App for EcmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("ECM Tracker");
            ui.label(format!(
                "Rust + egui host — Phase 0 skeleton (ecm-core v{})",
                ecm_core::version()
            ));
        });

        // In smoke mode, exit cleanly once the first frame has been drawn.
        if self.smoke {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
