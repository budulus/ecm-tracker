//! App-wide light theme for the egui host.
//!
//! Translates the Qt `LIGHT_QSS` palette (`app/gui/theme.py`) to egui `Visuals`. egui has no
//! stylesheet, so this is a faithful **palette** port (window/surface/border/text/accent +
//! flat-until-hover buttons + an accent selection), not a 1:1 reproduction of every QSS rule.
//! The canvas viewport keeps its own dark frame background — the theme only styles egui widgets.

use eframe::egui::{self, Color32, Rounding, Stroke};

// Palette — mirrors `theme.py`: window #f4f5f7, surface #ffffff, border #d6dade,
// text #2b2f33, muted #8a9099, accent #2563eb (+ hover/pressed/checked tints).
const WINDOW: Color32 = Color32::from_rgb(0xf4, 0xf5, 0xf7);
const SURFACE: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
const BORDER: Color32 = Color32::from_rgb(0xd6, 0xda, 0xde);
const TEXT: Color32 = Color32::from_rgb(0x2b, 0x2f, 0x33);
const ACCENT: Color32 = Color32::from_rgb(0x25, 0x63, 0xeb);
const HOVER: Color32 = Color32::from_rgb(0xee, 0xf1, 0xf6);
const PRESSED: Color32 = Color32::from_rgb(0xe3, 0xe8, 0xf0);
const CHECKED_BG: Color32 = Color32::from_rgb(0xe7, 0xef, 0xff);
const CHECKED_BORDER: Color32 = Color32::from_rgb(0xb9, 0xd2, 0xff);

/// Apply the light theme to the egui context once at startup (mirrors `theme.apply_theme`).
pub fn apply(ctx: &egui::Context) {
    let mut v = egui::Visuals::light();

    v.panel_fill = WINDOW; // toolbar / central / side panels
    v.window_fill = SURFACE; // dialog (egui Window) bodies
    v.extreme_bg_color = SURFACE; // DragValue / text-edit field background
    v.faint_bg_color = WINDOW; // striped-grid alternate rows (cleanup panel)
    v.hyperlink_color = ACCENT;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_rounding = Rounding::same(10.0);

    // Selected toggle (Button::selected / selectable): light-blue fill + accent border,
    // matching `QToolButton:checked`.
    v.selection.bg_fill = CHECKED_BG;
    v.selection.stroke = Stroke::new(1.0, CHECKED_BORDER);

    let rounding = Rounding::same(8.0);
    let w = &mut v.widgets;

    // Labels, separators, status text. bg_stroke is the separator/divider color.
    w.noninteractive.bg_fill = WINDOW;
    w.noninteractive.weak_bg_fill = WINDOW;
    w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    w.noninteractive.rounding = rounding;

    // Buttons at rest: flat (transparent fill), dark text — like `QToolButton { background: transparent }`.
    w.inactive.bg_fill = Color32::TRANSPARENT;
    w.inactive.weak_bg_fill = Color32::TRANSPARENT;
    w.inactive.bg_stroke = Stroke::NONE;
    w.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    w.inactive.rounding = rounding;

    w.hovered.bg_fill = HOVER;
    w.hovered.weak_bg_fill = HOVER;
    w.hovered.bg_stroke = Stroke::new(1.0, BORDER);
    w.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    w.hovered.rounding = rounding;

    w.active.bg_fill = PRESSED;
    w.active.weak_bg_fill = PRESSED;
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    w.active.fg_stroke = Stroke::new(1.0, TEXT);
    w.active.rounding = rounding;

    // Open menus / combo popups.
    w.open.bg_fill = SURFACE;
    w.open.weak_bg_fill = HOVER;
    w.open.bg_stroke = Stroke::new(1.0, BORDER);
    w.open.fg_stroke = Stroke::new(1.0, TEXT);
    w.open.rounding = rounding;

    ctx.set_visuals(v);
}
