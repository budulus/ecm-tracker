//! Frame canvas: image→screen transform (fit-to-rect → zoom → pan), texture draw, and
//! scroll-zoom / drag-pan. Ports the geometry model of `app/gui/canvas_view.py` to egui.
//!
//! Overlay geometry is stored in *image* coordinates and mapped through `Transform` at paint
//! time, so markers/handles stay constant-size under zoom (same approach as the Qt canvas).

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, Vec2};

/// Maps image pixels ↔ screen pixels for the current view.
#[derive(Clone, Copy)]
pub struct Transform {
    pub origin: Pos2, // screen position of image pixel (0, 0)
    pub scale: f32,   // image px → screen px
}

impl Transform {
    pub fn image_to_screen(&self, x: f32, y: f32) -> Pos2 {
        self.origin + Vec2::new(x * self.scale, y * self.scale)
    }
    pub fn screen_to_image(&self, p: Pos2) -> Pos2 {
        let d = p - self.origin;
        Pos2::new(d.x / self.scale, d.y / self.scale)
    }
}

/// What `CanvasView::show` produced: the widget rect, the active transform (None when no
/// frame is shown), and the interaction response (for tool-specific image-space input).
pub struct CanvasOut {
    pub rect: Rect,
    pub transform: Option<Transform>,
    pub response: egui::Response,
}

/// Pan/zoom view state for the canvas.
pub struct CanvasView {
    pub zoom: f32,
    pub pan: Vec2, // screen-space offset applied after centering
}

impl Default for CanvasView {
    fn default() -> Self {
        Self { zoom: 1.0, pan: Vec2::ZERO }
    }
}

impl CanvasView {
    /// Reset to fit-to-window (zoom 1.0, no pan).
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }

    /// Aspect-preserving fit, scaled by zoom, centered in `rect`, then panned.
    fn transform(&self, rect: Rect, img_w: f32, img_h: f32) -> Transform {
        let fit = (rect.width() / img_w).min(rect.height() / img_h);
        let scale = fit * self.zoom;
        let disp = Vec2::new(img_w * scale, img_h * scale);
        let origin = rect.center() - disp * 0.5 + self.pan;
        Transform { origin, scale }
    }

    /// Draw the frame texture (if any) with the current view and handle pan/zoom.
    /// `allow_primary_pan` lets the active tool claim left-drag (e.g. ROI drawing); middle/right
    /// drag always pans. Returns the rect + transform + response for overlays and tools.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        tex: Option<(&TextureHandle, [usize; 2])>,
        allow_primary_pan: bool,
    ) -> CanvasOut {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(30));

        let Some((handle, [iw, ih])) = tex else {
            return CanvasOut { rect, transform: None, response };
        };
        let (img_w, img_h) = (iw as f32, ih as f32);

        // Pan: middle/right drag always; left drag only when the tool doesn't claim it.
        let pan = response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
            || (allow_primary_pan && response.dragged_by(egui::PointerButton::Primary));
        if pan {
            self.pan += response.drag_delta();
        }

        // Scroll to zoom, keeping the image point under the cursor fixed.
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(cursor) = response.hover_pos() {
                    let img_pt = self.transform(rect, img_w, img_h).screen_to_image(cursor);
                    self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.05, 50.0);
                    let new_screen =
                        self.transform(rect, img_w, img_h).image_to_screen(img_pt.x, img_pt.y);
                    self.pan += cursor - new_screen;
                }
            }
        }

        let t = self.transform(rect, img_w, img_h);
        let img_rect = Rect::from_min_size(t.origin, Vec2::new(img_w * t.scale, img_h * t.scale));
        painter.image(
            handle.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        CanvasOut { rect, transform: Some(t), response }
    }
}

const ROI_OUTLINE: Color32 = Color32::from_rgb(255, 215, 0); // gold
const ROI_VERTEX: Color32 = Color32::from_rgb(255, 0, 255); // magenta

/// Draw an ROI polygon (image-space corners) in screen space: outline + vertex handles.
/// `closed` joins the last corner back to the first.
pub fn draw_roi(painter: &egui::Painter, t: &Transform, corners: &[(f64, f64)], closed: bool) {
    if corners.is_empty() {
        return;
    }
    let pts: Vec<Pos2> = corners
        .iter()
        .map(|&(x, y)| t.image_to_screen(x as f32, y as f32))
        .collect();
    let stroke = Stroke::new(2.0, ROI_OUTLINE);
    let edges = if closed { pts.len() } else { pts.len().saturating_sub(1) };
    for i in 0..edges {
        painter.line_segment([pts[i], pts[(i + 1) % pts.len()]], stroke);
    }
    for p in &pts {
        painter.circle_filled(*p, 4.0, ROI_VERTEX);
    }
}

/// Draw a set of image-space points as constant-size filled markers (e.g. seed features).
pub fn draw_points(painter: &egui::Painter, t: &Transform, pts: &[(f32, f32)], color: Color32, radius: f32) {
    for &(x, y) in pts {
        painter.circle_filled(t.image_to_screen(x, y), radius, color);
    }
}

/// Draw the LK search window (a `win_size`-px square in *image* space, so it scales with zoom)
/// around each active, finite tracked point. Mirrors the optional window-box overlay in
/// `CanvasView._draw_tracked` (gated on `display_params.show_window_box`): green, alpha-matched
/// to the markers, color independent of the cleanup preview.
pub fn draw_window_boxes(
    painter: &egui::Painter,
    t: &Transform,
    pts: &[(f32, f32)],
    active: Option<&[bool]>,
    win_size: i32,
    alpha: u8,
) {
    let stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 220, 0, alpha)); // green
    let half = win_size as f32 * 0.5;
    for (i, &(x, y)) in pts.iter().enumerate() {
        if let Some(mask) = active {
            if !mask.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        // Map the image-space box corners through the transform (zoom-aware), then outline it.
        let tl = t.image_to_screen(x - half, y - half);
        let tr = t.image_to_screen(x + half, y - half);
        let br = t.image_to_screen(x + half, y + half);
        let bl = t.image_to_screen(x - half, y + half);
        painter.line_segment([tl, tr], stroke);
        painter.line_segment([tr, br], stroke);
        painter.line_segment([br, bl], stroke);
        painter.line_segment([bl, tl], stroke);
    }
}

/// Draw tracked points at the current frame. Skips points masked out by `active` and any with
/// non-finite coordinates (failed tracks). Mirrors `CanvasView._draw_tracked`: among the active
/// points, a cleanup `preview` keep-mask colors survivors green and would-be drops red; with no
/// preview everything kept is green. `alpha` (0–255) is the Display dialog's marker opacity.
pub fn draw_tracks(
    painter: &egui::Painter,
    t: &Transform,
    pts: &[(f32, f32)],
    active: Option<&[bool]>,
    preview: Option<&[bool]>,
    radius: f32,
    alpha: u8,
) {
    let kept = Color32::from_rgba_unmultiplied(0, 200, 0, alpha); // green
    let drop_color = Color32::from_rgba_unmultiplied(255, 60, 60, alpha); // red = will drop
    for (i, &(x, y)) in pts.iter().enumerate() {
        if let Some(mask) = active {
            if !mask.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let drop = preview.is_some_and(|pv| !pv.get(i).copied().unwrap_or(true));
        painter.circle_filled(t.image_to_screen(x, y), radius, if drop { drop_color } else { kept });
    }
}
