//! Toolbar SVG icons. The Lucide SVGs in `app/gui/icons/` are embedded at compile time
//! (`include_str!`), rendered with resvg + tiny-skia, recolored by alpha (the glyphs are authored
//! black, so we keep coverage-alpha and replace RGB with the tint), and uploaded as egui textures.
//! Ports `app/gui/icon_loader.py` — the Qt `SourceIn` recolor, now an egui texture cache.

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use resvg::{tiny_skia, usvg};
use std::collections::HashMap;

/// Default toolbar glyph color (mirrors `icon_loader.NORMAL`).
pub const NORMAL: Color32 = Color32::from_rgb(0x3a, 0x3f, 0x44);
/// Glyph on the accent "Run" button (mirrors `icon_loader.ACCENT`).
pub const ACCENT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);

/// Render resolution (px) of toolbar glyphs — 2× the ~18px display size for crisp edges.
const RENDER_PX: u32 = 36;

/// The embedded SVG source for a toolbar icon name, or `None` if we don't bundle one (callers
/// fall back to a text-only button). Paths are relative to this source file → repo `app/gui/icons`.
fn icon_svg(name: &str) -> Option<&'static str> {
    Some(match name {
        "hand" => include_str!("../../../../app/gui/icons/hand.svg"),
        "frame" => include_str!("../../../../app/gui/icons/frame.svg"),
        "square-x" => include_str!("../../../../app/gui/icons/square-x.svg"),
        "scan" => include_str!("../../../../app/gui/icons/scan.svg"),
        "grid" => include_str!("../../../../app/gui/icons/grid.svg"),
        "play" => include_str!("../../../../app/gui/icons/play.svg"),
        "trash" => include_str!("../../../../app/gui/icons/trash.svg"),
        "sliders" => include_str!("../../../../app/gui/icons/sliders.svg"),
        "download" => include_str!("../../../../app/gui/icons/download.svg"),
        "maximize" => include_str!("../../../../app/gui/icons/maximize.svg"),
        "eye" => include_str!("../../../../app/gui/icons/eye.svg"),
        _ => return None,
    })
}

/// Rasterize `svg` to a `px`×`px` RGBA pixmap, or `None` if it can't be parsed/allocated.
fn rasterize(svg: &str, px: u32) -> Option<tiny_skia::Pixmap> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).ok()?;
    let mut pixmap = tiny_skia::Pixmap::new(px, px)?;
    let scale = px as f32 / tree.size().width().max(1.0);
    resvg::render(&tree, tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    Some(pixmap)
}

/// Render `svg` and recolor every pixel to `color`, preserving the rendered alpha (Qt `SourceIn`).
fn render_tinted(svg: &str, color: Color32, px: u32) -> ColorImage {
    let mut img = ColorImage::new([px as usize, px as usize], Color32::TRANSPARENT);
    let Some(pixmap) = rasterize(svg, px) else {
        return img; // unparsable → blank glyph; the button still functions
    };
    let [r, g, b, _] = color.to_array();
    for (dst, src) in img.pixels.iter_mut().zip(pixmap.pixels()) {
        *dst = Color32::from_rgba_unmultiplied(r, g, b, src.alpha());
    }
    img
}

/// Render the full-color `app_icon.svg` as unmultiplied RGBA for the OS window/taskbar icon
/// (no recolor — the SVG's own colors are kept). `None` if it can't be rendered.
pub fn app_icon(px: u32) -> Option<egui::IconData> {
    let pixmap = rasterize(include_str!("../../../../app/gui/icons/app_icon.svg"), px)?;
    let mut rgba = Vec::with_capacity((px * px * 4) as usize);
    for p in pixmap.pixels() {
        let c = p.demultiply(); // tiny-skia stores premultiplied; egui wants straight alpha
        rgba.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    Some(egui::IconData { rgba, width: px, height: px })
}

/// Caches rendered + tinted icon textures by (name, color). Held by the app for its lifetime.
#[derive(Default)]
pub struct IconStore {
    cache: HashMap<(&'static str, [u8; 4]), Option<TextureHandle>>,
}

impl IconStore {
    /// A `display_px`-sized egui image for `name` tinted `color`, or `None` when no such icon is
    /// bundled (caller uses a text-only button). Renders + uploads once per (name, color).
    pub fn image(
        &mut self,
        ctx: &egui::Context,
        name: &'static str,
        color: Color32,
        display_px: f32,
    ) -> Option<egui::Image<'static>> {
        let handle = self
            .cache
            .entry((name, color.to_array()))
            .or_insert_with(|| {
                let svg = icon_svg(name)?;
                let img = render_tinted(svg, color, RENDER_PX);
                let [r, g, b, a] = color.to_array();
                let tex_name = format!("icon:{name}:{r:02x}{g:02x}{b:02x}{a:02x}");
                Some(ctx.load_texture(tex_name, img, TextureOptions::LINEAR))
            })
            .as_ref()?;
        let src = egui::load::SizedTexture::new(handle.id(), egui::vec2(display_px, display_px));
        Some(egui::Image::new(src))
    }
}
