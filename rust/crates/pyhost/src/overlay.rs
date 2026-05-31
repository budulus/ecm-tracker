//! The plugin overlay draw-command API (Phase 3 slice 3f).
//!
//! Replaces the Qt `QPainter` handed to overlays in the Python app: a plugin defines an
//! `overlay(self, painter)` method and draws onto the host-provided [`OverlayPainter`], which
//! accumulates [`DrawCommand`]s in **image coordinates**. The host (GUI) reads the commands out and
//! renders them on the egui canvas (positions transformed image→screen; radii/widths/text sizes are
//! constant screen pixels, matching the built-in overlays). This keeps Python free of any live Rust
//! painter handle — the plugin only ever touches the `OverlayPainter` pyclass.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An RGBA color, 0–255 per channel.
#[derive(Clone, Copy, Debug)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// A stroke: width in **screen** pixels (constant regardless of zoom) + color.
#[derive(Clone, Copy, Debug)]
pub struct Stroke {
    pub width: f64,
    pub color: Rgba,
}

/// One overlay primitive, with geometry in **image** coordinates. Radii/widths/sizes are screen px.
#[derive(Clone, Debug)]
pub enum DrawCommand {
    /// Circle at `center`, `radius` screen px, optionally filled and/or stroked.
    Circle {
        center: (f64, f64),
        radius: f64,
        fill: Option<Rgba>,
        stroke: Option<Stroke>,
    },
    /// Open polyline through `points` (a `line` is the 2-point case).
    Polyline {
        points: Vec<(f64, f64)>,
        stroke: Stroke,
    },
    /// Closed polygon, optionally filled. (Fill assumes a convex polygon, like egui.)
    Polygon {
        points: Vec<(f64, f64)>,
        fill: Option<Rgba>,
        stroke: Option<Stroke>,
    },
    /// Text anchored top-left at `pos`, `size` screen px.
    Text {
        pos: (f64, f64),
        text: String,
        color: Rgba,
        size: f64,
    },
}

/// Convert a Python `(r, g, b)` or `(r, g, b, a)` sequence into an [`Rgba`] (alpha defaults to 255).
fn rgba_from(v: &[u8]) -> PyResult<Rgba> {
    match *v {
        [r, g, b] => Ok(Rgba { r, g, b, a: 255 }),
        [r, g, b, a] => Ok(Rgba { r, g, b, a }),
        _ => Err(PyValueError::new_err(
            "color must be (r, g, b) or (r, g, b, a)",
        )),
    }
}

/// The drawing surface handed to a plugin's `overlay(self, painter)`. Each method appends a
/// [`DrawCommand`]; the host reads [`OverlayPainter::commands`] after the call. Coordinates are in
/// image space; colors are `(r, g, b)` / `(r, g, b, a)` sequences; widths/radii/sizes are screen px.
#[pyclass]
#[derive(Default)]
pub struct OverlayPainter {
    pub commands: Vec<DrawCommand>,
}

#[pymethods]
impl OverlayPainter {
    /// Draw a circle at `center` (image coords), `radius` screen px. `fill`/`stroke` are colors
    /// (omit for none); `width` is the stroke width in screen px.
    #[pyo3(signature = (center, radius=4.0, fill=None, stroke=None, width=1.0))]
    fn circle(
        &mut self,
        center: (f64, f64),
        radius: f64,
        fill: Option<Vec<u8>>,
        stroke: Option<Vec<u8>>,
        width: f64,
    ) -> PyResult<()> {
        let fill = fill.as_deref().map(rgba_from).transpose()?;
        let stroke = stroke
            .as_deref()
            .map(rgba_from)
            .transpose()?
            .map(|color| Stroke { width, color });
        self.commands.push(DrawCommand::Circle {
            center,
            radius,
            fill,
            stroke,
        });
        Ok(())
    }

    /// Draw a line segment `a`→`b` (image coords) with `color`, `width` screen px.
    #[pyo3(signature = (a, b, color, width=1.0))]
    fn line(&mut self, a: (f64, f64), b: (f64, f64), color: Vec<u8>, width: f64) -> PyResult<()> {
        let color = rgba_from(&color)?;
        self.commands.push(DrawCommand::Polyline {
            points: vec![a, b],
            stroke: Stroke { width, color },
        });
        Ok(())
    }

    /// Draw an open polyline through `points` (image coords) with `color`, `width` screen px.
    #[pyo3(signature = (points, color, width=1.0))]
    fn polyline(&mut self, points: Vec<(f64, f64)>, color: Vec<u8>, width: f64) -> PyResult<()> {
        let color = rgba_from(&color)?;
        self.commands.push(DrawCommand::Polyline {
            points,
            stroke: Stroke { width, color },
        });
        Ok(())
    }

    /// Draw a closed polygon through `points` (image coords). `fill`/`stroke` are colors (omit for
    /// none); `width` is the outline width in screen px.
    #[pyo3(signature = (points, fill=None, stroke=None, width=1.0))]
    fn polygon(
        &mut self,
        points: Vec<(f64, f64)>,
        fill: Option<Vec<u8>>,
        stroke: Option<Vec<u8>>,
        width: f64,
    ) -> PyResult<()> {
        let fill = fill.as_deref().map(rgba_from).transpose()?;
        let stroke = stroke
            .as_deref()
            .map(rgba_from)
            .transpose()?
            .map(|color| Stroke { width, color });
        self.commands.push(DrawCommand::Polygon {
            points,
            fill,
            stroke,
        });
        Ok(())
    }

    /// Draw `text` anchored top-left at `pos` (image coords), `size` screen px, in `color`.
    #[pyo3(signature = (pos, text, color, size=12.0))]
    fn text(&mut self, pos: (f64, f64), text: String, color: Vec<u8>, size: f64) -> PyResult<()> {
        let color = rgba_from(&color)?;
        self.commands.push(DrawCommand::Text {
            pos,
            text,
            color,
            size,
        });
        Ok(())
    }
}
