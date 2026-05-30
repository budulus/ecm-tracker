//! Seed-point generation: Shi-Tomasi corners or a regular grid clipped to the ROI.
//! Ports `app/core/feature_detection.py`.

use crate::roi::Roi;
use opencv::core::{Mat, Point2f, Vector};
use opencv::imgproc;
use serde::{Deserialize, Serialize};

/// Shi-Tomasi / Harris corner parameters (`DEFAULT_SHI_TOMASI`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShiTomasiParams {
    pub max_corners: i32,
    pub quality_level: f64,
    pub min_distance: f64,
    pub block_size: i32,
    pub use_harris_detector: bool,
    pub k: f64,
}

impl Default for ShiTomasiParams {
    fn default() -> Self {
        Self {
            max_corners: 500,
            quality_level: 0.01,
            min_distance: 7.0,
            block_size: 7,
            use_harris_detector: false,
            k: 0.04,
        }
    }
}

/// Regular-grid spacing parameters (`DEFAULT_GRID`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GridParams {
    pub spacing_x: i32,
    pub spacing_y: i32,
}

impl Default for GridParams {
    fn default() -> Self {
        Self { spacing_x: 20, spacing_y: 20 }
    }
}

/// Shi-Tomasi / Harris corners inside `mask` (None = whole image). Returns image-space
/// `(x, y)` points.
pub fn shi_tomasi(
    gray: &Mat,
    mask: Option<&Mat>,
    params: &ShiTomasiParams,
) -> opencv::Result<Vec<(f32, f32)>> {
    let mut corners = Vector::<Point2f>::new();
    let empty = Mat::default();
    let mask_arr: &Mat = mask.unwrap_or(&empty); // empty Mat == noArray() == no mask
    imgproc::good_features_to_track(
        gray,
        &mut corners,
        params.max_corners,
        params.quality_level,
        params.min_distance,
        mask_arr,
        params.block_size,
        params.use_harris_detector,
        params.k,
    )?;
    Ok(corners.iter().map(|p| (p.x, p.y)).collect())
}

/// `np.arange(start, stop, step)` — count = ceil((stop-start)/step), values start + i*step.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = (((stop - start) / step).ceil()).max(0.0) as usize;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Grid of points `spacing_x`/`spacing_y` apart, clipped to the ROI polygon.
pub fn regular_grid(roi: &Roi, spacing_x: f64, spacing_y: f64) -> opencv::Result<Vec<(f32, f32)>> {
    if !roi.is_complete() || spacing_x <= 0.0 || spacing_y <= 0.0 {
        return Ok(Vec::new());
    }
    let x0 = roi.corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let x1 = roi.corners.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
    let y0 = roi.corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let y1 = roi.corners.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);

    let xs = arange(x0, x1 + 1e-6, spacing_x);
    let ys = arange(y0, y1 + 1e-6, spacing_y);

    let mut pts = Vec::new();
    for &y in &ys {
        for &x in &xs {
            if roi.contains(x, y)? {
                pts.push((x as f32, y as f32));
            }
        }
    }
    Ok(pts)
}
