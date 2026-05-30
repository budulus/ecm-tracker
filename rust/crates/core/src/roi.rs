//! Polygonal region of interest (corners in image coordinates).
//! Ports `app/core/roi.py`.

use opencv::core::{Mat, Point2f, Point2i, Scalar, Vector, CV_8UC1};
use opencv::imgproc;

/// A polygonal ROI defined by N corners in image coordinates. Constructing with
/// `corners` yields an already-closed polygon (how the shape tools/tests build one).
#[derive(Clone, Debug, Default)]
pub struct Roi {
    pub corners: Vec<(f64, f64)>,
    pub closed: bool,
}

impl Roi {
    pub const MIN_CORNERS: usize = 3;

    /// Already-closed ROI from a full corner list.
    pub fn new(corners: Vec<(f64, f64)>) -> Self {
        let closed = !corners.is_empty();
        Self { corners, closed }
    }

    /// Closed *and* at least a triangle (guards degenerate 1–2 corner polygons).
    pub fn is_complete(&self) -> bool {
        self.closed && self.corners.len() >= Self::MIN_CORNERS
    }

    pub fn add_corner(&mut self, x: f64, y: f64) {
        if !self.closed {
            self.corners.push((x, y));
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn reset(&mut self) {
        self.corners.clear();
        self.closed = false;
    }

    fn contour(&self) -> Vector<Point2f> {
        self.corners
            .iter()
            .map(|&(x, y)| Point2f::new(x as f32, y as f32))
            .collect()
    }

    fn polygon_int(&self) -> Vector<Point2i> {
        self.corners
            .iter()
            .map(|&(x, y)| Point2i::new(x as i32, y as i32))
            .collect()
    }

    /// Binary (0/255) uint8 mask of the filled polygon at image resolution.
    pub fn mask(&self, height: i32, width: i32) -> opencv::Result<Mat> {
        let mut m = Mat::new_rows_cols_with_default(height, width, CV_8UC1, Scalar::all(0.0))?;
        if self.is_complete() {
            let mut polys = Vector::<Vector<Point2i>>::new();
            polys.push(self.polygon_int());
            imgproc::fill_poly_def(&mut m, &polys, Scalar::all(255.0))?;
        }
        Ok(m)
    }

    pub fn contains(&self, x: f64, y: f64) -> opencv::Result<bool> {
        if !self.is_complete() {
            return Ok(false);
        }
        let d = imgproc::point_polygon_test(&self.contour(), Point2f::new(x as f32, y as f32), false)?;
        Ok(d >= 0.0)
    }

    /// Boolean mask over `pts` that lie inside a *complete* ROI (all-false otherwise).
    pub fn contains_many(&self, pts: &[(f64, f64)]) -> opencv::Result<Vec<bool>> {
        if !self.is_complete() {
            return Ok(vec![false; pts.len()]);
        }
        let contour = self.contour();
        let mut out = Vec::with_capacity(pts.len());
        for &(x, y) in pts {
            let d = imgproc::point_polygon_test(&contour, Point2f::new(x as f32, y as f32), false)?;
            out.push(d >= 0.0);
        }
        Ok(out)
    }
}
