//! ECM Tracker core (Rust port target).
//!
//! Mirrors the dependency rule of the Python `app/core/`: pure logic, no GUI,
//! no Python bindings — so the whole pipeline can be exercised headlessly.
//! Phase 1 fills this in: image_sequence, roi, feature_detection, tracking,
//! cleanup, export, settings, plus the dual global/cut index model.

use opencv::core::{Mat, Scalar, CV_8UC3};
use opencv::imgproc;
use opencv::prelude::*;

/// Crate version, used by the GUI shell to confirm the workspace links.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Linked OpenCV runtime version — proves the `opencv` crate links + loads.
pub fn opencv_version() -> opencv::Result<String> {
    opencv::core::get_version_string()
}

/// Smoke: build a 4x4 BGR `Mat` and run `cvtColor` to gray (one of the 8
/// OpenCV functions the Python core uses). Returns the gray Mat's (rows, cols).
pub fn opencv_cvt_smoke() -> opencv::Result<(i32, i32)> {
    let src = Mat::new_rows_cols_with_default(4, 4, CV_8UC3, Scalar::all(0.0))?;
    let mut gray = Mat::default();
    imgproc::cvt_color_def(&src, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok((gray.rows(), gray.cols()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::version(), "0.1.0");
    }

    #[test]
    fn opencv_links_and_runs() {
        let v = super::opencv_version().expect("get OpenCV version");
        println!("OpenCV version: {v}");
        assert!(v.starts_with("4."), "unexpected OpenCV version: {v}");

        let (rows, cols) = super::opencv_cvt_smoke().expect("cvtColor smoke");
        assert_eq!((rows, cols), (4, 4));
    }
}
