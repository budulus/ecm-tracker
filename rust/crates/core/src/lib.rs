//! ECM Tracker core (Rust port of `app/core/` + `app/models/`).
//!
//! Mirrors the dependency rule of the Python core: pure logic, no GUI, no Python
//! bindings — so the whole pipeline can be exercised headlessly.
//!
//! Phase 1 port complete: image_sequence, roi, feature_detection, tracking, cleanup,
//! export, settings, the TrackerResult/LkParams data types, and the ProjectState
//! dual-index model.

pub mod cleanup;
pub mod export;
pub mod feature_detection;
pub mod image_sequence;
pub mod project_state;
pub mod result;
pub mod roi;
pub mod settings;
pub mod tracking;

/// Crate version (sanity check that the workspace links).
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Linked OpenCV runtime version — confirms the `opencv` crate links + loads.
pub fn opencv_version() -> opencv::Result<String> {
    opencv::core::get_version_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(super::version(), "0.1.0");
    }

    #[test]
    fn opencv_links() {
        let v = super::opencv_version().expect("get OpenCV version");
        assert!(v.starts_with("4."), "unexpected OpenCV version: {v}");
    }
}
