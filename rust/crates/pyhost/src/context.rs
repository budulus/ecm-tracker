//! The `PluginContext` façade exposed to Python plugins — the Rust port of
//! `app/plugins/api.py:PluginContext`.
//!
//! A plugin reaches everything it's allowed to touch through one `PluginContext` instance. Because
//! embedded Python runs in-process and a pyo3 `#[pyclass]` must be `'static`, the context wraps an
//! **immutable snapshot** of the app state (taken when the context is built) rather than a live
//! borrow of `EcmApp`/`ProjectState`. Reads are served from the snapshot; the single mutation
//! (`apply_keep_mask`) will be a host-applied command in a later slice.
//!
//! Index convention (mirrors the Python SDK): every index a plugin passes/receives is a **global**
//! frame index (`0 .. n_total_images-1`); tracked-data arrays are **cut**-indexed (`0` = reference).
//! Convert with `global_to_cut` / `cut_to_global`.

use pyo3::prelude::*;

/// Immutable copy of the read-relevant app state, handed to a `PluginContext`. Plain data only
/// (no `ecm-core` types) so `pyhost` stays decoupled from the core/opencv stack — the GUI layer,
/// which owns both, assembles this from `ProjectState`.
#[derive(Clone, Debug, Default)]
pub struct ContextSnapshot {
    pub n_total_images: usize,
    pub has_sequence: bool,
    pub has_result: bool,
    /// `(height, width)` of the frames, if a sequence is loaded.
    pub image_size: Option<(usize, usize)>,
    pub reference_index: usize,
    pub last_index: usize,
    pub current_index: usize,
    /// Tracked-range frame count (`coords` axis 0); 0 if no result.
    pub frame_count: usize,
    /// Total tracked points P (before filtering); 0 if no result.
    pub point_count: usize,
    /// Currently-kept points (active-mask true count); 0 if no result.
    pub n_active: usize,
    /// ROI corner points `[(x, y), …]` in image coordinates (empty if no ROI).
    pub roi_corners: Vec<(f64, f64)>,
}

/// The façade a plugin uses to reach the app. One instance per plugin (passed as `self.ctx`);
/// treat everything as read-only. Backed by an immutable [`ContextSnapshot`].
#[pyclass]
pub struct PluginContext {
    snap: ContextSnapshot,
}

impl PluginContext {
    /// Build a context over `snap`. Constructed by the host (Rust side) — plugins never make one,
    /// so there is no `#[new]`.
    pub fn new(snap: ContextSnapshot) -> Self {
        Self { snap }
    }
}

#[pymethods]
impl PluginContext {
    // ---- session / status ----------------------------------------------
    /// True once an image sequence is loaded.
    #[getter]
    fn has_sequence(&self) -> bool {
        self.snap.has_sequence
    }

    /// True once tracking has produced a result.
    #[getter]
    fn has_result(&self) -> bool {
        self.snap.has_result
    }

    /// Number of frames in the full loaded folder.
    #[getter]
    fn n_total_images(&self) -> usize {
        self.snap.n_total_images
    }

    /// `(height, width)` of the frames, or `None` if no sequence is loaded.
    fn image_size(&self) -> Option<(usize, usize)> {
        self.snap.image_size
    }

    // ---- frame indices --------------------------------------------------
    /// Global index of the reference frame (cut 0).
    #[getter]
    fn reference_index(&self) -> usize {
        self.snap.reference_index
    }

    /// Global index of the last tracked frame.
    #[getter]
    fn last_index(&self) -> usize {
        self.snap.last_index
    }

    /// Global index of the frame currently shown on the canvas.
    #[getter]
    fn current_index(&self) -> usize {
        self.snap.current_index
    }

    /// Cut index of the current frame, or `None` if it is outside the tracked range.
    #[getter]
    fn current_cut(&self) -> Option<i64> {
        let cur = self.snap.current_index;
        if self.snap.reference_index <= cur && cur <= self.snap.last_index {
            Some(self.global_to_cut(cur as i64))
        } else {
            None
        }
    }

    /// Convert a global frame index to a cut index (`global - reference`).
    fn global_to_cut(&self, global_index: i64) -> i64 {
        global_index - self.snap.reference_index as i64
    }

    /// Convert a cut index to a global frame index (`reference + cut`).
    fn cut_to_global(&self, cut_index: i64) -> i64 {
        self.snap.reference_index as i64 + cut_index
    }

    // ---- tracked data (scalars; arrays arrive in slice 3b) --------------
    /// Number of frames in the tracked range (`coords` axis 0). 0 if no result.
    #[getter]
    fn frame_count(&self) -> usize {
        self.snap.frame_count
    }

    /// Total number of tracked points P (before filtering). 0 if no result.
    #[getter]
    fn point_count(&self) -> usize {
        self.snap.point_count
    }

    /// Number of currently-kept points.
    #[getter]
    fn n_active(&self) -> usize {
        self.snap.n_active
    }

    // ---- ROI ------------------------------------------------------------
    /// The ROI corner points `[(x, y), …]` in image coordinates (empty if no ROI).
    #[getter]
    fn roi_corners(&self) -> Vec<(f64, f64)> {
        self.snap.roi_corners.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ContextSnapshot {
        ContextSnapshot {
            n_total_images: 20,
            has_sequence: true,
            has_result: true,
            image_size: Some((480, 640)),
            reference_index: 2,
            last_index: 13,
            current_index: 5,
            frame_count: 12,
            point_count: 267,
            n_active: 200,
            roi_corners: vec![(10.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)],
        }
    }

    /// Drive the pyclass through Python's object protocol (getattr / call_method) to prove the
    /// read API is actually exposed to embedded Python, and that the snapshot/index math is right.
    #[test]
    fn context_read_api_from_python() {
        Python::attach(|py| {
            let ctx = Py::new(py, PluginContext::new(sample())).unwrap();
            let b = ctx.bind(py);

            let pc: usize = b.getattr("point_count").unwrap().extract().unwrap();
            assert_eq!(pc, 267);
            let na: usize = b.getattr("n_active").unwrap().extract().unwrap();
            assert_eq!(na, 200);
            let total: usize = b.getattr("n_total_images").unwrap().extract().unwrap();
            assert_eq!(total, 20);

            // Index conversions (global ↔ cut), reference = 2.
            let cut: i64 = b.call_method1("global_to_cut", (5,)).unwrap().extract().unwrap();
            assert_eq!(cut, 3);
            let g: i64 = b.call_method1("cut_to_global", (3,)).unwrap().extract().unwrap();
            assert_eq!(g, 5);
            // current_index 5 is inside [2, 13] → current_cut = 3.
            let cc: Option<i64> = b.getattr("current_cut").unwrap().extract().unwrap();
            assert_eq!(cc, Some(3));

            let size: (usize, usize) = b.call_method0("image_size").unwrap().extract().unwrap();
            assert_eq!(size, (480, 640));

            let corners: Vec<(f64, f64)> = b.getattr("roi_corners").unwrap().extract().unwrap();
            assert_eq!(corners.len(), 4);
            assert_eq!(corners[1], (110.0, 10.0));
        });
    }

    /// current_cut is None when the current frame is outside the tracked range.
    #[test]
    fn current_cut_none_out_of_range() {
        let mut snap = sample();
        snap.current_index = 18; // > last_index (13)
        Python::attach(|py| {
            let ctx = Py::new(py, PluginContext::new(snap)).unwrap();
            let cc: Option<i64> = ctx.bind(py).getattr("current_cut").unwrap().extract().unwrap();
            assert_eq!(cc, None);
        });
    }
}
