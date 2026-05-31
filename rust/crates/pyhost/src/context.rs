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

use ndarray::{Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Indices (into the full P points) where `mask` is true, in order — the kept-point columns.
fn kept_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect()
}

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
    /// Absolute paths to every frame in the full loaded folder, global-indexed (`[global_i]`).
    /// Empty if no sequence is loaded. Lets native-window plugins load the frame image to draw on
    /// (the paths-only, Qt-free equivalent of the old `api.py` `frame_rgb`).
    pub image_paths: Vec<String>,
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

    /// Forward tracked coordinates `(frames, P, 2)` f32, cut-indexed (`[0]` = reference). `None`
    /// if no result.
    pub coords_fw: Option<Array3<f32>>,
    /// Forward per-frame tracking status `(frames, P)` u8 (1 = tracked OK, 0 = carried forward).
    /// `None` if no result.
    pub status_fw: Option<Array2<u8>>,
    /// Active (keep) mask `(P,)`: true where a point survived cleanup. `None` if no result.
    pub active_mask: Option<Vec<bool>>,

    /// The plugin's persisted settings section (`plugin:<id>`) as a compact JSON-object string, or
    /// `None` if it has saved none. The GUI loads it via `core::settings::get_section`; `get_settings`
    /// parses it back to a dict. It rides in the snapshot (rather than on `PluginContext`) because
    /// the host rebuilds the context from a fresh snapshot on every call (slice 3g-d).
    pub settings_json: Option<String>,
}

/// The façade a plugin uses to reach the app. One instance per plugin (passed as `self.ctx`);
/// treat everything as read-only. Backed by an immutable [`ContextSnapshot`].
#[pyclass]
pub struct PluginContext {
    snap: ContextSnapshot,
    /// Keep-mask recorded by `apply_keep_mask` during a plugin call (full length P). The host
    /// takes it after the call and applies it through its undoable mask path (slice 3g-b). The
    /// context itself stays read-only — this is just an outbox, not a live mutation.
    pending_keep: Option<Vec<bool>>,
    /// Settings dict recorded by `save_settings` during a plugin call, serialized to a JSON string.
    /// The host takes it after the call and persists it via `core::settings` (slice 3g-d) — the same
    /// collect-then-apply outbox as `pending_keep`, keeping the context read-only.
    pending_settings: Option<String>,
}

impl PluginContext {
    /// Build a context over `snap`. Constructed by the host (Rust side) — plugins never make one,
    /// so there is no `#[new]`.
    pub fn new(snap: ContextSnapshot) -> Self {
        Self { snap, pending_keep: None, pending_settings: None }
    }

    /// Take the keep-mask recorded by `apply_keep_mask` during the last plugin call (full length
    /// P), clearing it. Called host-side after a plugin method returns.
    pub(crate) fn take_pending_keep(&mut self) -> Option<Vec<bool>> {
        self.pending_keep.take()
    }

    /// Take the settings JSON recorded by `save_settings` during the last plugin call, clearing it.
    /// Called host-side after a plugin method returns; the GUI persists it via `core::settings`.
    pub(crate) fn take_pending_settings(&mut self) -> Option<String> {
        self.pending_settings.take()
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

    /// Absolute filesystem path of the frame at `global_i`, or `None` if no sequence is loaded or
    /// the index is out of range. Native-window plugins use it to load the image to draw on.
    fn frame_path(&self, global_i: usize) -> Option<String> {
        self.snap.image_paths.get(global_i).cloned()
    }

    /// All frame image paths in the full loaded folder, global-indexed (empty if no sequence).
    fn sequence_paths(&self) -> Vec<String> {
        self.snap.image_paths.clone()
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

    // ---- tracked-data arrays (rust-numpy) -------------------------------
    /// Tracked forward coordinates as an `(frames, points, 2)` float32 array. With `active_only`
    /// (default True) only kept points are returned (point axis aligns with `point_indices`);
    /// False returns all P points. `None` if no result. Cut-indexed on axis 0 (`coords[0]` =
    /// reference frame).
    #[pyo3(signature = (active_only = true))]
    fn coords<'py>(&self, py: Python<'py>, active_only: bool) -> Option<Bound<'py, PyArray3<f32>>> {
        let coords = self.snap.coords_fw.as_ref()?;
        match (active_only, self.snap.active_mask.as_ref()) {
            (true, Some(mask)) => Some(coords.select(Axis(1), &kept_indices(mask)).into_pyarray(py)),
            _ => Some(coords.to_pyarray(py)),
        }
    }

    /// Forward per-frame tracking status as an `(frames, points)` uint8 array, aligned to
    /// `coords` (1 = tracked OK, 0 = LK failed and the position was carried forward). Same
    /// `active_only` semantics as `coords`; `None` if no result.
    #[pyo3(signature = (active_only = true))]
    fn track_status<'py>(
        &self,
        py: Python<'py>,
        active_only: bool,
    ) -> Option<Bound<'py, PyArray2<u8>>> {
        let status = self.snap.status_fw.as_ref()?;
        match (active_only, self.snap.active_mask.as_ref()) {
            (true, Some(mask)) => Some(status.select(Axis(1), &kept_indices(mask)).into_pyarray(py)),
            _ => Some(status.to_pyarray(py)),
        }
    }

    /// The `(P,)` bool keep-mask: True where a point survived cleanup. `None` if no result.
    /// Read-only — change it via `apply_keep_mask` (a later slice).
    #[getter]
    fn active_mask<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<bool>>> {
        Some(self.snap.active_mask.as_ref()?.to_pyarray(py))
    }

    /// Original column indices (into the full P points) of the kept points, matching the point
    /// axis of `coords(active_only=True)`. `None` if no result. (`np.where(active_mask)[0]`.)
    fn point_indices<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        let mask = self.snap.active_mask.as_ref()?;
        let idx: Vec<i64> = mask
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i as i64)
            .collect();
        Some(idx.into_pyarray(py))
    }

    // ---- the one mutation: record a keep-mask (host applies it) ----------
    /// Filter the active points by a boolean keep-mask (the single plugin mutation). `keep` is a
    /// bool sequence/array of length `point_count` (all P points) or `n_active` (current kept
    /// points only); points marked False leave the active set. The context is a read-only
    /// snapshot, so this only *records* the mask — the host applies it through its undoable mask
    /// path after the plugin call returns (batched + undoable). No-op if there is no result.
    /// Raises `ValueError` if the length is neither P nor n_active. Mirrors
    /// `app/plugins/api.py:PluginContext.apply_keep_mask` (which we ravel to 1-D, as it assumes).
    fn apply_keep_mask(&mut self, py: Python<'_>, keep: &Bound<'_, PyAny>) -> PyResult<()> {
        // No result → nothing to filter (matches the Python no-op).
        let Some(active) = self.snap.active_mask.as_ref() else {
            return Ok(());
        };
        // Coerce to a 1-D bool array: np.asarray(keep).astype(bool).ravel().
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (keep,))?;
        let arr = arr.call_method1("astype", ("bool",))?;
        let arr = arr.call_method0("ravel")?;
        let flat: Vec<bool> = arr.extract::<PyReadonlyArray1<bool>>()?.as_array().to_vec();

        let p = self.snap.point_count;
        let full = if flat.len() == p {
            flat
        } else if flat.len() == self.snap.n_active {
            // Expand an n_active-length mask to full length P, placing values at the kept columns
            // (inactive points stay False — they're dropped by the host's `active & keep` anyway).
            let mut full = vec![false; p];
            for (slot, &k) in kept_indices(active).into_iter().zip(flat.iter()) {
                full[slot] = k;
            }
            full
        } else {
            return Err(PyValueError::new_err(format!(
                "keep mask length {} is neither P={} nor n_active={}",
                flat.len(),
                p,
                self.snap.n_active
            )));
        };
        self.pending_keep = Some(full);
        Ok(())
    }

    // ---- per-plugin settings (persistence) ------------------------------
    /// Return this plugin's persisted settings as a dict, or an empty dict if it has saved none.
    /// Mirrors `app/plugins/api.py:PluginContext.get_settings` (`settings.get_section(...) or {}`).
    /// The host loads the plugin's `plugin:<id>` section into the snapshot's `settings_json`; this
    /// parses it. Returns a fresh dict each call — mutate it then call `save_settings` to persist.
    fn get_settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.snap.settings_json.as_deref() {
            Some(s) => py.import("json")?.call_method1("loads", (s,)),
            None => Ok(PyDict::new(py).into_any()),
        }
    }

    /// Persist this plugin's settings (a JSON-serializable dict), overwriting any previous. Mirrors
    /// `save_settings(data)` (`settings.update_section(...)`). Because the context is a read-only
    /// snapshot, this only *records* the dict (as a JSON string) into an outbox; the host writes it
    /// through `core::settings` after the plugin call returns (collect-then-apply, like
    /// `apply_keep_mask`). Raises if `data` isn't JSON-serializable (Python's `json.dumps` does).
    fn save_settings(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let dumped: String = py.import("json")?.call_method1("dumps", (data,))?.extract()?;
        self.pending_settings = Some(dumped);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};

    fn sample() -> ContextSnapshot {
        ContextSnapshot {
            n_total_images: 20,
            has_sequence: true,
            has_result: true,
            image_size: Some((480, 640)),
            image_paths: vec![
                "/seq/frame_000.png".to_string(),
                "/seq/frame_001.png".to_string(),
                "/seq/frame_002.png".to_string(),
            ],
            reference_index: 2,
            last_index: 13,
            current_index: 5,
            frame_count: 12,
            point_count: 267,
            n_active: 200,
            roi_corners: vec![(10.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)],
            coords_fw: None,
            status_fw: None,
            active_mask: None,
            settings_json: None,
        }
    }

    /// A small, self-consistent "has result" snapshot for the array API: 3 frames × 4 points,
    /// mask keeps columns [0, 2, 3] (n_active 3). `coords[f,p,c] = f*100 + p*10 + c` so values
    /// are easy to check after column selection.
    fn sample_with_arrays() -> ContextSnapshot {
        let coords = Array3::from_shape_fn((3, 4, 2), |(f, p, c)| (f * 100 + p * 10 + c) as f32);
        let status = Array2::from_shape_fn((3, 4), |(f, p)| ((f + p) % 2) as u8);
        ContextSnapshot {
            n_total_images: 8,
            has_sequence: true,
            image_paths: vec![],
            has_result: true,
            image_size: Some((48, 64)),
            reference_index: 0,
            last_index: 2,
            current_index: 1,
            frame_count: 3,
            point_count: 4,
            n_active: 3,
            roi_corners: vec![],
            coords_fw: Some(coords),
            status_fw: Some(status),
            active_mask: Some(vec![true, false, true, true]),
            settings_json: None,
        }
    }

    /// Drive the pyclass through Python's object protocol (getattr / call_method) to prove the
    /// read API is actually exposed to embedded Python, and that the snapshot/index math is right.
    #[test]
    fn context_read_api_from_python() {
        let _g = crate::interp_test_lock();
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
        let _g = crate::interp_test_lock();
        let mut snap = sample();
        snap.current_index = 18; // > last_index (13)
        Python::attach(|py| {
            let ctx = Py::new(py, PluginContext::new(snap)).unwrap();
            let cc: Option<i64> = ctx.bind(py).getattr("current_cut").unwrap().extract().unwrap();
            assert_eq!(cc, None);
        });
    }

    /// frame_path / sequence_paths expose the global-indexed image paths to Python; an out-of-range
    /// frame_path is None.
    #[test]
    fn image_paths_from_python() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            let ctx = Py::new(py, PluginContext::new(sample())).unwrap();
            let b = ctx.bind(py);

            let p1: String = b.call_method1("frame_path", (1,)).unwrap().extract().unwrap();
            assert_eq!(p1, "/seq/frame_001.png");
            let oob: Option<String> =
                b.call_method1("frame_path", (99,)).unwrap().extract().unwrap();
            assert_eq!(oob, None);

            let all: Vec<String> = b.call_method0("sequence_paths").unwrap().extract().unwrap();
            assert_eq!(all.len(), 3);
            assert_eq!(all[0], "/seq/frame_000.png");
        });
    }

    /// coords / track_status / active_mask / point_indices return correctly-shaped NumPy arrays,
    /// and active_only selects the kept columns ([0, 2, 3]) in order.
    #[test]
    fn tracked_arrays_from_python() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            // Self-sufficient: ensure numpy is importable (rust-numpy's lazy array-API init imports
            // it) rather than relying on another test having added the embedded site to sys.path.
            crate::ensure_embedded_site(py).unwrap();
            let ctx = Py::new(py, PluginContext::new(sample_with_arrays())).unwrap();
            let b = ctx.bind(py);

            // Full coords: (3, 4, 2).
            let full: PyReadonlyArray3<f32> =
                b.call_method1("coords", (false,)).unwrap().extract().unwrap();
            assert_eq!(full.as_array().shape(), &[3, 4, 2]);

            // Active coords: (3, 3, 2), kept columns [0, 2, 3]. Column 1 of the active array is the
            // original point 2 → coords[0, 2, c] = 0*100 + 2*10 + c = (20, 21) at frame 0.
            let act: PyReadonlyArray3<f32> =
                b.call_method1("coords", (true,)).unwrap().extract().unwrap();
            let a = act.as_array();
            assert_eq!(a.shape(), &[3, 3, 2]);
            assert_eq!(a[[0, 1, 0]], 20.0);
            assert_eq!(a[[0, 1, 1]], 21.0);

            // Active status: (3, 3).
            let st: PyReadonlyArray2<u8> =
                b.call_method1("track_status", (true,)).unwrap().extract().unwrap();
            assert_eq!(st.as_array().shape(), &[3, 3]);

            // active_mask: (4,) bool, unfiltered.
            let m: PyReadonlyArray1<bool> = b.getattr("active_mask").unwrap().extract().unwrap();
            assert_eq!(m.as_array().to_vec(), vec![true, false, true, true]);

            // point_indices: the kept columns.
            let pi: PyReadonlyArray1<i64> =
                b.call_method0("point_indices").unwrap().extract().unwrap();
            assert_eq!(pi.as_array().to_vec(), vec![0_i64, 2, 3]);
        });
    }

    /// With no result, every array accessor returns Python None.
    #[test]
    fn no_result_arrays_are_none() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            let ctx = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let b = ctx.bind(py);
            assert!(b.call_method1("coords", (true,)).unwrap().is_none());
            assert!(b.call_method1("track_status", (false,)).unwrap().is_none());
            assert!(b.getattr("active_mask").unwrap().is_none());
            assert!(b.call_method0("point_indices").unwrap().is_none());
        });
    }

    /// apply_keep_mask records a full-length keep-mask: the n_active-length path expands at the
    /// kept columns, the P-length path passes through, a wrong length raises ValueError, and a
    /// no-result context records nothing. The recorded mask is taken (and cleared) host-side.
    #[test]
    fn apply_keep_mask_records_and_validates() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            crate::ensure_embedded_site(py).unwrap(); // apply_keep_mask uses numpy
            // P=4, n_active=3, active = [T, F, T, T] → kept columns [0, 2, 3].
            let ctx = Py::new(py, PluginContext::new(sample_with_arrays())).unwrap();
            let b = ctx.bind(py);

            // n_active-length (3) mask [T, F, T] expands at slots 0,2,3 → full P [T, F, F, T].
            b.call_method1("apply_keep_mask", (vec![true, false, true],)).unwrap();
            assert_eq!(b.borrow_mut().take_pending_keep(), Some(vec![true, false, false, true]));
            // Taken once, it's cleared.
            assert_eq!(b.borrow_mut().take_pending_keep(), None);

            // P-length (4) mask passes straight through.
            b.call_method1("apply_keep_mask", (vec![false, true, true, false],)).unwrap();
            assert_eq!(b.borrow_mut().take_pending_keep(), Some(vec![false, true, true, false]));

            // A length that is neither P nor n_active raises ValueError.
            assert!(b.call_method1("apply_keep_mask", (vec![true, false],)).is_err());

            // No result → no-op (records nothing).
            let empty = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let eb = empty.bind(py);
            eb.call_method1("apply_keep_mask", (vec![true],)).unwrap();
            assert_eq!(eb.borrow_mut().take_pending_keep(), None);
        });
    }

    /// get_settings parses the snapshot's `settings_json` into a dict (and returns an empty dict
    /// when there is none); save_settings records the dict as a JSON string into the outbox, which
    /// the host takes (and clears). Uses only the stdlib `json` module — no numpy/site needed.
    #[test]
    fn settings_round_trip_records_and_reads() {
        let _g = crate::interp_test_lock();
        Python::attach(|py| {
            // get_settings: a persisted section parses back to its values.
            let snap = ContextSnapshot {
                settings_json: Some(r#"{"stride": 5, "invert": true}"#.to_string()),
                ..Default::default()
            };
            let ctx = Py::new(py, PluginContext::new(snap)).unwrap();
            let b = ctx.bind(py);
            let got = b.call_method0("get_settings").unwrap();
            let stride: i64 = got.get_item("stride").unwrap().extract().unwrap();
            assert_eq!(stride, 5);
            let invert: bool = got.get_item("invert").unwrap().extract().unwrap();
            assert!(invert);

            // No persisted settings → an empty dict.
            let empty = Py::new(py, PluginContext::new(ContextSnapshot::default())).unwrap();
            let eb = empty.bind(py);
            let got2 = eb.call_method0("get_settings").unwrap();
            assert_eq!(got2.len().unwrap(), 0);

            // save_settings records the dict; the host takes it as a JSON string, then it's cleared.
            let data = PyDict::new(py);
            data.set_item("stride", 3).unwrap();
            data.set_item("invert", false).unwrap();
            eb.call_method1("save_settings", (&data,)).unwrap();
            let pending = eb.borrow_mut().take_pending_settings().unwrap();
            // Round-trip the recorded string back through json to assert its contents.
            let reparsed = py.import("json").unwrap().call_method1("loads", (pending,)).unwrap();
            let s: i64 = reparsed.get_item("stride").unwrap().extract().unwrap();
            assert_eq!(s, 3);
            assert_eq!(eb.borrow_mut().take_pending_settings(), None); // taken once, then cleared
        });
    }
}
