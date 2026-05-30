//! Immutable tracking output + LK parameters.
//! Ports `app/models/tracker_result.py` and `DEFAULT_LK` from `app/core/tracking.py`.

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// Flat Lucas-Kanade parameters, assembled into OpenCV's nested `winSize`/`criteria`
/// at call time (mirrors `_cv_lk_kwargs`). Edit the flat fields, not OpenCV shapes.
///
/// `#[serde(default)]` makes a partial persisted section fill present fields and default
/// the rest — the same merge as Python's `{**DEFAULT_LK, **section}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LkParams {
    pub win_size: i32,
    pub max_level: i32,
    pub max_iter: i32,
    pub epsilon: f64,
    pub flags: i32,
    pub min_eig_threshold: f64,
}

impl Default for LkParams {
    fn default() -> Self {
        // DEFAULT_LK
        Self {
            win_size: 21,
            max_level: 3,
            max_iter: 30,
            epsilon: 0.01,
            flags: 0,
            min_eig_threshold: 1e-4,
        }
    }
}

/// Immutable raw tracking output. Never mutated after creation — cleanup operates on a
/// separate active mask (in `ProjectState`), not on this.
///
/// All per-frame arrays are indexed by *cut* index (0 = reference, N-1 = last), so
/// `coords_fw[t]` and `coords_bw[t]` describe the same frame and compare directly.
#[derive(Clone, Debug)]
pub struct TrackerResult {
    pub reference_index: usize,
    pub last_index: usize,

    pub coords_fw: Array3<f32>, // (N, P, 2)
    pub status_fw: Array2<u8>,  // (N, P)
    pub err_fw: Array2<f32>,    // (N, P)

    pub coords_bw: Array3<f32>,
    pub status_bw: Array2<u8>,
    pub err_bw: Array2<f32>,

    pub fb_mean_error: Array1<f32>, // (P,)
    pub fb_max_error: Array1<f32>,  // (P,)

    /// LK window size (px) used to produce this result, kept so overlays can show the
    /// real search window even if the user later edits the Tracker params.
    pub win_size: i32,
}

impl TrackerResult {
    pub fn n_frames(&self) -> usize {
        self.coords_fw.shape()[0]
    }

    pub fn n_points(&self) -> usize {
        self.coords_fw.shape()[1]
    }
}
