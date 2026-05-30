//! Mutable per-session state with the dual global/cut index system.
//! Ports `app/models/project_state.py`.
//!
//! Index vocabulary:
//!   - global index: position in the full folder (0 .. total_images-1)
//!   - cut index:    position within reference..last (0 .. n_cut-1), cut 0 = reference
//! Invariant enforced by the setters: 0 <= reference_index <= last_index < total_images.

use crate::feature_detection::{GridParams, ShiTomasiParams};
use crate::image_sequence::ImageSequence;
use crate::result::{LkParams, TrackerResult};
use crate::roi::Roi;
use crate::settings;
use opencv::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Marker-rendering preferences (display-only; persisted under "display").
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayParams {
    pub show_markers: bool,
    pub marker_size: i32,
    pub marker_opacity: i32,
    pub show_window_box: bool,
    pub show_roi: bool,
}

impl Default for DisplayParams {
    fn default() -> Self {
        Self {
            show_markers: true,
            marker_size: 3,
            marker_opacity: 100,
            show_window_box: false,
            show_roi: true,
        }
    }
}

/// `{**DEFAULT, **(section or {})}`: deserialize a persisted section over defaults.
fn merge_section<T: Default + DeserializeOwned>(name: &str) -> T {
    match settings::get_section(name) {
        Some(v) => serde_json::from_value(v).unwrap_or_default(),
        None => T::default(),
    }
}

pub struct ProjectState {
    pub sequence: Option<ImageSequence>,
    pub source_dir: Option<String>,
    pub reference_index: usize,
    pub last_index: usize,
    pub current_index: usize,
    pub roi: Option<Roi>,

    pub shi_tomasi_params: ShiTomasiParams,
    pub grid_params: GridParams,
    pub lk_params: LkParams,
    pub display_params: DisplayParams,

    pub features: Option<Vec<(f32, f32)>>, // reference-frame seed points
    pub result: Option<TrackerResult>,
    pub active_mask: Option<Vec<bool>>, // (P,) aligned to result points
    pub undo_stack: Vec<Vec<bool>>,     // previous active_mask snapshots (cleanup undo)
}

impl Default for ProjectState {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectState {
    pub fn new() -> Self {
        Self {
            sequence: None,
            source_dir: None,
            reference_index: 0,
            last_index: 0,
            current_index: 0,
            roi: None,
            shi_tomasi_params: merge_section("shi_tomasi"),
            grid_params: merge_section("grid"),
            lk_params: merge_section("lk"),
            display_params: merge_section("display"),
            features: None,
            result: None,
            active_mask: None,
            undo_stack: Vec::new(),
        }
    }

    pub fn load_sequence(&mut self, sequence: ImageSequence, source_dir: Option<String>) {
        self.last_index = sequence.len().saturating_sub(1);
        self.sequence = Some(sequence);
        self.source_dir = source_dir;
        self.reference_index = 0;
        self.current_index = 0;
        self.roi = None;
        self.features = None;
        self.result = None;
        self.active_mask = None;
        self.undo_stack.clear();
    }

    /// (height, width) of the frames, or None if no sequence.
    pub fn image_size(&self) -> opencv::Result<Option<(i32, i32)>> {
        match &self.sequence {
            None => Ok(None),
            Some(seq) => {
                let bgr = seq.load_bgr(0)?;
                Ok(Some((bgr.rows(), bgr.cols())))
            }
        }
    }

    pub fn total_images(&self) -> usize {
        self.sequence.as_ref().map_or(0, |s| s.len())
    }

    pub fn has_sequence(&self) -> bool {
        self.sequence.is_some()
    }

    // ---- index helpers --------------------------------------------------

    /// Frames in the reference..last range (both endpoints inclusive).
    pub fn n_cut(&self) -> usize {
        self.last_index - self.reference_index + 1
    }

    /// Convert a global index to a cut index (may be negative if before the reference frame).
    pub fn global_to_cut(&self, global_index: usize) -> isize {
        global_index as isize - self.reference_index as isize
    }

    pub fn cut_to_global(&self, cut_index: isize) -> isize {
        self.reference_index as isize + cut_index
    }

    pub fn current_in_range(&self) -> bool {
        self.reference_index <= self.current_index && self.current_index <= self.last_index
    }

    pub fn on_reference_frame(&self) -> bool {
        self.has_sequence() && self.current_index == self.reference_index
    }

    // ---- validated setters (mirror the Python min/max clamping) ---------

    pub fn set_current(&mut self, index: i64) {
        let hi = self.total_images() as i64 - 1;
        self.current_index = index.min(hi).max(0) as usize;
    }

    pub fn set_reference(&mut self, index: i64) {
        self.reference_index = index.min(self.last_index as i64).max(0) as usize;
    }

    pub fn set_last(&mut self, index: i64) {
        let hi = self.total_images() as i64 - 1;
        self.last_index = index.min(hi).max(self.reference_index as i64) as usize;
    }
}
