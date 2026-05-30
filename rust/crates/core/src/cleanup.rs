//! Per-point quality metrics + threshold-based filtering for the cleanup flow.
//! Ports the computational core of `app/core/cleanup.py`. (The `*_to_dict`/`*_from_dict`
//! JSON persistence and the UI band-scaling tables land with the settings/GUI port.)

use crate::result::TrackerResult;
use crate::roi::Roi;

/// Spin-box ceiling for band bounds; also the "no upper limit" value.
pub const BAND_CAP: f64 = 1_000_000.0;

/// Per-point quality metrics, derived once from a `TrackerResult`. Points that never
/// tracked validly get +inf error/step so the error filters can drop them once tightened.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub n_frames: usize,
    pub n_points: usize,
    pub fail_count_fw: Vec<i64>,  // (P,)
    pub fail_count_bw: Vec<i64>,  // (P,)
    pub max_err_fw: Vec<f32>,     // (P,)
    pub mean_err_fw: Vec<f32>,    // (P,)
    pub fb_mean: Vec<f32>,        // (P,)
    pub fb_max: Vec<f32>,         // (P,)
    pub max_step: Vec<f32>,       // (P,) max single-frame displacement
    pub left_image: Vec<bool>,    // (P,)
    pub left_roi: Vec<bool>,      // (P,)
}

/// A max threshold for one metric. Disabled = ignored entirely (so +inf metrics survive);
/// enabled = keep a point iff `metric <= hi`. `cap` is the UI slider scale (unused here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandFilter {
    pub enabled: bool,
    pub hi: f64,
    pub cap: f64,
}

impl Default for BandFilter {
    fn default() -> Self {
        Self { enabled: false, hi: BAND_CAP, cap: BAND_CAP }
    }
}

/// Per-filter bands plus two boolean filters. A point is kept iff it passes every *enabled*
/// filter. Defaults (all disabled, both bools off) keep all points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Thresholds {
    pub fw_failures: BandFilter,
    pub bw_failures: BandFilter,
    pub opencv_error: BandFilter,
    pub mean_error: BandFilter,
    pub fb_mean: BandFilter,
    pub fb_max: BandFilter,
    pub distance: BandFilter,
    pub drop_left_image: bool,
    pub drop_left_roi: bool,
}

/// Factory defaults: every filter disabled, so all points are kept.
pub fn default_thresholds() -> Thresholds {
    Thresholds::default()
}

/// Derive per-point metrics from a tracking result. `image_size` is `(height, width)`.
pub fn compute_metrics(
    result: &TrackerResult,
    roi: Option<&Roi>,
    image_size: (i32, i32),
) -> opencv::Result<Metrics> {
    let cf = &result.coords_fw; // (N, P, 2)
    let sf = &result.status_fw; // (N, P)
    let ef = &result.err_fw; // (N, P)
    let sb = &result.status_bw;
    let n = result.n_frames();
    let p = result.n_points();
    let (h, w) = image_size;

    // Failure counts = number of frames with status 0.
    let mut fail_fw = vec![0i64; p];
    let mut fail_bw = vec![0i64; p];
    for t in 0..n {
        for j in 0..p {
            if sf[[t, j]] == 0 {
                fail_fw[j] += 1;
            }
            if sb[[t, j]] == 0 {
                fail_bw[j] += 1;
            }
        }
    }

    // Max / mean forward OpenCV error over valid frames; +inf if never valid.
    let mut max_err = vec![f32::INFINITY; p];
    let mut mean_err = vec![f32::INFINITY; p];
    for j in 0..p {
        let mut vals = Vec::new();
        for t in 0..n {
            if sf[[t, j]] == 1 {
                vals.push(ef[[t, j]]);
            }
        }
        if !vals.is_empty() {
            max_err[j] = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            mean_err[j] = vals.iter().sum::<f32>() / vals.len() as f32;
        }
    }

    // Max single-frame displacement over consecutive frames where both endpoints are valid.
    let mut step = vec![f32::INFINITY; p];
    if n >= 2 {
        for j in 0..p {
            let mut vals = Vec::new();
            for t in 1..n {
                if sf[[t, j]] == 1 && sf[[t - 1, j]] == 1 {
                    let dx = cf[[t, j, 0]] - cf[[t - 1, j, 0]];
                    let dy = cf[[t, j, 1]] - cf[[t - 1, j, 1]];
                    vals.push((dx * dx + dy * dy).sqrt());
                }
            }
            if !vals.is_empty() {
                step[j] = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
        }
    }

    // Left the image: any valid frame whose point is out of bounds.
    let mut left_image = vec![false; p];
    for j in 0..p {
        for t in 0..n {
            if sf[[t, j]] == 1 {
                let x = cf[[t, j, 0]];
                let y = cf[[t, j, 1]];
                if x < 0.0 || x >= w as f32 || y < 0.0 || y >= h as f32 {
                    left_image[j] = true;
                    break;
                }
            }
        }
    }

    // Left the ROI: any valid frame whose point falls outside a complete ROI.
    let mut left_roi = vec![false; p];
    if let Some(r) = roi {
        if r.is_complete() {
            for j in 0..p {
                for t in 0..n {
                    if sf[[t, j]] == 1
                        && !r.contains(cf[[t, j, 0]] as f64, cf[[t, j, 1]] as f64)?
                    {
                        left_roi[j] = true;
                        break;
                    }
                }
            }
        }
    }

    Ok(Metrics {
        n_frames: n,
        n_points: p,
        fail_count_fw: fail_fw,
        fail_count_bw: fail_bw,
        max_err_fw: max_err,
        mean_err_fw: mean_err,
        fb_mean: result.fb_mean_error.to_vec(),
        fb_max: result.fb_max_error.to_vec(),
        max_step: step,
        left_image,
        left_roi,
    })
}

fn apply_band(keep: &mut [bool], band: &BandFilter, value: impl Fn(usize) -> f64) {
    if !band.enabled {
        return;
    }
    for (j, k) in keep.iter_mut().enumerate() {
        *k &= value(j) <= band.hi;
    }
}

/// Boolean (P,) keep mask. Disabled bands are skipped entirely; an enabled band keeps a point
/// iff `metric <= hi` (so +inf metrics survive unless an enabled band excludes them).
pub fn build_mask(m: &Metrics, thr: &Thresholds) -> Vec<bool> {
    let mut keep = vec![true; m.n_points];
    apply_band(&mut keep, &thr.fw_failures, |j| m.fail_count_fw[j] as f64);
    apply_band(&mut keep, &thr.bw_failures, |j| m.fail_count_bw[j] as f64);
    apply_band(&mut keep, &thr.opencv_error, |j| m.max_err_fw[j] as f64);
    apply_band(&mut keep, &thr.mean_error, |j| m.mean_err_fw[j] as f64);
    apply_band(&mut keep, &thr.fb_mean, |j| m.fb_mean[j] as f64);
    apply_band(&mut keep, &thr.fb_max, |j| m.fb_max[j] as f64);
    apply_band(&mut keep, &thr.distance, |j| m.max_step[j] as f64);
    if thr.drop_left_image {
        for (j, k) in keep.iter_mut().enumerate() {
            *k &= !m.left_image[j];
        }
    }
    if thr.drop_left_roi {
        for (j, k) in keep.iter_mut().enumerate() {
            *k &= !m.left_roi[j];
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_2pts() -> Metrics {
        // point 0 = clean, point 1 = bad (high error, +inf step, left image+roi).
        Metrics {
            n_frames: 5,
            n_points: 2,
            fail_count_fw: vec![0, 3],
            fail_count_bw: vec![0, 2],
            max_err_fw: vec![1.0, 50.0],
            mean_err_fw: vec![0.5, f32::INFINITY],
            fb_mean: vec![0.1, f32::INFINITY],
            fb_max: vec![0.2, f32::INFINITY],
            max_step: vec![2.0, f32::INFINITY],
            left_image: vec![false, true],
            left_roi: vec![false, true],
        }
    }

    #[test]
    fn default_keeps_all() {
        let m = metrics_2pts();
        assert_eq!(build_mask(&m, &default_thresholds()), vec![true, true]);
    }

    #[test]
    fn disabled_band_ignores_inf() {
        // fb_mean disabled -> +inf point survives.
        let mut thr = Thresholds::default();
        thr.fb_mean = BandFilter { enabled: false, hi: 1.0, cap: BAND_CAP };
        assert_eq!(build_mask(&metrics_2pts(), &thr), vec![true, true]);
    }

    #[test]
    fn enabled_band_drops_inf_and_over_threshold() {
        let mut thr = Thresholds::default();
        thr.fb_mean = BandFilter { enabled: true, hi: 1.0, cap: BAND_CAP }; // inf > 1 -> drop pt 1
        assert_eq!(build_mask(&metrics_2pts(), &thr), vec![true, false]);
    }

    #[test]
    fn count_band_and_bool_filters() {
        let mut thr = Thresholds::default();
        thr.fw_failures = BandFilter { enabled: true, hi: 0.0, cap: BAND_CAP }; // drop any fw failure
        assert_eq!(build_mask(&metrics_2pts(), &thr), vec![true, false]);

        let mut thr2 = Thresholds::default();
        thr2.drop_left_image = true;
        assert_eq!(build_mask(&metrics_2pts(), &thr2), vec![true, false]);
    }
}
