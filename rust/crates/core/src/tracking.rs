//! Pyramidal Lucas-Kanade tracking: forward (reference->last) then backward
//! (last->reference, seeded from the forward pass's last frame), plus the per-point
//! forward-backward error. Ports `app/core/tracking.py`.

use crate::image_sequence::ImageSequence;
use crate::result::{LkParams, TrackerResult};
use ndarray::{s, Array1, Array2, Array3};
use opencv::core::{Point2f, Size, TermCriteria, TermCriteria_Type, Vector};
use opencv::video;

/// `(done, total) -> cancel?`
pub type ProgressCb<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

struct PassOut {
    coords: Array3<f32>, // (n, p, 2)
    status: Array2<u8>,  // (n, p)
    err: Array2<f32>,    // (n, p)
}

/// Run LK along `frame_order` (global indices). Results are stored at the position
/// within `frame_order` (index 0 = the seed frame). Returns `None` if cancelled.
#[allow(clippy::too_many_arguments)]
fn track_pass(
    seq: &ImageSequence,
    frame_order: &[usize],
    seed: &[Point2f],
    lk: &LkParams,
    progress: &mut Option<ProgressCb>,
    total: usize,
    done0: usize,
) -> opencv::Result<Option<(PassOut, usize)>> {
    let n = frame_order.len();
    let p = seed.len();
    let mut coords = Array3::<f32>::zeros((n, p, 2));
    let mut status = Array2::<u8>::zeros((n, p));
    let mut err = Array2::<f32>::zeros((n, p));

    for (j, pt) in seed.iter().enumerate() {
        coords[[0, j, 0]] = pt.x;
        coords[[0, j, 1]] = pt.y;
        status[[0, j]] = 1;
    }

    let criteria = TermCriteria::new(
        TermCriteria_Type::EPS as i32 | TermCriteria_Type::COUNT as i32,
        lk.max_iter,
        lk.epsilon,
    )?;
    let win = Size::new(lk.win_size, lk.win_size);

    let mut prev_gray = seq.load_gray(frame_order[0])?;
    let mut prev_pts: Vector<Point2f> = seed.iter().copied().collect();
    let mut done = done0;

    for i in 1..n {
        let cur_gray = seq.load_gray(frame_order[i])?;
        let mut next_pts = Vector::<Point2f>::new();
        let mut st = Vector::<u8>::new();
        let mut er = Vector::<f32>::new();
        video::calc_optical_flow_pyr_lk(
            &prev_gray,
            &cur_gray,
            &prev_pts,
            &mut next_pts,
            &mut st,
            &mut er,
            win,
            lk.max_level,
            criteria,
            lk.flags,
            lk.min_eig_threshold,
        )?;
        for j in 0..p {
            let pt = next_pts.get(j)?;
            coords[[i, j, 0]] = pt.x;
            coords[[i, j, 1]] = pt.y;
            status[[i, j]] = st.get(j)?;
            err[[i, j]] = er.get(j)?;
        }
        prev_gray = cur_gray;
        prev_pts = next_pts; // carry forward predicted positions (failures flagged by status)
        done += 1;
        if let Some(cb) = progress.as_deref_mut() {
            if cb(done, total) {
                return Ok(None);
            }
        }
    }
    Ok(Some((PassOut { coords, status, err }, done)))
}

fn compute_fb_errors(
    coords_fw: &Array3<f32>,
    coords_bw: &Array3<f32>,
    status_fw: &Array2<u8>,
    status_bw: &Array2<u8>,
) -> (Array1<f32>, Array1<f32>) {
    let n = coords_fw.shape()[0];
    let p = coords_fw.shape()[1];
    let mut fb_mean = Array1::<f32>::from_elem(p, f32::INFINITY);
    let mut fb_max = Array1::<f32>::from_elem(p, f32::INFINITY);
    for j in 0..p {
        let mut vals: Vec<f32> = Vec::new();
        for t in 0..n {
            if status_fw[[t, j]] == 1 && status_bw[[t, j]] == 1 {
                let dx = coords_fw[[t, j, 0]] - coords_bw[[t, j, 0]];
                let dy = coords_fw[[t, j, 1]] - coords_bw[[t, j, 1]];
                vals.push((dx * dx + dy * dy).sqrt());
            }
        }
        if !vals.is_empty() {
            fb_mean[j] = vals.iter().sum::<f32>() / vals.len() as f32;
            fb_max[j] = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
    }
    (fb_mean, fb_max)
}

/// Track `seed_pts` forward then backward; returns `None` if cancelled via `progress`.
/// `coords_bw` is reindexed so `coords_bw[t]` aligns with `coords_fw[t]` (cut 0 = reference).
pub fn track(
    seq: &ImageSequence,
    reference_index: usize,
    last_index: usize,
    seed_pts: &[(f32, f32)],
    lk: &LkParams,
    mut progress: Option<ProgressCb>,
) -> opencv::Result<Option<TrackerResult>> {
    let seed: Vec<Point2f> = seed_pts.iter().map(|&(x, y)| Point2f::new(x, y)).collect();
    let n = last_index - reference_index + 1;
    let total = 2 * (n - 1);

    let forward_order: Vec<usize> = (reference_index..=last_index).collect();
    let (fw, done) = match track_pass(seq, &forward_order, &seed, lk, &mut progress, total, 0)? {
        Some(x) => x,
        None => return Ok(None),
    };

    // Backward pass: seed from forward's last-frame positions, walk last -> reference.
    let p = seed.len();
    let bw_seed: Vec<Point2f> = (0..p)
        .map(|j| Point2f::new(fw.coords[[n - 1, j, 0]], fw.coords[[n - 1, j, 1]]))
        .collect();
    let backward_order: Vec<usize> = (reference_index..=last_index).rev().collect();
    let (bw, _) = match track_pass(seq, &backward_order, &bw_seed, lk, &mut progress, total, done)? {
        Some(x) => x,
        None => return Ok(None),
    };

    // Reindex backward arrays from pass-order (last..reference) to cut-order (reference..last).
    let coords_bw = bw.coords.slice(s![..;-1, .., ..]).to_owned();
    let status_bw = bw.status.slice(s![..;-1, ..]).to_owned();
    let err_bw = bw.err.slice(s![..;-1, ..]).to_owned();

    let (fb_mean_error, fb_max_error) =
        compute_fb_errors(&fw.coords, &coords_bw, &fw.status, &status_bw);

    Ok(Some(TrackerResult {
        reference_index,
        last_index,
        coords_fw: fw.coords,
        status_fw: fw.status,
        err_fw: fw.err,
        coords_bw,
        status_bw,
        err_bw,
        fb_mean_error,
        fb_max_error,
        win_size: lk.win_size,
    }))
}
