from typing import Callable, Optional

import cv2
import numpy as np

from app.core.image_sequence import ImageSequence
from app.models.tracker_result import TrackerResult

# Flat LK parameters (assembled into cv2 kwargs by _cv_lk_kwargs).
DEFAULT_LK = dict(
    win_size=21,
    max_level=3,
    max_iter=30,
    epsilon=0.01,
    flags=0,
    min_eig_threshold=1e-4,
)

ProgressCb = Callable[[int, int], bool]  # (done, total) -> cancel?


def _cv_lk_kwargs(p: dict) -> dict:
    return dict(
        winSize=(int(p["win_size"]), int(p["win_size"])),
        maxLevel=int(p["max_level"]),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(p["max_iter"]),
            float(p["epsilon"]),
        ),
        flags=int(p["flags"]),
        minEigThreshold=float(p["min_eig_threshold"]),
    )


def _track_pass(get_gray, frame_order, seed_pts, lk_kwargs, progress, total, done0):
    """Run LK along `frame_order` (list of global indices). Results are stored at the
    *position within frame_order* (index 0 = the seed frame)."""
    n = len(frame_order)
    p = len(seed_pts)
    coords = np.zeros((n, p, 2), dtype=np.float32)
    status = np.zeros((n, p), dtype=np.uint8)
    err = np.zeros((n, p), dtype=np.float32)

    coords[0] = seed_pts
    status[0] = 1

    prev_gray = get_gray(frame_order[0])
    prev_pts = seed_pts.reshape(-1, 1, 2).astype(np.float32)
    done = done0
    for i in range(1, n):
        cur_gray = get_gray(frame_order[i])
        nxt, st, er = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None, **lk_kwargs
        )
        coords[i] = nxt.reshape(-1, 2)
        status[i] = st.reshape(-1)
        err[i] = er.reshape(-1)
        prev_gray = cur_gray
        prev_pts = nxt  # carry forward predicted positions (failures flagged by status)
        done += 1
        if progress is not None and progress(done, total):
            return None
    return coords, status, err, done


def _compute_fb_errors(coords_fw, coords_bw, status_fw, status_bw):
    diff = np.linalg.norm(coords_fw - coords_bw, axis=2)  # (N, P)
    valid = (status_fw == 1) & (status_bw == 1)  # (N, P)
    p = diff.shape[1]
    fb_mean = np.full(p, np.inf, dtype=np.float32)
    fb_max = np.full(p, np.inf, dtype=np.float32)
    for j in range(p):
        col = diff[valid[:, j], j]
        if col.size:
            fb_mean[j] = col.mean()
            fb_max[j] = col.max()
    return fb_mean, fb_max


def track(
    sequence: ImageSequence,
    reference_index: int,
    last_index: int,
    seed_pts: np.ndarray,
    lk_params: dict,
    progress_cb: Optional[ProgressCb] = None,
) -> Optional[TrackerResult]:
    """Track seed_pts forward (reference->last) then backward (last->reference, seeded from
    the forward result's last-frame positions). Returns None if cancelled via progress_cb.

    coords_bw is reindexed so coords_bw[t] aligns with coords_fw[t] (cut 0 = reference).
    """
    seed_pts = np.asarray(seed_pts, dtype=np.float32).reshape(-1, 2)
    n = last_index - reference_index + 1
    lk_kwargs = _cv_lk_kwargs(lk_params)
    total = 2 * (n - 1)

    def get_gray(global_index):
        return sequence.load_gray(global_index)

    forward_order = list(range(reference_index, last_index + 1))
    fw = _track_pass(get_gray, forward_order, seed_pts, lk_kwargs, progress_cb, total, 0)
    if fw is None:
        return None
    coords_fw, status_fw, err_fw, done = fw

    # Backward pass: seed from forward's last-frame positions, walk last -> reference.
    backward_order = list(range(last_index, reference_index - 1, -1))
    bw_seed = coords_fw[n - 1]
    bw = _track_pass(get_gray, backward_order, bw_seed, lk_kwargs, progress_cb, total, done)
    if bw is None:
        return None
    coords_bw_rev, status_bw_rev, err_bw_rev, _ = bw

    # Reindex backward arrays from pass-order (last..reference) to cut-order (reference..last).
    coords_bw = coords_bw_rev[::-1].copy()
    status_bw = status_bw_rev[::-1].copy()
    err_bw = err_bw_rev[::-1].copy()

    fb_mean, fb_max = _compute_fb_errors(coords_fw, coords_bw, status_fw, status_bw)

    return TrackerResult(
        reference_index=reference_index,
        last_index=last_index,
        coords_fw=coords_fw,
        status_fw=status_fw,
        err_fw=err_fw,
        coords_bw=coords_bw,
        status_bw=status_bw,
        err_bw=err_bw,
        fb_mean_error=fb_mean,
        fb_max_error=fb_max,
        win_size=int(lk_params["win_size"]),
    )
