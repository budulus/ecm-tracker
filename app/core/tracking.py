from typing import Callable, Optional

import cv2
import numpy as np

from app.core.image_sequence import ImageSequence
from app.core.result import TrackerResult

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
    required = set(DEFAULT_LK)
    missing = sorted(required - set(p))
    if missing:
        raise ValueError(f"Missing LK parameter(s): {', '.join(missing)}")
    try:
        win_size = int(p["win_size"])
        max_level = int(p["max_level"])
        max_iter = int(p["max_iter"])
        epsilon = float(p["epsilon"])
        flags = int(p["flags"])
        min_eig = float(p["min_eig_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("LK parameters must be numeric") from exc
    if win_size < 3 or max_level < 0 or max_iter < 1 or epsilon <= 0 or min_eig < 0:
        raise ValueError("LK parameters are outside their valid ranges")
    if flags not in (0, cv2.OPTFLOW_LK_GET_MIN_EIGENVALS):
        raise ValueError("LK flags must select standard error (0) or minimum-eigenvalue error (8)")
    if not np.isfinite(epsilon) or not np.isfinite(min_eig):
        raise ValueError("LK parameters must be finite")
    return dict(
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            epsilon,
        ),
        flags=flags,
        minEigThreshold=min_eig,
    )


def _track_pass(
    get_gray,
    frame_order,
    seed_pts,
    lk_kwargs,
    progress,
    total,
    done0,
    seed_valid=None,
):
    """Run LK along `frame_order` (list of global indices). Results are stored at the
    *position within frame_order* (index 0 = the seed frame)."""
    n = len(frame_order)
    p = len(seed_pts)
    coords = np.zeros((n, p, 2), dtype=np.float32)
    status = np.zeros((n, p), dtype=np.uint8)
    err = np.zeros((n, p), dtype=np.float32)

    coords[0] = seed_pts
    alive = (
        np.ones(p, dtype=bool)
        if seed_valid is None
        else np.asarray(seed_valid, dtype=bool).reshape(p).copy()
    )
    status[0] = alive.astype(np.uint8)
    err[0, ~alive] = np.inf

    prev_gray = get_gray(frame_order[0])
    prev_pts = seed_pts.reshape(-1, 1, 2).astype(np.float32)
    done = done0
    for i in range(1, n):
        cur_gray = get_gray(frame_order[i])
        nxt, st, er = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None, **lk_kwargs
        )
        # OpenCV's status describes only this one frame-to-frame transition. A point that has
        # already been lost must never become valid again: its returned location is undefined and
        # feeding that location into the next transition can create a plausible-looking "revived"
        # track. Keep validity cumulative and freeze dead points at their last valid coordinate.
        if nxt is None or st is None:
            next_pts = coords[i - 1].copy()
            step_valid = np.zeros(p, dtype=bool)
            step_err = np.full(p, np.inf, dtype=np.float32)
        else:
            next_pts = np.asarray(nxt, dtype=np.float32).reshape(-1, 2)
            step_valid = np.asarray(st).reshape(-1).astype(bool)
            if next_pts.shape != (p, 2) or step_valid.shape != (p,):
                raise ValueError("OpenCV returned an unexpected optical-flow result shape")
            step_valid &= np.isfinite(next_pts).all(axis=1)
            step_err = (
                np.asarray(er, dtype=np.float32).reshape(-1)
                if er is not None
                else np.full(p, np.inf, dtype=np.float32)
            )
            if step_err.shape != (p,):
                raise ValueError("OpenCV returned an unexpected optical-flow error shape")
            step_err[~np.isfinite(step_err)] = np.inf

        alive &= step_valid
        next_pts[~alive] = coords[i - 1, ~alive]
        step_err[~alive] = np.inf
        coords[i] = next_pts
        status[i] = alive.astype(np.uint8)
        err[i] = step_err
        prev_gray = cur_gray
        prev_pts = next_pts.reshape(-1, 1, 2)
        done += 1
        if progress is not None and progress(done, total):
            return None
    return coords, status, err, done


def _compute_fb_errors(coords_fw, coords_bw, status_fw, status_bw):
    diff = np.linalg.norm(coords_fw - coords_bw, axis=2)  # (N, P)
    valid = (status_fw == 1) & (status_bw == 1)  # (N, P)
    if diff.shape[0] == 1:
        zeros = np.zeros(diff.shape[1], dtype=np.float32)
        return zeros, zeros.copy()
    # The last frame is the backward seed, so its discrepancy is identically zero and carries no
    # round-trip information. Including it would systematically bias every FB mean downward.
    valid[-1] = False
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
    if not 0 <= reference_index <= last_index < len(sequence):
        raise ValueError("Tracking range is outside the loaded image sequence")
    if seed_pts.shape[0] == 0 or not np.isfinite(seed_pts).all():
        raise ValueError("Tracking requires at least one finite seed point")
    h, w = sequence.load_gray(reference_index).shape[:2]
    if np.any(seed_pts < 0) or np.any(seed_pts[:, 0] >= w) or np.any(seed_pts[:, 1] >= h):
        raise ValueError("Tracking seeds must be inside the reference image")
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
    # A point that did not survive the complete forward pass has no trustworthy last-frame seed.
    # Keep it invalid throughout the backward pass so FB metrics cannot be rescued by garbage.
    bw_seed_valid = status_fw[n - 1].astype(bool)
    bw = _track_pass(
        get_gray,
        backward_order,
        bw_seed,
        lk_kwargs,
        progress_cb,
        total,
        done,
        seed_valid=bw_seed_valid,
    )
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
        error_kind="min_eigenvalue" if int(lk_params.get("flags", 0)) & 8 else "photometric",
        tracking_params=dict(lk_params),
    )
