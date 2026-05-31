"""Pure-numpy core for the Affine Zones plugin (Phase 4 slice 4c).

Point-in-polygon, per-zone least-squares affine fit, principal stretches, a RANSAC affine estimator
(reimplementing ``cv2.estimateAffine2D(method=RANSAC)`` — there is **no cv2** in the embedded env),
and the CSV writer. Faithful to ``plugins/affine_zones/zones.py``. No Qt, no cv2.
"""

import csv

import numpy as np

MIN_ZONE_POINTS = 3


def points_in_polygon(polygon, pts):
    """Boolean mask of which (x, y) rows of ``pts`` fall inside ``polygon`` (a list of (x, y)).

    Uses ``matplotlib.path.Path`` (the original delegated to ``app.core.roi.ROI`` which isn't
    importable here; Path.contains_points is the standard even-odd test)."""
    from matplotlib.path import Path

    return Path(np.asarray(polygon, dtype=float)).contains_points(np.asarray(pts, dtype=float))


def fit_zone_deformation(polygon, ref_pts, cur_pts, valid=None):
    """Least-squares affine fit ``[x y 1] -> [x' y']`` over the points inside ``polygon``. Returns
    ``(local_idx, F, b)`` (F = 2x2 linear part, b = translation), or ``None`` if fewer than
    ``MIN_ZONE_POINTS`` points are in the zone."""
    inside = points_in_polygon(polygon, ref_pts)
    if valid is not None:
        inside = inside & np.asarray(valid, dtype=bool)
    local_idx = np.where(inside)[0]
    if local_idx.size < MIN_ZONE_POINTS:
        return None
    src = ref_pts[local_idx].astype(np.float64)
    dst = cur_pts[local_idx].astype(np.float64)
    design = np.column_stack([src, np.ones(src.shape[0])])
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)  # (3, 2)
    F = np.array([[sol[0, 0], sol[1, 0]], [sol[0, 1], sol[1, 1]]])
    b = sol[2, :].copy()
    return local_idx, F, b


def principal_stretches(F):
    """Principal stretches of F: ``C = FᵀF``; ``λ = √max(eig(C), 0)`` sorted descending ->
    ``(λ1, λ2, v1, v2)`` with v1/v2 the matching eigenvectors (columns)."""
    C = F.T @ F
    vals, vecs = np.linalg.eigh(C)  # ascending eigenvalues, orthonormal columns
    lam = np.sqrt(np.maximum(vals, 0.0))
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    vecs = vecs[:, order]
    return float(lam[0]), float(lam[1]), vecs[:, 0].copy(), vecs[:, 1].copy()


def _fit_affine(src, dst):
    """Exact/least-squares affine (2x3 ``[A | t]``) from N>=3 correspondences."""
    design = np.column_stack([src, np.ones(src.shape[0])])
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)  # (3, 2)
    return sol.T  # (2, 3): [[a, b, tx], [c, d, ty]]


def estimate_affine_ransac(src, dst, reproj=3.0, max_iters=2000, rng_seed=0):
    """Reimplements ``cv2.estimateAffine2D(method=cv2.RANSAC)``: repeatedly sample 3 correspondences,
    fit an affine, count inliers within ``reproj`` px, keep the best; refit on the inliers. Returns
    ``(M, inliers)`` (M is 2x3, inliers a bool mask over the inputs), or ``(None, all-False)``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < MIN_ZONE_POINTS:
        return None, np.zeros(n, dtype=bool)
    rng = np.random.default_rng(rng_seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    thr2 = float(reproj) ** 2
    for _ in range(int(max_iters)):
        idx = rng.choice(n, size=3, replace=False)
        M = _fit_affine(src[idx], dst[idx])
        pred = src @ M[:, :2].T + M[:, 2]
        err2 = np.sum((pred - dst) ** 2, axis=1)
        inliers = err2 <= thr2
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
            if count == n:
                break
    if best_count < MIN_ZONE_POINTS:
        return None, best_inliers
    M = _fit_affine(src[best_inliers], dst[best_inliers])
    return M, best_inliers


def bbox_polygon(pts):
    """Axis-aligned bounding-box polygon of ``pts`` (used as a default zone in the headless selftest)."""
    p = np.asarray(pts, dtype=float)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def write_zones_csv(path, polygons, coords, status, ref_global, frame_count):
    """CSV: one row per (frame, zone). Header
    ``[zone, frame_global, n_points, lambda1, lambda2, v1x, v1y, v2x, v2y]``."""
    ref_pts = coords[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zone", "frame_global", "n_points", "lambda1", "lambda2",
                    "v1x", "v1y", "v2x", "v2y"])
        for cut in range(frame_count):
            cur_pts = coords[cut]
            valid = None if status is None else (status[cut] == 1)
            for z, poly in enumerate(polygons):
                fit = fit_zone_deformation(poly, ref_pts, cur_pts, valid=valid)
                if fit is None:
                    continue
                local_idx, F, _b = fit
                lam1, lam2, v1, v2 = principal_stretches(F)
                w.writerow([z + 1, ref_global + cut, local_idx.size,
                            f"{lam1:.6f}", f"{lam2:.6f}",
                            f"{v1[0]:.6f}", f"{v1[1]:.6f}",
                            f"{v2[0]:.6f}", f"{v2[1]:.6f}"])
