"""Homogeneous in-plane kinematics from tracked points (Qt-free, unit-tested).

For each frame we fit **one** homogeneous deformation gradient ``F`` across all valid tracked
points (least-squares affine ``x' = F x + b``, reference → current), then form the left
Cauchy–Green tensor ``B = F Fᵀ``. Its eigendecomposition gives the principal stretches
``λ₁ ≥ λ₂`` and the principal directions *in the current (deformed) configuration*, from which we
define the linear strains ``εᵢ = λᵢ − 1``. The translation ``b`` carries rigid-body motion and is
discarded; a rigid rotation is absorbed into ``B``'s eigenvectors (they rotate with the body), so
the stretches are rotation-invariant.

The affine fit and the RANSAC routine are ported (verbatim, deterministic seed 0) from the
``affine_zones`` plugin; ``principal_decomposition_B`` differs only in using ``B = F Fᵀ`` (current
config) rather than ``C = Fᵀ F`` (reference config) — ``eig(B) == eig(C)`` so the stretches are
identical and only the direction vectors differ (B's live on the deformed frame the gauge draws).

Pure numpy — no Qt, no plugin API — so it runs headlessly and is unit-tested in
``tests/test_mts_uniaxial.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MIN_FIT_POINTS = 3  # an affine fit needs at least 3 correspondences


def fit_deformation_gradient(ref_pts, cur_pts, valid=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Least-squares homogeneous affine fit ``x' = F x + b`` over all valid points.

    ``ref_pts``/``cur_pts`` are ``(P, 2)`` reference and current positions. ``valid`` (optional) is
    a bool mask over the points; False entries (e.g. a track LK lost, carried forward) are excluded
    so dead tracks don't bias the fit. Returns ``(F 2×2, b 2,)`` or ``None`` if fewer than
    ``MIN_FIT_POINTS`` valid correspondences.
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    cur_pts = np.asarray(cur_pts, dtype=np.float64)
    if valid is None:
        idx = np.arange(ref_pts.shape[0])
    else:
        idx = np.where(np.asarray(valid, dtype=bool))[0]
    if idx.size < MIN_FIT_POINTS:
        return None
    src = ref_pts[idx]
    dst = cur_pts[idx]
    # Solve [x y 1] @ sol = [x' y'] in the least-squares sense; sol is 3×2.
    P = np.column_stack([src, np.ones(src.shape[0])])
    sol, *_ = np.linalg.lstsq(P, dst, rcond=None)
    F = np.array([[sol[0, 0], sol[1, 0]], [sol[0, 1], sol[1, 1]]])
    b = sol[2, :].copy()
    return F, b


def principal_decomposition_B(F) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Principal stretches and *current-configuration* directions from ``B = F Fᵀ``.

    Returns ``(λ₁, λ₂, v1, v2)`` with ``λ₁ ≥ λ₂`` the square roots of the eigenvalues of the left
    Cauchy–Green tensor ``B = F Fᵀ`` and ``v1, v2`` the matching unit eigenvectors (principal
    directions in the deformed frame). ``F = I`` ⇒ ``λ₁ = λ₂ = 1``. ``eig(B) == eig(C)`` so the
    stretches equal those from ``Fᵀ F``; only the directions differ.
    """
    F = np.asarray(F, dtype=np.float64)
    B = F @ F.T
    vals, vecs = np.linalg.eigh(B)  # ascending eigenvalues, orthonormal columns
    lam = np.sqrt(np.maximum(vals, 0.0))
    order = np.argsort(lam)[::-1]  # descending → λ₁ first
    lam = lam[order]
    vecs = vecs[:, order]
    return float(lam[0]), float(lam[1]), vecs[:, 0].copy(), vecs[:, 1].copy()


def ransac_affine(src, dst, sample_size=6, reproj=3.0, max_iters=2000, confidence=0.99
                  ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Deterministic RANSAC affine fit of ``src`` → ``dst`` (ported from ``affine_zones``).

    Each hypothesis is a least-squares affine fit over ``sample_size`` randomly drawn
    correspondences; inliers are points whose reprojection error is ``<= reproj`` px; the best
    consensus set wins and the model is refit on it. The iteration count adapts to the running best
    inlier ratio (capped at ``max_iters``) so ``confidence`` keeps OpenCV's meaning. Fixed seed, so
    the result is stable across re-runs. Returns ``(M, inliers)``: ``M`` is the 2×3 affine (``None``
    if no 3+-point consensus), ``inliers`` a bool array aligned to the input rows. ``n <= sample_size``
    ⇒ every point is an inlier (nothing to clean).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    s = max(MIN_FIT_POINTS, int(sample_size))
    P = np.column_stack([src, np.ones(n)])  # homogeneous src, reused for every residual eval
    if n <= s:
        sol, *_ = np.linalg.lstsq(P, dst, rcond=None)
        return sol.T, np.ones(n, dtype=bool)

    rng = np.random.default_rng(0)
    thresh = float(reproj)
    conf = min(max(float(confidence), 0.0), 1.0 - 1e-12)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    dynamic_iters = int(max_iters)

    it = 0
    while it < min(int(max_iters), dynamic_iters):
        it += 1
        idx = rng.choice(n, size=s, replace=False)
        sol, *_ = np.linalg.lstsq(P[idx], dst[idx], rcond=None)  # 3×2
        err = np.linalg.norm(P @ sol - dst, axis=1)
        inliers = err <= thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            w = count / n
            if w >= 1.0:
                break
            denom = np.log1p(-(w ** s))  # log(1 - wᵏ), strictly < 0 for 0 < w < 1
            dynamic_iters = int(np.ceil(np.log1p(-conf) / denom))

    if best_count < MIN_FIT_POINTS:
        return None, best_inliers
    sol, *_ = np.linalg.lstsq(P[best_inliers], dst[best_inliers], rcond=None)
    return sol.T, best_inliers


def eps_2_incompressible(eps_1) -> np.ndarray:
    """Theoretical lateral linear strain under incompressibility (isotropic uniaxial).

    For an incompressible material in uniaxial tension the transverse stretch is
    ``λ₂ = λ₁^(−1/2)``, so ``eps_2_ico = (1 + eps_1)^(−1/2) − 1`` — computed purely from the
    measured tensile strain ``eps_1``. NaN where ``1 + eps_1 <= 0`` (no real stretch).
    """
    eps_1 = np.asarray(eps_1, dtype=np.float64)
    base = 1.0 + eps_1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(base > 0.0, base ** -0.5 - 1.0, np.nan)


@dataclass
class KinematicsSeries:
    """Per-frame homogeneous kinematics; every array's leading axis is the cut frame index."""
    time_s: np.ndarray      # (frames,) image time relative to the reference (ref = 0.0), seconds
    lambda_1: np.ndarray    # (frames,) major principal stretch (tensile)
    lambda_2: np.ndarray    # (frames,) minor principal stretch (lateral)
    eps_1: np.ndarray       # (frames,) λ₁ − 1
    eps_2: np.ndarray       # (frames,) λ₂ − 1
    eps_2_ico: np.ndarray   # (frames,) incompressible prediction from eps_1 (NaN where undefined)
    angle_deg: np.ndarray   # (frames,) atan2(v1y, v1x) in degrees (tensile dir, image coords)
    n_points: np.ndarray    # (frames,) valid-point count used for each fit
    v1: np.ndarray          # (frames, 2) unit tensile direction (current config)
    v2: np.ndarray          # (frames, 2) unit lateral direction (current config)


def compute_series(coords, image_time_ms, status=None) -> KinematicsSeries:
    """Per-frame homogeneous kinematics from tracked coordinates.

    ``coords`` is ``(frames, points, 2)`` cut-indexed (``coords[0]`` = reference). ``image_time_ms``
    is the ``(frames,)`` image time of each frame (any zero — it is re-zeroed to the reference).
    ``status`` (optional) is ``(frames, points)`` uint8 (1 = tracked OK at that frame); points not
    OK at a frame are excluded from that frame's fit. Frames with fewer than ``MIN_FIT_POINTS``
    valid points yield a NaN row (``n_points`` records the count). Frame 0 is forced to ``F = I``
    exactly (λ = 1, eps = 0, v1 = e_x, v2 = e_y) — at zero deformation the eigenbasis is degenerate.
    """
    coords = np.asarray(coords, dtype=np.float64)
    frames = coords.shape[0]
    image_time_ms = np.asarray(image_time_ms, dtype=np.float64)

    time_s = (image_time_ms - image_time_ms[0]) / 1000.0
    lambda_1 = np.full(frames, np.nan)
    lambda_2 = np.full(frames, np.nan)
    angle_deg = np.full(frames, np.nan)
    n_points = np.zeros(frames, dtype=int)
    v1 = np.full((frames, 2), np.nan)
    v2 = np.full((frames, 2), np.nan)

    ref_pts = coords[0]
    for t in range(frames):
        valid = None if status is None else (np.asarray(status[t]) == 1)
        n_points[t] = ref_pts.shape[0] if valid is None else int(valid.sum())
        if t == 0:
            # Reference frame: F = I by definition; seed directions deterministically.
            lambda_1[t] = lambda_2[t] = 1.0
            v1[t], v2[t] = (1.0, 0.0), (0.0, 1.0)
            angle_deg[t] = 0.0
            continue
        fit = fit_deformation_gradient(ref_pts, coords[t], valid=valid)
        if fit is None:
            continue
        l1, l2, e1, e2 = principal_decomposition_B(fit[0])
        lambda_1[t], lambda_2[t] = l1, l2
        v1[t], v2[t] = e1, e2
        angle_deg[t] = np.degrees(np.arctan2(e1[1], e1[0]))

    eps_1 = lambda_1 - 1.0
    eps_2 = lambda_2 - 1.0
    return KinematicsSeries(
        time_s=time_s, lambda_1=lambda_1, lambda_2=lambda_2,
        eps_1=eps_1, eps_2=eps_2, eps_2_ico=eps_2_incompressible(eps_1),
        angle_deg=angle_deg, n_points=n_points, v1=v1, v2=v2,
    )
