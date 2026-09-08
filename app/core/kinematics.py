"""Homogeneous in-plane kinematics from tracked points (Qt-free, unit-tested).

For each frame we fit **one** homogeneous deformation gradient ``F`` across all valid tracked
points (least-squares affine ``x' = F x + b``, reference → current), then form the left
Cauchy–Green tensor ``B = F Fᵀ``. Its eigendecomposition gives the principal stretches
``λ₁ ≥ λ₂`` and the principal directions *in the current (deformed) configuration*, from which we
define the linear strains ``εᵢ = λᵢ − 1``. The translation ``b`` carries rigid-body motion and is
discarded; a rigid rotation is absorbed into ``B``'s eigenvectors (they rotate with the body), so
the stretches are rotation-invariant.

The affine fit and deterministic RANSAC routine are shared with ``affine_zones`` through the
guarded Qt-free implementation in :mod:`app.core.affine`. ``principal_decomposition_B`` uses
``B = F Fᵀ`` (current config); its directions therefore live on the deformed frame the gauge draws.

Pure numpy — no Qt, no plugin API — so it runs headlessly and is unit-tested in
``tests/test_mts_uniaxial.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.core.affine import (
    fit_affine,
    principal_directions_defined,
    principal_stretches,
    ransac_affine as _ransac_affine,
)

MIN_FIT_POINTS = 3  # an affine fit needs at least 3 correspondences


def fit_deformation_gradient(ref_pts, cur_pts, valid=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Least-squares homogeneous affine fit ``x' = F x + b`` over all valid points.

    ``ref_pts``/``cur_pts`` are ``(P, 2)`` reference and current positions. ``valid`` (optional) is
    a bool mask over the points; False entries (e.g. a track LK lost, carried forward) are excluded
    so dead tracks don't bias the fit. Returns ``(F 2×2, b 2,)`` or ``None`` if fewer than
    ``MIN_FIT_POINTS`` valid correspondences.
    """
    fit = fit_affine(ref_pts, cur_pts, valid)
    return None if fit is None else (fit[1], fit[2])


def principal_decomposition_B(F) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Principal stretches and *current-configuration* directions from ``B = F Fᵀ``.

    Returns ``(λ₁, λ₂, v1, v2)`` with ``λ₁ ≥ λ₂`` the square roots of the eigenvalues of the left
    Cauchy–Green tensor ``B = F Fᵀ`` and ``v1, v2`` the matching unit eigenvectors (principal
    directions in the deformed frame). ``F = I`` ⇒ ``λ₁ = λ₂ = 1``. ``eig(B) == eig(C)`` so the
    stretches equal those from ``Fᵀ F``; only the directions differ.
    """
    return principal_stretches(F)


def ransac_affine(src, dst, sample_size=6, reproj=3.0, max_iters=2000, confidence=0.99
                  ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Deterministic RANSAC affine fit of ``src`` → ``dst`` (ported from ``affine_zones``).

    Each hypothesis is a least-squares affine fit over ``sample_size`` randomly drawn
    correspondences; inliers are points whose reprojection error is ``<= reproj`` px; the best
    consensus set wins and the model is refit on it. The iteration count adapts to the running best
    inlier ratio (capped at ``max_iters``) so ``confidence`` keeps OpenCV's meaning. Fixed seed, so
    the result is stable across re-runs. Returns ``(M, inliers)``: ``M`` is the 2×3 affine (``None``
    if no 3+-point consensus), ``inliers`` a bool array aligned to the input rows. Even when
    n <= sample_size, returned inliers must pass the returned model threshold.
    """
    return _ransac_affine(
        src,
        dst,
        sample_size=sample_size,
        reproj=reproj,
        max_iters=max_iters,
        confidence=confidence,
    )


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
    lambda_1: np.ndarray    # (frames,) major principal stretch (not necessarily axial)
    lambda_2: np.ndarray    # (frames,) minor principal stretch (not necessarily transverse)
    eps_1: np.ndarray       # (frames,) λ₁ − 1
    eps_2: np.ndarray       # (frames,) λ₂ − 1
    eps_2_ico: np.ndarray   # (frames,) incompressible prediction from eps_1 (NaN where undefined)
    angle_deg: np.ndarray   # (frames,) atan2(v1y, v1x) in degrees (major principal axis, image coords)
    n_points: np.ndarray    # (frames,) valid-point count used for each fit
    v1: np.ndarray          # (frames, 2) unit major principal direction (current config)
    v2: np.ndarray          # (frames, 2) unit minor principal direction (current config)
    axial_lambda: Optional[np.ndarray] = None  # |F a0|; legacy major stretch when axis unspecified
    fit_reason: tuple = ()
    residual_rms: Optional[np.ndarray] = None


def compute_series(coords, image_time_ms, status=None, *, loading_axis_deg=None) -> KinematicsSeries:
    """Per-frame homogeneous kinematics from tracked coordinates.

    ``coords`` is ``(frames, points, 2)`` cut-indexed (``coords[0]`` = reference). ``image_time_ms``
    is the ``(frames,)`` image time of each frame (any zero — it is re-zeroed to the reference).
    ``status`` (optional) is ``(frames, points)`` uint8 (1 = tracked OK at that frame); points not
    OK at a frame are excluded from that frame's fit. Frames with fewer than ``MIN_FIT_POINTS``
    valid well-conditioned points yield a NaN row (``n_points`` counts eligible rows).
    The reference obeys the same validity rules; isotropic stretch has undefined directions.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 2 or coords.shape[0] == 0:
        raise ValueError("coords must have nonempty shape (frames, points, 2)")
    frames = coords.shape[0]
    if status is not None:
        status = np.asarray(status)
        if status.shape != coords.shape[:2] or not np.isin(status, (0, 1)).all():
            raise ValueError("status must be a matching (frames, points) 0/1 array")
    image_time_ms = np.asarray(image_time_ms, dtype=np.float64)

    if image_time_ms.shape != (frames,) or not np.isfinite(image_time_ms).all() or np.any(np.diff(image_time_ms) < 0):
        raise ValueError("image_time_ms must be finite, nondecreasing, and frame-aligned")
    time_s = (image_time_ms - image_time_ms[0]) / 1000.0
    if loading_axis_deg is not None and not np.isfinite(loading_axis_deg):
        raise ValueError("Loading axis angle must be finite")
    angle = np.radians(loading_axis_deg or 0.0)
    axis = np.array([np.cos(angle), np.sin(angle)])
    axial = np.full(frames, np.nan)
    lambda_1 = np.full(frames, np.nan)
    lambda_2 = np.full(frames, np.nan)
    angle_deg = np.full(frames, np.nan)
    n_points = np.zeros(frames, dtype=int)
    reasons = ["insufficient_points"] * frames
    residual_rms = np.full(frames, np.nan)
    v1 = np.full((frames, 2), np.nan)
    v2 = np.full((frames, 2), np.nan)

    ref_pts = coords[0]
    for t in range(frames):
        valid = np.isfinite(ref_pts).all(axis=1) & np.isfinite(coords[t]).all(axis=1)
        if status is not None:
            valid &= (status[0] == 1) & (status[t] == 1)
        n_points[t] = int(valid.sum())
        fit = fit_deformation_gradient(ref_pts, coords[t], valid=valid)
        if fit is None:
            if n_points[t] >= MIN_FIT_POINTS:
                reasons[t] = "ill_conditioned_reference"
            continue
        if t == 0:
            fit = (np.eye(2), np.zeros(2))  # Exact identity only AFTER geometry/status validation.
        try:
            l1, l2, e1, e2 = principal_decomposition_B(fit[0])
        except ValueError:
            reasons[t] = "nonphysical_deformation"
            continue
        residual = ref_pts[valid] @ fit[0].T + fit[1] - coords[t, valid]
        residual_rms[t] = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        reasons[t] = "ok"
        lambda_1[t], lambda_2[t] = l1, l2
        axial[t] = l1 if loading_axis_deg is None else np.linalg.norm(fit[0] @ axis)
        if not principal_directions_defined(l1, l2):
            # At isotropic stretch the eigenspace is degenerate: reporting an angle would turn
            # numerical noise into a seemingly physical direction. Keep stretches, mark axes NaN.
            continue
        v1[t], v2[t] = e1, e2
        angle_deg[t] = np.degrees(np.arctan2(e1[1], e1[0]))

    eps_1 = lambda_1 - 1.0
    eps_2 = lambda_2 - 1.0
    return KinematicsSeries(
        time_s=time_s, lambda_1=lambda_1, lambda_2=lambda_2,
        eps_1=eps_1, eps_2=eps_2, eps_2_ico=eps_2_incompressible(axial - 1),
        angle_deg=angle_deg, n_points=n_points, v1=v1, v2=v2, axial_lambda=axial, fit_reason=tuple(reasons), residual_rms=residual_rms,
    )
