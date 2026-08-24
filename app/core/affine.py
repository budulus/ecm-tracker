"""Numerically guarded 2-D affine fitting shared by analysis plugins.

All routines are Qt-free. Fits reject non-finite, collinear, or otherwise rank-deficient point
sets instead of returning plausible-looking deformation values from an underdetermined system.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

MIN_AFFINE_POINTS = 3


def principal_directions_defined(lambda_1: float, lambda_2: float) -> bool:
    """Whether distinct stretches define a meaningful eigenvector basis."""
    return not np.isclose(lambda_1, lambda_2, rtol=1e-6, atol=1e-9)


def fit_affine(src, dst, valid=None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit ``dst = src @ F.T + b`` and return ``(indices, F, b)``, or ``None``.

    Centering the coordinates before solving improves conditioning. At least three finite,
    non-collinear correspondences are required; ``indices`` maps the fit rows to the input rows.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape:
        raise ValueError("Affine inputs must be matching (N, 2) arrays")

    keep = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (src.shape[0],):
            raise ValueError("Affine validity mask must have shape (N,)")
        keep &= valid
    indices = np.flatnonzero(keep)
    if indices.size < MIN_AFFINE_POINTS:
        return None

    x = src[indices]
    y = dst[indices]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    if np.linalg.matrix_rank(xc) < 2:
        return None

    linear_t, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    F = linear_t.T
    b = y_mean - F @ x_mean
    if not (np.isfinite(F).all() and np.isfinite(b).all()):
        return None
    return indices, F, b


def principal_stretches(F) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return principal stretches/directions from ``B = F F.T`` with deterministic signs."""
    F = np.asarray(F, dtype=np.float64)
    if F.shape != (2, 2) or not np.isfinite(F).all():
        raise ValueError("F must be a finite 2x2 matrix")
    B = F @ F.T
    vals, vecs = np.linalg.eigh(B)
    lam = np.sqrt(np.maximum(vals, 0.0))
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    vecs = vecs[:, order]
    # Eigenvector signs are arbitrary. A deterministic convention prevents needless 180-degree
    # flips in tables and plots while preserving their axis (modulo-pi) meaning.
    for column in range(2):
        vector = vecs[:, column]
        dominant = int(np.argmax(np.abs(vector)))
        if vector[dominant] < 0:
            vecs[:, column] *= -1
    return float(lam[0]), float(lam[1]), vecs[:, 0].copy(), vecs[:, 1].copy()


def ransac_affine(
    src,
    dst,
    *,
    sample_size=6,
    reproj=3.0,
    max_iters=2000,
    confidence=0.99,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Deterministic affine RANSAC that rejects degenerate samples and non-finite rows."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape:
        raise ValueError("RANSAC inputs must be matching (N, 2) arrays")
    n = src.shape[0]
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    eligible = np.flatnonzero(finite)
    output = np.zeros(n, dtype=bool)
    if eligible.size < MIN_AFFINE_POINTS:
        return None, output

    s = max(MIN_AFFINE_POINTS, int(sample_size))
    if eligible.size <= s:
        fit = fit_affine(src, dst, finite)
        if fit is None:
            return None, output
        indices, F, b = fit
        output[indices] = True
        return np.column_stack([F, b]), output

    threshold = max(0.0, float(reproj))
    limit = max(1, int(max_iters))
    conf = min(max(float(confidence), 0.0), 1.0 - 1e-12)
    rng = np.random.default_rng(0)
    best = np.zeros(n, dtype=bool)
    best_count = 0
    best_error = np.inf
    dynamic_limit = limit

    iteration = 0
    while iteration < min(limit, dynamic_limit):
        iteration += 1
        sample = rng.choice(eligible, size=s, replace=False)
        sample_mask = np.zeros(n, dtype=bool)
        sample_mask[sample] = True
        fit = fit_affine(src, dst, sample_mask)
        if fit is None:
            continue
        _indices, F, b = fit
        predicted = src[eligible] @ F.T + b
        residual = np.linalg.norm(predicted - dst[eligible], axis=1)
        inliers = np.zeros(n, dtype=bool)
        inliers[eligible] = residual <= threshold
        count = int(inliers.sum())
        error = float(residual[residual <= threshold].sum()) if count else np.inf
        if count > best_count or (count == best_count and error < best_error):
            best, best_count, best_error = inliers, count, error
            ratio = count / eligible.size
            if ratio >= 1.0:
                break
            success = ratio**s
            if success > 0.0 and conf > 0.0:
                dynamic_limit = max(
                    1,
                    int(np.ceil(np.log1p(-conf) / np.log1p(-success))),
                )

    if best_count < MIN_AFFINE_POINTS:
        return None, best
    fit = fit_affine(src, dst, best)
    if fit is None:
        return None, best
    _indices, F, b = fit
    return np.column_stack([F, b]), best
