"""Stable Qt-free SDK helpers. Import these instead of implementation modules.

fit_affine: (N,2) reference/current arrays -> (eligible indices, F, translation) or None.
Rejects fewer than three finite points or geometry condition >= 1e4.
principal_stretches: B=F F.T, descending stretches, current-image unit axes; axes NaN
for equal stretches. Rejects det(F)<=0 or condition(F)>1e8.
compute_series: (frames,points,2), milliseconds, optional 0/1 status -> aligned
engineering strains lambda-1. Invalid rows/undefined axes are NaN, not zero.
ransac_affine: deterministic; returned inliers satisfy the returned model's threshold.
ROI: plugin-owned polygon helper. Never mutates the host ROI.
atomic_open/atomic_save_npy: atomic replacement of ONE file, not a multi-file transaction.
"""
from app.core.affine import (
    fit_affine, principal_stretches, principal_directions_defined, ransac_affine,
)
from app.core.kinematics import (
    MIN_FIT_POINTS, KinematicsSeries, compute_series, eps_2_incompressible,
    fit_deformation_gradient, principal_decomposition_B,
)
from app.core.roi import ROI
from app.core.atomic_io import atomic_open, atomic_save_npy
