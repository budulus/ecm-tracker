from typing import Optional

import cv2
import numpy as np

from app.core.roi import ROI

DEFAULT_SHI_TOMASI = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=7.0,
    blockSize=7,
    useHarrisDetector=False,
    k=0.04,
)

DEFAULT_GRID = dict(spacing_x=20, spacing_y=20)


def shi_tomasi(gray: np.ndarray, mask: Optional[np.ndarray], params: dict) -> np.ndarray:
    """Shi-Tomasi / Harris corners inside `mask`. Returns (N, 2) float32 image-space points."""
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **params)
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def regular_grid(roi: ROI, spacing_x: float, spacing_y: float) -> np.ndarray:
    """Grid of points (spacing_x, spacing_y apart) clipped to the ROI polygon. (N, 2) float32."""
    if not roi.is_complete or spacing_x <= 0 or spacing_y <= 0:
        return np.empty((0, 2), dtype=np.float32)

    corners = np.array(roi.corners, dtype=np.float32)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    xs = np.arange(x0, x1 + 1e-6, spacing_x)
    ys = np.arange(y0, y1 + 1e-6, spacing_y)

    pts = [(float(x), float(y)) for y in ys for x in xs if roi.contains(x, y)]
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)
