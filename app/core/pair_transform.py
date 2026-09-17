"""Qt-free image-pair geometry. Pixel centers use OpenCV's integer coordinates.

The destination-to-reference linear map is L = R U: U is positive-definite symmetric
stretch, R is a proper rotation. Analysis restores C = R inv(U) R.T, not inv(L),
so alignment's rigid rotation is never reintroduced into the deformation gradient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import cv2
import numpy as np


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return int(value) if value.is_integer() else value


@dataclass(frozen=True)
class PairAlignment:
    """Immutable geometry plus remembered editor presets (excluded from fingerprints)."""

    translation: tuple = (0, 0)
    mode: str = "translation"
    scale: float = 1.0
    affine_stretch: tuple = ((1.0, 0.0), (0.0, 1.0))
    rotation_enabled: bool = False
    angle_degrees: float = 0.0  # Clockwise in image coordinates (y down).
    pivot: tuple | None = None  # Editor anchor in original destination pixel coordinates.

    def __post_init__(self):
        if self.mode not in ("translation", "scale", "affine"):
            raise ValueError("Unknown alignment mode")
        if type(self.rotation_enabled) is not bool:
            raise ValueError("Rotation enabled must be a boolean")
        if not isinstance(self.translation, (tuple, list)) or len(self.translation) != 2:
            raise ValueError("Alignment needs two offsets")
        shift = tuple(_number(v, "Offset") for v in self.translation)
        if self.pivot is not None:
            if not isinstance(self.pivot, (tuple, list)) or len(self.pivot) != 2:
                raise ValueError("Pivot needs two coordinates")
            object.__setattr__(self, "pivot", tuple(_number(v, "Pivot") for v in self.pivot))
        scale = _number(self.scale, "Scale")
        angle = _number(self.angle_degrees, "Rotation")
        if scale <= 0 or not math.isfinite(1.0 / scale):
            raise ValueError("Scale must be positive and invertible")
        try:
            raw = np.asarray(self.affine_stretch, dtype=object)
            if raw.shape != (2, 2):
                raise ValueError("Stretch must be a 2x2 matrix")
            stretch = np.array([[_number(v, "Stretch") for v in row] for row in raw])
        except (TypeError, OverflowError) as exc:
            raise ValueError("Invalid affine stretch") from exc
        if not np.allclose(stretch, stretch.T, rtol=0, atol=1e-12):
            raise ValueError("Affine stretch must be symmetric; rotation is stored separately")
        stretch = (stretch + stretch.T) * .5
        values = np.linalg.eigvalsh(stretch)
        if values[0] <= 0 or values[-1] / values[0] > 1e8:
            raise ValueError("Affine stretch must be positive definite and well conditioned")
        if not np.isfinite(np.linalg.inv(stretch)).all():
            raise ValueError("Affine stretch must be invertible")
        object.__setattr__(self, "translation", shift)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "angle_degrees", (angle + 180) % 360 - 180)
        object.__setattr__(self, "affine_stretch", tuple(tuple(float(v) for v in row) for row in stretch))

    @property
    def rotation(self):
        angle = math.radians(self.angle_degrees if self.rotation_enabled else 0)
        c, s = math.cos(angle), math.sin(angle)
        c, s = (0.0 if abs(v) < 1e-15 else v for v in (c, s))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    @property
    def stretch(self):
        if self.mode == "translation":
            return np.eye(2)
        if self.mode == "scale":
            return np.eye(2) * self.scale
        return np.array(self.affine_stretch, dtype=np.float64)

    @property
    def linear(self):
        return self.rotation @ self.stretch

    @property
    def correction(self):
        if np.array_equal(self.stretch, np.eye(2)):
            return np.eye(2)
        return self.rotation @ np.linalg.inv(self.stretch) @ self.rotation.T

    @property
    def integer_translation_only(self):
        return np.array_equal(self.linear, np.eye(2)) and all(float(v).is_integer() for v in self.translation)

    def matrix(self, destination_shape):
        """2x3 forward pixel-center map from original destination to reference."""
        h, w = destination_shape[:2]
        center = np.array([(w - 1) / 2, (h - 1) / 2])
        linear = self.linear
        return np.column_stack((linear, center + self.translation - linear @ center))

    def same_geometry(self, other):
        return self.translation == other.translation and np.array_equal(self.linear, other.linear)

    @property
    def scale_percent(self):
        return 100 * (math.sqrt(np.linalg.det(self.stretch)) if self.mode == "affine" else self.scale)

    def _preserve_pivot(self, updated, destination_shape):
        """Compensate a stretch edit without changing the stored center-based map."""
        if self.pivot is None or np.array_equal(self.linear, updated.linear):
            return updated
        if destination_shape is None:
            raise ValueError("Destination shape is required to scale about a pivot")
        h, w = destination_shape[:2]
        offset = np.asarray(self.pivot) - [(w - 1) / 2, (h - 1) / 2]
        shift = self.translation + (self.linear - updated.linear) @ offset
        return replace(updated, translation=tuple(shift))

    def with_mode(self, mode, destination_shape):
        return self._preserve_pivot(replace(self, mode=mode), destination_shape)

    def with_scale_percent(self, percent, destination_shape=None):
        value = _number(percent, "Scale percentage") / 100
        if self.mode == "affine":
            matrix = self.stretch * (value / (self.scale_percent / 100))
            updated = replace(self, affine_stretch=tuple(map(tuple, matrix)))
        else:
            updated = replace(self, scale=value)
        return self._preserve_pivot(updated, destination_shape)

    def drag_corner(self, shape, corner, target):
        """Anchor the chosen pivot or opposite corner and minimally change stretch.

        The symmetric minimum-Frobenius-norm update E satisfying E q = d is
        (d q.T + q d.T)/|q|² - (d.q) q q.T/|q|⁴. Rotation stays independent.
        Handles use image boundaries, half a pixel outside the corner pixel centers.
        """
        if self.mode == "translation":
            return self
        h, w = shape[:2]
        corners = np.array([[-.5, -.5], [w-.5, -.5], [w-.5, h-.5], [-.5, h-.5]])
        center = np.array([(w-1)/2, (h-1)/2])
        if self.pivot is None:
            q = corners[corner] - center
            anchor = center + self.translation - self.linear @ q
            desired = self.rotation.T @ (np.asarray(target) - anchor) / 2
        else:
            q = corners[corner] - self.pivot
            if q @ q <= 1e-12:
                return self  # A handle coincident with the fixed point cannot define a scale.
            anchor = self.matrix(shape) @ np.array([*self.pivot, 1])
            desired = self.rotation.T @ (np.asarray(target) - anchor)
        if self.mode == "scale":
            scale = float(desired @ q / (q @ q))
            updated = replace(self, scale=scale)
        else:
            d = desired - self.stretch @ q
            norm = q @ q
            change = (np.outer(d, q) + np.outer(q, d)) / norm - (d @ q) * np.outer(q, q) / norm**2
            updated = replace(self, affine_stretch=tuple(map(tuple, self.stretch + change)))
        if self.pivot is not None:
            return self._preserve_pivot(updated, shape)
        shift = anchor + updated.linear @ q - center
        return replace(updated, translation=tuple(shift))

    def metadata(self):
        return asdict(self)

    @classmethod
    def from_metadata(cls, value):
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) not in (fields, fields - {"pivot"}):
            raise ValueError("Invalid pair alignment metadata")
        try:
            return cls(**value)
        except (TypeError, OverflowError, np.linalg.LinAlgError) as exc:
            raise ValueError("Invalid pair alignment metadata") from exc


def largest_valid_rectangle(mask):
    """Largest rectangle in an affine footprint (each valid row is contiguous).

    Scan pairs of rows using NumPy cumulative interval intersections. O(H² + HW),
    O(H + W) auxiliary memory; transpose tall masks to bound the quadratic dimension.
    Tie order in original coordinates: topmost, leftmost, widest.
    """
    transposed = mask.shape[0] > mask.shape[1]
    if transposed:
        mask = mask.T
    height, width = mask.shape
    nonempty = mask.any(axis=1)
    left = np.where(nonempty, mask.argmax(axis=1), width)
    right = np.where(nonempty, width - mask[:, ::-1].argmax(axis=1), 0)
    best, best_key = (0, 0, 0, 0), (0, 0, 0, 0)
    for top in np.flatnonzero(nonempty):
        ls = np.maximum.accumulate(left[top:])
        rs = np.minimum.accumulate(right[top:])
        widths = np.maximum(0, rs - ls)
        heights = np.arange(1, height - top + 1)
        areas = widths * heights
        maximum = int(areas.max())
        if maximum == 0 or maximum < best_key[0]:
            continue
        for i in np.flatnonzero(areas == maximum):
            rect = (int(ls[i]), int(top), int(widths[i]), int(heights[i]))
            if transposed:
                rect = (rect[1], rect[0], rect[3], rect[2])
            x, y, w, h = rect
            key = (w*h, -y, -x, w)
            if key > best_key:
                best, best_key = rect, key
    return best


def alignment_crop(reference_shape, destination_shape, alignment):
    """Reference-space crop whose interpolation never samples border fill."""
    rh, rw = reference_shape[:2]
    dh, dw = destination_shape[:2]
    if alignment.integer_translation_only:
        dx, dy = map(int, alignment.translation)
        x, y = max(0, dx), max(0, dy)
        return x, y, max(0, min(rw, dx+dw)-x), max(0, min(rh, dy+dh)-y)
    support = cv2.warpAffine(np.ones((dh, dw), np.float32), alignment.matrix(destination_shape),
                             (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return largest_valid_rectangle(support == 1.0)


def compose_alignment_affines(F, current, reference):
    """Restore scale/shear in a residual gradient, in rotation-aligned coordinates."""
    return np.asarray(current) @ np.asarray(F) @ np.linalg.inv(reference)
