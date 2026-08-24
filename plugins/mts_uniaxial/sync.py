"""Temporal synchronization between the high-frequency sensor stream and the image timeline.

The two streams are aligned by their own elapsed-from-zero timelines plus a single user offset;
wall-clock stamps are never used. The sensor is interpolated *onto* each image's timestamp, so
every tracked frame carries exactly one (displacement, force) value.

Offset sign convention (locked):
    positive offset = the sensor recorded LATER than the images.
    The offset is ADDED to the sensor timeline: value = np.interp(image_t, sensor_t + offset, ch).
    With offset > 0 a given image frame therefore samples *earlier* sensor data.

Qt-free; all functions take/return numpy arrays.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

FORCE_CHANNELS = ("A", "B", "average")

#: The locked offset sign convention — shown verbatim in the UI and recorded in the manifest.
OFFSET_CONVENTION = (
    "Sensor time shift (ms) — positive = sensor recorded LATER than images. "
    "The offset is added to the sensor timeline: value = interp(image_t, sensor_t + offset)."
)


def sensor_time_ms(sensor) -> np.ndarray:
    """Sensor elapsed time in milliseconds (the file stores seconds)."""
    return sensor.time_s * 1000.0


def composite_displacement(sensor) -> np.ndarray:
    """Displacement is always the sum of both clamp displacements."""
    return sensor.disp_a + sensor.disp_b


def composite_force(sensor, channel: str) -> np.ndarray:
    """Force per the user-selected channel: clamp ``A``, clamp ``B``, or their ``average``."""
    if channel == "A":
        return sensor.force_a
    if channel == "B":
        return sensor.force_b
    if channel == "average":
        return (sensor.force_a + sensor.force_b) / 2.0
    raise ValueError(f"Unknown force channel {channel!r} (expected one of {FORCE_CHANNELS})")


def interp_to_images(
    image_time_ms: np.ndarray,
    sensor_time_ms: np.ndarray,
    channel: np.ndarray,
    offset_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a sensor ``channel`` onto the image timeline.

    Returns ``(values, in_range)``. ``np.interp`` clamps to the sensor endpoints outside the
    covered span, so every image frame gets a value; ``in_range`` flags which frames actually lie
    within sensor coverage (the rest are clamped, not real data).
    """
    image_time_ms = np.asarray(image_time_ms, dtype=np.float64)
    sensor_time_ms = np.asarray(sensor_time_ms, dtype=np.float64)
    channel = np.asarray(channel, dtype=np.float64)
    try:
        offset_ms = float(offset_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("Synchronization offset must be numeric") from exc
    if image_time_ms.ndim != 1 or sensor_time_ms.ndim != 1 or channel.ndim != 1:
        raise ValueError("Synchronization inputs must be one-dimensional")
    if sensor_time_ms.size == 0 or channel.shape != sensor_time_ms.shape:
        raise ValueError("Sensor timestamps and channel must be non-empty and equal-length")
    if not (
        np.isfinite(image_time_ms).all()
        and np.isfinite(sensor_time_ms).all()
        and np.isfinite(channel).all()
        and np.isfinite(offset_ms)
    ):
        raise ValueError("Synchronization inputs must be finite")
    if np.any(np.diff(sensor_time_ms) <= 0):
        raise ValueError("Sensor timestamps must be strictly increasing")
    xs = sensor_time_ms + offset_ms
    values = np.interp(image_time_ms, xs, channel)
    in_range = (image_time_ms >= xs[0]) & (image_time_ms <= xs[-1])
    return values, in_range


def sensor_index_to_image_index(
    sensor_index: int,
    sensor_time_ms: np.ndarray,
    image_time_ms: np.ndarray,
    offset_ms: float,
) -> int:
    """Map a sensor sample index to the index of the nearest image frame in time.

    The reference must be a real frame (it carries the ROI), so this picks the nearest image
    rather than interpolating a fractional position.
    """
    sensor_time_ms = np.asarray(sensor_time_ms, dtype=np.float64)
    image_time_ms = np.asarray(image_time_ms, dtype=np.float64)
    try:
        offset_ms = float(offset_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("Synchronization offset must be numeric") from exc
    if not 0 <= sensor_index < sensor_time_ms.size:
        raise IndexError(f"Sensor index {sensor_index} is out of range")
    if image_time_ms.size == 0 or not (
        np.isfinite(sensor_time_ms).all()
        and np.isfinite(image_time_ms).all()
        and np.isfinite(offset_ms)
    ):
        raise ValueError("Timeline inputs must be non-empty and finite")
    t = sensor_time_ms[sensor_index] + offset_ms
    return int(np.argmin(np.abs(image_time_ms - t)))
