import math
from typing import Optional

import numpy as np

from app.core import settings
from app.core.feature_detection import DEFAULT_GRID, DEFAULT_SHI_TOMASI
from app.core.image_sequence import ImageSequence
from app.core.roi import ROI
from app.core.tracking import DEFAULT_LK

# Marker-rendering preferences (display-only, persisted under the "display" settings section).
DEFAULT_DISPLAY = dict(
    show_markers=True,
    marker_size=3,        # marker radius in screen px (~ the previous hardcoded 2.5)
    marker_opacity=100,   # percent, 0-100 (100 preserves the original look)
    show_window_box=False,
    show_roi=True,
)


def _validated(defaults: dict, values, ranges: dict) -> dict:
    """Merge known, correctly typed finite values; replace invalid persisted input with defaults."""
    values = values if isinstance(values, dict) else {}
    result = dict(defaults)
    for key, default in defaults.items():
        if key not in values:
            continue
        value = values[key]
        if isinstance(default, bool):
            if isinstance(value, bool):
                result[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        if isinstance(default, int) and not number.is_integer():
            continue
        low, high = ranges.get(key, (None, None))
        if (low is not None and number < low) or (high is not None and number > high):
            continue
        result[key] = int(value) if isinstance(default, int) else number
    return result


def validate_shi_tomasi(values) -> dict:
    return _validated(
        DEFAULT_SHI_TOMASI,
        values,
        {
            "maxCorners": (1, 1_000_000),
            "qualityLevel": (0.0001, 1.0),
            "minDistance": (0.0, 1000.0),
            "blockSize": (1, 99),
            "k": (0.0, 1.0),
        },
    )


def validate_grid(values) -> dict:
    return _validated(DEFAULT_GRID, values, {"spacing_x": (1, 1000), "spacing_y": (1, 1000)})


def validate_lk(values) -> dict:
    result = _validated(
        DEFAULT_LK,
        values,
        {
            "win_size": (3, 201),
            "max_level": (0, 10),
            "max_iter": (1, 1000),
            "epsilon": (0.0001, 10.0),
            "flags": (0, 8),
            "min_eig_threshold": (0.0, 1.0),
        },
    )
    # OpenCV defines bit 4 as USE_INITIAL_FLOW, which this app intentionally cannot select because
    # it does not supply a separate initial next-point estimate. Bit 8 switches the error output
    # to minimum eigenvalues; all other values are either meaningless here or unsafe.
    if result["flags"] not in (0, 8):
        result["flags"] = DEFAULT_LK["flags"]
    return result


def validate_display(values) -> dict:
    return _validated(
        DEFAULT_DISPLAY,
        values,
        {"marker_size": (1, 15), "marker_opacity": (0, 100)},
    )


class ProjectState:
    """Mutable per-session state. Holds the loaded sequence and the global frame indices.

    Index vocabulary:
      - global index: position in the full folder sequence (0 .. total_images-1)
      - cut index:    position within the reference..last range (0 .. n_cut-1), cut 0 = reference
    Invariant enforced by the setters: 0 <= reference_index <= last_index < total_images.
    """

    def __init__(self) -> None:
        self.revision = 0
        self.sequence: Optional[ImageSequence] = None
        self.source_dir: Optional[str] = None
        self.reference_index: int = 0
        self.last_index: int = 0
        self.current_index: int = 0
        self.roi: Optional[ROI] = None

        # Built-in defaults merged with any persisted user defaults.
        self.shi_tomasi_params: dict = validate_shi_tomasi(settings.get_section("shi_tomasi"))
        self.grid_params: dict = validate_grid(settings.get_section("grid"))
        self.lk_params: dict = validate_lk(settings.get_section("lk"))
        self.display_params: dict = validate_display(settings.get_section("display"))
        self.features: Optional[np.ndarray] = None  # (N, 2) reference-frame seed points
        self.result = None  # TrackerResult, set after tracking
        self.active_mask: Optional[np.ndarray] = None  # (P,) bool, aligned to result points
        self.undo_stack: list = []  # previous active_mask snapshots for cleanup undo

    def touch(self) -> None:
        """Advance scientific state; display-only navigation does not change this revision."""
        self.revision += 1

    # ---- sequence -------------------------------------------------------
    def load_sequence(self, sequence: ImageSequence, source_dir: Optional[str]) -> None:
        self.touch()
        self.sequence = sequence
        self.source_dir = source_dir
        self.reference_index = 0
        self.last_index = len(sequence) - 1
        self.current_index = 0
        self.roi = None
        self.features = None
        self.result = None
        self.active_mask = None
        self.undo_stack = []

    def image_size(self):
        """Return (height, width) of the sequence frames, or None if no sequence."""
        if self.sequence is None:
            return None
        h, w = self.sequence.load_bgr(0).shape[:2]
        return h, w

    @property
    def total_images(self) -> int:
        return len(self.sequence) if self.sequence is not None else 0

    @property
    def has_sequence(self) -> bool:
        return self.sequence is not None

    # ---- index helpers --------------------------------------------------
    @property
    def n_cut(self) -> int:
        """Number of frames in the reference..last range (both endpoints inclusive)."""
        return self.last_index - self.reference_index + 1

    def global_to_cut(self, global_index: int) -> int:
        return global_index - self.reference_index

    def cut_to_global(self, cut_index: int) -> int:
        return self.reference_index + cut_index

    @property
    def current_in_range(self) -> bool:
        return self.reference_index <= self.current_index <= self.last_index

    @property
    def on_reference_frame(self) -> bool:
        return self.has_sequence and self.current_index == self.reference_index

    # ---- validated setters ---------------------------------------------
    def set_current(self, index: int) -> None:
        self.current_index = max(0, min(index, self.total_images - 1))

    def set_reference(self, index: int) -> None:
        index = max(0, min(index, self.last_index))
        self.reference_index = index

    def set_last(self, index: int) -> None:
        index = max(self.reference_index, min(index, self.total_images - 1))
        self.last_index = index
