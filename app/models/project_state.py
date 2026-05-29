import os
from typing import Optional

import numpy as np

from app.core import settings
from app.core.feature_detection import DEFAULT_GRID, DEFAULT_SHI_TOMASI
from app.core.image_sequence import ImageSequence
from app.core.roi import ROI
from app.core.tracking import DEFAULT_LK


class ProjectState:
    """Mutable per-session state. Holds the loaded sequence and the global frame indices.

    Index vocabulary:
      - global index: position in the full folder sequence (0 .. total_images-1)
      - cut index:    position within the reference..last range (0 .. n_cut-1), cut 0 = reference
    Invariant enforced by the setters: 0 <= reference_index <= last_index < total_images.
    """

    def __init__(self) -> None:
        self.sequence: Optional[ImageSequence] = None
        self.source_dir: Optional[str] = None
        self.reference_index: int = 0
        self.last_index: int = 0
        self.current_index: int = 0
        self.roi: Optional[ROI] = None

        # Built-in defaults merged with any persisted user defaults.
        self.shi_tomasi_params: dict = {**DEFAULT_SHI_TOMASI, **(settings.get_section("shi_tomasi") or {})}
        self.grid_params: dict = {**DEFAULT_GRID, **(settings.get_section("grid") or {})}
        self.lk_params: dict = {**DEFAULT_LK, **(settings.get_section("lk") or {})}
        self.features: Optional[np.ndarray] = None  # (N, 2) reference-frame seed points
        self.result = None  # TrackerResult, set after tracking
        self.active_mask: Optional[np.ndarray] = None  # (P,) bool, aligned to result points
        self.undo_stack: list = []  # previous active_mask snapshots for cleanup undo

    # ---- sequence -------------------------------------------------------
    def load_sequence(self, sequence: ImageSequence, source_dir: Optional[str]) -> None:
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
