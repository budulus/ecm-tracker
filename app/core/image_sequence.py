import os
import re
from collections import OrderedDict
from typing import List, Sequence, Union

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(name: str):
    """Numeric-aware key so that 'img_2' sorts before 'img_10'."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def discover(path_or_files: Union[str, Sequence[str]]) -> List[str]:
    """Return naturally sorted, supported image paths from a directory or a list of files."""
    if isinstance(path_or_files, (list, tuple)):
        files = [f for f in path_or_files if _is_supported(f)]
    else:
        directory = path_or_files
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if _is_supported(f)
        ]
    files.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return files


def _normalize_to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Coerce any decoded image to a 3-channel uint8 BGR array (deterministic across frames)."""
    if img.dtype == np.uint16:
        img = (img // 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


class ImageSequence:
    """Naturally ordered image paths with on-demand decoding and a small LRU frame cache."""

    def __init__(self, paths: List[str], cache_size: int = 8):
        if not paths:
            raise ValueError("ImageSequence requires at least one image path")
        self.paths = list(paths)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.paths)

    def load_bgr(self, index: int) -> np.ndarray:
        """Return the frame at `index` as a 3-channel uint8 BGR array."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        raw = cv2.imread(self.paths[index], cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Failed to load image: {self.paths[index]}")
        img = _normalize_to_bgr_u8(raw)

        self._cache[index] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def load_gray(self, index: int) -> np.ndarray:
        """Return the frame at `index` as a single-channel uint8 grayscale array."""
        return cv2.cvtColor(self.load_bgr(index), cv2.COLOR_BGR2GRAY)
