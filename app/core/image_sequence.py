import hashlib
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


def _normalize_to_bgr_u8(img: np.ndarray, float_scale=None) -> np.ndarray:
    """Coerce any decoded image to a 3-channel uint8 BGR array (deterministic across frames)."""
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    elif img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif np.issubdtype(img.dtype, np.floating):
        if not np.isfinite(img).all():
            raise ValueError("Floating-point image contains NaN or infinity")
        lo = float(img.min()) if img.size else 0.0
        hi = float(img.max()) if img.size else 0.0
        # The common floating image convention is [0, 1]. Preserve ordinary [0, 255]
        # floating images without per-frame contrast normalization, which would destabilize LK.
        unit_range = (lo >= 0.0 and hi <= 1.0)
        if float_scale == 255 and not unit_range:
            raise ValueError("Float image exceeds the sequence [0, 1] intensity convention")
        if float_scale == 255 or (float_scale is None and unit_range):
            img = np.rint(img * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels != 3:
            raise ValueError(f"Unsupported image channel count: {channels}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return np.ascontiguousarray(img)


class ImageSequence:
    """Naturally ordered image paths with on-demand decoding and a small LRU frame cache."""

    def __init__(self, paths: List[str], cache_size: int = 8):
        if not paths:
            raise ValueError("ImageSequence requires at least one image path")
        self.paths = list(paths)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self._frame_shape = None
        self._float_scale = None
        self._fingerprint = None
        self._file_signatures = None

    def __len__(self) -> int:
        return len(self.paths)

    def load_bgr(self, index: int) -> np.ndarray:
        """Return the frame at `index` as a 3-channel uint8 BGR array."""
        if not 0 <= index < len(self.paths):
            raise IndexError(f"Frame index {index} is outside 0..{len(self.paths) - 1}")
        if self._file_signatures is not None:
            stat = os.stat(self.paths[index])
            current = (stat.st_size, stat.st_mtime_ns)
            if current != self._file_signatures[index]:
                raise IOError(
                    f"Image changed on disk after the sequence was loaded: {self.paths[index]}"
                )
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        raw = cv2.imread(self.paths[index], cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Failed to load image: {self.paths[index]}")
        try:
            if np.issubdtype(raw.dtype, np.floating) and self._float_scale is None:
                self._float_scale = 255 if np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1 else 1
            img = _normalize_to_bgr_u8(raw, self._float_scale)
        except ValueError as exc:
            raise ValueError(f"Invalid image {self.paths[index]}: {exc}") from exc

        self._check_frame_shape(img, self.paths[index])
        img.setflags(write=False)

        self._cache[index] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def _check_frame_shape(self, img: np.ndarray, path: str) -> None:
        """Ordinary sequences require one shape; pair sources may override this check."""
        if self._frame_shape is None:
            self._frame_shape = img.shape
        elif img.shape != self._frame_shape:
            raise ValueError(
                f"Frame dimensions changed at {path}: got {img.shape[:2]}, "
                f"expected {self._frame_shape[:2]}"
            )

    def load_gray(self, index: int) -> np.ndarray:
        """Return the frame at `index` as a single-channel uint8 grayscale array."""
        return cv2.cvtColor(self.load_bgr(index), cv2.COLOR_BGR2GRAY)

    def validate_all(self, progress_cb=None) -> bool:
        """Decode every frame, enforce a common shape, and compute an ordered content fingerprint.

        Returns ``False`` when ``progress_cb(done, total)`` requests cancellation. Validation is
        performed before a sequence is installed into project state, making a failed load
        transactional. The LRU still bounds memory while this walks large sequences.
        """
        digest = hashlib.sha256()
        digest.update(f"ecmtracker-sequence-v1:{len(self.paths)}".encode("ascii"))
        signatures = []
        for index in range(len(self.paths)):
            before = os.stat(self.paths[index])
            image = self.load_bgr(index)
            digest.update(f"|{index}:{image.shape}".encode("ascii"))
            digest.update(memoryview(image))
            stat = os.stat(self.paths[index])
            if (before.st_size, before.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise IOError(f"Image changed while it was being validated: {self.paths[index]}")
            signatures.append((stat.st_size, stat.st_mtime_ns))
            if progress_cb is not None and progress_cb(index + 1, len(self.paths)):
                return False
        self._fingerprint = digest.hexdigest()
        self._file_signatures = signatures
        return True

    @property
    def fingerprint(self):
        """SHA-256 identity of the validated, ordered normalized frame content, or ``None``."""
        return self._fingerprint
