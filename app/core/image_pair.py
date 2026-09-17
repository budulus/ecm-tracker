"""Immutable aligned pairs and portable setups. Original pixels stay unchanged.

Output coordinates start at the shared crop's upper-left corner in reference space.
Integer translations retain the original, exact slicing and v1 fingerprint algorithm.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import cv2
import numpy as np

from app.core.atomic_io import atomic_open
from app.core.image_sequence import ImageSequence
from app.core.pair_transform import PairAlignment, alignment_crop

SETUP_FORMAT = "ecmtracker-image-pair"
SETUP_VERSION = 2
SETUP_SUFFIX = ".ecmpair.json"


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a whole number of pixels")
    return int(value)


def overlap_rect(reference_shape, destination_shape, dx, dy):
    """Return (x, y, width, height) in reference pixels; zero extent means no overlap."""
    dx, dy = _integer(dx, "X offset"), _integer(dy, "Y offset")
    rh, rw = reference_shape[:2]
    dh, dw = destination_shape[:2]
    x, y = max(0, dx), max(0, dy)
    return x, y, max(0, min(rw, dx + dw) - x), max(0, min(rh, dy + dh) - y)


def _image_fingerprint(image):
    digest = hashlib.sha256(str(image.shape).encode("ascii"))
    digest.update(memoryview(image))
    return digest.hexdigest()


class _PairSources(ImageSequence):
    """Share normal decoding, intensity conventions, caching, and disk-change checks."""

    def _check_frame_shape(self, img, path):
        pass  # Only the resulting aligned frames must have a common shape.


class AlignedImagePairSequence(ImageSequence):
    """Two source images exposed as equally sized, immutable overlap crops.

    Geometry is fixed for the lifetime of a sequence. ``with_translation`` creates a
    candidate sharing the validated source cache without modifying the active sequence.
    """

    def __init__(self, paths, dx=0, dy=0, *, alignment=None):
        paths = [os.path.abspath(os.fspath(path)) for path in paths]
        if len(paths) != 2:
            raise ValueError("An image pair requires a reference and a destination image")
        super().__init__(paths, cache_size=2)
        self._alignment = alignment if alignment is not None else PairAlignment(
            translation=(_integer(dx, "X offset"), _integer(dy, "Y offset")))
        if not isinstance(self._alignment, PairAlignment):
            raise ValueError("Invalid pair alignment")
        self._crop = None
        self._sources = _PairSources(paths, cache_size=2)
        self._source_fingerprints = None

    @property
    def translation(self):
        return self.alignment.translation

    @property
    def alignment(self):
        return self._alignment

    @property
    def frame_shape(self):
        """Validated output shape, available without accessing source files again."""
        return self._frame_shape

    @property
    def crop(self):
        shapes = (self.source_bgr(0).shape, self.source_bgr(1).shape)
        if self._crop is None:
            self._crop = alignment_crop(*shapes, self.alignment)
        rect = self._crop
        if rect[2] == 0 or rect[3] == 0:
            raise ValueError("The images do not overlap. Move the destination closer to the reference.")
        return rect

    def source_bgr(self, index):
        """Original normalized pixels for alignment preview, guarded against disk changes."""
        return self._sources.load_bgr(index)

    def with_translation(self, dx, dy):
        return self.with_alignment(replace(self.alignment, translation=(
            _integer(dx, "X offset"), _integer(dy, "Y offset"))))

    def with_alignment(self, alignment):
        candidate = AlignedImagePairSequence(self.paths, alignment=alignment)
        candidate._sources = self._sources
        return candidate

    def with_editor_settings(self, alignment):
        """Retain validated frames/identity when only inactive presets or mode changed."""
        if not self.alignment.same_geometry(alignment):
            raise ValueError("Editor-only update cannot change alignment geometry")
        candidate = self.with_alignment(alignment)
        candidate._cache = self._cache.copy()
        candidate._crop = self._crop
        candidate._frame_shape = self._frame_shape
        candidate._fingerprint = self._fingerprint
        candidate._source_fingerprints = self._source_fingerprints
        return candidate

    def load_bgr(self, index):
        if not 0 <= index < 2:
            raise IndexError(f"Frame index {index} is outside 0..1")
        x, y, width, height = self.crop  # Check both sources even when the crop is cached.
        if index not in self._cache:
            if index == 1 and not self.alignment.integer_translation_only:
                matrix = self.alignment.matrix(self.source_bgr(1).shape)
                rh, rw = self.source_bgr(0).shape[:2]
                # Use the identical reference-grid map as the support mask. Inverting a
                # crop-shifted matrix can round boundary samples differently in OpenCV.
                warped = cv2.warpAffine(self.source_bgr(1), matrix, (rw, rh),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                image = warped[y:y+height, x:x+width].copy(order="C")
            else:
                if index == 1:
                    x -= int(self.translation[0])
                    y -= int(self.translation[1])
                image = np.ascontiguousarray(self.source_bgr(index)[y:y + height, x:x + width])
            image.setflags(write=False)
            self._cache[index] = image
        return self._cache[index]

    def validate_all(self, progress_cb=None):
        if not self._sources.validate_all():
            return False
        fingerprints = tuple(_image_fingerprint(self.source_bgr(i)) for i in range(2))
        if self.alignment.integer_translation_only:
            digest = hashlib.sha256(b"ecmtracker-aligned-pair-v1")
            digest.update(json.dumps([fingerprints, tuple(map(int, self.translation)), self.crop]).encode("ascii"))
        else:
            digest = hashlib.sha256(b"ecmtracker-aligned-pair-v2:linear:constant:valid-rectangle-v1")
            digest.update(json.dumps([fingerprints, self.alignment.matrix(self.source_bgr(1).shape).tolist(),
                                      self.alignment.correction.tolist(), self.crop]).encode("ascii"))
        for index in range(2):
            image = self.load_bgr(index)
            digest.update(memoryview(image))
            if progress_cb is not None and progress_cb(index + 1, 2):
                return False
        # A callback may have processed GUI events; recheck source signatures before commit.
        for index in range(2):
            self.source_bgr(index)
        self._source_fingerprints = fingerprints
        self._frame_shape = self._cache[0].shape
        self._fingerprint = digest.hexdigest()
        return True

    def setup_metadata(self):
        if self.fingerprint is None:
            raise ValueError("Validate the image pair before saving its setup")
        return {
            "format": SETUP_FORMAT,
            "version": SETUP_VERSION,
            "source_fingerprints": list(self._source_fingerprints),
            "translation": list(self.translation),
            "alignment": self.alignment.metadata(),
            "crop": list(self.crop),
            "sequence_fingerprint": self.fingerprint,
        }


def save_pair_setup(path, sequence):
    """Atomically save geometry and source references, never image or tracker data."""
    if not isinstance(sequence, AlignedImagePairSequence):
        raise ValueError("The current sequence is not an aligned image pair")
    path = os.path.abspath(os.fspath(path))
    if not path.lower().endswith(SETUP_SUFFIX):
        path += SETUP_SUFFIX
    sequence.validate_all()
    metadata = sequence.setup_metadata()
    paths = []
    for source in sequence.paths:
        try:
            source = os.path.relpath(source, os.path.dirname(path))
        except ValueError:  # Different Windows drives cannot be expressed relatively.
            pass
        paths.append(source.replace(os.sep, "/"))
    metadata["sources"] = paths
    with atomic_open(path) as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def load_pair_setup(path):
    """Validate a saved setup completely before returning a candidate sequence."""
    path = os.path.abspath(os.fspath(path))
    try:
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read pair setup: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("format") != SETUP_FORMAT:
        raise ValueError("Not an ECM Tracker image pair setup")
    if type(metadata.get("version")) is not int or metadata["version"] not in (1, SETUP_VERSION):
        raise ValueError("Unsupported image pair setup version")
    paths = metadata.get("sources")
    if not isinstance(paths, list) or len(paths) != 2 or any(
        not isinstance(p, str) or not p for p in paths
    ):
        raise ValueError("Pair setup must identify a reference and a destination image")
    shift = metadata.get("translation")
    if not isinstance(shift, list) or len(shift) != 2:
        raise ValueError("Pair setup must contain X and Y offsets")
    crop = metadata.get("crop")
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError("Pair setup crop geometry is invalid")
    paths = [os.path.join(os.path.dirname(path), p) for p in paths]
    if metadata["version"] == 1:
        sequence = AlignedImagePairSequence(paths, *shift)
    else:
        alignment = PairAlignment.from_metadata(metadata.get("alignment"))
        # Retain the legacy top-level offset field for readers, but reject conflicting geometry.
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in shift) or tuple(shift) != alignment.translation:
            raise ValueError("Pair setup offsets disagree with its alignment")
        sequence = AlignedImagePairSequence(paths, alignment=alignment)
    sequence.validate_all()
    expected = sequence.setup_metadata()
    if metadata.get("source_fingerprints") != expected["source_fingerprints"]:
        raise ValueError("Pair source images have changed. Open the images as a new pair to realign them.")
    if crop != expected["crop"] or metadata.get("sequence_fingerprint") != sequence.fingerprint:
        raise ValueError("Pair setup alignment does not match its saved fingerprint. Open a new pair to realign it.")
    return sequence
