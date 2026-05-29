from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class ROI:
    """A polygonal region of interest defined by N corners in image coordinates.

    Corners connect in click order; the polygon is finalized when ``close()`` is called.
    Constructing with ``corners`` yields an already-closed ROI (the shape tools and tests
    build a complete polygon in one shot this way).
    """

    def __init__(self, corners: Optional[Sequence[Tuple[float, float]]] = None):
        self.corners: List[Tuple[float, float]] = [
            (float(x), float(y)) for x, y in (corners or [])
        ]
        self.closed: bool = bool(self.corners)

    @property
    def is_complete(self) -> bool:
        return self.closed

    def add_corner(self, x: float, y: float) -> None:
        if not self.closed:
            self.corners.append((float(x), float(y)))

    def close(self) -> None:
        self.closed = True

    def reset(self) -> None:
        self.corners = []
        self.closed = False

    def _contour(self) -> np.ndarray:
        return np.array(self.corners, dtype=np.float32).reshape(-1, 1, 2)

    def polygon_int(self) -> np.ndarray:
        return np.array(self.corners, dtype=np.int32)

    def mask(self, height: int, width: int) -> np.ndarray:
        """Binary (0/255) uint8 mask of the filled polygon at image resolution."""
        m = np.zeros((height, width), dtype=np.uint8)
        if self.is_complete:
            cv2.fillPoly(m, [self.polygon_int()], 255)
        return m

    def contains(self, x: float, y: float) -> bool:
        if not self.is_complete:
            return False
        return cv2.pointPolygonTest(self._contour(), (float(x), float(y)), False) >= 0
