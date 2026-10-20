"""Interactive drag tools for defining Rectangle and Circle ROIs.

Each tool is a duck-typed ``CanvasInteraction`` (see ``canvas_view.set_interaction``): the
canvas delivers ``on_press`` / ``on_move`` / ``on_release`` in image coordinates and calls
``on_cancel`` when the interaction is replaced or cleared. A tool drives a live preview by
setting ``window.state.roi`` to the in-progress polygon and calling ``canvas.update()`` (a
cheap repaint that reuses the built-in ROI overlay), then commits on release.
"""

import math

from app.core.roi import CIRCLE_SEGMENTS, ROI, circle_corners, rectangle_corners

MIN_SIZE = 3.0  # image px below which a drag is treated as an accidental click (cancelled)
__all__ = ["CIRCLE_SEGMENTS", "CircleTool", "NGonTool", "RectangleTool"]


class _DragTool:
    """Shared press/move/release/cancel plumbing for the rubber-band shape tools."""

    def __init__(self, window):
        self.window = window
        self._start = None  # (x, y) in image coords, set on press
        self._done = False  # True once committed, so on_cancel becomes a no-op

    def on_press(self, image_pt, event) -> None:
        self._start = (image_pt.x(), image_pt.y())

    def on_move(self, image_pt, event) -> None:
        if self._start is None:
            return
        corners = self._corners(image_pt.x(), image_pt.y())
        self.window.state.roi = ROI(corners) if corners else ROI()
        self.window.canvas.update()

    def on_release(self, image_pt, event) -> None:
        if self._start is None:
            return
        x, y = image_pt.x(), image_pt.y()
        corners = self._corners(x, y)
        args = self._shape_args(x, y)
        self._start = None
        if not corners:  # degenerate (click without a real drag): abort cleanly
            self.window._cancel_roi_definition()
            return
        self._done = True
        self.window._commit_interactive_roi(self.SHAPE, args)

    def on_cancel(self) -> None:
        self._start = None
        if self._done:
            return
        self.window.state.roi = None
        self.window.canvas.update()

    def _corners(self, x, y):
        """Return the polygon corners for the current drag, or None if too small."""
        raise NotImplementedError

    def _shape_args(self, x, y):
        """The session command arguments describing the dragged shape."""
        raise NotImplementedError


class RectangleTool(_DragTool):
    """Drag from one corner to the opposite to build an axis-aligned rectangle."""

    SHAPE = "rectangle"

    def _corners(self, x, y):
        x0, y0 = self._start
        if abs(x - x0) < MIN_SIZE or abs(y - y0) < MIN_SIZE:
            return None
        return rectangle_corners(x0, y0, x, y)

    def _shape_args(self, x, y):
        return (*self._start, x, y)


class CircleTool(_DragTool):
    """Press at the center and drag out the radius; stored as a polygon approximation."""

    SHAPE = "circle"

    def _corners(self, x, y):
        cx, cy = self._start
        r = math.hypot(x - cx, y - cy)
        if r < MIN_SIZE:
            return None
        return circle_corners(cx, cy, r)

    def _shape_args(self, x, y):
        cx, cy = self._start
        return cx, cy, math.hypot(x - cx, y - cy)


class NGonTool:
    """Collect left-clicks to define an arbitrary polygon ROI; a right-click closes it.

    A click-based ``CanvasInteraction`` (duck-typed like the drag tools). Each left-click adds a
    corner and previews the open polyline on ``window.state.roi`` (an incomplete ROI draws open);
    a right-click finishes the polygon once it has at least ``ROI.MIN_CORNERS`` vertices (the
    closing click is not added as a point). Built on the single canvas-interaction mechanism.
    """

    def __init__(self, window):
        self.window = window

    def on_press(self, image_pt, event) -> None:
        roi = self.window.state.roi
        if roi is None:
            return
        roi.add_corner(image_pt.x(), image_pt.y())
        self.window.canvas.refresh()
        self.window._update_tool_states()

    def on_right_press(self, image_pt, event) -> None:
        roi = self.window.state.roi
        if roi is None:
            return
        if len(roi.corners) < ROI.MIN_CORNERS:
            self.window.statusBar().showMessage(
                f"Need at least {ROI.MIN_CORNERS} points to close the ROI.", 3000
            )
            return
        roi.close()
        self.window._finish_roi_definition()

    def on_cancel(self) -> None:
        # Aborting before the polygon is closed is handled by the window's _cancel_roi_definition
        # (it discards the incomplete ROI); nothing tool-local to undo.
        pass
