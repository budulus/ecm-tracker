"""Interactive click tools for editing individual tracker points on the canvas.

Each class is a duck-typed canvas interaction (see ``CanvasView.set_interaction``): while installed,
the canvas delivers left-button presses as ``on_press(image_pt, event)`` in image coordinates. Both
tools are driven by their toolbar QAction's checked state (the single source of truth) — see
``MainWindow._on_point_tool_toggled``. A single click is self-contained, so ``on_cancel`` is a no-op.

- ``AddPointsTool``    appends a seed point to ``state.features`` (reference frame, before tracking).
- ``DeletePointsTool`` removes the point nearest the click: a seed point before tracking, or — once a
  result exists — an individual tracked point, dropped via the undoable ``apply_keep_mask`` path.
"""
import numpy as np
from PySide6.QtCore import QPointF

# Extra screen-space slack (px) added to the marker radius when hit-testing a delete click, so a
# click just outside the drawn marker still lands on it.
DELETE_SLACK_PX = 4.0


def nearest_point(canvas, marker_size, image_pt: QPointF, coords, candidates) -> int:
    """Index (into ``coords``) of the candidate nearest ``image_pt`` within the hit radius, or -1.

    Distance is measured in screen space so the threshold tracks the on-screen marker size at any
    zoom (markers are drawn at a constant pixel radius regardless of image scale). Shared by the
    Delete tool and the Point Manager so both pick the same point for a given click."""
    click = canvas.image_to_screen(image_pt.x(), image_pt.y())
    radius = marker_size + DELETE_SLACK_PX
    best_i, best_d2 = -1, radius * radius
    for i in candidates:
        x, y = coords[i]
        scr = canvas.image_to_screen(float(x), float(y))
        dx, dy = scr.x() - click.x(), scr.y() - click.y()
        d2 = dx * dx + dy * dy
        if d2 <= best_d2:
            best_i, best_d2 = i, d2
    return best_i


class AddPointsTool:
    def __init__(self, window):
        self.window = window

    def on_press(self, image_pt: QPointF, event) -> None:
        state = self.window.state
        size = state.image_size()
        if size is None or not (0 <= image_pt.x() < size[1] and 0 <= image_pt.y() < size[0]):
            return
        pt = np.array([[image_pt.x(), image_pt.y()]], dtype=np.float32)
        if state.features is None or len(state.features) == 0:
            state.features = pt
        else:
            state.features = np.vstack([state.features, pt]).astype(np.float32)
        state.touch()
        self.window.signals.seeds_changed.emit()
        self.window.canvas.update()        # overlay-only change: cheaper than a full refresh()
        self.window._update_tool_states()  # the first point enables Run Tracking

    def on_cancel(self) -> None:
        pass


class DeletePointsTool:
    def __init__(self, window):
        self.window = window

    def on_press(self, image_pt: QPointF, event) -> None:
        if self.window.state.result is None:
            self._delete_seed(image_pt)
        else:
            self._delete_tracked(image_pt)

    def on_cancel(self) -> None:
        pass

    # -- internals ---------------------------------------------------------
    def _nearest(self, image_pt: QPointF, coords, candidates) -> int:
        return nearest_point(
            self.window.canvas,
            self.window.state.display_params["marker_size"],
            image_pt,
            coords,
            candidates,
        )

    def _delete_seed(self, image_pt: QPointF) -> None:
        state = self.window.state
        feats = state.features
        if feats is None or len(feats) == 0:
            return
        i = self._nearest(image_pt, feats, range(len(feats)))
        if i < 0:
            return
        feats = np.delete(feats, i, axis=0)
        state.features = feats if len(feats) else None  # match detection's empty convention
        state.touch()
        self.window.signals.seeds_changed.emit()
        self.window.canvas.update()
        self.window._update_tool_states()

    def _delete_tracked(self, image_pt: QPointF) -> None:
        state = self.window.state
        result, active = state.result, state.active_mask
        if result is None or active is None or not state.current_in_range:
            return
        cut = state.global_to_cut(state.current_index)
        coords = result.coords_fw[cut]
        valid = result.status_fw[cut].astype(bool)
        candidates = [p for p in range(result.n_points) if active[p] and valid[p]]
        i = self._nearest(image_pt, coords, candidates)
        if i < 0:
            return
        keep = np.ones(result.n_points, dtype=bool)
        keep[i] = False
        self.window.apply_keep_mask(keep)  # snapshots undo, ANDs the mask, refreshes, emits
