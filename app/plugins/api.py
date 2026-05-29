"""Public plugin SDK for the Feature Tracker.

This is the **only** file a plugin author needs to read. A plugin is a small Python package
under ``plugins/`` that subclasses :class:`TrackerPlugin` and, when launched from the
``Plugins`` menu, builds a Qt window. Everything a plugin is allowed to touch — the tracked
coordinates, the images, the ROI, the active mask, the canvas, mouse input, persistent
settings — is reached through a single :class:`PluginContext` instance handed to the plugin
as ``self.ctx``.

Three capabilities, three entry points:

* **Read data** — ``ctx.coords()``, ``ctx.frame_bgr(i)``, ``ctx.roi``, ``ctx.metrics()`` …
* **Draw overlays** — ``ctx.add_overlay(fn)`` where ``fn(painter, ctx)`` paints in *screen*
  space (use ``ctx.image_to_screen(x, y)`` to place things).
* **Capture the mouse** — ``ctx.begin_canvas_interaction(handler)`` with a
  :class:`CanvasInteraction`; you receive clicks/drags in *image* coordinates.

Plugins react to state changes through ``ctx.signals`` (a Qt signal hub) instead of polling.
The only state a plugin may mutate is the keep-mask, via the safe, undoable
``ctx.apply_keep_mask(...)``.

Index convention (important): every index in this API is a **global** frame index (position in
the loaded folder, ``0 .. ctx.n_total_images-1``). The tracked-data arrays returned by
``coords()`` are indexed by **cut** index (``0`` = the reference frame). Use
``ctx.global_to_cut`` / ``ctx.cut_to_global`` to convert; you never need to know the internals.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QPointF, pyqtSignal

from app.core import settings
from app.core.cleanup import Metrics, compute_metrics

# Overlay painter signature: fn(painter: QPainter, ctx: PluginContext) -> None
OverlayFn = Callable[..., None]


class PluginSignals(QObject):
    """A Qt signal hub broadcasting tracker state changes to plugins.

    Connect to these to refresh your view reactively, e.g.::

        self.ctx.signals.frame_changed.connect(self._redraw)

    * ``sequence_changed`` — a new image sequence was loaded (all downstream state reset).
    * ``frame_changed(int)`` — the displayed frame changed; argument is the global index.
    * ``result_changed`` — a tracking result was produced or discarded.
    * ``mask_changed`` — the active (keep) mask changed (cleanup applied/undone).
    * ``roi_changed`` — the ROI was completed or cleared.
    """

    sequence_changed = pyqtSignal()
    frame_changed = pyqtSignal(int)
    result_changed = pyqtSignal()
    mask_changed = pyqtSignal()
    roi_changed = pyqtSignal()


class CanvasInteraction:
    """Base class for a temporary canvas mouse-capture mode.

    Install one with ``ctx.begin_canvas_interaction(self)`` and remove it with
    ``ctx.end_canvas_interaction()``. While installed, left-button press/move/release are
    delivered here (instead of the built-in ROI click); middle/right-drag still pans, and the
    ROI/Pan tools are switched off to avoid conflicts. All points are in **image** coordinates.

    Override only the methods you need; the defaults are no-ops. Exceptions raised here are
    swallowed by the canvas so a bug can't crash the app — log inside your handler if needed.
    """

    def on_press(self, image_pt: QPointF, event) -> None:  # noqa: D401
        """Left-button press at ``image_pt`` (a QPointF in image coordinates)."""

    def on_move(self, image_pt: QPointF, event) -> None:
        """Mouse move at ``image_pt`` (fires continuously; ignore if you only want clicks)."""

    def on_release(self, image_pt: QPointF, event) -> None:
        """Left-button release at ``image_pt``."""

    def on_cancel(self) -> None:
        """Called when the interaction is force-removed (replaced or ended). Clean up here."""


class PluginContext:
    """The façade a plugin uses to reach everything in the app.

    One instance is created per plugin and passed as ``self.ctx``. It is a thin, stable wrapper
    over the main window; it deliberately hides ``MainWindow`` / ``ProjectState`` internals so
    plugins keep working as the core evolves. Treat every array returned here as **read-only**
    unless a method name says otherwise.
    """

    def __init__(self, main_window, plugin_id: str = "plugin"):
        self._window = main_window
        self._plugin_id = plugin_id
        self._overlay_wrappers: dict = {}
        self._metrics_cache: Optional[Tuple[int, Metrics]] = None

    # ---- identity / plumbing -------------------------------------------
    @property
    def window(self):
        """The ``MainWindow`` — use as the parent for your plugin's Qt windows/dialogs."""
        return self._window

    @property
    def canvas(self):
        """The :class:`CanvasView`. Prefer the helpers below, but it's here for advanced use."""
        return self._window.canvas

    @property
    def signals(self) -> PluginSignals:
        """The :class:`PluginSignals` hub — connect to react to state changes."""
        return self._window.signals

    @property
    def _state(self):
        return self._window.state

    # ---- session / status ----------------------------------------------
    @property
    def has_sequence(self) -> bool:
        """True once an image sequence is loaded."""
        return self._state.has_sequence

    @property
    def has_result(self) -> bool:
        """True once tracking has produced a result."""
        return self._state.result is not None

    @property
    def n_total_images(self) -> int:
        """Number of frames in the full loaded folder."""
        return self._state.total_images

    def status(self, message: str, timeout: int = 4000) -> None:
        """Show a transient message in the main-window status bar."""
        self._window.statusBar().showMessage(message, timeout)

    def request_redraw(self) -> None:
        """Repaint the canvas (call after changing what your overlay draws)."""
        self._window.canvas.update()

    # ---- frame indices --------------------------------------------------
    @property
    def reference_index(self) -> int:
        """Global index of the reference frame (cut 0)."""
        return self._state.reference_index

    @property
    def last_index(self) -> int:
        """Global index of the last tracked frame."""
        return self._state.last_index

    @property
    def current_index(self) -> int:
        """Global index of the frame currently shown on the canvas."""
        return self._state.current_index

    @property
    def current_cut(self) -> Optional[int]:
        """Cut index of the current frame, or ``None`` if it is outside the tracked range."""
        s = self._state
        return s.global_to_cut(s.current_index) if s.current_in_range else None

    def global_to_cut(self, global_index: int) -> int:
        """Convert a global frame index to a cut index (``global - reference``)."""
        return self._state.global_to_cut(global_index)

    def cut_to_global(self, cut_index: int) -> int:
        """Convert a cut index to a global frame index (``reference + cut``)."""
        return self._state.cut_to_global(cut_index)

    # ---- images ---------------------------------------------------------
    def image_size(self) -> Optional[Tuple[int, int]]:
        """``(height, width)`` of the frames, or ``None`` if no sequence is loaded."""
        return self._state.image_size()

    def frame_bgr(self, global_index: int) -> np.ndarray:
        """Frame at ``global_index`` as an ``(H, W, 3)`` uint8 **BGR** array (OpenCV order)."""
        return self._require_sequence().load_bgr(global_index)

    def frame_gray(self, global_index: int) -> np.ndarray:
        """Frame at ``global_index`` as an ``(H, W)`` uint8 grayscale array."""
        return self._require_sequence().load_gray(global_index)

    def frame_rgb(self, global_index: int) -> np.ndarray:
        """Frame at ``global_index`` as an ``(H, W, 3)`` uint8 **RGB** array (for Qt/matplotlib)."""
        return cv2.cvtColor(self.frame_bgr(global_index), cv2.COLOR_BGR2RGB)

    def _require_sequence(self):
        seq = self._state.sequence
        if seq is None:
            raise RuntimeError("No image sequence is loaded.")
        return seq

    # ---- tracked data ---------------------------------------------------
    @property
    def result(self):
        """The raw, immutable ``TrackerResult`` (escape hatch for advanced arrays such as the
        backward pass and FB errors), or ``None``. Prefer ``coords()`` for the common case."""
        return self._state.result

    @property
    def frame_count(self) -> int:
        """Number of frames in the tracked range (``coords`` axis 0). 0 if no result."""
        return self._state.result.n_frames if self.has_result else 0

    @property
    def point_count(self) -> int:
        """Total number of tracked points P (before filtering). 0 if no result."""
        return self._state.result.n_points if self.has_result else 0

    @property
    def active_mask(self) -> Optional[np.ndarray]:
        """The ``(P,)`` bool keep-mask: True where a point survived cleanup. ``None`` if no
        result. Read-only — change it with :meth:`apply_keep_mask`."""
        return self._state.active_mask

    @property
    def n_active(self) -> int:
        """Number of currently-kept points."""
        m = self.active_mask
        return int(m.sum()) if m is not None else 0

    def point_indices(self) -> Optional[np.ndarray]:
        """Original column indices (into the full P points) of the kept points, matching the
        order of ``coords(active_only=True)``'s second axis. ``None`` if no result."""
        m = self.active_mask
        if m is None:
            return None
        return np.where(m)[0]

    def coords(self, active_only: bool = True) -> Optional[np.ndarray]:
        """Tracked forward coordinates as an ``(frames, points, 2)`` float32 array of ``(x, y)``.

        With ``active_only=True`` (default) only kept points are returned, so the point axis
        matches :meth:`point_indices`. With ``active_only=False`` all P points are returned.
        Returns ``None`` if there is no result. Indexed by **cut** index on axis 0
        (``coords[0]`` is the reference frame).
        """
        if not self.has_result:
            return None
        coords = self._state.result.coords_fw
        mask = self.active_mask
        if active_only and mask is not None:
            return coords[:, mask, :]
        return coords

    def metrics(self) -> Optional[Metrics]:
        """Per-point quality metrics (FB error, failure counts, max step, out-of-bounds …),
        computed once and cached. ``None`` if no result. See ``app/core/cleanup.py:Metrics``."""
        if not self.has_result:
            return None
        result = self._state.result
        key = id(result)
        if self._metrics_cache is None or self._metrics_cache[0] != key:
            h, w = self._state.image_size()
            self._metrics_cache = (key, compute_metrics(result, self._state.roi, (h, w)))
        return self._metrics_cache[1]

    # ---- ROI ------------------------------------------------------------
    @property
    def roi(self):
        """The current :class:`~app.core.roi.ROI`, or ``None``."""
        return self._state.roi

    @property
    def roi_corners(self) -> List[Tuple[float, float]]:
        """The ROI corner points ``[(x, y), …]`` in image coordinates (empty if no ROI)."""
        roi = self._state.roi
        return list(roi.corners) if roi is not None else []

    def roi_contains(self, x: float, y: float) -> bool:
        """True if image point ``(x, y)`` lies inside a *complete* ROI."""
        roi = self._state.roi
        return bool(roi is not None and roi.contains(x, y))

    def roi_mask(self) -> Optional[np.ndarray]:
        """A filled ``(H, W)`` uint8 (0/255) ROI mask at image resolution, or ``None``."""
        roi = self._state.roi
        size = self._state.image_size()
        if roi is None or not roi.is_complete or size is None:
            return None
        return roi.mask(size[0], size[1])

    # ---- safe state mutation -------------------------------------------
    def apply_keep_mask(self, keep: np.ndarray) -> None:
        """Filter the tracked points by a boolean keep-mask, undoably.

        ``keep`` may be length P (all points) or length ``n_active`` (current kept points only).
        Points marked False are removed from the active set; the operation is pushed onto the
        cleanup undo stack (the user can Undo it in the Cleanup dialog) and emits
        ``signals.mask_changed``. No-op if there is no result. Points already filtered out stay
        filtered out — this only ever shrinks the active set, never resurrects points.
        """
        mask = self.active_mask
        if mask is None:
            return
        keep = np.asarray(keep, dtype=bool)
        if keep.shape[0] == self.point_count:
            full = keep
        elif keep.shape[0] == self.n_active:
            # Expand to full length; inactive points are dropped by the `active & full` step
            # regardless, so they need no special handling here.
            full = np.zeros(self.point_count, dtype=bool)
            full[self.point_indices()] = keep
        else:
            raise ValueError(
                f"keep mask length {keep.shape[0]} is neither P={self.point_count} "
                f"nor n_active={self.n_active}"
            )
        self._window.apply_keep_mask(full)

    # ---- canvas overlays ------------------------------------------------
    def add_overlay(self, fn: OverlayFn) -> None:
        """Register an overlay painter ``fn(painter, ctx)``.

        ``fn`` is called on every repaint, in **screen** space, after the built-in overlays.
        Convert image coordinates with :meth:`image_to_screen`. Keep it fast (it runs on the UI
        thread each paint). A painter that raises is auto-removed. Remember to
        :meth:`remove_overlay` in your window's ``closeEvent`` / plugin ``on_unload``.
        """
        if fn in self._overlay_wrappers:
            return

        def wrapper(painter, _canvas, _fn=fn):
            _fn(painter, self)

        self._overlay_wrappers[fn] = wrapper
        self._window.canvas.add_overlay(wrapper)

    def remove_overlay(self, fn: OverlayFn) -> None:
        """Unregister an overlay painter previously added with :meth:`add_overlay`."""
        wrapper = self._overlay_wrappers.pop(fn, None)
        if wrapper is not None:
            self._window.canvas.remove_overlay(wrapper)

    def image_to_screen(self, x: float, y: float) -> QPointF:
        """Map an image-space point to widget/screen coordinates (for overlay painting)."""
        return self._window.canvas.image_to_screen(x, y)

    def screen_to_image(self, point: QPointF) -> QPointF:
        """Map a widget/screen point back to image coordinates."""
        return self._window.canvas.screen_to_image(point)

    # ---- canvas interaction --------------------------------------------
    def begin_canvas_interaction(self, handler: CanvasInteraction) -> None:
        """Capture canvas mouse input with ``handler`` (a :class:`CanvasInteraction`).

        The ROI and Pan tools are switched off so they don't conflict. Always pair with
        :meth:`end_canvas_interaction` (e.g. when your window closes)."""
        win = self._window
        if getattr(win, "define_roi_action", None) is not None and win.define_roi_action.isChecked():
            win.define_roi_action.setChecked(False)
        if getattr(win, "pan_tool_action", None) is not None and win.pan_tool_action.isChecked():
            win.pan_tool_action.setChecked(False)
        win.canvas.set_interaction(handler)

    def end_canvas_interaction(self) -> None:
        """Release canvas mouse capture (restores the default ROI-click behavior)."""
        self._window.canvas.clear_interaction()

    # ---- persistent settings -------------------------------------------
    def get_settings(self) -> dict:
        """Return this plugin's persisted settings dict (empty if none saved yet). Stored in the
        shared ``settings.json`` under a key unique to your plugin."""
        return settings.get_section(self._section()) or {}

    def save_settings(self, data: dict) -> None:
        """Persist this plugin's settings dict (overwrites the previous one)."""
        settings.update_section(self._section(), data)

    def _section(self) -> str:
        return f"plugin:{self._plugin_id}"


class TrackerPlugin:
    """Base class for a plugin. Subclass it and implement :meth:`launch`.

    Declare the package's plugin by assigning ``PLUGIN = YourClass`` in the package
    ``__init__.py`` (the manager also auto-detects a lone ``TrackerPlugin`` subclass).

    Lifecycle: the manager instantiates your class once (passing the :class:`PluginContext`),
    then calls :meth:`launch` each time the user clicks your menu entry. The window you return
    is kept alive by the manager; relaunching while it's still open just re-focuses it.

    Minimal example::

        from PyQt5.QtWidgets import QLabel
        from app.plugins import TrackerPlugin

        class HelloPlugin(TrackerPlugin):
            NAME = "Hello"
            DESCRIPTION = "Reports how many points are tracked."

            def launch(self):
                n = self.ctx.n_active
                w = QLabel(f"{n} points tracked", parent=self.ctx.window)
                w.setWindowFlags(w.windowFlags() | 0x00000001)  # Qt.Window
                w.setWindowTitle(self.NAME)
                w.show()
                return w

        PLUGIN = HelloPlugin
    """

    #: Display name shown in the Plugins menu.
    NAME: str = "Unnamed Plugin"
    #: One-line description (used as the menu item's tooltip).
    DESCRIPTION: str = ""

    def __init__(self, ctx: PluginContext):
        self.ctx = ctx

    def launch(self):
        """Build and show the plugin's window; return it (a ``QWidget``) so it stays alive.

        Called on the Qt main thread each time the menu entry is clicked. Returning ``None`` is
        allowed (e.g. for a plugin that just runs an action), but then the plugin is responsible
        for keeping any windows it creates referenced.
        """
        raise NotImplementedError

    def on_unload(self) -> None:
        """Optional: called when plugins are reloaded. Remove overlays / interactions here."""
