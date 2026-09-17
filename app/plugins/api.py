"""Public plugin SDK for the ECM Tracker.

Start with ``plugins/PLUGIN_CONTRACT.md`` for the standalone authoring contract. A plugin is a small Python package
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
The keep-mask is editable through undoable ``ctx.apply_keep_mask(...)``. Explicit
workflow methods can also load a sequence/session and change the tracking range;
changed bounds discard dependent tracking. See plugins/PLUGIN_CONTRACT.md.

Index convention (important): every index in this API is a **global** frame index (position in
the loaded folder, ``0 .. ctx.n_total_images-1``). The tracked-data arrays returned by
``coords()`` are indexed by **cut** index (``0`` = the reference frame). Use
``ctx.global_to_cut`` / ``ctx.cut_to_global`` to convert; you never need to know the internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from types import MappingProxyType
from typing import Mapping

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, Signal

from app.core import settings
from app.core.cleanup import Metrics, compute_metrics

# Overlay painter signature: fn(painter: QPainter, ctx: PluginContext) -> None
OverlayFn = Callable[..., None]
API_VERSION = 1


def _readonly(array):
    value = np.array(array, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class TrackingSnapshot:
    """Aligned, detached read-only arrays. Axis 0 is frame_indices; axis 1 is point_ids.

    A false valid entry is a frozen/failed track, NEVER a measured displacement.
    revision belongs to the session that produced this snapshot.
    """
    revision: int
    frame_indices: np.ndarray
    point_ids: np.ndarray
    coords: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    error_kind: str
    reference_index: int
    last_index: int
    tracking_params: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "tracking_params", MappingProxyType(deepcopy(dict(self.tracking_params))))



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

    sequence_changed = Signal()
    frame_changed = Signal(int)
    result_changed = Signal()
    mask_changed = Signal()
    roi_changed = Signal()
    range_changed = Signal()
    seeds_changed = Signal()


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
        self._metrics_cache: Optional[Tuple[tuple, Metrics]] = None
        self._interaction_handler = None
        self._subscriptions = []
        self._snapshot_cache = {}
        self._disposed = False

    @property
    def revision(self) -> int:
        """Monotonic scientific-state version; display-only frame changes do not increment it."""
        return self._state.revision

    def subscribe(self, signal, callback):
        """Connect a callback until unsubscribe() or host disposal; returns unsubscribe()."""
        if self._disposed:
            raise RuntimeError("Plugin context has been disposed")
        signal.connect(callback)
        pair = (signal, callback)
        self._subscriptions.append(pair)

        def unsubscribe():
            if pair in self._subscriptions:
                self._subscriptions.remove(pair)
                try:
                    signal.disconnect(callback)
                except (RuntimeError, TypeError):
                    pass
        return unsubscribe

    def dispose(self) -> None:
        """Idempotently release host-managed callbacks, overlays and mouse capture."""
        if self._disposed:
            return
        self._disposed = True
        for signal, callback in self._subscriptions:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        self._subscriptions.clear()
        for fn in list(self._overlay_wrappers):
            self.remove_overlay(fn)
        self.end_canvas_interaction()
        self._snapshot_cache.clear()
        self._metrics_cache = None

    def tracks(self, active_only: bool = True) -> Optional[TrackingSnapshot]:
        """Return a coherent snapshot, cached until scientific state changes."""
        result = self._state.result
        if result is None:
            self._snapshot_cache.clear()
            return None
        key = (self.revision, id(result), bool(active_only),
               self._state.active_mask.tobytes() if self._state.active_mask is not None else None)
        if key not in self._snapshot_cache:
            ids = np.arange(result.n_points)
            if active_only and self._state.active_mask is not None:
                ids = ids[self._state.active_mask]
            snapshot = TrackingSnapshot(
                self.revision, _readonly(np.arange(result.reference_index, result.last_index + 1)),
                _readonly(ids), _readonly(result.coords_fw[:, ids]),
                _readonly(result.status_fw[:, ids].astype(bool)), _readonly(result.err_fw[:, ids]),
                result.error_kind, result.reference_index, result.last_index, result.tracking_params)
            self._snapshot_cache = {key: snapshot}
        return self._snapshot_cache[key]

    def frame_tracks(self, global_index: int, active_only: bool = True):
        """Return (point_ids, xy, valid) for one global frame, or None outside the result."""
        if isinstance(global_index, bool) or not isinstance(global_index, (int, np.integer)):
            raise ValueError("Frame index must be an integer")
        snap = self.tracks(active_only)
        if snap is None or not snap.reference_index <= global_index <= snap.last_index:
            return None
        i = global_index - snap.reference_index
        return snap.point_ids, snap.coords[i], snap.valid[i]

    def set_frame_range(self, reference: int, last: int) -> None:
        """Atomically set both global bounds; changed bounds discard the tracking result."""
        self._window.set_frame_range(reference, last)

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
        if self._disposed:
            raise RuntimeError("Plugin context has been disposed")
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

    @property
    def source_dir(self) -> Optional[str]:
        """Directory the active image sequence was loaded from, or ``None``.

        This is the same directory the core application uses as the default location for
        sequence-related open/save dialogs.  It is exposed read-only so plugins can choose a
        nearby default without reaching through the façade into :class:`ProjectState`.
        """
        return self._state.source_dir

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

    def set_current_frame(self, global_index: int) -> None:
        """Move the app's current frame to ``global_index`` (clamped to the loaded range).

        Drives the same path as the main Current slider: updates the canvas, status bar and tool
        states, and emits ``signals.frame_changed``. No-op if no sequence is loaded.
        """
        self._window._go_to_frame(global_index)

    def set_reference_frame(self, global_index: int) -> None:
        """Move the reference (start) frame to ``global_index`` (clamped to ``0 .. last_index``).

        Drives the same path as the main Reference slider: re-scopes the tracked range and, because
        the ROI is defined on the reference frame, clears any existing ROI and seed features
        (emitting ``signals.roi_changed``). If a tracking result exists and the reference actually
        moves, the result is discarded (its cut-indexed arrays are tied to the old reference),
        emitting ``signals.result_changed``. No-op if no sequence is loaded.
        """
        self._window._set_reference_frame(global_index)

    def set_last_frame(self, global_index: int) -> None:
        """Move the last frame to ``global_index`` (clamped to ``reference_index .. last loaded
        frame``).

        Drives the same path as the main Last slider: re-scopes the tracked range. If a tracking
        result exists and the last frame actually moves, the result is discarded (re-scoping the
        range invalidates its cut-indexed arrays), emitting ``signals.result_changed``. No-op if
        no sequence is loaded.
        """
        self._window._set_last_frame(global_index)

    # ---- images ---------------------------------------------------------
    @property
    def frame_paths(self) -> tuple[str, ...]:
        """Ordered image paths for all global frames as an immutable tuple.

        The order exactly matches global frame indices and :attr:`n_total_images`.  An empty
        tuple is returned when no sequence is loaded.  Plugins may inspect file metadata through
        these paths; pixel access should continue to use :meth:`frame_bgr` and its variants.
        For aligned pairs these remain the original source paths; their raw pixels/dimensions
        may differ from the cropped frames returned by the image helpers.
        """
        sequence = self._state.sequence
        return tuple(sequence.paths) if sequence is not None else ()

    def load_sequence(self, paths: List[str], source_dir: Optional[str] = None) -> bool:
        """Load an explicit, pre-ordered list of image paths as the active sequence.

        Order is preserved verbatim (no filename sort) — pass the paths in the exact order you
        want them indexed (e.g. acquisition-log order). Resets all downstream state (ROI, seed
        features, result, mask) and emits ``signals.sequence_changed``, just like File -> Open.
        ``source_dir`` is remembered as the default save/open directory. Intended for loader
        plugins; most plugins never need this and should work with the already-loaded sequence.
        """
        return self._window.load_sequence_from_paths(list(paths), source_dir)

    def save_trackers(self, path: str) -> None:
        """Save the current reference points and (if tracked) the full result to ``path``.

        Writes a single self-contained ``.npz`` (see :mod:`app.core.tracker_io`) holding the
        seed points, the tracking arrays + active mask, the ROI, the frame range and the
        detection/LK parameters — the same format as ``File -> Save Trackers``. Use it to
        persist the tracking state alongside your plugin's own project files, then restore it
        with :meth:`load_trackers`. Raises ``ValueError`` if there are no reference points and
        ``OSError`` if the write fails."""
        self._window.save_trackers_to(path)

    def load_trackers(self, path: str, *, expected_range=None) -> None:
        """Load a ``.npz`` written by :meth:`save_trackers` and overlay it onto the open sequence.

        Restores the seeds, tracking result, active mask, ROI and frame range, emitting
        ``signals.result_changed`` / ``mask_changed`` / ``roi_changed`` so the UI and plugins
        refresh. Current files require the same ordered image-content fingerprint; legacy v1 files
        can only be checked by frame count. A mismatch or bad/foreign file raises ``ValueError``.
        This is the only way a plugin can install a full tracking result back into the core."""
        self._window.load_trackers_from(path, expected_range=expected_range)

    def image_size(self) -> Optional[Tuple[int, int]]:
        """``(height, width)`` of the frames, or ``None`` if no sequence is loaded."""
        return self._state.image_size()

    def frame_bgr(self, global_index: int) -> np.ndarray:
        """Frame at ``global_index`` as an ``(H, W, 3)`` uint8 **BGR** array (OpenCV order).

        Aligned pairs return the shared crop, in the same coordinates as tracking results.
        """
        return self._require_sequence().load_bgr(global_index)

    def frame_gray(self, global_index: int) -> np.ndarray:
        """Frame at ``global_index`` as an ``(H, W)`` uint8 grayscale array."""
        return self._require_sequence().load_gray(global_index)

    def alignment_affine(self, global_index: int) -> np.ndarray:
        """Detached read-only 2x2 deformation-restoration matrix C for a global frame.

        Identity for ordinary sequences, the original pair reference, or rigid-only alignment.
        For a destination warped by L=R U (polar rotation/stretch), C=R inv(U) R.T.
        Restore a residual fit with C_current @ F @ inv(C_reference). This restores scale/shear,
        never alignment translation/rotation. Reported axes live in restored, rotation-aligned
        coordinates; map them through inv(C_current) to overlay them on the aligned pixels.
        All frame/track accessors continue to return aligned coordinates.
        """
        from app.core.image_pair import AlignedImagePairSequence

        sequence = self._require_sequence()
        if isinstance(global_index, bool) or not isinstance(global_index, (int, np.integer)) or not 0 <= global_index < len(sequence):
            raise IndexError("Alignment frame index is outside the loaded sequence")
        matrix = sequence.alignment.correction if isinstance(sequence, AlignedImagePairSequence) and global_index == 1 else np.eye(2)
        return _readonly(matrix)

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
        backward pass and FB errors), or ``None``. Prefer validity-aware ``tracks()``."""
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
        mask = self._state.active_mask
        if mask is None:
            return None
        view = mask.view()
        view.setflags(write=False)
        return view

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
        snapshot = self.tracks(active_only)
        return snapshot.coords if snapshot is not None else None

    def track_status(self, active_only: bool = True) -> Optional[np.ndarray]:
        """Forward per-frame tracking status as an ``(frames, points)`` uint8 array, aligned to
        :meth:`coords` (``1`` = the point was tracked OK at that frame, ``0`` = LK failed and the
        position was carried forward). Use it to drop dead/frozen tracks before fitting. Same
        ``active_only`` semantics as :meth:`coords`; ``None`` if there is no result."""
        snapshot = self.tracks(active_only)
        return snapshot.valid.view(np.uint8) if snapshot is not None else None

    def metrics(self) -> Optional[Metrics]:
        """Per-point quality metrics (FB error, failure counts, max step, out-of-bounds …),
        computed once and cached. ``None`` if no result. See ``app/core/cleanup.py:Metrics``."""
        if not self.has_result:
            return None
        result = self._state.result
        # Key on both the result and the ROI: the ``left_roi`` metric depends on the ROI, so the
        # cache must refresh when the ROI is set/cleared/replaced, not only when the result changes.
        key = (id(result), tuple(self.roi_corners))
        if self._metrics_cache is None or self._metrics_cache[0] != key:
            h, w = self._state.image_size()
            self._metrics_cache = (key, compute_metrics(result, self._state.roi, (h, w)))
        return self._metrics_cache[1]

    # ---- ROI ------------------------------------------------------------
    @property
    def roi(self):
        """Detached ROI copy, or None; editing it cannot change the host."""
        return deepcopy(self._state.roi)

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
    def apply_keep_mask(self, keep: np.ndarray, *, revision: Optional[int] = None) -> None:
        """Filter the tracked points by a boolean keep-mask, undoably.

        ``keep`` may be length P (all points) or length ``n_active`` (current kept points only).
        Points marked False are removed from the active set; the operation is pushed onto the
        cleanup undo stack (the user can Undo it in the Cleanup dialog) and emits
        ``signals.mask_changed``. No-op if there is no result. Points already filtered out stay
        filtered out — this only ever shrinks the active set, never resurrects points.
        """
        if revision is not None and revision != self.revision:
            raise ValueError("Stale tracking snapshot; refresh before applying a mask.")
        mask = self.active_mask
        if mask is None:
            return
        keep = np.asarray(keep, dtype=bool)
        if keep.ndim != 1:
            raise ValueError(f"keep mask must be one-dimensional, got shape {keep.shape}")
        if keep.size == self.point_count:
            full = keep
        elif keep.size == self.n_active:
            # Expand to full length; inactive points are dropped by the `active & full` step
            # regardless, so they need no special handling here.
            full = np.zeros(self.point_count, dtype=bool)
            full[self.point_indices()] = keep
        else:
            raise ValueError(
                f"keep mask length {keep.size} is neither P={self.point_count} "
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
            wrapper = self._overlay_wrappers[fn]
            # Canvas removes a painter that raises. Permit an explicit re-add after the plugin has
            # corrected transient state instead of leaving the context's registry permanently stale.
            if wrapper not in self._window.canvas._overlays:
                self._window.canvas.add_overlay(wrapper)
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
            win._cancel_roi_definition()
        if hasattr(win, "_deactivate_point_tools"):
            win._deactivate_point_tools()
        if getattr(win, "pan_tool_action", None) is not None and win.pan_tool_action.isChecked():
            win.pan_tool_action.setChecked(False)
        win.canvas.set_interaction(handler)
        self._interaction_handler = handler

    def end_canvas_interaction(self) -> None:
        """Release this context's mouse capture without disturbing a newer owner."""
        handler, self._interaction_handler = self._interaction_handler, None
        if handler is not None and self._window.canvas._interaction is handler:
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

    Lifecycle: the manager owns your context and returned window. A visible-window relaunch
    re-focuses it. Relaunch after close/deletion disposes the old instance and creates a new one.
    Action-only plugins returning None may receive repeated launch calls on the same instance.

    Minimal example::

        from PySide6.QtWidgets import QLabel
        from app.plugins import TrackerPlugin

        class HelloPlugin(TrackerPlugin):
            NAME = "Hello"
            DESCRIPTION = "Reports how many points are tracked."

            def launch(self):
                n = self.ctx.n_active
                w = QLabel(f"{n} points tracked", parent=self.ctx.window)
                w.setWindowFlags(w.windowFlags() | 0x00000001)  # Qt.WindowType.Window
                w.setWindowTitle(self.NAME)
                w.show()
                return w

        PLUGIN = HelloPlugin
    """

    #: Display name shown in the Plugins menu.
    API_VERSION: int = API_VERSION
    NAME: str = "Unnamed Plugin"
    #: One-line description (used as the menu item's tooltip).
    DESCRIPTION: str = ""
    #: Sort key for the Plugins menu / side-pane (ascending; ties broken by NAME). Lower floats
    #: to the top. Leave at the default to be ordered after pinned plugins, alphabetically.
    ORDER: int = 100

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
