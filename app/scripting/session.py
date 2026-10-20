"""The scripting kernel: one object that runs the whole tracking workflow.

A :class:`Session` owns a :class:`~app.models.project_state.ProjectState` and exposes every
workflow step as a plain Python method — open images, set the frame range, define the ROI,
seed points, track, clean up, export, save/load. It is **Qt-free**, so the identical code runs

* headless, in your own program::

      from ecm_tracker import Session
      s = Session()
      s.open_directory(r"C:/data/experiment")
      s.set_roi_rectangle(60, 50, 240, 180)
      s.detect_corners(maxCorners=300)
      s.track()
      s.cleanup(fb_mean=0.5, drop_left_image=True)
      s.export_npy(r"C:/data/experiment/coords.npy")

* inside the GUI, where ``MainWindow`` routes its own toolbar/menu actions through the same
  methods (``window.session``) and the Python console / Script Editor get it as ``session``.

That single path is what makes macro recording exact: while a
:class:`~app.scripting.recorder.Recorder` is attached, each *top-level* command is written out
as the Python call that reproduces it (nested commands and commands issued by signal handlers
are consequences, not user actions, and are not recorded).

Observers learn about state changes through :meth:`Session.add_listener`; callbacks receive a
``frozenset`` of change kinds (``SEQUENCE``, ``RANGE``, ``FRAME``, ``ROI``, ``SEEDS``,
``RESULT``, ``MASK``, ``PARAMS``, ``DISPLAY``, ``PAIR``). ``before`` callbacks run just before
a change is applied (the GUI tears down editors bound to the old state there), ``after``
callbacks once the outermost command has finished.

Index convention: every frame index here is **global** (position in the loaded sequence). Arrays
of tracked data are indexed by **cut** index on axis 0 (``0`` = the reference frame); use
:meth:`Session.global_to_cut` / :meth:`Session.cut_to_global` to convert.
"""
from __future__ import annotations

import functools
import inspect
import math
import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.core import tracker_io
from app.core.cleanup import Metrics, build_mask, compute_metrics, thresholds_from_filters
from app.core.export import export, export_csv
from app.core.feature_detection import regular_grid, shi_tomasi
from app.core.image_pair import AlignedImagePairSequence, load_pair_setup, save_pair_setup
from app.core.image_sequence import ImageSequence, discover
from app.core.pair_transform import PairAlignment
from app.core.result import TrackerResult
from app.core.roi import CIRCLE_SEGMENTS, ROI, circle_corners, rectangle_corners
from app.core.tracking import track
from app.models.project_state import (
    ProjectState,
    validate_display,
    validate_grid,
    validate_lk,
    validate_shi_tomasi,
)

# Change kinds delivered to listeners.
SEQUENCE = "sequence"  # a new sequence replaced everything downstream
RANGE = "range"  # reference/last bounds changed
FRAME = "frame"  # the current (displayed) frame changed
ROI_KIND = "roi"
SEEDS = "seeds"  # reference seed points changed
RESULT = "result"  # tracking result installed or discarded (resets the mask)
MASK = "mask"  # active keep-mask changed on an existing result
PARAMS = "params"  # detection / tracking parameters changed
DISPLAY = "display"  # marker display preferences changed
PAIR = "pair"  # aligned-pair editor settings changed without touching geometry

# ProgressCb: (done, total) -> True to cancel.
ProgressCb = Callable[[int, int], bool]


class SessionError(ValueError):
    """A command cannot run in the session's current state (e.g. tracking without seeds)."""


class StaleResultError(SessionError):
    """The tracking inputs changed while tracking ran, so the result was discarded."""


class OperationCancelled(Exception):
    """A long-running command was cancelled through its progress callback."""


def command(_fn=None, *, record: bool = True, coalesce: bool = False):
    """Mark a Session method as a command: event batching + macro recording.

    Listener notifications are deferred until the outermost command returns, so observers see
    one consistent update. A successful *top-level* call is recorded as Python source (keyword
    arguments equal to the method's defaults and ``progress`` callbacks are left out).
    ``coalesce`` collapses consecutive recordings of the same command into its last call.
    """

    def deco(fn):
        signature = inspect.signature(fn)
        defaults = {
            name: p.default
            for name, p in signature.parameters.items()
            if p.default is not inspect.Parameter.empty
        }

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            top = self._depth == 0
            self._depth += 1
            ok = False
            try:
                value = fn(self, *args, **kwargs)
                ok = True
                return value
            finally:
                self._depth -= 1
                if self._depth == 0:
                    if ok and top and record:
                        recorded = {
                            k: v for k, v in kwargs.items()
                            if k != "progress" and not (k in defaults and _same(v, defaults[k]))
                        }
                        self._record("session", fn.__name__, args, recorded, coalesce=coalesce)
                    self._flush_events()

        wrapper.__ecm_command__ = True
        return wrapper

    return deco(_fn) if _fn is not None else deco


def _same(a, b) -> bool:
    try:
        return bool(a == b) and type(a) is type(b)
    except (TypeError, ValueError):
        return False


def _plain(value):
    """numpy scalar -> Python scalar (the parameter validators only accept Python numbers)."""
    return value.item() if isinstance(value, np.generic) else value


def _index(value, name: str) -> int:
    value = _plain(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer frame index, got {value!r}")
    return value


def _strict_params(section: str, validator, current: dict, updates: dict) -> dict:
    """Merge ``updates`` over ``current`` and reject unknown keys or out-of-range values.

    The ``validate_*`` helpers silently fall back to defaults for bad persisted input; a script
    must instead fail loudly, so every requested value has to survive validation unchanged.
    """
    unknown = sorted(set(updates) - set(current))
    if unknown:
        raise TypeError(
            f"Unknown {section} parameter(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(current)}"
        )
    updates = {k: _plain(v) for k, v in updates.items()}
    merged = validator({**current, **updates})
    for key, value in updates.items():
        if merged[key] != value or isinstance(merged[key], bool) != isinstance(value, bool):
            raise ValueError(f"Invalid value for {section} parameter {key!r}: {value!r}")
    return merged


def _points(points, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be a sequence of (x, y) pairs")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


class Session:
    """A complete, scriptable tracking session (see the module docstring for an overview).

    ``Session()`` starts empty with the built-in parameter defaults (independent of the GUI's
    saved preferences, so a script behaves identically on every machine). Pass an existing
    ``ProjectState`` to drive state that something else (the GUI) also displays.

    ``progress_handler``, when set, is a context-manager factory
    ``handler(label, total, delay_ms) -> ProgressCb`` used for long commands (loading,
    tracking) that were not given an explicit ``progress=`` callback.
    """

    def __init__(self, state: Optional[ProjectState] = None):
        self.state = state if state is not None else ProjectState(use_saved_defaults=False)
        self.progress_handler = None
        self._before: List[Callable[[frozenset], None]] = []
        self._after: List[Callable[[frozenset], None]] = []
        self._depth = 0
        self._pending: set = set()
        self._dispatching = False
        self._validating = False
        self._recorder = None
        self._record_suspended = 0
        self._metrics_cache = None
        self._export_cache = None

    # ==================================================================== events
    def add_listener(self, after=None, *, before=None) -> Callable[[], None]:
        """Subscribe to state changes; returns a function that unsubscribes.

        ``after(kinds)`` runs once the outermost command finished; ``before(kinds)`` runs just
        before each change is applied. ``kinds`` is a ``frozenset`` of change-kind strings."""
        if after is not None:
            self._after.append(after)
        if before is not None:
            self._before.append(before)

        def unsubscribe():
            if after in self._after:
                self._after.remove(after)
            if before in self._before:
                self._before.remove(before)

        return unsubscribe

    @contextmanager
    def _change(self, *kinds):
        kinds = frozenset(kinds)
        for listener in list(self._before):
            listener(kinds)
        self._depth += 1
        try:
            yield
        finally:
            self._pending |= kinds
            self._depth -= 1
            if self._depth == 0:
                self._flush_events()

    def _flush_events(self) -> None:
        if not self._pending or self._dispatching:
            return
        kinds, self._pending = frozenset(self._pending), set()
        self._dispatching = True
        error = None
        try:
            for listener in list(self._after):
                try:
                    listener(kinds)
                except Exception as exc:  # deliver to every listener, re-raise the first failure
                    error = error or exc
        finally:
            self._dispatching = False
        if self._pending:  # a listener issued further commands
            self._flush_events()
        if error is not None:
            raise error

    # ================================================================= recording
    def start_recording(self, recorder=None):
        """Attach a :class:`~app.scripting.recorder.Recorder` (a new one by default) and
        return it. Every top-level command from now on is recorded as Python source."""
        from app.scripting.recorder import Recorder

        self._recorder = recorder if recorder is not None else Recorder()
        return self._recorder

    def stop_recording(self):
        """Detach and return the active recorder (``None`` if not recording).
        ``recorder.script()`` is the runnable macro."""
        recorder, self._recorder = self._recorder, None
        return recorder

    @property
    def recorder(self):
        """The active recorder, or ``None``."""
        return self._recorder

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    @contextmanager
    def suspend_recording(self):
        """Commands issued inside this block are not recorded (used while the console or a
        script runs: their *source* is recorded instead of the commands it expands to)."""
        self._record_suspended += 1
        try:
            yield
        finally:
            self._record_suspended -= 1

    def _can_record(self) -> bool:
        return self._recorder is not None and not self._record_suspended and not self._dispatching

    def _record(self, target, name, args=(), kwargs=None, *, coalesce=False) -> None:
        if self._can_record():
            self._recorder.record_call(target, name, args, kwargs, coalesce=coalesce)

    def record_command(self, target: str, name: str, *args, **kwargs) -> None:
        """Record ``target.name(*args, **kwargs)`` as if a top-level command had run.

        For front ends that perform an action through lower-level calls (e.g. the GUI's pair
        alignment editor, or launching a plugin as ``app.launch_plugin(...)``)."""
        if self._depth == 0:
            self._record(target, name, args, kwargs)

    # ============================================================ read-only view
    @property
    def revision(self) -> int:
        """Monotonic version of the scientific state (navigation does not change it)."""
        return self.state.revision

    @property
    def has_sequence(self) -> bool:
        return self.state.has_sequence

    @property
    def n_images(self) -> int:
        """Number of frames in the loaded sequence (0 if none)."""
        return self.state.total_images

    @property
    def frame_paths(self) -> Tuple[str, ...]:
        """Ordered source image paths; index = global frame index."""
        sequence = self.state.sequence
        return tuple(sequence.paths) if sequence is not None else ()

    @property
    def source_dir(self) -> Optional[str]:
        return self.state.source_dir

    @property
    def is_image_pair(self) -> bool:
        return isinstance(self.state.sequence, AlignedImagePairSequence)

    @property
    def pair_alignment(self) -> Optional[dict]:
        """The aligned pair's alignment as keyword arguments, or ``None`` for a plain sequence."""
        if not self.is_image_pair:
            return None
        return self.state.sequence.alignment.metadata()

    @property
    def reference_index(self) -> int:
        return self.state.reference_index

    @property
    def last_index(self) -> int:
        return self.state.last_index

    @property
    def current_index(self) -> int:
        return self.state.current_index

    @property
    def n_range_frames(self) -> int:
        """Frames in the reference..last range (both inclusive); 0 without a sequence."""
        return self.state.n_cut if self.has_sequence else 0

    def global_to_cut(self, global_index: int) -> int:
        return self.state.global_to_cut(global_index)

    def cut_to_global(self, cut_index: int) -> int:
        return self.state.cut_to_global(cut_index)

    def image_size(self) -> Optional[Tuple[int, int]]:
        """``(height, width)`` of the frames, or ``None`` without a sequence."""
        return self.state.image_size()

    def frame(self, index: int) -> np.ndarray:
        """Frame at global ``index`` as a read-only ``(H, W, 3)`` uint8 **BGR** array."""
        return self._require_sequence().load_bgr(_index(index, "index"))

    def frame_gray(self, index: int) -> np.ndarray:
        """Frame at global ``index`` as an ``(H, W)`` uint8 grayscale array."""
        return self._require_sequence().load_gray(_index(index, "index"))

    @property
    def roi(self) -> Optional[ROI]:
        """A detached copy of the ROI (editing it does not change the session), or ``None``."""
        return deepcopy(self.state.roi)

    @property
    def roi_corners(self) -> List[Tuple[float, float]]:
        roi = self.state.roi
        return list(roi.corners) if roi is not None else []

    @property
    def seeds(self) -> Optional[np.ndarray]:
        """Copy of the ``(N, 2)`` float32 reference-frame seed points, or ``None``."""
        features = self.state.features
        return None if features is None else np.array(features, copy=True)

    @property
    def n_seeds(self) -> int:
        features = self.state.features
        return 0 if features is None else len(features)

    @property
    def corner_params(self) -> dict:
        """Shi-Tomasi / Harris parameters (copy) used by :meth:`detect_corners`."""
        return dict(self.state.shi_tomasi_params)

    @property
    def grid_params(self) -> dict:
        return dict(self.state.grid_params)

    @property
    def lk_params(self) -> dict:
        """Lucas-Kanade parameters (copy) used by :meth:`track`."""
        return dict(self.state.lk_params)

    @property
    def display_params(self) -> dict:
        return dict(self.state.display_params)

    @property
    def result(self) -> Optional[TrackerResult]:
        """The immutable raw :class:`TrackerResult` (all points, cut-indexed), or ``None``."""
        return self.state.result

    @property
    def has_result(self) -> bool:
        return self.state.result is not None

    @property
    def n_points(self) -> int:
        """Total tracked points P (before cleanup); 0 without a result."""
        return self.state.result.n_points if self.has_result else 0

    @property
    def active_mask(self) -> Optional[np.ndarray]:
        """Copy of the ``(P,)`` keep-mask (True = survived cleanup), or ``None``."""
        mask = self.state.active_mask
        return None if mask is None else mask.copy()

    @property
    def n_active(self) -> int:
        mask = self.state.active_mask
        return int(mask.sum()) if mask is not None else 0

    @property
    def can_undo_cleanup(self) -> bool:
        return bool(self.state.undo_stack)

    def point_ids(self, active_only: bool = True) -> np.ndarray:
        """Column indices (into all P points) matching :meth:`coords`' point axis."""
        result = self._require_result()
        ids = np.arange(result.n_points)
        return ids[self.state.active_mask] if active_only else ids

    def frame_indices(self) -> np.ndarray:
        """Global frame index of each row of :meth:`coords` (``reference .. last``)."""
        result = self._require_result()
        return np.arange(result.reference_index, result.last_index + 1)

    def coords(self, active_only: bool = True) -> np.ndarray:
        """Forward-tracked ``(frames, points, 2)`` float32 ``(x, y)`` coordinates (a copy).

        Axis 0 is the cut index (row 0 = reference frame). Positions where a track has failed are
        frozen at the last valid location; check :meth:`valid` before using them."""
        return np.array(self._require_result().coords_fw[:, self.point_ids(active_only)])

    def valid(self, active_only: bool = True) -> np.ndarray:
        """``(frames, points)`` bool: True where the forward track was valid."""
        return self._require_result().status_fw[:, self.point_ids(active_only)].astype(bool)

    def metrics(self) -> Metrics:
        """Per-point quality metrics of the current result (cached; see ``app.core.cleanup``)."""
        result = self._require_result()
        roi_key = tuple(self.roi_corners)
        cached = self._metrics_cache
        if cached is None or cached[0] is not result or cached[1] != roi_key:
            self._metrics_cache = (result, roi_key, compute_metrics(result, self.state.roi, self.image_size()))
        return self._metrics_cache[2]

    def cleanup_mask(self, **filters) -> np.ndarray:
        """The ``(P,)`` keep-mask :meth:`cleanup` would AND into the active mask, without
        applying it. Filters are as for :meth:`cleanup`."""
        return build_mask(self.metrics(), thresholds_from_filters(**filters))

    def exportable_mask(self) -> Optional[np.ndarray]:
        """Active points that stayed valid on every frame of the tracked range (what the
        exports write), or ``None`` without a result."""
        result, active = self.state.result, self.state.active_mask
        if result is None or active is None:
            return None
        if self._export_cache is None or self._export_cache[0] is not result:
            self._export_cache = (result, np.all(result.status_fw == 1, axis=0))
        return active & self._export_cache[1]

    def summary(self) -> str:
        """A human-readable multi-line description of the session state."""
        s = self.state
        lines = ["ECM Tracker session"]
        if not s.has_sequence:
            lines.append("  sequence : none loaded")
        else:
            h, w = self.image_size()
            kind = "image pair" if self.is_image_pair else "frames"
            lines.append(f"  sequence : {s.total_images} {kind} ({w}x{h} px) from {s.source_dir}")
            lines.append(
                f"  range    : reference {s.reference_index} -> last {s.last_index} "
                f"({s.n_cut} frames), current {s.current_index}"
            )
        roi = s.roi
        lines.append(f"  ROI      : {len(roi.corners)} corners" if roi is not None and roi.is_complete
                     else "  ROI      : none")
        lines.append(f"  seeds    : {self.n_seeds}")
        if s.result is not None:
            lines.append(
                f"  tracking : {s.result.n_points} points x {s.result.n_frames} frames, "
                f"{self.n_active} active, {int(self.exportable_mask().sum())} exportable"
            )
        else:
            lines.append("  tracking : none")
        lines.append(f"  recording: {'on' if self.is_recording else 'off'}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.state
        if not s.has_sequence:
            return "<Session: no sequence loaded>"
        parts = [f"{s.total_images} frames", f"range {s.reference_index}..{s.last_index}"]
        if s.roi is not None and s.roi.is_complete:
            parts.append("ROI")
        parts.append(f"{self.n_seeds} seeds")
        if s.result is not None:
            parts.append(f"tracked {self.n_active}/{s.result.n_points} active")
        return f"<Session: {', '.join(parts)}>"

    __str__ = summary

    # ====================================================================== guards
    def _require_sequence(self):
        if self.state.sequence is None:
            raise SessionError("No image sequence is loaded. Use open_directory() first.")
        return self.state.sequence

    def _require_roi(self) -> ROI:
        self._require_sequence()
        roi = self.state.roi
        if roi is None or not roi.is_complete:
            raise SessionError("Define a complete ROI first (set_roi / set_roi_rectangle / set_roi_circle).")
        return roi

    def _require_result(self) -> TrackerResult:
        if self.state.result is None:
            raise SessionError("No tracking result. Run track() first.")
        return self.state.result

    def _require_pair(self) -> AlignedImagePairSequence:
        sequence = self._require_sequence()
        if not isinstance(sequence, AlignedImagePairSequence):
            raise SessionError("The loaded sequence is not an aligned image pair.")
        return sequence

    def _discard_result(self) -> None:
        """Drop the result and everything derived from it (call inside a RESULT change)."""
        self.state.result = None
        self.state.active_mask = None
        self.state.undo_stack = []

    def _result_kinds(self):
        return (RESULT,) if self.state.result is not None else ()

    def _progress(self, label: str, total: int, delay_ms: int, override):
        if override is not None:
            return nullcontext(override)
        if self.progress_handler is None:
            return nullcontext(None)
        return self.progress_handler(label, total, delay_ms)

    # ============================================================ sequence loading
    @command
    def open_directory(self, path, *, progress: Optional[ProgressCb] = None) -> int:
        """Load every supported image in folder ``path`` (natural filename order).
        Returns the number of frames. Resets ROI, seeds and tracking."""
        path = os.fspath(path)
        paths = discover(path)
        if not paths:
            raise SessionError(f"No supported images were found in {path}")
        return self.load_sequence(paths, path, progress=progress)

    @command
    def open_files(self, paths: Sequence, *, progress: Optional[ProgressCb] = None) -> int:
        """Load the given image files, sorted in natural filename order."""
        files = discover([os.fspath(p) for p in paths])
        if not files:
            raise SessionError("None of the given files is a supported image.")
        return self.load_sequence(files, os.path.dirname(files[0]), progress=progress)

    @command
    def load_sequence(self, paths: Sequence, source_dir=None, *, progress: Optional[ProgressCb] = None) -> int:
        """Load an explicit, pre-ordered list of image paths (order is kept verbatim).

        Every frame is decoded and checked for a common size before anything changes, so a
        failed load leaves the session untouched. Returns the number of frames."""
        paths = [os.fspath(p) for p in paths]
        if not paths:
            raise SessionError("load_sequence() needs at least one image path.")
        sequence = ImageSequence(paths)
        if source_dir is None:
            source_dir = os.path.dirname(os.path.abspath(paths[0]))
        self.adopt_sequence(sequence, os.fspath(source_dir), progress=progress)
        return len(sequence)

    @command(record=False)
    def adopt_sequence(self, sequence: ImageSequence, source_dir=None, *, setup_path=None,
                       progress: Optional[ProgressCb] = None) -> None:
        """Install an already-constructed sequence object (validated first if needed).

        Low-level entry for front ends and plugins that build their own sequence; not recorded
        (record the high-level command that produced the sequence instead)."""
        if sequence.fingerprint is None:
            self.validate_sequence(sequence, progress)
        with self._change(SEQUENCE):
            self.state.load_sequence(sequence, source_dir)
            self.state.pair_setup_path = setup_path

    def validate_sequence(self, sequence, progress: Optional[ProgressCb] = None) -> None:
        """Decode and fingerprint every frame of ``sequence`` (no state change). Raises
        :class:`OperationCancelled` if the progress callback cancels."""
        if self._validating:
            raise RuntimeError("An image sequence is already being validated.")
        self._validating = True
        try:
            with self._progress("Validating images...", len(sequence), 500, progress) as cb:
                valid = sequence.validate_all(cb)
        finally:
            self._validating = False
        if not valid:
            raise OperationCancelled("Image loading cancelled.")

    @command
    def open_image_pair(self, reference_path, destination_path, **alignment) -> None:
        """Load two images as an aligned pair (frame 0 = reference, 1 = destination).

        ``alignment`` keywords build a :class:`~app.core.pair_transform.PairAlignment`
        (``translation=(dx, dy)``, ``mode``, ``scale``, ``affine_stretch``,
        ``rotation_enabled``, ``angle_degrees``); unspecified fields keep their defaults."""
        paths = [os.fspath(reference_path), os.fspath(destination_path)]
        sequence = AlignedImagePairSequence(paths, alignment=PairAlignment(**alignment))
        self.validate_sequence(sequence, None)
        sequence.source_bgr(0)
        sequence.source_bgr(1)
        self.adopt_sequence(sequence, os.path.dirname(sequence.paths[0]))

    @command
    def set_pair_alignment(self, **alignment) -> None:
        """Re-align the loaded image pair (same keywords as :meth:`open_image_pair`).

        A geometry change re-crops the pair and resets ROI, seeds and tracking; a change of
        editor-only presets keeps everything."""
        sequence = self._require_pair()
        new = PairAlignment(**alignment)
        if new.same_geometry(sequence.alignment):
            if new != sequence.alignment:
                with self._change(PAIR):
                    self.state.sequence = sequence.with_editor_settings(new)
            return
        candidate = sequence.with_alignment(new)
        self.validate_sequence(candidate, None)
        candidate.source_bgr(0)
        candidate.source_bgr(1)
        self.adopt_sequence(candidate, os.path.dirname(sequence.paths[0]),
                            setup_path=self.state.pair_setup_path)

    @command
    def open_pair_setup(self, path) -> None:
        """Load an image pair from a saved ``.ecmpair.json`` setup (sources are re-verified)."""
        sequence = load_pair_setup(path)
        self.adopt_sequence(sequence, os.path.dirname(sequence.paths[0]),
                            setup_path=os.path.abspath(os.fspath(path)))

    @command
    def save_pair_setup(self, path) -> str:
        """Save the aligned pair's geometry to ``path`` (``.ecmpair.json``); returns the path."""
        sequence = self._require_pair()
        with self._change(PAIR):
            self.state.pair_setup_path = save_pair_setup(path, sequence)
        return self.state.pair_setup_path

    # ============================================================== frame range
    @command(coalesce=True)
    def set_frame_range(self, reference: Optional[int] = None, last: Optional[int] = None) -> None:
        """Set the tracked range ``reference .. last`` (global indices; ``None`` keeps a bound).

        Moving the reference clears the ROI and seeds (they live on the reference frame); any
        change of the range discards the tracking result."""
        s = self.state
        self._require_sequence()
        reference = s.reference_index if reference is None else _index(reference, "reference")
        last = s.last_index if last is None else _index(last, "last")
        if not 0 <= reference <= last < s.total_images:
            raise IndexError(
                f"Frame range must satisfy 0 <= reference <= last < {s.total_images}; "
                f"got reference={reference}, last={last}"
            )
        if (reference, last) == (s.reference_index, s.last_index):
            return
        ref_changed = reference != s.reference_index
        kinds = [RANGE, *self._result_kinds()]
        if ref_changed:
            kinds += [ROI_KIND, SEEDS]
        with self._change(*kinds):
            s.reference_index, s.last_index = reference, last
            self._discard_result()
            if ref_changed:
                s.roi = s.features = None
            s.touch()

    @command(record=False)
    def set_current_frame(self, index: int) -> None:
        """Show global frame ``index`` (display/navigation only; not recorded)."""
        index = _index(index, "index")
        self._require_sequence()
        if not 0 <= index < self.state.total_images:
            raise IndexError(f"Frame index {index} is outside 0..{self.state.total_images - 1}")
        with self._change(FRAME):
            self.state.set_current(index)

    # ======================================================================= ROI
    @command
    def set_roi(self, corners: Sequence[Tuple[float, float]]) -> None:
        """Set a polygonal ROI from ``[(x, y), ...]`` image coordinates (>= 3 corners).
        Discards an existing tracking result; seed points are kept."""
        self._require_sequence()
        array = _points(corners, "ROI corners")
        if len(array) < ROI.MIN_CORNERS:
            raise ValueError(f"An ROI needs at least {ROI.MIN_CORNERS} corners")
        with self._change(ROI_KIND, *self._result_kinds()):
            self._discard_result()
            self.state.roi = ROI([(float(x), float(y)) for x, y in array])
            self.state.touch()

    @command
    def set_roi_rectangle(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Rectangular ROI spanned by the opposite corners ``(x0, y0)`` and ``(x1, y1)``."""
        x0, y0, x1, y1 = (float(_plain(v)) for v in (x0, y0, x1, y1))
        if x0 == x1 or y0 == y1:
            raise ValueError("A rectangle ROI needs a non-zero width and height")
        self.set_roi(rectangle_corners(x0, y0, x1, y1))

    @command
    def set_roi_circle(self, cx: float, cy: float, radius: float, segments: int = CIRCLE_SEGMENTS) -> None:
        """Circular ROI (a ``segments``-gon) centred on ``(cx, cy)``."""
        cx, cy, radius = (float(_plain(v)) for v in (cx, cy, radius))
        segments = _index(segments, "segments")
        if not (radius > 0 and math.isfinite(radius)) or segments < ROI.MIN_CORNERS:
            raise ValueError("A circle ROI needs a positive radius and at least 3 segments")
        self.set_roi(circle_corners(cx, cy, radius, segments))

    @command
    def clear_roi(self) -> None:
        """Remove the ROI and the seed points (and any tracking result)."""
        self._require_sequence()
        with self._change(ROI_KIND, SEEDS, *self._result_kinds()):
            self._discard_result()
            self.state.roi = None
            self.state.features = None
            self.state.touch()

    # ================================================================ parameters
    @command
    def set_corner_params(self, **params) -> dict:
        """Update Shi-Tomasi parameters (``maxCorners``, ``qualityLevel``, ``minDistance``,
        ``blockSize``, ``useHarrisDetector``, ``k``). Returns the full parameter set."""
        merged = _strict_params("corner", validate_shi_tomasi, self.state.shi_tomasi_params, params)
        if merged != self.state.shi_tomasi_params:
            with self._change(PARAMS):
                self.state.shi_tomasi_params = merged
        return dict(merged)

    @command
    def set_grid_params(self, **params) -> dict:
        """Update the regular-grid spacing (``spacing_x``, ``spacing_y`` in px)."""
        merged = _strict_params("grid", validate_grid, self.state.grid_params, params)
        if merged != self.state.grid_params:
            with self._change(PARAMS):
                self.state.grid_params = merged
        return dict(merged)

    @command
    def set_lk_params(self, **params) -> dict:
        """Update Lucas-Kanade parameters (``win_size``, ``max_level``, ``max_iter``,
        ``epsilon``, ``flags`` 0/8, ``min_eig_threshold``)."""
        merged = _strict_params("tracking", validate_lk, self.state.lk_params, params)
        if merged != self.state.lk_params:
            with self._change(PARAMS):
                self.state.lk_params = merged
        return dict(merged)

    @command(record=False)
    def set_display_params(self, **params) -> dict:
        """Update marker display preferences (``show_markers``, ``marker_size``,
        ``marker_opacity``, ``show_window_box``, ``show_roi``). Cosmetic; not recorded."""
        merged = _strict_params("display", validate_display, self.state.display_params, params)
        if merged != self.state.display_params:
            with self._change(DISPLAY):
                self.state.display_params = merged
        return dict(merged)

    # ===================================================================== seeds
    def _install_seeds(self, features) -> None:
        with self._change(SEEDS, *self._result_kinds()):
            self._discard_result()
            self.state.features = features
            self.state.touch()

    def _check_inside(self, points: np.ndarray) -> None:
        h, w = self.image_size()
        inside = (points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] < w) & (points[:, 1] < h)
        if not inside.all():
            bad = points[~inside][0]
            raise ValueError(f"Seed point ({bad[0]:g}, {bad[1]:g}) is outside the {w}x{h} image")

    @command
    def detect_corners(self, **params) -> int:
        """Detect Shi-Tomasi corners inside the ROI on the reference frame; returns the count.

        Keyword arguments update :attr:`corner_params` first (recorded macros pass all of them,
        so a replay does not depend on saved preferences). Replaces the seeds and discards any
        tracking result."""
        roi = self._require_roi()
        if params:
            self.set_corner_params(**params)
        gray = self.state.sequence.load_gray(self.state.reference_index)
        h, w = gray.shape[:2]
        features = shi_tomasi(gray, roi.mask(h, w), self.state.shi_tomasi_params)
        self._install_seeds(features)
        return len(features)

    @command
    def detect_grid(self, **params) -> int:
        """Seed a regular grid (``spacing_x``, ``spacing_y``) inside the ROI; returns the count.
        Points of an ROI that extends past the image border are dropped."""
        roi = self._require_roi()
        if params:
            self.set_grid_params(**params)
        h, w = self.image_size()
        grid = regular_grid(roi, **self.state.grid_params)
        inside = (grid[:, 0] >= 0) & (grid[:, 1] >= 0) & (grid[:, 0] < w) & (grid[:, 1] < h)
        features = grid[inside]
        self._install_seeds(features)
        return len(features)

    @command
    def set_seeds(self, points) -> int:
        """Replace the seed points with ``[(x, y), ...]`` (inside the image); returns the count."""
        self._require_sequence()
        array = _points(points, "Seed points") if len(points) else np.empty((0, 2))
        self._check_inside(array)
        self._install_seeds(array.astype(np.float32) if len(array) else None)
        return len(array)

    @command
    def add_seed(self, x: float, y: float) -> int:
        """Append one seed point; returns the new seed count."""
        return self.add_seeds([(x, y)])

    @command
    def add_seeds(self, points) -> int:
        """Append seed points ``[(x, y), ...]``; returns the new seed count."""
        self._require_sequence()
        array = _points(points, "Seed points").astype(np.float32)
        self._check_inside(array)
        current = self.state.features
        combined = array if current is None or len(current) == 0 else np.vstack([current, array])
        self._install_seeds(combined.astype(np.float32))
        return len(combined)

    @command
    def remove_seeds(self, indices: Iterable[int]) -> int:
        """Delete seed points by index; returns the remaining count."""
        features = self.state.features
        if features is None or len(features) == 0:
            raise SessionError("There are no seed points to remove.")
        indices = sorted({_index(i, "seed index") for i in indices})
        if any(not 0 <= i < len(features) for i in indices):
            raise IndexError(f"Seed indices must be in 0..{len(features) - 1}")
        remaining = np.delete(features, indices, axis=0)
        self._install_seeds(remaining if len(remaining) else None)
        return len(remaining)

    @command
    def clear_seeds(self) -> None:
        """Remove all seed points (and any tracking result)."""
        self._require_sequence()
        self._install_seeds(None)

    # ================================================================== tracking
    @command
    def track(self, *, progress: Optional[ProgressCb] = None, **lk_params) -> TrackerResult:
        """Track the seeds forward (reference -> last) and backward with pyramidal LK.

        Keyword arguments update :attr:`lk_params` first. Returns the new result and resets
        the active mask to all points. Raises :class:`OperationCancelled` if cancelled."""
        sequence = self._require_sequence()
        features = self.state.features
        if features is None or len(features) == 0:
            raise SessionError("No seed points to track. Detect or add seeds first.")
        if lk_params:
            self.set_lk_params(**lk_params)
        s = self.state
        revision = s.revision
        n = s.last_index - s.reference_index + 1
        with self._progress("Tracking...", max(1, 2 * (n - 1)), 0, progress) as cb:
            result = track(sequence, s.reference_index, s.last_index, features.copy(),
                           dict(s.lk_params), cb)
        if s.revision != revision:
            raise StaleResultError("Tracking inputs changed while tracking ran; the stale result was discarded.")
        if result is None:
            raise OperationCancelled("Tracking cancelled.")
        with self._change(RESULT):
            s.undo_stack = []
            s.touch()
            s.result = result
            s.active_mask = np.ones(result.n_points, dtype=bool)
        return result

    @command
    def clear_tracking(self) -> None:
        """Discard the tracking result (seeds and ROI are kept)."""
        if self.state.result is None:
            return
        with self._change(RESULT):
            self._discard_result()
            self.state.touch()

    # =================================================================== cleanup
    @command
    def cleanup(self, **filters) -> int:
        """Drop low-quality points; returns how many were removed. Undo with :meth:`undo_cleanup`.

        Each keyword enables one filter: band limits keep a point iff its metric is <= the
        limit — ``fw_failures``, ``bw_failures``, ``opencv_error``, ``mean_error``,
        ``fb_mean``, ``fb_max``, ``distance`` (max per-frame step, px) — except
        ``min_eigenvalue`` (keep iff >= limit); ``drop_left_image=True`` / ``drop_left_roi=True``
        drop points that ever left the image / ROI. Unlisted filters are off."""
        return self.apply_keep_mask(self.cleanup_mask(**filters))

    @command(record=False)
    def apply_keep_mask(self, keep) -> int:
        """AND a full-length ``(P,)`` bool keep-mask into the active mask, undoably; returns how
        many points were removed. Points only ever leave the active set. Recorded as the
        equivalent :meth:`remove_points` call."""
        active = self.state.active_mask
        if active is None:
            return 0
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != active.shape:
            raise ValueError(f"keep mask shape {keep.shape} does not match active mask shape {active.shape}")
        dropped = np.flatnonzero(active & ~keep)
        if dropped.size == 0:
            return 0
        with self._change(MASK):
            self.state.undo_stack.append(active.copy())
            self.state.touch()
            self.state.active_mask = active & keep
        if self._depth == 1:  # called directly, not from cleanup()/remove_points()
            self._record("session", "remove_points", (dropped.tolist(),))
        return int(dropped.size)

    @command
    def remove_points(self, indices: Iterable[int]) -> int:
        """Remove tracked points by column index (into all P points), undoably."""
        result = self._require_result()
        indices = [_index(i, "point index") for i in indices]
        if any(not 0 <= i < result.n_points for i in indices):
            raise IndexError(f"Point indices must be in 0..{result.n_points - 1}")
        keep = np.ones(result.n_points, dtype=bool)
        keep[indices] = False
        return self.apply_keep_mask(keep)

    @command
    def undo_cleanup(self) -> bool:
        """Restore the active mask from before the last cleanup/removal; False if none."""
        if not self.state.undo_stack:
            return False
        with self._change(MASK):
            self.state.touch()
            self.state.active_mask = self.state.undo_stack.pop()
        return True

    # ==================================================================== export
    def _require_exportable(self) -> np.ndarray:
        self._require_result()
        mask = self.exportable_mask()
        if not mask.any():
            raise SessionError("No fully valid tracked points remain to export.")
        return mask

    @command
    def export_npy(self, path):
        """Write surviving coordinates as ``.npy`` (+ ``.bundle.npz`` and a sequence range file).
        Returns ``(coords_path, sequence_path, (frames, points, 2))``."""
        mask = self._require_exportable()
        result = self.state.result
        path = os.fspath(path)
        return export(result.coords_fw, mask, result.reference_index, result.last_index,
                      os.path.dirname(path), os.path.basename(path))

    @command
    def export_csv(self, path):
        """Write surviving coordinates as CSV (one row per frame, ``x, y`` per point).
        Returns ``(csv_path, (frames, points, 2))``."""
        mask = self._require_exportable()
        result = self.state.result
        names = [os.path.basename(self.state.sequence.paths[g])
                 for g in range(result.reference_index, result.last_index + 1)]
        path = os.fspath(path)
        return export_csv(result.coords_fw, mask, names, os.path.dirname(path), os.path.basename(path))

    # ============================================================ save / load
    @command
    def save_trackers(self, path) -> str:
        """Save seeds, ROI, range, parameters and (if tracked) the full result + mask to one
        ``.npz``; returns the written path."""
        s = self.state
        self._require_sequence()
        if s.features is None or len(s.features) == 0:
            raise SessionError("No reference points to save.")
        result = s.result
        arrays = None if result is None else {name: getattr(result, name) for name in tracker_io.RESULT_ARRAY_DTYPES}
        return tracker_io.save_trackers(
            path,
            total_images=s.total_images,
            reference_index=s.reference_index,
            last_index=s.last_index,
            current_index=s.current_index,
            features=s.features,
            roi_corners=(s.roi.corners if s.roi is not None and s.roi.is_complete else None),
            active_mask=s.active_mask if result is not None else None,
            result_arrays=arrays,
            win_size=(result.win_size if result is not None else s.lk_params.get("win_size", 0)),
            lk_params=s.lk_params,
            shi_tomasi_params=s.shi_tomasi_params,
            grid_params=s.grid_params,
            sequence_fingerprint=s.sequence.fingerprint,
            error_kind=result.error_kind if result is not None else "unknown",
            tracking_params=dict(result.tracking_params) if result is not None else {},
        )

    @command
    def load_trackers(self, path, *, expected_range=None) -> dict:
        """Overlay a tracker ``.npz`` onto the loaded sequence (it must be the same images);
        restores seeds, ROI, range, parameters, result and mask. Returns the loaded bundle."""
        s = self.state
        self._require_sequence()
        bundle = tracker_io.load_trackers(path)
        if bundle["total_images"] != s.total_images:
            raise ValueError(
                f"This tracker file is for a {bundle['total_images']}-frame sequence, but the "
                f"open sequence has {s.total_images} frames."
            )
        expected, actual = bundle.get("sequence_fingerprint"), s.sequence.fingerprint
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                "This tracker file belongs to a different image sequence. The frame count "
                "matches, but the ordered image-content fingerprint does not."
            )
        if expected_range is not None and tuple(expected_range) != (bundle["reference_index"], bundle["last_index"]):
            raise ValueError("Saved tracker range does not match the expected project reference.")
        candidate = (TrackerResult(reference_index=bundle["reference_index"],
                                   last_index=bundle["last_index"], win_size=bundle["win_size"],
                                   error_kind=bundle.get("error_kind", "unknown"),
                                   tracking_params=bundle.get("tracking_params", {}),
                                   **bundle["result_arrays"]) if bundle["has_result"] else None)
        with self._change(RANGE, SEEDS, ROI_KIND, RESULT, MASK, FRAME, PARAMS):
            s.touch()
            s.undo_stack = []
            # tracker_io has already validated the complete frame-range invariant.
            s.reference_index = bundle["reference_index"]
            s.last_index = bundle["last_index"]
            s.set_current(bundle["current_index"])
            s.roi = ROI(bundle["roi_corners"]) if bundle["roi_corners"] is not None else None
            s.features = bundle["features"]
            s.result = candidate
            s.active_mask = bundle["active_mask"] if candidate is not None else None
            # Parameters that produced this session (merged over built-ins, like ProjectState).
            s.lk_params = validate_lk(bundle["lk_params"])
            s.shi_tomasi_params = validate_shi_tomasi(bundle["shi_tomasi_params"])
            s.grid_params = validate_grid(bundle["grid_params"])
        return bundle
