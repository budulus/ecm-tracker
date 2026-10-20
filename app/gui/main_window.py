import os
from contextlib import contextmanager

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core import settings
from app.gui.icon_loader import ACCENT, load_icon
from app.core.cleanup import build_mask, filters_from_thresholds, thresholds_from_dict
from app.core.image_sequence import discover
from app.core.image_pair import AlignedImagePairSequence
from app.core.pair_transform import PairAlignment
from app.core.roi import ROI
from app.gui.canvas_view import CanvasView
from app.gui.point_tools import AddPointsTool, DeletePointsTool, snap
from app.gui.pair_alignment import ImagePairDialog, PairAlignmentDialog
from app.gui.point_manager import PointManagerDialog, PointSelectInteraction
from app.gui.roi_tools import CircleTool, NGonTool, RectangleTool
from app.gui.cleanup_dialog import CleanupDialog
from app.gui.dialogs import CornerDetectionDialog, DisplayDialog, GridDialog, TrackerDialog
from app.gui.scripting.host import ScriptingHost
from app.models.project_state import ProjectState
from app.plugins.api import PluginSignals
from app.plugins.manager import PluginManager
from app.scripting.session import (
    DISPLAY,
    FRAME,
    MASK,
    PAIR,
    PARAMS,
    RANGE,
    RESULT,
    ROI_KIND,
    SEEDS,
    SEQUENCE,
    OperationCancelled,
    Session,
    StaleResultError,
)

# Plugin signals emitted after a session change, in this order (``frame_changed`` comes last).
_KIND_SIGNALS = (
    (RANGE, "range_changed"),
    (SEEDS, "seeds_changed"),
    (ROI_KIND, "roi_changed"),
    (RESULT, "result_changed"),
    (MASK, "mask_changed"),
)


def _alignment_kwargs(alignment: PairAlignment) -> dict:
    """The non-default fields of ``alignment``: the keywords that rebuild it in a script."""
    default = PairAlignment().metadata()
    return {k: v for k, v in alignment.metadata().items() if v != default[k] or k == "translation"}


class LabeledSlider(QWidget):
    """A label + horizontal slider + spinbox kept in sync.

    `valueChanged` fires only on user interaction; `setValue`/`setRange` are programmatic and
    do not emit, which avoids reentrant update loops between interdependent sliders.
    """

    valueChanged = Signal(int)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.spin = QSpinBox()

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.spin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addLayout(row)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.setEnabled(False)

    def _on_slider(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _on_spin(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def setRange(self, low: int, high: int) -> None:
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setRange(low, high)
            widget.blockSignals(False)

    def setValue(self, value: int) -> None:
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def value(self) -> int:
        return self.slider.value()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ECM Tracker")
        self.resize(1400, 850)

        self.state = ProjectState()
        # Every state change goes through the scripting kernel, so toolbar actions, plugins and
        # Python scripts share one code path (and the macro recorder sees all of them). The two
        # listeners tear down editors bound to old state and refresh the UI afterwards.
        self.session = Session(self.state)
        self.session.progress_handler = self._progress_dialog
        self.session.add_listener(self._on_session_changed, before=self._on_session_changing)
        self.canvas = CanvasView(self.state)
        self.signals = PluginSignals()  # state-change hub broadcast to plugins
        self._status_label = QLabel()
        self.statusBar().addPermanentWidget(self._status_label)

        self._cleanup_dialog = None
        self._cleanup_metrics = None
        self._preview_keep = None
        self._point_manager = None

        self.current_slider = LabeledSlider("Current")
        self.reference_slider = LabeledSlider("Reference")
        self.last_slider = LabeledSlider("Last")
        self.current_slider.valueChanged.connect(self._on_current_changed)
        self.reference_slider.valueChanged.connect(self._on_reference_changed)
        self.last_slider.valueChanged.connect(self._on_last_changed)

        # ---- left pane: titled cards --------------------------------------
        range_card = QGroupBox("Frame Range")
        range_layout = QVBoxLayout(range_card)
        range_layout.setContentsMargins(10, 8, 10, 10)
        range_layout.setSpacing(8)
        range_layout.addWidget(self.current_slider)
        range_layout.addWidget(self.reference_slider)
        range_layout.addWidget(self.last_slider)

        plugins_card = QGroupBox("Plugins")
        plugins_layout = QVBoxLayout(plugins_card)
        plugins_layout.setContentsMargins(10, 8, 10, 10)
        plugins_layout.setSpacing(6)  # populated with launch buttons after plugin discovery

        self.side_pane = QWidget()
        pane_layout = QVBoxLayout(self.side_pane)
        pane_layout.setContentsMargins(8, 8, 8, 8)
        pane_layout.setSpacing(10)
        pane_layout.addWidget(range_card)
        pane_layout.addWidget(plugins_card)
        pane_layout.addStretch(1)  # cards hug the top; future cards stack downward

        # ---- splitter as central widget: [pane | canvas] ------------------
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.side_pane)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 0)   # pane keeps its size on window resize
        self.splitter.setStretchFactor(1, 1)   # canvas absorbs extra space
        self.splitter.setCollapsible(0, True)  # pane can be dragged shut
        self.splitter.setCollapsible(1, False)  # never collapse the canvas
        self.setCentralWidget(self.splitter)

        self._restore_pane_width()

        self._activating_point_tool = False  # reentrancy guard for Add/Delete mutual exclusivity
        self._roi_before_definition = None  # complete ROI to restore if a redefinition is cancelled

        self.scripting = ScriptingHost(self)  # console dock, Script Editor, macro recorder
        self._recording_label = QLabel("● REC")
        self._recording_label.setObjectName("recordingIndicator")
        self._recording_label.setToolTip("A macro is being recorded")
        self._recording_label.hide()
        self.statusBar().insertPermanentWidget(0, self._recording_label)  # stays visible over messages
        self.scripting.recording_changed.connect(self._recording_label.setVisible)

        self._build_menus()
        self._build_toolbar()

        # Esc cancels an in-progress ROI definition (the menu-only Define button has no
        # toggle-off affordance).
        self._roi_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._roi_escape.activated.connect(self._cancel_roi_definition)

        # ←/→ step the Current frame, scoped to the canvas (WidgetWithChildrenShortcut) so
        # they don't hijack arrow keys from the sliders/spin boxes or from plugin windows
        # (a window-wide shortcut on these navigation keys conflicts widely and is unstable).
        self._prev_frame_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self.canvas)
        self._prev_frame_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._prev_frame_shortcut.activated.connect(
            lambda: self._go_to_frame(self.state.current_index - 1)
        )
        self._next_frame_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self.canvas)
        self._next_frame_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_frame_shortcut.activated.connect(
            lambda: self._go_to_frame(self.state.current_index + 1)
        )

        # n/r/c: jump to the reference frame and start an N-Gon / Rectangle / Circle ROI.
        # Window-scoped (letters don't conflict with the slider/spin-box widgets the way ←/→ do).
        self._roi_shape_shortcuts = []
        for key, shape in ((Qt.Key.Key_N, "ngon"),
                           (Qt.Key.Key_R, "rectangle"),
                           (Qt.Key.Key_C, "circle")):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda shape=shape: self._begin_roi_on_reference(shape))
            self._roi_shape_shortcuts.append(sc)

        # Discover installed plugins; populate the pane launch buttons + the utility menu.
        self.plugin_manager = PluginManager(self)
        self.plugin_manager.discover()
        self.plugin_manager.build_menu(self._plugins_menu)
        self.plugin_manager.build_panel(plugins_layout)

        self._update_status()
        self._update_tool_states()

    # ---- menus ----------------------------------------------------------
    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_dir = file_menu.addAction("Open &Directory...")
        open_dir.setShortcut("Ctrl+O")
        open_dir.triggered.connect(self._open_directory)
        open_files = file_menu.addAction("Open &Files...")
        open_files.triggered.connect(self._open_files)
        file_menu.addAction("Open Image &Pair...").triggered.connect(self._open_image_pair)
        file_menu.addAction("Open Pair Setup...").triggered.connect(self._open_pair_setup)
        self.save_pair_setup_action = file_menu.addAction("Save Pair Setup...")
        self.save_pair_setup_action.triggered.connect(self._save_pair_setup)
        file_menu.addSeparator()
        self.save_trackers_action = file_menu.addAction("Save &Trackers...")
        self.save_trackers_action.setShortcut("Ctrl+S")
        self.save_trackers_action.triggered.connect(self._save_trackers)
        self.load_trackers_action = file_menu.addAction("&Load Trackers...")
        self.load_trackers_action.setShortcut("Ctrl+L")
        self.load_trackers_action.triggered.connect(self._load_trackers)
        file_menu.addSeparator()
        self.export_action = file_menu.addAction("&Export...")
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self._export)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        params_menu = self.menuBar().addMenu("&Parameters")
        params_menu.addAction("Corner Detection...").triggered.connect(
            self._open_corner_dialog
        )
        params_menu.addAction("Grid...").triggered.connect(self._open_grid_dialog)
        params_menu.addAction("Tracker...").triggered.connect(self._open_tracker_dialog)
        params_menu.addSeparator()
        self.adjust_pair_action = params_menu.addAction("Adjust Pair Alignment...")
        self.adjust_pair_action.triggered.connect(self._adjust_pair_alignment)

        # Populated by the PluginManager after construction.
        self._plugins_menu = self.menuBar().addMenu("&Plugins")

        scripting = self.scripting
        script_menu = self.menuBar().addMenu("&Scripting")
        script_menu.addAction(scripting.console_action)
        script_menu.addAction(scripting.editor_action)
        script_menu.addSeparator()
        script_menu.addAction(scripting.record_action)
        script_menu.addSeparator()
        script_menu.addAction(scripting.run_file_action)
        script_menu.addAction(scripting.stop_action)

        help_menu = self.menuBar().addMenu("&Help")
        view_help = help_menu.addAction("View &Help")
        view_help.setShortcut("F1")
        view_help.triggered.connect(self._open_help)
        help_menu.addAction("&About...").triggered.connect(self._open_about)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Tools")
        toolbar.setMovable(False)

        # Each action keeps its original slot / enable-disable wiring; we only attach
        # an icon and host it inside a QToolButton, so _update_tool_states is unchanged.
        # Button labels are kept short because the group caption carries the context.
        # Checkable only for the active-while-defining highlight; the button is an
        # instant-popup menu (below), so it is never toggled by a click.
        self.define_roi_action = QAction(load_icon("frame"), "Define", self)
        self.define_roi_action.setCheckable(True)
        self.define_roi_action.setToolTip("Define an ROI — Rectangle, Circle, or N-Gon")

        self._roi_menu = QMenu(self)
        self._roi_menu.addAction("Rectangle", lambda: self._begin_roi_definition("rectangle"))
        self._roi_menu.addAction("Circle", lambda: self._begin_roi_definition("circle"))
        self._roi_menu.addAction("N-Gon", lambda: self._begin_roi_definition("ngon"))

        self.clear_roi_action = QAction(load_icon("square-x"), "Clear", self)
        self.clear_roi_action.setToolTip("Remove the current ROI")
        self.clear_roi_action.triggered.connect(self._clear_roi)

        self.detect_corners_action = QAction(load_icon("scan"), "Corners", self)
        self.detect_corners_action.setToolTip("Shi-Tomasi corners inside the ROI")
        self.detect_corners_action.triggered.connect(self._detect_shi_tomasi)

        self.detect_grid_action = QAction(load_icon("grid"), "Grid", self)
        self.detect_grid_action.setToolTip("Regular grid of points inside the ROI")
        self.detect_grid_action.triggered.connect(self._detect_grid)

        self.add_points_action = QAction(load_icon("circle-plus"), "Add", self)
        self.add_points_action.setCheckable(True)
        self.add_points_action.setToolTip("Click the image to add seed points")
        self.add_points_action.toggled.connect(self._on_add_points_toggled)

        self.delete_points_action = QAction(load_icon("circle-minus"), "Delete", self)
        self.delete_points_action.setCheckable(True)
        self.delete_points_action.setToolTip(
            "Click a point to remove it (a seed point, or a tracked point after tracking)"
        )
        self.delete_points_action.toggled.connect(self._on_delete_points_toggled)

        self.point_manager_action = QAction(load_icon("list"), "Manage", self)
        self.point_manager_action.setToolTip(
            "Open the Point Manager: list, select, and delete points"
        )
        self.point_manager_action.triggered.connect(self._open_point_manager)

        self.run_tracking_action = QAction(load_icon("play", ACCENT), "Run", self)
        self.run_tracking_action.setToolTip("Track features forward and backward")
        self.run_tracking_action.triggered.connect(self._run_tracking)

        self.clear_tracking_action = QAction(load_icon("trash"), "Clear", self)
        self.clear_tracking_action.setToolTip("Discard the tracking result")
        self.clear_tracking_action.triggered.connect(self._on_clear_tracking_clicked)

        self.cleanup_action = QAction(load_icon("sliders"), "Cleanup", self)
        self.cleanup_action.setToolTip("Filter out low-quality tracks")
        self.cleanup_action.triggered.connect(self._open_cleanup)

        self.export_toolbar_action = QAction(load_icon("download"), "Export", self)
        self.export_toolbar_action.setToolTip("Export surviving coordinates")

        self._export_menu = QMenu(self)
        self._export_menu.addAction("Save as NumPy array (.npy)", self._export)
        self._export_menu.addAction("Save as CSV (.csv)", self._export_csv)

        self.zoom_in_action = QAction(load_icon("zoom-in"), "Zoom In", self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.setToolTip("Zoom in (Cmd+=)")
        self.zoom_in_action.triggered.connect(self.canvas.zoom_in)

        self.zoom_out_action = QAction(load_icon("zoom-out"), "Zoom Out", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.setToolTip("Zoom out (Cmd+-)")
        self.zoom_out_action.triggered.connect(self.canvas.zoom_out)

        self.reset_view_action = QAction(load_icon("maximize"), "Fit", self)
        self.reset_view_action.setShortcut("Ctrl+0")
        self.reset_view_action.setToolTip("Fit the image to the window (Cmd+0)")
        self.reset_view_action.triggered.connect(self.canvas.reset_view)

        self.pan_tool_action = QAction(load_icon("hand"), "Pan", self)
        self.pan_tool_action.setCheckable(True)
        self.pan_tool_action.setToolTip(
            "Hand tool: drag to pan. Or hold Spacebar and drag at any time."
        )
        self.pan_tool_action.toggled.connect(self._on_pan_tool_toggled)

        self.display_action = QAction(load_icon("eye"), "Display", self)
        self.display_action.setToolTip(
            "Marker size, opacity, visibility, and the window-size box"
        )
        self.display_action.triggered.connect(self._open_display_dialog)

        # Navigation: convenience jumps for the Current frame. Lambdas read live
        # state at click time so a later reference/last change is honored.
        self.go_reference_action = QAction(load_icon("skip-back"), "Reference", self)
        self.go_reference_action.setToolTip("Jump to the reference frame")
        self.go_reference_action.triggered.connect(
            lambda: self._go_to_frame(self.state.reference_index)
        )

        self.prev_frame_action = QAction(load_icon("chevron-left"), "Prev", self)
        self.prev_frame_action.setToolTip("Previous frame (←)")
        self.prev_frame_action.triggered.connect(
            lambda: self._go_to_frame(self.state.current_index - 1)
        )

        self.next_frame_action = QAction(load_icon("chevron-right"), "Next", self)
        self.next_frame_action.setToolTip("Next frame (→)")
        self.next_frame_action.triggered.connect(
            lambda: self._go_to_frame(self.state.current_index + 1)
        )

        self.go_last_action = QAction(load_icon("skip-forward"), "Last", self)
        self.go_last_action.setToolTip("Jump to the last frame")
        self.go_last_action.triggered.connect(
            lambda: self._go_to_frame(self.state.last_index)
        )

        toolbar.addWidget(
            self._toolbar_group(
                "NAVIGATE",
                [
                    self.go_reference_action,
                    self.prev_frame_action,
                    self.next_frame_action,
                    self.go_last_action,
                ],
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "ROI",
                [self.define_roi_action, self.clear_roi_action],
                menus={self.define_roi_action: self._roi_menu},
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "DETECT", [self.detect_corners_action, self.detect_grid_action]
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "POINTS",
                [
                    self.add_points_action,
                    self.delete_points_action,
                    self.point_manager_action,
                ],
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "TRACK",
                [
                    self.run_tracking_action,
                    self.clear_tracking_action,
                    self.cleanup_action,
                    self.export_toolbar_action,
                ],
                primary=self.run_tracking_action,
                menus={self.export_toolbar_action: self._export_menu},
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "VIEW",
                [
                    self.zoom_in_action,
                    self.zoom_out_action,
                    self.reset_view_action,
                    self.pan_tool_action,
                    self.display_action,
                ],
            )
        )
        toolbar.addSeparator()
        toolbar.addWidget(
            self._toolbar_group(
                "SCRIPT",
                [
                    self.scripting.console_action,
                    self.scripting.editor_action,
                    self.scripting.record_action,
                ],
            )
        )
        self._toolbar = toolbar

    def _toolbar_group(self, title: str, actions, primary=None, menus=None) -> QWidget:
        """A captioned cluster of QToolButtons hosting the given actions.

        Each button proxies its QAction via setDefaultAction, so the action's
        existing enabled/checked state drives the button automatically. ``primary``
        marks one action's button as the accent button (objectName for QSS).
        ``menus`` maps an action to a QMenu, turning that button into an
        instant-popup dropdown (objectName for the chevron QSS).
        """
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        for action in actions:
            button = QToolButton()
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setAutoRaise(True)
            if action is primary:
                button.setObjectName("primaryAction")
            if menus and action in menus:
                button.setMenu(menus[action])
                button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
                button.setObjectName("menuButton")
            row.addWidget(button)
        outer.addLayout(row)

        caption = QLabel(title)
        caption.setObjectName("toolGroupCaption")
        caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(caption)
        return box

    # ---- sequence loading ----------------------------------------------
    def _open_directory(self) -> None:
        start = self.state.source_dir or ""
        directory = QFileDialog.getExistingDirectory(self, "Open Image Directory", start)
        if not directory:
            return
        if not discover(directory):
            QMessageBox.warning(self, "No images", "No supported images were found.")
            return
        self._run_load(self.session.open_directory, directory)

    def _open_files(self) -> None:
        start = self.state.source_dir or ""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Images",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not files:
            return
        if not discover(files):
            QMessageBox.warning(self, "No images", "No supported images were found.")
            return
        self._run_load(self.session.open_files, files)

    def _run_load(self, command, *args, **kwargs) -> bool:
        """Run a session loading command with the GUI's error reporting; True on success."""
        try:
            command(*args, **kwargs)
        except OperationCancelled:
            self.statusBar().showMessage("Image loading cancelled.", 4000)
            return False
        except (IOError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return False
        return True

    def load_sequence_from_paths(self, paths, source_dir) -> bool:
        """Public entry to load an explicit, pre-ordered image-path list.

        Order is preserved verbatim (``ImageSequence`` does not re-sort). Resets all downstream
        state and emits ``sequence_changed``, exactly like File -> Open. Used by loader plugins
        that order frames by an acquisition log rather than by filename."""
        return self._load_paths(paths, source_dir)

    def _load_paths(self, paths, source_dir) -> bool:
        if not paths:
            QMessageBox.warning(self, "No images", "No supported images were found.")
            return False
        return self._run_load(self.session.load_sequence, list(paths), source_dir)

    def _validate_sequence(self, sequence) -> bool:
        """Decode/fingerprint a candidate sequence with GUI error reporting (no state change)."""
        return self._run_load(self.session.validate_sequence, sequence)

    def _load_sequence_candidate(self, sequence, source_dir, *, setup_path=None) -> bool:
        if not self._validate_sequence(sequence):
            return False
        self._commit_sequence(sequence, source_dir, setup_path=setup_path)
        return True

    def _commit_sequence(self, sequence, source_dir, *, setup_path=None) -> None:
        # Commit only after every frame has decoded and passed the common-shape check. A failed
        # candidate therefore leaves the existing project completely untouched.
        self.session.adopt_sequence(sequence, source_dir, setup_path=setup_path)

    @contextmanager
    def _progress_dialog(self, label: str, total: int, delay_ms: int):
        """Session progress handler: a cancellable, window-modal progress dialog. A script's
        Stop request cancels the operation too."""
        dialog = QProgressDialog(label, "Cancel", 0, total, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(delay_ms)
        if delay_ms == 0:
            dialog.setValue(0)

        def progress(done, _total):
            dialog.setValue(done)
            QApplication.processEvents()
            return dialog.wasCanceled() or self.scripting.stop_requested

        try:
            yield progress
        finally:
            dialog.close()
            dialog.deleteLater()

    # ---- session change notifications -----------------------------------
    def _on_session_changing(self, kinds) -> None:
        """Before a session change: release UI bound to the state that is about to change."""
        if kinds & {SEQUENCE, RANGE}:
            self._teardown_session_ui()
            return
        if kinds - {FRAME, PARAMS, DISPLAY, PAIR}:
            self._cancel_roi_definition(silent=True)
        if RESULT in kinds:
            if self._cleanup_dialog is not None:
                self._cleanup_dialog.close()  # bound to the old result's metrics
            self._preview_keep = None
            self.canvas.set_preview_mask(None)

    def _on_session_changed(self, kinds) -> None:
        """After a session command: refresh widgets, then notify plugins."""
        if SEQUENCE in kinds:
            self._configure_sliders()
            self.canvas.reset_view()
            self._update_window_title()
        elif kinds & {RANGE, FRAME}:
            self._configure_sliders()
        if kinds & {SEQUENCE, FRAME}:
            self.canvas.refresh()  # a different image is shown
        else:
            self.canvas.update()  # overlays only
        self._update_status()
        self._update_tool_states()
        if SEQUENCE in kinds:
            self.signals.sequence_changed.emit()
        else:
            for kind, name in _KIND_SIGNALS:
                if kind in kinds:
                    getattr(self.signals, name).emit()
            if FRAME in kinds:
                self.signals.frame_changed.emit(self.state.current_index)
        if MASK in kinds:
            self._cleanup_preview()  # keep an open Cleanup dialog's counts current

    def set_scripts_running(self, running: bool) -> None:
        """Lock the window's controls while console/editor code runs (it pumps events)."""
        if running:
            self._deactivate_point_tools()
        for widget in (self.menuBar(), self._toolbar, self.side_pane, self.canvas):
            widget.setEnabled(not running)
        for shortcut in (self._roi_escape, *self._roi_shape_shortcuts):
            shortcut.setEnabled(not running)

    def _open_image_pair(self) -> None:
        chooser = ImagePairDialog(self.state.source_dir or "", self)
        try:
            if chooser.exec() != QDialog.DialogCode.Accepted:
                return
            sequence = AlignedImagePairSequence(chooser.paths)
        finally:
            chooser.deleteLater()
        if self._validate_sequence(sequence):
            self._align_pair(sequence, editing=False)

    def _adjust_pair_alignment(self) -> None:
        sequence = self.state.sequence
        if isinstance(sequence, AlignedImagePairSequence):
            self._align_pair(sequence, editing=True)

    def _align_pair(self, sequence, *, editing) -> None:
        revision = self.state.revision
        try:
            dialog = PairAlignmentDialog(sequence, self)
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                alignment = dialog.alignment
            finally:
                dialog.deleteLater()  # Release the full-resolution preview images after closing.
            if self.state.revision != revision:
                self.statusBar().showMessage("Session changed while aligning. Reopen alignment to try again.", 5000)
                return
            if editing and alignment.same_geometry(sequence.alignment):
                if alignment != sequence.alignment:
                    self.session.set_pair_alignment(**_alignment_kwargs(alignment))
                return
            candidate = sequence.with_alignment(alignment)
            if not self._validate_sequence(candidate):
                return
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Alignment failed", str(exc))
            return
        if self.state.revision != revision:
            self.statusBar().showMessage("Session changed while aligning. Reopen alignment to try again.", 5000)
            return
        if editing and (self.state.roi is not None or self.state.features is not None
                        or self.state.result is not None):
            answer = QMessageBox.question(
                self, "Apply changed alignment?",
                "Changing alignment clears the ROI, reference points, tracking, and cleanup history. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # Confirmation runs a nested Qt event loop. Check again before discarding current work.
        if self.state.revision != revision:
            self.statusBar().showMessage("Session changed while aligning. Reopen alignment to try again.", 5000)
            return
        try:
            candidate.source_bgr(0)
            candidate.source_bgr(1)
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Alignment failed", str(exc))
            return
        setup_path = self.state.pair_setup_path if editing else None
        self._commit_sequence(candidate, os.path.dirname(sequence.paths[0]), setup_path=setup_path)
        if editing:
            self.session.record_command("session", "set_pair_alignment", **_alignment_kwargs(alignment))
        else:
            self.session.record_command("session", "open_image_pair", *candidate.paths,
                                        **_alignment_kwargs(alignment))
        self.statusBar().showMessage("Aligned pair loaded. Use File → Save Pair Setup to save this alignment.", 6000)

    def _open_pair_setup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pair Setup", self.state.source_dir or "",
            "Image pair setup (*.ecmpair.json)",
        )
        if not path:
            return
        try:
            self.session.open_pair_setup(path)
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Open pair setup failed", str(exc))

    def _save_pair_setup(self) -> None:
        if not isinstance(self.state.sequence, AlignedImagePairSequence):
            return
        default = self.state.pair_setup_path or os.path.join(self.state.source_dir or "", "pair.ecmpair.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pair Setup", default, "Image pair setup (*.ecmpair.json)",
        )
        if not path:
            return
        try:
            self.session.save_pair_setup(path)
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Save pair setup failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved pair setup to {self.state.pair_setup_path}", 6000)

    def _update_window_title(self) -> None:
        """Reflect the loaded folder's full absolute path in the title bar."""
        name = os.path.abspath(self.state.source_dir) if self.state.source_dir else None
        if isinstance(self.state.sequence, AlignedImagePairSequence):
            name = "Image Pair — " + " → ".join(os.path.basename(p) for p in self.state.sequence.paths)
        self.setWindowTitle(f"ECM Tracker - {name}" if name else "ECM Tracker")

    # ---- left pane ------------------------------------------------------
    def _restore_pane_width(self) -> None:
        ui = settings.get_section("ui") or {}
        try:
            width = max(0, int(ui.get("left_pane_width", 280)))
        except (TypeError, ValueError):
            width = 280
        total = max(self.width(), width + 100)  # window not yet shown -> sane fallback
        self.splitter.setSizes([width, total - width])

    def closeEvent(self, event) -> None:
        if self.scripting.busy:
            self.scripting.request_stop()
            event.ignore()
            return
        if not self.scripting.confirm_quit():
            event.ignore()
            return
        self.scripting.shutdown()
        sizes = self.splitter.sizes()
        if sizes:
            try:
                settings.update_section("ui", {"left_pane_width": int(sizes[0])})
            except OSError as exc:
                self.statusBar().showMessage(f"Could not save UI settings: {exc}", 5000)
        self.plugin_manager.shutdown()
        super().closeEvent(event)

    def _teardown_session_ui(self) -> None:
        """Release old-state UI before committing a replacement."""
        self._cancel_roi_definition(silent=True)
        self._deactivate_point_tools()
        self.canvas.clear_interaction()
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.close()
        if self._point_manager is not None:
            self._point_manager.close()
        self._preview_keep = None
        self.canvas.set_preview_mask(None)

    def _configure_sliders(self) -> None:
        total = self.state.total_images
        for slider in (self.current_slider, self.reference_slider, self.last_slider):
            slider.setEnabled(True)
            slider.setRange(0, total - 1)
        self.current_slider.setValue(self.state.current_index)
        self.reference_slider.setValue(self.state.reference_index)
        self.last_slider.setValue(self.state.last_index)
        self._sync_range_constraints()

    def _sync_range_constraints(self) -> None:
        """Enforce 0 <= reference <= last < total at the widget level."""
        total = self.state.total_images
        self.reference_slider.setRange(0, self.state.last_index)
        self.last_slider.setRange(self.state.reference_index, total - 1)

    # ---- slider handlers ------------------------------------------------
    def _on_current_changed(self, value: int) -> None:
        previous = self.state.current_index
        try:
            self.session.set_current_frame(value)
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            # The canvas could not decode the frame; it still shows `previous`.
            self.state.set_current(previous)
            self.current_slider.setValue(previous)
            self._update_status()
            self._update_tool_states()
            QMessageBox.critical(self, "Frame load failed", str(exc))

    def _on_reference_changed(self, value: int) -> None:
        self.set_frame_range(value, self.state.last_index)

    def _on_last_changed(self, value: int) -> None:
        self.set_frame_range(self.state.reference_index, value)

    def set_frame_range(self, reference: int, last: int) -> None:
        """Commit the final range once. Component notifications observe the complete change."""
        if not self.state.has_sequence:
            return
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (reference, last)):
            raise ValueError("Frame indices must be integers")
        reference = max(0, min(int(reference), self.state.total_images - 1))
        last = max(reference, min(int(last), self.state.total_images - 1))
        self.session.set_frame_range(reference, last)  # no-op if unchanged

    def _go_to_frame(self, global_index: int) -> None:
        """Move Current to ``global_index`` (clamped), driving the same refresh
        path as the Current slider."""
        if not self.state.has_sequence:
            return
        target = max(0, min(global_index, self.state.total_images - 1))
        self.current_slider.setValue(target)  # sync widget (setValue blocks signals)
        self._on_current_changed(target)      # session -> canvas + status + tools + signal

    def _set_reference_frame(self, global_index: int) -> None:
        self.set_frame_range(min(global_index, self.state.last_index), self.state.last_index)

    def _set_last_frame(self, global_index: int) -> None:
        self.set_frame_range(self.state.reference_index, global_index)

    # ---- ROI ------------------------------------------------------------
    def _on_pan_tool_toggled(self, checked: bool) -> None:
        self.canvas.set_pan_tool(checked)
        if checked:
            self._deactivate_point_tools()
            self._cancel_roi_definition()

    # ---- point editing tools -------------------------------------------
    def _on_add_points_toggled(self, checked: bool) -> None:
        self._on_point_tool_toggled(self.add_points_action, AddPointsTool, checked)

    def _on_delete_points_toggled(self, checked: bool) -> None:
        self._on_point_tool_toggled(self.delete_points_action, DeletePointsTool, checked)

    def _on_point_tool_toggled(self, action, tool_cls, checked: bool) -> None:
        """Drive a point tool from its QAction's checked state (the single source of truth).

        On activate, deactivate every competing tool (the other point tool, Pan, an in-progress ROI
        definition) under a reentrancy guard, then install the canvas interaction. On deactivate,
        drop the interaction. Cascaded ``setChecked(False)`` calls re-enter the point-tool handlers,
        but only ever on the deactivate path, so this cannot recurse.
        """
        if checked:
            if not self._activating_point_tool:
                self._activating_point_tool = True
                try:
                    sibling = (
                        self.delete_points_action
                        if action is self.add_points_action
                        else self.add_points_action
                    )
                    sibling.setChecked(False)            # tears down its interaction first
                    self.pan_tool_action.setChecked(False)
                    self._cancel_roi_definition()        # no-op unless mid-definition
                finally:
                    self._activating_point_tool = False
            self.canvas.set_interaction(tool_cls(self))
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            # Only clear if THIS family owns the current interaction (guards a foreign handler, e.g.
            # an ROI tool or plugin, that may have replaced ours).
            if isinstance(self.canvas._interaction, (AddPointsTool, DeletePointsTool)):
                self.canvas.clear_interaction()
            self.canvas.unsetCursor()

    def _deactivate_point_tool(self, action) -> None:
        """Uncheck one point tool (its toggled handler clears the interaction). Guarded so the
        cascade stays pure teardown."""
        self._activating_point_tool = True
        try:
            action.setChecked(False)
        finally:
            self._activating_point_tool = False

    def _deactivate_point_tools(self) -> None:
        self._deactivate_point_tool(self.add_points_action)
        self._deactivate_point_tool(self.delete_points_action)

    def _begin_roi_on_reference(self, shape: str) -> None:
        """Jump to the reference frame, then start defining an ROI of ``shape``.

        Bound to the n/r/c window shortcuts (n-gon / rectangle / circle)."""
        if not self.state.has_sequence:
            return
        self._go_to_frame(self.state.reference_index)
        self._begin_roi_definition(shape)

    def _begin_roi_definition(self, shape: str) -> None:
        """Start defining an ROI of the given shape on the reference frame."""
        if not (self.state.has_sequence and self.state.on_reference_frame):
            return
        if self.pan_tool_action.isChecked():
            self.pan_tool_action.setChecked(False)
        self._deactivate_point_tools()
        if not self._confirm_discard_tracking():
            return
        self._cancel_roi_definition(silent=True)
        self._roi_before_definition = self.state.roi
        self.state.roi = ROI()
        self.define_roi_action.setChecked(True)
        if shape == "ngon":
            self.canvas.set_interaction(NGonTool(self))
            message = "Left-click to add points; right-click to close the ROI (Esc to cancel)."
        elif shape == "rectangle":
            self.canvas.set_interaction(RectangleTool(self))
            message = "Drag a rectangle from corner to corner (Esc to cancel)."
        else:  # circle
            self.canvas.set_interaction(CircleTool(self))
            message = "Press the center and drag out the radius (Esc to cancel)."
        self.statusBar().showMessage(message, 6000)
        self.canvas.update()
        self._update_tool_states()

    def _commit_interactive_roi(self, shape: str, args) -> None:
        """Finalize an ROI built by a drag tool: ``shape`` is "rectangle" (x0, y0, x1, y1) or
        "circle" (cx, cy, radius), in image coordinates."""
        command = self.session.set_roi_rectangle if shape == "rectangle" else self.session.set_roi_circle
        self._finish_roi_definition(command, *(snap(v) for v in args))

    def _finish_roi_definition(self, command=None, *args) -> None:
        """Shared completion for every ROI tool: drop the interaction, then commit the shape
        through the session (one recorded command). Without ``command`` (the N-Gon tool), the
        closed polygon in ``state.roi`` is committed."""
        if command is None:
            command, args = self.session.set_roi, ([(snap(x), snap(y)) for x, y in self.state.roi.corners],)
        previous = self._roi_before_definition
        self._roi_before_definition = None
        self.define_roi_action.setChecked(False)
        self.canvas.clear_interaction()
        try:
            command(*args)
        except ValueError as exc:
            self.state.roi = previous
            self.canvas.update()
            self._update_tool_states()
            QMessageBox.warning(self, "Invalid ROI", str(exc))
            return
        self.statusBar().showMessage("ROI complete.", 4000)

    def _cancel_roi_definition(self, silent: bool = False) -> None:
        """Abort an in-progress ROI definition (Esc, pan tool, or starting a new shape).

        No-op unless *we* are mid-definition. ``define_roi_action`` is checked only while an
        ROI is being defined, and ``begin_canvas_interaction`` unchecks it when a plugin takes
        over the canvas — so this never tears down a plugin's interaction. (Esc while a plugin
        is capturing canvas clicks must leave its handler intact.)
        """
        if not self.define_roi_action.isChecked():
            return
        self.canvas.clear_interaction()
        self.define_roi_action.setChecked(False)
        if self.state.roi is None or not self.state.roi.is_complete:
            # Nothing new was committed: keep the ROI that existed before this definition began.
            self.state.roi = self._roi_before_definition
        self._roi_before_definition = None
        self.canvas.update()
        if not silent:
            self.canvas.refresh()
            self._update_tool_states()

    def _clear_roi(self) -> None:
        if not self._confirm_discard_tracking():
            return
        self.canvas.clear_interaction()
        self._roi_before_definition = None
        if self.define_roi_action.isChecked():
            self.define_roi_action.setChecked(False)
        self.session.clear_roi()

    # ---- feature detection ---------------------------------------------
    def _open_corner_dialog(self) -> None:
        dialog = CornerDetectionDialog(self.state.shi_tomasi_params, self)
        if dialog.exec():
            self.state.shi_tomasi_params = dialog.values()
        dialog.deleteLater()

    def _open_grid_dialog(self) -> None:
        dialog = GridDialog(self.state.grid_params, self)
        if dialog.exec():
            self.state.grid_params = dialog.values()
        dialog.deleteLater()

    def _open_tracker_dialog(self) -> None:
        dialog = TrackerDialog(self.state.lk_params, self)
        if dialog.exec():
            self.state.lk_params = dialog.values()
        dialog.deleteLater()

    def _open_display_dialog(self) -> None:
        snapshot = dict(self.state.display_params)

        def _apply(values: dict) -> None:
            self.state.display_params = values
            self.canvas.update()

        dialog = DisplayDialog(snapshot, _apply, self)
        if dialog.exec():
            self.state.display_params = dialog.values()
        else:
            self.state.display_params = snapshot
        dialog.deleteLater()
        self.canvas.update()

    # ---- help -----------------------------------------------------------
    def _open_help(self) -> None:
        help_path = os.path.join(os.path.dirname(__file__), "help.html")
        QDesktopServices.openUrl(QUrl.fromLocalFile(help_path))

    def _open_about(self) -> None:
        QMessageBox.about(
            self,
            "About ECM Tracker",
            "<h3>ECM Tracker</h3>"
            "<p>Version 0.1</p>"
            "<p>Image-feature tracking for frame sequences.</p>"
            "<p>&copy; 2026 Senecell AG</p>"
            "<p>Licensed under the "
            "<a href=\"https://polyformproject.org/licenses/noncommercial/1.0.0/\">"
            "PolyForm Noncommercial License 1.0.0</a>.</p>",
        )

    def _detect_shi_tomasi(self) -> None:
        if not self._roi_ready() or not self._confirm_discard_tracking():
            return
        # Pass every parameter explicitly so a recorded macro replays without saved preferences.
        self._run_detection(self.session.detect_corners, self.state.shi_tomasi_params)

    def _detect_grid(self) -> None:
        if not self._roi_ready() or not self._confirm_discard_tracking():
            return
        self._run_detection(self.session.detect_grid, self.state.grid_params)

    def _run_detection(self, command, params) -> None:
        try:
            n = command(**params)
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            QMessageBox.critical(self, "Detection failed", str(exc))
            return
        self.statusBar().showMessage(f"Detected {n} feature points.", 4000)

    def _roi_ready(self) -> bool:
        return (
            self.state.has_sequence
            and self.state.roi is not None
            and self.state.roi.is_complete
        )

    # ---- tracking -------------------------------------------------------
    def _run_tracking(self) -> None:
        feats = self.state.features
        if feats is None or len(feats) == 0:
            return
        # The session discards a result whose inputs changed during tracking, and its RESULT
        # change closes a Cleanup dialog / undo history bound to the previous result.
        try:
            result = self.session.track(**self.state.lk_params)
        except StaleResultError:
            self.statusBar().showMessage("Tracking inputs changed; discarded the stale result.", 5000)
            return
        except OperationCancelled:
            self.statusBar().showMessage("Tracking cancelled.", 4000)
            return
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            QMessageBox.critical(self, "Tracking failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Tracked {result.n_points} points over {result.n_frames} frames.", 4000
        )

    def _on_clear_tracking_clicked(self) -> None:
        self._confirm_discard_tracking()

    def _confirm_discard_tracking(self) -> bool:
        """True if there is no result, or the user agrees to discard it (which clears it)."""
        if self.state.result is None:
            return True
        resp = QMessageBox.question(
            self,
            "Discard tracking?",
            "This will discard the existing tracking result. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self._clear_tracking()
            return True
        return False

    def _clear_tracking(self) -> None:
        self.session.clear_tracking()

    # ---- cleanup --------------------------------------------------------
    def _open_cleanup(self) -> None:
        if self.state.result is None:
            return
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.raise_()
            self._cleanup_dialog.activateWindow()
            return
        self._cleanup_metrics = self.session.metrics()
        saved = settings.get_section("cleanup")
        defaults = thresholds_from_dict(saved) if saved else None
        dialog = CleanupDialog(self._cleanup_metrics, defaults, self)
        dialog.thresholdsChanged.connect(self._cleanup_preview)
        dialog.applyRequested.connect(self._cleanup_apply)
        dialog.undoRequested.connect(self._cleanup_undo)
        dialog.finished.connect(self._cleanup_closed)
        dialog.finished.connect(dialog.deleteLater)
        self._cleanup_dialog = dialog
        self._cleanup_preview()
        dialog.show()

    def _cleanup_preview(self) -> None:
        if self._cleanup_dialog is None or self.state.active_mask is None:
            return
        keep = build_mask(self._cleanup_metrics, self._cleanup_dialog.thresholds())
        self._preview_keep = keep
        self.canvas.set_preview_mask(keep)
        active = self.state.active_mask
        kept = int((active & keep).sum())
        self._cleanup_dialog.set_counts(kept, int(active.sum()))

    def apply_keep_mask(self, keep) -> None:
        """Filter the active point set by a full-length ``(P,)`` bool keep-mask, undoably.

        Shared by the Cleanup dialog and the plugin API (``PluginContext.apply_keep_mask``):
        snapshots the current mask onto the undo stack, ANDs in ``keep`` (so points only ever
        leave the active set), refreshes, and emits ``mask_changed``. No-op without a result."""
        self.session.apply_keep_mask(keep)

    def _cleanup_apply(self) -> None:
        self._cleanup_preview()  # Apply current controls, never a cached preview.
        if self._preview_keep is None or self._cleanup_dialog is None:
            return
        # Recorded as e.g. session.cleanup(fb_mean=0.5, drop_left_image=True).
        self.session.cleanup(**filters_from_thresholds(self._cleanup_dialog.thresholds()))

    def _cleanup_undo(self) -> None:
        if not self.state.undo_stack:
            return
        self.session.undo_cleanup()

    def _cleanup_closed(self, _result=None) -> None:
        self._cleanup_dialog = None
        self._cleanup_metrics = None
        self._preview_keep = None
        self.canvas.set_preview_mask(None)
        self.canvas.refresh()
        self._update_tool_states()

    # ---- point manager --------------------------------------------------
    def _open_point_manager(self) -> None:
        has_seed = self.state.features is not None and len(self.state.features) > 0
        if not (has_seed or self.state.result is not None):
            return
        if self._point_manager is not None:
            self._point_manager.raise_()
            self._point_manager.activateWindow()
            return
        # Stand down conflicting canvas modes so our selection interaction owns the slot.
        self._deactivate_point_tools()
        if self.pan_tool_action.isChecked():
            self.pan_tool_action.setChecked(False)
        self._cancel_roi_definition()
        dialog = PointManagerDialog(self)
        dialog.deleteRequested.connect(self._point_manager_delete)
        dialog.finished.connect(self._point_manager_closed)
        dialog.finished.connect(dialog.deleteLater)
        # Reactive refresh: rebuild on any change to the point set or result mode.
        self.signals.mask_changed.connect(dialog.rebuild)
        self.signals.result_changed.connect(dialog.rebuild)
        self.signals.sequence_changed.connect(dialog.rebuild)
        self.signals.seeds_changed.connect(dialog.rebuild)
        self._point_manager = dialog
        self.canvas.add_overlay(dialog._paint_selected)
        self.canvas.set_interaction(PointSelectInteraction(dialog))
        dialog.rebuild()
        dialog.show()

    def _point_manager_delete(self) -> None:
        dialog = self._point_manager
        if dialog is None:
            return
        indices = dialog.selected_point_indices()
        if not indices:
            return
        if self.state.result is not None:
            self.session.remove_points(indices)  # emits mask_changed -> dialog.rebuild
        elif self.state.features is not None:
            self.session.remove_seeds(indices)  # emits seeds_changed -> dialog.rebuild

    def _point_manager_closed(self, _result=None) -> None:
        dialog = self._point_manager
        self._point_manager = None
        if dialog is not None:
            for sig in (
                self.signals.mask_changed,
                self.signals.result_changed,
                self.signals.sequence_changed,
                self.signals.seeds_changed,
            ):
                try:
                    sig.disconnect(dialog.rebuild)
                except TypeError:
                    pass
            self.canvas.remove_overlay(dialog._paint_selected)
        # Only release the slot if we still own it (a tool may have reclaimed it while open).
        if isinstance(self.canvas._interaction, PointSelectInteraction):
            self.canvas.clear_interaction()
        self.canvas.update()

    # ---- export ---------------------------------------------------------
    def _export_target(self, default_name: str, caption: str, file_filter: str):
        """Shared export precondition + save dialog. Returns the chosen path, or ``None`` if
        there is nothing to export or the user cancelled."""
        if self.state.result is None or self.state.active_mask is None:
            return None
        export_mask = self._exportable_mask()
        if export_mask is None or not export_mask.any():
            QMessageBox.warning(
                self,
                "Nothing to export",
                "No fully valid tracked points remain to export.",
            )
            return None
        default_path = os.path.join(self.state.source_dir or "", default_name)
        path, _ = QFileDialog.getSaveFileName(self, caption, default_path, file_filter)
        return path or None

    def _exportable_mask(self):
        """Active points that remained valid for every frame in the tracked range."""
        return self.session.exportable_mask()

    def _export(self) -> None:
        path = self._export_target("coords.npy", "Export coordinates", "NumPy array (*.npy)")
        if path is None:
            return
        try:
            coords_path, seq_path, shape = self.session.export_npy(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {shape[1]} points x {shape[0]} frames to "
            f"{os.path.basename(coords_path)} + {os.path.basename(seq_path)}",
            6000,
        )

    def _export_csv(self) -> None:
        path = self._export_target("coords.csv", "Export coordinates as CSV", "CSV file (*.csv)")
        if path is None:
            return
        try:
            csv_path, shape = self.session.export_csv(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {shape[1]} points x {shape[0]} frames to "
            f"{os.path.basename(csv_path)}",
            6000,
        )

    # ---- save / load trackers ------------------------------------------
    def save_trackers_to(self, path) -> str:
        """Write the current seeds + (optional) tracking result to ``path`` (a ``.npz``).

        The dialog-free save path shared by ``File -> Save Trackers`` and the plugin API
        (``PluginContext.save_trackers``). Raises ``ValueError`` if there are no reference
        points; lets ``OSError`` from the write propagate. Returns the written path."""
        return self.session.save_trackers(path)

    def load_trackers_from(self, path, *, expected_range=None):
        """Load a tracker ``.npz`` and overlay it onto the open sequence; return the bundle.

        The dialog-free load path shared by ``File -> Load Trackers`` and the plugin API
        (``PluginContext.load_trackers``). Raises ``ValueError`` if no sequence is open, the
        file can't be read, or its frame count doesn't match the open sequence."""
        if not self.state.has_sequence:
            raise ValueError("Open an image sequence before loading trackers.")
        bundle = self.session.load_trackers(path, expected_range=expected_range)
        s = self.state
        if bundle["has_result"]:
            msg = f"Loaded {s.result.n_points} points, tracked over {s.result.n_frames} frames."
        else:
            msg = f"Loaded {len(s.features)} reference points."
        if bundle.get("sequence_fingerprint") is None:
            msg += " Legacy v1 file: sequence identity could not be verified."
        self.statusBar().showMessage(msg, 6000)
        return bundle

    def _save_trackers(self) -> None:
        """Save the reference seed points and (if tracked) the full result to one .npz."""
        s = self.state
        if not s.has_sequence or s.features is None or len(s.features) == 0:
            QMessageBox.warning(self, "Nothing to save", "Seed some reference points first.")
            return
        default_path = os.path.join(s.source_dir or "", "trackers.npz")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Trackers", default_path, "Tracker file (*.npz)"
        )
        if not path:
            return
        try:
            self.save_trackers_to(path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        where = "with tracking" if s.result is not None else "reference points only"
        self.statusBar().showMessage(
            f"Saved trackers ({where}) to {os.path.basename(path)}", 6000
        )

    def _load_trackers(self) -> None:
        """Overlay a saved tracker file onto the open sequence (frame counts must match)."""
        if not self.state.has_sequence:
            QMessageBox.warning(
                self, "No sequence", "Open the image folder first, then load trackers onto it."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trackers", self.state.source_dir or "", "Tracker file (*.npz)"
        )
        if not path:
            return
        if self.state.features is not None or self.state.result is not None:
            resp = QMessageBox.question(
                self,
                "Replace current trackers?",
                "This will replace the current points and tracking with the loaded file. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        try:
            self.load_trackers_from(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load failed", str(exc))

    # ---- tool enablement ------------------------------------------------
    def _update_tool_states(self) -> None:
        if self.define_roi_action.isChecked() and not self.state.on_reference_frame:
            # Leaving the reference frame mid-definition: discard the partial ROI too, or the
            # half-placed N-Gon polyline is orphaned (kept painting, detection silently blocked
            # until the user Clears). silent=True avoids recursing back into _update_tool_states.
            # Done first so the enable states below reflect the restored ROI.
            self._cancel_roi_definition(silent=True)
        has = self.state.has_sequence
        is_pair = isinstance(self.state.sequence, AlignedImagePairSequence)
        self.save_pair_setup_action.setEnabled(is_pair)
        self.adjust_pair_action.setEnabled(is_pair)
        on_ref = self.state.on_reference_frame
        roi_ready = self._roi_ready()
        has_features = self.state.features is not None and len(self.state.features) > 0
        has_result = self.state.result is not None

        self.define_roi_action.setEnabled(has and on_ref)
        self.clear_roi_action.setEnabled(has and self.state.roi is not None)
        self.detect_corners_action.setEnabled(roi_ready)
        self.detect_grid_action.setEnabled(roi_ready)
        # Add edits seed points (reference frame, pre-track); Delete also removes individual tracked
        # points once a result exists (current frame must be in the tracked range).
        pre_track_edit = has and on_ref and not has_result
        post_track_delete = has_result and self.state.current_in_range
        self.add_points_action.setEnabled(pre_track_edit)
        self.delete_points_action.setEnabled(pre_track_edit or post_track_delete)
        self.point_manager_action.setEnabled(has_features or has_result)
        for act in (self.add_points_action, self.delete_points_action):
            if act.isChecked() and not act.isEnabled():
                self._deactivate_point_tool(act)
        self.run_tracking_action.setEnabled(has_features)
        self.clear_tracking_action.setEnabled(has_result)
        self.cleanup_action.setEnabled(has_result)
        export_mask = self._exportable_mask() if has_result else None
        can_export = export_mask is not None and bool(export_mask.any())
        self.export_action.setEnabled(can_export)
        self.export_toolbar_action.setEnabled(can_export)
        # Save needs seed points; Load overlays onto any open sequence.
        self.save_trackers_action.setEnabled(has_features)
        self.load_trackers_action.setEnabled(has)

        # The result is tied to a fixed reference..last range; lock those sliders until the
        # user explicitly discards the tracking (current stays free for browsing frames).
        self.current_slider.setEnabled(has)
        self.reference_slider.setEnabled(has and not has_result)
        self.last_slider.setEnabled(has and not has_result)

        self.go_reference_action.setEnabled(has)
        self.go_last_action.setEnabled(has)
        self.prev_frame_action.setEnabled(has and self.state.current_index > 0)
        self.next_frame_action.setEnabled(
            has and self.state.current_index < self.state.total_images - 1
        )

    # ---- status ---------------------------------------------------------
    def _update_status(self) -> None:
        if not self.state.has_sequence:
            self._status_label.setText("No sequence loaded — File → Open Directory or Open Image Pair.")
            return
        s = self.state
        if s.current_in_range:
            cut = f"cut {s.global_to_cut(s.current_index)}"
        else:
            cut = "out of range"
        pair_status = ""
        if isinstance(s.sequence, AlignedImagePairSequence):
            dx, dy = s.sequence.translation
            # Dimensions are already validated/cached; do not reread source files just for status.
            h, w = s.sequence.frame_shape[:2]
            alignment = s.sequence.alignment
            mode = {"translation": "translation", "scale": "uniform scale", "affine": "affine"}[alignment.mode]
            rotation = f" · {alignment.angle_degrees:+.2f}°" if alignment.rotation_enabled else ""
            pair_status = (f"  |  Pair {mode} ({dx:+g}, {dy:+g}) px{rotation} · "
                           f"{w} × {h} px · aligned coordinates")
        self._status_label.setText(
            f"Frame {s.current_index + 1}/{s.total_images} ({cut})  |  "
            f"reference {s.reference_index}  last {s.last_index}  "
            f"({s.n_cut} frames in range){pair_status}"
        )
