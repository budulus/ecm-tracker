import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
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

from app.core import settings, tracker_io
from app.gui.icon_loader import ACCENT, load_icon
from app.core.cleanup import build_mask, compute_metrics, thresholds_from_dict
from app.core.export import export, export_csv
from app.core.feature_detection import regular_grid, shi_tomasi
from app.core.image_sequence import ImageSequence, discover
from app.core.roi import ROI
from app.core.tracking import track
from app.gui.canvas_view import CanvasView
from app.gui.point_tools import AddPointsTool, DeletePointsTool
from app.gui.point_manager import PointManagerDialog, PointSelectInteraction
from app.gui.roi_tools import CircleTool, NGonTool, RectangleTool
from app.gui.cleanup_dialog import CleanupDialog
from app.gui.dialogs import CornerDetectionDialog, DisplayDialog, GridDialog, TrackerDialog
from app.models.project_state import ProjectState, validate_grid, validate_lk, validate_shi_tomasi
from app.models.tracker_result import TrackerResult
from app.plugins.api import PluginSignals
from app.plugins.manager import PluginManager


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

        self._export_cache = None
        self._loading_sequence = False
        self.state = ProjectState()
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

        # Populated by the PluginManager after construction.
        self._plugins_menu = self.menuBar().addMenu("&Plugins")

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
        if directory:
            self._load_paths(discover(directory), directory)

    def _open_files(self) -> None:
        start = self.state.source_dir or ""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Images",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if files:
            self._load_paths(discover(files), os.path.dirname(files[0]))

    def load_sequence_from_paths(self, paths, source_dir) -> bool:
        """Public entry to load an explicit, pre-ordered image-path list.

        Order is preserved verbatim (``ImageSequence`` does not re-sort). Resets all downstream
        state and emits ``sequence_changed``, exactly like File -> Open. Used by loader plugins
        that order frames by an acquisition log rather than by filename."""
        return self._load_paths(paths, source_dir)

    def _load_paths(self, paths, source_dir) -> bool:
        if self._loading_sequence:
            raise RuntimeError("An image sequence is already being validated.")
        if not paths:
            QMessageBox.warning(self, "No images", "No supported images were found.")
            return False
        try:
            sequence = ImageSequence(paths)
            dialog = QProgressDialog("Validating images...", "Cancel", 0, len(sequence), self)
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            dialog.setMinimumDuration(500)

            def progress(done, _total):
                dialog.setValue(done)
                QApplication.processEvents()
                return dialog.wasCanceled()

            self._loading_sequence = True
            try:
                valid = sequence.validate_all(progress)
            finally:
                self._loading_sequence = False
                dialog.close()
            if not valid:
                self.statusBar().showMessage("Image loading cancelled.", 4000)
                return False
        except (IOError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return False
        # Commit only after every frame has decoded and passed the common-shape check. A failed
        # candidate therefore leaves the existing project completely untouched.
        self._teardown_session_ui()
        self.state.load_sequence(sequence, source_dir)
        self._configure_sliders()
        self.canvas.reset_view()
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()
        self._update_window_title()
        self.signals.sequence_changed.emit()
        return True

    def _update_window_title(self) -> None:
        """Reflect the loaded folder in the title bar (e.g. 'ECM Tracker - experiment_1')."""
        name = os.path.basename(self.state.source_dir) if self.state.source_dir else None
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
        self._export_cache = None
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
        self.state.set_current(value)
        try:
            self.canvas.refresh()
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            self.state.set_current(previous)
            self.current_slider.setValue(previous)
            QMessageBox.critical(self, "Frame load failed", str(exc))
            return
        self._update_status()
        self._update_tool_states()
        self.signals.frame_changed.emit(self.state.current_index)

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
        s = self.state
        if (reference, last) == (s.reference_index, s.last_index):
            return
        ref_changed = reference != s.reference_index
        had_result = s.result is not None
        self._teardown_session_ui()
        s.reference_index, s.last_index = reference, last
        s.result = s.active_mask = None
        s.undo_stack = []
        if ref_changed:
            s.roi = s.features = None
        s.touch()
        self._configure_sliders()
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()
        self.signals.range_changed.emit()
        if ref_changed:
            self.signals.roi_changed.emit()
            self.signals.seeds_changed.emit()
        if had_result:
            self.signals.result_changed.emit()

    def _go_to_frame(self, global_index: int) -> None:
        """Move Current to ``global_index`` (clamped), driving the same refresh
        path as the Current slider."""
        if not self.state.has_sequence:
            return
        target = max(0, min(global_index, self.state.total_images - 1))
        self.current_slider.setValue(target)  # sync widget (setValue blocks signals)
        self._on_current_changed(target)      # state + canvas + status + tools + signal

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

    def _commit_interactive_roi(self, corners) -> None:
        """Finalize an ROI built by a drag tool (Rectangle / Circle)."""
        self.state.roi = ROI(corners)
        self._finish_roi_definition()

    def _finish_roi_definition(self) -> None:
        """Shared completion for every ROI tool: drop the interaction and notify. The tool has
        already populated ``state.roi`` (drag tools build it fresh; the N-Gon tool closes it)."""
        self.define_roi_action.setChecked(False)
        self.canvas.clear_interaction()
        self.statusBar().showMessage("ROI complete.", 4000)
        self.state.touch()
        self.signals.roi_changed.emit()
        self.canvas.refresh()
        self._update_tool_states()

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
        if self.state.roi is not None and not self.state.roi.is_complete:
            self.state.roi = None
        if not silent:
            self.canvas.refresh()
            self._update_tool_states()

    def _clear_roi(self) -> None:
        if not self._confirm_discard_tracking():
            return
        self.canvas.clear_interaction()
        self.state.roi = None
        self.state.features = None
        self.state.touch()
        self.signals.seeds_changed.emit()
        if self.define_roi_action.isChecked():
            self.define_roi_action.setChecked(False)
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.roi_changed.emit()

    # ---- feature detection ---------------------------------------------
    def _open_corner_dialog(self) -> None:
        dialog = CornerDetectionDialog(self.state.shi_tomasi_params, self)
        if dialog.exec():
            self.state.shi_tomasi_params = dialog.values()

    def _open_grid_dialog(self) -> None:
        dialog = GridDialog(self.state.grid_params, self)
        if dialog.exec():
            self.state.grid_params = dialog.values()

    def _open_tracker_dialog(self) -> None:
        dialog = TrackerDialog(self.state.lk_params, self)
        if dialog.exec():
            self.state.lk_params = dialog.values()

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
        try:
            gray = self.state.sequence.load_gray(self.state.reference_index)
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            QMessageBox.critical(self, "Detection failed", str(exc))
            return
        h, w = gray.shape[:2]
        mask = self.state.roi.mask(h, w)
        self.state.features = shi_tomasi(gray, mask, self.state.shi_tomasi_params)
        self._after_detection()

    def _detect_grid(self) -> None:
        if not self._roi_ready() or not self._confirm_discard_tracking():
            return
        self.state.features = regular_grid(self.state.roi, **self.state.grid_params)
        self._after_detection()

    def _after_detection(self) -> None:
        self.state.touch()
        self.signals.seeds_changed.emit()
        n = 0 if self.state.features is None else len(self.state.features)
        self.statusBar().showMessage(f"Detected {n} feature points.", 4000)
        self.canvas.refresh()
        self._update_tool_states()

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
        revision = self.state.revision
        sequence = self.state.sequence
        reference, last = self.state.reference_index, self.state.last_index
        feats = feats.copy()
        params = dict(self.state.lk_params)
        n = last - reference + 1
        total = max(1, 2 * (n - 1))
        dialog = QProgressDialog("Tracking...", "Cancel", 0, total, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        def progress(done, _total):
            dialog.setValue(done)
            QApplication.processEvents()
            return dialog.wasCanceled()

        try:
            result = track(
                sequence,
                reference,
                last,
                feats,
                params,
                progress,
            )
        except (OSError, ValueError, IndexError, cv2.error) as exc:
            QMessageBox.critical(self, "Tracking failed", str(exc))
            return
        finally:
            dialog.close()

        if self.state.revision != revision:
            self.statusBar().showMessage("Tracking inputs changed; discarded the stale result.", 5000)
            return
        if result is None:
            self.statusBar().showMessage("Tracking cancelled.", 4000)
            return
        # Discard any derived state tied to a previous result before installing the new one: an
        # open Cleanup dialog is bound to the old metrics and a leftover undo stack holds old-mask
        # snapshots, either of which would otherwise be applied against the new result.
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.close()
        self.state.undo_stack = []
        self.canvas.set_preview_mask(None)
        self._preview_keep = None
        self.state.touch()
        self.state.result = result
        self.state.active_mask = np.ones(result.n_points, dtype=bool)
        self.statusBar().showMessage(
            f"Tracked {result.n_points} points over {result.n_frames} frames.", 4000
        )
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.result_changed.emit()

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
        self._export_cache = None
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.close()
        self.state.touch()
        self.state.result = None
        self.state.active_mask = None
        self.state.undo_stack = []
        self.canvas.set_preview_mask(None)
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.result_changed.emit()

    # ---- cleanup --------------------------------------------------------
    def _open_cleanup(self) -> None:
        if self.state.result is None:
            return
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.raise_()
            self._cleanup_dialog.activateWindow()
            return
        h, w = self.state.image_size()
        self._cleanup_metrics = compute_metrics(self.state.result, self.state.roi, (h, w))
        saved = settings.get_section("cleanup")
        defaults = thresholds_from_dict(saved) if saved else None
        dialog = CleanupDialog(self._cleanup_metrics, defaults, self)
        dialog.thresholdsChanged.connect(self._cleanup_preview)
        dialog.applyRequested.connect(self._cleanup_apply)
        dialog.undoRequested.connect(self._cleanup_undo)
        dialog.finished.connect(self._cleanup_closed)
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
        if self.state.active_mask is None:
            return
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != self.state.active_mask.shape:
            raise ValueError(
                f"keep mask shape {keep.shape} does not match active mask "
                f"shape {self.state.active_mask.shape}"
            )
        updated = self.state.active_mask & keep
        if np.array_equal(updated, self.state.active_mask):
            return
        self.state.undo_stack.append(self.state.active_mask.copy())
        self.state.touch()
        self.state.active_mask = updated
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.mask_changed.emit()

    def _cleanup_apply(self) -> None:
        self._cleanup_preview()  # Apply current controls, never a cached preview.
        if self._preview_keep is None:
            return
        self.apply_keep_mask(self._preview_keep)
        self._cleanup_preview()

    def _cleanup_undo(self) -> None:
        if not self.state.undo_stack:
            return
        self.state.touch()
        self.state.active_mask = self.state.undo_stack.pop()
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.mask_changed.emit()
        self._cleanup_preview()

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
            keep = np.ones(self.state.result.n_points, dtype=bool)
            keep[indices] = False
            self.apply_keep_mask(keep)  # emits mask_changed -> dialog.rebuild
        else:
            feats = self.state.features
            if feats is None:
                return
            feats = np.delete(feats, indices, axis=0)
            self.state.features = feats if len(feats) else None  # match detection's empty convention
            self.canvas.update()
            self._update_tool_states()
            self.state.touch()
            self.signals.seeds_changed.emit()

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
        """Shared export precondition + save dialog. Returns ``(out_dir, filename)``, or ``None``
        if there is nothing to export or the user cancelled."""
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
        if not path:
            return None
        return os.path.dirname(path), os.path.basename(path)

    def _exportable_mask(self):
        """Active points that remained valid for every frame in the tracked range."""
        result = self.state.result
        active = self.state.active_mask
        if result is None or active is None:
            return None
        if self._export_cache is None or self._export_cache[0] is not result:
            self._export_cache = (result, np.all(result.status_fw == 1, axis=0))
        return active & self._export_cache[1]

    def _export(self) -> None:
        target = self._export_target("coords.npy", "Export coordinates", "NumPy array (*.npy)")
        if target is None:
            return
        out_dir, filename = target
        result = self.state.result
        export_mask = self._exportable_mask()
        try:
            coords_path, seq_path, shape = export(
                result.coords_fw,
                export_mask,
                result.reference_index,
                result.last_index,
                out_dir,
                filename,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {shape[1]} points x {shape[0]} frames to "
            f"{os.path.basename(coords_path)} + sequence.txt",
            6000,
        )

    def _export_csv(self) -> None:
        target = self._export_target(
            "coords.csv", "Export coordinates as CSV", "CSV file (*.csv)"
        )
        if target is None:
            return
        out_dir, filename = target
        result = self.state.result
        export_mask = self._exportable_mask()
        frame_names = [
            os.path.basename(self.state.sequence.paths[g])
            for g in range(result.reference_index, result.last_index + 1)
        ]
        try:
            csv_path, shape = export_csv(
                result.coords_fw,
                export_mask,
                frame_names,
                out_dir,
                filename,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {shape[1]} points x {shape[0]} frames to "
            f"{os.path.basename(csv_path)}",
            6000,
        )

    # ---- save / load trackers ------------------------------------------
    def save_trackers_to(self, path) -> None:
        """Write the current seeds + (optional) tracking result to ``path`` (a ``.npz``).

        The dialog-free save path shared by ``File -> Save Trackers`` and the plugin API
        (``PluginContext.save_trackers``). Raises ``ValueError`` if there are no reference
        points; lets ``OSError`` from the write propagate."""
        s = self.state
        if s.features is None or len(s.features) == 0:
            raise ValueError("No reference points to save.")
        result = s.result
        result_arrays = (
            {
                "coords_fw": result.coords_fw,
                "status_fw": result.status_fw,
                "err_fw": result.err_fw,
                "coords_bw": result.coords_bw,
                "status_bw": result.status_bw,
                "err_bw": result.err_bw,
                "fb_mean_error": result.fb_mean_error,
                "fb_max_error": result.fb_max_error,
            }
            if result is not None
            else None
        )
        tracker_io.save_trackers(
            path,
            total_images=s.total_images,
            reference_index=s.reference_index,
            last_index=s.last_index,
            current_index=s.current_index,
            features=s.features,
            roi_corners=(s.roi.corners if s.roi is not None and s.roi.is_complete else None),
            active_mask=s.active_mask if result is not None else None,
            result_arrays=result_arrays,
            win_size=(result.win_size if result is not None else s.lk_params.get("win_size", 0)),
            lk_params=s.lk_params,
            shi_tomasi_params=s.shi_tomasi_params,
            grid_params=s.grid_params,
            sequence_fingerprint=s.sequence.fingerprint,
            error_kind=result.error_kind if result is not None else "unknown",
            tracking_params=dict(result.tracking_params) if result is not None else {},
        )

    def load_trackers_from(self, path, *, expected_range=None):
        """Load a tracker ``.npz`` and overlay it onto the open sequence; return the bundle.

        The dialog-free load path shared by ``File -> Load Trackers`` and the plugin API
        (``PluginContext.load_trackers``). Raises ``ValueError`` if no sequence is open, the
        file can't be read, or its frame count doesn't match the open sequence."""
        if not self.state.has_sequence:
            raise ValueError("Open an image sequence before loading trackers.")
        bundle = tracker_io.load_trackers(path)  # raises ValueError on a bad/foreign file
        if bundle["total_images"] != self.state.total_images:
            raise ValueError(
                f"This tracker file is for a {bundle['total_images']}-frame sequence, but the "
                f"open sequence has {self.state.total_images} frames."
            )
        expected = bundle.get("sequence_fingerprint")
        actual = self.state.sequence.fingerprint
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(
                "This tracker file belongs to a different image sequence. The frame count "
                "matches, but the ordered image-content fingerprint does not."
            )
        if expected_range is not None and tuple(expected_range) != (bundle["reference_index"], bundle["last_index"]):
            raise ValueError("Saved tracker range does not match the expected project reference.")
        self._apply_loaded_trackers(bundle)
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

    def _apply_loaded_trackers(self, bundle) -> None:
        """Install a loaded tracker bundle onto the open sequence (overlay model).

        Mirrors the teardown in ``_run_tracking``/``_clear_tracking``, then restores state and
        drives the same refresh path as a fresh load. Safe because ``LabeledSlider.setValue``
        (used by ``_configure_sliders``) blocks signals, so restoring a non-zero reference index
        does not re-enter ``_on_reference_changed`` and wipe the ROI we just restored.
        """
        s = self.state
        candidate = (TrackerResult(reference_index=bundle["reference_index"],
                                   last_index=bundle["last_index"], win_size=bundle["win_size"],
                                   error_kind=bundle.get("error_kind", "unknown"),
                                   tracking_params=bundle.get("tracking_params", {}),
                                   **bundle["result_arrays"]) if bundle["has_result"] else None)
        self._teardown_session_ui()
        s.touch()
        # Tear down UI bound to any previous result (cached metrics / stale selections).
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.close()
        if self._point_manager is not None:
            self._point_manager.close()
        s.undo_stack = []
        self._preview_keep = None
        self.canvas.set_preview_mask(None)

        # tracker_io has already validated the complete frame-range invariant.
        s.reference_index = bundle["reference_index"]
        s.last_index = bundle["last_index"]
        s.set_current(bundle["current_index"])

        # ROI (toggle off, mirroring _finish_roi_definition).
        s.roi = ROI(bundle["roi_corners"]) if bundle["roi_corners"] is not None else None
        self.define_roi_action.setChecked(False)

        # Seed points + (optional) tracking result + active mask.
        s.features = bundle["features"]
        if bundle["has_result"]:
            s.result = candidate
            s.active_mask = bundle["active_mask"]
        else:
            s.result = None
            s.active_mask = None

        # Parameters that produced this session (merged over built-ins, like ProjectState).
        s.lk_params = validate_lk(bundle["lk_params"])
        s.shi_tomasi_params = validate_shi_tomasi(bundle["shi_tomasi_params"])
        s.grid_params = validate_grid(bundle["grid_params"])

        self._configure_sliders()  # safe: LabeledSlider.setValue blocks signals
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()

        # Refresh reactive consumers (plugins, point manager) atomically after the full swap.
        self.signals.range_changed.emit()
        self.signals.seeds_changed.emit()
        self.signals.roi_changed.emit()
        self.signals.result_changed.emit()
        self.signals.mask_changed.emit()
        self.signals.frame_changed.emit(s.current_index)

        if bundle["has_result"]:
            msg = f"Loaded {s.result.n_points} points, tracked over {s.result.n_frames} frames."
        else:
            msg = f"Loaded {len(s.features)} reference points."
        if bundle.get("sequence_fingerprint") is None:
            msg += " Legacy v1 file: sequence identity could not be verified."
        self.statusBar().showMessage(msg, 6000)

    # ---- tool enablement ------------------------------------------------
    def _update_tool_states(self) -> None:
        has = self.state.has_sequence
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

        if self.define_roi_action.isChecked() and not on_ref:
            # Leaving the reference frame mid-definition: discard the partial ROI too, or the
            # half-placed N-Gon polyline is orphaned (kept painting, detection silently blocked
            # until the user Clears). silent=True avoids recursing back into _update_tool_states.
            self._cancel_roi_definition(silent=True)

    # ---- status ---------------------------------------------------------
    def _update_status(self) -> None:
        if not self.state.has_sequence:
            self._status_label.setText("No sequence loaded — File → Open Directory.")
            return
        s = self.state
        if s.current_in_range:
            cut = f"cut {s.global_to_cut(s.current_index)}"
        else:
            cut = "out of range"
        self._status_label.setText(
            f"Frame {s.current_index + 1}/{s.total_images} ({cut})  |  "
            f"reference {s.reference_index}  last {s.last_index}  "
            f"({s.n_cut} frames in range)"
        )
