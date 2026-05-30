import os

import numpy as np
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QShortcut,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core import settings
from app.gui.icon_loader import ACCENT, load_icon
from app.core.cleanup import build_mask, compute_metrics, thresholds_from_dict
from app.core.export import export, export_csv
from app.core.feature_detection import regular_grid, shi_tomasi
from app.core.image_sequence import ImageSequence, discover
from app.core.roi import ROI
from app.core.tracking import track
from app.gui.canvas_view import CanvasView
from app.gui.roi_tools import CircleTool, NGonTool, RectangleTool
from app.gui.cleanup_dialog import CleanupDialog
from app.gui.dialogs import CornerDetectionDialog, DisplayDialog, GridDialog, TrackerDialog
from app.models.project_state import ProjectState
from app.plugins.api import PluginSignals
from app.plugins.manager import PluginManager


class LabeledSlider(QWidget):
    """A label + horizontal slider + spinbox kept in sync.

    `valueChanged` fires only on user interaction; `setValue`/`setRange` are programmatic and
    do not emit, which avoids reentrant update loops between interdependent sliders.
    """

    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = QLabel(label)
        self._label.setMinimumWidth(90)
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QSpinBox()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)

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
        self.resize(1100, 800)

        self.state = ProjectState()
        self.canvas = CanvasView(self.state)
        self.signals = PluginSignals()  # state-change hub broadcast to plugins
        self._status_label = QLabel()
        self.statusBar().addPermanentWidget(self._status_label)

        self._cleanup_dialog = None
        self._cleanup_metrics = None
        self._preview_keep = None

        self.current_slider = LabeledSlider("Current")
        self.reference_slider = LabeledSlider("Reference")
        self.last_slider = LabeledSlider("Last")
        self.current_slider.valueChanged.connect(self._on_current_changed)
        self.reference_slider.valueChanged.connect(self._on_reference_changed)
        self.last_slider.valueChanged.connect(self._on_last_changed)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 4, 8, 8)
        controls_layout.addWidget(self.current_slider)
        controls_layout.addWidget(self.reference_slider)
        controls_layout.addWidget(self.last_slider)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(controls)
        self.setCentralWidget(container)

        self._build_menus()
        self._build_toolbar()

        # Esc cancels an in-progress ROI definition (the menu-only Define button has no
        # toggle-off affordance).
        self._roi_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._roi_escape.activated.connect(self._cancel_roi_definition)

        # ←/→ step the Current frame, scoped to the canvas (WidgetWithChildrenShortcut) so
        # they don't hijack arrow keys from the sliders/spin boxes or from plugin windows
        # (a window-wide shortcut on these navigation keys conflicts widely and is unstable).
        self._prev_frame_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self.canvas)
        self._prev_frame_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._prev_frame_shortcut.activated.connect(
            lambda: self._go_to_frame(self.state.current_index - 1)
        )
        self._next_frame_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self.canvas)
        self._next_frame_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_frame_shortcut.activated.connect(
            lambda: self._go_to_frame(self.state.current_index + 1)
        )

        # Discover installed plugins and populate the Plugins menu.
        self.plugin_manager = PluginManager(self)
        self.plugin_manager.discover()
        self.plugin_manager.build_menu(self._plugins_menu)

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
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setAutoRaise(True)
            if action is primary:
                button.setObjectName("primaryAction")
            if menus and action in menus:
                button.setMenu(menus[action])
                button.setPopupMode(QToolButton.InstantPopup)
                button.setObjectName("menuButton")
            row.addWidget(button)
        outer.addLayout(row)

        caption = QLabel(title)
        caption.setObjectName("toolGroupCaption")
        caption.setAlignment(Qt.AlignHCenter)
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

    def _load_paths(self, paths, source_dir) -> None:
        if not paths:
            QMessageBox.warning(self, "No images", "No supported images were found.")
            return
        try:
            sequence = ImageSequence(paths)
            self.state.load_sequence(sequence, source_dir)
            sequence.load_bgr(0)  # surface decode errors early
        except (IOError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._configure_sliders()
        self.canvas.reset_view()
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()
        self.signals.sequence_changed.emit()

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
        self.state.set_current(value)
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()
        self.signals.frame_changed.emit(self.state.current_index)

    def _on_reference_changed(self, value: int) -> None:
        self.state.set_reference(value)
        self._sync_range_constraints()
        roi_cleared = self.state.roi is not None
        if roi_cleared:
            self.state.roi = None
            self.state.features = None
            self.define_roi_action.setChecked(False)
            self.statusBar().showMessage("ROI cleared (reference frame changed).", 4000)
        self.canvas.refresh()
        self._update_status()
        self._update_tool_states()
        if roi_cleared:
            self.signals.roi_changed.emit()

    def _on_last_changed(self, value: int) -> None:
        self.state.set_last(value)
        self._sync_range_constraints()
        self._update_status()
        self._update_tool_states()

    def _go_to_frame(self, global_index: int) -> None:
        """Move Current to ``global_index`` (clamped), driving the same refresh
        path as the Current slider."""
        if not self.state.has_sequence:
            return
        target = max(0, min(global_index, self.state.total_images - 1))
        self.current_slider.setValue(target)  # sync widget (setValue blocks signals)
        self._on_current_changed(target)      # state + canvas + status + tools + signal

    # ---- ROI ------------------------------------------------------------
    def _on_pan_tool_toggled(self, checked: bool) -> None:
        self.canvas.set_pan_tool(checked)
        if checked:
            self._cancel_roi_definition()

    def _begin_roi_definition(self, shape: str) -> None:
        """Start defining an ROI of the given shape on the reference frame."""
        if not (self.state.has_sequence and self.state.on_reference_frame):
            return
        if self.pan_tool_action.isChecked():
            self.pan_tool_action.setChecked(False)
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
        if self.define_roi_action.isChecked():
            self.define_roi_action.setChecked(False)
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.roi_changed.emit()

    # ---- feature detection ---------------------------------------------
    def _open_corner_dialog(self) -> None:
        dialog = CornerDetectionDialog(self.state.shi_tomasi_params, self)
        if dialog.exec_():
            self.state.shi_tomasi_params = dialog.values()

    def _open_grid_dialog(self) -> None:
        dialog = GridDialog(self.state.grid_params, self)
        if dialog.exec_():
            self.state.grid_params = dialog.values()

    def _open_tracker_dialog(self) -> None:
        dialog = TrackerDialog(self.state.lk_params, self)
        if dialog.exec_():
            self.state.lk_params = dialog.values()

    def _open_display_dialog(self) -> None:
        snapshot = dict(self.state.display_params)

        def _apply(values: dict) -> None:
            self.state.display_params = values
            self.canvas.update()

        dialog = DisplayDialog(snapshot, _apply, self)
        if dialog.exec_():
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
            "<p>Image-feature tracking for frame sequences.</p>"
            "<p>&copy; 2026 Senecell AG</p>",
        )

    def _detect_shi_tomasi(self) -> None:
        if not self._roi_ready() or not self._confirm_discard_tracking():
            return
        gray = self.state.sequence.load_gray(self.state.reference_index)
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
        n = self.state.n_cut
        total = max(1, 2 * (n - 1))
        dialog = QProgressDialog("Tracking...", "Cancel", 0, total, self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        def progress(done, _total):
            dialog.setValue(done)
            QApplication.processEvents()
            return dialog.wasCanceled()

        try:
            result = track(
                self.state.sequence,
                self.state.reference_index,
                self.state.last_index,
                feats,
                self.state.lk_params,
                progress,
            )
        finally:
            dialog.close()

        if result is None:
            self.statusBar().showMessage("Tracking cancelled.", 4000)
            return
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
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp == QMessageBox.Yes:
            self._clear_tracking()
            return True
        return False

    def _clear_tracking(self) -> None:
        if self._cleanup_dialog is not None:
            self._cleanup_dialog.close()
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
        if self._cleanup_dialog is None:
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
        self.state.undo_stack.append(self.state.active_mask.copy())
        self.state.active_mask = self.state.active_mask & keep
        self.canvas.refresh()
        self._update_tool_states()
        self.signals.mask_changed.emit()

    def _cleanup_apply(self) -> None:
        if self._preview_keep is None:
            return
        self.apply_keep_mask(self._preview_keep)
        self._cleanup_preview()

    def _cleanup_undo(self) -> None:
        if not self.state.undo_stack:
            return
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

    # ---- export ---------------------------------------------------------
    def _export_target(self, default_name: str, caption: str, file_filter: str):
        """Shared export precondition + save dialog. Returns ``(out_dir, filename)``, or ``None``
        if there is nothing to export or the user cancelled."""
        if self.state.result is None or self.state.active_mask is None:
            return None
        if not self.state.active_mask.any():
            QMessageBox.warning(self, "Nothing to export", "No points remain to export.")
            return None
        default_path = os.path.join(self.state.source_dir or "", default_name)
        path, _ = QFileDialog.getSaveFileName(self, caption, default_path, file_filter)
        if not path:
            return None
        return os.path.dirname(path), os.path.basename(path)

    def _export(self) -> None:
        target = self._export_target("coords.npy", "Export coordinates", "NumPy array (*.npy)")
        if target is None:
            return
        out_dir, filename = target
        result = self.state.result
        coords_path, seq_path, shape = export(
            result.coords_fw,
            self.state.active_mask,
            result.reference_index,
            result.last_index,
            out_dir,
            filename,
        )
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
        frame_names = [
            os.path.basename(self.state.sequence.paths[g])
            for g in range(result.reference_index, result.last_index + 1)
        ]
        csv_path, shape = export_csv(
            result.coords_fw,
            self.state.active_mask,
            frame_names,
            out_dir,
            filename,
        )
        self.statusBar().showMessage(
            f"Exported {shape[1]} points x {shape[0]} frames to "
            f"{os.path.basename(csv_path)}",
            6000,
        )

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
        self.run_tracking_action.setEnabled(has_features)
        self.clear_tracking_action.setEnabled(has_result)
        self.cleanup_action.setEnabled(has_result)
        can_export = has_result and bool(self.state.active_mask.any())
        self.export_action.setEnabled(can_export)
        self.export_toolbar_action.setEnabled(can_export)

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
