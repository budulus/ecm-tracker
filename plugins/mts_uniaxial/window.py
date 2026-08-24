"""The MTS Uniaxial workflow window.

A top-to-bottom column of sections, one per step of the dependency chain (Load -> Channels &
offset -> Crop -> Find reference -> Handoff -> Export). Each section is disabled until its
upstream step is satisfied, and editing any step conservatively wipes everything downstream (in
memory and on disk) so the user can never reach an inconsistent state. Intermediate state is saved
after each step, so closing and reopening resumes the work.

This module owns the only :class:`MtsProjectState`; it drives the core app through ``ctx``
(``load_sequence``, ``set_reference_frame``, ``set_last_frame``) and reads tracked coordinates
back through ``ctx`` after the user tracks in the main window.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QBrush, QColor, QDoubleValidator, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import kinematics, parsers, project_io, sync
from .crop_plot import CropPlotWidget, ReferencePlotWidget, matplotlib_available
from .reference_algorithms import PREFORCE, REGISTRY
from .state import STEP_ARTIFACTS, MtsProjectState, Step

# Per-frame derived-measures columns offered in the export panel: (key, header, always_on).
# "always_on" columns (the time/frame anchors) have no checkbox and are written unconditionally.
_MEASURE_COLUMNS = [
    ("frame_global", "frame_global", True),
    ("time_s", "time_s", True),
    ("lambda_1", "lambda_1", False),
    ("lambda_2", "lambda_2", False),
    ("eps_1", "eps_1", False),
    ("eps_2", "eps_2", False),
    ("eps_2_ico", "eps_2_ico", False),
    ("pk_stress_MPa", "pk_stress_MPa", False),
    ("cauchy_stress_MPa", "cauchy_stress_MPa", False),
    ("force_N", "force_N", False),
    ("displacement_mm", "displacement_mm", False),
    ("angle_deg", "angle_deg", False),
    ("n_points", "n_points", False),
    ("in_range", "in_range", False),
]
_MEASURE_INT_KEYS = {"frame_global", "n_points", "in_range"}

# Every artifact filename, for clearing a project folder before a fresh load.
_ALL_FILES = [f for files in STEP_ARTIFACTS.values() for f in files]

_DEFAULT_IMAGES_SUBDIR = "veddac"
_DEFAULT_SENSOR_REL = "mts/specimen.dat"
_LOG_NAME = "VDCCam.log"

_OUTLIER_COLOR = QColor(255, 60, 60)  # red — RANSAC outlier preview on the main canvas


class MtsUniaxialWindow(QWidget):
    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self.pstate = MtsProjectState()
        self._syncing = False     # True while pushing state -> widgets (suppresses handlers)
        self._loading = False     # True while we drive ctx.load_sequence (ignore our own signal)
        self._connected = False   # signal subscription lives for the cached instance's lifetime
        self._root_candidate = None

        # Lazily-computed per-frame kinematics (invalidated by mask/result/reference changes), and
        # the post-processing child windows (created on demand, cached, nulled on their close).
        self._kin_cache = None
        self._kin_plot = None
        self._pk_plot = None
        self._cauchy_plot = None
        self._gauge = None

        # "Show outliers" preview: a bool mask over active points (True = RANSAC would drop it) drawn
        # red on the main canvas, plus whether our canvas overlay is currently registered.
        self._outlier_mask = None
        self._outlier_overlay_on = False

        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("MTS Uniaxial")
        self.resize(560, 820)

        content = QWidget()
        outer = QVBoxLayout(content)
        outer.addWidget(self._build_load_section())
        outer.addWidget(self._build_channel_section())
        outer.addWidget(self._build_crop_section())
        outer.addWidget(self._build_reference_section())
        outer.addWidget(self._build_handoff_section())
        outer.addWidget(self._build_ransac_section())
        outer.addWidget(self._build_plot_section())
        outer.addWidget(self._build_export_section())
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Prefill the last-used root for convenience (no auto-load / no dialog on open).
        last_root = ctx.get_settings().get("last_root")
        if last_root and os.path.isdir(last_root):
            self._set_root_candidate(last_root)

        self._update_gating()
        self._connect_signals()

    # ----------------------------------------------------------------- section builders
    def _build_load_section(self) -> QGroupBox:
        box = QGroupBox("1 · Load experiment")
        self.sec_load = box
        v = QVBoxLayout(box)

        open_row = QHBoxLayout()
        self.root_edit = QLineEdit()
        self.root_edit.setReadOnly(True)
        self.root_edit.setPlaceholderText("Experiment root folder…")
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._on_open_root)
        open_row.addWidget(QLabel("Root:"))
        open_row.addWidget(self.root_edit, 1)
        open_row.addWidget(open_btn)
        v.addLayout(open_row)

        form = QFormLayout()
        self.images_edit = QLineEdit(_DEFAULT_IMAGES_SUBDIR)
        img_browse = QPushButton("...")
        img_browse.clicked.connect(self._on_browse_images)
        img_row = QHBoxLayout()
        img_row.addWidget(self.images_edit, 1)
        img_row.addWidget(img_browse)
        form.addRow("Images folder:", img_row)

        self.sensor_edit = QLineEdit(_DEFAULT_SENSOR_REL)
        sen_browse = QPushButton("...")
        sen_browse.clicked.connect(self._on_browse_sensor)
        sen_row = QHBoxLayout()
        sen_row.addWidget(self.sensor_edit, 1)
        sen_row.addWidget(sen_browse)
        form.addRow("Sensor file:", sen_row)

        # Reference cross-section of the specimen (always millimetres) + constitutive flag. These
        # are parameters, not a workflow step — editing them never invalidates tracking.
        self.width_edit = QLineEdit(f"{self.pstate.material_width:g}")
        self.width_edit.setValidator(self._mm_validator())
        self.width_edit.editingFinished.connect(self._on_material_param_changed)
        form.addRow("Width (mm):", self.width_edit)
        self.thickness_edit = QLineEdit(f"{self.pstate.material_thickness:g}")
        self.thickness_edit.setValidator(self._mm_validator())
        self.thickness_edit.editingFinished.connect(self._on_material_param_changed)
        form.addRow("Thickness (mm):", self.thickness_edit)
        v.addLayout(form)

        self.incompressible_chk = QCheckBox("Incompressible material")
        self.incompressible_chk.setChecked(self.pstate.incompressible)
        self.incompressible_chk.toggled.connect(self._on_material_param_changed)
        v.addWidget(self.incompressible_chk)
        self.a0_label = QLabel()
        self.a0_label.setStyleSheet("color: #555;")
        v.addWidget(self.a0_label)
        self._update_a0_label()

        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load)
        load_row.addStretch(1)
        load_row.addWidget(self.load_btn)
        v.addLayout(load_row)

        self.load_summary = QLabel("No experiment loaded.")
        self.load_summary.setWordWrap(True)
        v.addWidget(self.load_summary)
        return box

    def _build_channel_section(self) -> QGroupBox:
        box = QGroupBox("2 · Channels & temporal offset")
        self.sec_channel = box
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Displacement = clamp A + clamp B (sum, always)."))

        form = QFormLayout()
        self.force_box = QComboBox()
        self._populate_force_box(("Clamp A", "Clamp B"))
        self.force_box.currentIndexChanged.connect(self._on_channel_changed)
        form.addRow("Force channel:", self.force_box)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-60000.0, 60000.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setSuffix(" ms")
        self.offset_spin.valueChanged.connect(self._on_channel_changed)
        form.addRow("Temporal offset:", self.offset_spin)
        v.addLayout(form)

        note = QLabel(sync.OFFSET_CONVENTION)
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        v.addWidget(note)
        return box

    def _build_crop_section(self) -> QGroupBox:
        box = QGroupBox("3 · Crop experiment window")
        self.sec_crop = box
        v = QVBoxLayout(box)

        self.crop_plot = CropPlotWidget()
        v.addWidget(self.crop_plot)

        self.show_images_chk = QCheckBox("Show image frames")
        self.show_images_chk.setChecked(True)
        self.show_images_chk.toggled.connect(lambda *_: self._refresh_crop_plot())
        v.addWidget(self.show_images_chk)

        self.crop_lo = QSlider(Qt.Orientation.Horizontal)
        self.crop_hi = QSlider(Qt.Orientation.Horizontal)
        self.crop_lo.valueChanged.connect(self._on_crop_changed)
        self.crop_hi.valueChanged.connect(self._on_crop_changed)
        self.crop_lo_label = QLabel("—")
        self.crop_hi_label = QLabel("—")
        for slider, label, name in (
            (self.crop_lo, self.crop_lo_label, "First sample:"),
            (self.crop_hi, self.crop_hi_label, "Last sample:"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            row.addWidget(slider, 1)
            row.addWidget(label)
            v.addLayout(row)
        return box

    def _build_reference_section(self) -> QGroupBox:
        box = QGroupBox("4 · Find reference")
        self.sec_reference = box
        v = QVBoxLayout(box)

        # A force-vs-displacement view of the search sub-window (set by the two sliders below).
        self.ref_plot = ReferencePlotWidget()
        v.addWidget(self.ref_plot)

        # The two sliders bound a search sub-window inside the crop; reference finding runs on it.
        self.ref_lo = QSlider(Qt.Orientation.Horizontal)
        self.ref_hi = QSlider(Qt.Orientation.Horizontal)
        self.ref_lo.valueChanged.connect(self._on_ref_window_changed)
        self.ref_hi.valueChanged.connect(self._on_ref_window_changed)
        self.ref_lo_label = QLabel("—")
        self.ref_hi_label = QLabel("—")
        for slider, label, name in (
            (self.ref_lo, self.ref_lo_label, "Search from:"),
            (self.ref_hi, self.ref_hi_label, "Search to:"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            row.addWidget(slider, 1)
            row.addWidget(label)
            v.addLayout(row)

        row = QHBoxLayout()
        self.algo_box = QComboBox()
        for name in REGISTRY:
            self.algo_box.addItem(name)
        self.detect_btn = QPushButton("Detect reference")
        self.detect_btn.clicked.connect(self._on_detect_reference)
        row.addWidget(self.algo_box, 1)
        row.addWidget(self.detect_btn)
        v.addLayout(row)

        # Pre-force threshold (N) for the "Set preforce" algorithm; a parameter, not a step input,
        # so it is read only at detect time and never invalidates downstream state.
        pf = QHBoxLayout()
        pf.addWidget(QLabel("Pre-force threshold:"))
        self.preforce_spin = QDoubleSpinBox()
        self.preforce_spin.setRange(0.0, 1e6)
        self.preforce_spin.setDecimals(3)
        self.preforce_spin.setSingleStep(0.1)
        self.preforce_spin.setSuffix(" N")
        self.preforce_spin.setValue(0.1)
        pf.addWidget(self.preforce_spin)
        pf.addStretch(1)
        v.addLayout(pf)

        override = QHBoxLayout()
        override.addWidget(QLabel("Reference frame (override):"))
        self.ref_spin = QSpinBox()
        self.ref_spin.valueChanged.connect(self._on_ref_override)
        override.addWidget(self.ref_spin)
        override.addStretch(1)
        v.addLayout(override)

        self.ref_label = QLabel("No reference set.")
        self.ref_label.setWordWrap(True)
        v.addWidget(self.ref_label)
        return box

    def _build_handoff_section(self) -> QGroupBox:
        box = QGroupBox("5 · Track in the main window")
        self.sec_handoff = box
        v = QVBoxLayout(box)
        msg = QLabel(
            "Reference and last frame are set. Now define the ROI, seed points, run tracking, "
            "and filter in the main window. Export enables once a tracking result exists."
        )
        msg.setWordWrap(True)
        v.addWidget(msg)
        return box

    def _build_ransac_section(self) -> QGroupBox:
        box = QGroupBox("6 · Homogeneous RANSAC filtering")
        self.sec_ransac = box
        v = QVBoxLayout(box)
        intro = QLabel(
            "Fit one affine map reference → last frame across all active points and drop the "
            "outliers (points not moving with the homogeneous deformation). Filtering uses the "
            "app's undoable active set, so the kinematics plots refresh and you can undo it in the "
            "main window's Cleanup."
        )
        intro.setWordWrap(True)
        v.addWidget(intro)

        form = QFormLayout()
        self.ransac_sample = QSpinBox()
        self.ransac_sample.setRange(kinematics.MIN_FIT_POINTS, 50)
        self.ransac_sample.setValue(6)
        form.addRow("Points per fit:", self.ransac_sample)
        self.ransac_reproj = QDoubleSpinBox()
        self.ransac_reproj.setRange(0.1, 50.0)
        self.ransac_reproj.setSingleStep(0.5)
        self.ransac_reproj.setDecimals(2)
        self.ransac_reproj.setValue(3.0)
        form.addRow("Reproj threshold (px):", self.ransac_reproj)
        self.ransac_iters = QSpinBox()
        self.ransac_iters.setRange(10, 100000)
        self.ransac_iters.setValue(2000)
        form.addRow("Max iterations:", self.ransac_iters)
        self.ransac_conf = QDoubleSpinBox()
        self.ransac_conf.setRange(0.50, 0.99999)
        self.ransac_conf.setDecimals(5)
        self.ransac_conf.setSingleStep(0.001)
        self.ransac_conf.setValue(0.99)
        form.addRow("Confidence:", self.ransac_conf)
        v.addLayout(form)

        # Recompute the red outlier preview live as any RANSAC parameter changes (a no-op while the
        # "Show outliers" box is unchecked).
        for w in (self.ransac_sample, self.ransac_reproj, self.ransac_iters, self.ransac_conf):
            w.valueChanged.connect(self._refresh_outlier_preview)

        self.show_outliers_chk = QCheckBox("Show outliers")
        self.show_outliers_chk.setToolTip(
            "Color the points RANSAC would drop in red on the main canvas, live, without applying.")
        self.show_outliers_chk.toggled.connect(self._on_show_outliers_toggled)
        v.addWidget(self.show_outliers_chk)

        row = QHBoxLayout()
        self.ransac_status = QLabel("—")
        self.ransac_status.setWordWrap(True)
        self.ransac_btn = QPushButton("Apply RANSAC")
        self.ransac_btn.clicked.connect(self._on_apply_ransac)
        row.addWidget(self.ransac_status, 1)
        row.addWidget(self.ransac_btn)
        v.addLayout(row)
        return box

    def _build_plot_section(self) -> QGroupBox:
        box = QGroupBox("7 · Plotting")
        self.sec_plot = box
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Open a window per view (refresh live as the active set changes)."))
        grid = QGridLayout()
        self.plot_kinematics_btn = QPushButton("Kinematics")
        self.plot_kinematics_btn.clicked.connect(self._open_kinematics_plot)
        self.plot_pk_btn = QPushButton("Piola–Kirchhoff stress")
        self.plot_pk_btn.clicked.connect(self._open_pk_plot)
        self.plot_cauchy_btn = QPushButton("Cauchy stress")
        self.plot_cauchy_btn.clicked.connect(self._open_cauchy_plot)
        self.gauge_btn = QPushButton("Direction gauge")
        self.gauge_btn.clicked.connect(self._open_gauge)
        grid.addWidget(self.plot_kinematics_btn, 0, 0)
        grid.addWidget(self.plot_pk_btn, 0, 1)
        grid.addWidget(self.plot_cauchy_btn, 1, 0)
        grid.addWidget(self.gauge_btn, 1, 1)
        v.addLayout(grid)
        return box

    def _build_export_section(self) -> QGroupBox:
        box = QGroupBox("8 · Export aligned data & measures")
        self.sec_export = box
        v = QVBoxLayout(box)
        self.export_info = QLabel("—")
        self.export_info.setWordWrap(True)
        v.addWidget(self.export_info)
        row = QHBoxLayout()
        self.export_btn = QPushButton("Export aligned data")
        self.export_btn.clicked.connect(self._on_export)
        row.addStretch(1)
        row.addWidget(self.export_btn)
        v.addLayout(row)

        # Per-frame derived-measures CSV with a checkbox column picker.
        measures_box = QGroupBox("Derived measures (one row per frame)")
        mv = QVBoxLayout(measures_box)
        grid = QGridLayout()
        self.measure_checks = {}
        col = rowi = 0
        for key, header, always in _MEASURE_COLUMNS:
            if always:
                continue
            chk = QCheckBox(header)
            chk.setChecked(True)
            self.measure_checks[key] = chk
            grid.addWidget(chk, rowi, col)
            col += 1
            if col == 3:
                col, rowi = 0, rowi + 1
        mv.addLayout(grid)
        note = QLabel("frame_global + time_s are always included. Force sign is as recorded; "
                      "stresses are in MPa (N/mm²). Cauchy needs the incompressible flag.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        mv.addWidget(note)
        mrow = QHBoxLayout()
        self.measures_info = QLabel("—")
        self.measures_info.setWordWrap(True)
        self.measures_btn = QPushButton("Export measures (CSV)")
        self.measures_btn.clicked.connect(self._on_export_measures)
        mrow.addWidget(self.measures_info, 1)
        mrow.addWidget(self.measures_btn)
        mv.addLayout(mrow)
        v.addWidget(measures_box)
        return box

    # ----------------------------------------------------------------- load handlers
    def _set_root_candidate(self, root: str) -> None:
        self._root_candidate = root
        self.root_edit.setText(root)
        self.images_edit.setText(_DEFAULT_IMAGES_SUBDIR)
        self.sensor_edit.setText(_DEFAULT_SENSOR_REL)

    def _rel(self, path: str) -> str:
        """Show a picked path relative to the current root; fall back to absolute if impossible."""
        root = self._root_candidate
        if not root:
            return path
        try:
            return os.path.relpath(path, root)
        except ValueError:  # different drive on Windows — no relative form exists
            return path

    def _on_open_root(self) -> None:
        start = self._root_candidate or ""
        directory = QFileDialog.getExistingDirectory(self, "Open experiment root", start)
        if directory:
            self._set_root_candidate(directory)

    def _on_browse_images(self) -> None:
        start = self._root_candidate or ""
        directory = QFileDialog.getExistingDirectory(self, "Select images folder", start)
        if directory:
            self.images_edit.setText(self._rel(directory))

    def _on_browse_sensor(self) -> None:
        start = self._root_candidate or ""
        path, _ = QFileDialog.getOpenFileName(self, "Select sensor file", start, "Sensor data (*.dat *.txt);;All files (*)")
        if path:
            self.sensor_edit.setText(self._rel(path))

    def _resolve(self, text: str, root: str):
        text = text.strip()
        if not text:
            return None
        return text if os.path.isabs(text) else os.path.join(root, text)

    def _on_load(self) -> None:
        root = self._root_candidate
        if not root:
            QMessageBox.warning(self, "No root", "Open an experiment root folder first.")
            return
        images_dir = self._resolve(self.images_edit.text(), root)
        sensor_path = self._resolve(self.sensor_edit.text(), root)

        if sensor_path and os.path.isdir(sensor_path):  # a folder was given — find the .dat inside
            dats = sorted(glob.glob(os.path.join(sensor_path, "*.dat")))
            sensor_path = dats[0] if dats else sensor_path
        if not (images_dir and os.path.isdir(images_dir)):
            QMessageBox.critical(self, "Load failed", f"Images folder not found:\n{images_dir}")
            return
        log_path = self._find_log(images_dir)
        if log_path is None:
            QMessageBox.critical(self, "Load failed",
                                 f"No image log ({_LOG_NAME} or *.log) found in:\n{images_dir}")
            return
        if not (sensor_path and os.path.isfile(sensor_path)):
            QMessageBox.critical(self, "Load failed", f"Sensor file not found:\n{sensor_path}")
            return

        try:
            image_log = parsers.parse_image_log(log_path)
            sensor = parsers.parse_sensor(sensor_path)
            ordered = parsers.resolve_image_paths(image_log, images_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        # Offer to resume an existing project rather than overwrite it.
        if os.path.isfile(os.path.join(project_io.project_dir(root), project_io.MANIFEST)):
            choice = QMessageBox.question(
                self, "Resume project?",
                "An MTS Uniaxial project already exists in this folder.\n\n"
                "Resume it (restore channel, crop and reference), or start fresh (overwrite)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes,
            )
            if choice == QMessageBox.StandardButton.Yes:
                resumed = project_io.load_project(root)
                if resumed is not None:
                    self._adopt_state(resumed)
                    return
                QMessageBox.information(self, "Resume failed",
                                        "Could not resume the saved project; starting fresh.")

        self._fresh_load(root, images_dir, sensor_path, log_path, image_log, sensor, ordered)

    def _find_log(self, images_dir: str):
        preferred = os.path.join(images_dir, _LOG_NAME)
        if os.path.isfile(preferred):
            return preferred
        logs = sorted(glob.glob(os.path.join(images_dir, "*.log")))
        return logs[0] if logs else None

    def _wipe_artifacts(self, root: str, filenames) -> None:
        failures = project_io.wipe_files(project_io.project_dir(root), list(filenames))
        if failures:
            self.ctx.status("Could not remove stale artifact(s): " + "; ".join(failures), 8000)

    def _fresh_load(self, root, images_dir, sensor_path, log_path, image_log, sensor, ordered) -> None:
        st = MtsProjectState()
        st.root, st.images_dir, st.sensor_file, st.log_path = root, images_dir, sensor_path, log_path
        st.image_log, st.sensor, st.ordered_paths = image_log, sensor, ordered
        st.force_channel, st.offset_ms = "average", 0.0
        st.crop_start, st.crop_end = 0, sensor.n_samples - 1
        # Carry the material parameters the user has entered (they're specimen properties, not tied
        # to the loaded experiment) into the new project so a Load doesn't reset them.
        st.material_width = self._read_float(self.width_edit, st.material_width)
        st.material_thickness = self._read_float(self.thickness_edit, st.material_thickness)
        st.incompressible = self.incompressible_chk.isChecked()
        st.completed_through = int(Step.CROP)  # channel + crop have valid defaults
        # Validate and install the sequence before touching an existing on-disk project. A corrupt
        # candidate therefore cannot wipe a resumable project. If persistence later fails, the
        # successfully loaded experiment remains usable in-memory and the user gets a clear warning.
        self._loading = True
        try:
            loaded = self.ctx.load_sequence(ordered, root)
        finally:
            self._loading = False
        if not loaded:
            self.ctx.status("Image validation failed; the existing MTS project was left untouched.")
            return
        self.pstate = st

        try:
            self._wipe_artifacts(root, _ALL_FILES)  # clear stale project
            project_io.save_load(st)
            project_io.save_channel(st)
            project_io.save_crop(st)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                "The experiment is loaded in memory, but its project files could not be "
                f"written:\n\n{exc}",
            )

        try:
            self.ctx.save_settings({"last_root": root})
        except OSError as exc:
            self.ctx.status(f"Loaded experiment, but could not remember its folder: {exc}")

        self._sync_widgets_from_state()
        self._update_gating()
        self.ctx.status(f"Loaded {image_log.n_images} images and {sensor.n_samples} sensor samples.")

    def _adopt_state(self, st: MtsProjectState) -> None:
        """Take over a resumed state: load the sequence and re-apply the frame range."""
        self._loading = True
        try:
            loaded = self.ctx.load_sequence(st.ordered_paths, st.root)
        finally:
            self._loading = False
        if not loaded:
            self.ctx.status("Could not resume MTS project: its image sequence failed validation.")
            return
        self.pstate = st
        if st.done(Step.REFERENCE) and st.ref_image_global is not None:
            total = self.ctx.n_total_images
            self.ctx.set_last_frame(total - 1)
            self.ctx.set_reference_frame(int(st.ref_image_global))
            self.ctx.set_last_frame(int(st.last_image_global))
            self.ctx.set_current_frame(int(st.ref_image_global))
            if st.done(Step.TRACK):
                self._restore_trackers(st)
        try:
            self.ctx.save_settings({"last_root": st.root})
        except OSError as exc:
            self.ctx.status(f"Resumed project, but could not remember its folder: {exc}")
        self._sync_widgets_from_state()
        self._update_gating()
        self.ctx.status("Resumed MTS Uniaxial project.")

    def _restore_trackers(self, st: MtsProjectState) -> None:
        """Reinstall the saved core tracking result (``trackers.npz``) on resume.

        Guarded by ``self._loading`` so the ``result_changed``/``mask_changed`` the install emits
        don't re-enter our handlers (which would wipe the export and re-save). A missing or
        unreadable file degrades gracefully to the REFERENCE-only resume."""
        path = os.path.join(project_io.project_dir(st.root), "trackers.npz")
        self._loading = True
        try:
            self.ctx.load_trackers(path)
        except (OSError, ValueError) as exc:
            st.completed_through = int(Step.REFERENCE)
            self.ctx.status(f"Could not restore trackers: {exc}")
            return
        finally:
            self._loading = False
        self._invalidate_kinematics()
        self._refresh_outlier_preview()

    # ----------------------------------------------------------------- step handlers
    def _on_channel_changed(self, *_) -> None:
        st = self.pstate
        if self._syncing or not st.done(Step.LOAD):
            return
        st.force_channel = self.force_box.currentData()
        st.offset_ms = float(self.offset_spin.value())
        files = st.invalidate_from(Step.REFERENCE)  # offset/channel feed the reference, not the crop
        self._wipe_artifacts(st.root, files)
        st.completed_through = int(Step.CROP)
        try:
            project_io.save_channel(st)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save channel settings:\n\n{exc}")
        self._refresh_crop_plot()
        self._refresh_reference_plot()
        self._sync_reference_widgets()
        self._update_gating()

    def _on_crop_changed(self, *_) -> None:
        st = self.pstate
        if self._syncing or not st.done(Step.LOAD):
            return
        lo, hi = self.crop_lo.value(), self.crop_hi.value()
        if lo > hi:  # keep the two handles ordered
            if self.sender() is self.crop_lo:
                hi = lo
                self._set_slider(self.crop_hi, hi)
            else:
                lo = hi
                self._set_slider(self.crop_lo, lo)
        st.crop_start, st.crop_end = lo, hi
        files = st.invalidate_from(Step.REFERENCE)
        self._wipe_artifacts(st.root, files)
        st.completed_through = int(Step.CROP)
        try:
            project_io.save_crop(st)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save crop settings:\n\n{exc}")
        self._reset_ref_window(lo, hi)  # the search sub-window lives inside the new crop
        self._update_crop_labels()
        self._refresh_crop_plot()
        self._refresh_reference_plot()
        self._sync_reference_widgets()
        self._update_gating()

    def _on_ref_window_changed(self, *_) -> None:
        st = self.pstate
        if self._syncing or not st.done(Step.CROP):
            return
        lo, hi = self.ref_lo.value(), self.ref_hi.value()
        if lo > hi:  # keep the two handles ordered
            if self.sender() is self.ref_lo:
                hi = lo
                self._set_slider(self.ref_hi, hi)
            else:
                lo = hi
                self._set_slider(self.ref_lo, lo)
        self._update_ref_labels()
        self._refresh_reference_plot()

    def _on_detect_reference(self) -> None:
        st = self.pstate
        if not st.done(Step.CROP):
            return
        sen = st.sensor
        lo, hi = self.ref_lo.value(), self.ref_hi.value()  # the search sub-window
        disp = sync.composite_displacement(sen)[lo:hi + 1]
        force = sync.composite_force(sen, st.force_channel)[lo:hi + 1]
        algo = self.algo_box.currentText()
        fn = REGISTRY.get(algo)
        if fn is None:
            QMessageBox.warning(self, "No algorithm", "Select a reference-finding algorithm.")
            return
        # local index within the sub-window -> absolute sensor index -> nearest global image frame.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # some fits (spring-hinge) block for ~1-3 s
        try:
            local = int(fn(disp, force, self.preforce_spin.value()) if algo == PREFORCE else fn(disp, force))
        finally:
            QApplication.restoreOverrideCursor()
        abs_idx = lo + max(0, min(local, hi - lo))
        last_idx = self._sensor_to_image(st.crop_end)  # last frame is the crop end, not the sub-window
        ref_idx = self._sensor_to_image(abs_idx)
        self._apply_reference(ref_idx, max(ref_idx, last_idx), algo, abs_idx)

    def _on_ref_override(self, value: int) -> None:
        st = self.pstate
        if self._syncing or not st.done(Step.CROP):
            return
        last = st.last_image_global
        if last is None:
            last = self._sensor_to_image(st.crop_end)
        self._apply_reference(int(value), max(int(value), int(last)), "manual override", None)

    def _apply_reference(self, ref_global: int, last_global: int, algo: str, sensor_idx) -> None:
        st = self.pstate
        sen, log = st.sensor, st.image_log
        m = log.n_images
        ref_global = max(0, min(int(ref_global), m - 1))
        last_global = max(ref_global, min(int(last_global), m - 1))

        stms = sync.sensor_time_ms(sen)
        disp_img, _ = sync.interp_to_images(log.time_ms, stms, sync.composite_displacement(sen), st.offset_ms)
        force_img, _ = sync.interp_to_images(log.time_ms, stms, sync.composite_force(sen, st.force_channel), st.offset_ms)

        st.ref_algorithm = algo
        st.ref_sensor_index = int(sensor_idx) if sensor_idx is not None else None
        st.ref_image_global = ref_global
        st.last_image_global = last_global
        st.zero_disp = float(disp_img[ref_global])
        st.zero_force = float(force_img[ref_global])

        files = st.invalidate_from(Step.TRACK)  # a new reference invalidates tracking/export
        self._wipe_artifacts(st.root, files)
        st.completed_through = int(Step.REFERENCE)
        try:
            project_io.save_reference(st)
        except OSError as exc:
            st.invalidate_from(Step.REFERENCE)
            QMessageBox.critical(self, "Save failed", f"Could not save the reference:\n\n{exc}")
            self._sync_reference_widgets()
            self._update_gating()
            return

        # Drive the core: open the range fully, then set reference (clears ROI), then last.
        total = self.ctx.n_total_images
        self.ctx.set_last_frame(total - 1)
        self.ctx.set_reference_frame(ref_global)
        self.ctx.set_last_frame(last_global)
        self.ctx.set_current_frame(ref_global)  # show the reference frame so the ROI can be drawn there

        self._sync_reference_widgets()
        self._refresh_crop_plot()       # show the reference marker on the two global plots
        self._refresh_reference_plot()  # and on the sub-window plot
        self._invalidate_kinematics()   # a new reference re-scopes cut 0 / the range
        self._update_gating()
        self.ctx.status(f"Reference frame {ref_global}, last frame {last_global} ({algo}).")

    def _sensor_to_image(self, sensor_index: int) -> int:
        st = self.pstate
        return sync.sensor_index_to_image_index(
            sensor_index, sync.sensor_time_ms(st.sensor), st.image_log.time_ms, st.offset_ms
        )

    # ----------------------------------------------------------------- result / export
    def _on_result_changed(self) -> None:
        if self._loading:  # we're reinstalling a saved result during resume; don't re-handle it
            return
        st = self.pstate
        if st.done(Step.REFERENCE):
            if self.ctx.has_result:
                files = st.invalidate_from(Step.EXPORT)  # any new/changed result voids a stale export
                self._wipe_artifacts(st.root, files)
                # Persist the reloadable tracking result (the TRACK artifact) plus the manifest, so
                # resume reaches TRACK and reinstalls it. Only mark TRACK once the file is written.
                self._persist_trackers()
            else:
                # Result cleared: drop the saved trackers + the now-orphaned export.
                files = st.invalidate_from(Step.TRACK)
                self._wipe_artifacts(st.root, files)
                try:
                    project_io.save_manifest(st)
                except OSError as exc:
                    self.ctx.status(f"Could not update the MTS project manifest: {exc}")
        self._invalidate_kinematics()  # new/cleared result → recompute on next access
        self._refresh_outlier_preview()  # active set / result changed → re-judge the preview
        self._update_gating()

    def _on_mask_changed(self) -> None:
        if self._loading:  # mask_changed also fires while reinstalling a saved result
            return
        # The active set changed (our RANSAC, or the main window's Cleanup). It is part of the
        # reloadable tracking artifact, and any previous export now contains the wrong point set.
        st = self.pstate
        if st.done(Step.REFERENCE) and self.ctx.has_result:
            self._wipe_artifacts(st.root, st.invalidate_from(Step.EXPORT))
            self._persist_trackers()
        self._invalidate_kinematics()
        self._refresh_outlier_preview()  # re-judge the preview on the new active set
        self._update_gating()

    def _persist_trackers(self) -> None:
        """Atomically persist the live core result/mask or degrade to REFERENCE on failure."""
        st = self.pstate
        pdir = project_io.project_dir(st.root)
        path = os.path.join(pdir, "trackers.npz")
        try:
            self.ctx.save_trackers(path)
            st.completed_through = int(Step.TRACK)
            project_io.save_manifest(st)
        except (OSError, ValueError) as exc:
            # An old tracking file would be more dangerous than no resume at all.
            self._wipe_artifacts(st.root, ["trackers.npz"])
            st.completed_through = int(Step.REFERENCE)
            try:
                project_io.save_manifest(st)
            except OSError:
                pass
            self.ctx.status(f"Could not save trackers: {exc}")

    def _on_export(self) -> None:
        st = self.pstate
        if not (st.done(Step.REFERENCE) and self.ctx.has_result):
            return
        coords = self.ctx.coords(active_only=True)
        if coords is None or coords.shape[1] == 0:
            QMessageBox.warning(self, "Nothing to export", "There are no active points to export.")
            return
        status = self.ctx.track_status(active_only=True)
        fully_valid = (
            np.all(status == 1, axis=0)
            if status is not None
            else np.ones(coords.shape[1], dtype=bool)
        )
        if not fully_valid.any():
            QMessageBox.warning(
                self, "Nothing to export", "There are no active points valid for every frame."
            )
            return
        coords = coords[:, fully_valid, :]
        point_ids = self.ctx.point_indices()[fully_valid]
        sen, log = st.sensor, st.image_log
        stms = sync.sensor_time_ms(sen)
        disp_img, _ = sync.interp_to_images(log.time_ms, stms, sync.composite_displacement(sen), st.offset_ms)
        force_img, in_range = sync.interp_to_images(log.time_ms, stms, sync.composite_force(sen, st.force_channel), st.offset_ms)

        # Zero against the actual current reference frame, so the reference always exports as
        # (0, 0) no matter how the range was set — and keep the recorded zero in sync.
        ref = self.ctx.reference_index
        st.zero_disp, st.zero_force = float(disp_img[ref]), float(force_img[ref])
        rows = []
        for t in range(self.ctx.frame_count):
            g = ref + t
            rows.append([
                g,
                float(log.time_ms[g]),
                float(disp_img[g] - st.zero_disp),
                float(force_img[g] - st.zero_force),
                bool(in_range[g]),
            ])
        try:
            paths = project_io.save_export(st, coords, point_ids, rows)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._update_gating()
        names = ", ".join(os.path.basename(p) for p in paths)
        self.export_info.setText(f"Exported {coords.shape[1]} points × {self.ctx.frame_count} frames "
                                 f"→ {project_io.project_dir(st.root)} ({names}).")
        self.ctx.status(f"Exported {coords.shape[1]} points × {self.ctx.frame_count} frames to the project folder.")

    # ----------------------------------------------------------------- kinematics / post-processing
    def kinematics(self):
        """The cached per-frame :class:`KinematicsSeries`, computed lazily from the current active
        points. ``None`` if there is no result yet. Invalidated by mask/result/reference changes."""
        if self._kin_cache is None:
            if not self.ctx.has_result:
                return None
            coords = self.ctx.coords(active_only=True)
            log = self.pstate.image_log
            if coords is None or coords.shape[1] == 0 or log is None:
                return None
            status = self.ctx.track_status(active_only=True)
            ref = self.ctx.reference_index
            img_t = np.array([log.time_ms[ref + t] for t in range(self.ctx.frame_count)])
            self._kin_cache = kinematics.compute_series(coords, img_t, status)
        return self._kin_cache

    def _invalidate_kinematics(self) -> None:
        self._kin_cache = None
        self._refresh_open_plots()

    def _refresh_open_plots(self) -> None:
        for win in (self._kin_plot, self._pk_plot, self._cauchy_plot):
            if win is not None and win.isVisible():
                win.replot()
        if self._gauge is not None and self._gauge.isVisible():
            self._gauge.refresh()

    def _close_child_windows(self) -> None:
        for attr in ("_kin_plot", "_pk_plot", "_cauchy_plot", "_gauge"):
            win = getattr(self, attr, None)
            if win is not None:
                win.close()
                setattr(self, attr, None)

    def _plot_ready(self) -> bool:
        return self.pstate.done(Step.REFERENCE) and self.ctx.has_result

    def _aligned_force_disp(self):
        """Per-global-image displacement/force interpolated onto the image timeline (not zeroed),
        plus the in-range flag. Mirrors the export path so plots and CSV agree."""
        st = self.pstate
        sen, log = st.sensor, st.image_log
        stms = sync.sensor_time_ms(sen)
        disp_img, _ = sync.interp_to_images(
            log.time_ms, stms, sync.composite_displacement(sen), st.offset_ms)
        force_img, in_range = sync.interp_to_images(
            log.time_ms, stms, sync.composite_force(sen, st.force_channel), st.offset_ms)
        return disp_img, force_img, in_range

    def per_frame_force_N(self):
        """Per-frame force in N over the tracked range, zeroed at the reference frame. ``None`` if
        no result. (Consumed by the stress plots.)"""
        if not self._plot_ready():
            return None
        _disp, force_img, _ir = self._aligned_force_disp()
        ref = self.ctx.reference_index
        zero = float(force_img[ref])
        return np.array([force_img[ref + t] - zero for t in range(self.ctx.frame_count)])

    # ---- material parameters -----------------------------------------
    def _on_material_param_changed(self, *_) -> None:
        if self._syncing:
            return
        st = self.pstate
        st.material_width = self._read_float(self.width_edit, st.material_width)
        st.material_thickness = self._read_float(self.thickness_edit, st.material_thickness)
        st.incompressible = self.incompressible_chk.isChecked()
        self._update_a0_label()
        if st.done(Step.LOAD):
            # Material parameters feed measures.csv. Remove that derived file rather than leaving
            # a stale stress table beside a manifest containing the new specimen dimensions.
            self._wipe_artifacts(st.root, ["measures.csv"])
            self.measures_info.setText("Material parameters changed; re-export measures.")
            try:
                project_io.save_manifest(st)  # persist — but NOT a Step, so tracking stays valid
            except OSError as exc:
                self.ctx.status(f"Could not save material parameters: {exc}")
        # Geometry (λ/ε/directions) is independent of the cross-section, so the kinematics cache
        # stays valid; only the stress curves and the ε₂-incompressible overlay need a redraw.
        # The Cauchy (true-stress) curve σ=λ₁·P is valid only under incompressibility, so if the
        # user just turned that off, tear its window down rather than keep redrawing an invalid
        # curve titled "Cauchy (true) stress" (the Cauchy button is likewise disabled while
        # compressible — keep the two consistent).
        if not st.incompressible and self._cauchy_plot is not None:
            self._cauchy_plot.close()
            self._cauchy_plot = None
        self._refresh_open_plots()
        self._update_gating()

    def _read_float(self, edit, default):
        """Parse a positive float from a line edit; revert to ``default`` (rewritten) on bad input."""
        try:
            val = float(edit.text().strip())
            if not np.isfinite(val) or val <= 0:
                raise ValueError
        except ValueError:
            edit.setText(f"{default:g}")
            return default
        edit.setText(f"{val:g}")
        return val

    def _update_a0_label(self) -> None:
        self.a0_label.setText(f"Reference cross-section A₀ = {self.pstate.reference_area_mm2:.4g} mm²")

    @staticmethod
    def _mm_validator() -> QDoubleValidator:
        """A positive-millimetre validator with C-locale '.' decimals (independent of system locale)."""
        validator = QDoubleValidator(1e-6, 1e6, 4)
        validator.setLocale(QLocale(QLocale.Language.C))
        return validator

    # ---- RANSAC -------------------------------------------------------
    def _ransac_outliers(self):
        """Run RANSAC with the current panel parameters and return ``(outliers, info)``.

        ``outliers`` is a bool mask over the active points (True = a point RANSAC would drop) or
        ``None`` when the fit cannot run, in which case ``info`` explains why (otherwise ``info`` is
        ``None``). Points not valid at both the reference and the last frame are never judged (kept).
        Shared by the live "Show outliers" preview and the apply action so both stay in lockstep.
        """
        if not self._plot_ready():
            return None, "Run tracking and set a reference frame first."
        coords = self.ctx.coords(active_only=True)
        if coords is None or coords.shape[1] < kinematics.MIN_FIT_POINTS:
            return None, "Need at least 3 active points to filter."
        last = self.ctx.frame_count - 1
        status = self.ctx.track_status(active_only=True)
        if status is not None:
            valid = (status[0] == 1) & (status[last] == 1)
        else:
            valid = np.ones(coords.shape[1], dtype=bool)
        vidx = np.where(valid)[0]
        if vidx.size < kinematics.MIN_FIT_POINTS:
            return None, "Too few points are valid at both the reference and the last frame."
        M, inliers = kinematics.ransac_affine(
            coords[0][vidx], coords[last][vidx],
            sample_size=self.ransac_sample.value(), reproj=self.ransac_reproj.value(),
            max_iters=self.ransac_iters.value(), confidence=self.ransac_conf.value())
        if M is None:
            return None, "No consensus affine found."
        outliers = np.zeros(self.ctx.n_active, dtype=bool)
        outliers[vidx[~inliers]] = True  # points invalid at ref-or-last are not judged, so kept
        return outliers, None

    def _on_apply_ransac(self) -> None:
        outliers, info = self._ransac_outliers()
        if outliers is None:
            QMessageBox.warning(self, "RANSAC", f"{info} Nothing applied.")
            return
        dropped = int(outliers.sum())
        if dropped == 0:
            self.ransac_status.setText(f"No outliers — kept all {outliers.size} points.")
            return
        # → mask_changed → _on_mask_changed → kinematics invalidated + preview recomputed → refresh.
        self.ctx.apply_keep_mask(~outliers)
        self.ransac_status.setText(
            f"Dropped {dropped} outlier(s); kept {int((~outliers).sum())} of {outliers.size}. "
            "Undo in the main window's Cleanup.")

    # ---- "Show outliers" preview --------------------------------------
    def _on_show_outliers_toggled(self, checked: bool) -> None:
        if checked:
            if not self._outlier_overlay_on:
                self.ctx.add_overlay(self._paint_outliers)
                self._outlier_overlay_on = True
            self._refresh_outlier_preview()
        else:
            self._outlier_mask = None
            if self._outlier_overlay_on:
                self.ctx.remove_overlay(self._paint_outliers)
                self._outlier_overlay_on = False
            self.ctx.request_redraw()

    def _refresh_outlier_preview(self) -> None:
        """Recompute the red outlier mask and redraw. No-op unless "Show outliers" is checked."""
        if not self.show_outliers_chk.isChecked():
            return
        outliers, info = self._ransac_outliers()
        self._outlier_mask = outliers
        if outliers is None:
            self.ransac_status.setText(info)
        else:
            self.ransac_status.setText(
                f"Preview: {int(outliers.sum())} outlier(s) shown in red of {outliers.size} points.")
        self.ctx.request_redraw()

    def _paint_outliers(self, painter, ctx) -> None:
        """Canvas overlay: draw the previewed RANSAC outliers as red dots at the current frame."""
        mask = self._outlier_mask
        if mask is None:
            return
        coords = ctx.coords(active_only=True)
        cut = ctx.current_cut
        if coords is None or cut is None or coords.shape[1] != mask.size:
            return
        pts = coords[cut]
        painter.setPen(QPen(_OUTLIER_COLOR, 1))
        painter.setBrush(QBrush(_OUTLIER_COLOR))
        for i in np.nonzero(mask)[0]:
            painter.drawEllipse(ctx.image_to_screen(float(pts[i][0]), float(pts[i][1])), 4, 4)

    # ---- plot windows -------------------------------------------------
    def _ensure_matplotlib(self) -> bool:
        if not matplotlib_available():
            QMessageBox.critical(
                self, "matplotlib unavailable",
                "matplotlib with a working Qt backend is required for plotting. "
                "Run `uv sync` and try again.")
            return False
        return True

    def _open_kinematics_plot(self) -> None:
        if self.kinematics() is None or not self._ensure_matplotlib():
            return
        if self._kin_plot is None:
            from .plots import KinematicsPlotWindow
            self._kin_plot = KinematicsPlotWindow(self)
        self._kin_plot.show()
        self._kin_plot.raise_()
        self._kin_plot.replot()

    def _open_pk_plot(self) -> None:
        if self.kinematics() is None or not self._ensure_matplotlib():
            return
        if self._pk_plot is None:
            from .plots import StressPlotWindow
            self._pk_plot = StressPlotWindow(self, "pk")
        self._pk_plot.show()
        self._pk_plot.raise_()
        self._pk_plot.replot()

    def _open_cauchy_plot(self) -> None:
        if not self.pstate.incompressible:
            return
        if self.kinematics() is None or not self._ensure_matplotlib():
            return
        if self._cauchy_plot is None:
            from .plots import StressPlotWindow
            self._cauchy_plot = StressPlotWindow(self, "cauchy")
        self._cauchy_plot.show()
        self._cauchy_plot.raise_()
        self._cauchy_plot.replot()

    def _open_gauge(self) -> None:
        if self.kinematics() is None:
            return
        if self._gauge is None:
            from .gauge import DirectionGaugeWindow
            self._gauge = DirectionGaugeWindow(self)
        self._gauge.show()
        self._gauge.raise_()
        self._gauge.refresh()

    def on_plot_closed(self, win) -> None:
        if win is self._kin_plot:
            self._kin_plot = None
        elif win is self._pk_plot:
            self._pk_plot = None
        elif win is self._cauchy_plot:
            self._cauchy_plot = None

    def on_gauge_closed(self) -> None:
        self._gauge = None

    # ---- measures CSV -------------------------------------------------
    def _on_export_measures(self) -> None:
        st = self.pstate
        if not (st.done(Step.REFERENCE) and self.ctx.has_result):
            return
        series = self.kinematics()
        if series is None:
            QMessageBox.warning(self, "Nothing to export", "No active points / kinematics to export.")
            return
        disp_img, force_img, in_range = self._aligned_force_disp()
        ref = self.ctx.reference_index
        zero_disp, zero_force = float(disp_img[ref]), float(force_img[ref])
        a0 = st.reference_area_mm2

        keys = [key for key, _h, always in _MEASURE_COLUMNS
                if always or (self.measure_checks[key].isChecked()
                              and self.measure_checks[key].isEnabled())]
        header = [h for key, h, _a in _MEASURE_COLUMNS if key in keys]
        rows = []
        for t in range(self.ctx.frame_count):
            g = ref + t
            pk = (force_img[g] - zero_force) / a0 if a0 > 0 else float("nan")
            values = {
                "frame_global": g,
                "time_s": series.time_s[t],
                "lambda_1": series.lambda_1[t],
                "lambda_2": series.lambda_2[t],
                "eps_1": series.eps_1[t],
                "eps_2": series.eps_2[t],
                "eps_2_ico": series.eps_2_ico[t],
                "pk_stress_MPa": pk,
                "cauchy_stress_MPa": series.lambda_1[t] * pk,
                "force_N": force_img[g] - zero_force,
                "displacement_mm": disp_img[g] - zero_disp,
                "angle_deg": series.angle_deg[t],
                "n_points": int(series.n_points[t]),
                "in_range": int(bool(in_range[g])),
            }
            rows.append([self._fmt_cell(k, values[k]) for k in keys])
        try:
            path = project_io.save_measures(st, header, rows)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.measures_info.setText(
            f"Exported {len(rows)} frames × {len(header)} columns → {os.path.basename(path)}.")
        self.ctx.status(f"Exported measures CSV ({len(header)} columns).")

    @staticmethod
    def _fmt_cell(key, val) -> str:
        if key in _MEASURE_INT_KEYS:
            return str(int(val))
        v = float(val)
        return "nan" if not np.isfinite(v) else f"{v:.6g}"

    # ----------------------------------------------------------------- external sequence
    def _on_sequence_changed(self) -> None:
        if self._loading:
            return  # our own load
        if self.pstate.done(Step.LOAD):
            self.pstate = MtsProjectState()  # an external File->Open replaced our sequence
            self._close_child_windows()
            self._invalidate_kinematics()
            self._sync_widgets_from_state()
            self._update_gating()
            self.ctx.status("MTS Uniaxial detached: the image sequence was changed outside the plugin.")

    # ----------------------------------------------------------------- widget sync
    def _populate_force_box(self, labels) -> None:
        self.force_box.clear()
        self.force_box.addItem(f"Clamp A ({labels[0]})", "A")
        self.force_box.addItem(f"Clamp B ({labels[1]})", "B")
        self.force_box.addItem("Average (A+B)/2", "average")

    def _set_slider(self, slider: QSlider, value: int) -> None:
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)

    def _sync_widgets_from_state(self) -> None:
        st = self.pstate
        self._syncing = True
        try:
            self.width_edit.setText(f"{st.material_width:g}")
            self.thickness_edit.setText(f"{st.material_thickness:g}")
            self.incompressible_chk.setChecked(st.incompressible)
            self._update_a0_label()
            if st.root:
                self.root_edit.setText(st.root)
            if st.sensor is not None:
                self._populate_force_box(st.sensor.clamp_labels)
                idx = max(0, self.force_box.findData(st.force_channel))
                self.force_box.setCurrentIndex(idx)
                self.offset_spin.setValue(st.offset_ms)
                n = st.sensor.n_samples
                for slider in (self.crop_lo, self.crop_hi):
                    slider.setRange(0, max(0, n - 1))
                self._set_slider(self.crop_lo, st.crop_start or 0)
                self._set_slider(self.crop_hi, st.crop_end if st.crop_end is not None else n - 1)
                self._update_crop_labels()
                cs = st.crop_start or 0
                ce = st.crop_end if st.crop_end is not None else n - 1
                for slider in (self.ref_lo, self.ref_hi):
                    slider.setRange(cs, ce)
                self._set_slider(self.ref_lo, cs)
                self._set_slider(self.ref_hi, ce)
                self._update_ref_labels()
                self.ref_spin.setRange(0, st.image_log.n_images - 1)
                self._load_summary_text()
                self._refresh_crop_plot()
                self._refresh_reference_plot()
            else:
                self.load_summary.setText("No experiment loaded.")
                self.crop_plot.clear()
                self.ref_plot.clear()
            self._sync_reference_widgets()
        finally:
            self._syncing = False

    def _sync_reference_widgets(self) -> None:
        st = self.pstate
        was = self._syncing
        self._syncing = True
        try:
            if st.ref_algorithm and st.ref_image_global is not None:
                if st.ref_algorithm in REGISTRY:
                    self.algo_box.setCurrentText(st.ref_algorithm)
                self.ref_spin.setValue(int(st.ref_image_global))
                self.ref_label.setText(
                    f"Reference frame {st.ref_image_global} (t={st.image_log.time_ms[st.ref_image_global]:.0f} ms), "
                    f"last frame {st.last_image_global}. "
                    f"Zero: displacement={st.zero_disp:.4g}, force={st.zero_force:.4g} [{st.ref_algorithm}]."
                )
            else:
                self.ref_label.setText("No reference set.")
        finally:
            self._syncing = was

    def _load_summary_text(self) -> None:
        st = self.pstate
        sen, log = st.sensor, st.image_log
        img_span = f"{log.time_ms[0]:.0f}–{log.time_ms[-1]:.0f} ms"
        sen_span = f"{sen.time_s[0] * 1000:.0f}–{sen.time_s[-1] * 1000:.0f} ms"
        extra = f", {sen.n_skipped} rows skipped" if sen.n_skipped else ""
        self.load_summary.setText(
            f"{log.n_images} images ({img_span}) · {sen.n_samples} sensor samples ({sen_span}){extra}."
        )

    def _update_crop_labels(self) -> None:
        st = self.pstate
        if st.sensor is None:
            return
        t = st.sensor.time_s * 1000.0
        lo, hi = self.crop_lo.value(), self.crop_hi.value()
        self.crop_lo_label.setText(f"{lo} ({t[lo]:.0f} ms)")
        self.crop_hi_label.setText(f"{hi} ({t[hi]:.0f} ms)")

    def _reset_ref_window(self, lo: int, hi: int) -> None:
        """Clamp the reference search sub-window to the crop [lo, hi] and reset it to the full crop."""
        for slider in (self.ref_lo, self.ref_hi):
            slider.blockSignals(True)
            slider.setRange(lo, hi)
            slider.blockSignals(False)
        self._set_slider(self.ref_lo, lo)
        self._set_slider(self.ref_hi, hi)
        self._update_ref_labels()

    def _update_ref_labels(self) -> None:
        st = self.pstate
        if st.sensor is None:
            return
        t = st.sensor.time_s * 1000.0
        lo, hi = self.ref_lo.value(), self.ref_hi.value()
        self.ref_lo_label.setText(f"{lo} ({t[lo]:.0f} ms)")
        self.ref_hi_label.setText(f"{hi} ({t[hi]:.0f} ms)")

    def _refresh_crop_plot(self) -> None:
        st = self.pstate
        if st.sensor is None or not self.crop_plot.available:
            return
        sen, log = st.sensor, st.image_log
        stms = sync.sensor_time_ms(sen)
        disp = sync.composite_displacement(sen)
        force = sync.composite_force(sen, st.force_channel)
        img_force, _ = sync.interp_to_images(log.time_ms, stms, force, st.offset_ms)
        img_disp, _ = sync.interp_to_images(log.time_ms, stms, disp, st.offset_ms)
        self.crop_plot.update_data(stms, disp, force, st.crop_start or 0,
                                   st.crop_end if st.crop_end is not None else sen.n_samples - 1,
                                   log.time_ms, img_disp, img_force,
                                   show_images=self.show_images_chk.isChecked(),
                                   ref=st.ref_image_global)

    def _refresh_reference_plot(self) -> None:
        st = self.pstate
        if st.sensor is None or not self.ref_plot.available:
            return
        sen, log = st.sensor, st.image_log
        disp = sync.composite_displacement(sen)
        force = sync.composite_force(sen, st.force_channel)
        ref_disp = ref_force = None
        if st.ref_image_global is not None:  # mark the reference at its image-interpolated position
            stms = sync.sensor_time_ms(sen)
            img_disp, _ = sync.interp_to_images(log.time_ms, stms, disp, st.offset_ms)
            img_force, _ = sync.interp_to_images(log.time_ms, stms, force, st.offset_ms)
            r = st.ref_image_global
            ref_disp, ref_force = float(img_disp[r]), float(img_force[r])
        self.ref_plot.update_data(disp, force, self.ref_lo.value(), self.ref_hi.value(),
                                  ref_disp, ref_force)

    def _update_gating(self) -> None:
        st = self.pstate
        self.sec_channel.setEnabled(st.done(Step.LOAD))
        self.sec_crop.setEnabled(st.done(Step.CHANNEL))
        self.sec_reference.setEnabled(st.done(Step.CROP))
        self.sec_handoff.setEnabled(st.done(Step.REFERENCE))
        self.sec_export.setEnabled(st.done(Step.REFERENCE))
        self.export_btn.setEnabled(st.done(Step.TRACK) and self.ctx.has_result)

        # Post-processing panels need a reference + a tracking result; Cauchy also needs the flag.
        ready = self._plot_ready()
        self.sec_ransac.setEnabled(ready)
        self.sec_plot.setEnabled(ready)
        incompressible = self.incompressible_chk.isChecked()
        self.plot_cauchy_btn.setEnabled(ready and incompressible)
        self.measures_btn.setEnabled(st.done(Step.TRACK) and self.ctx.has_result)
        self.measure_checks["cauchy_stress_MPa"].setEnabled(incompressible)

        if not st.done(Step.REFERENCE):
            self.export_info.setText("Set a reference frame, then track in the main window.")
        elif not self.ctx.has_result:
            self.export_info.setText("No tracking result yet — run tracking in the main window.")
        else:
            self.export_info.setText(f"{self.ctx.n_active} points × {self.ctx.frame_count} frames ready.")

    # ----------------------------------------------------------------- lifecycle
    def _connect_signals(self) -> None:
        """Observe project-integrity changes for the plugin instance's full lifetime."""
        if self._connected:
            return
        self.ctx.signals.result_changed.connect(self._on_result_changed)
        self.ctx.signals.sequence_changed.connect(self._on_sequence_changed)
        self.ctx.signals.mask_changed.connect(self._on_mask_changed)
        self._connected = True

    def _disconnect_signals(self) -> None:
        if not self._connected:
            return
        for sig, slot in (
            (self.ctx.signals.result_changed, self._on_result_changed),
            (self.ctx.signals.sequence_changed, self._on_sequence_changed),
            (self.ctx.signals.mask_changed, self._on_mask_changed),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connected = False

    def dispose(self) -> None:
        """Final teardown used on plugin reload; ordinary window closes keep state observation."""
        self._disconnect_signals()
        self._close_child_windows()
        if self._outlier_overlay_on:
            self.ctx.remove_overlay(self._paint_outliers)
            self._outlier_overlay_on = False

    def showEvent(self, event) -> None:
        # Connections normally live for the cached instance's full lifetime; the idempotent call
        # also makes a manually disposed/reused development window recover safely.
        super().showEvent(event)
        self._connect_signals()
        # Window is cached and reused: re-register the outlier overlay if it was left checked.
        if self.show_outliers_chk.isChecked() and not self._outlier_overlay_on:
            self.ctx.add_overlay(self._paint_outliers)
            self._outlier_overlay_on = True
            self._refresh_outlier_preview()
        self._update_gating()

    def closeEvent(self, event) -> None:
        self._close_child_windows()
        if self._outlier_overlay_on:
            self.ctx.remove_overlay(self._paint_outliers)
            self._outlier_overlay_on = False  # checkbox state kept, so reopen restores the preview
        super().closeEvent(event)
