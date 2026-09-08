"""Pressure–strain plugin window and reactive lifecycle."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import analysis


_MODE_LABELS = (
    ("ε₁ — major principal strain", "epsilon_1"),
    ("ε₂ — minor principal strain", "epsilon_2"),
    ("Mean — (ε₁ + ε₂) / 2", "mean"),
)


class PressureStrainWindow(QWidget):
    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self._pressure = None
        self._aligned = None
        self._strain_cache = None
        self._plot_window = None
        self._connected = False

        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Pressure–Strain")
        self.resize(560, 390)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(self._build_pressure_section())
        root.addWidget(self._build_plot_section())
        root.addStretch(1)

        saved = self.ctx.get_settings()
        saved_mode = saved.get("strain_mode", "epsilon_1")
        index = next(
            (i for i in range(self.strain_combo.count())
             if self.strain_combo.itemData(i) == saved_mode),
            0,
        )
        self.strain_combo.setCurrentIndex(index)
        self.zero_check.setChecked(bool(saved.get("zero_pressure", False)))

        self._connect_signals()
        self._update_gating()

    def _build_pressure_section(self) -> QGroupBox:
        box = QGroupBox("1 · Pressure data")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Load a pressure logger CSV…")
        self.load_button = QPushButton("Load CSV…")
        self.load_button.clicked.connect(self._choose_pressure)
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.load_button)
        layout.addLayout(row)

        self.pressure_info = QLabel(
            "Load images first; the file picker opens one directory above the image folder."
        )
        self.pressure_info.setWordWrap(True)
        self.pressure_info.setStyleSheet("color: #6b727a;")
        layout.addWidget(self.pressure_info)
        return box

    def _build_plot_section(self) -> QGroupBox:
        box = QGroupBox("2 · Plot and export")
        layout = QVBoxLayout(box)

        form = QFormLayout()
        self.strain_combo = QComboBox()
        for label, mode in _MODE_LABELS:
            self.strain_combo.addItem(label, mode)
        self.strain_combo.currentIndexChanged.connect(self._on_controls_changed)
        form.addRow("Strain:", self.strain_combo)
        layout.addLayout(form)

        self.zero_check = QCheckBox("Shift pressure at first plotted frame to zero")
        self.zero_check.toggled.connect(self._on_controls_changed)
        layout.addWidget(self.zero_check)

        self.tracking_info = QLabel("Run tracking in the main window to calculate strain.")
        self.tracking_info.setWordWrap(True)
        self.tracking_info.setStyleSheet("color: #6b727a;")
        layout.addWidget(self.tracking_info)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.plot_button = QPushButton("Plot")
        self.plot_button.setDefault(True)
        self.plot_button.clicked.connect(self._open_plot)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self._export)
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.export_button)
        layout.addLayout(buttons)
        return box

    # --------------------------------------------------------------- pressure loading
    def _picker_start_directory(self) -> str:
        source = self.ctx.source_dir
        return os.path.dirname(os.path.abspath(source)) if source else ""

    def _choose_pressure(self) -> None:
        if not self.ctx.has_sequence:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load pressure data",
            self._picker_start_directory(),
            "Pressure CSV (*.csv);;All files (*)",
        )
        if path:
            self.load_pressure_file(path)

    def load_pressure_file(self, path: str) -> bool:
        """Transactionally parse and align ``path``; public to keep UI regression tests direct."""
        try:
            if not self.ctx.has_sequence:
                raise ValueError("Load an image sequence before loading pressure data")
            paths = self.ctx.frame_paths
            if len(paths) != self.ctx.n_total_images:
                raise ValueError("The image sequence changed while pressure data was loading")
            pressure = analysis.parse_pressure_csv(path)
            timestamps = analysis.image_modified_times(paths)
            aligned = analysis.align_pressure_to_images(pressure, timestamps)
        except ValueError as exc:
            QMessageBox.critical(self, "Pressure load failed", str(exc))
            return False

        self._pressure = pressure
        self._aligned = aligned
        self.path_edit.setText(pressure.source_path)
        self.path_edit.setToolTip(pressure.source_path)
        self.pressure_info.setText(
            f"Assumed endpoint / file-mtime alignment: {pressure.source_rows:,} CSV rows to "
            f"{aligned.pressure_mbar.size:,} frames · "
            f"{aligned.pressure_mbar[0]:.4g} to {aligned.pressure_mbar[-1]:.4g} mbar"
        )
        self._update_gating()
        self._refresh_plot()
        self.ctx.status(f"Loaded pressure data from {os.path.basename(path)}")
        return True

    # --------------------------------------------------------------- derived data
    def _strain_series(self):
        if self._strain_cache is not None:
            return self._strain_cache
        if not self.ctx.has_result or self.ctx.n_active < 3:
            return None
        coords = self.ctx.coords(active_only=True)
        status = self.ctx.track_status(active_only=True)
        if coords is None or coords.shape[0] != self.ctx.frame_count:
            return None
        self._strain_cache = analysis.compute_principal_strains(coords, status)
        return self._strain_cache

    def current_plot_series(self):
        strains = self._strain_series()
        if self._aligned is None or strains is None:
            return None
        try:
            return analysis.build_plot_series(
                self._aligned,
                strains,
                self.ctx.reference_index,
                self.ctx.last_index,
                self.strain_combo.currentData(),
                self.zero_check.isChecked(),
            )
        except ValueError:
            return None

    # --------------------------------------------------------------- plotting/export
    def _open_plot(self) -> None:
        series = self.current_plot_series()
        if series is None:
            QMessageBox.warning(
                self,
                "Plot unavailable",
                "Load pressure data and calculate tracking with at least three active points.",
            )
            return
        if self._plot_window is None:
            try:
                from .plot import PressureStrainPlotWindow

                self._plot_window = PressureStrainPlotWindow(self)
            except Exception as exc:
                self._plot_window = None
                QMessageBox.critical(
                    self,
                    "matplotlib unavailable",
                    "A working matplotlib Qt backend is required for plotting.\n\n" + str(exc),
                )
                return
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.replot()

    def _refresh_plot(self) -> None:
        if self._plot_window is not None and self._plot_window.isVisible():
            self._plot_window.replot()

    def on_plot_closed(self, window) -> None:
        if window is self._plot_window:
            self._plot_window = None

    def _export(self) -> None:
        series = self.current_plot_series()
        if series is None:
            return
        start = os.path.dirname(self._pressure.source_path) if self._pressure is not None else ""
        default_path = os.path.join(start, "pressure_strain.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export pressure–strain data", default_path, "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            analysis.export_plot_csv(path, series, self.ctx.frame_paths)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.ctx.status(f"Exported pressure–strain data to {path}")

    # --------------------------------------------------------------- reactive state
    def _on_controls_changed(self, *_) -> None:
        if not hasattr(self, "zero_check"):
            return
        self.ctx.save_settings(
            {
                "strain_mode": self.strain_combo.currentData(),
                "zero_pressure": self.zero_check.isChecked(),
            }
        )
        self._refresh_plot()

    def _on_sequence_changed(self) -> None:
        self._pressure = None
        self._aligned = None
        self._strain_cache = None
        self.path_edit.clear()
        self.path_edit.setToolTip("")
        self.pressure_info.setText(
            "Choose a pressure CSV for the newly loaded image sequence."
        )
        if self._plot_window is not None:
            self._plot_window.close()
            self._plot_window = None
        self._update_gating()

    def _on_result_or_mask_changed(self) -> None:
        self._strain_cache = None
        self._update_gating()
        self._refresh_plot()

    def _update_gating(self) -> None:
        self.load_button.setEnabled(self.ctx.has_sequence)
        ready = self._aligned is not None and self.ctx.has_result and self.ctx.n_active >= 3
        self.plot_button.setEnabled(ready)
        self.export_button.setEnabled(ready)
        self.strain_combo.setEnabled(self.ctx.has_result)
        self.zero_check.setEnabled(self._aligned is not None)

        if not self.ctx.has_result:
            self.tracking_info.setText("Run tracking in the main window to calculate strain.")
        elif self.ctx.n_active < 3:
            self.tracking_info.setText("At least three active tracked points are required.")
        else:
            self.tracking_info.setText(
                f"{self.ctx.n_active:,} active points × {self.ctx.frame_count:,} tracked frames."
            )

    def _connect_signals(self) -> None:
        if self._connected:
            return
        self.ctx.signals.sequence_changed.connect(self._on_sequence_changed)
        self.ctx.signals.result_changed.connect(self._on_result_or_mask_changed)
        self.ctx.signals.mask_changed.connect(self._on_result_or_mask_changed)
        self._connected = True

    def dispose(self) -> None:
        if self._connected:
            for signal, slot in (
                (self.ctx.signals.sequence_changed, self._on_sequence_changed),
                (self.ctx.signals.result_changed, self._on_result_or_mask_changed),
                (self.ctx.signals.mask_changed, self._on_result_or_mask_changed),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            self._connected = False
        if self._plot_window is not None:
            self._plot_window.close()
            self._plot_window = None

    def showEvent(self, event):
        super().showEvent(event)
        self._connect_signals()
        self._update_gating()

    def closeEvent(self, event):
        if self._plot_window is not None:
            self._plot_window.close()
            self._plot_window = None
        super().closeEvent(event)
