from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core import settings
from app.core.cleanup import (
    BAND_CAP,
    BAND_METRICS,
    COUNT_BANDS,
    BandFilter,
    Metrics,
    Thresholds,
    thresholds_to_dict,
)

# Human-readable labels for each band filter.
BAND_LABELS = {
    "fw_failures": "Forward status failures",
    "bw_failures": "Backward status failures",
    "opencv_error": "Max OpenCV error",
    "mean_error": "Mean OpenCV error",
    "fb_mean": "FB mean error",
    "fb_max": "FB max error",
    "distance": "Distance (max single step)",
}

# Resolution of the float-metric sliders (integer count metrics use 0..cap directly).
SLIDER_STEPS = 1000


def _finite_max(arr: np.ndarray, fallback: float = 1.0) -> float:
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size and finite.max() > 0 else fallback


class _BandRow(QWidget):
    """Enable checkbox + a slider (the applied max threshold) + a live value readout + a "max"
    box that sets the slider's full-scale. The spin/slider are greyed while the checkbox is off
    and the filter is ignored; when enabled a point is kept iff metric <= the slider value.

    `self._hi` is the source of truth for the applied threshold: the slider derives it while
    dragging, and editing the max box keeps it fixed (clamped to the new scale) by repositioning
    the slider."""

    changed = pyqtSignal()

    def __init__(self, label: str, integer: bool, hi_cap: float, hi_initial: float):
        super().__init__()
        self._integer = integer
        self._hi_cap = hi_cap  # hard ceiling the max box may be set to

        self.enable = QCheckBox(label)
        self.slider = QSlider(Qt.Horizontal)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(64)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        if integer:
            self.cap_box = QSpinBox()
            self.cap_box.setRange(1, max(1, int(hi_cap)))
        else:
            self.cap_box = QDoubleSpinBox()
            self.cap_box.setRange(0.0, float(hi_cap))
            self.cap_box.setDecimals(3)
            self.cap_box.setSingleStep(0.1)

        self._set_cap_box(min(hi_initial, hi_cap))
        self._hi = self._cap()  # default: full-scale -> keep every finite point
        self._configure_slider_range()
        self._sync_slider()
        self._set_enabled(False)

        self.enable.toggled.connect(self._on_enable)
        self.slider.valueChanged.connect(self._on_slider)
        self.cap_box.valueChanged.connect(self._on_cap)

    # ---- helpers --------------------------------------------------------
    def _cap(self) -> float:
        return float(self.cap_box.value())

    def _set_cap_box(self, value: float) -> None:
        self.cap_box.blockSignals(True)
        self.cap_box.setValue(int(round(value)) if self._integer else float(value))
        self.cap_box.blockSignals(False)

    def _configure_slider_range(self) -> None:
        self.slider.blockSignals(True)
        if self._integer:
            self.slider.setRange(0, max(1, int(round(self._cap()))))
            self.slider.setSingleStep(1)
        else:
            self.slider.setRange(0, SLIDER_STEPS)
        self.slider.blockSignals(False)

    def _sync_slider(self) -> None:
        """Clamp self._hi to the current cap and position the slider to represent it."""
        cap = self._cap()
        self._hi = max(0.0, min(self._hi, cap))
        if self._integer:
            pos = int(round(self._hi))
        else:
            pos = 0 if cap <= 0 else int(round(self._hi / cap * SLIDER_STEPS))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self._refresh_label()

    def _hi_from_slider(self) -> float:
        if self._integer:
            return float(self.slider.value())
        return self.slider.value() / SLIDER_STEPS * self._cap()

    def _refresh_label(self) -> None:
        self.value_label.setText(
            str(int(round(self._hi))) if self._integer else f"{self._hi:.3f}"
        )

    def _set_enabled(self, on: bool) -> None:
        self.slider.setEnabled(on)
        self.cap_box.setEnabled(on)
        self.value_label.setEnabled(on)

    # ---- signal handlers ------------------------------------------------
    def _on_enable(self, checked: bool) -> None:
        self._set_enabled(checked)
        self.changed.emit()

    def _on_slider(self, _value: int) -> None:
        self._hi = self._hi_from_slider()
        self._refresh_label()
        self.changed.emit()

    def _on_cap(self, _value: float) -> None:
        self._configure_slider_range()
        self._sync_slider()  # keep self._hi (clamped to the new cap), reposition the slider
        self.changed.emit()

    # ---- threshold state ------------------------------------------------
    def band(self) -> BandFilter:
        return BandFilter(self.enable.isChecked(), self._hi, self._cap())

    def set_band(self, band: BandFilter) -> None:
        self.enable.blockSignals(True)
        self.enable.setChecked(band.enabled)
        self.enable.blockSignals(False)
        cap = band.cap if band.cap > 0 else self._cap()
        self._set_cap_box(min(cap, self._hi_cap))
        self._hi = band.hi
        self._configure_slider_range()
        self._sync_slider()
        self._set_enabled(band.enabled)


class CleanupDialog(QDialog):
    """Max-threshold filters (slider + max box) with a live green/red preview and apply/undo/
    reset actions, plus a Save button that persists the thresholds as defaults.

    The dialog reports threshold state and emits action requests; the MainWindow owns the
    active mask and undo stack and performs the actual mutations.
    """

    thresholdsChanged = pyqtSignal()
    applyRequested = pyqtSignal()
    undoRequested = pyqtSignal()

    def __init__(self, metrics: Metrics, defaults: Optional[Thresholds] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cleanup")
        self.setModal(False)

        initial_hi = {
            "fw_failures": metrics.n_frames,
            "bw_failures": metrics.n_frames,
            "opencv_error": _finite_max(metrics.max_err_fw),
            "mean_error": _finite_max(metrics.mean_err_fw),
            "fb_mean": _finite_max(metrics.fb_mean),
            "fb_max": _finite_max(metrics.fb_max),
            "distance": _finite_max(metrics.max_step),
        }

        # Data-driven baseline: every band disabled, scale + threshold at the metric's max
        # (keep everything). Used as the Reset target and when no settings are saved.
        baseline = Thresholds()
        for name in BAND_METRICS:
            setattr(baseline, name, BandFilter(False, initial_hi[name], initial_hi[name]))

        if defaults is not None:
            merged = Thresholds(
                drop_left_image=defaults.drop_left_image,
                drop_left_roi=defaults.drop_left_roi,
            )
            for name in BAND_METRICS:
                saved = getattr(defaults, name)
                # A missing/sentinel cap falls back to the data-driven scale.
                cap = saved.cap if 0 < saved.cap < BAND_CAP else initial_hi[name]
                setattr(merged, name, BandFilter(saved.enabled, min(saved.hi, cap), cap))
            self._defaults = merged
        else:
            self._defaults = baseline

        grid = QGridLayout()
        grid.addWidget(QLabel("threshold (keep ≤)"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 4)
        grid.setColumnStretch(1, 1)
        self._rows = {}
        for i, name in enumerate(BAND_METRICS, start=1):
            integer = name in COUNT_BANDS
            cap = metrics.n_frames if integer else BAND_CAP
            row = _BandRow(BAND_LABELS[name], integer, cap, initial_hi[name])
            row.changed.connect(self.thresholdsChanged)
            setattr(self, name, row)
            self._rows[name] = row
            grid.addWidget(row.enable, i, 0)
            grid.addWidget(row.slider, i, 1)
            grid.addWidget(row.value_label, i, 2)
            grid.addWidget(QLabel("max"), i, 3)
            grid.addWidget(row.cap_box, i, 4)

        self.drop_left_image = QCheckBox("Drop points that leave image bounds")
        self.drop_left_roi = QCheckBox("Drop points that leave ROI bounds")
        self.drop_left_image.toggled.connect(self.thresholdsChanged)
        self.drop_left_roi.toggled.connect(self.thresholdsChanged)

        filters_box = QGroupBox("Filters (keep when metric ≤ threshold; unchecked = ignored)")
        box_layout = QVBoxLayout(filters_box)
        box_layout.addLayout(grid)
        box_layout.addWidget(self.drop_left_image)
        box_layout.addWidget(self.drop_left_roi)

        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignCenter)

        reset_btn = QPushButton("Reset thresholds")
        self._save_btn = QPushButton("Save parameters")
        save_btn = self._save_btn
        apply_btn = QPushButton("Apply cleanup")
        undo_btn = QPushButton("Undo last cleanup")
        reset_btn.clicked.connect(self._on_reset)
        save_btn.clicked.connect(self._on_save)
        apply_btn.clicked.connect(self.applyRequested)
        undo_btn.clicked.connect(self.undoRequested)
        buttons = QHBoxLayout()
        for b in (reset_btn, save_btn, apply_btn, undo_btn):
            buttons.addWidget(b)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(filters_box)
        layout.addWidget(self.count_label)
        layout.addLayout(buttons)
        layout.addWidget(close_box)

        self._apply_thresholds(self._defaults)

    def thresholds(self) -> Thresholds:
        thr = Thresholds(
            drop_left_image=self.drop_left_image.isChecked(),
            drop_left_roi=self.drop_left_roi.isChecked(),
        )
        for name, row in self._rows.items():
            setattr(thr, name, row.band())
        return thr

    def _apply_thresholds(self, thr: Thresholds) -> None:
        for name, row in self._rows.items():
            row.set_band(getattr(thr, name))
        for box, value in (
            (self.drop_left_image, thr.drop_left_image),
            (self.drop_left_roi, thr.drop_left_roi),
        ):
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)
        self.thresholdsChanged.emit()

    def set_counts(self, kept: int, total: int) -> None:
        self.count_label.setText(f"Keeping {kept} / {total} points")

    def _on_reset(self) -> None:
        self._apply_thresholds(self._defaults)

    def _on_save(self) -> None:
        settings.update_section("cleanup", thresholds_to_dict(self.thresholds()))
        self._defaults = self.thresholds()
        self._save_btn.setText("Saved ✓")
        QTimer.singleShot(1500, lambda: self._save_btn.setText("Save parameters"))
