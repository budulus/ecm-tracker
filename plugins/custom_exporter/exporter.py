"""The exporter window and the format writers (kept separate from the plugin entry point)."""
import csv
import json
import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# (label, format-key, file extension, save-dialog filter)
FORMATS = [
    ("Long-form CSV (frame, point, x, y)", "csv", ".csv", "CSV (*.csv)"),
    ("MATLAB .mat", "mat", ".mat", "MATLAB (*.mat)"),
    ("NumPy .npz (+ metadata)", "npz", ".npz", "NumPy archive (*.npz)"),
]


def _frame_global_indices(ctx) -> np.ndarray:
    """Global frame index for each cut row of the coords array."""
    ref = ctx.reference_index
    return np.arange(ref, ref + ctx.frame_count)


def write_csv(path, coords, point_ids, frame_globals):
    """coords: (frames, points, 2). One row per (frame, point)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_global", "frame_cut", "point_id", "x", "y"])
        for t in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                x, y = coords[t, j]
                w.writerow([int(frame_globals[t]), t, int(point_ids[j]), f"{x:.4f}", f"{y:.4f}"])


def write_mat(path, coords, point_ids, frame_globals, ref, last):
    from scipy.io import savemat  # imported lazily so the dep is only needed on use

    savemat(
        path,
        {
            "coords": coords.astype(np.float64),  # (frames, points, 2)
            "point_ids": point_ids.astype(np.int64).reshape(-1, 1),
            "frame_global_indices": frame_globals.astype(np.int64).reshape(-1, 1),
            "reference_index": int(ref),
            "last_index": int(last),
        },
    )


def write_npz(path, coords, point_ids, frame_globals, ref, last):
    np.savez(
        path,
        coords=coords.astype(np.float32),
        point_ids=point_ids.astype(np.int64),
        frame_global_indices=frame_globals.astype(np.int64),
        reference_index=np.int64(ref),
        last_index=np.int64(last),
    )
    meta = {
        "format": "feature-tracker coords export",
        "shape": list(coords.shape),
        "axes": ["frame (cut order; 0 = reference)", "point", "(x, y)"],
        "reference_index": int(ref),
        "last_index": int(last),
        "n_points": int(coords.shape[1]),
    }
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


class CustomExporterWindow(QWidget):
    """A small top-level window to export the current tracked coordinates."""

    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Custom Exporter")
        self.setMinimumWidth(380)

        saved = ctx.get_settings()

        self.format_box = QComboBox()
        for label, key, _ext, _filter in FORMATS:
            self.format_box.addItem(label, key)
        idx = max(0, self.format_box.findData(saved.get("format", "csv")))
        self.format_box.setCurrentIndex(idx)

        self.active_only = QCheckBox("Export kept (active) points only")
        self.active_only.setChecked(bool(saved.get("active_only", True)))

        self.info = QLabel()
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self._on_export)

        form = QFormLayout()
        form.addRow("Format:", self.format_box)
        form.addRow(self.active_only)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.info)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.export_btn)
        layout.addLayout(row)

        # Stay in sync with the app: refresh availability when tracking/mask changes.
        ctx.signals.result_changed.connect(self._refresh)
        ctx.signals.mask_changed.connect(self._refresh)
        self.active_only.toggled.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        if not self.ctx.has_result:
            self.info.setText("No tracking result yet — run tracking first.")
            self.export_btn.setEnabled(False)
            return
        n_pts = self.ctx.n_active if self.active_only.isChecked() else self.ctx.point_count
        self.info.setText(f"{n_pts} points × {self.ctx.frame_count} frames ready to export.")
        self.export_btn.setEnabled(n_pts > 0)

    def _on_export(self):
        coords = self.ctx.coords(active_only=self.active_only.isChecked())
        if coords is None or coords.shape[1] == 0:
            QMessageBox.warning(self, "Nothing to export", "There are no points to export.")
            return
        if self.active_only.isChecked():
            point_ids = self.ctx.point_indices()
        else:
            point_ids = np.arange(self.ctx.point_count)
        frame_globals = _frame_global_indices(self.ctx)

        key = self.format_box.currentData()
        _label, _key, ext, file_filter = next(f for f in FORMATS if f[1] == key)
        start_dir = self.ctx.get_settings().get("last_dir") or ""
        default = os.path.join(start_dir, f"tracked_coords{ext}")
        path, _ = QFileDialog.getSaveFileName(self, "Export coordinates", default, file_filter)
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext

        ref, last = self.ctx.reference_index, self.ctx.last_index
        try:
            if key == "csv":
                write_csv(path, coords, point_ids, frame_globals)
            elif key == "mat":
                write_mat(path, coords, point_ids, frame_globals, ref, last)
            else:
                write_npz(path, coords, point_ids, frame_globals, ref, last)
        except Exception as exc:  # surface writer/dependency errors to the user
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        self.ctx.save_settings(
            {"format": key, "active_only": self.active_only.isChecked(),
             "last_dir": os.path.dirname(path)}
        )
        self.ctx.status(f"Exported {coords.shape[1]} points to {os.path.basename(path)}")
        QMessageBox.information(self, "Export complete", f"Wrote:\n{path}")
