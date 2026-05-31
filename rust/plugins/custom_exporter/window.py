"""Standalone PyQt5 export window for the Custom Exporter plugin (Phase 4 slice 4a).

Run as a subprocess by the plugin's ``launch()``::

    python window.py <data_dir> <outbox_path>

Loads the host's data snapshot (``snapshot.npz`` + ``settings.json``), shows a small dialog (format
combo + active-only checkbox + count + Export), and writes the chosen format with the shared
``exporter`` module. Settings the user picks are reported back to the host by appending a JSON line to
the outbox file, which the retained plugin instance drains on its next reactive event.
"""

import json
import os
import sys

import numpy as np

# Make ``plugins.custom_exporter.exporter`` importable when run as a standalone script: the rust root
# (grandparent of plugins/) goes on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RUST_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../rust
if _RUST_ROOT not in sys.path:
    sys.path.insert(0, _RUST_ROOT)

from plugins.custom_exporter import exporter  # noqa: E402

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FORMATS = [
    ("Long-form CSV (frame, point, x, y)", "csv"),
    ("MATLAB .mat", "mat"),
    ("NumPy .npz (+ metadata)", "npz"),
]


class ExportWindow(QWidget):
    def __init__(self, data_dir, outbox):
        super().__init__()
        self.outbox = outbox
        npz = np.load(os.path.join(data_dir, "snapshot.npz"))
        self.coords_full = npz["coords"]  # (F, P, 2)
        self.active_idx = npz["active_indices"]  # (n_active,)
        self.frame_globals = npz["frame_globals"]  # (F,)
        self.ref = int(npz["reference_index"])
        self.last = int(npz["last_index"])
        try:
            with open(os.path.join(data_dir, "settings.json")) as f:
                self.saved = json.load(f)
        except (OSError, ValueError):
            self.saved = {}

        self.setWindowTitle("Custom Exporter")
        self.setMinimumWidth(380)

        self.format_box = QComboBox()
        for label, key in FORMATS:
            self.format_box.addItem(label, key)
        want = self.saved.get("format", "csv")
        for i, (_, key) in enumerate(FORMATS):
            if key == want:
                self.format_box.setCurrentIndex(i)

        self.active_only = QCheckBox("Export kept (active) points only")
        self.active_only.setChecked(bool(self.saved.get("active_only", True)))
        self.active_only.toggled.connect(self._refresh)

        self.info = QLabel()
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self._on_export)

        form = QFormLayout()
        form.addRow("Format", self.format_box)
        form.addRow(self.active_only)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.export_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.info)
        layout.addLayout(row)
        self._refresh()

    def _current(self):
        if self.active_only.isChecked():
            return self.coords_full[:, self.active_idx, :], self.active_idx
        return self.coords_full, np.arange(self.coords_full.shape[1], dtype=np.int64)

    def _refresh(self):
        coords, _ = self._current()
        self.info.setText(f"{coords.shape[1]} points × {coords.shape[0]} frames ready to export.")
        self.export_btn.setEnabled(coords.shape[1] > 0)

    def _on_export(self):
        fmt = self.format_box.currentData()
        start = self.saved.get("last_dir") or ""
        path, _ = QFileDialog.getSaveFileName(self, "Export", start, f"*.{fmt}")
        if not path:
            return
        coords, point_ids = self._current()
        exporter.write(fmt, path, coords, point_ids, self.frame_globals, self.ref, self.last)
        self.saved = {
            "format": fmt,
            "active_only": self.active_only.isChecked(),
            "last_dir": os.path.dirname(path),
        }
        self._report(self.saved)
        self.info.setText(f"Exported {coords.shape[1]} points to {os.path.basename(path)}.")

    def _report(self, settings):
        with open(self.outbox, "a") as f:
            f.write(json.dumps({"type": "save_settings", "data": settings}) + "\n")


def main():
    if len(sys.argv) < 3:
        print("usage: window.py <data_dir> <outbox_path>", file=sys.stderr)
        sys.exit(2)
    app = QApplication(sys.argv)
    win = ExportWindow(sys.argv[1], sys.argv[2])
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
