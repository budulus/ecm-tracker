"""Standalone PyQt5 + embedded-matplotlib window for the Affine Zones plugin (Phase 4 slice 4c).

Run as a subprocess by the plugin's ``launch()``::

    python window.py <data_dir> <outbox_path>

Loads the host's data snapshot (``snapshot.npz`` with coords/status + ``meta.json``), shows the
reference frame image, and lets the user draw polygon zones with the mouse, read per-zone principal
stretches in a table, plot λ1/λ2 vs frame, run RANSAC outlier removal, and export a CSV. The RANSAC
keep-mask is reported back to the host by appending a JSON line to the outbox; the retained plugin
instance applies it when the user clicks the plugin panel's "Apply changes from window" button.
"""

import json
import os
import sys

import numpy as np

# Make ``plugins.affine_zones.zones`` importable when run as a standalone script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RUST_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../rust
if _RUST_ROOT not in sys.path:
    sys.path.insert(0, _RUST_ROOT)

from plugins.affine_zones import zones as Z  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Qt5Agg")
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PyQt5.QtGui import QColor  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# A distinct color per zone, cycling.
PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153),
]


class Zone:
    def __init__(self, polygon, color):
        self.polygon = polygon  # list of (x, y)
        self.color = color  # QColor


class ZonesWindow(QWidget):
    def __init__(self, data_dir, outbox):
        super().__init__()
        self.outbox = outbox
        npz = np.load(os.path.join(data_dir, "snapshot.npz"))
        self.coords = npz["coords"]  # (F, P, 2)
        self.status = npz["status"]  # (F, P)
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.ref_pts = self.coords[0]
        self.n_active = int(self.meta["n_active"])
        self.frame_count = int(self.meta["frame_count"])
        self.ref_global = int(self.meta["reference_index"])

        self.zones = []
        self._drawing = []  # vertices of the in-progress zone
        self._outliers = np.zeros(self.n_active, dtype=bool)

        self.setWindowTitle("Affine Zone Tool")
        self.resize(900, 640)
        self._build_ui()
        self._show_reference()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, max(0, self.frame_count - 1))
        self.frame_spin.setValue(self.frame_count - 1)
        self.frame_spin.valueChanged.connect(self._refresh_table)

        self.new_btn = QPushButton("New Zone")
        self.finish_btn = QPushButton("Finish Zone")
        self.clear_btn = QPushButton("Clear Zones")
        self.ransac_btn = QPushButton("RANSAC…")
        self.plot_btn = QPushButton("Plot Curves")
        self.export_btn = QPushButton("Export CSV…")
        self.new_btn.clicked.connect(self._new_zone)
        self.finish_btn.clicked.connect(self._finish_zone)
        self.clear_btn.clicked.connect(self._clear_zones)
        self.ransac_btn.clicked.connect(self._run_ransac)
        self.plot_btn.clicked.connect(self._plot_curves)
        self.export_btn.clicked.connect(self._export_csv)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Zone", "Color", "Pts", "λ1", "λ2", "v1 (x,y)", "v2 (x,y)"]
        )
        self.info = QLabel("Click 'New Zone', click vertices on the image, then 'Finish Zone'.")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Frame:"))
        controls.addWidget(self.frame_spin)
        for b in (self.new_btn, self.finish_btn, self.clear_btn,
                  self.ransac_btn, self.plot_btn, self.export_btn):
            controls.addWidget(b)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 3)
        layout.addWidget(self.table, 2)
        layout.addWidget(self.info)

    def _show_reference(self):
        self.ax.clear()
        path = self.meta.get("ref_path") or ""
        if path and os.path.exists(path):
            try:
                self.ax.imshow(mpimg.imread(path))
            except Exception:
                self.ax.scatter(self.ref_pts[:, 0], self.ref_pts[:, 1], s=4, c="0.5")
                self.ax.invert_yaxis()
        else:
            self.ax.scatter(self.ref_pts[:, 0], self.ref_pts[:, 1], s=4, c="0.5")
            self.ax.invert_yaxis()
        self.ax.set_title("Reference frame — draw zones here")
        self._draw_zones()

    def _draw_zones(self):
        # Remove previous zone/vertex/outlier artists, keep the image.
        for art in list(self.ax.lines) + list(self.ax.collections[1:] if self.ax.images else self.ax.collections):
            try:
                art.remove()
            except Exception:
                pass
        for zone in self.zones:
            poly = np.asarray(zone.polygon + [zone.polygon[0]], dtype=float)
            rgbf = (zone.color.red() / 255, zone.color.green() / 255, zone.color.blue() / 255)
            self.ax.plot(poly[:, 0], poly[:, 1], "-", color=rgbf, linewidth=1.5)
        if self._drawing:
            d = np.asarray(self._drawing, dtype=float)
            self.ax.plot(d[:, 0], d[:, 1], "y--o", linewidth=1.0, markersize=3)
        if self._outliers.any():
            out = self.ref_pts[self._outliers]
            self.ax.scatter(out[:, 0], out[:, 1], s=20, facecolors="none", edgecolors="r")
        self.canvas.draw_idle()

    # ---- zone drawing -----------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self._drawing.append((float(event.xdata), float(event.ydata)))
        self._draw_zones()

    def _new_zone(self):
        self._drawing = []
        self.info.setText("Click vertices on the image, then 'Finish Zone'.")
        self._draw_zones()

    def _finish_zone(self):
        if len(self._drawing) < 3:
            self.info.setText("A zone needs at least 3 vertices.")
            return
        color = QColor(*PALETTE[len(self.zones) % len(PALETTE)])
        self.zones.append(Zone(list(self._drawing), color))
        self._drawing = []
        self._add_table_row()
        self._refresh_table()
        self._draw_zones()

    def _clear_zones(self):
        self.zones = []
        self._drawing = []
        self._outliers = np.zeros(self.n_active, dtype=bool)
        self.table.setRowCount(0)
        self._show_reference()

    # ---- table ------------------------------------------------------------
    def _add_table_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        btn = QPushButton("color")
        z = self.zones[row]
        btn.setStyleSheet(f"background-color: {z.color.name()};")
        btn.clicked.connect(lambda _checked, r=row: self._pick_color(r))
        self.table.setCellWidget(row, 1, btn)

    def _pick_color(self, row):
        c = QColorDialog.getColor(self.zones[row].color, self, "Zone color")
        if c.isValid():
            self.zones[row].color = c
            self.table.cellWidget(row, 1).setStyleSheet(f"background-color: {c.name()};")
            self._draw_zones()

    def _refresh_table(self):
        t = self.frame_spin.value()
        cur_pts = self.coords[t]
        valid = self.status[t] == 1
        for row, zone in enumerate(self.zones):
            fit = Z.fit_zone_deformation(zone.polygon, self.ref_pts, cur_pts, valid=valid)
            if fit is None:
                vals = [str(row + 1), None, "0", "—", "—", "—", "—"]
            else:
                local_idx, F, _b = fit
                lam1, lam2, v1, v2 = Z.principal_stretches(F)
                vals = [
                    str(row + 1), None, str(local_idx.size),
                    f"{lam1:.4f}", f"{lam2:.4f}",
                    f"{v1[0]:.3f}, {v1[1]:.3f}", f"{v2[0]:.3f}, {v2[1]:.3f}",
                ]
            for col, text in enumerate(vals):
                if col == 1 or text is None:
                    continue
                self.table.setItem(row, col, QTableWidgetItem(text))

    # ---- RANSAC -----------------------------------------------------------
    def _run_ransac(self):
        if not self.zones:
            self.info.setText("Draw a zone first.")
            return
        dlg = RansacDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        reproj = dlg.reproj.value()
        t = self.frame_spin.value()
        cur_pts = self.coords[t]
        valid = self.status[t] == 1
        self._outliers = np.zeros(self.n_active, dtype=bool)
        total_out = 0
        for zone in self.zones:
            fit = Z.fit_zone_deformation(zone.polygon, self.ref_pts, cur_pts, valid=valid)
            if fit is None:
                continue
            local_idx, _F, _b = fit
            _M, inliers = Z.estimate_affine_ransac(
                self.ref_pts[local_idx], cur_pts[local_idx], reproj=reproj
            )
            self._outliers[local_idx[~inliers]] = True
            total_out += int((~inliers).sum())
        keep = (~self._outliers).tolist()
        self._report({"type": "apply_keep_mask", "keep": keep})
        self.info.setText(
            f"RANSAC marked {total_out} outliers (red). Click the plugin panel's "
            "'Apply changes from window' to drop them."
        )
        self._draw_zones()

    # ---- plot -------------------------------------------------------------
    def _plot_curves(self):
        if not self.zones:
            self.info.setText("Draw a zone first.")
            return
        PlotDialog(self).show()

    # ---- export -----------------------------------------------------------
    def _export_csv(self):
        if not self.zones:
            self.info.setText("Draw a zone first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export zones CSV", "", "*.csv")
        if not path:
            return
        Z.write_zones_csv(
            path, [z.polygon for z in self.zones], self.coords, self.status,
            self.ref_global, self.frame_count,
        )
        self.info.setText(f"Exported {os.path.basename(path)}.")

    def _report(self, command):
        with open(self.outbox, "a") as f:
            f.write(json.dumps(command) + "\n")


class RansacDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("RANSAC outlier removal")
        self.reproj = QDoubleSpinBox()
        self.reproj.setRange(0.1, 50.0)
        self.reproj.setSingleStep(0.5)
        self.reproj.setValue(3.0)
        ok = QPushButton("Run")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(QLabel("Reprojection threshold (px):"))
        row.addWidget(self.reproj)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addLayout(btns)


class PlotDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.setWindowTitle("Principal stretches vs frame")
        self.resize(640, 420)
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        frames = np.array([owner.ref_global + t for t in range(owner.frame_count)])
        for z, zone in enumerate(owner.zones):
            lam1 = np.full(owner.frame_count, np.nan)
            lam2 = np.full(owner.frame_count, np.nan)
            for t in range(owner.frame_count):
                valid = owner.status[t] == 1
                fit = Z.fit_zone_deformation(zone.polygon, owner.ref_pts, owner.coords[t], valid=valid)
                if fit is None:
                    continue
                lam1[t], lam2[t], _v1, _v2 = Z.principal_stretches(fit[1])
            rgbf = (zone.color.red() / 255, zone.color.green() / 255, zone.color.blue() / 255)
            ax.plot(frames, lam1, "-", color=rgbf, label=f"Zone {z + 1} λ1")
            ax.plot(frames, lam2, "--", color=rgbf, label=f"Zone {z + 1} λ2")
        ax.set_xlabel("global frame")
        ax.set_ylabel("principal stretch λ")
        ax.legend(fontsize="small")
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)


def main():
    if len(sys.argv) < 3:
        print("usage: window.py <data_dir> <outbox_path>", file=sys.stderr)
        sys.exit(2)
    app = QApplication(sys.argv)
    win = ZonesWindow(sys.argv[1], sys.argv[2])
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
