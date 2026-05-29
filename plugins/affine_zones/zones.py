"""Zone drawing interaction, per-zone RANSAC affine fit, and the control window."""
import csv

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.plugins import CanvasInteraction

MIN_ZONE_POINTS = 3  # an affine fit needs at least 3 correspondences


def points_in_polygon(polygon, pts):
    """Boolean mask of which (x, y) rows of ``pts`` fall inside ``polygon`` (list of (x, y))."""
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return np.array(
        [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in pts],
        dtype=bool,
    )


def fit_zone_affine(polygon, ref_pts, cur_pts):
    """Fit a RANSAC affine for the points inside ``polygon``.

    Returns ``(local_idx, M, inliers)`` where ``local_idx`` indexes into the active-point arrays,
    ``M`` is the 2×3 affine (or None), and ``inliers`` is a bool array aligned to ``local_idx``.
    Returns None if the zone holds too few points.
    """
    inside = points_in_polygon(polygon, ref_pts)
    local_idx = np.where(inside)[0]
    if local_idx.size < MIN_ZONE_POINTS:
        return None
    src = ref_pts[local_idx].astype(np.float32)
    dst = cur_pts[local_idx].astype(np.float32)
    M, inlier_col = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
    inliers = (
        inlier_col.ravel().astype(bool)
        if inlier_col is not None
        else np.zeros(local_idx.size, dtype=bool)
    )
    return local_idx, M, inliers


class ZoneDrawTool(CanvasInteraction):
    """Collects polygon vertices from canvas clicks while the user draws a zone."""

    def __init__(self, window):
        self.window = window
        self.vertices = []

    def on_press(self, image_pt, event):
        self.vertices.append((image_pt.x(), image_pt.y()))
        self.window.on_draw_progress()


class AffineZonesWindow(QWidget):
    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Affine Zone Tool")
        self.setMinimumWidth(560)
        self.zones = []  # list of polygons (each a list of (x, y) in image coords)
        self._tool = None

        self.new_btn = QPushButton("New Zone")
        self.finish_btn = QPushButton("Finish Zone")
        self.clear_btn = QPushButton("Clear Zones")
        self.export_btn = QPushButton("Export CSV…")
        self.drop_btn = QPushButton("Drop Outliers")
        self.new_btn.clicked.connect(self._start_zone)
        self.finish_btn.clicked.connect(self._finish_zone)
        self.clear_btn.clicked.connect(self._clear_zones)
        self.export_btn.clicked.connect(self._export)
        self.drop_btn.clicked.connect(self._drop_outliers)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Zone", "Pts", "Inliers", "a11", "a12", "tx", "a21", "a22", "ty"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hint = QLabel()

        top = QHBoxLayout()
        for b in (self.new_btn, self.finish_btn, self.clear_btn, self.export_btn, self.drop_btn):
            top.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.hint)
        layout.addWidget(self.table)

        ctx.signals.frame_changed.connect(self._refresh)
        ctx.signals.result_changed.connect(self._refresh)
        ctx.signals.mask_changed.connect(self._refresh)
        ctx.add_overlay(self._paint)
        self._update_buttons()
        self._refresh()

    # ---- lifecycle ------------------------------------------------------
    def closeEvent(self, event):
        if self._tool is not None:
            self.ctx.end_canvas_interaction()
            self._tool = None
        self.ctx.remove_overlay(self._paint)
        super().closeEvent(event)

    # ---- drawing zones --------------------------------------------------
    def _start_zone(self):
        self._tool = ZoneDrawTool(self)
        self.ctx.begin_canvas_interaction(self._tool)
        self.ctx.status("Click to add zone vertices, then press “Finish Zone”.")
        self._update_buttons()

    def on_draw_progress(self):
        self.ctx.request_redraw()

    def _finish_zone(self):
        if self._tool is not None and len(self._tool.vertices) >= MIN_ZONE_POINTS:
            self.zones.append(list(self._tool.vertices))
        self.ctx.end_canvas_interaction()
        self._tool = None
        self._update_buttons()
        self._refresh()

    def _clear_zones(self):
        self.zones.clear()
        self._refresh()

    def _update_buttons(self):
        drawing = self._tool is not None
        self.new_btn.setEnabled(not drawing)
        self.finish_btn.setEnabled(drawing)
        has = bool(self.zones)
        self.clear_btn.setEnabled(has)
        self.export_btn.setEnabled(has)
        self.drop_btn.setEnabled(has)

    # ---- compute --------------------------------------------------------
    def _current_fits(self):
        """Return a list aligned to self.zones of (local_idx, M, inliers) or None per zone."""
        ctx = self.ctx
        cut = ctx.current_cut
        coords = ctx.coords(active_only=True)
        if cut is None or coords is None or coords.shape[1] == 0:
            return [None] * len(self.zones)
        ref_pts, cur_pts = coords[0], coords[cut]
        return [fit_zone_affine(poly, ref_pts, cur_pts) for poly in self.zones]

    def _refresh(self):
        fits = self._current_fits()
        self.table.setRowCount(len(self.zones))
        for z, fit in enumerate(fits):
            self._set_cell(z, 0, str(z + 1))
            if fit is None or fit[1] is None:
                for col in range(1, 9):
                    self._set_cell(z, col, "—")
                continue
            local_idx, M, inliers = fit
            self._set_cell(z, 1, str(local_idx.size))
            self._set_cell(z, 2, str(int(inliers.sum())))
            for col, val in zip(range(3, 9), (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])):
                self._set_cell(z, col, f"{val:.4f}")
        if not self.ctx.has_result:
            self.hint.setText("No tracking result yet — run tracking first.")
        elif self.ctx.current_cut == 0:
            self.hint.setText("On the reference frame the affine is identity; move the slider.")
        else:
            self.hint.setText(f"{len(self.zones)} zone(s). Affine maps reference → current frame.")
        self.ctx.request_redraw()

    def _set_cell(self, row, col, text):
        self.table.setItem(row, col, QTableWidgetItem(text))

    # ---- actions --------------------------------------------------------
    def _drop_outliers(self):
        """Drop RANSAC outliers (in every zone) from the active set, at the current frame."""
        fits = self._current_fits()
        keep = np.ones(self.ctx.n_active, dtype=bool)
        dropped = 0
        for fit in fits:
            if fit is None or fit[1] is None:
                continue
            local_idx, _M, inliers = fit
            keep[local_idx[~inliers]] = False
            dropped += int((~inliers).sum())
        if dropped == 0:
            QMessageBox.information(self, "No outliers", "No RANSAC outliers to drop.")
            return
        self.ctx.apply_keep_mask(keep)
        self.ctx.status(f"Dropped {dropped} outlier point(s). Undo in the Cleanup dialog.")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export per-frame affines", "zone_affines.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        coords = self.ctx.coords(active_only=True)
        if coords is None:
            return
        ref_pts = coords[0]
        ref_global = self.ctx.reference_index
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["zone", "frame_global", "n_points", "n_inliers",
                            "a11", "a12", "tx", "a21", "a22", "ty"])
                for cut in range(self.ctx.frame_count):
                    cur_pts = coords[cut]
                    for z, poly in enumerate(self.zones):
                        fit = fit_zone_affine(poly, ref_pts, cur_pts)
                        if fit is None or fit[1] is None:
                            continue
                        local_idx, M, inliers = fit
                        w.writerow([z + 1, ref_global + cut, local_idx.size, int(inliers.sum()),
                                    *[f"{v:.6f}" for v in
                                      (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])]])
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.ctx.status(f"Exported zone affines to {path}")

    # ---- overlay --------------------------------------------------------
    def _paint(self, painter, ctx):
        # finished zones + their inlier/outlier points at the current frame
        fits = self._current_fits()
        for poly, fit in zip(self.zones, fits):
            screen = QPolygonF([ctx.image_to_screen(x, y) for x, y in poly])
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.setBrush(QBrush(QColor(0, 200, 255, 30)))
            painter.drawPolygon(screen)
            if fit is None or fit[1] is None:
                continue
            local_idx, _M, inliers = fit
            coords = ctx.coords(active_only=True)
            cut = ctx.current_cut
            cur_pts = coords[cut]
            for k, j in enumerate(local_idx):
                color = QColor(0, 220, 0) if inliers[k] else QColor(255, 60, 60)
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(ctx.image_to_screen(*cur_pts[j]), 3, 3)

        # in-progress polygon being drawn
        if self._tool is not None and self._tool.vertices:
            pts = [ctx.image_to_screen(x, y) for x, y in self._tool.vertices]
            painter.setPen(QPen(QColor(255, 215, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            if len(pts) >= 2:
                painter.drawPolyline(QPolygonF(pts))
            for p in pts:
                painter.drawEllipse(p, 4, 4)
