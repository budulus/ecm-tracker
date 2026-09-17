"""Zone drawing, per-zone homogeneous-motion (principal-stretch) analysis, RANSAC cleaning.

For each polygon zone the tracked points inside it are fit (least squares, reference → current
frame) with an affine map ``x' = F x + b``. The linear part ``F`` is the homogenized
**deformation gradient** of the zone; its left Cauchy–Green tensor ``B = F Fᵀ`` yields the two
**principal stretches** ``λ1 ≥ λ2`` (square roots of the eigenvalues) and their normalized
eigenvectors (principal directions in the current/deformed configuration). The translation ``b`` is
irrelevant for motion analysis and is discarded. A separate RANSAC tool cleans a zone's points; a
plot window shows the stretches over frames.
"""
import csv

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.plugins.analysis import ROI
from app.plugins.analysis import fit_affine, principal_stretches, ransac_affine
from app.plugins.analysis import atomic_open, compose_alignment_affines
from app.plugins import CanvasInteraction

MIN_ZONE_POINTS = 3  # an affine fit needs at least 3 correspondences

# Distinct, non-red palette — red is reserved for RANSAC-preview outliers.
PALETTE = [
    QColor("#2563eb"),  # blue
    QColor("#16a34a"),  # green
    QColor("#f59e0b"),  # amber
    QColor("#9333ea"),  # purple
    QColor("#0891b2"),  # teal
    QColor("#db2777"),  # magenta
    QColor("#65a30d"),  # olive
    QColor("#92400e"),  # brown
    QColor("#0ea5e9"),  # sky
    QColor("#7c3aed"),  # violet
]
OUTLIER_COLOR = QColor(255, 60, 60)


def default_zone_color(index):
    """A distinct, non-red color for the zone at ``index`` (cycles through ``PALETTE``)."""
    return QColor(PALETTE[index % len(PALETTE)])


class Zone:
    """A drawn polygon zone plus its display color."""

    def __init__(self, polygon, color):
        self.polygon = list(polygon)  # list of (x, y) in image coords
        self.color = QColor(color)


def points_in_polygon(polygon, pts):
    """Boolean mask of which (x, y) rows of ``pts`` fall inside ``polygon``. Delegates to
    ``ROI.contains_many`` so the point-in-polygon test lives in one place (``app.core.roi``)."""
    return ROI(list(polygon)).contains_many(pts)


def fit_zone_deformation(polygon, ref_pts, cur_pts, valid=None):
    """Least-squares affine fit for the points inside ``polygon``.

    Returns ``(local_idx, F, b)`` where ``local_idx`` indexes into the active-point arrays, ``F``
    is the 2×2 deformation gradient (linear part of ``x' = F x + b``) and ``b`` is the translation
    (discarded by callers). Returns ``None`` if the zone holds too few points.

    ``valid`` (optional) is a bool mask over the active points; points that are False — e.g. a
    track LK lost at this frame, whose last position was carried forward — are excluded so dead
    tracks don't bias the deformation gradient.
    """
    inside = points_in_polygon(polygon, ref_pts)
    if valid is not None:
        inside &= np.asarray(valid, dtype=bool)
    fit = fit_affine(ref_pts, cur_pts, inside)
    if fit is not None and (np.linalg.det(fit[1]) <= 0 or np.linalg.cond(fit[1]) > 1e8):
        return None
    return fit


def _ransac_affine(src, dst, sample_size, reproj, max_iters, confidence):
    """Custom RANSAC affine fit of ``src`` → ``dst`` using ``sample_size`` points per hypothesis.

    Generalizes OpenCV's minimal-sample (3-point) RANSAC: each hypothesis is a least-squares affine
    fit over ``sample_size`` randomly drawn correspondences (exact when ``sample_size == 3``, an
    averaging fit when larger — less sensitive to noise on any single inlier, at the cost of needing
    more iterations to draw an all-inlier sample). Inliers are points whose reprojection error is
    ``<= reproj`` px; the best consensus set wins and the model is refit on it. The iteration count
    adapts to the running best inlier ratio (capped at ``max_iters``) so ``confidence`` keeps
    OpenCV's meaning. Deterministic (fixed seed) so the live preview is stable across re-runs.

    Returns ``(M, inliers)``: ``M`` is the 2×3 affine (``None`` if no 3+-point consensus is found),
    ``inliers`` a bool array aligned to the input rows. If there are too few points to separate
    signal from noise (``n <= sample_size``), every point is an inlier (nothing to clean).
    """
    return ransac_affine(
        src,
        dst,
        sample_size=sample_size,
        reproj=reproj,
        max_iters=max_iters,
        confidence=confidence,
    )


def fit_zone_affine(polygon, ref_pts, cur_pts, reproj=3.0, sample_size=3, max_iters=2000,
                    confidence=0.99, valid=None):
    """RANSAC affine fit for the points inside ``polygon`` (used by the cleaning dialog).

    Uses a custom RANSAC (:func:`_ransac_affine`) with ``sample_size`` points per hypothesis fit
    (3 = minimal exact sample, more = least-squares averaging). Returns ``(local_idx, M, inliers)``
    where ``M`` is the 2×3 affine (or None) and ``inliers`` is a bool array aligned to ``local_idx``.
    Returns ``None`` if the zone holds too few points. ``valid`` (optional) excludes lost tracks
    before RANSAC, as in :func:`fit_zone_deformation`.
    """
    inside = points_in_polygon(polygon, ref_pts)
    if valid is not None:
        inside = inside & np.asarray(valid, dtype=bool)
    local_idx = np.where(inside)[0]
    if local_idx.size < MIN_ZONE_POINTS:
        return None
    src = ref_pts[local_idx].astype(np.float64)
    dst = cur_pts[local_idx].astype(np.float64)
    M, inliers = _ransac_affine(
        src, dst, sample_size=sample_size, reproj=reproj,
        max_iters=max_iters, confidence=confidence,
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


# Table columns
COL_ZONE, COL_COLOR, COL_PTS, COL_L1, COL_L2, COL_V1, COL_V2 = range(7)


class AffineZonesWindow(QWidget):
    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Affine Zone Tool")
        self.setMinimumWidth(640)
        self.zones = []  # list of Zone
        self._tool = None
        self._ransac_dialog = None
        self._ransac_preview = None  # (zone_idx, local_idx, inliers) while previewing
        self._plot_window = None
        self._gauge_window = None
        self._fits_cache = None  # per-zone fits for the current state; invalidated in _refresh

        self.new_btn = QPushButton("New Zone")
        self.finish_btn = QPushButton("Finish Zone")
        self.all_points_btn = QPushButton("All Points Zone")
        self.clear_btn = QPushButton("Clear Zones")
        self.ransac_btn = QPushButton("RANSAC…")
        self.plot_btn = QPushButton("Plot Curves")
        self.gauge_btn = QPushButton("Direction Gauge")
        self.export_btn = QPushButton("Export CSV…")
        self.new_btn.clicked.connect(self._start_zone)
        self.finish_btn.clicked.connect(self._finish_zone)
        self.all_points_btn.clicked.connect(self._add_all_points_zone)
        self.clear_btn.clicked.connect(self._clear_zones)
        self.ransac_btn.clicked.connect(self._open_ransac)
        self.plot_btn.clicked.connect(self._open_plot)
        self.gauge_btn.clicked.connect(self._open_gauge)
        self.export_btn.clicked.connect(self._export)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Zone", "Color", "Pts", "λ1", "λ2", "v1 (x,y)", "v2 (x,y)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.include_alignment = QCheckBox("Include alignment affine maps if present")
        self.include_alignment.setChecked(ctx.get_settings().get("include_alignment_affines", True) is not False)
        self.include_alignment.setToolTip(
            "Restore alignment scale/shear in all reported stretches and directions. "
            "Alignment translation and rigid rotation remain excluded; tracking and RANSAC use aligned pixels.")
        self.include_alignment.toggled.connect(self._alignment_toggled)

        top = QHBoxLayout()
        for b in (self.new_btn, self.finish_btn, self.all_points_btn, self.clear_btn):
            top.addWidget(b)
        analysis = QHBoxLayout()
        for b in (self.ransac_btn, self.plot_btn, self.gauge_btn, self.export_btn):
            analysis.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(analysis)
        layout.addWidget(self.include_alignment)
        layout.addWidget(self.hint)
        layout.addWidget(self.table)

        self._reference_owner = ctx.reference_index
        ctx.subscribe(ctx.signals.sequence_changed, self._on_sequence_changed)
        ctx.subscribe(ctx.signals.range_changed, self._on_range_changed)
        ctx.signals.frame_changed.connect(self._refresh)
        ctx.signals.result_changed.connect(self._refresh)
        ctx.signals.mask_changed.connect(self._refresh)
        self._refresh()  # also refreshes button-enable state (see _refresh's tail)

    # ---- lifecycle ------------------------------------------------------
    def showEvent(self, event):
        # Re-register the overlay on every show (not just construction): closeEvent removes
        # it, and the window is reused — relaunching only calls show(), so __init__ won't run
        # again. add_overlay is idempotent, so the initial show is harmless.
        super().showEvent(event)
        self.ctx.add_overlay(self._paint)
        self.ctx.request_redraw()

    def closeEvent(self, event):
        if self._tool is not None:
            self.ctx.end_canvas_interaction()
            self._tool = None
        if self._ransac_dialog is not None:
            self._ransac_dialog.close()
            self._ransac_dialog = None
        if self._plot_window is not None:
            self._plot_window.close()
            self._plot_window = None
        if self._gauge_window is not None:
            self._gauge_window.close()
            self._gauge_window = None
        self._ransac_preview = None
        self.ctx.remove_overlay(self._paint)
        super().closeEvent(event)

    def dispose(self):
        """Final signal teardown for plugin reload; ordinary closes keep the cached view current."""
        for signal, slot in (
            (self.ctx.signals.frame_changed, self._refresh),
            (self.ctx.signals.result_changed, self._refresh),
            (self.ctx.signals.mask_changed, self._refresh),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _on_sequence_changed(self):
        self.ctx.end_canvas_interaction()
        self._tool = None
        self._reference_owner = self.ctx.reference_index
        self._clear_zones()

    def _on_range_changed(self):
        if self._reference_owner != self.ctx.reference_index:
            self._on_sequence_changed()
        else:
            self._refresh()

    # ---- drawing zones --------------------------------------------------
    def _start_zone(self):
        if not self.ctx.has_sequence:
            return
        self.ctx.set_current_frame(self.ctx.reference_index)
        self._tool = ZoneDrawTool(self)
        self.ctx.begin_canvas_interaction(self._tool)
        self.ctx.status("Click to add zone vertices, then press “Finish Zone”.")
        self._update_buttons()

    def on_draw_progress(self):
        self.ctx.request_redraw()

    def _finish_zone(self):
        if self._tool is not None and len(self._tool.vertices) >= MIN_ZONE_POINTS:
            self.zones.append(Zone(self._tool.vertices, default_zone_color(len(self.zones))))
        self.ctx.end_canvas_interaction()
        self._tool = None
        self._refresh()  # rebuilds the table and refreshes button-enable state

    def _add_all_points_zone(self):
        """Create one zone — the convex hull of all active reference-frame points — enclosing
        every tracked point, then refresh like any other zone add."""
        coords = self.ctx.coords(active_only=True)
        if coords is None or coords.shape[1] < MIN_ZONE_POINTS:
            self.ctx.status("Need at least 3 tracked points to make an all-points zone.")
            return
        ref_pts = coords[0].astype(np.float32)
        hull = cv2.convexHull(ref_pts).reshape(-1, 2)
        if len(hull) < MIN_ZONE_POINTS:  # collinear points -> degenerate hull
            self.ctx.status("Tracked points are collinear — can't form an all-points zone.")
            return
        polygon = [(float(x), float(y)) for x, y in hull]
        self.zones.append(Zone(polygon, default_zone_color(len(self.zones))))
        self._refresh()

    def _clear_zones(self):
        self.zones.clear()
        self._ransac_preview = None
        if self._ransac_dialog is not None:
            self._ransac_dialog.close()
        self._refresh()
        self._refresh_plot()

    def _update_buttons(self):
        drawing = self._tool is not None
        self.new_btn.setEnabled(not drawing and self.ctx.has_sequence)
        self.finish_btn.setEnabled(drawing)
        self.all_points_btn.setEnabled(not drawing and self.ctx.has_result)
        has = bool(self.zones)
        self.clear_btn.setEnabled(has)
        self.export_btn.setEnabled(has and self.ctx.has_result)
        self.plot_btn.setEnabled(has and self.ctx.has_result)
        self.gauge_btn.setEnabled(has and self.ctx.has_result)
        self.ransac_btn.setEnabled(
            not drawing and self.ctx.has_result and self.table.currentRow() >= 0
        )

    # ---- compute --------------------------------------------------------
    def _alignment_toggled(self):
        settings = self.ctx.get_settings()
        settings["include_alignment_affines"] = self.include_alignment.isChecked()
        self.ctx.save_settings(settings)
        self._refresh()

    def correction_at(self, cut):
        if not self.include_alignment.isChecked():
            return np.eye(2)
        return self.ctx.alignment_affine(self.ctx.cut_to_global(cut))

    def fit_zone(self, zone, ref_pts, cur_pts, cut, valid=None):
        """Shared reported (indices, F) for tables, plots, export and gauge.

        Membership/validity are determined before restoring deformation. Translation is
        deliberately omitted; RANSAC continues to use the uncorrected pixel-space fit.
        """
        fit = fit_zone_deformation(zone.polygon, ref_pts, cur_pts, valid=valid)
        if fit is None:
            return None
        indices, F, _b = fit
        if self.include_alignment.isChecked():
            F = compose_alignment_affines(F, self.correction_at(cut), self.correction_at(0))
        if not np.isfinite(F).all() or np.linalg.det(F) <= 0 or np.linalg.cond(F) > 1e8:
            return None
        return indices, F

    def analysis_label(self):
        return ("Alignment scale/shear included; axes in restored, rotation-aligned coordinates."
                if self.include_alignment.isChecked() else "Residual deformation in aligned image coordinates.")

    def _valid_at(self, cut):
        """Per-active-point bool mask: True where the track is valid at frame ``cut`` (LK didn't
        lose it). ``None`` if no result, which the fit functions treat as 'all points valid'."""
        status = self.ctx.track_status(active_only=True)
        if status is None or cut is None:
            return None
        return status[cut] == 1

    def _get_fits(self):
        """Per-zone fits for the current frame, cached so a refresh and the paint it triggers
        (plus incidental repaints from zoom/pan) don't each redo the least-squares fits. The
        cache is invalidated in _refresh, which every state-change signal funnels through."""
        if self._fits_cache is None:
            self._fits_cache = self._compute_fits()
        return self._fits_cache

    def _compute_fits(self):
        """Return a list aligned to self.zones of (local_idx, F) or None per zone."""
        ctx = self.ctx
        cut = ctx.current_cut
        coords = ctx.coords(active_only=True)
        if cut is None or coords is None or coords.shape[1] == 0:
            return [None] * len(self.zones)
        ref_pts, cur_pts = coords[0], coords[cut]
        valid = self._valid_at(cut)
        out = []
        for zone in self.zones:
            out.append(self.fit_zone(zone, ref_pts, cur_pts, cut, valid=valid))
        return out

    def _refresh(self):
        self._fits_cache = None  # state changed: recompute fits once, then reuse in _paint
        fits = self._get_fits()
        self.table.setRowCount(len(self.zones))
        for z, fit in enumerate(fits):
            self._set_cell(z, COL_ZONE, str(z + 1))
            if self.table.cellWidget(z, COL_COLOR) is None:
                self._install_color_button(z)  # only build it once per row, not every refresh
            if fit is None:
                for col in (COL_PTS, COL_L1, COL_L2, COL_V1, COL_V2):
                    self._set_cell(z, col, "—")
                continue
            local_idx, F = fit
            lam1, lam2, v1, v2 = principal_stretches(F)
            self._set_cell(z, COL_PTS, str(local_idx.size))
            self._set_cell(z, COL_L1, f"{lam1:.4f}")
            self._set_cell(z, COL_L2, f"{lam2:.4f}")
            self._set_cell(z, COL_V1, f"{v1[0]:.3f}, {v1[1]:.3f}")
            self._set_cell(z, COL_V2, f"{v2[0]:.3f}, {v2[1]:.3f}")
        if not self.ctx.has_result:
            self.hint.setText("No tracking result yet — run tracking first.")
        elif self.ctx.current_cut == 0:
            self.hint.setText("On the reference frame F = I (λ1 = λ2 = 1); move the slider.")
        else:
            self.hint.setText(
                f"{len(self.zones)} zone(s). Stretches map reference → current frame. {self.analysis_label()}"
            )
        self._update_buttons()
        if self._gauge_window is not None and self._gauge_window.isVisible():
            self._gauge_window.refresh()  # propagate zone add/clear/recolor + RANSAC edits
        self._refresh_plot()
        self.ctx.request_redraw()

    def _set_cell(self, row, col, text):
        self.table.setItem(row, col, QTableWidgetItem(text))

    def _install_color_button(self, row):
        btn = QPushButton()
        btn.setFixedHeight(20)
        self._style_color_button(btn, self.zones[row].color)
        btn.clicked.connect(lambda _=False, r=row: self._pick_color(r))
        self.table.setCellWidget(row, COL_COLOR, btn)

    @staticmethod
    def _style_color_button(btn, color):
        btn.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #888; border-radius: 3px;"
        )

    def _pick_color(self, row):
        if not (0 <= row < len(self.zones)):
            return
        color = QColorDialog.getColor(self.zones[row].color, self, "Zone color")
        if color.isValid():
            self.zones[row].color = color
            self._style_color_button(self.table.cellWidget(row, COL_COLOR), color)
            self.ctx.request_redraw()
            self._refresh_plot()

    # ---- RANSAC ---------------------------------------------------------
    def selected_zone_index(self):
        row = self.table.currentRow()
        return row if 0 <= row < len(self.zones) else None

    def _open_ransac(self):
        if self.selected_zone_index() is None:
            return
        if self._ransac_dialog is None:
            self._ransac_dialog = RansacDialog(self)
        self._ransac_dialog.sync_frame_range()  # set range + jump to last frame
        self._ransac_dialog.show()
        self._ransac_dialog.raise_()
        self._ransac_dialog.preview()

    def set_ransac_preview(self, preview):
        self._ransac_preview = preview
        self.ctx.request_redraw()

    def on_ransac_closed(self):
        self._ransac_dialog = None
        self._ransac_preview = None
        self.ctx.request_redraw()

    # ---- plotting -------------------------------------------------------
    def _open_plot(self):
        if not (self.zones and self.ctx.has_result):
            return
        if self._plot_window is None:
            try:
                mpl = _load_matplotlib()  # before constructing the widget, so nothing leaks
            except Exception:
                QMessageBox.critical(
                    self, "matplotlib unavailable",
                    "matplotlib with a working Qt backend is required for plotting. "
                    "Run `uv sync` and try again.",
                )
                return
            self._plot_window = StretchPlotWindow(self, mpl)
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.replot()

    def _refresh_plot(self):
        if self._plot_window is not None and self._plot_window.isVisible():
            self._plot_window.replot()

    def on_plot_closed(self):
        self._plot_window = None

    # ---- direction gauge ------------------------------------------------
    def _open_gauge(self):
        if not (self.zones and self.ctx.has_result):
            return
        if self._gauge_window is None:
            from .gauge import ZoneDirectionGaugeWindow
            self._gauge_window = ZoneDirectionGaugeWindow(self)
        self._gauge_window.show()
        self._gauge_window.raise_()
        self._gauge_window.refresh()

    def on_gauge_closed(self):
        self._gauge_window = None

    # ---- export ---------------------------------------------------------
    def _export(self):
        corrected = self.ctx.has_sequence and self.include_alignment.isChecked() and any(
            not np.array_equal(self.correction_at(cut), np.eye(2)) for cut in range(self.ctx.frame_count))
        filename = "zone_stretches_with_alignment.csv" if corrected else "zone_stretches.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export per-frame stretches — " + self.analysis_label(), filename, "CSV (*.csv)"
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
        status = self.ctx.track_status(active_only=True)
        try:
            with atomic_open(path, newline="") as f:
                w = csv.writer(f)
                w.writerow(["zone", "frame_global", "n_points",
                            "lambda1", "lambda2", "v1x", "v1y", "v2x", "v2y"])
                for cut in range(self.ctx.frame_count):
                    cur_pts = coords[cut]
                    valid = None if status is None else (status[cut] == 1)
                    for z, zone in enumerate(self.zones):
                        fit = self.fit_zone(zone, ref_pts, cur_pts, cut, valid=valid)
                        if fit is None:
                            continue
                        local_idx, F = fit
                        lam1, lam2, v1, v2 = principal_stretches(F)
                        w.writerow([z + 1, ref_global + cut, local_idx.size,
                                    f"{lam1:.6f}", f"{lam2:.6f}",
                                    f"{v1[0]:.6f}", f"{v1[1]:.6f}",
                                    f"{v2[0]:.6f}", f"{v2[1]:.6f}"])
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.ctx.status(f"Exported zone stretches to {path}")

    # ---- overlay --------------------------------------------------------
    def _paint(self, painter, ctx):
        fits = self._get_fits()
        coords = ctx.coords(active_only=True)
        cut = ctx.current_cut
        cur_pts = coords[cut] if (coords is not None and cut is not None) else None
        for z, (zone, fit) in enumerate(zip(self.zones, fits)):
            color = zone.color
            screen = QPolygonF([ctx.image_to_screen(x, y) for x, y in zone.polygon])
            painter.setPen(QPen(color, 2))
            fill = QColor(color)
            fill.setAlpha(30)
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(screen)
            if fit is None or cur_pts is None:
                continue
            local_idx, _F = fit
            # RANSAC outliers (red) for the zone being previewed; everything else in zone color.
            outlier_set = set()
            if self._ransac_preview is not None and self._ransac_preview[0] == z:
                _zi, prev_idx, prev_inliers = self._ransac_preview
                outlier_set = set(int(i) for i in prev_idx[~prev_inliers])
            for j in local_idx:
                pt_color = OUTLIER_COLOR if int(j) in outlier_set else color
                painter.setPen(QPen(pt_color, 1))
                painter.setBrush(QBrush(pt_color))
                painter.drawEllipse(ctx.image_to_screen(*cur_pts[j]), 3, 3)

        # in-progress polygon being drawn
        if self._tool is not None and self._tool.vertices:
            pts = [ctx.image_to_screen(x, y) for x, y in self._tool.vertices]
            painter.setPen(QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if len(pts) >= 2:
                painter.drawPolyline(QPolygonF(pts))
            for p in pts:
                painter.drawEllipse(p, 4, 4)


class RansacDialog(QDialog):
    """Tune RANSAC parameters and clean the selected zone's points (current frame)."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.setWindowTitle("RANSAC zone cleaning")
        self.setWindowFlags(Qt.WindowType.Window)

        self.frame_spin = QSpinBox()  # range/value set per-open by sync_frame_range()
        self.sample_size = QSpinBox()
        self.sample_size.setRange(MIN_ZONE_POINTS, 50)
        self.sample_size.setValue(6)
        self.reproj = QDoubleSpinBox()
        self.reproj.setRange(0.1, 50.0)
        self.reproj.setSingleStep(0.5)
        self.reproj.setValue(3.0)
        self.reproj.setDecimals(2)
        self.max_iters = QSpinBox()
        self.max_iters.setRange(10, 100000)
        self.max_iters.setValue(2000)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.50, 0.99999)
        self.confidence.setDecimals(5)
        self.confidence.setSingleStep(0.001)
        self.confidence.setValue(0.99)

        form = QFormLayout()
        form.addRow("Frame", self.frame_spin)
        form.addRow("Points per fit", self.sample_size)
        form.addRow("Reproj threshold (px)", self.reproj)
        form.addRow("Max iterations", self.max_iters)
        form.addRow("Confidence", self.confidence)

        self.count_label = QLabel("—")
        self.apply_btn = QPushButton("Apply Cleaning")
        self.apply_btn.clicked.connect(self._apply)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.count_label)
        layout.addWidget(self.apply_btn)
        layout.addWidget(buttons)

        for w in (self.reproj, self.confidence):
            w.valueChanged.connect(self.preview)
        self.max_iters.valueChanged.connect(self.preview)
        self.sample_size.valueChanged.connect(self.preview)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        self.ctx.signals.frame_changed.connect(self._on_frame_changed)
        self.ctx.signals.mask_changed.connect(self.preview)
        # Follow the zone table: clicking another row re-previews that zone with the current
        # parameters, so the user can tune once and run through all zones without reopening.
        self.owner.table.itemSelectionChanged.connect(self.preview)

    def _fit(self):
        z = self.owner.selected_zone_index()
        coords = self.ctx.coords(active_only=True)
        cut = self.ctx.current_cut
        if z is None or coords is None or cut is None or coords.shape[1] == 0:
            return None, None
        fit = fit_zone_affine(
            self.owner.zones[z].polygon, coords[0], coords[cut],
            reproj=self.reproj.value(), sample_size=self.sample_size.value(),
            max_iters=self.max_iters.value(),
            confidence=self.confidence.value(), valid=self.owner._valid_at(cut),
        )
        return z, fit

    def _on_frame_spin(self, value):
        self.ctx.set_current_frame(value)  # → frame_changed → _on_frame_changed → preview

    def _on_frame_changed(self):
        self._sync_frame_spin()
        self.preview()

    def _sync_frame_spin(self):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self.ctx.current_index)  # QSpinBox clamps to its range
        self.frame_spin.blockSignals(False)

    def sync_frame_range(self):
        """Set the spinbox to the tracked range and jump to the last frame. Called on every open."""
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(self.ctx.reference_index, self.ctx.last_index)
        self.frame_spin.blockSignals(False)
        self.ctx.set_current_frame(self.ctx.last_index)  # drives main slider + emits frame_changed

    def preview(self):
        z, fit = self._fit()
        if fit is None or fit[1] is None:
            self.count_label.setText("Selected zone has no valid fit at this frame.")
            self.apply_btn.setEnabled(False)
            self.owner.set_ransac_preview(None)
            return
        local_idx, _M, inliers = fit
        n_out = int((~inliers).sum())
        self.count_label.setText(
            f"Zone {z + 1}: {int(inliers.sum())} inliers / {n_out} outliers (red)."
        )
        self.apply_btn.setEnabled(n_out > 0)
        self.owner.set_ransac_preview((z, local_idx, inliers))

    def _apply(self):
        z, fit = self._fit()
        if fit is None or fit[1] is None:
            return
        local_idx, _M, inliers = fit
        keep = np.ones(self.ctx.n_active, dtype=bool)
        keep[local_idx[~inliers]] = False
        dropped = int((~inliers).sum())
        if dropped == 0:
            return
        self.ctx.apply_keep_mask(keep)
        self.ctx.status(f"Zone {z + 1}: dropped {dropped} outlier(s). Undo in the Cleanup dialog.")
        self.preview()

    def closeEvent(self, event):
        try:
            self.ctx.signals.frame_changed.disconnect(self._on_frame_changed)
            self.ctx.signals.mask_changed.disconnect(self.preview)
            self.owner.table.itemSelectionChanged.disconnect(self.preview)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_ransac_closed()
        super().closeEvent(event)


def _load_matplotlib():
    """Lazily import the matplotlib Qt backend, returning ``(FigureCanvasQTAgg, Figure)``.

    Deferred (not module-level) so the plugin's mere import doesn't pull in matplotlib at app
    startup. Raises on *any* failure — a missing package (ImportError) or a backend that fails to
    initialize (e.g. a Qt-binding mismatch, which raises non-ImportError) — so the caller can show
    a message and skip constructing the plot widget entirely.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    return FigureCanvasQTAgg, NavigationToolbar2QT, Figure


class StretchPlotWindow(QWidget):
    """Embedded matplotlib plot of λ1 (solid) / λ2 (dashed) over frames, per zone color."""

    def __init__(self, owner, mpl):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Principal stretches over frames")
        self.resize(720, 480)

        FigureCanvasQTAgg, NavigationToolbar2QT, Figure = mpl

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.replot)

        layout = QVBoxLayout(self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)

        self.ctx.signals.result_changed.connect(self.replot)
        self.ctx.signals.mask_changed.connect(self.replot)

    def replot(self):
        self.ax.clear()
        coords = self.ctx.coords(active_only=True)
        if coords is not None and coords.shape[1] > 0:
            ref_pts = coords[0]
            status = self.ctx.track_status(active_only=True)
            frames = np.array([self.ctx.cut_to_global(t) for t in range(self.ctx.frame_count)])
            for z, zone in enumerate(self.owner.zones):
                lam1 = np.full(self.ctx.frame_count, np.nan)
                lam2 = np.full(self.ctx.frame_count, np.nan)
                for t in range(self.ctx.frame_count):
                    valid = None if status is None else (status[t] == 1)
                    fit = self.owner.fit_zone(zone, ref_pts, coords[t], t, valid=valid)
                    if fit is None:
                        continue
                    lam1[t], lam2[t], _v1, _v2 = principal_stretches(fit[1])
                rgb = zone.color.getRgbF()[:3]
                self.ax.plot(frames, lam1, "-", color=rgb, label=f"Zone {z + 1} λ1")
                self.ax.plot(frames, lam2, "--", color=rgb, label=f"Zone {z + 1} λ2")
        self.ax.set_xlabel("frame (global index)")
        self.ax.set_ylabel("principal stretch λ")
        self.ax.set_title("Including alignment scale/shear" if self.owner.include_alignment.isChecked()
                          else "Residual deformation", fontsize="small")
        self.ax.grid(True, alpha=0.3)
        if self.owner.zones:
            self.ax.legend(fontsize="small", ncol=2)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            self.ctx.signals.result_changed.disconnect(self.replot)
            self.ctx.signals.mask_changed.disconnect(self.replot)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_plot_closed()
        super().closeEvent(event)
