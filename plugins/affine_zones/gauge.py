"""Direction gauge: image sequence with its own frame slider and per-zone principal-strain axes.

A standalone window (adapted from ``mts_uniaxial/gauge.py``) that shows the tracked frames with an
**independent** local slider (unrelated to the app's global current frame) and overlays, at the
centroid of **every** drawn zone, the two principal-direction axes — the eigenvectors of
``B = F Fᵀ`` in the current (deformed) configuration — so the user can read off which way the
principal strains point in each zone at any frame. Each axis length is scaled by its principal
stretch (``λ₁``/``λ₂``) so the arrows grow in tension and shrink in compression; the base length is
kept short so several zones don't crowd each other.

Like the mts_uniaxial gauge it does **not** use ``ctx.add_overlay`` (that overlay tracks the global
current frame); it renders the frame itself into a small widget with a plain fit-to-widget scale and
paints the axes on top. It computes each zone's deformation gradient live (reusing
``zones.fit_zone_deformation`` / ``zones.principal_stretches``) for the local slider frame.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .zones import principal_stretches

# Fraction of the displayed frame's short side used as the (half) arrow length. Kept short so the
# per-zone arrows don't overlap when several zones are close together.
_AXIS_LENGTH_FRAC = 0.07


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an ``(H, W, 3)`` uint8 RGB array to an independent (copied) QImage."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _ZoneGaugeCanvas(QWidget):
    """Paints the current frame fit-to-widget plus, per zone, two principal axes at its centroid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self._img = None        # QImage of the current frame
        self._entries = []      # list of (centroid_xy, v1, v2, lam1, lam2, QColor)

    def set_frame(self, img, entries):
        self._img = img
        self._entries = entries or []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self._img is None:
            return
        iw, ih = self._img.width(), self._img.height()
        if iw == 0 or ih == 0:
            return
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox, oy = (self.width() - dw) / 2.0, (self.height() - dh) / 2.0
        painter.drawImage(QRectF(ox, oy, dw, dh), self._img)
        if not self._entries:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Base length = the undeformed (λ = 1) axis; scale each axis by its principal stretch so the
        # arrows grow in tension (λ > 1) and shrink in compression (λ < 1) live with the frame.
        base = _AXIS_LENGTH_FRAC * min(dw, dh)
        for centroid, v1, v2, lam1, lam2, color in self._entries:
            c = QPointF(ox + centroid[0] * scale, oy + centroid[1] * scale)
            qc = QColor(color)
            self._draw_axis(painter, c, v1, base * lam1, qc, dashed=False)  # major (solid)
            self._draw_axis(painter, c, v2, base * lam2, qc, dashed=True)   # minor (dashed)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(c, 2.5, 2.5)

    def _draw_axis(self, painter, c, v, length, color, dashed):
        vx, vy = float(v[0]), float(v[1])
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1e-9:
            return
        vx, vy = vx / norm, vy / norm
        p1 = QPointF(c.x() - vx * length, c.y() - vy * length)
        p2 = QPointF(c.x() + vx * length, c.y() + vy * length)
        pen = QPen(color, 1.5 if dashed else 2.5)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))
        painter.drawLine(p1, p2)
        self._arrow_head(painter, p2, vx, vy, color)
        self._arrow_head(painter, p1, -vx, -vy, color)

    @staticmethod
    def _arrow_head(painter, tip, vx, vy, color):
        size = 7.0
        px, py = -vy, vx  # perpendicular
        bx, by = tip.x() - vx * size, tip.y() - vy * size
        left = QPointF(bx + px * size * 0.5, by + py * size * 0.5)
        right = QPointF(bx - px * size * 0.5, by - py * size * 0.5)
        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([tip, left, right]))


class ZoneDirectionGaugeWindow(QWidget):
    """Frame viewer with a local slider and a per-zone principal-axis overlay."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Direction gauge — per-zone principal strain axes")
        self.resize(560, 600)

        self.canvas = _ZoneGaugeCanvas()
        self.info = QLabel("—")
        self.info.setWordWrap(True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda *_: self._update_display())

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        legend = QLabel(
            "Per zone, in the zone color: "
            "<b>solid</b> = major axis (λ₁, tensile)   "
            "<b>dashed</b> = minor axis (λ₂, lateral). "
            "Axis length ∝ stretch (grows in tension, shrinks in compression)."
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        layout.addWidget(self.info)
        row = QHBoxLayout()
        row.addWidget(QLabel("Frame:"))
        row.addWidget(self.slider, 1)
        layout.addLayout(row)

        self.ctx.signals.result_changed.connect(self.refresh)
        self.ctx.signals.mask_changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        """Re-clamp the slider to the tracked range and redraw (range/zones may have changed)."""
        n = self.ctx.frame_count
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        if self.slider.value() > max(0, n - 1):
            self.slider.setValue(max(0, n - 1))
        self.slider.blockSignals(False)
        self._update_display()

    def _update_display(self):
        if not self.ctx.has_result or not self.ctx.has_sequence or self.ctx.frame_count == 0:
            self.canvas.set_frame(None, None)
            self.info.setText("No tracking result.")
            return
        coords = self.ctx.coords(active_only=True)
        if coords is None or coords.shape[1] == 0:
            self.canvas.set_frame(None, None)
            self.info.setText("No active points.")
            return
        cut = max(0, min(self.slider.value(), self.ctx.frame_count - 1))
        g = self.ctx.cut_to_global(cut)
        try:
            qimg = _rgb_to_qimage(self.ctx.frame_rgb(g))
        except Exception:
            return
        ref_pts, cur_pts = coords[0], coords[cut]
        status = self.ctx.track_status(active_only=True)
        valid = None if status is None else (status[cut] == 1)
        entries = []
        for zone in self.owner.zones:
            fit = self.owner.fit_zone(zone, ref_pts, cur_pts, cut, valid=valid)
            if fit is None:
                continue
            local_idx, F = fit
            lam1, lam2, v1, v2 = principal_stretches(F)
            if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))
                    and np.isfinite(lam1) and np.isfinite(lam2)):
                continue
            pts = cur_pts[local_idx]
            centroid = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            # Reported axes belong to restored coordinates, but the background is aligned.
            # _draw_axis normalizes these mapped directions; lengths remain the reported λ.
            inverse = np.linalg.inv(self.owner.correction_at(cut))
            v1, v2 = inverse @ v1, inverse @ v2
            entries.append((centroid, v1, v2, lam1, lam2, zone.color))
        self.canvas.set_frame(qimg, entries)
        self.info.setText(
            f"frame {g} (cut {cut}/{self.ctx.frame_count - 1}) — "
            f"{len(entries)}/{len(self.owner.zones)} zone(s) with a valid fit. {self.owner.analysis_label()}"
        )

    def closeEvent(self, event):
        try:
            self.ctx.signals.result_changed.disconnect(self.refresh)
            self.ctx.signals.mask_changed.disconnect(self.refresh)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_gauge_closed()
        super().closeEvent(event)
