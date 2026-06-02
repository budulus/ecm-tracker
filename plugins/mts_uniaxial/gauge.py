"""Direction gauge: image sequence with its own frame slider and the principal-strain axes.

A standalone window that shows the tracked frames with an **independent** local slider (unrelated
to the app's global current frame) and overlays the two principal-direction axes — the eigenvectors
of ``B = F Fᵀ`` in the current (deformed) configuration — anchored at the active-points centroid, so
the user can read off which way the principal strains point at any frame. Each axis length is scaled
by its principal stretch (``λ₁``/``λ₂``), so the arrows grow in tension and shrink in compression.

It deliberately does **not** use ``ctx.add_overlay`` (that overlay tracks the global current frame);
instead it renders the frame itself into a small widget with a plain fit-to-widget scale and paints
the axes on top. It reads the owner's cached :class:`KinematicsSeries` (``v1``/``v2`` per frame) and
never recomputes the kinematics.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

_V1_COLOR = "#2563eb"  # major / tensile axis
_V2_COLOR = "#f59e0b"  # minor / lateral axis


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an ``(H, W, 3)`` uint8 RGB array to an independent (copied) QImage."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class _GaugeCanvas(QWidget):
    """Paints the current frame fit-to-widget plus the two principal axes through the centroid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self._img = None        # QImage of the current frame
        self._centroid = None   # (x, y) in image coords
        self._v1 = self._v2 = None
        self._lam1 = self._lam2 = 1.0  # principal stretches; scale the arrow lengths

    def set_frame(self, img, centroid, v1, v2, lam1=1.0, lam2=1.0):
        self._img = img
        self._centroid = centroid
        self._v1, self._v2 = v1, v2
        self._lam1, self._lam2 = lam1, lam2
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
        if self._centroid is None or self._v1 is None or self._v2 is None:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        c = QPointF(ox + self._centroid[0] * scale, oy + self._centroid[1] * scale)
        # Base length = the undeformed (λ = 1) axis; scale each axis by its principal stretch so the
        # arrows grow in tension (λ > 1) and shrink in compression (λ < 1) live with the frame.
        base = 0.20 * min(dw, dh)
        self._draw_axis(painter, c, self._v1, base * self._lam1, QColor(_V1_COLOR))
        self._draw_axis(painter, c, self._v2, base * self._lam2, QColor(_V2_COLOR))
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(c, 3.0, 3.0)

    def _draw_axis(self, painter, c, v, length, color):
        vx, vy = float(v[0]), float(v[1])
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1e-9:
            return
        vx, vy = vx / norm, vy / norm
        p1 = QPointF(c.x() - vx * length, c.y() - vy * length)
        p2 = QPointF(c.x() + vx * length, c.y() + vy * length)
        painter.setPen(QPen(color, 2.5))
        painter.setBrush(QBrush(color))
        painter.drawLine(p1, p2)
        self._arrow_head(painter, p2, vx, vy, color)
        self._arrow_head(painter, p1, -vx, -vy, color)

    @staticmethod
    def _arrow_head(painter, tip, vx, vy, color):
        size = 9.0
        px, py = -vy, vx  # perpendicular
        bx, by = tip.x() - vx * size, tip.y() - vy * size
        left = QPointF(bx + px * size * 0.5, by + py * size * 0.5)
        right = QPointF(bx - px * size * 0.5, by - py * size * 0.5)
        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([tip, left, right]))


class DirectionGaugeWindow(QWidget):
    """Frame viewer with a local slider and the principal-axis overlay."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Direction gauge — principal strain axes")
        self.resize(560, 600)

        self.canvas = _GaugeCanvas()
        self.info = QLabel("—")
        self.info.setWordWrap(True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda *_: self._update_display())

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        legend = QLabel(
            f"<span style='color:{_V1_COLOR}'>■</span> major axis (λ₁, tensile)   "
            f"<span style='color:{_V2_COLOR}'>■</span> minor axis (λ₂, lateral)"
        )
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
        """Re-clamp the slider to the tracked range and redraw (range may change with the result)."""
        n = self.ctx.frame_count
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        if self.slider.value() > max(0, n - 1):
            self.slider.setValue(max(0, n - 1))
        self.slider.blockSignals(False)
        self._update_display()

    def _update_display(self):
        series = self.owner.kinematics()
        if series is None or not self.ctx.has_sequence or self.ctx.frame_count == 0:
            self.canvas.set_frame(None, None, None, None)
            self.info.setText("No tracking result.")
            return
        cut = max(0, min(self.slider.value(), self.ctx.frame_count - 1))
        g = self.ctx.cut_to_global(cut)
        try:
            qimg = _rgb_to_qimage(self.ctx.frame_rgb(g))
        except Exception:
            return
        centroid = self._centroid_at(cut)
        v1, v2 = series.v1[cut], series.v2[cut]
        if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
            v1 = v2 = None
        l1, l2 = series.lambda_1[cut], series.lambda_2[cut]
        sl1 = l1 if np.isfinite(l1) else 1.0
        sl2 = l2 if np.isfinite(l2) else 1.0
        self.canvas.set_frame(qimg, centroid, v1, v2, sl1, sl2)
        self.info.setText(f"frame {g} (cut {cut}/{self.ctx.frame_count - 1}) — "
                          f"λ₁ = {l1:.4f}, λ₂ = {l2:.4f}")

    def _centroid_at(self, cut):
        coords = self.ctx.coords(active_only=True)
        if coords is None or coords.shape[1] == 0:
            return None
        pts = coords[cut]
        status = self.ctx.track_status(active_only=True)
        if status is not None:
            m = status[cut] == 1
            if m.any():
                pts = pts[m]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def closeEvent(self, event):
        try:
            self.ctx.signals.result_changed.disconnect(self.refresh)
            self.ctx.signals.mask_changed.disconnect(self.refresh)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_gauge_closed()
        super().closeEvent(event)
