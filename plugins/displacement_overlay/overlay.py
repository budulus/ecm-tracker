"""Strain computation + the overlay painter and its control window."""
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# component key -> human label
COMPONENTS = [
    ("von_mises", "von Mises strain"),
    ("exx", "εxx (normal, x)"),
    ("eyy", "εyy (normal, y)"),
    ("exy", "εxy (shear)"),
]


def small_strain_per_triangle(ref_pts, cur_pts, simplices):
    """Return a (M,) array per triangle for each strain component.

    For each triangle, the affine deformation gradient F maps reference edge vectors to current
    edge vectors; the small-strain tensor is ε = ½(F + Fᵀ) − I. Degenerate triangles yield NaN.
    Returns a dict of component-key -> (M,) float array.
    """
    m = len(simplices)
    exx = np.full(m, np.nan)
    eyy = np.full(m, np.nan)
    exy = np.full(m, np.nan)
    for i, (a, b, c) in enumerate(simplices):
        dX = np.array([ref_pts[b] - ref_pts[a], ref_pts[c] - ref_pts[a]]).T  # (2,2) edge cols
        dx = np.array([cur_pts[b] - cur_pts[a], cur_pts[c] - cur_pts[a]]).T
        det = dX[0, 0] * dX[1, 1] - dX[0, 1] * dX[1, 0]
        if abs(det) < 1e-9:
            continue
        F = dx @ np.linalg.inv(dX)
        eps = 0.5 * (F + F.T) - np.eye(2)
        exx[i], eyy[i], exy[i] = eps[0, 0], eps[1, 1], eps[0, 1]
    vm = np.sqrt(exx ** 2 - exx * eyy + eyy ** 2 + 3 * exy ** 2)
    return {"exx": exx, "eyy": eyy, "exy": exy, "von_mises": vm}


def _values_to_colors(values, vmin, vmax, alpha):
    """Map a (M,) value array to a list of QColor via OpenCV's JET colormap (NaN -> None)."""
    span = max(vmax - vmin, 1e-9)
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET).reshape(-1, 3)
    colors = []
    for i, v in enumerate(values):
        if not np.isfinite(v):
            colors.append(None)
        else:
            b, g, r = (int(c) for c in bgr[i])
            colors.append(QColor(r, g, b, alpha))
    return colors


class StrainOverlayWindow(QWidget):
    """Controls for the overlay; also owns the overlay painter registered on the canvas."""

    def __init__(self, ctx):
        super().__init__(ctx.window)
        self.ctx = ctx
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Displacement / Strain Overlay")
        self.setMinimumWidth(320)
        self._cache = None  # (cut, component) -> (simplices, colors) for the current frame

        self.component = QComboBox()
        for key, label in COMPONENTS:
            self.component.addItem(label, key)
        self.show_field = QCheckBox("Show strain field")
        self.show_field.setChecked(True)
        self.show_vectors = QCheckBox("Show displacement vectors")
        self.show_vectors.setChecked(True)
        self.vector_scale = QDoubleSpinBox()
        self.vector_scale.setRange(0.1, 50.0)
        self.vector_scale.setValue(1.0)
        self.vector_scale.setSingleStep(0.5)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setValue(0.45)
        self.alpha.setSingleStep(0.05)
        self.range_label = QLabel("—")

        form = QFormLayout()
        form.addRow("Component:", self.component)
        form.addRow(self.show_field)
        form.addRow(self.show_vectors)
        form.addRow("Vector scale:", self.vector_scale)
        form.addRow("Field opacity:", self.alpha)
        form.addRow("Value range:", self.range_label)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Overlay follows the frame slider. Cut 0 (reference) is zero."))

        # Redraw on any control change; invalidate the cache when the data changes.
        for w in (self.component,):
            w.currentIndexChanged.connect(self._invalidate)
        for w in (self.show_field, self.show_vectors):
            w.toggled.connect(self._redraw)
        for w in (self.vector_scale, self.alpha):
            w.valueChanged.connect(self._redraw)
        ctx.signals.frame_changed.connect(self._redraw)
        ctx.signals.result_changed.connect(self._invalidate)
        ctx.signals.mask_changed.connect(self._invalidate)

        ctx.add_overlay(self._paint)
        self._redraw()

    # ---- overlay lifecycle ---------------------------------------------
    def closeEvent(self, event):
        self.ctx.remove_overlay(self._paint)
        super().closeEvent(event)

    def _invalidate(self):
        self._cache = None
        self._redraw()

    def _redraw(self):
        self.ctx.request_redraw()

    # ---- computation ----------------------------------------------------
    def _field_for_current(self):
        """Return (cur_pts, simplices, colors) for the current frame, or None."""
        ctx = self.ctx
        cut = ctx.current_cut
        if cut is None or not ctx.has_result:
            return None
        coords = ctx.coords(active_only=True)  # (frames, P, 2)
        if coords is None or coords.shape[1] < 3:
            return None
        ref_pts = coords[0]
        cur_pts = coords[cut]
        key = self.component.currentData()
        if self._cache is not None and self._cache[0] == (cut, key):
            return cur_pts, self._cache[1], self._cache[2]
        try:
            from scipy.spatial import Delaunay
            simplices = Delaunay(ref_pts).simplices
        except Exception:
            return None
        values = small_strain_per_triangle(ref_pts, cur_pts, simplices)[key]
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = float(np.nanmax(np.abs(finite)))
            vmin = -vmax if key in ("exx", "eyy", "exy") else 0.0
            self.range_label.setText(f"[{vmin:.4f}, {vmax:.4f}]")
        else:
            vmin, vmax = 0.0, 1.0
            self.range_label.setText("—")
        colors = _values_to_colors(values, vmin, vmax, int(self.alpha.value() * 255))
        self._cache = ((cut, key), simplices, colors)
        return cur_pts, simplices, colors

    # ---- painting -------------------------------------------------------
    def _paint(self, painter, ctx):
        field = self._field_for_current()
        if field is None:
            return
        cur_pts, simplices, colors = field

        if self.show_field.isChecked():
            painter.setPen(Qt.NoPen)
            for tri, color in zip(simplices, colors):
                if color is None:
                    continue
                poly = QPolygonF([ctx.image_to_screen(*cur_pts[v]) for v in tri])
                painter.setBrush(QBrush(color))
                painter.drawPolygon(poly)

        if self.show_vectors.isChecked():
            coords = ctx.coords(active_only=True)
            ref_pts = coords[0]
            scale = self.vector_scale.value()
            painter.setPen(QPen(QColor(255, 255, 0), 1.2))
            painter.setBrush(Qt.NoBrush)
            for j in range(cur_pts.shape[0]):
                disp = (cur_pts[j] - ref_pts[j]) * scale
                tip = ref_pts[j] + disp
                painter.drawLine(ctx.image_to_screen(*ref_pts[j]), ctx.image_to_screen(*tip))
