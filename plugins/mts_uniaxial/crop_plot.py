"""Embedded matplotlib widget: force-vs-time (top) and force-vs-displacement (bottom).

Used by the Crop section to visualise the experiment window. The cropped span is drawn in the
accent colour over the full (greyed) curve, and the matched image frames are scattered on both
axes so the user sees where images fall. matplotlib is imported lazily (and broadly guarded) so
the plugin's mere import never pulls it in at app startup; if it's unavailable the widget shows a
fallback label and the crop sliders still work.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

_ACCENT = "#2563eb"
_IMG = "#dc2626"
_REF = "#16a34a"  # reference-frame marker (distinct from the crop accent and the image red)


def _load_matplotlib():
    """Import the matplotlib Qt5 backend, returning ``(FigureCanvas, Figure)``. Raises on failure."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    return FigureCanvasQTAgg, Figure


def matplotlib_available() -> bool:
    try:
        _load_matplotlib()
        return True
    except Exception:
        return False


class CropPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.available = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            FigureCanvas, Figure = _load_matplotlib()
        except Exception:
            layout.addWidget(QLabel("matplotlib is not available — crop plot disabled.\n"
                                    "The crop sliders still define the experiment window."))
            return
        self.available = True
        self.figure = Figure(figsize=(5.0, 4.0), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_t = self.figure.add_subplot(2, 1, 1)
        self.ax_d = self.figure.add_subplot(2, 1, 2)
        self.canvas.setMinimumHeight(320)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        if not self.available:
            return
        self.ax_t.clear()
        self.ax_d.clear()
        self.canvas.draw_idle()

    def update_data(self, t_ms, disp, force, lo, hi,
                    img_t=None, img_disp=None, img_force=None,
                    show_images=True, ref=None) -> None:
        """Redraw both subplots. ``lo``/``hi`` are inclusive sensor indices of the crop window.

        ``show_images`` toggles the per-frame image overlay; ``ref`` is the reference image index,
        marked at its image-interpolated position (so it lands on the image-frame scatter).
        """
        if not self.available:
            return
        self.ax_t.clear()
        self.ax_d.clear()
        n = int(t_ms.size)
        lo = max(0, min(int(lo), n - 1))
        hi = max(lo, min(int(hi), n - 1))

        self.ax_t.plot(t_ms, force, color="0.8", lw=1.0)
        self.ax_t.plot(t_ms[lo:hi + 1], force[lo:hi + 1], color=_ACCENT, lw=1.6)
        self.ax_d.plot(disp, force, color="0.8", lw=1.0)
        self.ax_d.plot(disp[lo:hi + 1], force[lo:hi + 1], color=_ACCENT, lw=1.6)

        # Overlay only the image frames that fall within the sensor's time coverage, so a long
        # post-sensor image tail doesn't squash the curve.
        if show_images and img_t is not None and img_force is not None:
            m = (img_t >= float(t_ms.min())) & (img_t <= float(t_ms.max()))
            if np.any(m):
                self.ax_t.scatter(img_t[m], img_force[m], s=14, c=_IMG, zorder=3, label="images")
                if img_disp is not None:
                    self.ax_d.scatter(img_disp[m], img_force[m], s=14, c=_IMG, zorder=3)
                self.ax_t.legend(fontsize="small", loc="best")

        if ref is not None and img_t is not None and img_force is not None and 0 <= ref < img_t.size:
            self.ax_t.scatter([img_t[ref]], [img_force[ref]], s=70, marker="*", c=_REF, zorder=5)
            if img_disp is not None:
                self.ax_d.scatter([img_disp[ref]], [img_force[ref]], s=70, marker="*", c=_REF, zorder=5)

        self.ax_t.set_xlabel("time (ms)")
        self.ax_t.set_ylabel("force (N)")
        self.ax_d.set_xlabel("displacement (mm)")
        self.ax_d.set_ylabel("force (N)")
        for ax in (self.ax_t, self.ax_d):
            ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


class ReferencePlotWidget(QWidget):
    """Single-axis force-vs-displacement plot over the reference-search sub-window.

    Mirrors :class:`CropPlotWidget`'s lazy-matplotlib handling: if the backend is unavailable the
    widget shows a fallback label and ``available`` stays ``False``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            FigureCanvas, Figure = _load_matplotlib()
        except Exception:
            layout.addWidget(QLabel("matplotlib is not available — reference plot disabled."))
            return
        self.available = True
        self.figure = Figure(figsize=(5.0, 2.4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.canvas.setMinimumHeight(180)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        if not self.available:
            return
        self.ax.clear()
        self.canvas.draw_idle()

    def update_data(self, disp, force, lo, hi, ref_disp=None, ref_force=None) -> None:
        """Plot force-vs-displacement over the sub-window ``[lo, hi]`` (inclusive sensor indices)."""
        if not self.available:
            return
        self.ax.clear()
        n = int(disp.size)
        lo = max(0, min(int(lo), n - 1))
        hi = max(lo, min(int(hi), n - 1))
        self.ax.plot(disp[lo:hi + 1], force[lo:hi + 1], color=_ACCENT, lw=1.6)
        if ref_disp is not None and ref_force is not None:
            self.ax.scatter([ref_disp], [ref_force], s=70, marker="*", c=_REF, zorder=5,
                            label="reference")
            self.ax.legend(fontsize="small", loc="best")
        self.ax.set_xlabel("displacement (mm)")
        self.ax.set_ylabel("force (N)")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
