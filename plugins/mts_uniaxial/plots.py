"""Standalone matplotlib windows for the kinematics / stress curves.

Each window embeds a ``FigureCanvasQTAgg`` in a top-level ``QWidget`` (never ``plt.show()``, which
would fight the app's Qt event loop) and pulls the owner's cached :class:`KinematicsSeries` at
replot time, so RANSAC apply/undo (``mask_changed``) and new results (``result_changed``) refresh
the plot reactively. matplotlib is imported lazily and broadly guarded — exactly like
``crop_plot.py`` — so the plugin's mere import never pulls it in at startup.

The owner (``MtsUniaxialWindow``) supplies the data through ``owner.kinematics()`` (the cached
series), ``owner.per_frame_force_N()`` (per-frame force in N, zeroed at the reference) and
``owner.pstate`` (the reference cross-section and the incompressible flag).
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

_ACCENT = "#2563eb"   # primary curve / nominal stress
_E1 = "#dc2626"       # ε₁ tensile (red)
_E2 = "#2563eb"       # ε₂ lateral (blue)
_ICO = "#16a34a"      # incompressible prediction (green, dashed)


def _load_matplotlib():
    """Import the matplotlib Qt5 backend, returning ``(FigureCanvas, Toolbar, Figure)``. Raises on
    failure so the caller can show a message and skip building the window."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    return FigureCanvasQTAgg, NavigationToolbar2QT, Figure


class KinematicsPlotWindow(QWidget):
    """Linear strains: ε₁/ε₂ over time (left) and ε₂ vs ε₁ (right), with the incompressible
    prediction overlaid on both when the material is flagged incompressible."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Kinematics — linear strains")
        self.resize(860, 460)

        FigureCanvas, Toolbar, Figure = _load_matplotlib()
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_t = self.figure.add_subplot(1, 2, 1)
        self.ax_c = self.figure.add_subplot(1, 2, 2)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.replot)

        layout = QVBoxLayout(self)
        layout.addWidget(Toolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)

        self.ctx.signals.result_changed.connect(self.replot)
        self.ctx.signals.mask_changed.connect(self.replot)

    def replot(self):
        self.ax_t.clear()
        self.ax_c.clear()
        series = self.owner.kinematics()
        if series is not None:
            t = series.time_s
            self.ax_t.plot(t, series.eps_1, "-", color=_E1, label=r"$\varepsilon_1$ (tensile)")
            self.ax_t.plot(t, series.eps_2, "-", color=_E2, label=r"$\varepsilon_2$ (lateral)")
            self.ax_c.plot(series.eps_1, series.eps_2, "-", color=_E2,
                           label=r"$\varepsilon_2$ measured")
            if self.owner.pstate.incompressible:
                self.ax_t.plot(t, series.eps_2_ico, "--", color=_ICO,
                               label=r"$\varepsilon_2$ incompressible")
                self.ax_c.plot(series.eps_1, series.eps_2_ico, "--", color=_ICO,
                               label=r"$\varepsilon_2$ incompressible")
        self.ax_t.set_xlabel("time (s)")
        self.ax_t.set_ylabel(r"linear strain $\varepsilon$")
        self.ax_c.set_xlabel(r"$\varepsilon_1$ (tensile)")
        self.ax_c.set_ylabel(r"$\varepsilon_2$ (lateral)")
        for ax in (self.ax_t, self.ax_c):
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small", loc="best")
        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            self.ctx.signals.result_changed.disconnect(self.replot)
            self.ctx.signals.mask_changed.disconnect(self.replot)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_plot_closed(self)
        super().closeEvent(event)


class StressPlotWindow(QWidget):
    """First Piola–Kirchhoff (``mode='pk'``) or Cauchy (``mode='cauchy'``) stress vs ε₁.

    One parameterized class — the two differ only in the y-array and label:
      • PK (nominal):  P = force / A₀                       (MPa)
      • Cauchy (true): σ = λ₁ · P   (incompressible only)   (MPa)
    """

    def __init__(self, owner, mode):
        super().__init__(owner)
        self.owner = owner
        self.ctx = owner.ctx
        self.mode = mode
        is_pk = mode == "pk"
        self._ylabel = "P (MPa)" if is_pk else r"$\sigma$ (MPa)"
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("First Piola–Kirchhoff stress" if is_pk else "Cauchy (true) stress")
        self.resize(640, 480)

        FigureCanvas, Toolbar, Figure = _load_matplotlib()
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.replot)

        layout = QVBoxLayout(self)
        layout.addWidget(Toolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)

        self.ctx.signals.result_changed.connect(self.replot)
        self.ctx.signals.mask_changed.connect(self.replot)

    def replot(self):
        self.ax.clear()
        series = self.owner.kinematics()
        force_N = self.owner.per_frame_force_N()
        a0 = self.owner.pstate.reference_area_mm2
        if series is not None and force_N is not None and a0 > 0:
            pk = force_N / a0  # N/mm² = MPa
            y = pk if self.mode == "pk" else series.lambda_1 * pk
            self.ax.plot(series.eps_1, y, "-", color=_ACCENT)
        self.ax.set_xlabel(r"$\varepsilon_1$ (tensile)")
        self.ax.set_ylabel(self._ylabel)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            self.ctx.signals.result_changed.disconnect(self.replot)
            self.ctx.signals.mask_changed.disconnect(self.replot)
        except (TypeError, RuntimeError):
            pass
        self.owner.on_plot_closed(self)
        super().closeEvent(event)
