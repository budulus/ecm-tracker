"""Standalone matplotlib window for the selected pressure–strain curve."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


MODE_LABELS = {
    "epsilon_1": "ε₁ — major principal strain",
    "epsilon_2": "ε₂ — minor principal strain",
    "mean": "Mean principal strain — (ε₁ + ε₂) / 2",
}


def _load_matplotlib():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    return FigureCanvasQTAgg, NavigationToolbar2QT, Figure


class PressureStrainPlotWindow(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Pressure–Strain")
        self.resize(760, 560)

        FigureCanvas, Toolbar, Figure = _load_matplotlib()
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.note = QLabel()
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setStyleSheet("color: #6b727a; padding: 2px;")

        layout = QVBoxLayout(self)
        layout.addWidget(Toolbar(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.note)

    def replot(self) -> None:
        self.ax.clear()
        series = self.owner.current_plot_series()
        if series is None:
            self.note.setText("Pressure and a valid tracking result are required.")
            self.canvas.draw_idle()
            return

        finite = np.isfinite(series.pressure_mbar) & np.isfinite(series.strain)
        self.ax.plot(
            series.pressure_mbar,
            series.strain,
            "-o",
            color="#2563eb",
            linewidth=1.8,
            markersize=3.2,
            markerfacecolor="#ffffff",
            markeredgewidth=1.0,
        )
        self.ax.axhline(0.0, color="#9aa1aa", linewidth=0.8, alpha=0.55)
        if series.pressure_zeroed:
            self.ax.axvline(0.0, color="#9aa1aa", linewidth=0.8, alpha=0.55)
        self.ax.set_xlabel(
            "Pressure change from reference (mbar)"
            if series.pressure_zeroed
            else "Pressure (mbar)"
        )
        self.ax.set_ylabel("Principal linear strain ε")
        self.ax.set_title(MODE_LABELS[series.strain_mode])
        self.ax.grid(True, alpha=0.25)
        self.ax.margins(x=0.03, y=0.08)

        missing = int(finite.size - finite.sum())
        if not finite.any():
            self.note.setText("No frame has enough valid, non-collinear tracks for a strain fit.")
        elif missing:
            self.note.setText(f"{missing} frame(s) omitted because strain could not be fitted.")
        else:
            self.note.clear()
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self.owner.on_plot_closed(self)
        super().closeEvent(event)
