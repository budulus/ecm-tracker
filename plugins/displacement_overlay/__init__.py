"""Displacement & strain overlay plugin.

Draws, on top of the live canvas, the per-point displacement of each tracked point relative to
the reference frame, plus an interpolated small-strain field (Delaunay triangulation of the
points, per-triangle deformation gradient → strain, colour-mapped). The overlay follows the
frame slider. Demonstrates: overlay painting, reacting to ``ctx.signals``, and reading frames
and coordinates through the context.
"""
from app.plugins import TrackerPlugin

from .overlay import StrainOverlayWindow


class DisplacementOverlayPlugin(TrackerPlugin):
    NAME = "Displacement / Strain Overlay"
    DESCRIPTION = "Overlay displacement vectors and an interpolated strain field on the canvas."

    def __init__(self, ctx):
        super().__init__(ctx)
        self._window = None

    def launch(self):
        if self._window is None:
            self._window = StrainOverlayWindow(self.ctx)
        self._window.show()
        return self._window

    def on_unload(self):
        if self._window is not None:
            self._window.close()
            self._window = None


PLUGIN = DisplacementOverlayPlugin
