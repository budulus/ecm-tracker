"""Interactive affine / RANSAC zone tool.

Lets you draw polygon zones on the canvas; for each zone the tracked points inside it are fit
with a RANSAC affine transform between the reference frame and the current frame. Shows the
affine parameters and inlier/outlier split (live, following the frame slider), can export the
per-frame affines to CSV, and can drop the RANSAC outliers via the safe keep-mask API.
Demonstrates: canvas mouse capture, overlays, per-frame compute, and ``ctx.apply_keep_mask``.
"""
from app.plugins import TrackerPlugin

from .zones import AffineZonesWindow


class AffineZonesPlugin(TrackerPlugin):
    NAME = "Affine Zone Tool"
    DESCRIPTION = "Draw zones and fit a RANSAC affine deformation per zone."

    def __init__(self, ctx):
        super().__init__(ctx)
        self._window = None

    def launch(self):
        if self._window is None:
            self._window = AffineZonesWindow(self.ctx)
        self._window.show()
        return self._window

    def on_unload(self):
        if self._window is not None:
            self._window.close()
            self._window = None


PLUGIN = AffineZonesPlugin
