"""Interactive homogeneous-motion (principal-stretch) zone tool.

Lets you draw polygon zones on the canvas; for each zone the tracked points inside it are fit
(least squares, reference → current frame) with an affine map whose linear part is the
homogenized deformation gradient ``F``. From the left Cauchy–Green tensor ``B = F Fᵀ`` the tool
reports the principal stretches ``λ1, λ2`` and their normalized eigenvectors — the principal
directions in the current (deformed) configuration (live, following the frame slider). Each
zone gets a unique color (with a per-row color picker); a RANSAC dialog cleans the selected
zone's points; a plot window shows λ1 (solid) / λ2 (dashed) over frames in the zone color; and
the per-frame stretches export to CSV. Demonstrates: canvas mouse capture, overlays, per-frame
compute, embedded matplotlib, and ``ctx.apply_keep_mask``.
"""
from app.plugins import TrackerPlugin

from .zones import AffineZonesWindow


class AffineZonesPlugin(TrackerPlugin):
    NAME = "Affine Zone Tool"
    DESCRIPTION = "Draw zones and fit a RANSAC affine deformation per zone."
    ORDER = 20  # pinned second, after MTS Uniaxial

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
