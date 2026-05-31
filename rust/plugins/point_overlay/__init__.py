"""Point Overlay — example overlay plugin for the Rust host (slice 3f).

Demonstrates the canvas overlay API: define ``overlay(self, painter)`` and draw
with the host-provided painter using IMAGE coordinates. The host re-invokes it
when the session state changes (frame scrub, tracking, cleanup) and renders the
draw-commands on top of the built-in markers.

Draws the ROI outline (gold) and the active tracked points at the current frame
(cyan). Reads ``ctx.coords`` (a NumPy array), so it exercises the array API too.
"""

from ecm_host import TrackerPlugin


class PointOverlay(TrackerPlugin):
    NAME = "Point Overlay"
    DESCRIPTION = "Draw the ROI and current-frame tracked points on the canvas."

    def launch(self):
        return "Point Overlay active — scrub frames to see the tracked points."

    def overlay(self, painter):
        ctx = self.ctx
        # ROI outline (image coords; the host transforms to screen).
        if ctx.roi_corners:
            painter.polygon(ctx.roi_corners, stroke=(255, 215, 0, 255), width=2.0)
        # Active tracked points at the current frame.
        if ctx.has_result:
            cut = ctx.current_cut
            coords = ctx.coords(True)  # (frames, P, 2) float32, active points only
            if cut is not None and coords is not None and 0 <= cut < coords.shape[0]:
                for x, y in coords[cut]:
                    painter.circle((float(x), float(y)), radius=3.0, fill=(0, 200, 255, 255))


PLUGIN = PointOverlay
