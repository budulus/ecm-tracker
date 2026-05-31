"""Decimate Points — example plugin exercising the apply_keep_mask mutation (slice 3g-b).

Demonstrates the single plugin mutation: ctx.apply_keep_mask(keep). On launch it drops every
other active point (a simple, deterministic downsample). The context is a read-only snapshot, so
the call only *records* the mask; the host applies it through its undoable path, so the user can
Undo it in the Cleanup dialog.

keep is a plain list of bools of length n_active (numpy-free); the host coerces it. A list of
length point_count would also work — see ctx.apply_keep_mask's docstring.
"""

from ecm_host import TrackerPlugin


class DecimatePoints(TrackerPlugin):
    NAME = "Decimate Points"
    DESCRIPTION = "Drop every other active point (demo of apply_keep_mask)."

    def launch(self):
        ctx = self.ctx
        if not ctx.has_result or ctx.n_active == 0:
            return "Decimate Points: no tracking result to filter."
        n = ctx.n_active
        keep = [i % 2 == 0 for i in range(n)]  # keep even-indexed active points
        ctx.apply_keep_mask(keep)
        return f"Decimate Points: kept {sum(keep)} of {n} active points."


PLUGIN = DecimatePoints
