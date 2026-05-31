"""Session Summary — a minimal example ECM Tracker plugin for the Rust host.

Demonstrates the `ecm_host` SDK end-to-end: subclass `TrackerPlugin`, read the
`PluginContext` handed in as ``self.ctx``, and return a short string from
``launch()`` (the host shows it in the status bar). Uses only scalar/list reads,
so it needs no numpy — copy it as the starting point for a real plugin.

Declared via the module-level ``PLUGIN`` attribute the host looks for.
"""

from ecm_host import TrackerPlugin


class SessionSummary(TrackerPlugin):
    NAME = "Session Summary"
    DESCRIPTION = "Summarize the current tracking session in the status bar."

    def launch(self):
        ctx = self.ctx
        if not ctx.has_sequence:
            return "Session Summary: no image sequence loaded."

        parts = [f"{ctx.n_total_images} frames"]
        size = ctx.image_size()
        if size is not None:
            parts.append(f"{size[1]}×{size[0]} px")  # width x height
        if ctx.roi_corners:
            parts.append(f"ROI {len(ctx.roi_corners)} corners")
        if ctx.has_result:
            parts.append(
                f"{ctx.point_count} points ({ctx.n_active} active) "
                f"over {ctx.frame_count} frames"
            )
        else:
            parts.append("no tracking result yet")
        return "Session Summary: " + ", ".join(parts) + "."


PLUGIN = SessionSummary
