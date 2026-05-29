"""Custom exporter plugin.

Exports the tracked coordinates in a choice of formats (long-form CSV, MATLAB ``.mat``, or
NumPy ``.npz`` with metadata). Demonstrates the simplest plugin shape: read data through the
context, open a window, and persist the user's last choices via ``ctx.get_settings`` /
``ctx.save_settings``.
"""
from app.plugins import TrackerPlugin

from .exporter import CustomExporterWindow


class CustomExporterPlugin(TrackerPlugin):
    NAME = "Custom Exporter"
    DESCRIPTION = "Export tracked coordinates to CSV, MATLAB .mat, or NumPy .npz."

    def launch(self):
        window = CustomExporterWindow(self.ctx)
        window.show()
        return window


PLUGIN = CustomExporterPlugin
