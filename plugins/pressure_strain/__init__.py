"""Pressure–strain analysis synchronized to image modification timestamps."""
from app.plugins import TrackerPlugin


class PressureStrainPlugin(TrackerPlugin):
    NAME = "Pressure–Strain"
    DESCRIPTION = "Synchronize pressure data to frames and plot principal strain over pressure."
    ORDER = 30

    def __init__(self, ctx):
        super().__init__(ctx)
        self._window = None

    def launch(self):
        if self._window is None:
            from .window import PressureStrainWindow

            self._window = PressureStrainWindow(self.ctx)
        self._window.show()
        self._window.raise_()
        return self._window

    def on_unload(self):
        if self._window is not None:
            self._window.dispose()
            self._window.close()
            self._window = None


PLUGIN = PressureStrainPlugin
