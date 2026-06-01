"""MTS Uniaxial — synchronize MTS force/displacement sensor data with the image sequence.

Loads an experiment (images in acquisition-log order + the MTS ``.dat`` sensor stream),
interpolates the sensor onto the image timeline (with a user temporal offset), lets the user crop
the experiment window and pick a zero-stress reference frame, zeroes the channels, and sets the
core app's tracked range. After the user tracks in the main window, it exports per-frame tracked
coordinates aligned with the interpolated, zeroed force & displacement.

See ``plugins/mts_uniaxial/README.md`` for the workflow, the project-folder file set, and resume
behaviour.
"""
from app.plugins import TrackerPlugin


class MtsUniaxialPlugin(TrackerPlugin):
    NAME = "MTS Uniaxial"
    DESCRIPTION = "Sync MTS force/displacement with the image sequence and set the tracked range."

    def __init__(self, ctx):
        super().__init__(ctx)
        self._window = None

    def launch(self):
        if self._window is None:
            from .window import MtsUniaxialWindow  # deferred so importing the package stays cheap
            self._window = MtsUniaxialWindow(self.ctx)
        self._window.show()
        self._window.raise_()
        return self._window

    def on_unload(self):
        if self._window is not None:
            self._window.close()
            self._window = None


PLUGIN = MtsUniaxialPlugin
