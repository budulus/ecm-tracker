"""Copy this folder to your user plugins directory and rename it."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel
from app.plugins import TrackerPlugin
from app.plugins.analysis import compute_series


def mean_major_strain(snapshot):
    """Pure calculation: ignore failed tracks; use only the documented SDK."""
    if snapshot is None:
        return None
    import numpy as np
    # These artificial monotonic times are ONLY for a time-independent strain summary.
    # For time derivatives, supply independently calibrated acquisition timestamps.
    series = compute_series(snapshot.coords, np.arange(len(snapshot.frame_indices)), snapshot.valid)
    finite = series.eps_1[np.isfinite(series.eps_1)]
    return float(finite.mean()) if finite.size else None


class StrainSummary(TrackerPlugin):
    API_VERSION = 1
    NAME = "Strain summary"
    DESCRIPTION = "Example: validity-aware analysis and managed subscriptions."
    ORDER = 100

    def launch(self):
        label = QLabel(parent=self.ctx.window)
        label.setWindowFlags(Qt.WindowType.Window)
        label.setWindowTitle(self.NAME)

        def refresh(*_):
            value = mean_major_strain(self.ctx.tracks())
            label.setText("Track points first." if value is None else f"Mean major strain: {value:.4g}")

        # Automatic unload cleanup; window destruction also disconnects these closures.
        unsub = [self.ctx.subscribe(signal, refresh) for signal in
                 (self.ctx.signals.result_changed, self.ctx.signals.mask_changed,
                  self.ctx.signals.sequence_changed)]
        label.destroyed.connect(lambda *_: [disconnect() for disconnect in unsub])
        refresh()
        label.show()
        return label


PLUGIN = StrainSummary
