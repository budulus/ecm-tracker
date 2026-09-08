"""Small SDK-only test fixture for plugin calculations (no MainWindow knowledge required)."""
from types import SimpleNamespace
import numpy as np
from app.plugins import TrackingSnapshot


def sample_tracks():
    """Three frames/four points: identity, 20% x stretch, then one failed track."""
    ref = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    coords = np.stack([ref, ref * [1.2, 1], ref * [1.3, 1]])
    valid = np.ones((3, 4), dtype=bool)
    valid[2, 3] = False
    coords[2, 3] = coords[1, 3]
    arrays = [np.arange(5, 8), np.arange(4), coords, valid, np.zeros((3, 4))]
    for value in arrays:
        value.setflags(write=False)
    return TrackingSnapshot(1, *arrays, "photometric", 5, 7)


class FakeContext:
    """Read-only fixture for calculation tests, NOT a replacement Qt lifecycle emulator."""
    def __init__(self, snapshot=None):
        self.snapshot = snapshot

    def tracks(self, active_only=True):
        return self.snapshot

    @property
    def revision(self):
        return self.snapshot.revision if self.snapshot is not None else 0
