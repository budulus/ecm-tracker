"""Small reproducible trajectory-access benchmark; no files or GUI windows are created."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from time import perf_counter
import tracemalloc
import numpy as np
from app.models.tracker_result import TrackerResult
from app.plugins import PluginContext


def main():
    n, p = 400, 1500
    xy = np.zeros((n, p, 2), np.float32)
    status = np.ones((n, p), np.uint8)
    error = np.zeros((n, p), np.float32)
    result = TrackerResult(0, n - 1, xy, status, error, xy, status, error,
                           np.zeros(p), np.zeros(p))
    ctx = PluginContext(SimpleNamespace(state=SimpleNamespace(
        result=result, active_mask=np.ones(p, bool), revision=1)))
    start = perf_counter()
    ctx.tracks()
    cold = perf_counter() - start
    tracemalloc.start()
    start = perf_counter()
    for i in range(2000):
        ctx.frame_tracks(i % n)
    elapsed = perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{n} frames x {p} points; initial snapshot {cold:.4f}s; "
          f"2000 cached frame accesses {elapsed:.4f}s; peak additional allocation {peak} bytes")


if __name__ == "__main__":
    main()
