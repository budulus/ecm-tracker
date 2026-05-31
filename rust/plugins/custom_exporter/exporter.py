"""Pure export writers for the Custom Exporter plugin — CSV / MATLAB .mat / NumPy .npz.

Byte-for-byte faithful to the original ``app/plugins/custom_exporter/exporter.py``. numpy is
required; scipy is imported lazily (only for ``.mat``). No cv2, no Qt — safe to import in the host
process and in the standalone window subprocess.
"""

import csv
import json
import os

import numpy as np


def write_csv(path, coords, point_ids, frame_globals):
    """Long-form CSV: one row per (frame, point). coords is (frames, points, 2)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_global", "frame_cut", "point_id", "x", "y"])
        for t in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                x, y = coords[t, j]
                w.writerow([int(frame_globals[t]), t, int(point_ids[j]), f"{x:.4f}", f"{y:.4f}"])


def write_mat(path, coords, point_ids, frame_globals, ref, last):
    """MATLAB .mat via scipy.io.savemat (imported lazily)."""
    from scipy.io import savemat

    savemat(
        path,
        {
            "coords": coords.astype(np.float64),
            "point_ids": point_ids.astype(np.int64).reshape(-1, 1),
            "frame_global_indices": frame_globals.astype(np.int64).reshape(-1, 1),
            "reference_index": int(ref),
            "last_index": int(last),
        },
    )


def write_npz(path, coords, point_ids, frame_globals, ref, last):
    """NumPy .npz plus a sidecar ``*_meta.json`` describing the arrays."""
    np.savez(
        path,
        coords=coords.astype(np.float32),
        point_ids=point_ids.astype(np.int64),
        frame_global_indices=frame_globals.astype(np.int64),
        reference_index=np.int64(ref),
        last_index=np.int64(last),
    )
    meta = {
        "format": "feature-tracker coords export",
        "shape": list(coords.shape),
        "axes": ["frame (cut order; 0 = reference)", "point", "(x, y)"],
        "reference_index": int(ref),
        "last_index": int(last),
        "n_points": int(coords.shape[1]),
    }
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def write(fmt, path, coords, point_ids, frame_globals, ref, last):
    """Dispatch to the writer for ``fmt`` ("csv" | "mat" | "npz")."""
    if fmt == "csv":
        write_csv(path, coords, point_ids, frame_globals)
    elif fmt == "mat":
        write_mat(path, coords, point_ids, frame_globals, ref, last)
    else:
        write_npz(path, coords, point_ids, frame_globals, ref, last)
