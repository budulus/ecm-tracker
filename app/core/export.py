import csv
import os

import numpy as np


def export(
    coords_fw: np.ndarray,
    active_mask: np.ndarray,
    reference_index: int,
    last_index: int,
    out_dir: str,
    filename: str = "coords.npy",
):
    """Write cleaned forward coordinates and the sequence range.

    - coords.npy : forward coords for kept points, shape (N_frames, N_points_kept, 2)
    - sequence.txt : "<reference_index> <last_index>" (global indices)

    Returns (coords_path, sequence_path, coords_shape).
    """
    if not filename.endswith(".npy"):
        filename += ".npy"

    coords = coords_fw[:, active_mask, :].astype(np.float32)
    coords_path = os.path.join(out_dir, filename)
    np.save(coords_path, coords)

    sequence_path = os.path.join(out_dir, "sequence.txt")
    with open(sequence_path, "w") as f:
        f.write(f"{reference_index} {last_index}\n")

    return coords_path, sequence_path, coords.shape


def export_csv(
    coords_fw: np.ndarray,
    active_mask: np.ndarray,
    frame_names,
    out_dir: str,
    filename: str = "coords.csv",
):
    """Write cleaned forward coords as CSV: one row per frame, columns interleaved x/y.

    Header:  filename,p1x,p1y,p2x,p2y,...
    Row i :  <frame_names[i]>, x0, y0, x1, y1, ...   for the kept points only.

    Returns (csv_path, coords_shape).
    """
    if not filename.endswith(".csv"):
        filename += ".csv"

    coords = coords_fw[:, active_mask, :].astype(np.float32)  # (N, P_kept, 2)
    n_frames, n_pts, _ = coords.shape
    flat = coords.reshape(n_frames, n_pts * 2)  # p0x,p0y,p1x,p1y,...

    header = ["filename"]
    for p in range(n_pts):
        header += [f"p{p + 1}x", f"p{p + 1}y"]  # 1-based labels

    csv_path = os.path.join(out_dir, filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_frames):
            # Fixed 6-decimal format (matches the affine_zones exporter); repr() would emit
            # noisy full-precision float64 expansions of the float32 values.
            writer.writerow([frame_names[i], *(f"{v:.6f}" for v in flat[i])])

    return csv_path, coords.shape
