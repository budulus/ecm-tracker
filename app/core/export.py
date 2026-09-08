import csv
import os

import numpy as np

from app.core.atomic_io import atomic_open, atomic_save_npy


def _clean_coords(coords_fw, active_mask):
    coords_fw = np.asarray(coords_fw)
    active_mask = np.asarray(active_mask, dtype=bool)
    if coords_fw.ndim != 3 or coords_fw.shape[2] != 2:
        raise ValueError("Coordinates must have shape (frames, points, 2)")
    if active_mask.shape != (coords_fw.shape[1],):
        raise ValueError("Active mask length does not match the coordinate point axis")
    if not np.isfinite(coords_fw[:, active_mask, :]).all():
        raise ValueError("Selected export coordinates contain NaN or infinity")
    return coords_fw[:, active_mask, :].astype(np.float32)


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

    coords = _clean_coords(coords_fw, active_mask)
    coords_path = os.path.join(out_dir, filename)
    if reference_index < 0 or last_index - reference_index + 1 != coords.shape[0]:
        raise ValueError("Export frame range does not match coordinates")
    # One atomic file is authoritative; loose arrays/text below are legacy convenience views.
    with atomic_open(os.path.splitext(coords_path)[0] + ".bundle.npz", "wb") as handle:
        np.savez(handle, coords=coords, point_ids=np.flatnonzero(active_mask),
                 frame_indices=np.arange(reference_index, last_index + 1))
    atomic_save_npy(coords_path, coords)

    sequence_path = os.path.join(out_dir, "sequence.txt" if filename == "coords.npy" else os.path.splitext(filename)[0] + ".sequence.txt")
    with atomic_open(sequence_path) as f:
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

    coords = _clean_coords(coords_fw, active_mask)  # (N, P_kept, 2)
    n_frames, n_pts, _ = coords.shape
    if len(frame_names) != n_frames:
        raise ValueError("Frame-name count does not match the coordinate frame axis")
    flat = coords.reshape(n_frames, n_pts * 2)  # p0x,p0y,p1x,p1y,...

    header = ["filename"]
    for p in range(n_pts):
        header += [f"p{p + 1}x", f"p{p + 1}y"]  # 1-based labels

    csv_path = os.path.join(out_dir, filename)
    with atomic_open(csv_path, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_frames):
            # Fixed 6-decimal format (matches the affine_zones exporter); repr() would emit
            # noisy full-precision float64 expansions of the float32 values.
            writer.writerow([frame_names[i], *(f"{v:.6f}" for v in flat[i])])

    return csv_path, coords.shape
