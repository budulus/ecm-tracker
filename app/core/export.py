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
