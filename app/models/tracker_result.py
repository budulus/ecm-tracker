from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerResult:
    """Immutable raw tracking output. Never mutated after creation — cleanup operates on a
    separate active mask held by ProjectState.

    All per-frame arrays are indexed by *cut* index (0 = reference frame, n_frames-1 = last),
    so coords_fw[t] and coords_bw[t] describe the same frame and can be compared directly.
    """

    reference_index: int
    last_index: int

    coords_fw: np.ndarray  # (N, P, 2) float32
    status_fw: np.ndarray  # (N, P) uint8
    err_fw: np.ndarray  # (N, P) float32

    coords_bw: np.ndarray  # (N, P, 2) float32
    status_bw: np.ndarray  # (N, P) uint8
    err_bw: np.ndarray  # (N, P) float32

    fb_mean_error: np.ndarray  # (P,) float32
    fb_max_error: np.ndarray  # (P,) float32

    @property
    def n_frames(self) -> int:
        return self.coords_fw.shape[0]

    @property
    def n_points(self) -> int:
        return self.coords_fw.shape[1]
