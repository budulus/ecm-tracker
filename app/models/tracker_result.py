from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
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

    # LK window size (px) used to produce this result, so overlays can show the actual search
    # window even if the user later edits the Tracker params without re-running.
    win_size: int = 0

    def __post_init__(self) -> None:
        """Validate all aligned-array invariants and enforce the immutability contract."""
        # Own every buffer: merely marking a caller-owned array read-only would still allow its
        # original owner to mutate the supposedly immutable result through another reference.
        dtypes = {
            "coords_fw": np.float32,
            "status_fw": np.uint8,
            "err_fw": np.float32,
            "coords_bw": np.float32,
            "status_bw": np.uint8,
            "err_bw": np.float32,
            "fb_mean_error": np.float32,
            "fb_max_error": np.float32,
        }
        raw_arrays = {name: np.asarray(getattr(self, name)) for name in dtypes}
        coords = raw_arrays["coords_fw"]
        if coords.ndim != 3 or coords.shape[2] != 2:
            raise ValueError("coords_fw must have shape (frames, points, 2)")
        n, p = coords.shape[:2]
        if n == 0 or p == 0:
            raise ValueError("TrackerResult must contain at least one frame and one point")
        expected = {
            "coords_fw": (n, p, 2),
            "status_fw": (n, p),
            "err_fw": (n, p),
            "coords_bw": (n, p, 2),
            "status_bw": (n, p),
            "err_bw": (n, p),
            "fb_mean_error": (p,),
            "fb_max_error": (p,),
        }
        for name, shape in expected.items():
            if raw_arrays[name].shape != shape:
                raise ValueError(f"{name} has shape {raw_arrays[name].shape}; expected {shape}")
        if self.reference_index < 0 or self.last_index < self.reference_index:
            raise ValueError("TrackerResult contains an invalid frame range")
        if n != self.last_index - self.reference_index + 1:
            raise ValueError("TrackerResult frame axis does not match its reference/last range")
        if self.win_size < 0:
            raise ValueError("TrackerResult win_size cannot be negative")
        for name in ("coords_fw", "coords_bw"):
            if not np.isfinite(raw_arrays[name]).all():
                raise ValueError(f"{name} must contain only finite coordinates")
        for name in ("status_fw", "status_bw"):
            if not np.isin(raw_arrays[name], (0, 1)).all():
                raise ValueError(f"{name} must contain only 0/1")
        for name in ("err_fw", "err_bw", "fb_mean_error", "fb_max_error"):
            if np.isnan(raw_arrays[name].astype(np.float64, copy=False)).any():
                raise ValueError(f"{name} cannot contain NaN")

        arrays = {
            name: np.array(raw_arrays[name], dtype=dtype, copy=True)
            for name, dtype in dtypes.items()
        }

        for name in (
            "coords_fw",
            "status_fw",
            "err_fw",
            "coords_bw",
            "status_bw",
            "err_bw",
            "fb_mean_error",
            "fb_max_error",
        ):
            value = arrays[name]
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_frames(self) -> int:
        return self.coords_fw.shape[0]

    @property
    def n_points(self) -> int:
        return self.coords_fw.shape[1]
