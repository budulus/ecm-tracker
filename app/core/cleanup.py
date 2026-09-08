from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from app.core.roi import ROI
from app.core.result import TrackerResult

# Spin-box ceiling for band bounds; also the "no upper limit" display value.
BAND_CAP = 1_000_000.0

# Numeric band filters and the metric attribute each one tests.
BAND_METRICS = {
    "fw_failures": "fail_count_fw",
    "bw_failures": "fail_count_bw",
    "opencv_error": "max_err_fw",
    "mean_error": "mean_err_fw",
    "fb_mean": "fb_mean",
    "fb_max": "fb_max",
    "distance": "max_step",
    "min_eigenvalue": "min_eigenvalue",
}
COUNT_BANDS = ("fw_failures", "bw_failures")


@dataclass(frozen=True)
class Metrics:
    """Per-point quality metrics derived once from a TrackerResult. Points that never tracked
    validly get +inf error/step so the error filters can drop them once tightened."""

    n_frames: int
    n_points: int
    fail_count_fw: np.ndarray  # (P,) int
    fail_count_bw: np.ndarray  # (P,) int
    max_err_fw: np.ndarray  # (P,) float
    mean_err_fw: np.ndarray  # (P,) float
    fb_mean: np.ndarray  # (P,) float
    fb_max: np.ndarray  # (P,) float
    max_step: np.ndarray  # (P,) float, max single-frame displacement
    left_image: np.ndarray  # (P,) bool
    left_roi: np.ndarray  # (P,) bool
    error_kind: str = "photometric"
    min_eigenvalue: Optional[np.ndarray] = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.setflags(write=False)
                object.__setattr__(self, name, value)


@dataclass
class BandFilter:
    """A max threshold for one metric. When disabled the filter is ignored entirely (so points
    with +inf metrics survive); when enabled a point is kept iff metric <= hi. `cap` is the
    UI slider scale (the value of the "max" box) and is not used by build_mask."""

    enabled: bool = False
    hi: float = BAND_CAP
    cap: float = BAND_CAP


@dataclass
class Thresholds:
    """Per-filter bands plus two boolean filters. A point is kept iff it passes every
    *enabled* filter. The defaults (all disabled, both bools off) keep all points."""

    fw_failures: BandFilter = field(default_factory=BandFilter)
    bw_failures: BandFilter = field(default_factory=BandFilter)
    opencv_error: BandFilter = field(default_factory=BandFilter)
    mean_error: BandFilter = field(default_factory=BandFilter)
    fb_mean: BandFilter = field(default_factory=BandFilter)
    fb_max: BandFilter = field(default_factory=BandFilter)
    distance: BandFilter = field(default_factory=BandFilter)
    min_eigenvalue: BandFilter = field(default_factory=lambda: BandFilter(hi=0.0, cap=BAND_CAP))
    drop_left_image: bool = False
    drop_left_roi: bool = False


def compute_metrics(
    result: TrackerResult, roi: Optional[ROI], image_size: Tuple[int, int]
) -> Metrics:
    cf = result.coords_fw
    sf = result.status_fw
    ef = result.err_fw
    sb = result.status_bw
    n, p = result.n_frames, result.n_points
    h, w = image_size

    fail_fw = (sf == 0).sum(axis=0).astype(int)
    fail_bw = (sb == 0).sum(axis=0).astype(int)

    max_err = np.full(p, np.inf, dtype=np.float32)
    mean_err = np.full(p, np.inf, dtype=np.float32)
    if n == 1:
        max_err.fill(0.0)
        mean_err.fill(0.0)
    for j in range(p):
        # Frame zero is a seed, not an LK transition; its synthetic zero error must not dilute the
        # mean. A one-frame result is handled above as the legitimate no-transition case.
        valid = sf[1:, j] == 1
        if valid.any():
            e = ef[1:, j][valid]
            max_err[j] = float(e.max())
            mean_err[j] = float(e.mean())

    # max single-frame displacement over consecutive frames where both endpoints are valid
    # A valid one-frame range has no movement, so its maximum step is exactly zero. Starting at
    # +inf is still useful for multi-frame points that never have a valid consecutive pair.
    step = np.zeros(p, dtype=np.float32) if n < 2 else np.full(p, np.inf, dtype=np.float32)
    if n >= 2:
        d = np.linalg.norm(cf[1:] - cf[:-1], axis=2)  # (N-1, P)
        valid_pair = (sf[1:] == 1) & (sf[:-1] == 1)
        for j in range(p):
            dj = d[valid_pair[:, j], j]
            if dj.size:
                step[j] = float(dj.max())

    x = cf[..., 0]
    y = cf[..., 1]
    out_of_bounds = (x < 0) | (x >= w) | (y < 0) | (y >= h)
    left_image = (out_of_bounds & (sf == 1)).any(axis=0)

    left_roi = np.zeros(p, dtype=bool)
    if roi is not None and roi.is_complete:
        for j in range(p):
            for t in range(n):
                if sf[t, j] == 1 and not roi.contains(float(x[t, j]), float(y[t, j])):
                    left_roi[j] = True
                    break

    eigen = np.zeros(p, dtype=np.float32)
    if result.error_kind == "min_eigenvalue" and n > 1:
        for j in range(p):
            values = ef[1:, j][sf[1:, j] == 1]
            eigen[j] = float(values.min()) if values.size else 0.0
    return Metrics(
        n_frames=n,
        n_points=p,
        fail_count_fw=fail_fw,
        fail_count_bw=fail_bw,
        max_err_fw=max_err,
        mean_err_fw=mean_err,
        fb_mean=result.fb_mean_error,
        fb_max=result.fb_max_error,
        max_step=step,
        left_image=left_image,
        left_roi=left_roi,
        error_kind=result.error_kind,
        min_eigenvalue=eigen,
    )


def default_thresholds() -> Thresholds:
    """Factory defaults: every filter disabled, so all points are kept."""
    return Thresholds()


def build_mask(metrics: Metrics, thr: Thresholds) -> np.ndarray:
    """Boolean (P,) keep mask. Disabled bands are skipped entirely; an enabled band keeps a
    point iff metric <= hi (so +inf metrics survive unless an enabled band excludes them)."""
    keep = np.ones(metrics.n_points, dtype=bool)
    for name, metric_attr in BAND_METRICS.items():
        band: BandFilter = getattr(thr, name)
        if not band.enabled:
            continue
        if name in {"opencv_error", "mean_error"} and metrics.error_kind != "photometric":
            continue  # Eigenvalue quality is higher-is-better; legacy quality is unknown.
        if name == "min_eigenvalue":
            if metrics.error_kind == "min_eigenvalue" and metrics.min_eigenvalue is not None:
                keep &= metrics.min_eigenvalue >= band.hi
            continue
        values = getattr(metrics, metric_attr)
        keep &= values <= band.hi
    if thr.drop_left_image:
        keep &= ~metrics.left_image
    if thr.drop_left_roi:
        keep &= ~metrics.left_roi
    return keep


def thresholds_to_dict(thr: Thresholds) -> dict:
    out = {
        name: {"enabled": b.enabled, "hi": b.hi, "cap": b.cap}
        for name in BAND_METRICS
        for b in (getattr(thr, name),)
    }
    out["drop_left_image"] = thr.drop_left_image
    out["drop_left_roi"] = thr.drop_left_roi
    return out


def thresholds_from_dict(data: dict) -> Thresholds:
    """Rebuild Thresholds from persisted data, tolerant of missing/partial keys."""
    data = data if isinstance(data, dict) else {}
    thr = Thresholds()
    for name in BAND_METRICS:
        d = data.get(name) or {}
        d = d if isinstance(d, dict) else {}
        try:
            hi = float(d.get("hi", BAND_CAP))
            cap = float(d.get("cap", BAND_CAP))
        except (TypeError, ValueError):
            hi = cap = BAND_CAP
        if not np.isfinite(hi) or hi < 0:
            hi = BAND_CAP
        if not np.isfinite(cap) or cap <= 0:
            cap = BAND_CAP
        setattr(
            thr,
            name,
            BandFilter(
                enabled=bool(d.get("enabled", False)),
                hi=hi,
                cap=cap,
            ),
        )
    thr.drop_left_image = bool(data.get("drop_left_image", False))
    thr.drop_left_roi = bool(data.get("drop_left_roi", False))
    return thr
