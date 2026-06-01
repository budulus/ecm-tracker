"""Reference-finding algorithms: locate the zero-stress reference point in a tensile test.

Each algorithm shares one signature::

    find_reference(displacement, force) -> int   # SENSOR index into the CROPPED window

``displacement`` and ``force`` are 1-D numpy arrays over the cropped experiment window (so every
algorithm sees the same domain). The returned index is relative to that window; the caller adds
the crop start to get an absolute sensor index, then maps it to the nearest image frame.

These are baked in (not dynamically loaded). Register a new one by decorating a function with
``@register("Display name")`` and matching the signature; it appears in the UI dropdown
automatically. The two bundled algorithms are working placeholders — drop real ones in the same
way. Each is defensive: empty/too-short input returns 0.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

ReferenceFn = Callable[[np.ndarray, np.ndarray], int]

REGISTRY: Dict[str, ReferenceFn] = {}

PREFORCE = "Set preforce"  # the one algorithm the UI passes a threshold argument to


def register(name: str) -> Callable[[ReferenceFn], ReferenceFn]:
    """Decorator that registers a reference-finding function under ``name``."""

    def deco(fn: ReferenceFn) -> ReferenceFn:
        REGISTRY[name] = fn
        return fn

    return deco


@register("Force onset (5x baseline noise)")
def force_onset(displacement: np.ndarray, force: np.ndarray, n_sigma: float = 5.0) -> int:
    """First index where force rises above the baseline noise band.

    The baseline mean/std come from the first ~5% of the window; the reference is the first sample
    exceeding ``mean + n_sigma * std``. Marks the instant just before mechanical engagement.
    """
    force = np.asarray(force, dtype=float)
    if force.size < 3:
        return 0
    k = max(3, int(round(0.05 * force.size)))
    base = force[:k]
    thr = float(np.mean(base)) + n_sigma * max(float(np.std(base)), 1e-9)
    above = np.flatnonzero(force > thr)
    return int(above[0]) if above.size else 0


@register("Force-displacement knee (max curvature)")
def fd_knee(displacement: np.ndarray, force: np.ndarray) -> int:
    """Kneedle-style knee of the force-displacement curve (toe-to-linear transition).

    Both axes are normalised to [0, 1]; the reference is the point of maximum perpendicular
    distance from the chord joining the first and last points.
    """
    d = np.asarray(displacement, dtype=float)
    f = np.asarray(force, dtype=float)
    n = d.size
    if n < 3:
        return 0

    def _norm(a: np.ndarray) -> np.ndarray:
        rng = float(a.max() - a.min())
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    x, y = _norm(d), _norm(f)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    denom = float(np.hypot(dx, dy))
    if denom == 0:
        return 0
    dist = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / denom
    return int(np.argmax(dist))


@register(PREFORCE)
def preforce_reference(displacement: np.ndarray, force: np.ndarray, threshold: float = 0.0) -> int:
    """First index whose force exceeds ``threshold`` (a user-set pre-force, in N).

    Unlike the other algorithms this one takes a threshold; the UI passes the value from the
    pre-force spin box. Returns 0 if nothing exceeds the threshold (or the window is empty).
    """
    force = np.asarray(force, dtype=float)
    if force.size == 0:
        return 0
    above = np.flatnonzero(force > float(threshold))
    return int(above[0]) if above.size else 0
