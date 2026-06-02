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
from scipy.optimize import brentq, least_squares

ReferenceFn = Callable[[np.ndarray, np.ndarray], int]

REGISTRY: Dict[str, ReferenceFn] = {}

PREFORCE = "Set preforce"  # the one algorithm the UI passes a threshold argument to
ELASTIC_ENERGY = "Elastic-energy minimum (spring-hinge fit)"


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


@register(ELASTIC_ENERGY)
def elastic_energy_minimum(displacement: np.ndarray, force: np.ndarray) -> int:
    """Reference at the minimum of fitted elastic distortion energy (spring-hinge model).

    Fits a lumped spring-hinge-mass model (linear axial spring + rotary bending spring +
    gravity) to the force-displacement curve, then returns the index minimising the stored
    elastic distortion energy ``U_el = 1/2 (lambda-1)^2 + 1/2 beta phi^2`` -- the start of the
    clean tensile branch, just past the bending->tension knee, independent of axis scaling.

    The whole passed window is the fit window (crop it upstream with the search sliders).
    Needs >= 20 points; returns 0 on too-short input or a failed/degenerate fit.
    """
    u = np.asarray(displacement, dtype=float)
    f = np.asarray(force, dtype=float)
    if u.size < 20:
        return 0
    try:
        return _elastic_energy_index(u, f)
    except (ValueError, RuntimeError, FloatingPointError):
        return 0


def _elastic_energy_index(u: np.ndarray, f: np.ndarray) -> int:
    """Spring-hinge fit + elastic-energy argmin. Assumes ``u``, ``f`` are 1-D, equal length, >= 20.

    Raises ``ValueError`` on a degenerate fit window or an all-NaN energy curve.
    """
    HALF_PI = np.pi / 2.0

    # --- forward model: dimensionless horizontal force at grip ratio xi = x/L ---
    def f_hat(xi, gamma, beta):
        if xi <= 1e-9:
            return -gamma
        def equilibrium(phi):                       # vertical balance -> phi(xi)
            lam = xi / np.cos(phi)
            return (lam - 1.0) * np.sin(phi) + beta * phi * np.cos(phi) / lam - gamma
        try:
            phi = brentq(equilibrium, 1e-9, HALF_PI - 1e-4, xtol=1e-10, maxiter=100)
        except ValueError:
            return np.nan
        lam = xi / np.cos(phi)
        return (lam - 1.0) * np.cos(phi) - beta * phi * np.sin(phi) / lam

    # --- forward model in physical units: F(u; k, L, c, u0, W) ---
    def forward(uu, k, L, c, u0, W):
        gamma, beta = W / (k * L), c / (k * L * L)
        xi = (np.asarray(uu) + u0) / L
        return k * L * np.array([f_hat(x, gamma, beta) for x in xi])

    # --- the whole window is the fit window (already cropped by the search sliders upstream) ---
    uc, fc = u, f

    # --- subsample for speed (forward model does a root-find per point) ---
    step = max(1, uc.size // 250)
    uf, ff = uc[::step], fc[::step]

    # --- initial guess from the steepest tangent (toe-compensation style) ---
    order = np.argsort(uc)
    us, fs = uc[order], fc[order]
    w = max(5, (us.size // 30) | 1)                 # odd smoothing width
    usm = np.convolve(us, np.ones(w) / w, "same")
    fsm = np.convolve(fs, np.ones(w) / w, "same")
    slope = np.gradient(fsm, usm)
    core = slice(w, -w) if us.size > 2 * w else slice(None)
    k0 = max(np.percentile(slope[core], 80), 1e-2)
    ip = np.argmax(slope[core]) + (w if us.size > 2 * w else 0)
    onset0 = us[ip] - fs[ip] / slope[ip] if slope[ip] > 0 else us[0]
    L0 = max(2.0, 5.0 * (uc.max() - uc.min()))
    theta0 = [k0, L0, 0.05 * k0 * L0 * L0, L0 - onset0, 1e-3]

    # --- weighted nonlinear least-squares fit ---
    sigma = 0.01 * np.ptp(ff) + 1e-3
    sol = least_squares(
        lambda p: (forward(uf, *p) - ff) / sigma, theta0, method="trf",
        bounds=([1e-4, 1.0, 1e-6, -1e3, 1e-8], [1e5, 1e4, 1e9, 1e3, 1e3]),
        xtol=1e-12, ftol=1e-12, max_nfev=1000)
    k, L, c, u0, W = sol.x
    gamma, beta = W / (k * L), c / (k * L * L)

    # --- minimize elastic distortion energy over xi to get the reference ---
    xs = np.linspace(0.30, 1.80, 1500)
    phi = np.empty_like(xs)
    for i, xi in enumerate(xs):
        def eq(p, xi=xi):
            lam = xi / np.cos(p)
            return (lam - 1.0) * np.sin(p) + beta * p * np.cos(p) / lam - gamma
        try:
            phi[i] = brentq(eq, 1e-9, HALF_PI - 1e-4, xtol=1e-10, maxiter=100)
        except ValueError:
            phi[i] = np.nan
    lam = xs / np.cos(phi)
    U_el = 0.5 * (lam - 1.0) ** 2 + 0.5 * beta * phi ** 2
    xi_star = xs[np.nanargmin(U_el)]

    # --- map back to a grip reading and to the nearest input index ---
    u_star = L * xi_star - u0
    return int(np.argmin(np.abs(u - u_star)))
