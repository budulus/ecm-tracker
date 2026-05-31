"""Displacement / Strain Overlay — in-process egui canvas overlay + control panel (Phase 4 slice 4b).

Ported from ``plugins/displacement_overlay/``. The original was a separate PyQt window that registered
a canvas painter; the Rust host exposes the canvas directly, so this is now a true in-process overlay:
``overlay(self, painter)`` draws on the tracker image each refresh, and ``panel(self, ui)`` declares
the controls as an egui window (the host owns their live values and reports changes via
``on_control``).

Per triangle of a Delaunay mesh over the reference (cut-0) active points it computes the small-strain
tensor from the deformation gradient and colors the **current**-frame mesh by a chosen component with
a JET colormap; it also draws yellow displacement vectors. Uses numpy + ``scipy.spatial.Delaunay``;
**no cv2** — the JET colormap is reimplemented in numpy.

Deviations from the original (the egui panel API has only label/slider/checkbox/button, and labels are
fixed at declaration): the component selector is a 0–3 **slider** (live value) rather than a dropdown,
and the numeric value-range readout is dropped (panel labels can't update live).
"""

import numpy as np

from ecm_host import TrackerPlugin

# Component keys in selector order; index 0 (von Mises) is the default. Signed components use a range
# symmetric about 0; von Mises is non-negative ([0, max]).
COMPONENTS = ["von_mises", "exx", "eyy", "exy"]
SIGNED = {"exx", "eyy", "exy"}


def small_strain_per_triangle(ref_pts, cur_pts, simplices):
    """Per-triangle small-strain components. F maps reference edge vectors to current; ε = ½(F+Fᵀ) − I.
    Degenerate triangles (|det| < 1e-9) yield NaN. Returns dict component-key -> (M,) array."""
    m = len(simplices)
    exx = np.full(m, np.nan)
    eyy = np.full(m, np.nan)
    exy = np.full(m, np.nan)
    for i, (a, b, c) in enumerate(simplices):
        dX = np.array([ref_pts[b] - ref_pts[a], ref_pts[c] - ref_pts[a]]).T  # (2,2) edge cols
        dx = np.array([cur_pts[b] - cur_pts[a], cur_pts[c] - cur_pts[a]]).T
        det = dX[0, 0] * dX[1, 1] - dX[0, 1] * dX[1, 0]
        if abs(det) < 1e-9:
            continue
        F = dx @ np.linalg.inv(dX)
        eps = 0.5 * (F + F.T) - np.eye(2)
        exx[i], eyy[i], exy[i] = eps[0, 0], eps[1, 1], eps[0, 1]
    vm = np.sqrt(exx ** 2 - exx * eyy + eyy ** 2 + 3 * exy ** 2)
    return {"von_mises": vm, "exx": exx, "eyy": eyy, "exy": exy}


def _jet(norm):
    """Classic JET colormap (reimplements cv2.COLORMAP_JET visually). norm: (N,) in [0,1] -> (N,3)
    uint8 RGB (blue=low → cyan → green → yellow → red=high)."""
    v = np.clip(norm, 0.0, 1.0)
    four = 4.0 * v
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)


class DisplacementOverlay(TrackerPlugin):
    NAME = "Displacement / Strain Overlay"
    DESCRIPTION = "Overlay displacement vectors and an interpolated strain field on the canvas."

    def __init__(self, ctx):
        super().__init__(ctx)
        self.component_idx = 0
        self.show_field = True
        self.show_vectors = True
        self.vector_scale = 1.0
        self.opacity = 0.45
        self._simplices = None
        self._ref_count = -1

    def launch(self):
        return "Displacement / Strain Overlay active — open its panel to choose a component."

    # ---- panel ----------------------------------------------------------
    def panel(self, ui):
        ui.label("Strain field (JET): blue=low, red=high; signed components are symmetric about 0.")
        ui.slider("component", "Component 0=vonMises 1=exx 2=eyy 3=exy",
                  float(self.component_idx), 0.0, 3.0)
        ui.checkbox("show_field", "Show strain field", self.show_field)
        ui.checkbox("show_vectors", "Show displacement vectors", self.show_vectors)
        ui.slider("vector_scale", "Vector scale", self.vector_scale, 0.1, 50.0)
        ui.slider("opacity", "Field opacity", self.opacity, 0.0, 1.0)

    def on_control(self, key, value):
        if key == "component":
            self.component_idx = int(round(value)) % len(COMPONENTS)
        elif key == "show_field":
            self.show_field = bool(value)
        elif key == "show_vectors":
            self.show_vectors = bool(value)
        elif key == "vector_scale":
            self.vector_scale = float(value)
        elif key == "opacity":
            self.opacity = float(value)

    # ---- reactive: invalidate the Delaunay cache when the point set changes ----
    def on_result_changed(self):
        self._simplices = None

    def on_mask_changed(self):
        self._simplices = None

    # ---- overlay --------------------------------------------------------
    def overlay(self, painter):
        ctx = self.ctx
        cut = ctx.current_cut
        if cut is None or not ctx.has_result:
            return
        coords = ctx.coords(True)  # (frames, P, 2) float32, active points only
        if coords is None or coords.shape[1] < 3 or not (0 <= cut < coords.shape[0]):
            return
        ref_pts = coords[0]
        cur_pts = coords[cut]

        # Delaunay over the reference points (cached; recompute when the active count changes or a
        # reactive event cleared the cache).
        n_pts = coords.shape[1]
        if self._simplices is None or self._ref_count != n_pts:
            try:
                from scipy.spatial import Delaunay

                self._simplices = Delaunay(ref_pts).simplices
            except Exception:
                self._simplices = None
                return
            self._ref_count = n_pts
        simplices = self._simplices

        if self.show_field:
            key = COMPONENTS[self.component_idx]
            values = small_strain_per_triangle(ref_pts, cur_pts, simplices)[key]
            finite = values[np.isfinite(values)]
            if finite.size:
                vmax = float(np.max(np.abs(finite)))
                vmin = -vmax if key in SIGNED else 0.0
            else:
                vmin, vmax = 0.0, 1.0
            span = max(vmax - vmin, 1e-9)
            rgb = _jet(np.clip((values - vmin) / span, 0.0, 1.0))
            alpha = int(self.opacity * 255)
            for tri, v, color in zip(simplices, values, rgb):
                if not np.isfinite(v):
                    continue
                pts = [(float(cur_pts[i][0]), float(cur_pts[i][1])) for i in tri]
                painter.polygon(pts, fill=[int(color[0]), int(color[1]), int(color[2]), alpha])

        if self.show_vectors:
            scale = self.vector_scale
            for j in range(cur_pts.shape[0]):
                rx, ry = float(ref_pts[j][0]), float(ref_pts[j][1])
                tx = rx + (float(cur_pts[j][0]) - rx) * scale
                ty = ry + (float(cur_pts[j][1]) - ry) * scale
                painter.line((rx, ry), (tx, ty), color=[255, 255, 0, 255], width=1.2)


PLUGIN = DisplacementOverlay
