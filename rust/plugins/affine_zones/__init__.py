"""Affine Zones — draw zones and fit a RANSAC affine deformation per zone (Phase 4 slice 4c).

Native-window plugin. ``launch()`` spawns a standalone PyQt5 + embedded-matplotlib window
(``window.py``) as a detached child process; the host hands it a data snapshot (active coords +
track status + the reference frame image path + ref/frame_count/n_active + ROI) via a temp
``snapshot.npz`` / ``meta.json``. In the window the user draws polygon zones on the reference image,
reads per-zone principal stretches in a table, plots λ1/λ2 vs frame, runs RANSAC outlier removal, and
exports a CSV.

The window reports its one mutation — a RANSAC keep-mask — back through a JSON-line **outbox**. The
retained instance drains it when the user clicks this plugin's panel "Apply changes from window"
button: ``on_control`` is the *only* host path that applies a recorded keep-mask (reactive ``on_*``
events persist settings but don't apply keep-masks), so the drain lives there. A minimal ``panel()``
keeps the instance retained.

Headless ``ECM_SMOKE``: ``launch()`` runs an in-process selftest (affine fit + principal stretches +
RANSAC + CSV over the ROI as a default zone), no GUI. ``zones.py`` is pure numpy; **no cv2**.
"""

import json
import os
import subprocess
import sys
import tempfile

from ecm_host import TrackerPlugin


def _child_python():
    """Resolve the interpreter for the window subprocess (venv python beside ``sys.prefix``; the
    embedded host's ``sys.executable`` is the host exe)."""
    for c in (
        os.path.join(sys.prefix, "Scripts", "pythonw.exe"),
        os.path.join(sys.prefix, "Scripts", "python.exe"),
        os.path.join(sys.prefix, "bin", "python"),
    ):
        if os.path.exists(c):
            return c
    return sys.executable


class AffineZones(TrackerPlugin):
    NAME = "Affine Zone Tool"
    DESCRIPTION = "Draw zones and fit a RANSAC affine deformation per zone."

    def __init__(self, ctx):
        super().__init__(ctx)
        self._outbox = None

    # ---- snapshot building ------------------------------------------------
    def _build_snapshot(self):
        import numpy as np

        ctx = self.ctx
        if not ctx.has_result or ctx.frame_count == 0 or ctx.n_active < 3:
            return None
        coords = ctx.coords(True)  # (F, n_active, 2) active points only
        status = ctx.track_status(True)  # (F, n_active) uint8, or None
        ref = ctx.reference_index
        return {
            "coords": coords,
            "status": status if status is not None else np.ones(coords.shape[:2], dtype=np.uint8),
            "reference_index": int(ref),
            "frame_count": int(ctx.frame_count),
            "n_active": int(ctx.n_active),
            "ref_path": ctx.frame_path(ref) or "",
            "roi": [list(c) for c in ctx.roi_corners],
        }

    # ---- launch -----------------------------------------------------------
    def launch(self):
        snap = self._build_snapshot()
        if snap is None:
            return "Affine Zone Tool: need a tracking result with >= 3 active points."
        if os.environ.get("ECM_SMOKE"):
            return self._selftest(snap)
        return self._spawn_window(snap)

    def _spawn_window(self, snap):
        import numpy as np

        data_dir = tempfile.mkdtemp(prefix="ecm_zones_")
        np.savez(os.path.join(data_dir, "snapshot.npz"), coords=snap["coords"], status=snap["status"])
        meta = {k: snap[k] for k in ("reference_index", "frame_count", "n_active", "ref_path", "roi")}
        with open(os.path.join(data_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        self._outbox = os.path.join(data_dir, "outbox.jsonl")
        window = os.path.join(os.path.dirname(__file__), "window.py")
        creationflags = 0x00000008 if os.name == "nt" else 0  # DETACHED_PROCESS
        subprocess.Popen(
            [_child_python(), window, data_dir, self._outbox],
            env=os.environ.copy(),
            creationflags=creationflags,
            close_fds=True,
        )
        return f"Affine Zone Tool: window opened ({snap['n_active']} active points)."

    def _selftest(self, snap):
        """Headless ECM_SMOKE path: exercise the zone math (affine fit + stretches + RANSAC + CSV)
        in-process over the ROI (or the point bbox) as a default zone. No GUI."""
        from . import zones as Z

        coords = snap["coords"]
        status = snap["status"]
        ref_pts = coords[0]
        zone = snap["roi"] if len(snap["roi"]) >= 3 else Z.bbox_polygon(ref_pts)
        t = snap["frame_count"] - 1
        valid = status[t] == 1
        fit = Z.fit_zone_deformation(zone, ref_pts, coords[t], valid=valid)
        if fit is None:
            return "Affine Zone Tool: selftest — fewer than 3 points in the default zone."
        local_idx, F, _b = fit
        lam1, lam2, _v1, _v2 = Z.principal_stretches(F)
        _M, inliers = Z.estimate_affine_ransac(ref_pts[local_idx], coords[t][local_idx])
        out_dir = tempfile.mkdtemp(prefix="ecm_zones_smoke_")
        path = os.path.join(out_dir, "zones.csv")
        Z.write_zones_csv(path, [zone], coords, status, snap["reference_index"], snap["frame_count"])
        return (
            f"Affine Zone Tool: selftest fit zone ({local_idx.size} pts) "
            f"lambda1={lam1:.4f} lambda2={lam2:.4f}, RANSAC {int(inliers.sum())}/{local_idx.size} "
            f"inliers, wrote {path}."
        )

    # ---- panel (retention + apply/reopen) --------------------------------
    def panel(self, ui):
        ui.label(f"{self.ctx.n_active} active points. Draw zones in the window, then apply changes here.")
        ui.button("apply", "Apply changes from window")
        ui.button("reopen", "Reopen window")

    def on_control(self, key, value):
        if key == "reopen" and not os.environ.get("ECM_SMOKE"):
            snap = self._build_snapshot()
            if snap is not None:
                self._spawn_window(snap)
        # Drain here (not in on_* events): only on_control's dispatch path applies a recorded keep-mask.
        self._drain_outbox()

    def _drain_outbox(self):
        """Apply the RANSAC keep-mask(s) the child window queued. Multiple same-length masks are
        ANDed (apply_keep_mask only ever removes points), then applied once."""
        import numpy as np

        path = self._outbox
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as f:
                lines = f.readlines()
            open(path, "w").close()
        except OSError:
            return
        combined = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except ValueError:
                continue
            if cmd.get("type") == "apply_keep_mask":
                keep = np.asarray(cmd.get("keep", []), dtype=bool)
                if combined is None or combined.shape != keep.shape:
                    combined = keep
                else:
                    combined = combined & keep
        if combined is not None and combined.size:
            self.ctx.apply_keep_mask(combined.tolist())


PLUGIN = AffineZones
