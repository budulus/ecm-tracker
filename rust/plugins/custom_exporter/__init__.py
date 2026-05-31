"""Custom Exporter — export tracked coordinates to CSV, MATLAB .mat, or NumPy .npz.

Native-window plugin (Phase 4 slice 4a). ``launch()`` spawns a standalone PyQt5 window
(``window.py``) as a detached child process — egui/winit owns the main thread (and macOS forbids GUI
off the main thread), so the plugin UI lives in its own process with its own event loop. The host
hands the child a data snapshot (full coords + active-point indices + frame-global indices + ref/last)
plus the saved settings via a temp ``snapshot.npz`` / ``settings.json``, and the child reports the
chosen settings back through a JSON-line **outbox** file that this retained instance drains on each
reactive event (``ctx.save_settings``).

A minimal ``panel()`` keeps this instance retained — the host only retains plugins that declare an
``overlay()`` or a ``panel()`` — so the outbox can be drained and the window reopened.

Headless ``ECM_SMOKE``: ``launch()`` runs an in-process export selftest instead of spawning the
window, so the export path is exercised deterministically without a GUI.
"""

import json
import os
import subprocess
import sys
import tempfile

from ecm_host import TrackerPlugin


def _child_python():
    """Resolve the interpreter for the window subprocess. Under the embedded host ``sys.executable``
    is the host exe, so prefer the venv python beside ``sys.prefix``; fall back to ``sys.executable``."""
    for c in (
        os.path.join(sys.prefix, "pythonw.exe"),  # python-build-standalone base (packaged layout)
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "Scripts", "pythonw.exe"),  # Windows venv (dev layout)
        os.path.join(sys.prefix, "Scripts", "python.exe"),
        os.path.join(sys.prefix, "bin", "python"),  # Unix venv
    ):
        if os.path.exists(c):
            return c
    return sys.executable


class CustomExporter(TrackerPlugin):
    NAME = "Custom Exporter"
    DESCRIPTION = "Export tracked coordinates to CSV, MATLAB .mat, or NumPy .npz."

    def __init__(self, ctx):
        super().__init__(ctx)
        self._outbox = None

    # ---- snapshot building ------------------------------------------------
    def _build_snapshot(self):
        """Gather a data snapshot from ctx: full coords + the kept-point indices, so the window's
        "active only" checkbox can switch locally. Returns ``None`` if there is no result."""
        import numpy as np

        ctx = self.ctx
        if not ctx.has_result or ctx.frame_count == 0:
            return None
        coords = ctx.coords(False)  # (F, P, 2) float32, all P points
        active = ctx.point_indices()  # (n_active,) int64 original indices of kept points
        ref, last = ctx.reference_index, ctx.last_index
        return {
            "coords": coords,
            "active_indices": active if active is not None else np.arange(coords.shape[1], dtype=np.int64),
            "frame_globals": np.arange(ref, ref + ctx.frame_count, dtype=np.int64),
            "reference_index": int(ref),
            "last_index": int(last),
        }

    # ---- launch -----------------------------------------------------------
    def launch(self):
        snap = self._build_snapshot()
        if snap is None:
            return "Custom Exporter: no tracking result to export."
        if os.environ.get("ECM_SMOKE"):
            return self._selftest(snap)
        return self._spawn_window(snap)

    def _spawn_window(self, snap):
        import numpy as np

        data_dir = tempfile.mkdtemp(prefix="ecm_export_")
        np.savez(
            os.path.join(data_dir, "snapshot.npz"),
            coords=snap["coords"],
            active_indices=snap["active_indices"],
            frame_globals=snap["frame_globals"],
            reference_index=np.int64(snap["reference_index"]),
            last_index=np.int64(snap["last_index"]),
        )
        with open(os.path.join(data_dir, "settings.json"), "w") as f:
            json.dump(self.ctx.get_settings(), f)
        self._outbox = os.path.join(data_dir, "outbox.jsonl")
        window = os.path.join(os.path.dirname(__file__), "window.py")
        creationflags = 0x00000008 if os.name == "nt" else 0  # DETACHED_PROCESS
        subprocess.Popen(
            [_child_python(), window, data_dir, self._outbox],
            env=os.environ.copy(),
            creationflags=creationflags,
            close_fds=True,
        )
        coords = snap["coords"]
        return f"Custom Exporter: window opened ({coords.shape[1]} points x {coords.shape[0]} frames)."

    def _selftest(self, snap):
        """Headless export for ECM_SMOKE: export the saved/default format in-process via the shared
        writers, and exercise the settings round-trip. Exits without a GUI."""
        import numpy as np

        from . import exporter

        saved = self.ctx.get_settings()
        fmt = saved.get("format", "csv")
        active_only = bool(saved.get("active_only", True))
        coords = snap["coords"]
        if active_only:
            point_ids = snap["active_indices"]
            coords = coords[:, point_ids, :]
        else:
            point_ids = np.arange(coords.shape[1], dtype=np.int64)
        out_dir = tempfile.mkdtemp(prefix="ecm_export_smoke_")
        path = os.path.join(out_dir, f"export.{fmt}")
        exporter.write(fmt, path, coords, point_ids, snap["frame_globals"],
                       snap["reference_index"], snap["last_index"])
        self.ctx.save_settings({"format": fmt, "active_only": active_only, "last_dir": out_dir})
        return f"Custom Exporter: selftest wrote {fmt} ({coords.shape[1]} points) to {path}."

    # ---- panel (retention + reopen) --------------------------------------
    def panel(self, ui):
        ctx = self.ctx
        if ctx.has_result:
            ui.label(f"{ctx.n_active} active / {ctx.point_count} points x {ctx.frame_count} frames.")
        else:
            ui.label("No tracking result yet.")
        ui.button("reopen", "Open export window")

    def on_control(self, key, value):
        if key == "reopen" and not os.environ.get("ECM_SMOKE"):
            snap = self._build_snapshot()
            if snap is not None:
                self._spawn_window(snap)
        self._drain_outbox()

    # ---- reactive: drain the child->host settings outbox -----------------
    def on_frame_changed(self, global_index):
        self._drain_outbox()

    def on_result_changed(self):
        self._drain_outbox()

    def on_mask_changed(self):
        self._drain_outbox()

    def _drain_outbox(self):
        """Apply any ``save_settings`` commands the child window appended to the outbox file. The
        host persists the recorded settings after this reactive call returns."""
        path = self._outbox
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as f:
                lines = f.readlines()
            open(path, "w").close()  # truncate once read
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except ValueError:
                continue
            if cmd.get("type") == "save_settings":
                self.ctx.save_settings(cmd.get("data", {}))


PLUGIN = CustomExporter
