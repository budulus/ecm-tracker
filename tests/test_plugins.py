"""Headless checks for the plugin system (SDK, manager, context, canvas hooks).

Run with:  QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.test_plugins
All checks here need the offscreen Qt platform (they build a real MainWindow).
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKER_CONFIG_DIR", tempfile.mkdtemp(prefix="cfg_"))

import numpy as np
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QWidget

from app.core.image_sequence import ImageSequence, discover
from app.plugins import CanvasInteraction, PluginContext
from app.plugins.manager import PluginManager
from tests.synthetic import make_sequence

EXPECTED_PLUGINS = {"custom_exporter", "displacement_overlay", "affine_zones"}


_APP = None


def _app():
    # Keep a module-level reference; an unreferenced QApplication gets GC'd, after which
    # constructing any QWidget aborts with "Must construct a QApplication before a QWidget".
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _tracked_window():
    """Build a MainWindow, load a synthetic sequence, and run tracking through the real UI path."""
    _app()
    from app.gui.main_window import MainWindow

    d = tempfile.mkdtemp(prefix="trk_")
    make_sequence(d, n_frames=10, dx=2.0, dy=1.0)
    w = MainWindow()
    w.resize(900, 600)
    w.show()
    w._load_paths(discover(d), d)
    w.define_roi_action.setChecked(True)
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        w._on_image_clicked(QPointF(*c))
    w._detect_shi_tomasi()
    w._run_tracking()
    assert w.state.result is not None
    return w


def test_discovery_finds_examples():
    mgr = PluginManager(None)
    records = mgr.discover()
    found = {r.plugin_id: r for r in records}
    for pid in EXPECTED_PLUGINS:
        assert pid in found, f"{pid} not discovered"
        assert found[pid].error is None, f"{pid} failed to load:\n{found[pid].error}"
        assert found[pid].cls is not None and found[pid].name


def test_window_builds_plugin_menu():
    w = _tracked_window()
    ids = {r.plugin_id for r in w.plugin_manager._records.values()}
    assert EXPECTED_PLUGINS <= ids
    # the &Plugins menu has an entry per plugin plus the two utility actions
    actions = [a.text() for a in w._plugins_menu.actions() if a.text()]
    assert any("Reload Plugins" in t for t in actions)


def test_context_accessors():
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    assert ctx.has_sequence and ctx.has_result
    assert ctx.frame_count == w.state.result.n_frames
    assert ctx.point_count == w.state.result.n_points

    coords = ctx.coords(active_only=True)
    assert coords.shape == (ctx.frame_count, ctx.n_active, 2)
    assert coords.dtype == np.float32
    assert ctx.coords(active_only=False).shape[1] == ctx.point_count
    assert len(ctx.point_indices()) == ctx.n_active

    # index conversions round-trip; reference is cut 0
    assert ctx.global_to_cut(ctx.reference_index) == 0
    assert ctx.cut_to_global(0) == ctx.reference_index

    img = ctx.frame_bgr(ctx.current_index)
    assert img.ndim == 3 and img.shape[2] == 3
    assert ctx.image_size() == img.shape[:2]
    assert ctx.frame_rgb(ctx.current_index).shape == img.shape

    # ROI + metrics
    assert ctx.roi is not None and len(ctx.roi_corners) == 4
    assert ctx.roi_contains(150, 115)
    assert ctx.metrics() is not None and ctx.metrics().n_points == ctx.point_count


def test_apply_keep_mask_is_undoable():
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    start = ctx.n_active
    assert start >= 4

    # full-length mask: keep only the first half
    keep = np.zeros(ctx.point_count, dtype=bool)
    keep[ctx.point_indices()[: start // 2]] = True
    n_undo = len(w.state.undo_stack)
    ctx.apply_keep_mask(keep)
    assert ctx.n_active == start // 2
    assert len(w.state.undo_stack) == n_undo + 1

    # active-length mask: drop the first remaining point
    active_len_mask = np.ones(ctx.n_active, dtype=bool)
    active_len_mask[0] = False
    expected = ctx.n_active - 1
    ctx.apply_keep_mask(active_len_mask)
    assert ctx.n_active == expected

    # undo restores the previous active set
    w._cleanup_undo()
    assert ctx.n_active == start // 2


def test_overlay_is_painted():
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    calls = {"n": 0}

    def overlay(painter, c):
        calls["n"] += 1
        # exercise the screen mapping that real overlays use
        c.image_to_screen(10.0, 20.0)

    ctx.add_overlay(overlay)
    w.canvas.repaint()
    assert calls["n"] >= 1
    ctx.remove_overlay(overlay)
    calls["n"] = 0
    w.canvas.repaint()
    assert calls["n"] == 0


def test_canvas_interaction_receives_clicks():
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    got = []

    class Tool(CanvasInteraction):
        def on_press(self, image_pt, event):
            got.append((image_pt.x(), image_pt.y()))

    ctx.begin_canvas_interaction(Tool())
    pos = QPointF(w.canvas.width() / 2, w.canvas.height() / 2)
    ev = QMouseEvent(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    w.canvas.mousePressEvent(ev)
    assert len(got) == 1
    ctx.end_canvas_interaction()
    # after ending, a left click is no longer routed to the tool
    w.canvas.mousePressEvent(ev)
    assert len(got) == 1


def test_example_plugins_launch():
    w = _tracked_window()
    app = _app()
    for rec in w.plugin_manager._records.values():
        assert rec.cls is not None
        plugin = rec.cls(PluginContext(w, rec.plugin_id))
        window = plugin.launch()
        assert isinstance(window, QWidget)
        app.processEvents()
        plugin.on_unload() if hasattr(plugin, "on_unload") else None
        window.close()


def test_settings_round_trip():
    w = _tracked_window()
    ctx = PluginContext(w, "roundtrip")
    assert ctx.get_settings() == {}
    ctx.save_settings({"hello": 1, "mode": "x"})
    assert ctx.get_settings() == {"hello": 1, "mode": "x"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
