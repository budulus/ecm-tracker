"""Headless checks for the plugin system (SDK, manager, context, canvas hooks).

Run with:  QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.test_plugins
All checks here need the offscreen Qt platform (they build a real MainWindow).
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKER_CONFIG_DIR", tempfile.mkdtemp(prefix="cfg_"))

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from app.core.image_sequence import ImageSequence, discover
from app.plugins import CanvasInteraction, PluginContext
from app.plugins.manager import PluginManager
from tests.synthetic import make_sequence

EXPECTED_PLUGINS = {"affine_zones", "mts_uniaxial", "pressure_strain"}


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
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    ngon.on_right_press(QPointF(60, 180), None)  # right-click closes the polygon
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


def test_discovery_orders_by_plugin_order():
    # ORDER (ascending, ties broken by NAME) controls the menu/pane order: mts_uniaxial (10)
    # is pinned above affine_zones (20) despite sorting later alphabetically by folder name.
    mgr = PluginManager(None)
    order = [r.plugin_id for r in mgr.discover()]
    assert order.index("mts_uniaxial") < order.index("affine_zones")
    assert order.index("affine_zones") < order.index("pressure_strain")


def test_window_builds_plugin_menu():
    w = _tracked_window()
    ids = {r.plugin_id for r in w.plugin_manager._records.values()}
    assert EXPECTED_PLUGINS <= ids
    # the &Plugins menu holds only the utility actions (launching is via the pane buttons)
    actions = [a.text() for a in w._plugins_menu.actions() if a.text()]
    assert any("Reload Plugins" in t for t in actions)


def test_window_builds_plugin_buttons():
    w = _tracked_window()
    layout = w.plugin_manager._panel_layout
    assert layout is not None
    texts = [
        layout.itemAt(i).widget().text()
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QPushButton)
    ]
    names = {r.name for r in w.plugin_manager._records.values()}
    assert names <= set(texts)  # one launch button per plugin


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
    assert ctx.source_dir == w.state.source_dir
    assert isinstance(ctx.frame_paths, tuple)
    assert ctx.frame_paths == tuple(w.state.sequence.paths)
    assert len(ctx.frame_paths) == ctx.n_total_images

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


def test_keep_mask_rejects_multidimensional_input_and_access_is_read_only():
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    original = w.state.active_mask.copy()
    try:
        ctx.apply_keep_mask(np.ones((ctx.point_count, 1), dtype=bool))
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("a two-dimensional mask should be rejected")
    assert w.state.active_mask.shape == original.shape
    assert np.array_equal(w.state.active_mask, original)
    try:
        ctx.active_mask[0] = False
    except ValueError:
        pass
    else:
        raise AssertionError("PluginContext.active_mask must be read-only")


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
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    w.canvas.mousePressEvent(ev)
    assert len(got) == 1
    ctx.end_canvas_interaction()
    # after ending, a left click is no longer routed to the tool
    w.canvas.mousePressEvent(ev)
    assert len(got) == 1


def test_roi_cancel_leaves_plugin_interaction_intact():
    # The window-level Esc shortcut calls _cancel_roi_definition. While a plugin owns the
    # canvas (begin_canvas_interaction unchecks define_roi_action), Esc/cancel must NOT tear
    # down the plugin's interaction. Regression for the ROI shape-tools refactor.
    w = _tracked_window()
    ctx = PluginContext(w, "test")

    class Tool(CanvasInteraction):
        def on_press(self, image_pt, event):
            pass

    tool = Tool()
    ctx.begin_canvas_interaction(tool)
    assert w.canvas._interaction is tool
    assert not w.define_roi_action.isChecked()  # plugin capture switched ROI off

    w._cancel_roi_definition()  # what pressing Esc triggers
    assert w.canvas._interaction is tool  # plugin interaction survives the cancel
    ctx.end_canvas_interaction()


def test_plugin_interaction_cancels_partial_roi_and_respects_new_owner():
    _app()
    from app.gui.main_window import MainWindow

    d = tempfile.mkdtemp(prefix="interaction_")
    make_sequence(d, n_frames=3)
    w = MainWindow()
    w._load_paths(discover(d), d)
    w._begin_roi_definition("ngon")
    w.canvas._interaction.on_press(QPointF(20.0, 20.0), None)
    assert w.state.roi is not None and not w.state.roi.is_complete

    class Tool(CanvasInteraction):
        pass

    first_ctx = PluginContext(w, "first")
    first = Tool()
    first_ctx.begin_canvas_interaction(first)
    assert w.state.roi is None and not w.define_roi_action.isChecked()

    second_ctx = PluginContext(w, "second")
    second = Tool()
    second_ctx.begin_canvas_interaction(second)
    first_ctx.end_canvas_interaction()
    assert w.canvas._interaction is second
    second_ctx.end_canvas_interaction()
    assert w.canvas._interaction is None


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


def test_affine_zones_overlay_survives_reopen():
    w = _tracked_window()
    rec = w.plugin_manager._records["affine_zones"]
    plugin = rec.cls(PluginContext(w, "affine_zones"))
    before = len(w.canvas._overlays)
    win = plugin.launch()                       # first open registers the overlay
    assert len(w.canvas._overlays) == before + 1
    win.close()                                 # close removes it
    assert len(w.canvas._overlays) == before
    plugin.launch()                             # reopen must re-register it
    assert len(w.canvas._overlays) == before + 1
    plugin.on_unload()


def test_settings_round_trip():
    w = _tracked_window()
    ctx = PluginContext(w, "roundtrip")
    assert ctx.get_settings() == {}
    ctx.save_settings({"hello": 1, "mode": "x"})
    assert ctx.get_settings() == {"hello": 1, "mode": "x"}


def test_principal_stretches_known():
    """Qt-free math: deformation gradient → principal stretches & directions."""
    from plugins.affine_zones.zones import fit_zone_deformation, principal_stretches

    # Identity → both stretches 1.
    lam1, lam2, _v1, _v2 = principal_stretches(np.eye(2))
    assert abs(lam1 - 1.0) < 1e-9 and abs(lam2 - 1.0) < 1e-9

    # Pure stretch diag(2, 0.5) → λ = (2, 0.5) along the axes.
    F = np.diag([2.0, 0.5])
    lam1, lam2, v1, v2 = principal_stretches(F)
    assert abs(lam1 - 2.0) < 1e-9 and abs(lam2 - 0.5) < 1e-9
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9 and abs(np.linalg.norm(v2) - 1.0) < 1e-9
    assert abs(np.dot(v1, v2)) < 1e-9  # orthogonal directions
    assert abs(abs(v1[0]) - 1.0) < 1e-9  # λ1 direction is the x-axis

    # A rotation is rigid → both stretches 1.
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    lam1, lam2, _v1, _v2 = principal_stretches(R)
    assert abs(lam1 - 1.0) < 1e-9 and abs(lam2 - 1.0) < 1e-9

    # Round-trip through the LS fit: build points, apply F, recover it.
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ref = np.array([[1, 1], [9, 2], [3, 8], [6, 6], [5, 4]], dtype=np.float32)
    cur = (ref @ F.T).astype(np.float32)
    local_idx, F_fit, _b = fit_zone_deformation(polygon, ref, cur)
    assert local_idx.size == ref.shape[0]
    assert np.allclose(F_fit, F, atol=1e-4)


def test_zone_fit_excludes_lost_tracks():
    """Qt-free: a track LK lost (valid=False) must be dropped before the deformation fit, or its
    frozen position biases F. Regression for the plain-lstsq robustness gap."""
    from plugins.affine_zones.zones import fit_zone_deformation

    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    F = np.diag([2.0, 0.5])
    ref = np.array([[1, 1], [9, 2], [3, 8], [6, 6], [5, 4]], dtype=np.float32)
    cur = (ref @ F.T).astype(np.float32)
    cur[2] = [999.0, -999.0]  # a dead/frozen track that would skew a plain LS fit
    valid = np.array([True, True, False, True, True])

    local_idx, F_fit, _b = fit_zone_deformation(polygon, ref, cur, valid=valid)
    assert local_idx.size == 4  # the corrupt point was excluded
    assert np.allclose(F_fit, F, atol=1e-4)

    # Without the mask the corrupt point poisons the fit.
    _, F_bad, _ = fit_zone_deformation(polygon, ref, cur)
    assert not np.allclose(F_bad, F, atol=1e-4)


def test_ransac_zone_sample_size():
    """Qt-free: the configurable points-per-fit RANSAC flags injected outliers for any sample size
    and is deterministic (fixed seed), and degrades gracefully when a zone has too few points."""
    from plugins.affine_zones.zones import fit_zone_affine

    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    rng = np.random.default_rng(42)
    ref = rng.uniform(5, 95, size=(24, 2))
    F = np.array([[1.4, 0.1], [-0.05, 0.85]])
    cur = ref @ F.T + np.array([3.0, -2.0])
    cur += rng.normal(0, 0.2, size=cur.shape)  # sub-pixel inlier noise
    outliers = [4, 11, 19]
    cur[outliers] += np.array([40.0, -35.0])   # gross errors, ~53 px off

    ref32, cur32 = ref.astype(np.float32), cur.astype(np.float32)
    for k in (3, 6, 10):  # minimal exact sample through least-squares-averaged samples
        local_idx, M, inliers = fit_zone_affine(polygon, ref32, cur32, sample_size=k, reproj=2.0)
        assert local_idx.size == ref.shape[0] and M is not None
        for o in outliers:
            assert not inliers[o], f"outlier {o} kept at sample_size={k}"
        assert inliers.sum() >= ref.shape[0] - len(outliers) - 2  # clean points survive

    # Deterministic: identical inlier sets across repeated calls (fixed RANSAC seed).
    inl_a = fit_zone_affine(polygon, ref32, cur32, sample_size=6, reproj=2.0)[2]
    inl_b = fit_zone_affine(polygon, ref32, cur32, sample_size=6, reproj=2.0)[2]
    assert np.array_equal(inl_a, inl_b)

    # Too few points to separate signal from noise → everything is an inlier (nothing to clean).
    _li, _M, few = fit_zone_affine(polygon, ref32[:5], cur32[:5], sample_size=10, reproj=2.0)
    assert few.all()


def test_track_status_accessor():
    """ctx.track_status mirrors coords: (frames, points), reference frame all-valid."""
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    st = ctx.track_status(active_only=True)
    assert st is not None and st.shape == (ctx.frame_count, ctx.n_active)
    assert (st[0] == 1).all()  # every seed is valid on the reference frame


def test_stretch_plot_opens():
    """The matplotlib import is deferred + broadly guarded, and the plot window builds and
    replots. Closing the plugin window also tears the plot window down."""
    from plugins.affine_zones.zones import Zone, default_zone_color

    w = _tracked_window()
    rec = w.plugin_manager._records["affine_zones"]
    plugin = rec.cls(PluginContext(w, "affine_zones"))
    win = plugin.launch()
    win.zones.append(Zone([(60, 50), (240, 50), (240, 180), (60, 180)], default_zone_color(0)))

    win._open_plot()
    assert win._plot_window is not None and win._plot_window.isVisible()
    win._plot_window.replot()  # exercises the per-frame fit + status filtering

    win.close()  # AffineZonesWindow.closeEvent must close the plot window too
    assert win._plot_window is None
    plugin.on_unload()


def test_ransac_dialog_follows_row_selection():
    """With the RANSAC dialog open, clicking a different zone row must immediately re-preview that
    zone (so params are tuned once and applied across zones without reopening)."""
    from plugins.affine_zones.zones import Zone, default_zone_color

    w = _tracked_window()
    rec = w.plugin_manager._records["affine_zones"]
    plugin = rec.cls(PluginContext(w, "affine_zones"))
    win = plugin.launch()
    roi = [(60, 50), (240, 50), (240, 180), (60, 180)]
    win.zones.append(Zone(roi, default_zone_color(0)))
    win.zones.append(Zone(roi, default_zone_color(1)))
    win._refresh()  # build table rows for the appended zones

    win.table.selectRow(0)
    win._open_ransac()
    assert win._ransac_dialog is not None
    assert win._ransac_preview is not None and win._ransac_preview[0] == 0

    win.table.selectRow(1)  # clicking another row switches the dialog's target zone
    assert win._ransac_preview[0] == 1

    win.close()
    plugin.on_unload()


def test_set_current_frame():
    """ctx.set_current_frame moves the app's current frame (clamped) and emits frame_changed."""
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    seen = []
    w.signals.frame_changed.connect(lambda g: seen.append(g))

    ctx.set_current_frame(ctx.last_index)
    assert w.state.current_index == ctx.last_index
    assert seen and seen[-1] == ctx.last_index

    ctx.set_current_frame(-5)  # clamped to the first frame
    assert w.state.current_index == 0
    ctx.set_current_frame(10 ** 6)  # clamped to the last loaded frame
    assert w.state.current_index == w.state.total_images - 1


def test_set_reference_and_last_frame():
    """ctx.set_reference_frame / set_last_frame re-scope the tracked range, clamped to the
    invariant 0 <= reference <= last < total, mirroring the Reference/Last sliders."""
    w = _tracked_window()
    ctx = PluginContext(w, "test")
    total = w.state.total_images
    assert total >= 3  # synthetic sequence is long enough for the indices below

    ctx.set_last_frame(10 ** 6)  # clamped to the last loaded frame
    assert w.state.last_index == total - 1

    mid = total // 2
    ctx.set_last_frame(mid)
    assert w.state.last_index == mid

    ctx.set_reference_frame(10 ** 6)  # cannot exceed last_index
    assert w.state.reference_index == mid == w.state.last_index

    ctx.set_reference_frame(-5)  # clamped to the first frame
    assert w.state.reference_index == 0
    assert 0 <= w.state.reference_index <= w.state.last_index < total


def test_ransac_dialog_frame_defaults_to_last():
    """Opening RANSAC jumps the current frame to the last frame; the Frame spinbox mirrors it and,
    when edited, drives the app's current frame."""
    from plugins.affine_zones.zones import Zone, default_zone_color

    w = _tracked_window()
    rec = w.plugin_manager._records["affine_zones"]
    plugin = rec.cls(PluginContext(w, "affine_zones"))
    win = plugin.launch()
    win.zones.append(Zone([(60, 50), (240, 50), (240, 180), (60, 180)], default_zone_color(0)))
    win._refresh()  # build the table row for the appended zone
    win.table.selectRow(0)

    last = w.state.last_index
    win._open_ransac()
    dlg = win._ransac_dialog
    assert dlg is not None
    assert w.state.current_index == last  # auto-jumped to the last frame on open
    assert dlg.frame_spin.value() == last
    assert dlg.frame_spin.minimum() == w.state.reference_index
    assert dlg.frame_spin.maximum() == last

    dlg.frame_spin.setValue(last - 1)  # editing the spinbox navigates the app
    assert w.state.current_index == last - 1

    win.close()
    plugin.on_unload()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
