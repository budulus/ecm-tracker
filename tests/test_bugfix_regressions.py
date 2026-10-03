"""Desired-behavior regressions for the 2026-09-26 bug-fix pass."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QPointF
from PySide6.QtWidgets import QApplication, QWidget
from app.core.cleanup import BAND_CAP, Thresholds, build_mask, compute_metrics, thresholds_from_dict, thresholds_to_dict
from app.core.image_sequence import discover
from app.core.result import TrackerResult
from app.core.roi import ROI
from app.core.tracking import track
from app.plugins import PluginContext, TrackerPlugin
from app.plugins.manager import PluginRecord
from tests.synthetic import make_sequence


def flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class BugfixRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecm-bugfix-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {"TRACKER_CONFIG_DIR": str(self.root / "config")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def host(self, n_frames=4):
        from app.gui.main_window import MainWindow
        host = MainWindow()
        self.addCleanup(host.close)
        self.addCleanup(host.deleteLater)
        folder = self.root / "images"
        folder.mkdir(exist_ok=True)
        make_sequence(str(folder), n_frames=n_frames)
        self.assertTrue(host._load_paths(discover(str(folder)), str(folder)))
        return host

    def tracked_host(self):
        host = self.host()
        host.state.roi = ROI([(40, 40), (280, 40), (280, 200), (40, 200)])
        host._detect_grid()
        host._run_tracking()
        self.assertIsNotNone(host.state.result)
        return host

    # ---- core window ----------------------------------------------------------------------
    def test_grid_from_oversized_roi_stays_inside_image(self):
        host = self.host()
        h, w = host.state.image_size()
        host.state.roi = ROI([(-30, -30), (w + 30, -30), (w + 30, h + 30), (-30, h + 30)])
        host._detect_grid()
        f = host.state.features
        self.assertGreater(len(f), 0)
        self.assertTrue(((f[:, 0] >= 0) & (f[:, 1] >= 0) & (f[:, 0] < w) & (f[:, 1] < h)).all())
        self.assertIsNotNone(track(host.state.sequence, 0, 3, f, host.state.lk_params))

    def test_cancelled_roi_redefinition_keeps_previous_roi(self):
        host = self.host()
        original = ROI([(10, 10), (100, 10), (100, 100), (10, 100)])
        host.state.roi = original
        host._begin_roi_definition("rectangle")
        host._cancel_roi_definition()
        self.assertIs(host.state.roi, original)
        # Leaving the reference frame mid-definition also restores it.
        host._begin_roi_definition("ngon")
        host.canvas._interaction.on_press(QPointF(5, 5), None)
        host._go_to_frame(1)
        self.assertIs(host.state.roi, original)
        self.assertFalse(host.define_roi_action.isChecked())
        self.assertTrue(host.clear_roi_action.isEnabled())

    def test_export_status_names_range_sidecar(self):
        host = self.tracked_host()
        out = self.root / "out" / "run1.npy"
        out.parent.mkdir()
        with patch("app.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(out), "")):
            host._export()
        self.assertTrue((out.parent / "run1.sequence.txt").is_file())
        self.assertIn("run1.sequence.txt", host.statusBar().currentMessage())

    # ---- cleanup --------------------------------------------------------------------------
    def test_enabled_band_at_default_keeps_every_point(self):
        from app.gui.cleanup_dialog import CleanupDialog
        ref = np.array([[10, 10], [40, 10], [10, 40], [40, 40]], np.float32)
        xy = np.stack([ref, ref + 1, ref + 2])
        status = np.ones((3, 4), np.uint8)
        err = np.array([[0, 0, 0, 0], [.1, .2, .3, 1.2345674], [.1, .2, .3, 1.2345674]], np.float32)
        result = TrackerResult(0, 2, xy, status, err, xy, status, err, np.zeros(4), np.zeros(4))
        metrics = compute_metrics(result, None, (100, 100))
        dialog = CleanupDialog(metrics, None)
        self.addCleanup(dialog.deleteLater)
        dialog.opencv_error.enable.setChecked(True)
        self.assertTrue(build_mask(metrics, dialog.thresholds()).all())

    def test_legacy_saved_thresholds_default_min_eigenvalue_to_keep_all(self):
        data = thresholds_to_dict(Thresholds())
        data.pop("min_eigenvalue")
        restored = thresholds_from_dict(data)
        self.assertEqual(restored.min_eigenvalue.hi, 0.0)
        self.assertEqual(restored.fb_max.hi, BAND_CAP)

    def test_cleanup_counts_follow_external_mask_changes(self):
        host = self.tracked_host()
        host._open_cleanup()
        n = host.state.result.n_points
        keep = np.ones(n, dtype=bool)
        keep[0] = False
        host.apply_keep_mask(keep)  # e.g. the Delete tool or a plugin, not the dialog
        self.assertEqual(host._cleanup_dialog.count_label.text(), f"Keeping {n - 1} / {n - 1} points")

    def test_closed_dialogs_are_released(self):
        from app.gui.cleanup_dialog import CleanupDialog
        from PySide6.QtWidgets import QProgressDialog
        host = self.tracked_host()
        for _ in range(3):
            host._open_cleanup()
            host._cleanup_dialog._on_save()  # starts the "Saved" timer on a soon-deleted button
            host._cleanup_dialog.reject()
            flush_deletes()
        self.assertEqual(host.findChildren(CleanupDialog), [])
        self.assertEqual(host.findChildren(QProgressDialog), [])

    # ---- plugins --------------------------------------------------------------------------
    def test_relaunch_after_close_keeps_plugin_state(self):
        from plugins.affine_zones.zones import Zone
        host = self.host()
        manager = host.plugin_manager
        self.addCleanup(manager.shutdown)
        manager.launch("affine_zones")
        first = manager._records["affine_zones"].window
        first.zones.append(Zone([(0, 0), (10, 0), (10, 10)], "#2563eb"))
        first.close()
        manager.launch("affine_zones")
        again = manager._records["affine_zones"].window
        self.assertIs(again, first)
        self.assertEqual(len(again.zones), 1)
        self.assertTrue(again.isVisible())

    def test_relaunch_with_new_widget_releases_closed_one(self):
        host = self.host()
        manager = host.plugin_manager

        class Plugin(TrackerPlugin):
            def launch(self):
                window = QWidget()
                window.show()
                return window

        rec = PluginRecord("temp", "temp", cls=Plugin)
        manager._records["temp"] = rec
        self.addCleanup(manager.shutdown)
        manager.launch("temp")
        old = rec.window
        instance = rec.instance
        old.close()
        manager.launch("temp")
        flush_deletes()
        self.assertIs(rec.instance, instance)
        self.assertFalse(shiboken6.isValid(old))
        self.assertTrue(rec.window.isVisible())

    def test_ransac_escape_tears_down_preview(self):
        from plugins.affine_zones.zones import AffineZonesWindow, Zone
        host = self.tracked_host()
        ctx = PluginContext(host, "affine_zones")
        self.addCleanup(ctx.dispose)
        window = AffineZonesWindow(ctx)
        self.addCleanup(window.close)
        window.show()
        window.zones.append(Zone([(30, 30), (290, 30), (290, 210), (30, 210)], "#2563eb"))
        window._refresh()
        window.table.selectRow(0)
        window._open_ransac()
        self.assertIsNotNone(window._ransac_preview)
        window._ransac_dialog.reject()  # what Esc does
        self.assertIsNone(window._ransac_preview)
        self.assertIsNone(window._ransac_dialog)

    def test_zone_drawing_cancelled_by_foreign_interaction(self):
        from plugins.affine_zones.zones import AffineZonesWindow
        host = self.host()
        ctx = PluginContext(host, "affine_zones")
        self.addCleanup(ctx.dispose)
        window = AffineZonesWindow(ctx)
        self.addCleanup(window.close)
        window.show()
        window._start_zone()
        host.canvas._interaction.on_press(QPointF(5, 5), None)
        host.canvas.clear_interaction()  # e.g. a range change or another canvas tool
        self.assertIsNone(window._tool)
        self.assertFalse(window.finish_btn.isEnabled())
        self.assertTrue(window.new_btn.isEnabled())

    def test_mts_spin_boxes_apply_on_commit_not_per_keystroke(self):
        from plugins.mts_uniaxial.window import MtsUniaxialWindow
        host = self.host()
        ctx = PluginContext(host, "mts_uniaxial")
        self.addCleanup(ctx.dispose)
        window = MtsUniaxialWindow(ctx)
        self.addCleanup(window.dispose)
        self.assertFalse(window.ref_spin.keyboardTracking())
        self.assertFalse(window.offset_spin.keyboardTracking())

    def test_mts_manifest_tolerates_paths_without_relative_form(self):
        from plugins.mts_uniaxial import project_io
        from plugins.mts_uniaxial.state import MtsProjectState
        state = MtsProjectState()
        state.root = str(self.root / "experiment")
        state.sensor_file = "D:/elsewhere/specimen.dat"
        state.log_path = str(self.root / "experiment" / "images" / "VDCCam.log")
        real = os.path.relpath

        def relpath(path, start=None):
            if str(path).startswith("D:"):
                raise ValueError("path is on mount 'D:', start on mount 'C:'")
            return real(path, start)

        with patch("os.path.relpath", side_effect=relpath):
            project_io.save_manifest(state)
        manifest = json.loads((Path(project_io.project_dir(state.root)) / project_io.MANIFEST).read_text())
        self.assertEqual(set(manifest["relative_paths"]), {"log_path"})
        manifest["relative_paths"] = ["not", "a", "mapping"]
        (Path(project_io.project_dir(state.root)) / project_io.MANIFEST).write_text(json.dumps(manifest))
        self.assertIsNone(project_io.load_project(state.root))

    def test_plugin_metrics_cache_follows_result_identity(self):
        host = self.host(n_frames=3)
        ctx = PluginContext(host, "metrics")
        self.addCleanup(ctx.dispose)

        def make(points):
            ref = (np.arange(points * 2, dtype=np.float32).reshape(points, 2) % 50) + 10
            xy = np.stack([ref, ref + 1, ref + 2])
            status = np.ones((3, points), np.uint8)
            err = np.zeros((3, points), np.float32)
            return TrackerResult(0, 2, xy, status, err, xy, status, err, np.zeros(points), np.zeros(points))

        host.state.result, host.state.active_mask = make(4), np.ones(4, bool)
        self.assertEqual(ctx.metrics().n_points, 4)
        # Simulate a freed result whose id() is reused: the cache must not match on id alone.
        stale_id = id(host.state.result)
        host.state.result, host.state.active_mask = make(7), np.ones(7, bool)
        with patch("builtins.id", return_value=stale_id):
            self.assertEqual(ctx.metrics().n_points, 7)

    # ---- discovery ------------------------------------------------------------------------
    def test_directory_scan_skips_hidden_files(self):
        folder = self.root / "scan"
        folder.mkdir()
        for name in ("img_1.png", "._img_1.png", ".hidden.png"):
            (folder / name).write_bytes(b"")
        self.assertEqual([os.path.basename(p) for p in discover(str(folder))], ["img_1.png"])
        explicit = [str(folder / "._img_1.png"), str(folder / "img_1.png")]
        self.assertEqual(len(discover(explicit)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
