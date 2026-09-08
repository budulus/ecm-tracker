"""Desired-behavior regressions for the full-review fixes (not old-bug probes)."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QWidget
from app.core.affine import fit_affine, principal_stretches, ransac_affine
from app.core.cleanup import compute_metrics, build_mask, Thresholds, BandFilter
from app.core.export import export
from app.core.kinematics import compute_series
from app.core.roi import ROI
from app.models.tracker_result import TrackerResult
from app.plugins import PluginContext, TrackerPlugin
from app.plugins.manager import PluginManager, PluginRecord
from plugins.mts_uniaxial import parsers, project_io
from plugins.mts_uniaxial.state import MtsProjectState, Step
from plugins.pressure_strain.analysis import compute_principal_strains
from tests.synthetic import make_sequence
from tests.test_mts_uniaxial import make_mts_experiment


def result(kind="photometric"):
    ref = np.array([[10, 10], [40, 10], [10, 40], [40, 40]], dtype=np.float32)
    xy = np.stack([ref, ref + 1, ref + 2])
    status = np.ones((3, 4), dtype=np.uint8)
    err = np.tile([.1, .2, .3, .4], (3, 1)).astype(np.float32)
    err[0] = 0
    return TrackerResult(0, 2, xy, status, err, xy, status, err,
                         np.zeros(4), np.zeros(4), error_kind=kind)


class ReviewFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ecm-review-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {"TRACKER_CONFIG_DIR": str(self.root / "config")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def host(self):
        from app.gui.main_window import MainWindow
        host = MainWindow()
        self.addCleanup(host.close)
        self.addCleanup(host.deleteLater)
        folder = self.root / "images"
        folder.mkdir(exist_ok=True)
        make_sequence(str(folder), n_frames=3)
        from app.core.image_sequence import discover
        self.assertTrue(host._load_paths(discover(str(folder)), str(folder)))
        host.state.features = result().coords_fw[0].copy()
        host.state.result = result()
        host.state.active_mask = np.ones(4, dtype=bool)
        host.state.touch()
        return host

    def mts_state(self):
        exp = make_mts_experiment(str(self.root / "experiment"), n_frames=3)
        st = MtsProjectState()
        for key in ("root", "images_dir", "sensor_file", "log_path"):
            setattr(st, key, exp[key])
        st.sensor = parsers.parse_sensor(st.sensor_file)
        st.image_log = parsers.parse_image_log(st.log_path)
        st.ordered_paths = parsers.resolve_image_paths(st.image_log, st.images_dir)
        st.crop_start, st.crop_end = 0, st.sensor.n_samples - 1
        st.ref_image_global, st.last_image_global = 0, 2
        st.zero_disp = st.zero_force = 0
        st.completed_through = int(Step.REFERENCE)
        project_io.save_load(st)
        return st

    def test_energy_implementation_unchanged(self):
        path = Path(__file__).resolve().parents[1] / "plugins/mts_uniaxial/reference_algorithms.py"
        self.assertEqual(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper(),
                         "C3BECE51ED1677E69B4597E7EE71D4BD50EA890C31F472577D043FBD98174D22")

    def test_snapshot_alignment_and_stale_mask(self):
        host = self.host()
        ctx = PluginContext(host)
        first = ctx.tracks()
        self.assertIs(first, ctx.tracks())
        self.assertFalse(first.coords.flags.writeable)
        ctx.apply_keep_mask([True, False, True, False], revision=first.revision)
        snap = ctx.tracks()
        np.testing.assert_array_equal(snap.point_ids, [0, 2])
        np.testing.assert_array_equal(snap.coords, first.coords[:, [0, 2]])
        with self.assertRaises(ValueError):
            ctx.apply_keep_mask([True, True], revision=first.revision)
        self.assertIsNone(ctx.frame_tracks(3))
        self.assertEqual(ctx.frame_tracks(1)[1].shape, (2, 2))

    def test_roi_and_metrics_do_not_mutate_host(self):
        host = self.host()
        host.state.roi = ROI([(0, 0), (100, 0), (0, 100)])
        ctx = PluginContext(host)
        ctx.roi.reset()
        self.assertTrue(host.state.roi.is_complete)
        metrics = ctx.metrics()
        with self.assertRaises(ValueError):
            metrics.max_step[0] = 9
        host.state.roi.corners[1] = (5, 0)
        self.assertIsNot(ctx.metrics(), metrics)

    def test_load_releases_cleanup_and_old_drag(self):
        host = self.host()
        host._open_cleanup()
        seq = host.state.sequence
        self.assertTrue(host._load_paths(seq.paths, str(self.root)))
        self.assertIsNone(host._cleanup_dialog)
        self.assertIsNone(host._preview_keep)
        host._begin_roi_definition("rectangle")
        handler = host.canvas._interaction
        if handler is not None:
            handler.on_press(QPointF(5, 5), None)
            host._load_paths(seq.paths, str(self.root))
            handler.on_release(QPointF(30, 30), None)
            self.assertIsNone(host.state.roi)

    def test_stale_tracking_is_not_installed(self):
        host = self.host()
        def track_with_change(*args):
            host.set_frame_range(1, 2)
            return result()
        with patch("app.gui.main_window.track", side_effect=track_with_change):
            host._run_tracking()
        self.assertIsNone(host.state.result)
        self.assertEqual(host.state.reference_index, 1)

    def test_atomic_range_noop_preserves_result(self):
        host = self.host()
        original = host.state.result
        events = []
        host.signals.range_changed.connect(lambda: events.append(host.state.result))
        host.set_frame_range(0, 2)
        self.assertIs(host.state.result, original)
        host.set_frame_range(1, 2)
        self.assertEqual(events, [None])

    def test_context_disposal_owns_callbacks_and_overlays(self):
        host = self.host()
        ctx = PluginContext(host)
        events = []
        ctx.subscribe(ctx.signals.mask_changed, lambda: events.append(1))
        ctx.add_overlay(lambda *_: None)
        ctx.dispose()
        ctx.dispose()
        host.signals.mask_changed.emit()
        self.assertEqual(events, [])
        self.assertFalse(ctx._overlay_wrappers)

    def test_bad_plugin_metadata_is_isolated(self):
        host = self.host()
        class Bad(TrackerPlugin):
            NAME = "Bad"
            ORDER = "first"
        with patch("app.plugins.manager.importlib.import_module", return_value=types.SimpleNamespace(PLUGIN=Bad)):
            self.assertIsNotNone(PluginManager(host)._load_record("bad").error)
        class Good(TrackerPlugin):
            pass
        Good.__module__ = Bad.__module__ = "temporary"
        with self.assertRaises(ValueError):
            PluginManager._find_plugin_class(types.SimpleNamespace(__name__="temporary", a=Good, b=Bad))

    def test_deleted_plugin_window_relaunch(self):
        host = self.host()
        class Plugin(TrackerPlugin):
            def launch(self):
                window = QWidget()
                window.show()
                return window
        manager = PluginManager(host)
        rec = PluginRecord("temp", "temp", cls=Plugin)
        manager._records["temp"] = rec
        manager.launch("temp")
        import shiboken6
        shiboken6.delete(rec.window)
        manager.launch("temp")
        self.assertTrue(rec.window.isVisible())
        manager.shutdown()

    def test_eigenvalue_filter_direction(self):
        metrics = compute_metrics(result("min_eigenvalue"), None, (100, 100))
        thr = Thresholds(opencv_error=BandFilter(True, 0))
        self.assertTrue(build_mask(metrics, thr).all())
        thr.min_eigenvalue = BandFilter(True, .25)
        np.testing.assert_array_equal(build_mask(metrics, thr), [False, False, True, True])

    def test_bad_geometry_and_reflection(self):
        ref = np.array([[0, 0], [1, 1e-10], [2, 0], [3, 1e-10]])
        self.assertIsNone(fit_affine(ref, ref + [0, .1]))
        with self.assertRaises(ValueError):
            principal_stretches(np.diag([1, -1]))
        with self.assertRaises(ValueError):
            principal_stretches(np.diag([1, 0]))
        self.assertTrue(np.isnan(principal_stretches(np.eye(2))[2]).all())

    def test_shared_reference_validity_and_compression_axis(self):
        xy = result().coords_fw.copy()
        status = np.ones((3, 4), dtype=np.uint8)
        status[:, 2:] = 0
        mts = compute_series(xy, np.arange(3), status)
        pressure = compute_principal_strains(xy, status)
        self.assertTrue(np.isnan(mts.lambda_1).all())
        np.testing.assert_array_equal(mts.eps_1, pressure.epsilon_1)
        xy[1] = xy[0] * [.8, 1.1]
        mts = compute_series(xy, np.arange(3), loading_axis_deg=0)
        self.assertAlmostEqual(mts.axial_lambda[1], .8, places=6)
        self.assertAlmostEqual(mts.lambda_1[1], 1.1, places=6)

    def test_ransac_returned_mask_matches_model(self):
        rng = np.random.default_rng(83)
        ref = rng.normal(size=(40, 2)) * 10
        cur = ref * [1.2, .9] + rng.normal(size=ref.shape) * .9
        model, keep = ransac_affine(ref, cur, sample_size=3, reproj=1)
        self.assertIsNotNone(model)
        residual = np.linalg.norm(ref @ model[:, :2].T + model[:, 2] - cur, axis=1)
        self.assertTrue(np.all(residual[keep] <= 1))

    def test_units_both_clamps_and_clock_reset(self):
        path = self.root / "sensor.dat"
        header = "Time\tWeg Achse-2\tKraft Achse-2\tWeg Achse-4\tKraft Achse-4\n"
        path.write_text(header + "ms\tm\tkN\tmm\tN\n0\t.01\t.1\t10\t100\n1000\t.02\t.2\t20\t200\n")
        sensor = parsers.parse_sensor(str(path))
        np.testing.assert_array_equal(sensor.time_s, [0, 1])
        np.testing.assert_allclose(sensor.disp_a, sensor.disp_b)
        np.testing.assert_allclose(sensor.force_a, sensor.force_b)
        path.write_text(header + "s\tmm\tN\tmm\tN\n1\t0\t0\t0\t0\n0\t0\t0\t0\t0\n")
        with self.assertRaises(ValueError):
            parsers.parse_sensor(str(path))

    def test_changed_raw_data_rejects_resume(self):
        st = self.mts_state()
        self.assertIsNotNone(project_io.load_project(st.root))
        with open(st.sensor_file, "a") as handle:
            handle.write("\n")
        self.assertIsNone(project_io.load_project(st.root))

    def test_export_failure_keeps_old_generation(self):
        st = self.mts_state()
        xy = result().coords_fw
        rows = [[i, i, 0, 0, 1] for i in range(3)]
        old_paths = project_io.save_export(st, xy, np.arange(4), rows)
        old_generation = st.export_generation
        manifest_path = Path(project_io.project_dir(st.root)) / project_io.MANIFEST
        before = manifest_path.read_bytes()
        with patch.object(project_io, "atomic_save_npy", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                project_io.save_export(st, xy + 2, np.arange(4), rows)
        self.assertEqual(st.export_generation, old_generation)
        self.assertEqual(manifest_path.read_bytes(), before)
        np.testing.assert_array_equal(np.load(old_paths[0]), xy)

    def test_named_core_exports_bind_range_and_ids(self):
        for name, ref in (("one.npy", 2), ("two.npy", 8)):
            export(result().coords_fw, np.array([1, 0, 1, 0], bool), ref, ref + 2, str(self.root), name)
            with np.load(self.root / name.replace(".npy", ".bundle.npz")) as bundle:
                np.testing.assert_array_equal(bundle["frame_indices"], np.arange(ref, ref + 3))
                np.testing.assert_array_equal(bundle["point_ids"], [0, 2])
        self.assertNotEqual((self.root / "one.sequence.txt").read_text(),
                            (self.root / "two.sequence.txt").read_text())

    def test_mts_same_reference_preserves_tracking_and_crop_clears_cache(self):
        from plugins.mts_uniaxial.window import MtsUniaxialWindow
        host = self.host()
        st = self.mts_state()
        window = MtsUniaxialWindow(PluginContext(host))
        self.addCleanup(window.dispose)
        self.addCleanup(window.close)
        window._adopt_state(st)
        host.state.features = result().coords_fw[0].copy()
        host.state.result = result()
        host.state.active_mask = np.ones(4, bool)
        host.state.touch()
        host.signals.result_changed.emit()
        old = host.state.result
        self.assertIsNotNone(window.kinematics())
        window._apply_reference(0, 2, "manual override", None)
        self.assertIs(host.state.result, old)
        self.assertTrue(window.pstate.done(Step.TRACK))
        self.assertTrue((Path(project_io.project_dir(st.root)) / "trackers.npz").is_file())
        window._on_crop_changed()
        self.assertIsNone(window.kinematics())
        self.assertIsNone(window._kin_cache)

    def test_relocated_project_uses_new_root(self):
        import shutil
        st = self.mts_state()
        destination = self.root / "relocated"
        shutil.copytree(st.root, destination)
        loaded = project_io.load_project(str(destination))
        self.assertIsNotNone(loaded)
        self.assertTrue(Path(loaded.sensor_file).is_relative_to(destination))
        self.assertTrue(Path(loaded.log_path).is_relative_to(destination))

    def test_manifest_failure_does_not_publish_export(self):
        st = self.mts_state()
        rows = [[i, i, 0, 0, 1] for i in range(3)]
        project_io.save_export(st, result().coords_fw, np.arange(4), rows)
        old = st.export_generation
        with patch.object(project_io, "save_manifest", side_effect=OSError("injected commit failure")):
            with self.assertRaises(OSError):
                project_io.save_export(st, result().coords_fw + 9, np.arange(4), rows)
        self.assertEqual(st.export_generation, old)
        np.testing.assert_array_equal(np.load(project_io.export_paths(st)[0]), result().coords_fw)

    def test_packaged_sdk_staging(self):
        import build
        scratch = self.root / "build"
        (scratch / "run.dist").mkdir(parents=True)
        with patch.object(build, "BUILD_DIR", scratch):
            payload = build._stage_payload()
        kit = payload / "bundled_plugins"
        for name in ("PLUGIN_CONTRACT.md", "plugin-api.html", "_sdk/API_REFERENCE.txt",
                     "_sdk/host_dependencies.json", "_sdk/uv.lock", "_sdk/example/__init__.py"):
            self.assertTrue((kit / name).is_file(), name)
        self.assertFalse((payload / "plugins").exists())  # Do not overwrite legacy edited plugins.

    def test_frozen_discovery_without_source_plugins_package(self):
        import build
        import sys
        import app.plugins.manager as module
        scratch = self.root / "frozen"
        (scratch / "run.dist").mkdir(parents=True)
        with patch.object(build, "BUILD_DIR", scratch):
            payload = build._stage_payload()
        original_import = module.importlib.import_module
        def fresh_import(name, *args, **kwargs):
            if name == "plugins":
                raise ModuleNotFoundError("No source plugins package", name="plugins")
            return original_import(name, *args, **kwargs)
        with patch.dict(sys.modules), patch.object(module, "PROJECT_ROOT", payload), \
             patch.object(module, "PLUGINS_DIR", payload / "bundled_plugins"), \
             patch.object(module.importlib, "import_module", side_effect=fresh_import):
            for name in list(sys.modules):
                if name == "plugins" or name.startswith("plugins."):
                    del sys.modules[name]
            records = PluginManager(None).discover()
            self.assertEqual({r.plugin_id for r in records}, {"affine_zones", "mts_uniaxial", "pressure_strain"})
            self.assertTrue(all(r.error is None for r in records))

    def test_expected_range_rejects_before_install(self):
        host = self.host()
        path = str(self.root / "saved.npz")
        host.save_trackers_to(path)
        original = host.state.result
        with self.assertRaises(ValueError):
            PluginContext(host).load_trackers(path, expected_range=(1, 2))
        self.assertIs(host.state.result, original)

    def test_v3_producing_metadata_and_invalid_status(self):
        host = self.host()
        host.state.lk_params["flags"] = 8
        path = str(self.root / "trackers.npz")
        host.save_trackers_to(path)
        from app.core.tracker_io import load_trackers
        bundle = load_trackers(path)
        self.assertEqual(bundle["error_kind"], "photometric")
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["status_fw"][1, 0] = 0  # revived at frame 2
        np.savez(path, **arrays)
        with self.assertRaises(ValueError):
            load_trackers(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
