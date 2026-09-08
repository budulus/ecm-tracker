"""Read-only application audit probes; all generated experiments live in a temporary directory.

Run from the repository root: .venv/Scripts/python.exe review/reproduce_findings.py
These probes record current behavior, including defects; they are not regression acceptance tests.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMP = tempfile.TemporaryDirectory(prefix="ecm-review-")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TRACKER_CONFIG_DIR"] = str(Path(TEMP.name) / "config")
os.environ["MPLCONFIGDIR"] = str(Path(TEMP.name) / "matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QWidget

from app.core.affine import fit_affine, principal_stretches, ransac_affine
from app.core.cleanup import BandFilter, Thresholds, build_mask, compute_metrics
from app.core.roi import ROI
from app.core import tracker_io
from app.gui.main_window import MainWindow
from app.models.tracker_result import TrackerResult
from app.plugins import CanvasInteraction, PluginContext, TrackerPlugin
from app.plugins.manager import PluginManager, PluginRecord
from plugins.affine_zones.zones import AffineZonesWindow, Zone, default_zone_color
from plugins.mts_uniaxial import MtsUniaxialPlugin, kinematics, parsers, project_io
from plugins.mts_uniaxial.state import MtsProjectState, Step
from plugins.mts_uniaxial.window import MtsUniaxialWindow
from plugins.pressure_strain.analysis import compute_principal_strains
from tests.synthetic import make_sequence
from tests.test_mts_uniaxial import _write_sensor, make_mts_experiment

APP = QApplication.instance() or QApplication([])
WINDOWS = []


def record(name, **values):
    print(json.dumps({"probe": name, **values}, default=str), flush=True)


def result_for(coords, reference=0, errors=None, status=None):
    coords = np.asarray(coords, dtype=np.float32)
    n, p = coords.shape[:2]
    status = np.ones((n, p), np.uint8) if status is None else status
    errors = np.zeros((n, p), np.float32) if errors is None else errors
    return TrackerResult(reference, reference + n - 1, coords, status, errors,
                         coords, status, errors, np.zeros(p), np.zeros(p))


def main_window(name, frames=3):
    window = MainWindow()
    WINDOWS.append(window)
    folder = Path(TEMP.name) / name
    paths = make_sequence(str(folder), n_frames=frames)
    assert window.load_sequence_from_paths(paths, str(folder))
    return window, paths


def install_result(window):
    n = window.state.n_cut
    seeds = np.array([[70, 60], [150, 60], [150, 140], [70, 140]], np.float32)
    coords = np.stack([seeds * (1 + 0.01 * t) for t in range(n)])
    window.state.features = seeds
    window.state.roi = ROI([(50, 40), (200, 40), (200, 180), (50, 180)])
    window.state.result = result_for(coords, window.state.reference_index)
    window.state.active_mask = np.ones(len(seeds), bool)
    window.signals.result_changed.emit()
    return window.state.result


def load_mts(window, root_name):
    exp = make_mts_experiment(str(Path(TEMP.name) / root_name), n_frames=4)
    log = parsers.parse_image_log(exp["log_path"])
    sensor = parsers.parse_sensor(exp["sensor_file"])
    paths = parsers.resolve_image_paths(log, exp["images_dir"])
    window._fresh_load(exp["root"], exp["images_dir"], exp["sensor_file"],
                       exp["log_path"], log, sensor, paths)
    return exp


def probe_math():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = result_for(np.stack([square, square]), errors=np.array([[0.] * 4, [.001, .1, .001, .1]]))
    metrics = compute_metrics(result, None, (10, 10))
    keep = build_mask(metrics, Thresholds(opencv_error=BandFilter(enabled=True, hi=.01)))
    record("lk_eigenvalue_filter_direction", eigenvalues=metrics.max_err_fw.tolist(), kept=keep.tolist())
    for name, F in (("reflection", np.diag([-1., 1.])), ("collapse", np.diag([1., 0.]))):
        coords = np.stack([square, square @ F.T])
        fit = fit_affine(coords[0], coords[1])
        series = compute_principal_strains(coords)
        record(name, fit_accepted=fit is not None, determinant=float(np.linalg.det(fit[1])),
               strains=[float(series.epsilon_1[1]), float(series.epsilon_2[1])])
    line = np.array([[0., 0.], [1., 0.], [2., 0.]])
    series = np.stack([line, line])
    record("reference_fit_consistency", mts=kinematics.compute_series(series, [0, 1]).lambda_1.tolist(),
           pressure=compute_principal_strains(series).epsilon_1.tolist())
    skinny = np.array([[0., 0.], [1., 0.], [2., 1e-10], [3., 0.]])
    noisy = skinny.copy()
    noisy[2, 1] += .1
    fit = fit_affine(skinny, noisy)
    record("near_collinear_fit", accepted=fit is not None, max_stretch=principal_stretches(fit[1])[0])
    F = np.diag([.8, .8 ** -.5])
    series = kinematics.compute_series(np.stack([square, square @ F.T]), [0, 1])
    record("compression_major_is_not_axial", axial_stretch=.8, reported_tensile_stretch=float(series.lambda_1[1]),
           reported_tensile_strain=float(series.eps_1[1]), lateral_prediction=float(series.eps_2_ico[1]))
    rng = np.random.default_rng(0)
    for trial in range(100):
        src = rng.uniform(-10, 10, (20, 2))
        dst = src + rng.normal(0, .5, src.shape)
        M, inliers = ransac_affine(src, dst, sample_size=3, reproj=.8, max_iters=200)
        residual = np.linalg.norm(src @ M[:, :2].T + M[:, 2] - dst, axis=1)
        if (residual[inliers] > .8).any():
            record("ransac_final_membership", trial=trial, threshold=.8,
                   max_inlier_residual=float(residual[inliers].max()))
            break


def probe_units():
    path = Path(TEMP.name) / "units.dat"
    _write_sensor(str(path), [[0, 1000], [0, 1], [0, 2], [0, 1], [0, 2]],
                  ["Time", "Achse-2 Weg", "Achse-2 Kraft", "Achse-4 Weg", "Achse-4 Kraft"],
                  ["ms", "mm", "kN", "mm", "kN"])
    sensor = parsers.parse_sensor(str(path))
    record("sensor_units", units=sensor.units, time_s=sensor.time_s.tolist(), force_a=sensor.force_a.tolist(),
           warnings=sensor.warnings)


def probe_energy():
    from scipy.optimize import brentq, minimize_scalar
    gamma, beta = .1, .05

    def model(xi):
        def equilibrium(phi):
            lam = xi / np.cos(phi)
            return (lam - 1) * np.sin(phi) + beta * phi * np.cos(phi) / lam - gamma
        phi = brentq(equilibrium, 1e-9, np.pi / 2 - 1e-4)
        lam = xi / np.cos(phi)
        energy = .5 * (lam - 1) ** 2 + .5 * beta * phi ** 2
        force = (lam - 1) * np.cos(phi) - beta * phi * np.sin(phi) / lam
        return energy, force

    minimum = minimize_scalar(lambda xi: model(xi)[0], bounds=(.3, 1.8), method="bounded")
    record("energy_minimum_is_not_zero_force", gamma=gamma, beta=beta,
           xi_at_minimum=minimum.x, dimensionless_force_at_minimum=model(minimum.x)[1])


def probe_export_transaction():
    state = MtsProjectState(root=str(Path(TEMP.name) / "export"))
    state.completed_through = int(Step.TRACK)
    project_io.save_export(state, np.zeros((2, 1, 2)), np.array([1]), [[0, 0, 0, 0, True], [1, 1, 1, 1, True]])
    original = project_io.atomic_save_npy
    calls = 0

    def interrupted(path, array):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected disk failure on second artifact")
        original(path, array)

    with patch.object(project_io, "atomic_save_npy", interrupted):
        try:
            project_io.save_export(state, np.ones((3, 2, 2)), np.array([2, 3]), [])
        except OSError:
            pass
    pdir = Path(project_io.project_dir(state.root))
    record("partial_export", coordinates_shape=np.load(pdir / "tracked_coords.npy").shape,
           ids_shape=np.load(pdir / "point_indices.npy").shape,
           manifest_progress=json.loads((pdir / "manifest.json").read_text())["completed_through"])


def probe_sequence_ui():
    window, paths = main_window("cleanup")
    install_result(window)
    window._open_cleanup()
    original = window._cleanup_dialog
    window.load_sequence_from_paths(paths, str(Path(paths[0]).parent))
    error = None
    try:
        window._cleanup_preview()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record("cleanup_survives_sequence_replacement", same_dialog=window._cleanup_dialog is original,
           preview_retained=window.canvas._preview_keep_mask is not None, preview_error=error)
    original.close()
    window._begin_roi_definition("rectangle")
    handler = window.canvas._interaction
    handler.on_press(QPointF(10, 10), None)
    window.load_sequence_from_paths(paths, str(Path(paths[0]).parent))
    handler.on_release(QPointF(30, 30), None)
    record("roi_drag_crosses_sequence", committed_to_new_sequence=window.state.roi is not None,
           corners=window.state.roi.corners if window.state.roi else None)
    window.state.features = np.array([[10, 10], [20, 20], [30, 30]], np.float32)
    window._open_point_manager()
    manager = window._point_manager
    window.delete_points_action.setChecked(True)
    window.canvas._interaction.on_press(QPointF(30, 30), None)
    error = None
    manager.table.selectRow(2)
    try:
        window._point_manager_delete()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record("point_manager_seed_edit", table_rows=manager.table.rowCount(),
           actual_seeds=len(window.state.features), delete_error=error)
    manager.close()


def probe_tracking_reentrancy():
    import app.gui.main_window as module
    window, _ = main_window("reentrant", frames=4)
    window.state.features = np.array([[70, 60], [150, 60], [150, 140]], np.float32)
    original = module.track
    context = PluginContext(window)

    def with_plugin_edit(sequence, reference, last, points, params, progress):
        edited = False

        def callback(done, total):
            nonlocal edited
            if not edited:
                edited = True
                context.set_reference_frame(1)
            return progress(done, total)

        return original(sequence, reference, last, points, params, callback)

    with patch.object(module, "track", with_plugin_edit):
        window._run_tracking()
    record("tracking_reentrant_mutation", state_reference=window.state.reference_index,
           result_reference=window.state.result.reference_index,
           features_cleared=window.state.features is None)


def probe_tracker_validation():
    window, _ = main_window("tracker-validation")
    result = install_result(window)
    path = Path(TEMP.name) / "revived.npz"
    arrays = {name: getattr(result, name).copy() for name in tracker_io.RESULT_ARRAY_DTYPES}
    arrays["status_fw"][:, 0] = [1, 0, 1]
    arrays["err_fw"][1, 0] = -10
    tracker_io.save_trackers(path, total_images=3, reference_index=0, last_index=2,
                            current_index=0, features=window.state.features + 10,
                            roi_corners=None, active_mask=np.ones(4, bool), result_arrays=arrays,
                            win_size=0, lk_params={}, shi_tomasi_params={}, grid_params={},
                            sequence_fingerprint=window.state.sequence.fingerprint)
    window.load_trackers_from(path)
    record("tracker_v2_semantics", revived_status=window.state.result.status_fw[:, 0].tolist(),
           negative_error=float(window.state.result.err_fw[1, 0]),
           seed_matches_reference=bool(np.array_equal(window.state.features, window.state.result.coords_fw[0])))


def probe_plugin_lifecycle():
    with patch.object(MtsUniaxialPlugin, "ORDER", "first"):
        try:
            PluginManager(None).discover()
        except Exception as exc:
            record("bad_plugin_metadata", escaped=f"{type(exc).__name__}: {exc}")
    window, _ = main_window("plugins")
    manager = PluginManager(window)

    class DeletedWindowPlugin(TrackerPlugin):
        def launch(self):
            widget = QWidget(self.ctx.window)
            widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            widget.setWindowFlags(Qt.WindowType.Window)
            widget.show()
            return widget

    rec = PluginRecord("deleted", "Deleted", cls=DeletedWindowPlugin)
    manager._records["deleted"] = rec
    manager.launch("deleted")
    rec.window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    try:
        manager.launch("deleted")
    except Exception as exc:
        record("deleted_plugin_window_relaunch", escaped=f"{type(exc).__name__}: {exc}")
    ctx = PluginContext(window, "leak")
    ctx.add_overlay(lambda painter, context: None)
    handler = CanvasInteraction()
    ctx.begin_canvas_interaction(handler)
    record_to_unload = PluginRecord("leak", "Leak", instance=TrackerPlugin(ctx))
    manager._unload(record_to_unload)
    record("manager_unload_resource_ownership", overlay_count=len(window.canvas._overlays),
           capture_retained=window.canvas._interaction is handler)
    ctx.end_canvas_interaction()
    for overlay in list(ctx._overlay_wrappers):
        ctx.remove_overlay(overlay)
    install_result(window)
    ctx = PluginContext(window, "read_only")
    before = ctx.roi_corners
    ctx.roi.reset()
    metrics = ctx.metrics()
    metrics.max_step[:] = -999
    record("sdk_mutable_objects", roi_changed=ctx.roi_corners != before,
           cached_metric=ctx.metrics().max_step.tolist())
    zones = AffineZonesWindow(ctx)
    WINDOWS.append(zones)
    zones.zones.append(Zone([(0, 0), (300, 0), (300, 200), (0, 200)], default_zone_color(0)))
    zones._refresh()
    cached = zones._get_fits()
    paths = window.state.sequence.paths
    window.load_sequence_from_paths(paths, str(Path(paths[0]).parent))
    record("zones_sequence_reset", zones_retained=len(zones.zones), cached_fit_retained=zones._get_fits() is cached,
           export_enabled=zones.export_btn.isEnabled(), result_present=ctx.has_result)
    zones.dispose()


def probe_mts():
    main = MainWindow()
    WINDOWS.append(main)
    window = MtsUniaxialWindow(PluginContext(main, "mts_audit"))
    WINDOWS.append(window)
    load_mts(window, "experiment_a")
    window._apply_reference(0, 3, "audit", 0)
    install_result(main)
    cache = window.kinematics()
    window.force_box.setCurrentIndex(0)
    record("mts_channel_invalidation", reference_valid=window.pstate.done(Step.REFERENCE),
           core_result_retained=main.state.result is not None, kinematics_cache_retained=window.kinematics() is cache)
    window._apply_reference(0, 3, "audit", 0)
    record("mts_same_reference", core_result_present=main.state.result is not None,
           completed_through=window.pstate.completed_through,
           trackers_file_exists=(Path(project_io.project_dir(window.pstate.root)) / "trackers.npz").exists())
    install_result(main)
    cache = window.kinematics()
    load_mts(window, "experiment_b")
    record("mts_fresh_load_cache", has_result=main.state.result is not None,
           old_cache_returned=window.kinematics() is cache)
    window.pstate.offset_ms = 100.
    with patch.object(window.crop_plot, "update_data") as update:
        window._refresh_crop_plot()
        record("mts_crop_plot_unshifted_sensor_time", offset_ms=100.,
               plotted_sensor_start_ms=float(update.call_args.args[0][0]),
               correct_sensor_start_ms=float(window.pstate.sensor.time_s[0] * 1000 + 100))
    window.pstate.offset_ms = 10000.
    window._apply_reference(0, 3, "audit outside coverage", 0)
    install_result(main)
    record("mts_no_sensor_coverage", in_range=window._aligned_force_disp()[2].tolist(),
           force_plotted=window.per_frame_force_N().tolist())
    # Resume reads the original stream again without checking it against the saved snapshot.
    sensor_file = window.pstate.sensor_file
    original = window.pstate.sensor
    _write_sensor(sensor_file, [original.time_s, original.disp_a, original.force_a + 100,
                               original.disp_b, original.force_b + 100],
                  ["Time", "Achse-2 Weg", "Achse-2 Kraft", "Achse-4 Weg", "Achse-4 Kraft"],
                  ["Sec", "mm", "N", "mm", "N"])
    resumed = project_io.load_project(window.pstate.root)
    record("mts_changed_raw_stream_resume", restored_step=resumed.completed_through,
           reference_zero=resumed.zero_force, new_first_force=float(resumed.sensor.force_a[0]))
    window.dispose()


def main():
    try:
        probe_math()
        probe_units()
        probe_energy()
        probe_export_transaction()
        probe_sequence_ui()
        probe_tracking_reentrancy()
        probe_tracker_validation()
        probe_plugin_lifecycle()
        probe_mts()
    finally:
        for window in reversed(WINDOWS):
            try:
                window.close()
            except RuntimeError:
                pass
        APP.processEvents()
        TEMP.cleanup()


if __name__ == "__main__":
    main()
