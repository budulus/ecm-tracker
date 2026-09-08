"""Headless checks for the MTS Uniaxial plugin.

Run with:  QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.test_mts_uniaxial

The parsing / sync / state / project-io checks are Qt-free; the loader, reference and export
checks build a real MainWindow (offscreen). A module-level QApplication is kept referenced so
constructing widgets doesn't abort.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKER_CONFIG_DIR", tempfile.mkdtemp(prefix="cfg_"))

import csv

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from app.core.image_sequence import discover
from app.plugins import PluginContext
from tests.synthetic import make_sequence

import json

from plugins.mts_uniaxial import kinematics, parsers, project_io, sync
from plugins.mts_uniaxial.reference_algorithms import (
    ELASTIC_ENERGY,
    PREFORCE,
    REGISTRY,
    elastic_energy_minimum,
    fd_knee,
    force_onset,
    preforce_reference,
)
from plugins.mts_uniaxial.state import STEP_ARTIFACTS, MtsProjectState, Step

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _main_window():
    _app()
    from app.gui.main_window import MainWindow

    w = MainWindow()
    w.resize(900, 600)
    w.show()
    return w


# --------------------------------------------------------------------------- fixtures


_SENSOR_HEADER = (
    "MTS793|MPT|ENU|1|2|.|/|:|1|0|0|A\n"
    "\n"
    "Data Acquisition\t\t\t\t\t\tTime:\t10.0\tSec\t3/29/2016 1:54:32 PM\n"
)


def _write_sensor(path, cols, names, units):
    """Write a sensor .dat with the given (name, unit, values) columns."""
    with open(path, "w") as f:
        f.write(_SENSOR_HEADER)
        f.write("\t".join(names) + "\n")
        f.write("\t".join(units) + "\n")
        k = len(cols[0])
        for i in range(k):
            f.write("\t".join(f"{c[i]:.6g}" for c in cols) + "\n")


def make_mts_experiment(root, n_frames=10, reverse_log=False):
    """Build a synthetic experiment root matching the plugin defaults: veddac/ + mts/specimen.dat."""
    images_dir = os.path.join(root, "veddac")
    sensor_dir = os.path.join(root, "mts")
    os.makedirs(sensor_dir, exist_ok=True)
    paths = make_sequence(images_dir, n_frames=n_frames, dx=2.0, dy=1.0)
    names = [os.path.basename(p) for p in paths]  # motion / natural order
    log_names = list(reversed(names)) if reverse_log else names
    times = [i * 100 for i in range(len(log_names))]  # ms, 100 ms apart

    log_path = os.path.join(images_dir, "VDCCam.log")
    with open(log_path, "w") as f:
        f.write("Start\t2016-03-29\t14:02:51\t203\n")
        f.write("\tAbsolute\t\tSequence\n")
        f.write("File\tDate\tTime\tTime [hh:mm:ss]\tTime [ms]\n")
        for nm, t in zip(log_names, times):
            f.write(f"{nm}\t2016-03-29\t14:02:51\t00:00:00\t{t}\n")

    k = 200
    t_ms = times[-1]
    ts = np.linspace(0.0, t_ms / 1000.0, k)
    disp_a = np.linspace(0.0, 1.0, k)
    disp_b = np.linspace(0.0, 0.8, k)
    flat = max(1, k // 5)
    force_a = np.concatenate([np.full(flat, 0.01), np.linspace(0.01, 5.0, k - flat)])
    force_b = force_a * 0.9
    sensor_file = os.path.join(sensor_dir, "specimen.dat")
    _write_sensor(
        sensor_file,
        [ts, disp_a, force_a, disp_b, force_b],
        ["Time", "Achse-2 Weg", "Achse-2 Kraft Filter", "Achse-4 Weg", "Achse-4 Kraft Filter"],
        ["Sec", "mm", "N", "mm", "N"],
    )
    return dict(root=root, images_dir=images_dir, sensor_file=sensor_file,
                log_path=log_path, names=log_names)


def _plugin_window(w):
    from plugins.mts_uniaxial.window import MtsUniaxialWindow

    win = MtsUniaxialWindow(PluginContext(w, "mts_uniaxial"))
    win.show()  # showEvent connects the signal handlers
    return win


# --------------------------------------------------------------------------- parsers


def test_parse_real_assets():
    """The bundled mts_assets parse with the expected shapes, labels and units."""
    root = os.path.join(os.path.dirname(__file__), os.pardir, "mts_assets")
    log = parsers.parse_image_log(os.path.join(root, "images", "VDCCam.log"))
    assert log.n_images == 12
    assert log.time_ms[0] == 15.0 and log.time_ms[-1] == 5609.0

    sen = parsers.parse_sensor(os.path.join(root, "sensor", "specimen.dat"))
    assert sen.n_samples == 6
    assert sen.clamp_labels == ("Achse-2", "Achse-4")
    assert {k: sen.units[k] for k in ("time", "disp", "force")} == {"time": "Sec", "disp": "mm", "force": "N"}
    assert sen.n_skipped == 0 and sen.warnings == []


def test_parser_reordered_columns():
    """Clamp A is always the lowest Achse number, regardless of column order."""
    d = tempfile.mkdtemp(prefix="mts_")
    path = os.path.join(d, "specimen.dat")
    ts = np.linspace(0, 1, 5)
    # Achse-4 columns come first; force-A (Achse-2) values are the distinctive 7.x.
    _write_sensor(
        path,
        [ts, np.full(5, 9.0), np.full(5, 4.0), np.full(5, 2.0), np.full(5, 7.0)],
        ["Time", "Achse-4 Weg", "Achse-4 Kraft Filter", "Achse-2 Weg", "Achse-2 Kraft Filter"],
        ["Sec", "mm", "N", "mm", "N"],
    )
    sen = parsers.parse_sensor(path)
    assert sen.clamp_labels == ("Achse-2", "Achse-4")
    assert np.allclose(sen.force_a, 7.0)  # clamp A == Achse-2 column, not the first column
    assert np.allclose(sen.disp_a, 2.0)
    assert np.allclose(sen.force_b, 4.0)


def test_parser_positional_fallback():
    """A header with no recognisable 'Time' row falls back to fixed positions, with a warning."""
    d = tempfile.mkdtemp(prefix="mts_")
    path = os.path.join(d, "weird.dat")
    with open(path, "w") as f:
        f.write("garbage header line\n")
        for i in range(4):
            f.write(f"{i*0.1:.3f}\t{i}\t{i*2}\t{i*3}\t{i*4}\n")
    sen = parsers.parse_sensor(path)
    assert sen.n_samples == 4
    assert any("by position" in w for w in sen.warnings)
    assert np.allclose(sen.force_a, [0, 2, 4, 6])


def test_missing_image_raises():
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=4)
    log = parsers.parse_image_log(exp["log_path"])
    os.remove(os.path.join(exp["images_dir"], log.filenames[1]))
    try:
        parsers.resolve_image_paths(log, exp["images_dir"])
        assert False, "expected a missing-image error"
    except ValueError as e:
        assert "missing" in str(e)


# --------------------------------------------------------------------------- sync math


def test_offset_sign_and_in_range():
    sensor_t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ramp = sensor_t.copy()
    img_t = np.array([2.0])
    base, inr = sync.interp_to_images(img_t, sensor_t, ramp, 0.0)
    pos, _ = sync.interp_to_images(img_t, sensor_t, ramp, 1.0)  # +offset => earlier sensor sample
    assert base[0] == 2.0 and pos[0] == 1.0 and inr[0]
    # outside coverage clamps and flags
    vals, inr2 = sync.interp_to_images(np.array([-1.0, 5.0]), sensor_t, ramp, 0.0)
    assert list(vals) == [0.0, 4.0] and not inr2.any()


def test_composite_and_zero():
    class S:
        disp_a = np.array([1.0, 2.0, 3.0])
        disp_b = np.array([0.5, 0.5, 0.5])
        force_a = np.array([10.0, 20.0, 30.0])
        force_b = np.array([2.0, 4.0, 6.0])

    s = S()
    assert np.allclose(sync.composite_displacement(s), [1.5, 2.5, 3.5])
    assert np.allclose(sync.composite_force(s, "A"), [10, 20, 30])
    assert np.allclose(sync.composite_force(s, "average"), [6, 12, 18])


def test_sensor_index_to_image():
    sensor_t = np.linspace(0, 1000, 101)  # dense
    img_t = np.array([0.0, 100.0, 200.0, 300.0])  # coarse
    # sensor sample at 220 ms -> nearest image is index 2 (200 ms)
    idx = np.argmin(np.abs(sensor_t - 220))
    assert sync.sensor_index_to_image_index(idx, sensor_t, img_t, 0.0) == 2


# --------------------------------------------------------------------------- reference algos


def test_reference_algorithms():
    assert set(REGISTRY) and force_onset.__name__ and fd_knee.__name__
    force = np.concatenate([np.zeros(20), np.linspace(0.1, 5.0, 30)])  # flat, then a clear jump
    disp = np.linspace(0, 1, 50)
    assert force_onset(disp, force) == 20
    assert 0 < fd_knee(disp, force) < 49
    # guards
    assert force_onset(np.array([]), np.array([])) == 0
    assert fd_knee(np.array([0.0]), np.array([0.0])) == 0

    # "Set preforce": registered, and returns the first sample strictly above the threshold.
    assert PREFORCE in REGISTRY
    assert preforce_reference(disp, force, 0.05) == 20
    assert preforce_reference(disp, force, 100.0) == 0  # nothing exceeds -> 0
    assert preforce_reference(np.array([]), np.array([]), 0.0) == 0


def test_elastic_energy_reference_algorithm():
    # registered and discoverable via the standard 2-arg path
    assert ELASTIC_ENERGY in REGISTRY
    assert REGISTRY[ELASTIC_ENERGY] is elastic_energy_minimum

    # smooth toe -> linear tensile branch; assert the contract, not the model's exact frame
    n = 200
    u = np.linspace(0.0, 2.0, n)
    toe = 0.4
    f = np.where(u < toe, 0.5 * (u / toe) ** 2, 0.5 + 4.0 * (u - toe))
    idx = elastic_energy_minimum(u, f)
    assert isinstance(idx, int) and 0 <= idx < n  # in range, no exception

    # defensive contract: too-short input -> 0; degenerate (constant force) window doesn't raise
    assert elastic_energy_minimum(u[:10], f[:10]) == 0
    j = elastic_energy_minimum(u, np.zeros(n))
    assert isinstance(j, int) and 0 <= j < n


# --------------------------------------------------------------------------- state machine


def test_invalidate_from_wipes_downstream():
    st = MtsProjectState()
    st.force_channel = "A"
    st.crop_start, st.crop_end = 2, 90
    st.ref_image_global, st.last_image_global = 3, 9
    st.zero_disp, st.zero_force = 1.0, 2.0
    st.completed_through = int(Step.EXPORT)

    files = st.invalidate_from(Step.REFERENCE)
    # reference + track + export artifacts wiped; channel + crop kept
    assert set(files) == (
        set(STEP_ARTIFACTS[Step.REFERENCE])
        | set(STEP_ARTIFACTS[Step.TRACK])
        | set(STEP_ARTIFACTS[Step.EXPORT])
    )
    assert st.ref_image_global is None and st.zero_disp is None
    assert st.force_channel == "A" and st.crop_start == 2
    assert st.completed_through == int(Step.CROP)

    files2 = st.invalidate_from(Step.CROP)
    assert "crop.json" in files2 and st.crop_start is None
    assert st.completed_through == int(Step.CHANNEL)

    st.invalidate_from(Step.LOAD)
    assert st.root is None and st.completed_through == -1


# --------------------------------------------------------------------------- project io


def test_project_io_round_trip():
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=6)
    log = parsers.parse_image_log(exp["log_path"])
    sen = parsers.parse_sensor(exp["sensor_file"])
    ordered = parsers.resolve_image_paths(log, exp["images_dir"])

    st = MtsProjectState()
    st.root, st.images_dir, st.sensor_file, st.log_path = (
        exp["root"], exp["images_dir"], exp["sensor_file"], exp["log_path"])
    st.image_log, st.sensor, st.ordered_paths = log, sen, ordered
    st.force_channel, st.offset_ms = "A", 12.5
    st.crop_start, st.crop_end = 5, sen.n_samples - 1
    st.ref_image_global, st.last_image_global = 2, 5
    st.ref_sensor_index, st.ref_algorithm = 7, "manual override"
    st.zero_disp, st.zero_force = 0.5, 1.5
    st.completed_through = int(Step.REFERENCE)
    project_io.save_load(st)
    project_io.save_channel(st)
    project_io.save_crop(st)
    project_io.save_reference(st)

    loaded = project_io.load_project(exp["root"])
    assert loaded is not None
    assert loaded.force_channel == "A" and loaded.offset_ms == 12.5
    assert loaded.crop_start == 5 and loaded.crop_end == sen.n_samples - 1
    assert loaded.ref_image_global == 2 and loaded.last_image_global == 5
    assert loaded.completed_through == int(Step.REFERENCE)
    assert loaded.ordered_paths == ordered  # re-parsed in log order

    # Per-step files are inspection snapshots; the committed manifest is authoritative.
    os.remove(os.path.join(project_io.project_dir(exp["root"]), "reference.json"))
    capped = project_io.load_project(exp["root"])
    assert capped.completed_through == int(Step.REFERENCE)


# --------------------------------------------------------------------------- API + GUI wiring


def test_load_sequence_api_preserves_order():
    w = _main_window()
    ctx = PluginContext(w, "t")
    d = tempfile.mkdtemp(prefix="seq_")
    paths = make_sequence(d, n_frames=5)
    custom = list(reversed(paths))  # deliberately not natural-sorted
    seen = []
    w.signals.sequence_changed.connect(lambda: seen.append(1))
    ctx.load_sequence(custom, d)
    assert w.state.sequence.paths == custom
    assert custom != discover(d)  # proves order is preserved, not re-sorted
    assert seen


def test_loads_in_log_order():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=8, reverse_log=True)
    win._set_root_candidate(exp["root"])
    win._on_load()
    expected = [os.path.join(exp["images_dir"], n) for n in exp["names"]]
    assert w.state.sequence.paths == expected
    assert expected != discover(exp["images_dir"])  # log order differs from filename sort
    win.close()


def test_sets_reference_and_last():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    assert win.pstate.done(Step.CROP)  # channel + crop defaulted on load

    win._on_detect_reference()
    st = win.pstate
    assert st.done(Step.REFERENCE)
    assert w.state.reference_index == st.ref_image_global
    assert w.state.last_index == st.last_image_global
    assert 0 <= w.state.reference_index <= w.state.last_index < w.state.total_images
    assert w.state.roi is None  # set_reference_frame clears the ROI

    # Narrowing the reference search sub-window keeps the detected reference inside it
    # (guards the local -> absolute sub-window transform).
    win.ref_lo.setValue(30)
    win.ref_hi.setValue(60)
    win._on_detect_reference()
    assert win.pstate.ref_sensor_index is not None
    assert 30 <= win.pstate.ref_sensor_index <= 60
    win.close()


def test_export_after_tracking():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    win._on_detect_reference()

    # Track in the core app over the plugin-set range.
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    ngon.on_right_press(QPointF(60, 180), None)
    w._detect_shi_tomasi()
    w._run_tracking()
    assert w.state.result is not None
    assert win.pstate.done(Step.TRACK)  # result_changed bumped progress

    win._on_export()
    assert win.pstate.completed_through == int(Step.EXPORT)
    pdir = project_io.project_dir(exp["root"])
    pdir = os.path.dirname(project_io.export_paths(win.pstate)[0])
    for name in ("tracked_coords.npy", "point_indices.npy", "aligned_data.csv"):
        assert os.path.isfile(os.path.join(pdir, name)), name

    with open(os.path.join(pdir, "aligned_data.csv")) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["frame_global", "image_time_ms", "displacement", "force", "in_range"]
    assert len(rows) - 1 == win.ctx.frame_count
    for r in rows[1:]:
        assert np.isfinite(float(r[2])) and np.isfinite(float(r[3]))
    # the reference frame (first tracked row) is the zero-stress state
    assert int(rows[1][0]) == w.state.reference_index
    assert abs(float(rows[1][2])) < 1e-9 and abs(float(rows[1][3])) < 1e-9

    coords = np.load(os.path.join(pdir, "tracked_coords.npy"))
    assert coords.shape == (win.ctx.frame_count, win.ctx.n_active, 2)
    win.close()


def test_resume_restores_trackers():
    """Tracking persists trackers.npz; resuming a project reinstalls the in-app result."""
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    win._on_detect_reference()
    _track_in_app(w)
    assert w.state.result is not None
    assert win.pstate.done(Step.TRACK)
    pdir = project_io.project_dir(exp["root"])
    assert os.path.isfile(os.path.join(pdir, "trackers.npz"))  # persisted on track

    exp_coords = w.state.result.coords_fw.copy()
    exp_ref, exp_last = w.state.reference_index, w.state.last_index
    n_pts = w.state.result.n_points
    win.close()

    # load_project now reports TRACK as resumable because trackers.npz is present.
    loaded = project_io.load_project(exp["root"])
    assert loaded is not None and loaded.done(Step.TRACK)

    # A fresh window + plugin adopts the saved project and restores the in-app tracking result.
    w2 = _main_window()
    win2 = _plugin_window(w2)
    win2._adopt_state(loaded)
    assert w2.state.result is not None
    assert w2.state.result.n_points == n_pts
    assert (w2.state.reference_index, w2.state.last_index) == (exp_ref, exp_last)
    assert np.allclose(w2.state.result.coords_fw, exp_coords)
    assert win2.ctx.has_result and win2.pstate.done(Step.TRACK)

    # Export works immediately — no re-tracking needed — and the prior export wasn't wiped.
    win2._on_export()
    assert win2.pstate.completed_through == int(Step.EXPORT)
    assert os.path.isfile(os.path.join(pdir, "trackers.npz"))  # still present after export
    win2.close()


def test_channel_change_invalidates_reference():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    win._on_detect_reference()
    assert win.pstate.done(Step.REFERENCE)
    pdir = project_io.project_dir(exp["root"])
    assert os.path.isfile(os.path.join(pdir, "reference.json"))

    # Changing the force channel must wipe the reference (downstream) but keep the crop.
    idx = win.force_box.findData("A")
    win.force_box.setCurrentIndex(idx)
    assert win.pstate.ref_image_global is None
    assert win.pstate.completed_through == int(Step.CROP)
    assert not os.path.isfile(os.path.join(pdir, "reference.json"))
    win.close()


def test_crop_plot_builds():
    from plugins.mts_uniaxial.crop_plot import (
        CropPlotWidget,
        ReferencePlotWidget,
        matplotlib_available,
    )

    if not matplotlib_available():
        print("  (matplotlib unavailable — crop plot test skipped)")
        return
    _app()
    widget = CropPlotWidget()
    assert widget.available
    t = np.linspace(0, 100, 50)
    disp = np.linspace(0, 1, 50)
    force = np.linspace(0, 5, 50)
    img = dict(img_t=np.array([0.0, 50.0, 100.0]),
               img_disp=np.array([0.0, 0.5, 1.0]),
               img_force=np.array([0.0, 2.5, 5.0]))
    widget.update_data(t, disp, force, 10, 40, **img)
    widget.update_data(t, disp, force, 10, 40, show_images=False, ref=1, **img)  # toggle + marker

    ref_widget = ReferencePlotWidget()
    assert ref_widget.available
    ref_widget.update_data(disp, force, 10, 40, ref_disp=0.5, ref_force=2.5)


# --------------------------------------------------------------------------- kinematics math


def _grid(n=5, span=4.0):
    xs, ys = np.meshgrid(np.linspace(0, span, n), np.linspace(0, span, n))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_kinematics_known_F():
    """A known homogeneous F is recovered exactly: stretches, strains and tensile direction."""
    ref = _grid()
    F = np.diag([1.20, 0.90])
    cur = ref @ F.T + np.array([3.0, -2.0])  # x' = F x + t
    coords = np.stack([ref, cur])
    s = kinematics.compute_series(coords, np.array([0.0, 100.0]))
    assert abs(s.lambda_1[1] - 1.20) < 1e-6 and abs(s.lambda_2[1] - 0.90) < 1e-6
    assert abs(s.eps_1[1] - 0.20) < 1e-6 and abs(s.eps_2[1] + 0.10) < 1e-6
    assert abs(abs(s.v1[1][0]) - 1.0) < 1e-6 and abs(s.v1[1][1]) < 1e-6  # v1 ≈ ±x
    assert abs(s.time_s[1] - 0.1) < 1e-12  # ms → s, zeroed at the reference
    assert s.lambda_1[0] == 1.0 and s.eps_1[0] == 0.0  # reference frame is identity


def test_kinematics_rotation_invariance():
    """Stretches are rotation-invariant; the current-config tensile direction rotates with F."""
    theta = np.deg2rad(30.0)
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    F = R @ np.diag([1.25, 0.85])
    ref = _grid(4, 3.0)
    coords = np.stack([ref, ref @ F.T])
    s = kinematics.compute_series(coords, np.array([0.0, 1.0]))
    assert abs(s.lambda_1[1] - 1.25) < 1e-6 and abs(s.lambda_2[1] - 0.85) < 1e-6
    ang = np.arctan2(s.v1[1][1], s.v1[1][0])  # eigenvector sign is arbitrary → compare modulo π
    d = (ang - theta) % np.pi
    assert min(d, np.pi - d) < 1e-6

    # Pure rigid rotation has equal stretches, so no principal-strain direction exists.
    rigid = np.stack([ref, ref @ R.T])
    rigid_series = kinematics.compute_series(rigid, np.array([0.0, 1.0]))
    assert np.isnan(rigid_series.angle_deg[1])
    assert np.isnan(rigid_series.v1[1]).all()


def test_eps_2_incompressible_formula():
    out = kinematics.eps_2_incompressible(np.array([0.0, 0.21, -0.5, -1.0, -2.0]))
    assert abs(out[0]) < 1e-12
    assert abs(out[1] - (1.21 ** -0.5 - 1.0)) < 1e-12
    assert abs(out[2] - (0.5 ** -0.5 - 1.0)) < 1e-12
    assert np.isnan(out[3]) and np.isnan(out[4])  # 1 + eps_1 <= 0 is undefined


def test_kinematics_degenerate_frame():
    """A frame with < 3 valid points yields a NaN row; the reference is still identity."""
    ref = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], float)
    coords = np.stack([ref, ref * 1.1])
    status = np.ones((2, 5), dtype=np.uint8)
    status[1, 2:] = 0  # only points 0,1 valid at frame 1
    s = kinematics.compute_series(coords, np.array([0.0, 1.0]), status)
    assert s.lambda_1[0] == 1.0 and s.lambda_2[0] == 1.0
    assert np.isnan(s.lambda_1[1]) and np.isnan(s.eps_1[1]) and np.isnan(s.angle_deg[1])
    assert s.n_points[1] == 2


def test_affine_fit_rejects_collinear_points():
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    cur = ref * np.array([1.2, 0.8]) + np.array([2.0, 1.0])
    assert kinematics.fit_deformation_gradient(ref, cur) is None
    M, inliers = kinematics.ransac_affine(ref, cur, sample_size=3)
    assert M is None and not inliers.any()


def test_ransac_drops_planted_outlier():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 100, size=(40, 2))
    F = np.array([[1.10, 0.02], [0.0, 0.95]])
    dst = ref @ F.T + np.array([5.0, -3.0])
    dst[7] += np.array([40.0, -35.0])  # gross outlier
    M, inliers = kinematics.ransac_affine(ref, dst, sample_size=6, reproj=2.0,
                                          max_iters=2000, confidence=0.999)
    assert M is not None
    assert not inliers[7] and inliers.sum() >= 38


# --------------------------------------------------------------------------- material params


def test_material_params_survive_invalidate():
    """Width/thickness/incompressible are parameters, not a Step — invalidate_from leaves them be."""
    st = MtsProjectState()
    st.material_width, st.material_thickness, st.incompressible = 7.0, 0.3, False
    st.ref_image_global, st.zero_force = 4, 9.0
    st.completed_through = int(Step.EXPORT)
    st.invalidate_from(Step.REFERENCE)
    assert st.material_width == 7.0 and st.material_thickness == 0.3 and st.incompressible is False
    assert st.ref_image_global is None and st.zero_force is None  # reference fields did reset
    assert abs(st.reference_area_mm2 - 2.1) < 1e-12


def test_project_io_persists_material():
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=6)
    log = parsers.parse_image_log(exp["log_path"])
    sen = parsers.parse_sensor(exp["sensor_file"])
    ordered = parsers.resolve_image_paths(log, exp["images_dir"])
    st = MtsProjectState()
    st.root, st.images_dir, st.sensor_file, st.log_path = (
        exp["root"], exp["images_dir"], exp["sensor_file"], exp["log_path"])
    st.image_log, st.sensor, st.ordered_paths = log, sen, ordered
    st.crop_start, st.crop_end = 0, sen.n_samples - 1
    st.material_width, st.material_thickness, st.incompressible = 8.0, 0.25, False
    st.completed_through = int(Step.CROP)
    project_io.save_load(st)
    project_io.save_channel(st)
    project_io.save_crop(st)

    loaded = project_io.load_project(exp["root"])
    assert loaded.material_width == 8.0 and loaded.material_thickness == 0.25
    assert loaded.incompressible is False

    # A manifest predating the material keys falls back to the defaults (10 / 0.5 / True).
    mpath = os.path.join(project_io.project_dir(exp["root"]), project_io.MANIFEST)
    with open(mpath) as f:
        man = json.load(f)
    for k in ("material_width", "material_thickness", "incompressible"):
        man.pop(k, None)
    with open(mpath, "w") as f:
        json.dump(man, f)
    loaded2 = project_io.load_project(exp["root"])
    assert loaded2.material_width == 10.0 and loaded2.material_thickness == 0.5
    assert loaded2.incompressible is True


def test_resume_rejects_out_of_range_reference():
    # A stale/hand-edited reference.json whose ref_image_global lies past the re-parsed image log
    # must not be adopted: the resume caps at CROP instead of raising IndexError downstream.
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=6)
    log = parsers.parse_image_log(exp["log_path"])
    sen = parsers.parse_sensor(exp["sensor_file"])
    ordered = parsers.resolve_image_paths(log, exp["images_dir"])
    st = MtsProjectState()
    st.root, st.images_dir, st.sensor_file, st.log_path = (
        exp["root"], exp["images_dir"], exp["sensor_file"], exp["log_path"])
    st.image_log, st.sensor, st.ordered_paths = log, sen, ordered
    st.crop_start, st.crop_end = 0, sen.n_samples - 1
    st.ref_image_global, st.last_image_global = 2, 5
    st.completed_through = int(Step.REFERENCE)
    project_io.save_load(st)
    project_io.save_channel(st)
    project_io.save_crop(st)
    project_io.save_reference(st)

    # Corrupt the saved reference to point past the end of the (re-parsed) image log.
    rpath = os.path.join(project_io.project_dir(exp["root"]), project_io.MANIFEST)
    with open(rpath) as f:
        ref = json.load(f)
    ref["ref_image_global"] = log.n_images + 3
    with open(rpath, "w") as f:
        json.dump(ref, f)

    capped = project_io.load_project(exp["root"])  # must not raise IndexError
    assert capped is not None
    assert capped.completed_through == int(Step.CROP)


def test_resume_rejects_null_material_param():
    # A manifest with an explicit null material field is malformed: resume degrades to "start
    # fresh" (None) rather than crashing on float(None).
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=6)
    log = parsers.parse_image_log(exp["log_path"])
    sen = parsers.parse_sensor(exp["sensor_file"])
    ordered = parsers.resolve_image_paths(log, exp["images_dir"])
    st = MtsProjectState()
    st.root, st.images_dir, st.sensor_file, st.log_path = (
        exp["root"], exp["images_dir"], exp["sensor_file"], exp["log_path"])
    st.image_log, st.sensor, st.ordered_paths = log, sen, ordered
    st.crop_start, st.crop_end = 0, sen.n_samples - 1
    st.completed_through = int(Step.CROP)
    project_io.save_load(st)
    project_io.save_channel(st)
    project_io.save_crop(st)

    mpath = os.path.join(project_io.project_dir(exp["root"]), project_io.MANIFEST)
    with open(mpath) as f:
        man = json.load(f)
    man["material_width"] = None
    with open(mpath, "w") as f:
        json.dump(man, f)

    assert project_io.load_project(exp["root"]) is None  # must not raise TypeError


# --------------------------------------------------------------------------- kinematics GUI wiring


def _track_in_app(w):
    """Define a 4-corner ROI, seed Shi-Tomasi points and run tracking (mirrors the export test)."""
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    ngon.on_right_press(QPointF(60, 180), None)
    w._detect_shi_tomasi()
    w._run_tracking()


def test_measures_export():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    win._on_detect_reference()
    _track_in_app(w)
    assert w.state.result is not None

    series = win.kinematics()
    assert series is not None and series.lambda_1[0] == 1.0

    win._on_export_measures()
    mpath = os.path.join(project_io.project_dir(exp["root"]), "measures.csv")
    assert os.path.isfile(mpath)
    with open(mpath) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "frame_global" and "time_s" in header
    assert len(rows) - 1 == win.ctx.frame_count
    ti = header.index("time_s")
    assert abs(float(rows[1][ti])) < 1e-9  # reference row is t = 0
    if "force_N" in header:
        assert abs(float(rows[1][header.index("force_N")])) < 1e-9  # and zero force
    if "pk_stress_MPa" in header:
        assert np.isfinite(float(rows[1][header.index("pk_stress_MPa")]))
    assert "measures.csv" in STEP_ARTIFACTS[Step.EXPORT]  # wiped with the aligned data
    win.width_edit.setText("12.5")
    win._on_material_param_changed()
    assert not os.path.exists(mpath)  # stress table used the old cross-section and is now stale
    win.close()


def test_panels_gate_and_refresh():
    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    assert not win.sec_ransac.isEnabled() and not win.sec_plot.isEnabled()  # no result yet
    win._on_detect_reference()
    _track_in_app(w)
    assert win.sec_ransac.isEnabled() and win.sec_plot.isEnabled()
    assert win.measures_btn.isEnabled()
    assert win.kinematics() is not None and win._kin_cache is not None

    # A mask change (here a trivial keep) must invalidate the cache via the mask_changed wiring.
    keep = np.ones(win.ctx.n_active, dtype=bool)
    if keep.size > kinematics.MIN_FIT_POINTS:
        keep[0] = False
        win.ctx.apply_keep_mask(keep)
        assert win._kin_cache is None

    # Exercise the gauge (no matplotlib needed) and, if available, the plot windows.
    from plugins.mts_uniaxial.crop_plot import matplotlib_available
    from plugins.mts_uniaxial import gauge
    g = gauge.DirectionGaugeWindow(win)
    g.slider.setValue(min(2, win.ctx.frame_count - 1))
    g.close()
    if matplotlib_available():
        from plugins.mts_uniaxial import plots
        for win_ctor in (lambda: plots.KinematicsPlotWindow(win),
                         lambda: plots.StressPlotWindow(win, "pk"),
                         lambda: plots.StressPlotWindow(win, "cauchy")):
            pw = win_ctor()
            pw.replot()
            pw.close()
    win.close()


def test_mask_changes_persist_and_invalidate_export_while_plugin_is_closed():
    from app.core.tracker_io import load_trackers

    w = _main_window()
    win = _plugin_window(w)
    d = tempfile.mkdtemp(prefix="mts_mask_")
    exp = make_mts_experiment(d, n_frames=10)
    win._set_root_candidate(exp["root"])
    win._on_load()
    win._on_detect_reference()
    _track_in_app(w)
    win._on_export()
    pdir = project_io.project_dir(exp["root"])
    export_path = project_io.export_paths(win.pstate)[0]
    assert os.path.isfile(export_path)

    # Closing hides the UI but the cached plugin instance must continue protecting its project.
    win.close()
    keep = np.ones(win.ctx.n_active, dtype=bool)
    keep[0] = False
    win.ctx.apply_keep_mask(keep)
    saved = load_trackers(os.path.join(pdir, "trackers.npz"))
    assert np.array_equal(saved["active_mask"], w.state.active_mask)
    assert win.pstate.export_generation is None  # Old generation remains recoverable but uncommitted.
    assert win.pstate.completed_through == int(Step.TRACK)

    # An external sequence change while hidden must detach the old MTS state immediately.
    other = tempfile.mkdtemp(prefix="other_sequence_")
    make_sequence(other, n_frames=3)
    w._load_paths(discover(other), other)
    assert not win.pstate.done(Step.LOAD)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
