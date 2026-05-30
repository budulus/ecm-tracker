"""Headless regression checks for the tracker core pipeline.

Run with:  QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.test_pipeline
(QT platform only needed for the GUI flow check at the end.)
"""
import os
import sys
import tempfile

import numpy as np

from app.core import settings
from app.core.cleanup import (
    BandFilter,
    Metrics,
    Thresholds,
    build_mask,
    compute_metrics,
    default_thresholds,
    thresholds_from_dict,
    thresholds_to_dict,
)
from app.core.export import export, export_csv
from app.core.feature_detection import DEFAULT_SHI_TOMASI, regular_grid, shi_tomasi
from app.core.image_sequence import ImageSequence, discover, natural_sort_key
from app.core.roi import ROI
from app.core.tracking import DEFAULT_LK, track
from tests.synthetic import make_sequence


def test_natural_sort():
    names = ["img_10.png", "img_2.png", "img_1.png", "img_100.png"]
    assert sorted(names, key=natural_sort_key) == [
        "img_1.png",
        "img_2.png",
        "img_10.png",
        "img_100.png",
    ]


def _sequence(n=12, dx=2.0, dy=1.0):
    d = tempfile.mkdtemp(prefix="trk_")
    make_sequence(d, n_frames=n, dx=dx, dy=dy)
    return d, ImageSequence(discover(d))


def test_detection_inside_roi():
    _, seq = _sequence()
    gray = seq.load_gray(0)
    roi = ROI([(60, 50), (240, 50), (240, 180), (60, 180)])
    pts = shi_tomasi(gray, roi.mask(*gray.shape[:2]), DEFAULT_SHI_TOMASI)
    assert len(pts) > 20 and all(roi.contains(x, y) for x, y in pts)
    grid = regular_grid(roi, 20, 20)
    assert len(grid) > 0 and all(roi.contains(x, y) for x, y in grid)


def test_roi_general():
    # Empty ROI: not complete, empty mask, contains nothing.
    roi = ROI()
    assert not roi.is_complete
    assert roi.mask(50, 50).sum() == 0
    assert not roi.contains(10, 10)

    # N-Gon flow: add an arbitrary polyline, finalize with close() (any N >= 3).
    for p in [(10, 10), (40, 10), (25, 40)]:
        roi.add_corner(*p)
    assert not roi.is_complete  # not closed yet
    roi.close()
    assert roi.is_complete and len(roi.corners) == 3
    assert roi.mask(50, 50).sum() > 0 and roi.contains(25, 20)  # inside the triangle

    # Constructing with corners yields an already-closed ROI (the drag-tool/test path).
    rect = ROI([(10, 10), (40, 10), (40, 30), (10, 30)])
    assert rect.is_complete and rect.contains(25, 20) and not rect.contains(100, 100)
    # contains_many mirrors contains over many points at once.
    inside = rect.contains_many([(25, 20), (100, 100), (11, 11)])
    assert list(inside) == [True, False, True]

    # A degenerate 1–2 corner polygon is NOT complete (guards against a fillable empty region).
    assert not ROI([(10, 10)]).is_complete
    degenerate = ROI([(10, 10), (40, 10)])
    assert not degenerate.is_complete
    assert degenerate.mask(50, 50).sum() == 0 and not degenerate.contains(20, 10)

    # A many-vertex circle polygon behaves like any other polygon.
    cx, cy, r = 50.0, 50.0, 20.0
    circle = ROI(
        [
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ]
    )
    assert circle.is_complete and len(circle.corners) == 32
    assert circle.contains(cx, cy) and not circle.contains(cx + r + 10, cy)


def test_tracking_recovers_motion_and_fb():
    _, seq = _sequence(dx=2.0, dy=1.0)
    gray = seq.load_gray(0)
    roi = ROI([(60, 50), (240, 50), (240, 180), (60, 180)])
    seeds = shi_tomasi(gray, roi.mask(*gray.shape[:2]), DEFAULT_SHI_TOMASI)
    res = track(seq, 0, 11, seeds, DEFAULT_LK)
    assert res.coords_fw.shape == (12, res.n_points, 2)
    assert res.win_size == DEFAULT_LK["win_size"]  # recorded for the window-size overlay
    assert np.allclose(res.coords_bw[-1], res.coords_fw[-1])  # bw seeded from fw last frame
    ok = res.status_fw[1] == 1
    med = np.median(res.coords_fw[1][ok] - res.coords_fw[0][ok], axis=0)
    assert abs(med[0] - 2.0) < 0.4 and abs(med[1] - 1.0) < 0.4
    finite = np.isfinite(res.fb_mean_error)
    assert finite.sum() > 0.7 * res.n_points
    assert np.median(res.fb_mean_error[finite]) < 1.0
    assert track(seq, 0, 11, seeds, DEFAULT_LK, progress_cb=lambda a, b: True) is None


def _metrics():
    return Metrics(
        n_frames=10,
        n_points=4,
        fail_count_fw=np.array([0, 0, 5, 0]),
        fail_count_bw=np.array([0, 0, 0, 9]),
        max_err_fw=np.array([1.0, 2.0, np.inf, 3.0]),
        mean_err_fw=np.array([0.5, 1.0, np.inf, 1.5]),
        fb_mean=np.array([0.1, 5.0, np.inf, 0.2]),
        fb_max=np.array([0.2, 8.0, np.inf, 0.3]),
        max_step=np.array([2.0, 2.0, np.inf, 50.0]),
        left_image=np.array([False, False, True, True]),
        left_roi=np.array([False, True, False, False]),
    )


def test_build_mask_bands():
    m = _metrics()
    # factory defaults: every filter disabled -> keep all (including the inf point)
    assert build_mask(m, default_thresholds()).all()
    # enable an fb_mean max threshold of 1.0
    thr = Thresholds()
    thr.fb_mean = BandFilter(True, 1.0)
    assert list(build_mask(m, thr)) == [True, False, False, True]
    # add drop-left-image bool
    thr.drop_left_image = True
    assert list(build_mask(m, thr)) == [True, False, False, False]
    # distance max threshold drops the fast + inf points
    thr2 = Thresholds()
    thr2.distance = BandFilter(True, 10.0)
    assert list(build_mask(m, thr2)) == [True, True, False, False]
    # count band keeps points with <= 3 failures
    thr3 = Thresholds()
    thr3.fw_failures = BandFilter(True, 3)
    assert list(build_mask(m, thr3)) == [True, True, False, True]


def test_thresholds_serialization():
    m = _metrics()
    thr = Thresholds(drop_left_roi=True)
    thr.fb_max = BandFilter(True, 7.0, 8.0)
    rebuilt = thresholds_from_dict(thresholds_to_dict(thr))
    assert list(build_mask(m, rebuilt)) == list(build_mask(m, thr))
    assert build_mask(m, thresholds_from_dict({})).all()  # tolerant of empty/partial


def test_settings_persistence():
    d = tempfile.mkdtemp(prefix="cfg_")
    os.environ["TRACKER_CONFIG_DIR"] = d
    try:
        assert settings.get_section("lk") is None
        settings.update_section("lk", {"win_size": 31, "max_level": 2})
        settings.update_section("grid", {"spacing_x": 15})
        assert settings.get_section("lk") == {"win_size": 31, "max_level": 2}
        assert settings.load_settings()["grid"]["spacing_x"] == 15
    finally:
        os.environ.pop("TRACKER_CONFIG_DIR", None)


def test_export(tmp=None):
    cf = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
    mask = np.array([True, False, True, False, True])
    d = tempfile.mkdtemp(prefix="exp_")
    cpath, spath, shape = export(cf, mask, 12, 83, d, "coords")
    assert shape == (3, 3, 2)
    assert np.array_equal(np.load(cpath), cf[:, mask, :])
    assert open(spath).read().strip() == "12 83"


def test_export_csv(tmp=None):
    cf = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)
    mask = np.array([True, False, True, False, True])
    names = ["a.png", "b.png", "c.png"]
    d = tempfile.mkdtemp(prefix="expcsv_")
    cpath, shape = export_csv(cf, mask, names, d, "coords")
    assert shape == (3, 3, 2)
    assert cpath.endswith("coords.csv")

    lines = open(cpath).read().splitlines()
    assert lines[0] == "filename,p1x,p1y,p2x,p2y,p3x,p3y"
    assert len(lines) == 4  # header + 3 frames

    expected = cf[:, mask, :].reshape(3, 6)
    for i, name in enumerate(names):
        cells = lines[i + 1].split(",")
        assert cells[0] == name
        assert np.allclose(np.array(cells[1:], dtype=np.float32), expected[i])


def test_gui_pipeline():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["TRACKER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="cfg_")
    from PyQt5.QtCore import QPointF
    from PyQt5.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication(sys.argv)
    src, _ = _sequence()
    from app.gui.main_window import MainWindow

    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    w._load_paths(discover(src), src)
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    ngon.on_right_press(QPointF(60, 180), None)  # right-click closes the polygon
    w._detect_shi_tomasi()
    w._run_tracking()
    assert w.state.result is not None
    w._open_cleanup()
    # enable the OpenCV-error filter at the median so roughly half the points drop
    finite = w._cleanup_metrics.max_err_fw[np.isfinite(w._cleanup_metrics.max_err_fw)]
    thresh = round(float(np.median(finite)), 3)
    row = w._cleanup_dialog.opencv_error
    row.enable.setChecked(True)
    row.set_band(BandFilter(True, thresh, row.band().cap))
    applied = row.band().hi
    w._cleanup_apply()
    kept = int(w.state.active_mask.sum())
    assert 0 < kept < w.state.result.n_points, (kept, w.state.result.n_points, applied)
    w._cleanup_undo()
    assert int(w.state.active_mask.sum()) == w.state.result.n_points
    # Save parameters -> persists cleanup section; reopening loads it as defaults
    w._cleanup_dialog._on_save()
    saved = settings.get_section("cleanup")["opencv_error"]
    assert saved["enabled"] is True and "cap" in saved and "lo" not in saved
    w._cleanup_dialog.reject()
    app.processEvents()
    w._open_cleanup()
    assert w._cleanup_dialog.opencv_error.enable.isChecked()
    assert abs(w._cleanup_dialog.opencv_error.band().hi - applied) < 1e-6
    w._cleanup_dialog.reject()
    app.processEvents()

    out = os.path.join(tempfile.mkdtemp(prefix="out_"), "coords.npy")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))
    w._export()
    assert os.path.exists(out) and os.path.exists(
        os.path.join(os.path.dirname(out), "sequence.txt")
    )

    out_csv = os.path.join(tempfile.mkdtemp(prefix="outcsv_"), "coords.csv")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_csv, ""))
    w._export_csv()
    assert os.path.exists(out_csv)
    assert open(out_csv).readline().startswith("filename,p1x,p1y")


def test_roi_shape_tools():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["TRACKER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="cfg_")
    from PyQt5.QtCore import QPointF
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    src, _ = _sequence()
    from app.gui.main_window import MainWindow
    from app.gui.roi_tools import CIRCLE_SEGMENTS

    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    w._load_paths(discover(src), src)

    # Rectangle: full menu -> begin -> drag -> commit path.
    w._begin_roi_definition("rectangle")
    assert w.define_roi_action.isChecked()
    rect = w.canvas._interaction
    rect.on_press(QPointF(60, 50), None)
    rect.on_move(QPointF(240, 180), None)
    assert w.state.roi is not None and w.state.roi.is_complete  # live preview
    rect.on_release(QPointF(240, 180), None)
    assert w.state.roi.is_complete and len(w.state.roi.corners) == 4
    assert not w.define_roi_action.isChecked()

    # Circle: press center, drag out a radius -> a CIRCLE_SEGMENTS-gon.
    w._begin_roi_definition("circle")
    circ = w.canvas._interaction
    circ.on_press(QPointF(150, 110), None)
    circ.on_release(QPointF(150, 160), None)  # radius 50
    assert w.state.roi.is_complete and len(w.state.roi.corners) == CIRCLE_SEGMENTS
    assert w.state.roi.contains(150, 110)  # center inside

    # A degenerate (no-drag) gesture cancels the in-progress definition.
    w._begin_roi_definition("rectangle")
    tool = w.canvas._interaction
    tool.on_press(QPointF(10, 10), None)
    tool.on_release(QPointF(11, 11), None)
    assert w.state.roi is None and not w.define_roi_action.isChecked()

    # N-Gon: left-click arbitrary points (no preset count); right-click closes it.
    w._begin_roi_definition("ngon")
    assert w.define_roi_action.isChecked()
    ngon = w.canvas._interaction
    ngon.on_press(QPointF(60, 50), None)
    ngon.on_right_press(QPointF(60, 50), None)  # <3 points: must not close
    assert not w.state.roi.is_complete and w.define_roi_action.isChecked()
    for c in [(240, 50), (240, 180), (150, 220), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    assert not w.state.roi.is_complete  # left-clicks never auto-close
    assert w.define_roi_action.isChecked() and len(w.state.roi.corners) == 5
    ngon.on_right_press(QPointF(60, 180), None)  # right-click closes
    assert w.state.roi.is_complete and len(w.state.roi.corners) == 5
    assert not w.define_roi_action.isChecked()


def test_partial_ngon_discarded_on_frame_navigation():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["TRACKER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="cfg_")
    from PyQt5.QtCore import QPointF
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    src, _ = _sequence()
    from app.gui.main_window import MainWindow

    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    w._load_paths(discover(src), src)

    # Start an N-Gon and place only 2 corners without closing (definition still in progress).
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50)]:
        ngon.on_press(QPointF(*c), None)
    assert w.define_roi_action.isChecked() and not w.state.roi.is_complete

    # Navigating off the reference frame must discard the partial ROI, not orphan it.
    w._on_current_changed(1)
    assert w.state.roi is None
    assert not w.define_roi_action.isChecked()


def test_display_settings():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["TRACKER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="cfg_")
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    src, _ = _sequence()
    from app.gui.main_window import MainWindow
    from app.models.project_state import DEFAULT_DISPLAY, ProjectState

    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    # defaults merged onto fresh state
    assert set(w.state.display_params) == set(DEFAULT_DISPLAY)

    w._load_paths(discover(src), src)
    w._begin_roi_definition("ngon")
    ngon = w.canvas._interaction
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        ngon.on_press(QPointF(*c), None)
    ngon.on_right_press(QPointF(60, 180), None)  # right-click closes the polygon
    w._detect_shi_tomasi()
    w._run_tracking()
    assert w.state.result is not None

    # Each display config must paint without raising (render into an off-screen pixmap).
    def _paint():
        w.canvas.render(QPixmap(w.canvas.size()))

    for cfg in (
        dict(show_markers=True, marker_size=10, marker_opacity=50, show_window_box=True, show_roi=True),
        dict(show_markers=False, marker_size=1, marker_opacity=0, show_window_box=False, show_roi=False),
        dict(show_markers=True, marker_size=3, marker_opacity=100, show_window_box=True, show_roi=True),
    ):
        w.state.display_params = cfg
        _paint()

    # Live callback applies; cancel restores the pre-dialog snapshot.
    w.state.display_params = dict(DEFAULT_DISPLAY)
    from app.gui.dialogs import DisplayDialog

    seen = {}
    dlg = DisplayDialog(w.state.display_params, lambda v: seen.update(v), w)
    dlg.marker_size.setValue(12)
    assert seen["marker_size"] == 12
    assert dlg.values()["marker_size"] == 12
    dlg.reject()

    # The opened-dialog path saves on accept and restores on reject.
    w.state.display_params = dict(DEFAULT_DISPLAY, marker_size=7)
    settings.update_section("display", {"marker_opacity": 33})
    assert ProjectState().display_params["marker_opacity"] == 33


def test_app_icon():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # keep referenced (GC guard)
    assert app is not None
    from app.gui.icon_loader import load_app_icon

    icon = load_app_icon()
    assert not icon.isNull()
    assert icon.availableSizes()  # several sizes registered for window/taskbar use


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
