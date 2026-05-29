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
from app.core.export import export
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


def test_tracking_recovers_motion_and_fb():
    _, seq = _sequence(dx=2.0, dy=1.0)
    gray = seq.load_gray(0)
    roi = ROI([(60, 50), (240, 50), (240, 180), (60, 180)])
    seeds = shi_tomasi(gray, roi.mask(*gray.shape[:2]), DEFAULT_SHI_TOMASI)
    res = track(seq, 0, 11, seeds, DEFAULT_LK)
    assert res.coords_fw.shape == (12, res.n_points, 2)
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
    w.define_roi_action.setChecked(True)
    for c in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        w._on_image_clicked(QPointF(*c))
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
