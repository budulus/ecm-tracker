"""Capture real Qt widgets for the offline manual using disposable demonstration data.

Run from the repository: .venv/bin/python scripts/capture_help_screenshots.py
All experiment files and settings stay in a temporary directory. Only the manual's
PNG assets are written into the repository. No personal plugins are loaded.
"""
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix="ecm-manual-") as temporary:
        scratch = Path(temporary)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["QT_SCALE_FACTOR"] = "2"
        os.environ["TRACKER_CONFIG_DIR"] = str(scratch / "config")
        os.environ["MPLCONFIGDIR"] = str(scratch / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(scratch / "cache")
        capture(scratch)


def capture(scratch):
    import cv2
    import numpy as np
    from PySide6.QtCore import QPointF, QRect, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from unittest.mock import patch
    from app.core.cleanup import BandFilter
    from app.core.image_pair import AlignedImagePairSequence, save_pair_setup, load_pair_setup
    from app.core.pair_transform import PairAlignment
    from app.core.roi import ROI
    from app.core.export import export, export_csv
    from app.gui.main_window import MainWindow
    from app.gui.pair_alignment import PairAlignmentDialog
    from app.gui.theme import apply_theme
    from app.plugins.api import PluginContext
    from plugins.affine_zones.zones import AffineZonesWindow
    from plugins.mts_uniaxial.window import MtsUniaxialWindow
    from plugins.pressure_strain.window import PressureStrainWindow
    from plugins.pressure_strain.analysis import export_plot_csv
    from tests.test_mts_uniaxial import make_mts_experiment

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("ECM Tracker")
    apply_theme(app)
    assets = ROOT / "app" / "gui" / "help-assets"
    assets.mkdir(exist_ok=True)

    def settle():
        for _ in range(5):
            app.processEvents()

    def shot(widget, name, rect=None):
        settle()
        pixmap = widget.grab(rect) if rect is not None else widget.grab()
        assert pixmap.save(str(assets / name)), name
        print(f"Captured {name}: {pixmap.width()} × {pixmap.height()} px", flush=True)

    experiment = make_mts_experiment(str(scratch / "demonstration"), n_frames=12)
    image_dir = Path(experiment["images_dir"])
    # A reproducible speckle specimen with gentle extension and a locally slipping patch.
    rng = np.random.default_rng(2026)
    base = np.full((420, 640, 3), 207, dtype=np.uint8)
    for _ in range(2500):
        x, y = rng.integers([0, 0], [640, 420])
        value = int(rng.integers(35, 165))
        cv2.circle(base, (int(x), int(y)), int(rng.integers(1, 4)), (value,) * 3, -1, cv2.LINE_AA)
    base = cv2.GaussianBlur(base, (3, 3), .5)
    for t, name in enumerate(experiment["names"]):
        matrix = np.float32([[1 + .007 * t, 0, -.007 * t * 320], [0, 1 - .002 * t, .002 * t * 210]])
        frame = cv2.warpAffine(base, matrix, (640, 420), borderMode=cv2.BORDER_REFLECT)
        if t:
            shifted = cv2.warpAffine(frame, np.float32([[1, 0, 1.1*t], [0, 1, .3*t]]), (640, 420))
            frame[120:220, 410:540] = shifted[120:220, 410:540]
        path = image_dir / name
        assert cv2.imwrite(str(path), frame)
        os.utime(path, (1700000000 + t*.1, 1700000000 + t*.1))

    host = MainWindow()
    host.menuBar().setNativeMenuBar(False)
    host.resize(1600, 850)
    host.show()
    host.activateWindow()
    settle()
    with patch("app.gui.main_window.QDesktopServices.openUrl", return_value=True) as open_url:
        QTest.keyClick(host, Qt.Key.Key_F1)
        settle()
        open_url.assert_called_once()
        assert Path(open_url.call_args.args[0].toLocalFile()) == ROOT / "app" / "gui" / "help.html"
    paths = [str(image_dir / name) for name in experiment["names"]]
    assert host.load_sequence_from_paths(paths, str(image_dir))
    host.state.roi = ROI([(65, 55), (575, 55), (575, 365), (65, 365)])
    host.state.shi_tomasi_params.update(maxCorners=160, minDistance=18)
    host._detect_shi_tomasi()
    host._run_tracking()
    assert host.state.result is not None
    host._go_to_frame(11)
    host.statusBar().clearMessage()
    shot(host, "workspace.png")

    host._open_cleanup()
    cleanup = host._cleanup_dialog
    cleanup.distance.set_band(BandFilter(True, 2.5, 5.0))
    host._cleanup_preview()
    shot(cleanup, "cleanup-controls.png")
    # A real canvas crop makes the two rejected points visible beside the dialog.
    dropped = host.state.result.coords_fw[-1, ~host._preview_keep]
    center = np.mean(dropped, axis=0)
    center = host.canvas.image_to_screen(float(center[0]), float(center[1]))
    crop = QRect(max(0, min(int(center.x()) - 150, host.canvas.width() - 300)),
                 max(0, min(int(center.y()) - 249, host.canvas.height() - 498)), 300, 498)
    shot(host.canvas, "cleanup-preview.png", crop)
    original_mask = host.state.active_mask.copy()
    host._cleanup_apply()
    assert host.state.active_mask.sum() < original_mask.sum()
    host._cleanup_undo()
    np.testing.assert_array_equal(host.state.active_mask, original_mask)
    cleanup.close()

    snapshot = scratch / "trackers.npz"
    host.save_trackers_to(snapshot)
    host.load_trackers_from(snapshot)
    mask = host._exportable_mask()
    export(host.state.result.coords_fw, mask, 0, 11, str(scratch))
    export_csv(host.state.result.coords_fw, mask, experiment["names"], str(scratch))

    zones = AffineZonesWindow(PluginContext(host, "affine_zones"))
    zones.resize(820, 260)
    zones.show()
    zones._start_zone()
    for x, y in ((65, 55), (345, 55), (345, 365), (65, 365)):
        zones._tool.on_press(QPointF(x, y), None)
    zones._finish_zone(continue_drawing=True)
    for x, y in ((365, 55), (575, 55), (575, 365), (365, 365)):
        zones._tool.on_press(QPointF(x, y), None)
    zones._finish_zone()
    host._go_to_frame(11)
    zones.table.selectRow(0)
    shot(zones, "zones-controls.png")
    zones._open_gauge()
    zones._gauge_window.resize(820, 520)
    zones._gauge_window.slider.setValue(11)
    shot(zones._gauge_window, "zones-gauge.png")
    zones.close()

    pressure_path = scratch / "pressure.csv"
    pressure_path.write_text("Elapsed_s,Raw_mbar\n0,0\n0.4,30\n0.8,75\n1.1,100\n", encoding="utf-8")
    pressure = PressureStrainWindow(PluginContext(host, "pressure_strain"))
    assert pressure.load_pressure_file(str(pressure_path))
    series = pressure.current_plot_series()
    assert series is not None and len(series.strain) == 12
    export_plot_csv(str(scratch / "pressure-strain.csv"), series, paths)
    pressure.close()

    pair = AlignedImagePairSequence([paths[0], paths[-1]], alignment=PairAlignment(
        mode="affine", affine_stretch=((.96, .012), (.012, 1.02)),
        translation=(8, -3), rotation_enabled=True, angle_degrees=1.5, pivot=(320, 210)))
    assert pair.validate_all()
    setup_path = save_pair_setup(scratch / "demo", pair)
    assert load_pair_setup(setup_path).fingerprint == pair.fingerprint
    alignment = PairAlignmentDialog(pair)
    alignment.resize(1120, 760)
    alignment.show()
    shot(alignment, "alignment.png")
    alignment.close()

    mts = MtsUniaxialWindow(PluginContext(host, "mts_uniaxial"))
    mts.resize(870, 920)
    mts.show()
    mts._set_root_candidate(experiment["root"])
    mts._on_load()
    mts._on_detect_reference()
    settle()
    shot(mts.sec_crop, "mts-crop.png")
    shot(mts.sec_reference, "mts-reference.png")
    assert mts.pstate.ref_image_global is not None
    host.state.roi = ROI([(65, 55), (575, 55), (575, 365), (65, 365)])
    host._detect_shi_tomasi()
    host._run_tracking()
    assert mts.kinematics() is not None
    mts._on_export()
    mts._on_export_measures()
    assert (Path(experiment["root"]) / "mts_uniaxial_project" / "measures.csv").is_file()
    mts.close()
    host.close()
    print("Workflow checks passed: tracking, cleanup/undo, snapshot, exports, pair setup, all three plugins.")


if __name__ == "__main__":
    main()
