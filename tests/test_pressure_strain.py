"""Qt-free and offscreen regressions for the bundled Pressure–Strain plugin."""
from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKER_CONFIG_DIR", tempfile.mkdtemp(prefix="pressure_cfg_"))

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from app.core.image_sequence import discover
from app.plugins import PluginContext
from plugins.pressure_strain import analysis
from plugins.pressure_strain.window import PressureStrainWindow
from tests.synthetic import make_sequence


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


LOGGER_HEADER = ("Timestamp", "Elapsed_s", "Arduino_ms", "Raw_mbar", "Tared_mbar", "TareOffset_mbar")


def _write_pressure(path, rows, header=LOGGER_HEADER):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _local_stamp(epoch_s):
    """The logger's local wall-clock Timestamp text for an epoch time."""
    return datetime.fromtimestamp(epoch_s).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _logger_rows(start_epoch_s, arduino_ms, pressures, pc_lag_s=()):
    """Logger rows whose PC Timestamp = start + Arduino time (+ optional per-row receive lag)."""
    lags = list(pc_lag_s) + [0.0] * (len(arduino_ms) - len(pc_lag_s))
    t0 = arduino_ms[0] / 1000.0
    return [
        (_local_stamp(start_epoch_s + ms / 1000.0 - t0 + lag), 0.0, ms, p, p, 0.0)
        for ms, p, lag in zip(arduino_ms, pressures, lags)
    ]


def _pressure(time_s, pressure_mbar):
    return analysis.PressureData("memory.csv", np.asarray(time_s, float),
                                 np.asarray(pressure_mbar, float), len(time_s))


def test_parse_pressure_csv_uses_logger_clock_with_median_offset():
    directory = tempfile.mkdtemp(prefix="pressure_sample_")
    path = os.path.join(directory, "pressure.csv")
    start = datetime(2026, 8, 14, 14, 9, 15, 340000).timestamp()
    # Rows 1-3 arrive in one serial burst (same PC Timestamp); the median ignores the burst.
    rows = _logger_rows(start, [2412, 2512, 2612, 2712, 2812, 2912, 3012],
                        [964.84, 964.76, 964.89, 965.0, 970.0, 980.0, 990.0],
                        pc_lag_s=(0.2, 0.1, 0.0))
    _write_pressure(path, rows)
    parsed = analysis.parse_pressure_csv(path)
    assert parsed.source_rows == 7
    assert np.allclose(np.diff(parsed.time_s), 0.1)
    assert abs(parsed.time_s[0] - start) < 1e-3  # Timestamp has millisecond resolution
    assert abs(parsed.clock_residual_s - 0.2) < 2e-3
    assert np.array_equal(parsed.pressure_mbar, [964.84, 964.76, 964.89, 965.0, 970.0, 980.0, 990.0])


def test_pressure_parser_errors():
    directory = tempfile.mkdtemp(prefix="pressure_parser_")
    start = datetime(2026, 8, 14, 14, 0, 0).timestamp()

    bad_header = os.path.join(directory, "bad_header.csv")
    _write_pressure(bad_header, [(0, 1), (1, 2)], header=("Elapsed_s", "Raw_mbar"))
    try:
        analysis.parse_pressure_csv(bad_header)
    except ValueError as exc:
        assert "Missing required" in str(exc) and "Arduino_ms" in str(exc)
    else:
        raise AssertionError("logs without Timestamp/Arduino_ms should fail")

    restarted = os.path.join(directory, "restarted.csv")
    _write_pressure(restarted, _logger_rows(start, [5000, 5100, 100, 200], [1, 2, 3, 4]))
    try:
        analysis.parse_pressure_csv(restarted)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("an Arduino_ms reset should fail")

    bad_stamp = os.path.join(directory, "bad_stamp.csv")
    _write_pressure(bad_stamp, [("14:00:00", 0, 0, 1, 1, 0), ("14:00:01", 0, 100, 2, 2, 0)])
    try:
        analysis.parse_pressure_csv(bad_stamp)
    except ValueError as exc:
        assert "Timestamp" in str(exc)
    else:
        raise AssertionError("a malformed Timestamp should fail")


def test_image_capture_times_from_file_names():
    epoch = analysis.image_capture_times(["d/Image_1760360561123.bmp", "d/Image_0001760360561223.png"])
    assert np.allclose(epoch, [1760360561.123, 1760360561.223])

    local = analysis.image_capture_times(["Image_20260814140915340.bmp"])
    assert abs(local[0] - (datetime(2026, 8, 14, 14, 9, 15).timestamp() + 0.340)) < 1e-6

    for name in ("img_0.png", "Image_123.bmp", "Image_20261399000000000.bmp"):
        try:
            analysis.image_capture_times([name])
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f"{name} should be rejected")


def test_capture_time_alignment_and_later_reference_zeroing():
    # Logger started 3 s before the camera and stopped 1 s before the last frame.
    pressure = _pressure([97.0, 100.0, 105.0, 110.0], [90.0, 100.0, 150.0, 200.0])
    aligned = analysis.align_pressure_to_images(
        pressure, np.array([100.0, 101.0, 104.0, 110.0, 111.0])
    )
    assert np.allclose(aligned.pressure_mbar[:4], [100.0, 110.0, 140.0, 200.0])
    assert np.isnan(aligned.pressure_mbar[4])
    assert np.array_equal(aligned.covered, [True, True, True, True, False])
    assert np.allclose(aligned.image_elapsed_s, [0.0, 1.0, 4.0, 10.0, 11.0])
    assert aligned.provenance["clock_model"].startswith("filename_capture_time")

    strains = analysis.StrainSeries(
        epsilon_1=np.array([0.0, 0.2]),
        epsilon_2=np.array([0.0, -0.1]),
        mean=np.array([0.0, 0.05]),
        n_valid_points=np.array([5, 5]),
    )
    plotted = analysis.build_plot_series(aligned, strains, 2, 3, "mean", True)
    assert np.array_equal(plotted.global_indices, [2, 3])
    assert np.allclose(plotted.pressure_mbar, [0.0, 60.0])
    assert np.allclose(plotted.strain, [0.0, 0.05])

    try:
        analysis.build_plot_series(aligned, strains, 3, 4, "mean", False)
    except ValueError as exc:
        assert "outside the pressure log" in str(exc)
    else:
        raise AssertionError("tracked frames outside the log must not get pressure")


def test_image_time_validation():
    pressure = _pressure([0.0, 1.0], [10.0, 20.0])
    for stamps, message in (
        ([2.0, 1.0], "go backwards"),
        ([0.5, 0.5], "positive total span"),
        ([5.0, 6.0], "No image lies inside"),
    ):
        try:
            analysis.align_pressure_to_images(pressure, np.asarray(stamps))
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid image timestamps should fail")

    repeated = analysis.align_pressure_to_images(
        pressure, np.array([0.0, 0.5, 0.5, 1.0])
    )
    assert repeated.pressure_mbar[1] == repeated.pressure_mbar[2]


def test_principal_strains_and_status_filtering():
    reference = np.array([[0, 0], [3, 0], [0, 2], [2, 2], [1, 1]], dtype=float)
    deformation = np.diag([1.2, 0.9])
    current = reference @ deformation.T + np.array([7.0, -4.0])
    coords = np.stack([reference, current])
    status = np.ones((2, reference.shape[0]), dtype=np.uint8)
    series = analysis.compute_principal_strains(coords, status)
    assert series.epsilon_1[0] == series.epsilon_2[0] == 0.0
    assert abs(series.epsilon_1[1] - 0.2) < 1e-10
    assert abs(series.epsilon_2[1] + 0.1) < 1e-10
    assert abs(series.mean[1] - 0.05) < 1e-10

    corrupt = coords.copy()
    corrupt[1, 4] = [999.0, -999.0]
    status[1, 4] = 0
    filtered = analysis.compute_principal_strains(corrupt, status)
    assert abs(filtered.epsilon_1[1] - 0.2) < 1e-10
    assert filtered.n_valid_points[1] == 4

    too_few = analysis.compute_principal_strains(coords[:, :2], np.ones((2, 2)))
    assert np.isnan(too_few.epsilon_1).all()


def test_export_matches_current_plot_series():
    directory = tempfile.mkdtemp(prefix="pressure_export_")
    output = os.path.join(directory, "out.csv")
    plotted = analysis.PlotSeries(
        global_indices=np.array([1, 2]),
        image_elapsed_s=np.array([0.25, 0.5]),
        pressure_mbar=np.array([0.0, 12.5]),
        strain=np.array([0.0, np.nan]),
        n_valid_points=np.array([8, 2]),
        strain_mode="epsilon_2",
        pressure_zeroed=True,
    )
    analysis.export_plot_csv(output, plotted, ["a.png", "b.png", "c.png"])
    with open(output, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["image_filename"] == "b.png"
    assert rows[0]["pressure_mbar"] == "0"
    assert rows[0]["strain_mode"] == "epsilon_2" and rows[0]["pressure_zeroed"] == "1"
    assert rows[1]["strain"] == "" and rows[1]["n_valid_points"] == "2"


def _name_frames_by_capture_time(paths, start_ms, step_ms):
    """Rename synthetic frames to the camera's zero-padded Image_<16-digit epoch ms> naming."""
    renamed = []
    for index, path in enumerate(paths):
        target = os.path.join(os.path.dirname(path), f"Image_{start_ms + index * step_ms:016d}.png")
        os.replace(path, target)
        renamed.append(target)
    return renamed


def _tracked_window_with_capture_names(start_ms):
    _app()
    from app.gui.main_window import MainWindow

    directory = tempfile.mkdtemp(prefix="pressure_ui_images_")
    _name_frames_by_capture_time(make_sequence(directory, n_frames=6, dx=2.0, dy=1.0), start_ms, 500)

    window = MainWindow()
    window.resize(900, 600)
    window.show()
    window._load_paths(discover(directory), directory)
    window._begin_roi_definition("ngon")
    tool = window.canvas._interaction
    for corner in [(60, 50), (240, 50), (240, 180), (60, 180)]:
        tool.on_press(QPointF(*corner), None)
    tool.on_right_press(QPointF(60, 180), None)
    window._detect_shi_tomasi()
    window._run_tracking()
    assert window.state.result is not None
    return window, directory


def test_plugin_window_load_plot_and_sequence_invalidation():
    start_ms = 1_700_000_000_000
    main, image_dir = _tracked_window_with_capture_names(start_ms)
    pressure_path = os.path.join(tempfile.mkdtemp(prefix="pressure_ui_csv_"), "pressure.csv")
    # Logger runs 1 s before to 1 s after the 2.5 s of frames.
    _write_pressure(pressure_path, _logger_rows(start_ms / 1000.0 - 1.0, [0, 1000, 2000, 3000, 4500],
                                                [90.0, 100.0, 120.0, 160.0, 160.0]))

    plugin_window = PressureStrainWindow(PluginContext(main, "pressure_strain_test"))
    plugin_window.show()
    assert plugin_window._picker_start_directory() == os.path.dirname(image_dir)
    assert plugin_window.load_pressure_file(pressure_path)
    assert plugin_window.plot_button.isEnabled() and plugin_window.export_button.isEnabled()

    plugin_window.strain_combo.setCurrentIndex(2)  # mean
    plugin_window.zero_check.setChecked(True)
    current = plugin_window.current_plot_series()
    assert current is not None and current.strain_mode == "mean"
    assert current.pressure_zeroed and current.pressure_mbar[0] == 0.0

    plugin_window._open_plot()
    assert plugin_window._plot_window is not None and plugin_window._plot_window.isVisible()
    plugin_window._plot_window.replot()

    new_dir = tempfile.mkdtemp(prefix="pressure_ui_new_images_")
    _name_frames_by_capture_time(make_sequence(new_dir, n_frames=3), 1_800_000_000_000, 1000)
    main._load_paths(discover(new_dir), new_dir)
    assert plugin_window._aligned is None and plugin_window.path_edit.text() == ""
    assert plugin_window._plot_window is None

    plugin_window.dispose()
    plugin_window.close()
    main.close()


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} checks passed.")
