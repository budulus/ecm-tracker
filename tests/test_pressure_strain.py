"""Qt-free and offscreen regressions for the bundled Pressure–Strain plugin."""
from __future__ import annotations

import csv
import os
import tempfile

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


def _write_pressure(path, rows, header=("Elapsed_s", "Raw_mbar")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_parse_pressure_csv_with_logger_headers():
    directory = tempfile.mkdtemp(prefix="pressure_sample_")
    path = os.path.join(directory, "pressure.csv")
    header = (
        "Timestamp",
        "Elapsed_s",
        "Arduino_ms",
        "Raw_mbar",
        "Tared_mbar",
        "TareOffset_mbar",
    )
    rows = [
        ("2026-08-14 14:09:15.340", 0.083, 2412, 964.84, -0.01, 964.85),
        ("2026-08-14 14:09:15.340", 0.083, 2512, 964.76, -0.09, 964.85),
        ("2026-08-14 14:17:44.422", 509.166, 511612, 956.04, -8.81, 964.85),
    ]
    _write_pressure(path, rows, header=header)
    parsed = analysis.parse_pressure_csv(path)
    assert parsed.source_rows == 3
    assert parsed.elapsed_s[0] == 0.083 and parsed.elapsed_s[-1] == 509.166
    # Duplicate t=0.083 rows exist, but the explicit first-row endpoint must remain exact.
    assert parsed.pressure_mbar[0] == 964.84 and parsed.pressure_mbar[-1] == 956.04
    assert np.all(np.diff(parsed.elapsed_s) > 0)


def test_pressure_parser_duplicate_policy_and_errors():
    directory = tempfile.mkdtemp(prefix="pressure_parser_")
    path = os.path.join(directory, "pressure.csv")
    _write_pressure(path, [(0, 10), (0, 12), (0.5, 20), (0.5, 24), (1, 30), (1, 32)])
    parsed = analysis.parse_pressure_csv(path)
    assert np.array_equal(parsed.elapsed_s, [0.0, 0.5, 1.0])
    assert np.allclose(parsed.pressure_mbar, [10.0, 22.0, 32.0])

    bad_header = os.path.join(directory, "bad_header.csv")
    _write_pressure(bad_header, [(0, 1), (1, 2)], header=("time", "pressure"))
    try:
        analysis.parse_pressure_csv(bad_header)
    except ValueError as exc:
        assert "Missing required" in str(exc)
    else:
        raise AssertionError("missing pressure headers should fail")

    backwards = os.path.join(directory, "backwards.csv")
    _write_pressure(backwards, [(0, 1), (2, 2), (1, 3)])
    try:
        analysis.parse_pressure_csv(backwards)
    except ValueError as exc:
        assert "nondecreasing" in str(exc)
    else:
        raise AssertionError("decreasing Elapsed_s should fail")


def test_endpoint_alignment_and_later_reference_zeroing():
    pressure = analysis.PressureData(
        "memory.csv",
        np.array([0.0, 5.0, 10.0]),
        np.array([100.0, 150.0, 200.0]),
        3,
    )
    aligned = analysis.align_pressure_to_images(
        pressure, np.array([1000.0, 1001.0, 1004.0, 1010.0])
    )
    assert np.allclose(aligned.pressure_mbar, [100.0, 110.0, 140.0, 200.0])
    assert np.allclose(aligned.image_elapsed_s, [0.0, 1.0, 4.0, 10.0])

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


def test_image_time_validation():
    pressure = analysis.PressureData(
        "memory.csv", np.array([0.0, 1.0]), np.array([10.0, 20.0]), 2
    )
    for stamps, message in (
        ([2.0, 1.0], "go backwards"),
        ([2.0, 2.0], "positive total span"),
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


def _tracked_window_with_mtimes():
    _app()
    from app.gui.main_window import MainWindow

    directory = tempfile.mkdtemp(prefix="pressure_ui_images_")
    paths = make_sequence(directory, n_frames=6, dx=2.0, dy=1.0)
    base_ns = 1_700_000_000_000_000_000
    for index, path in enumerate(paths):
        timestamp = base_ns + index * 500_000_000
        os.utime(path, ns=(timestamp, timestamp))

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
    main, image_dir = _tracked_window_with_mtimes()
    pressure_path = os.path.join(tempfile.mkdtemp(prefix="pressure_ui_csv_"), "pressure.csv")
    _write_pressure(pressure_path, [(0.0, 100.0), (1.0, 120.0), (2.0, 160.0)])

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
    new_paths = make_sequence(new_dir, n_frames=3)
    base_ns = 1_800_000_000_000_000_000
    for index, path in enumerate(new_paths):
        timestamp = base_ns + index * 1_000_000_000
        os.utime(path, ns=(timestamp, timestamp))
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
