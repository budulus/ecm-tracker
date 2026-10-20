"""Checks for the scripting layer: headless Session, macro recorder, runners, console and editor.

Run with:  QT_QPA_PLATFORM=offscreen uv run python -m tests.test_scripting
The Session / recorder / CLI checks are Qt-free; the GUI checks build a real MainWindow.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKER_CONFIG_DIR", tempfile.mkdtemp(prefix="cfg_"))

import numpy as np

from app.core.image_sequence import discover
from app.scripting import (
    OperationCancelled,
    Recorder,
    Session,
    SessionError,
    format_call,
    format_value,
    run_script,
    set_current_session,
)
from app.scripting.executor import ScriptExecutor
from tests.synthetic import make_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP = None


def _sequence_dir(n=10):
    d = tempfile.mkdtemp(prefix="seq_")
    make_sequence(d, n_frames=n, dx=2.0, dy=1.0)
    return d


def _tracked_session(folder=None):
    s = Session()
    s.open_directory(folder or _sequence_dir())
    s.set_roi_rectangle(60, 50, 240, 180)
    s.detect_corners(maxCorners=150)
    s.track()
    return s


def _app():
    global _APP
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


def _window():
    _app()
    from app.gui.main_window import MainWindow

    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    return w


# ---------------------------------------------------------------------- headless session
def test_import_is_headless():
    code = (
        "import sys, ecm_tracker as et\n"
        "s = et.Session()\n"
        "assert 'PySide6' not in sys.modules, sorted(m for m in sys.modules if 'PySide' in m)\n"
        "print('ok', repr(s))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok <Session: no sequence loaded>")


def test_headless_pipeline_matches_core():
    from app.core.cleanup import build_mask, compute_metrics, thresholds_from_filters
    from app.core.feature_detection import shi_tomasi
    from app.core.image_sequence import ImageSequence
    from app.core.roi import ROI, rectangle_corners
    from app.core.tracking import track

    folder = _sequence_dir()
    s = Session()
    assert s.open_directory(folder) == 10
    assert s.n_images == 10 and s.frame_paths == tuple(discover(folder))
    s.set_frame_range(1, 8)
    s.set_roi_rectangle(60, 50, 240, 180)
    n = s.detect_corners(maxCorners=150, qualityLevel=0.02)
    s.track(win_size=15)
    removed = s.cleanup(fb_max=0.05, drop_left_image=True)

    # The same steps directly on the core library.
    seq = ImageSequence(discover(folder))
    roi = ROI(rectangle_corners(60.0, 50.0, 240.0, 180.0))
    gray = seq.load_gray(1)
    params = dict(s.corner_params)
    seeds = shi_tomasi(gray, roi.mask(*gray.shape[:2]), params)
    assert len(seeds) == n and np.array_equal(seeds, s.seeds)
    lk = dict(s.lk_params)
    assert lk["win_size"] == 15
    result = track(seq, 1, 8, seeds, lk)
    assert np.array_equal(result.coords_fw, s.result.coords_fw)
    keep = build_mask(compute_metrics(result, roi, gray.shape[:2]),
                      thresholds_from_filters(fb_max=0.05, drop_left_image=True))
    assert np.array_equal(keep, s.active_mask)
    assert removed == int((~keep).sum())

    assert s.coords().shape == (8, s.n_active, 2)
    assert np.array_equal(s.frame_indices(), np.arange(1, 9))
    assert s.valid().all() or s.valid().dtype == bool

    out = tempfile.mkdtemp(prefix="out_")
    coords_path, _seq_path, shape = s.export_npy(os.path.join(out, "coords.npy"))
    assert np.load(coords_path).shape == shape
    csv_path, _ = s.export_csv(os.path.join(out, "coords.csv"))
    assert open(csv_path).readline().startswith("filename,p1x")
    tracker_path = s.save_trackers(os.path.join(out, "session"))

    s2 = Session()
    s2.open_directory(folder)
    s2.load_trackers(tracker_path)
    assert (s2.reference_index, s2.last_index) == (1, 8)
    assert np.array_equal(s2.active_mask, s.active_mask)
    assert s2.roi_corners == s.roi_corners
    assert "tracking : " in s2.summary()


def test_session_guards_and_strict_parameters():
    s = Session()
    for call in (s.detect_corners, s.track, lambda: s.set_frame_range(0, 1), s.cleanup):
        try:
            call()
        except SessionError:
            pass
        else:
            raise AssertionError("expected SessionError without a sequence")
    s.open_directory(_sequence_dir(6))
    try:
        s.detect_corners()
    except SessionError as exc:
        assert "ROI" in str(exc)
    else:
        raise AssertionError("detect_corners needs an ROI")
    for bad in ({"maxCornerz": 5}, {"qualityLevel": 5.0}, {"maxCorners": "many"}):
        try:
            s.set_corner_params(**bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid parameters {bad}")
    try:
        s.set_lk_params(flags=4)
    except ValueError:
        pass
    else:
        raise AssertionError("LK flags 4 must be rejected")
    assert s.set_lk_params(win_size=np.int64(31))["win_size"] == 31  # numpy scalars are fine
    try:
        s.set_frame_range(4, 2)
    except IndexError:
        pass
    else:
        raise AssertionError("inverted range must be rejected")
    try:
        s.add_seed(1000, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("seed outside the image must be rejected")
    try:
        s.cleanup_mask(not_a_filter=1)
    except (TypeError, SessionError):
        pass
    else:
        raise AssertionError("unknown cleanup filters must be rejected")


def test_session_events_and_invalidation():
    s = _tracked_session()
    events = []
    before = []
    s.add_listener(lambda kinds: events.append(set(kinds)), before=lambda kinds: before.append(set(kinds)))
    revision = s.revision
    s.set_current_frame(3)  # navigation: event, but no scientific revision
    assert events[-1] == {"frame"} and s.revision == revision
    s.cleanup(fb_mean=0.0)
    assert events[-1] == {"mask"} and s.can_undo_cleanup
    assert s.undo_cleanup() and not s.can_undo_cleanup
    s.set_roi_circle(150, 110, 60)  # changing an input discards the result
    assert events[-1] == {"roi", "result"} and not s.has_result
    assert s.n_seeds > 0  # seeds survive an ROI change
    s.add_seed(10, 10)
    s.set_frame_range(2, 9)  # moving the reference clears ROI + seeds
    assert events[-1] == {"range", "roi", "seeds"} and s.n_seeds == 0 and s.roi is None
    # Nested commands notify once, after the outermost one: detect_corners sets params too.
    s.set_roi_rectangle(60, 50, 240, 180)
    events.clear()
    s.detect_corners(maxCorners=40)
    assert events == [{"params", "seeds"}]
    assert {"params"} in before and {"seeds"} in before


def test_progress_cancel_raises():
    folder = _sequence_dir(6)
    s = Session()
    try:
        s.open_directory(folder, progress=lambda done, total: True)
    except OperationCancelled:
        pass
    else:
        raise AssertionError("cancelled load must raise")
    assert not s.has_sequence
    s.open_directory(folder)
    s.set_roi_rectangle(60, 50, 240, 180)
    s.detect_corners()
    try:
        s.track(progress=lambda done, total: done >= 2)
    except OperationCancelled:
        pass
    else:
        raise AssertionError("cancelled tracking must raise")
    assert not s.has_result


# ---------------------------------------------------------------------- recorder
def test_value_and_call_formatting():
    assert format_value(np.float32(0.5)) == "0.5"
    assert format_value(np.int64(3)) == "3"
    assert format_value((1,)) == "(1,)"
    assert format_value(r"C:\data\run 1") == 'r"C:\\data\\run 1"'
    assert format_value("C:\\data\\") == repr("C:\\data\\")
    assert format_value(float("inf")) == 'float("inf")'
    assert format_value(np.array([[1, 2], [3, 4]])) == "[[1, 2], [3, 4]]"
    for value in ([(1.5, 2.0)] * 30, {"a": [1] * 40}, "x" * 120):
        assert eval(format_value(value)) == value
    short = format_call("session", "track", (), {"win_size": 21})
    assert short == "session.track(win_size=21)"
    long = format_call("session", "set_roi", ([(float(i), float(i)) for i in range(20)],))
    assert long.startswith("session.set_roi(\n") and long.endswith("\n)")
    ns = {"session": type("S", (), {"set_roi": staticmethod(lambda corners: corners)})}
    assert len(eval(long, ns)) == 20


def test_recorder_coalesces_and_skips_nested_commands():
    s = Session()
    rec = s.start_recording()
    s.open_directory(_sequence_dir(8))  # nested load_sequence is not recorded
    for last in (6, 5, 4):
        s.set_frame_range(0, last)  # a dragged slider -> one line
    s.set_current_frame(2)  # navigation is not recorded
    s.set_roi_rectangle(60, 50, 240, 180)  # nested set_roi is not recorded
    s.detect_corners(maxCorners=60)
    s.track()
    s.apply_keep_mask(np.arange(s.n_points) % 3 != 0)  # recorded as the equivalent remove_points
    s.set_display_params(marker_size=5)  # cosmetic; not recorded
    lines = rec.entries  # one entry per command (long calls wrap over several lines)
    names = [line.split("(")[0] for line in lines]
    assert names == [
        "session.open_directory",
        "session.set_frame_range",
        "session.set_roi_rectangle",
        "session.detect_corners",
        "session.track",
        "session.remove_points",
    ], names
    assert lines[1] == "session.set_frame_range(0, 4)"
    removed = eval(rec.entries[-1].split("(", 1)[1].rsplit(")", 1)[0])
    assert removed == [i for i in range(s.n_points) if i % 3 == 0]
    with s.suspend_recording():
        s.clear_tracking()
    assert len(rec.entries) == 6
    assert s.stop_recording() is rec and not s.is_recording
    s.clear_roi()
    assert len(rec.entries) == 6


def test_recorded_macro_replays_identically():
    folder = _sequence_dir()
    s = Session()
    rec = s.start_recording()
    s.open_directory(folder)
    s.set_frame_range(0, 8)
    s.set_roi([(60.5, 50.25), (240.0, 52.0), (238.0, 181.75), (61.0, 180.0)])
    s.detect_grid(spacing_x=12, spacing_y=14)
    s.remove_seeds([0, 3])
    s.add_seed(100.5, 100.25)
    s.track(win_size=17, max_level=2)
    s.cleanup(fb_mean=0.02)
    s.remove_points([1, 2])
    out = tempfile.mkdtemp(prefix="macro_")
    s.export_npy(os.path.join(out, "coords.npy"))
    script = rec.script()
    assert script.startswith("# ECM Tracker macro") and "session = get_session()" in script
    assert "app = get_app()" not in script

    path = os.path.join(out, "macro.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script.replace(os.path.join(out, "coords.npy"), os.path.join(out, "replay.npy")))
    replay = Session()
    run_script(path, session=replay)
    assert np.array_equal(replay.seeds, s.seeds)
    assert np.array_equal(replay.result.coords_fw, s.result.coords_fw)
    assert np.array_equal(replay.active_mask, s.active_mask)
    assert np.array_equal(np.load(os.path.join(out, "replay.npy")), np.load(os.path.join(out, "coords.npy")))


def test_run_script_namespace_and_cli():
    folder = _sequence_dir(6)
    out = tempfile.mkdtemp(prefix="cli_")
    script = os.path.join(out, "job.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(f"""
            import sys
            from ecm_tracker import get_session
            assert get_session() is session and __name__ == "__main__"
            assert sys.argv[1:] == ["alpha", "2"], sys.argv
            session.open_directory({folder!r})
            session.set_roi_rectangle(60, 50, 240, 180)
            session.detect_corners(maxCorners=50)
            session.track()
            session.export_csv(__file__.replace("job.py", "result.csv"))
            print("tracked", session.n_active, "pyside", "PySide6" in sys.modules)
        """))
    proc = subprocess.run([sys.executable, "-m", "app.main", "--run", script, "alpha", "2"], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "tracked" in proc.stdout and "pyside False" in proc.stdout  # headless: no Qt at all
    assert os.path.exists(os.path.join(out, "result.csv"))

    failing = os.path.join(out, "fail.py")
    with open(failing, "w", encoding="utf-8") as handle:
        handle.write("session.track()\n")
    proc = subprocess.run([sys.executable, "-m", "ecm_tracker", failing], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and "SessionError" in proc.stderr, proc.stderr

    # In-process: run_script binds get_session() to the given session and restores sys.argv.
    s = Session()
    argv = list(sys.argv)
    previous = set_current_session(None)
    try:
        globals_ = run_script(script, session=s, args=["alpha", 2])
    finally:
        set_current_session(previous)
    assert sys.argv == argv and globals_["session"] is s and s.has_result


def test_executor_output_errors_and_stop():
    out, err = [], []
    namespace = {"__name__": "__main__"}
    ticks = []
    armed = [False]

    def poll():
        if armed[0]:
            ticks.append(1)
            if len(ticks) == 3:
                executor.request_stop()

    executor = ScriptExecutor(namespace, out.append, err.append, poll=poll, poll_interval=0.0)
    assert executor.run_source("x = 6 * 7\nprint('x is', x)\n", "<t1>").ok
    assert namespace["x"] == 42 and "".join(out) == "x is 42\n"

    outcome = executor.run_source("def f():\n    return 1 / 0\n\nf()\n", "<t2>")
    assert not outcome.ok and isinstance(outcome.error, ZeroDivisionError) and outcome.line == 2
    text = "".join(err)
    assert "ZeroDivisionError" in text and "executor.py" not in text  # host frames trimmed

    err.clear()
    armed[0] = True
    outcome = executor.run_source("def spin():\n    while True:\n        pass\nspin()\n", "<t3>")
    assert outcome.interrupted and isinstance(outcome.error, KeyboardInterrupt)
    assert len(ticks) >= 3 and not executor.running

    outcome = executor.run_source("x = (", "<t4>")
    assert not outcome.ok and isinstance(outcome.error, SyntaxError)
    assert executor.run_source("import sys\nsys.exit(0)\n", "<t5>").ok


# ---------------------------------------------------------------------- GUI integration
def test_gui_actions_record_a_replayable_macro():
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QFileDialog

    folder = _sequence_dir()
    out = tempfile.mkdtemp(prefix="gui_macro_")
    w = _window()
    w.scripting.record_action.trigger()  # start recording (opens the editor's live tab)
    assert w.session.is_recording and w._recording_label.isVisibleTo(w)
    editor = w.scripting.editor
    assert editor is not None and editor._recording_tab is not None

    with patch.object(QFileDialog, "getExistingDirectory", return_value=folder):
        w._open_directory()
    w._on_last_changed(9)  # slider drags coalesce; no change from 9 -> no-op
    w._on_last_changed(8)
    w._go_to_frame(0)
    w._begin_roi_definition("rectangle")
    tool = w.canvas._interaction
    tool.on_press(QPointF(60.123, 50.456), None)
    tool.on_release(QPointF(240.0, 180.0), None)
    assert w.state.roi.corners[0] == (60.12, 50.46)  # interactive coordinates are snapped
    w.state.shi_tomasi_params = dict(w.state.shi_tomasi_params, maxCorners=120)  # as the dialog would
    w._detect_shi_tomasi()
    w._run_tracking()
    w._open_cleanup()
    row = w._cleanup_dialog.fb_mean
    from app.core.cleanup import BandFilter
    row.enable.setChecked(True)
    row.set_band(BandFilter(True, 0.02, row.band().cap))
    w._cleanup_apply()
    w._cleanup_dialog.reject()
    _APP.processEvents()
    from app.gui.point_tools import DeletePointsTool
    first_active = int(np.flatnonzero(w.state.active_mask & w.session.exportable_mask())[0])
    x, y = w.state.result.coords_fw[0, first_active]
    DeletePointsTool(w).on_press(QPointF(float(x), float(y)), None)
    with patch.object(QFileDialog, "getSaveFileName", return_value=(os.path.join(out, "gui.npy"), "")):
        w._export()

    live = editor._recording_tab.toPlainText()
    assert "session.export_npy(" in live
    w.scripting.record_action.trigger()  # stop
    assert not w.session.is_recording and editor._recording_tab is None
    macro = editor.current_editor()
    assert not macro.isReadOnly() and macro.document().isModified()
    script = macro.toPlainText()
    body = [line for line in script.splitlines() if line.startswith("session.")]
    names = [line.split("(")[0] for line in body]
    assert names == ["session.open_directory", "session.set_frame_range", "session.set_roi_rectangle",
                     "session.detect_corners", "session.track", "session.cleanup",
                     "session.remove_points", "session.export_npy"], names
    assert "session.set_roi_rectangle(60.12, 50.46, 240.0, 180.0)" in script
    assert "maxCorners=120" in script and "session.cleanup(fb_mean=" in script, script

    path = os.path.join(out, "macro.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script.replace("gui.npy", "headless.npy"))
    proc = subprocess.run([sys.executable, "-m", "app.main", "--run", path], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert np.array_equal(np.load(os.path.join(out, "headless.npy")), np.load(os.path.join(out, "gui.npy")))
    from PySide6.QtWidgets import QMessageBox

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard) as ask:
        w.close()  # the unsaved macro tab is offered for saving before quitting
    assert ask.called


def test_console_drives_the_gui():
    w = _window()
    console = w.scripting.console
    folder = _sequence_dir(6)
    seen = []
    w.signals.roi_changed.connect(lambda: seen.append("roi"))
    console.submit(f"session.open_directory({folder!r})")
    assert w.state.has_sequence and w.current_slider.isEnabled()
    assert not w.detect_corners_action.isEnabled()
    console.submit("session.set_roi_rectangle(60, 50, 240, 180)")
    assert seen == ["roi"] and w.detect_corners_action.isEnabled()
    console.submit("n = session.detect_corners(maxCorners=30)")
    assert w.scripting.namespace["n"] == w.session.n_seeds > 0
    assert w.run_tracking_action.isEnabled()
    console.submit("n * 2")  # expression results are echoed like the Python REPL
    console.submit("for i in range(2):")
    console.submit("    print('loop', i)")
    console.submit("")
    console.submit("1 / 0")
    text = console.toPlainText()
    assert str(w.session.n_seeds * 2) in text
    assert "loop 0" in text and "loop 1" in text
    assert "ZeroDivisionError" in text
    assert console.toPlainText().endswith(">>> ")
    assert w.menuBar().isEnabled() and not w.scripting.busy  # controls unlocked after each run
    # History recall and a pasted multi-line block.
    console._browse_history(-1)
    assert console._input_text() == "1 / 0"
    console._set_input_text("")
    console.submit("a = 1\nb = a + 1")
    assert w.scripting.namespace["b"] == 2
    w.close()


def test_editor_save_open_run_and_error_marking():
    w = _window()
    editor = w.scripting.show_editor()
    folder = tempfile.mkdtemp(prefix="scripts_")
    path = os.path.join(folder, "probe.py")
    code = editor.new_script("value = 21 * 2\nprint('value', value)\n")
    from PySide6.QtWidgets import QFileDialog

    with patch.object(QFileDialog, "getSaveFileName", return_value=(path, "")):
        assert editor.save_current()
    assert code.path == path and not code.document().isModified()
    assert open(path, encoding="utf-8").read().startswith("value = 21 * 2")
    outcome = editor.run_current()
    assert outcome.ok and w.scripting.namespace["value"] == 42
    assert w.scripting.namespace["__file__"] == path
    assert "value 42" in editor.output.toPlainText()

    broken = editor.new_script("x = 1\ny = 2\nraise RuntimeError('boom')\n")
    outcome = editor.run_current()
    assert not outcome.ok and outcome.line == 3 and broken._error_line == 3
    assert "RuntimeError: boom" in editor.output.toPlainText()

    sel = editor.new_script("a_sel = 5\nb_sel = a_sel + 1\n")
    from PySide6.QtGui import QTextCursor

    cursor = sel.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    sel.setTextCursor(cursor)
    assert editor.run_selection().ok and w.scripting.namespace["a_sel"] == 5
    assert "b_sel" not in w.scripting.namespace

    reopened = editor.open_script(path)
    assert reopened is code  # already open -> focused, not duplicated
    sel.toggle_comment()
    assert sel.toPlainText().startswith("# a_sel = 5")
    sel.toggle_comment()
    assert sel.toPlainText().startswith("a_sel = 5")
    # Unsaved tabs are discarded without a prompt in this test.
    from PySide6.QtWidgets import QMessageBox

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
        w.close()


def test_plugin_launch_is_recorded_as_gui_command():
    w = _window()
    w._load_paths(discover(_sequence_dir(6)), None)
    recorder = w.session.start_recording(Recorder())
    w.plugin_manager.launch("affine_zones")
    assert recorder.entries[-1] == "app.launch_plugin('affine_zones')"
    script = recorder.script()
    assert "app = get_app()" in script and "from ecm_tracker import get_app, get_session" in script
    w.session.stop_recording()
    assert "affine_zones" in w.scripting.app_api.plugins()
    # In the console `app.plugin(...)` reaches the running instance.
    assert w.scripting.app_api.plugin("affine_zones") is not None
    w.close()


def test_api_reference_lists_every_command():
    from app.scripting.reference import api_reference
    from app.scripting.session import Session as S

    text = api_reference()
    commands = [n for n, v in vars(S).items() if getattr(v, "__ecm_command__", False)]
    for name in commands:
        assert f"session.{name}(" in text, name


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
