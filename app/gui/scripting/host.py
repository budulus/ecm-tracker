"""The GUI side of scripting: one host shared by the console, the Script Editor and the recorder.

``MainWindow`` creates a single :class:`ScriptingHost`. It owns the namespace user code runs in
(``session`` = the window's :class:`~app.scripting.session.Session`, ``app`` =
:class:`ScriptApp`, ``np``), the interruptible executor, the Python console dock, the lazily
created Script Editor window, and the shared Console / Editor / Record actions that appear in
both the main toolbar and the editor.

While user code runs the main window's controls are disabled (the code calls
``processEvents`` to stay responsive, and a concurrent click must not change state under the
script); the console's Ctrl+C and the editor's Stop button interrupt it.
"""
import os
import sys

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QDockWidget, QFileDialog, QInputDialog, QLineEdit

from app.core import settings
from app.gui.icon_loader import load_icon
from app.gui.scripting.console import PythonConsole
from app.scripting.executor import ScriptExecutor
from app.scripting.recorder import Recorder
from app.scripting.reference import api_reference
from app.scripting.runner import make_namespace, set_current_app, set_current_session

SETTINGS_SECTION = "scripting"
RECORD_RED = "#dc2626"  # the conventional red record dot


class ScriptApp:
    """GUI helpers for scripts, available as ``app`` (only while the GUI runs)."""

    available = True

    def __init__(self, window):
        self._window = window

    @property
    def window(self):
        """The main window (parent for any dialogs a script opens)."""
        return self._window

    @property
    def canvas(self):
        return self._window.canvas

    @property
    def session(self):
        return self._window.session

    def plugins(self):
        """Ids of the enabled, loadable plugins."""
        return [r.plugin_id for r in self._window.plugin_manager.records() if r.cls is not None]

    def launch_plugin(self, plugin_id: str):
        """Open a plugin (like its side-pane button); returns its window, if it has one."""
        manager = self._window.plugin_manager
        record = manager._records.get(plugin_id)
        if record is None:
            raise KeyError(f"No plugin {plugin_id!r}. Available: {', '.join(self.plugins())}")
        if record.cls is None:
            raise RuntimeError(f"Plugin {plugin_id!r} is disabled or failed to load:\n{record.error or ''}")
        manager.launch(plugin_id)
        if record.instance is None:
            raise RuntimeError(f"Plugin {plugin_id!r} raised an error while launching.")
        return record.window

    def plugin(self, plugin_id: str):
        """The running plugin instance (``None`` until launched) for scripting its internals."""
        record = self._window.plugin_manager._records.get(plugin_id)
        return record.instance if record is not None else None

    def fit_view(self) -> None:
        self._window.canvas.reset_view()

    def zoom_in(self) -> None:
        self._window.canvas.zoom_in()

    def zoom_out(self) -> None:
        self._window.canvas.zoom_out()

    def save_view(self, path) -> str:
        """Save what the canvas currently shows (frame + overlays) as an image file."""
        path = os.fspath(path)
        if not self._window.canvas.grab().save(path):
            raise OSError(f"Could not save the view to {path}")
        return path

    def message(self, text: str, timeout: int = 4000) -> None:
        """Show ``text`` in the main window's status bar."""
        self._window.statusBar().showMessage(str(text), timeout)

    def process_events(self) -> None:
        """Let the GUI repaint during a long custom loop."""
        QApplication.processEvents()

    def __repr__(self) -> str:
        return "<ScriptApp: ECM Tracker GUI>"


class ScriptingHost(QObject):
    output = Signal(str, str, str)  # channel ("console" / "editor"), kind ("out"/"err"/"info"), text
    busy_changed = Signal(bool)
    recording_changed = Signal(bool)
    recording_updated = Signal(str)  # the complete macro script so far

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.session = window.session
        self.app_api = ScriptApp(window)
        set_current_session(self.session)
        set_current_app(self.app_api)
        self.namespace = make_namespace(self.session, input=self._input)
        self._channel = "console"
        self.executor = ScriptExecutor(self.namespace, self._write_out, self._write_err,
                                       poll=QApplication.processEvents)
        self.editor = None

        self.editor_action = QAction(load_icon("file-code"), "Editor", window)
        self.editor_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.editor_action.setToolTip("Open the Script Editor (Ctrl+Shift+E)")
        self.editor_action.triggered.connect(self.show_editor)
        self.record_action = QAction(load_icon("record", RECORD_RED), "Record", window)
        self.record_action.setCheckable(True)
        self.record_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.record_action.setToolTip("Record your actions as a Python macro (Ctrl+Shift+R)")
        self.record_action.toggled.connect(self._on_record_toggled)
        self.run_file_action = QAction("Run Script File…", window)
        self.run_file_action.triggered.connect(self.run_file_dialog)
        self.stop_action = QAction("Stop Script", window)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.request_stop)

        self.console = PythonConsole(self)
        self.console_dock = QDockWidget("Python Console", window)
        self.console_dock.setObjectName("pythonConsoleDock")
        self.console_dock.setWidget(self.console)
        self.console_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea
                                          | Qt.DockWidgetArea.RightDockWidgetArea)
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self.console_dock.hide()
        # Qt's own toggle action tracks the dock correctly through minimize/restore and floating.
        self.console_action = self.console_dock.toggleViewAction()
        self.console_action.setIcon(load_icon("terminal"))
        self.console_action.setText("Console")
        self.console_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.console_action.setToolTip("Show the Python console (Ctrl+Shift+P)")
        self.console_action.toggled.connect(lambda checked: checked and self.console.setFocus())

    # ---- settings ---------------------------------------------------------------
    def _settings(self) -> dict:
        return settings.get_section(SETTINGS_SECTION) or {}

    def _update_settings(self, **values) -> None:
        section = self._settings()
        section.update(values)
        try:
            settings.update_section(SETTINGS_SECTION, section)
        except OSError:
            pass  # history / last folder are conveniences; never fail a command over them

    def load_history(self):
        history = self._settings().get("history", [])
        return [h for h in history if isinstance(h, str)] if isinstance(history, list) else []

    def save_history(self, history) -> None:
        self._update_settings(history=list(history))

    def script_dir(self) -> str:
        folder = self._settings().get("script_dir")
        return folder if isinstance(folder, str) and os.path.isdir(folder) else os.path.expanduser("~")

    def remember_script_dir(self, folder: str) -> None:
        self._update_settings(script_dir=folder)

    @staticmethod
    def examples_dir() -> str:
        from app.plugins.manager import PROJECT_ROOT

        return str(PROJECT_ROOT / "examples" / "scripts")

    @staticmethod
    def api_reference() -> str:
        return api_reference()

    # ---- execution ----------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.executor.running

    @property
    def stop_requested(self) -> bool:
        return self.executor.stop_requested

    def request_stop(self) -> None:
        self.executor.request_stop()

    def _write_out(self, text: str) -> None:
        self.output.emit(self._channel, "out", text)

    def _write_err(self, text: str) -> None:
        self.output.emit(self._channel, "err", text)

    def _input(self, prompt: str = "") -> str:
        """``input()`` for scripts: a dialog instead of a (nonexistent) terminal."""
        text, ok = QInputDialog.getText(self.window, "Script input", str(prompt) or "Input:",
                                        QLineEdit.EchoMode.Normal)
        if not ok:
            raise EOFError("Input cancelled")
        return text

    def _run(self, runner, channel: str):
        if self.busy:
            raise RuntimeError("A script is already running.")
        self._channel = channel
        self._set_busy(True)
        try:
            return runner()
        finally:
            self._set_busy(False)

    def execute_code(self, code, source: str, filename: str, *, channel: str = "console"):
        """Run already-compiled ``code`` (its ``source`` is registered for tracebacks)."""
        self.executor.register_source(filename, source)
        return self._run(lambda: self.executor.run_code(code, filename), channel)

    def execute_source(self, source: str, filename: str, *, channel: str = "console",
                       mode: str = "exec", script_path=None):
        """Compile and run ``source``; returns a :class:`~app.scripting.executor.RunOutcome`."""
        if script_path:
            self.namespace["__file__"] = script_path
        return self._run(lambda: self.executor.run_source(source, filename, mode), channel)

    def run_file(self, path, *, args=(), channel: str = "console"):
        """Run a script file in the shared namespace (``sys.argv = [path, *args]``), reporting
        to the console."""
        path = os.path.abspath(os.fspath(path))
        self.show_console()
        try:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            self.output.emit(channel, "err", f"Cannot run {path}: {exc}\n")
            return None
        self.output.emit(channel, "info", f"Running {path}\n")
        old_argv = sys.argv
        sys.argv = [path, *map(str, args)]
        try:
            outcome = self.execute_source(source, path, channel=channel, script_path=path)
        finally:
            sys.argv = old_argv
        self.output.emit(channel, "info" if outcome.ok else "err",
                         "Done.\n" if outcome.ok else "Script failed.\n")
        return outcome

    def run_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.window, "Run Script", self.script_dir(),
                                              "Python scripts (*.py)")
        if path:
            self.remember_script_dir(os.path.dirname(path))
            self.run_file(path)

    def _set_busy(self, busy: bool) -> None:
        self.window.set_scripts_running(busy)
        self.stop_action.setEnabled(busy)
        self.busy_changed.emit(busy)

    # ---- windows ------------------------------------------------------------------
    def show_console(self) -> None:
        self.console_dock.show()
        self.console_dock.raise_()
        self.console.setFocus()

    def ensure_editor(self):
        if self.editor is None:
            from app.gui.scripting.editor import ScriptEditorWindow

            self.editor = ScriptEditorWindow(self)
        return self.editor

    def show_editor(self):
        editor = self.ensure_editor()
        editor.show()
        editor.raise_()
        editor.activateWindow()
        return editor

    # ---- recording ------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.session.is_recording

    def _on_record_toggled(self, checked: bool) -> None:
        if checked and not self.session.is_recording:
            self.start_recording()
        elif not checked and self.session.is_recording:
            self.stop_recording()

    def start_recording(self) -> Recorder:
        recorder = self.session.start_recording(Recorder())
        recorder.listeners.append(lambda _event, _text: self.recording_updated.emit(recorder.script()))
        self.record_action.setChecked(True)
        self.record_action.setText("Stop Rec")
        self.record_action.setToolTip("Stop recording the macro (Ctrl+Shift+R)")
        self.show_editor()
        self.recording_changed.emit(True)
        self.window.statusBar().showMessage("Recording a macro: your actions are being written as Python.", 5000)
        return recorder

    def stop_recording(self):
        recorder = self.session.stop_recording()
        self.record_action.setChecked(False)
        self.record_action.setText("Record")
        self.record_action.setToolTip("Record your actions as a Python macro (Ctrl+Shift+R)")
        self.recording_changed.emit(False)
        return recorder

    def recording_script(self) -> str:
        recorder = self.session.recorder
        return recorder.script() if recorder is not None else ""

    # ---- lifecycle -----------------------------------------------------------------
    def confirm_quit(self) -> bool:
        """Ask about unsaved scripts before the application closes; False to cancel."""
        if self.session.is_recording:
            self.stop_recording()
        return self.editor is None or self.editor.confirm_close_all()

    def shutdown(self) -> None:
        if self.editor is not None:
            self.editor.close()
            self.editor.deleteLater()
            self.editor = None
        set_current_app(None)
        set_current_session(None)
