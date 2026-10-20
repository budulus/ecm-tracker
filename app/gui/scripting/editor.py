"""The Script Editor window: tabbed Python editing, file save/load, Run / Stop, macro recording.

Scripts run in the same namespace as the Python console (``session``, ``app``, ``np`` …), so
variables a script defines can be inspected in the console afterwards. A run executes the
buffer as shown (saved or not); its output appears in the pane below the editor and an error
line is highlighted. While the macro recorder is on, a read-only "Recording" tab shows the
macro growing live; when recording stops it becomes an ordinary unsaved script.
"""
import os
import textwrap
import time

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from app.gui.icon_loader import ACCENT, load_icon
from app.gui.scripting.console import FORMATS, monospace_font
from app.gui.scripting.highlighter import PythonHighlighter

INDENT = "    "
FILE_FILTER = "Python scripts (*.py);;All files (*)"


class _LineNumbers(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:
        self.editor.paint_gutter(event)


class CodeEditor(QPlainTextEdit):
    """Plain-text Python editor: line numbers, current-line highlight, auto-indent, Tab/Shift+Tab
    block indent, Ctrl+/ comment toggle, and an error-line marker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("codeEditor")
        self.path = None  # file on disk, or None for an unsaved buffer
        self.untitled_name = "Untitled"
        self.setFont(monospace_font())
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.highlighter = PythonHighlighter(self.document())
        self._gutter = _LineNumbers(self)
        self._error_line = None
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_lines)
        self.textChanged.connect(self._clear_error)
        self._update_gutter_width()
        self._highlight_lines()

    @property
    def display_name(self) -> str:
        return os.path.basename(self.path) if self.path else self.untitled_name

    # ---- gutter -----------------------------------------------------------
    def gutter_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter_width(self, _count=0) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect, dy) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, event) -> None:
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), QColor("#f1f3f5"))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                if number + 1 == self._error_line:
                    painter.setPen(QColor("#b91c1c"))
                elif number == current:
                    painter.setPen(QColor("#2b2f33"))
                else:
                    painter.setPen(QColor("#9aa1a9"))
                painter.drawText(0, top, self._gutter.width() - 6, self.fontMetrics().height(),
                                 Qt.AlignmentFlag.AlignRight, str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1

    # ---- highlights -------------------------------------------------------
    def _highlight_lines(self) -> None:
        selections = []
        if not self.isReadOnly():
            current = QTextEdit.ExtraSelection()
            current.format.setBackground(QColor("#f5f8ff"))
            current.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            current.cursor = self.textCursor()
            current.cursor.clearSelection()
            selections.append(current)
        if self._error_line is not None:
            block = self.document().findBlockByNumber(self._error_line - 1)
            if block.isValid():
                error = QTextEdit.ExtraSelection()
                error.format.setBackground(QColor("#fde8e8"))
                error.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                error.cursor = QTextCursor(block)
                selections.append(error)
        self.setExtraSelections(selections)

    def mark_error(self, line) -> None:
        """Highlight 1-based ``line`` as the failing line and move the cursor there."""
        self._error_line = line
        if line is not None:
            block = self.document().findBlockByNumber(line - 1)
            if block.isValid():
                self.setTextCursor(QTextCursor(block))
                self.centerCursor()
        self._highlight_lines()
        self._gutter.update()

    def _clear_error(self) -> None:
        if self._error_line is not None:
            self._error_line = None
            self._highlight_lines()
            self._gutter.update()

    # ---- editing ------------------------------------------------------------
    def selected_lines(self):
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart()).blockNumber()
        last_block = doc.findBlock(cursor.selectionEnd())
        last = last_block.blockNumber()
        if cursor.hasSelection() and cursor.selectionEnd() == last_block.position() and last > first:
            last -= 1  # a selection ending at a line start does not include that line
        return first, last

    def _edit_lines(self, transform) -> None:
        first, last = self.selected_lines()
        doc = self.document()
        cursor = QTextCursor(doc.findBlockByNumber(first))
        cursor.beginEditBlock()
        for number in range(first, last + 1):
            block = doc.findBlockByNumber(number)
            line_cursor = QTextCursor(block)
            line_cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            new = transform(block.text())
            if new != block.text():
                line_cursor.insertText(new)
        cursor.endEditBlock()
        start = doc.findBlockByNumber(first).position()
        end_block = doc.findBlockByNumber(last)
        select = QTextCursor(doc)
        select.setPosition(start)
        select.setPosition(end_block.position() + len(end_block.text()), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(select)

    def indent_selection(self) -> None:
        self._edit_lines(lambda line: INDENT + line if line.strip() else line)

    def dedent_selection(self) -> None:
        def dedent(line):
            strip = len(line) - len(line.lstrip(" "))
            return line[min(strip, len(INDENT)):]
        self._edit_lines(dedent)

    def toggle_comment(self) -> None:
        first, last = self.selected_lines()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        code = [line for line in lines if line.strip()]
        uncomment = bool(code) and all(line.lstrip().startswith("#") for line in code)

        def toggle(line):
            if not line.strip():
                return line
            if uncomment:
                index = line.index("#")
                rest = line[index + 1:]
                return line[:index] + (rest[1:] if rest.startswith(" ") else rest)
            indent = len(line) - len(line.lstrip())
            return line[:indent] + "# " + line[indent:]
        self._edit_lines(toggle)

    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        cursor = self.textCursor()
        if key == Qt.Key.Key_Tab and not mods:
            if cursor.hasSelection() and " " in cursor.selectedText():
                self.indent_selection()
            else:
                column = cursor.positionInBlock()
                cursor.insertText(" " * (len(INDENT) - column % len(INDENT)))
            return
        if key == Qt.Key.Key_Backtab:
            self.dedent_selection()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (mods & Qt.KeyboardModifier.ControlModifier):
            line = cursor.block().text()[: cursor.positionInBlock()]
            indent = line[: len(line) - len(line.lstrip())]
            if line.rstrip().endswith(":"):
                indent += INDENT
            elif line.strip() in ("return", "pass", "break", "continue") or line.strip().startswith(("return ", "raise ")):
                indent = indent[: max(0, len(indent) - len(INDENT))]
            cursor.insertText("\n" + indent)
            self.ensureCursorVisible()
            return
        if key == Qt.Key.Key_Backspace and not cursor.hasSelection() and not mods:
            before = cursor.block().text()[: cursor.positionInBlock()]
            if before and not before.strip() and len(before) % len(INDENT) == 0:
                for _ in range(len(INDENT)):
                    cursor.deletePreviousChar()
                return
        super().keyPressEvent(event)


class ScriptEditorWindow(QMainWindow):
    """Tabbed script editor bound to a :class:`~app.gui.scripting.host.ScriptingHost`."""

    closed = Signal()

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.setWindowTitle("Script Editor — ECM Tracker")
        self.resize(900, 760)
        self._untitled = 0
        self._macros = 0
        self._recording_tab = None
        self._start_dir = host.script_dir()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("scriptTabs")
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._update_actions)

        self.output = QPlainTextEdit()
        self.output.setObjectName("scriptOutput")
        self.output.setReadOnly(True)
        self.output.setFont(monospace_font())
        self.output.setPlaceholderText("Script output appears here.")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.output)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 200])
        self.setCentralWidget(splitter)

        self._build_actions()
        self._build_menus_and_toolbar()
        self.statusBar().showMessage("F5 runs the current script; F9 runs the selection or current line.")

        host.output.connect(self._on_output)
        host.busy_changed.connect(self._update_actions)
        host.recording_changed.connect(self._on_recording_changed)
        host.recording_updated.connect(self._on_recording_updated)
        self.new_script()

    # ---- actions --------------------------------------------------------------
    def _action(self, text, slot, shortcut=None, icon=None, tip=None) -> QAction:
        action = QAction(load_icon(icon) if icon else QIcon(), text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setToolTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        self.new_action = self._action("New", self.new_script, QKeySequence.StandardKey.New, "file-plus", "New script")
        self.open_action = self._action("Open…", self.open_script_dialog, QKeySequence.StandardKey.Open, "folder-open", "Open a script")
        self.save_action = self._action("Save", self.save_current, QKeySequence.StandardKey.Save, "save", "Save the script")
        self.save_as_action = self._action("Save As…", self.save_current_as, "Ctrl+Shift+S")
        self.close_tab_action = self._action("Close Tab", lambda: self.close_tab(self.tabs.currentIndex()), QKeySequence.StandardKey.Close)
        self.run_action = self._action("Run", self.run_current, "F5", "play", "Run the whole script (F5)")
        self.run_action.setIcon(load_icon("play", ACCENT))
        self.run_selection_action = self._action("Run Selection", self.run_selection, "F9", "text-select",
                                                 "Run the selected lines, or the current line (F9)")
        self.stop_action = self._action("Stop", self.host.request_stop, "Shift+F5", "square", "Stop the running script (Shift+F5)")
        self.comment_action = self._action("Toggle Comment", lambda: self._editor_call("toggle_comment"), "Ctrl+/")
        self.indent_action = self._action("Indent", lambda: self._editor_call("indent_selection"), "Ctrl+]")
        self.dedent_action = self._action("Dedent", lambda: self._editor_call("dedent_selection"), "Ctrl+[")
        self.clear_output_action = self._action("Clear Output", self.output.clear, None, "trash", "Clear the output pane")

    def _build_menus_and_toolbar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (self.new_action, self.open_action):
            file_menu.addAction(action)
        self._examples_menu = file_menu.addMenu("Open Example")
        self._examples_menu.aboutToShow.connect(self._fill_examples_menu)
        file_menu.addSeparator()
        for action in (self.save_action, self.save_as_action, self.close_tab_action):
            file_menu.addAction(action)

        edit_menu = self.menuBar().addMenu("&Edit")
        for action in (self.comment_action, self.indent_action, self.dedent_action):
            edit_menu.addAction(action)

        run_menu = self.menuBar().addMenu("&Run")
        for action in (self.run_action, self.run_selection_action, self.stop_action):
            run_menu.addAction(action)
        run_menu.addSeparator()
        run_menu.addAction(self.host.record_action)
        run_menu.addAction(self.clear_output_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Session API Reference", self.show_api_reference)

        toolbar = self.addToolBar("Script")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for action in (self.new_action, self.open_action, self.save_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.run_selection_action)
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        toolbar.addAction(self.host.record_action)
        toolbar.addSeparator()
        toolbar.addAction(self.clear_output_action)
        run_button = toolbar.widgetForAction(self.run_action)
        if run_button is not None:
            run_button.setObjectName("primaryAction")

    def _editor_call(self, name: str) -> None:
        editor = self.current_editor()
        if editor is not None and not editor.isReadOnly():
            getattr(editor, name)()

    def _update_actions(self, *_args) -> None:
        busy = self.host.busy
        editor = self.current_editor()
        has_editor = editor is not None
        for action in (self.run_action, self.run_selection_action):
            action.setEnabled(has_editor and not busy)
        self.stop_action.setEnabled(busy)
        self.save_action.setEnabled(has_editor)
        self.save_as_action.setEnabled(has_editor)

    # ---- tabs -------------------------------------------------------------
    def editors(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def current_editor(self):
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def _add_editor(self, text: str = "", path=None, name=None) -> CodeEditor:
        editor = CodeEditor()
        editor.path = path
        if name is None:
            self._untitled += 1
            name = f"Untitled-{self._untitled}"
        editor.untitled_name = name
        editor.setPlainText(text)
        editor.document().setModified(False)
        editor.document().modificationChanged.connect(lambda _m, e=editor: self._refresh_tab(e))
        editor.cursorPositionChanged.connect(lambda e=editor: self._show_position(e))
        index = self.tabs.addTab(editor, "")
        self._refresh_tab(editor)
        self.tabs.setCurrentIndex(index)
        editor.setFocus()
        self._update_actions()
        return editor

    def _refresh_tab(self, editor) -> None:
        index = self.tabs.indexOf(editor)
        if index < 0:
            return
        title = editor.display_name
        if editor is self._recording_tab:
            title = "● " + title
        elif editor.document().isModified():
            title += " *"
        self.tabs.setTabText(index, title)
        self.tabs.setTabToolTip(index, editor.path or "Not saved yet")
        if editor is self.current_editor():
            self.setWindowTitle(f"{editor.display_name} — Script Editor — ECM Tracker")

    def _show_position(self, editor) -> None:
        cursor = editor.textCursor()
        self.statusBar().showMessage(f"Line {cursor.blockNumber() + 1}, Column {cursor.positionInBlock() + 1}")

    def new_script(self, text: str = "") -> CodeEditor:
        if not text:
            text = "# ECM Tracker script. `session`, `app` and `np` are predefined.\n\n"
        editor = self._add_editor(text)
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        return editor

    def open_script(self, path) -> CodeEditor:
        """Open ``path`` in a tab (or focus it if already open)."""
        path = os.path.abspath(path)
        for editor in self.editors():
            if editor.path and os.path.normcase(editor.path) == os.path.normcase(path):
                self.tabs.setCurrentWidget(editor)
                return editor
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        current = self.current_editor()
        replace = (current is not None and current.path is None and current is not self._recording_tab
                   and not current.document().isModified() and self.tabs.count() == 1)
        editor = self._add_editor(text, path)
        if replace:
            self.tabs.removeTab(self.tabs.indexOf(current))
            current.deleteLater()
        self._start_dir = os.path.dirname(path)
        self.host.remember_script_dir(self._start_dir)
        return editor

    def open_script_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Script", self._start_dir, FILE_FILTER)
        if path:
            try:
                self.open_script(path)
            except (OSError, UnicodeDecodeError) as exc:
                QMessageBox.critical(self, "Open failed", str(exc))

    def _fill_examples_menu(self) -> None:
        self._examples_menu.clear()
        folder = self.host.examples_dir()
        names = sorted(n for n in os.listdir(folder) if n.endswith(".py")) if folder and os.path.isdir(folder) else []
        if not names:
            self._examples_menu.addAction("(no examples found)").setEnabled(False)
        for name in names:
            path = os.path.join(folder, name)
            self._examples_menu.addAction(name, lambda p=path: self._open_example(p))

    def _open_example(self, path) -> None:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        editor = self._add_editor(text, name=os.path.basename(path).replace(".py", " (example)"))
        editor.document().setModified(True)  # saving asks for a location instead of overwriting the example

    def save_editor(self, editor, path=None) -> bool:
        path = path or editor.path
        if path is None:
            return self.save_editor_as(editor)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(editor.toPlainText())
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        editor.path = os.path.abspath(path)
        editor.document().setModified(False)
        self._refresh_tab(editor)
        self._start_dir = os.path.dirname(editor.path)
        self.host.remember_script_dir(self._start_dir)
        self.statusBar().showMessage(f"Saved {editor.path}", 4000)
        return True

    def save_editor_as(self, editor) -> bool:
        suggested = editor.path or os.path.join(self._start_dir, editor.display_name.split(" ")[0] + ".py")
        path, _ = QFileDialog.getSaveFileName(self, "Save Script", suggested, FILE_FILTER)
        if not path:
            return False
        if not os.path.splitext(path)[1]:
            path += ".py"
        return self.save_editor(editor, path)

    def save_current(self) -> bool:
        editor = self.current_editor()
        return editor is not None and self.save_editor(editor)

    def save_current_as(self) -> bool:
        editor = self.current_editor()
        return editor is not None and self.save_editor_as(editor)

    def _confirm_close(self, editor) -> bool:
        if editor is self._recording_tab:
            QMessageBox.information(self, "Recording", "Stop recording before closing the macro tab.")
            return False
        if not editor.document().isModified():
            return True
        self.tabs.setCurrentWidget(editor)
        answer = QMessageBox.question(
            self, "Unsaved script", f"Save changes to {editor.display_name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_editor(editor)
        return answer == QMessageBox.StandardButton.Discard

    def close_tab(self, index: int) -> bool:
        editor = self.tabs.widget(index)
        if editor is None or not self._confirm_close(editor):
            return False
        self.tabs.removeTab(index)
        editor.deleteLater()
        if self.tabs.count() == 0:
            self.new_script()
        self._update_actions()
        return True

    def confirm_close_all(self) -> bool:
        """Ask about every unsaved tab; False if the user cancelled."""
        return all(self._confirm_close(e) for e in self.editors() if e is not self._recording_tab)

    def closeEvent(self, event) -> None:
        # Unsaved tabs survive a plain window close (the window is only hidden); the main window
        # asks about them when the application quits.
        self.closed.emit()
        super().closeEvent(event)

    # ---- running ------------------------------------------------------------
    def _on_output(self, channel: str, kind: str, text: str) -> None:
        if channel != "editor":
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, FORMATS.get(kind, FORMATS["out"]))
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _filename(self, editor) -> str:
        return editor.path or f"<{editor.display_name}>"

    def run_current(self):
        editor = self.current_editor()
        if editor is None or self.host.busy:
            return None
        return self._run(editor, editor.toPlainText(), editor.display_name)

    def run_selection(self):
        editor = self.current_editor()
        if editor is None or self.host.busy:
            return None
        cursor = editor.textCursor()
        if cursor.hasSelection():
            first, last = editor.selected_lines()
            lines = [editor.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
            source = "\n".join(lines)
        else:
            first = cursor.blockNumber()
            source = cursor.block().text()
        # Keep line numbers in tracebacks aligned with the editor, dedent to allow running a
        # nested block on its own.
        source = "\n" * first + textwrap.dedent(source)
        label = f"{editor.display_name} (selection)"
        return self._run(editor, source, label)

    def _run(self, editor, source: str, label: str):
        editor.mark_error(None)
        self._on_output("editor", "info", f"▶ Running {label}\n")
        started = time.perf_counter()
        outcome = self.host.execute_source(source, self._filename(editor), channel="editor",
                                           script_path=editor.path)
        elapsed = time.perf_counter() - started
        if outcome.ok:
            self._on_output("editor", "info", f"✓ Finished in {elapsed:.2f} s\n\n")
        else:
            where = f" at line {outcome.line}" if outcome.line else ""
            verb = "Stopped" if outcome.interrupted else "Failed"
            self._on_output("editor", "err", f"✗ {verb}{where} after {elapsed:.2f} s\n\n")
            if outcome.line and self.tabs.indexOf(editor) >= 0:
                editor.mark_error(outcome.line)
        return outcome

    # ---- recording ----------------------------------------------------------
    def _on_recording_changed(self, recording: bool) -> None:
        if recording:
            self._macros += 1
            editor = self._add_editor(self.host.recording_script(), name=f"Macro-{self._macros}")
            editor.setReadOnly(True)
            self._recording_tab = editor
            self._refresh_tab(editor)
            self.statusBar().showMessage("Recording: every action you take is added to the macro tab.")
        elif self._recording_tab is not None:
            editor, self._recording_tab = self._recording_tab, None
            editor.setReadOnly(False)
            editor.document().setModified(True)
            self._refresh_tab(editor)
            self.tabs.setCurrentWidget(editor)
            self.statusBar().showMessage("Recording stopped. Edit the macro, run it (F5) or save it (Ctrl+S).", 8000)

    def _on_recording_updated(self, script: str) -> None:
        editor = self._recording_tab
        if editor is None:
            return
        editor.setPlainText(script)
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        editor.ensureCursorVisible()

    # ---- help ---------------------------------------------------------------
    def show_api_reference(self) -> None:
        editor = self._add_editor(self.host.api_reference(), name="Session API")
        editor.setReadOnly(True)
