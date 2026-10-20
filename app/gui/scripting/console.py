"""Interactive Python console widget (the "kernel command line").

A ``QPlainTextEdit`` that behaves like the standard Python REPL: ``>>>`` / ``...`` prompts,
multi-line statements, echo of expression values, Up/Down history, Tab completion, Ctrl+C to
interrupt a running command (or discard the current input), Ctrl+L to clear. Code runs through
the shared :class:`~app.gui.scripting.host.ScriptingHost`, so it sees the same ``session`` /
``app`` / ``np`` namespace as the Script Editor, and every session command it issues updates the
GUI exactly like the equivalent toolbar action.
"""
import codeop
import re
import rlcompleter

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QKeySequence, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

PS1 = ">>> "
PS2 = "... "
BANNER = (
    "ECM Tracker Python console. `session` drives the open project (help(session) lists every "
    "command), `app` the GUI, `np` is numpy.\n"
    "Tab completes, Up/Down browse history, Ctrl+C interrupts. Record macros with the Record button.\n"
)
_TOKEN = re.compile(r"[\w.]*$")
HISTORY_LIMIT = 500


def monospace_font() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(max(font.pointSize(), 10))
    return font


def _char_format(color: str) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    return fmt


FORMATS = {
    "out": _char_format("#2b2f33"),
    "err": _char_format("#b91c1c"),
    "info": _char_format("#6b727a"),
    "prompt": _char_format("#2563eb"),
    "input": _char_format("#111827"),
}


class PythonConsole(QPlainTextEdit):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.setObjectName("scriptConsole")
        self.host = host
        self.setFont(monospace_font())
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setTabChangesFocus(False)
        self._compiler = codeop.CommandCompiler()
        self._buffer = []  # lines of the statement being entered
        self._history = list(host.load_history())
        self._history_index = len(self._history)
        self._draft = ""
        self._prompt_block = None  # block number of the active prompt line
        self._prompt_pos = 0  # document position where editable input starts
        host.output.connect(self._on_output)
        host.busy_changed.connect(self._on_busy)
        self.write(BANNER, "info")
        self._show_prompt(PS1)

    # ---------------------------------------------------------------- output
    def write(self, text: str, kind: str = "out") -> None:
        """Insert ``text`` above the active prompt (or at the end while code runs)."""
        if not text:
            return
        cursor = QTextCursor(self.document())
        if self._prompt_block is not None:
            block = self.document().findBlockByNumber(self._prompt_block)
            cursor.setPosition(block.position())
            if not text.endswith("\n"):
                text += "\n"
            before = self.document().characterCount()
            cursor.insertText(text, FORMATS.get(kind, FORMATS["out"]))
            grown = self.document().characterCount() - before
            self._prompt_pos += grown
            self._prompt_block = self.document().findBlock(self._prompt_pos).blockNumber()
        else:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(text, FORMATS.get(kind, FORMATS["out"]))
        self._scroll_to_end()

    def _on_output(self, channel: str, kind: str, text: str) -> None:
        if channel == "console":
            self.write(text, kind)

    def _scroll_to_end(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_console(self) -> None:
        pending = self._input_text()
        self.clear()
        self._prompt_block = None
        self._show_prompt(PS2 if self._buffer else PS1, pending)

    # ---------------------------------------------------------------- prompt
    def _show_prompt(self, prompt: str, text: str = "") -> None:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.document().characterCount() > 1 and not self.toPlainText().endswith("\n"):
            cursor.insertText("\n")
        cursor.insertText(prompt, FORMATS["prompt"])
        self._prompt_pos = cursor.position()
        self._prompt_block = cursor.blockNumber()
        cursor.insertText(text, FORMATS["input"])
        self.setTextCursor(cursor)
        self.setCurrentCharFormat(FORMATS["input"])
        self._scroll_to_end()

    def _input_text(self) -> str:
        if self._prompt_block is None:
            return ""
        cursor = QTextCursor(self.document())
        cursor.setPosition(self._prompt_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText().replace(" ", "\n")

    def _set_input_text(self, text: str) -> None:
        cursor = QTextCursor(self.document())
        cursor.setPosition(self._prompt_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(text, FORMATS["input"])
        self.setTextCursor(cursor)

    # ------------------------------------------------------------- execution
    def submit(self, text: str) -> None:
        """Feed ``text`` (one or more lines) to the interpreter as if typed and entered."""
        if self._prompt_block is not None and self._input_text() != text:
            self._set_input_text(text)  # programmatic input is echoed like typed input
        self._move_to_end()
        cursor = self.textCursor()
        cursor.insertText("\n")
        self._prompt_block = None
        if "\n" in text and not self._buffer and self._run_block(text):
            self._show_prompt(PS1)
            return
        more = False
        for line in text.split("\n"):
            more = self._push(line)
        indent = ""
        if more:
            last = self._buffer[-1] if self._buffer else ""
            indent = re.match(r"\s*", last).group(0) + ("    " if last.rstrip().endswith(":") else "")
        self._show_prompt(PS2 if more else PS1, indent)

    def _push(self, line: str) -> bool:
        """Add one line; run the statement once complete. Returns True if more input is needed."""
        self._buffer.append(line)
        source = "\n".join(self._buffer)
        try:
            code = self._compiler(source, "<console>", "single")
        except (SyntaxError, OverflowError, ValueError):
            self._buffer = []
            self._remember(source)
            self.host.execute_source(source, "<console>", channel="console", mode="single")
            return False
        if code is None:
            return True
        self._buffer = []
        self._remember(source)
        self.host.execute_code(code, source, "<console>", channel="console")
        return False

    def _run_block(self, text: str) -> bool:
        """Run a pasted multi-statement block in one go; False if it is not a complete block."""
        try:
            code = compile(text, "<console>", "exec")
        except SyntaxError:
            return False
        self._remember(text)
        self.host.execute_code(code, text, "<console>", channel="console")
        return True

    def _remember(self, text: str) -> None:
        text = text.rstrip("\n")
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-HISTORY_LIMIT]
        self._history_index = len(self._history)
        self.host.save_history(self._history)

    def _on_busy(self, busy: bool) -> None:
        self.setReadOnly(busy)

    # ------------------------------------------------------------- keyboard
    def _move_to_end(self) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def _cursor_in_input(self) -> bool:
        cursor = self.textCursor()
        return self._prompt_block is not None and min(cursor.position(), cursor.anchor()) >= self._prompt_pos

    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if ctrl and key == Qt.Key.Key_C and not self.textCursor().hasSelection():
            self._interrupt()
            return
        if ctrl and key == Qt.Key.Key_L:
            self.clear_console()
            return
        if event.matches(QKeySequence.StandardKey.Copy) or (ctrl and key == Qt.Key.Key_A):
            super().keyPressEvent(event)
            return
        if self.host.busy or self._prompt_block is None:
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.submit(self._input_text())
            return
        if key == Qt.Key.Key_Up and not self._input_has_newline():
            self._browse_history(-1)
            return
        if key == Qt.Key.Key_Down and not self._input_has_newline():
            self._browse_history(+1)
            return
        if key == Qt.Key.Key_Tab:
            self._complete()
            return
        if key == Qt.Key.Key_Home and not (mods & Qt.KeyboardModifier.ShiftModifier):
            cursor = self.textCursor()
            cursor.setPosition(self._prompt_pos)
            self.setTextCursor(cursor)
            return
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Left):
            cursor = self.textCursor()
            if not cursor.hasSelection() and cursor.position() <= self._prompt_pos:
                return
        if not self._cursor_in_input():
            if event.text() or key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                self._move_to_end()
            else:  # navigation keys still move freely through the transcript
                super().keyPressEvent(event)
                return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source) -> None:
        if self.host.busy or self._prompt_block is None:
            return
        if not self._cursor_in_input():
            self._move_to_end()
        self.textCursor().insertText(source.text().replace("\r\n", "\n"), FORMATS["input"])

    def _input_has_newline(self) -> bool:
        return "\n" in self._input_text()

    def _interrupt(self) -> None:
        if self.host.busy:
            self.host.request_stop()
            return
        if self._prompt_block is None:
            return
        self._move_to_end()
        self.textCursor().insertText("\n")
        self._prompt_block = None
        self._buffer = []
        self.write("KeyboardInterrupt\n", "err")
        self._show_prompt(PS1)

    def _browse_history(self, step: int) -> None:
        if not self._history:
            return
        if self._history_index == len(self._history):
            self._draft = self._input_text()
        self._history_index = max(0, min(len(self._history), self._history_index + step))
        text = self._draft if self._history_index == len(self._history) else self._history[self._history_index]
        self._set_input_text(text)

    def _complete(self) -> None:
        cursor = self.textCursor()
        line = self._input_text()[: cursor.position() - self._prompt_pos]
        token = _TOKEN.search(line.split("\n")[-1]).group(0)
        if not token or line.split("\n")[-1].strip() == "":
            cursor.insertText("    ")
            return
        completer = rlcompleter.Completer(self.host.namespace)
        matches = []
        try:
            for state in range(1000):
                match = completer.complete(token, state)
                if match is None:
                    break
                if match not in matches:
                    matches.append(match)
        except Exception:  # completion must never break typing
            return
        if not matches:
            return
        common = matches[0]
        for match in matches[1:]:
            while not match.startswith(common):
                common = common[:-1]
        if len(common) > len(token):
            cursor.insertText(common[len(token):])
        elif len(matches) > 1:
            names = sorted({m.rsplit(".", 1)[-1] for m in matches})
            self.write("  ".join(names) + "\n", "info")
