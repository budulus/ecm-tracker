"""Execute user code with captured output, readable tracebacks, and a working Stop button.

:class:`ScriptExecutor` is what the console and Script Editor run code through. It is Qt-free:
the GUI passes ``poll`` (``QApplication.processEvents``) so the window stays responsive while a
script loops, and calls :meth:`ScriptExecutor.request_stop` from a Stop button.

Interrupting is done with ``sys.monitoring`` (PEP 669) LINE events enabled *only* on the code
objects compiled from the user's source, so library code (numpy, OpenCV, the app) runs at full
speed and debuggers that use ``sys.settrace`` are not displaced. Stop raises
``KeyboardInterrupt`` at the next executed line of user code; long session commands (tracking)
are cancelled through their progress dialog instead.
"""
from __future__ import annotations

import contextlib
import linecache
import sys
import time
import traceback
import types
from dataclasses import dataclass
from typing import Callable, Optional

_TOOL_NAME = "ecm-tracker-scripts"
_tool_id: Optional[int] = None
_active: Optional["ScriptExecutor"] = None


def _monitoring_tool() -> Optional[int]:
    """Claim a sys.monitoring tool id once per process (None if unavailable)."""
    global _tool_id
    if _tool_id is not None:
        return _tool_id
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return None
    for candidate in (4, 3):
        try:
            monitoring.use_tool_id(candidate, _TOOL_NAME)
        except ValueError:
            continue
        monitoring.register_callback(candidate, monitoring.events.LINE, _on_line)
        _tool_id = candidate
        return candidate
    return None


def _on_line(code, line_number):
    executor = _active
    if executor is None or not executor.running:
        return sys.monitoring.DISABLE
    executor._tick()
    return None


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


class _Stream:
    """File-like object forwarding writes to a callback (stdout/stderr of user code)."""

    def __init__(self, write: Callable[[str], None]):
        self._write = write
        self.encoding = "utf-8"

    def write(self, text) -> int:
        text = str(text)
        if text:
            self._write(text)
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@dataclass
class RunOutcome:
    """How a run ended: ``ok``; else the exception, and the failing line in the run's file."""

    ok: bool
    error: Optional[BaseException] = None
    line: Optional[int] = None
    interrupted: bool = False


class ScriptExecutor:
    """Runs source code in ``namespace`` with output routed to ``write_out`` / ``write_err``."""

    def __init__(self, namespace: dict, write_out: Callable[[str], None],
                 write_err: Callable[[str], None], *, poll: Optional[Callable[[], None]] = None,
                 poll_interval: float = 0.05):
        self.namespace = namespace
        self.write_out = write_out
        self.write_err = write_err
        self.poll = poll
        self.poll_interval = poll_interval
        self.running = False
        self._stop_requested = False
        self._in_poll = False
        self._last_poll = 0.0

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def request_stop(self) -> None:
        """Ask the running code to stop at its next line (raises KeyboardInterrupt there)."""
        if self.running:
            self._stop_requested = True

    def _tick(self) -> None:
        if self._stop_requested:
            self._stop_requested = False
            raise KeyboardInterrupt("Script stopped")
        if self.poll is None or self._in_poll:
            return
        now = time.monotonic()
        if now - self._last_poll >= self.poll_interval:
            self._last_poll = now
            self._in_poll = True
            try:
                self.poll()
            finally:
                self._in_poll = False
            if self._stop_requested:
                self._stop_requested = False
                raise KeyboardInterrupt("Script stopped")

    @staticmethod
    def register_source(filename: str, source: str) -> None:
        """Make ``source`` available to tracebacks under ``filename`` (also for unsaved buffers)."""
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    def compile(self, source: str, filename: str, mode: str = "exec"):
        """Compile, printing a SyntaxError like Python does; returns the code or the error."""
        self.register_source(filename, source)
        try:
            return compile(source, filename, mode)
        except (SyntaxError, ValueError, OverflowError) as exc:
            self.write_err("".join(traceback.format_exception_only(type(exc), exc)))
            return exc

    def run_source(self, source: str, filename: str, mode: str = "exec") -> RunOutcome:
        code = self.compile(source, filename, mode)
        if not isinstance(code, types.CodeType):
            return RunOutcome(False, code, getattr(code, "lineno", None))
        return self.run_code(code, filename)

    def run_code(self, code: types.CodeType, filename: str) -> RunOutcome:
        """Execute compiled ``code``; exceptions are printed (trimmed to user frames), not raised."""
        global _active
        if self.running:
            raise RuntimeError("A script is already running.")
        tool = _monitoring_tool()
        monitoring = getattr(sys, "monitoring", None)
        if tool is not None:
            monitoring.restart_events()
            for obj in _code_objects(code):
                monitoring.set_local_events(tool, obj, monitoring.events.LINE)
        previous, _active = _active, self
        self.running = True
        self._stop_requested = False
        self._last_poll = time.monotonic()
        try:
            with contextlib.redirect_stdout(_Stream(self.write_out)), \
                    contextlib.redirect_stderr(_Stream(self.write_err)):
                try:
                    exec(code, self.namespace)
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        self.write_err(f"SystemExit: {exc.code}\n")
                    return RunOutcome(exc.code in (None, 0), exc)
                except KeyboardInterrupt as exc:
                    self.write_err("KeyboardInterrupt: script stopped\n")
                    return RunOutcome(False, exc, _error_line(exc, filename), interrupted=True)
                except BaseException as exc:  # user code may raise anything; report it
                    self.write_err(format_user_traceback(exc, filename))
                    return RunOutcome(False, exc, _error_line(exc, filename))
            return RunOutcome(True)
        finally:
            self.running = False
            self._stop_requested = False
            _active = previous


def _error_line(exc: BaseException, filename: str) -> Optional[int]:
    """Line of the innermost frame of ``filename`` in the traceback (for editor highlighting)."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == filename:
            line = frame.lineno
    return line


def format_user_traceback(exc: BaseException, filename: str) -> str:
    """Format ``exc`` with the host's own frames (above the user's code) removed."""
    report = traceback.TracebackException.from_exception(exc)
    frames = list(report.stack)
    for index, frame in enumerate(frames):
        if frame.filename == filename:
            frames = frames[index:]
            break
    report.stack = traceback.StackSummary.from_list(frames)
    return "".join(report.format())
