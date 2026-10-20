"""Macro recorder: turns executed commands into a runnable Python script draft.

While a :class:`Recorder` is attached to a :class:`~app.scripting.session.Session`, every
top-level session command (whether triggered by a toolbar click, a plugin, or the console) is
appended here as one line of Python, e.g. ``session.detect_corners(maxCorners=500, ...)``.
:meth:`Recorder.script` wraps the lines in a header that makes the result runnable inside the
app's Script Editor, headless (``tracker --run macro.py``), or as plain ``python macro.py``.

Qt-free. Values are formatted as Python literals (:func:`format_value`); long calls wrap one
argument per line so a recorded macro stays readable and editable.
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

MAX_LINE = 99
INDENT = "    "


def format_value(value, indent: str = "") -> str:
    """Python source for ``value`` that evaluates back to an equal value.

    Handles the types commands receive: None/bool/int/float/str/path-likes, numpy scalars and
    arrays (as nested lists), tuples, lists and dicts. Long containers wrap one item per line,
    continuing at ``indent``.
    """
    flat = _format_flat(value)
    if len(indent) + len(flat) <= MAX_LINE or not isinstance(value, (list, tuple, dict, np.ndarray)):
        return flat
    inner = indent + INDENT
    if isinstance(value, dict):
        items = [f"{inner}{_format_flat(k)}: {format_value(v, inner)}," for k, v in value.items()]
        return "{\n" + "\n".join(items) + f"\n{indent}}}"
    seq = value.tolist() if isinstance(value, np.ndarray) else value
    items = [f"{inner}{format_value(v, inner)}," for v in seq]
    open_, close = ("(", ")") if isinstance(seq, tuple) else ("[", "]")
    return open_ + "\n" + "\n".join(items) + f"\n{indent}{close}"


def _format_flat(value) -> str:
    if isinstance(value, np.ndarray):
        return _format_flat(value.tolist())
    if isinstance(value, np.generic):
        return _format_flat(value.item())
    if value is None or isinstance(value, (bool, int)):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'float("nan")'
        if math.isinf(value):
            return 'float("inf")' if value > 0 else '-float("inf")'
        return repr(value)
    if isinstance(value, os.PathLike):
        return _format_flat(os.fspath(value))
    if isinstance(value, str):
        # Raw strings keep Windows paths readable; they cannot end in a backslash or hold quotes.
        if "\\" in value and not value.endswith("\\") and '"' not in value and "\n" not in value:
            return f'r"{value}"'
        return repr(value)
    if isinstance(value, tuple):
        inner = ", ".join(_format_flat(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_format_flat(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_format_flat(k)}: {_format_flat(v)}" for k, v in value.items()) + "}"
    return repr(value)


def format_call(target: str, name: str, args=(), kwargs=None) -> str:
    """``target.name(args..., key=value...)``, wrapped one argument per line when too long."""
    kwargs = kwargs or {}
    parts = [_format_flat(a) for a in args] + [f"{k}={_format_flat(v)}" for k, v in kwargs.items()]
    line = f"{target}.{name}({', '.join(parts)})"
    if len(line) <= MAX_LINE:
        return line
    parts = [format_value(a, INDENT) for a in args] + [
        f"{k}={format_value(v, INDENT)}" for k, v in kwargs.items()
    ]
    body = "\n".join(f"{INDENT}{p}," for p in parts)
    return f"{target}.{name}(\n{body}\n)"


class Recorder:
    """Collects recorded commands as Python source entries.

    ``listeners`` receive ``(event, text)`` as entries change: ``"append"`` adds an entry,
    ``"replace"`` rewrites the last one (consecutive coalescing commands such as a dragged
    frame-range slider collapse into a single line). The GUI uses this for a live macro view.
    """

    def __init__(self, title: Optional[str] = None):
        self.title = title
        self.started = datetime.now()
        self.entries: List[str] = []
        self.listeners: List[Callable[[str, str], None]] = []
        self._coalesce_key = None
        self._targets = set()

    def record_call(self, target: str, name: str, args=(), kwargs=None, *, coalesce: bool = False) -> None:
        """Record ``target.name(*args, **kwargs)``. With ``coalesce``, an immediately preceding
        call of the same command is replaced instead of followed."""
        text = format_call(target, name, args, kwargs)
        key = (target, name) if coalesce else None
        self._targets.add(target)
        if key is not None and key == self._coalesce_key and self.entries:
            self.entries[-1] = text
            self._emit("replace", text)
        else:
            self.entries.append(text)
            self._emit("append", text)
        self._coalesce_key = key

    def record_source(self, source: str) -> None:
        """Record arbitrary Python source verbatim (e.g. a statement typed in the console)."""
        text = source.rstrip()
        if not text:
            return
        if re.search(r"\bapp\.", text):
            self._targets.add("app")
        self._coalesce_key = None
        self.entries.append(text)
        self._emit("append", text)

    def record_comment(self, comment: str) -> None:
        self.record_source("\n".join(f"# {line}" for line in comment.splitlines()))

    def clear(self) -> None:
        self.entries.clear()
        self._coalesce_key = None
        self._targets.clear()

    def body(self) -> str:
        """Just the recorded statements, one entry per line/block."""
        return "\n".join(self.entries)

    def header(self) -> str:
        title = self.title or "ECM Tracker macro"
        lines = [
            f"# {title} - recorded {self.started:%Y-%m-%d %H:%M}",
            "# Run it in the Script Editor (F5), headless with `tracker --run <file>`,",
            "# or as a plain Python program with the ecm_tracker package importable.",
            "from ecm_tracker import get_app, get_session" if "app" in self._targets
            else "from ecm_tracker import get_session",
            "",
            "session = get_session()",
        ]
        if "app" in self._targets:
            lines.append("app = get_app()  # GUI-only commands (plugins, view); unavailable headless")
        return "\n".join(lines) + "\n\n"

    def script(self) -> str:
        """The complete, runnable script: header plus every recorded entry."""
        return self.header() + self.body() + ("\n" if self.entries else "")

    def _emit(self, event: str, text: str) -> None:
        for listener in list(self.listeners):
            listener(event, text)
