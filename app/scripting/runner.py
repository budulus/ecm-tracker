"""Running scripts against a session, headless or inside the GUI.

* :func:`get_session` / :func:`get_app` return the session (and GUI app object) a script should
  drive. Inside the running application they are the GUI's; in a plain Python program
  ``get_session()`` creates a headless :class:`~app.scripting.session.Session` on first use.
* :func:`run_script` executes a ``.py`` file with ``session``, ``app``, ``np`` … predefined.
* :func:`make_namespace` builds that predefined namespace (the console and Script Editor use it).

Qt-free: the GUI registers itself through :func:`set_current_session` / :func:`set_current_app`.
"""
from __future__ import annotations

import builtins
import os
import sys
from typing import Optional, Sequence

import numpy as np

from app.scripting.session import Session

_current_session: Optional[Session] = None
_current_app = None


class HeadlessApp:
    """Stand-in for the GUI ``app`` object when no GUI is running.

    Recorded macros may contain GUI-only commands such as ``app.launch_plugin(...)``; run
    headless, those raise a clear error instead of an obscure ``NameError``."""

    available = False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"app.{name} needs the ECM Tracker GUI; this script is running headless.")

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<HeadlessApp: no GUI running>"


HEADLESS_APP = HeadlessApp()


def get_session() -> Session:
    """The session scripts should drive: the GUI's when running inside the app (or the one a
    runner bound), otherwise a headless session created on first use."""
    global _current_session
    if _current_session is None:
        _current_session = Session()
    return _current_session


def set_current_session(session: Optional[Session]) -> Optional[Session]:
    """Bind the session returned by :func:`get_session`; returns the previous one."""
    global _current_session
    previous, _current_session = _current_session, session
    return previous


def get_app():
    """The GUI scripting object (``app``), or a :class:`HeadlessApp` when no GUI is running."""
    return _current_app if _current_app is not None else HEADLESS_APP


def set_current_app(app) -> None:
    global _current_app
    _current_app = app


def in_gui() -> bool:
    return _current_app is not None


def make_namespace(session: Optional[Session] = None, **extra) -> dict:
    """Globals for console input and scripts: ``session``, ``app``, ``np`` and the helpers."""
    namespace = {
        "__name__": "__main__",
        "__builtins__": builtins,
        "session": session if session is not None else get_session(),
        "app": get_app(),
        "np": np,
        "Session": Session,
        "get_session": get_session,
        "get_app": get_app,
        "run_script": run_script,
    }
    namespace.update(extra)
    return namespace


def run_script(path, *, session: Optional[Session] = None, args: Sequence[str] = (),
               namespace: Optional[dict] = None) -> dict:
    """Execute the Python script at ``path`` and return its globals.

    The script sees ``session`` (``session`` argument, else :func:`get_session`), ``app``,
    ``np`` and ``run_script`` predefined, ``__name__ == "__main__"``, ``__file__`` set,
    ``sys.argv == [path, *args]``, and its own folder importable — like ``python script.py``.
    While it runs, :func:`get_session` returns that same session. Pass ``namespace`` to run
    in existing globals instead of fresh ones."""
    path = os.path.abspath(os.fspath(path))
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    code = compile(source, path, "exec")
    session = session if session is not None else get_session()
    globals_ = namespace if namespace is not None else make_namespace(session)
    globals_["__file__"] = path
    previous_session = set_current_session(session)
    old_argv, script_dir = sys.argv, os.path.dirname(path)
    sys.argv = [path, *map(str, args)]
    sys.path.insert(0, script_dir)
    try:
        exec(code, globals_)
    finally:
        sys.argv = old_argv
        if script_dir in sys.path:
            sys.path.remove(script_dir)
        set_current_session(previous_session)
    return globals_
