"""Headless script runner: ``tracker --run script.py [args...]`` (also ``python -m ecm_tracker``).

Runs a script against a fresh headless :class:`~app.scripting.session.Session` without creating
any window (PySide6 is never imported). Long commands print a one-line progress indicator to
stderr. Exit status is 0 on success, 1 if the script raised, 2 for usage errors.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager

USAGE = "usage: tracker --run SCRIPT.py [ARGS...]   (headless; `session` is predefined)"


@contextmanager
def text_progress(label: str, total: int, delay_ms: int = 0):
    """Progress handler for headless runs: a single updating stderr line."""
    stream = sys.stderr
    interactive = stream is not None and hasattr(stream, "isatty") and stream.isatty()
    shown = [False]

    def callback(done, total_):
        if interactive:
            stream.write(f"\r{label} {done}/{total_}")
            stream.flush()
            shown[0] = True
        return False

    try:
        yield callback
    finally:
        if shown[0]:
            stream.write("\n")
            stream.flush()


def _attach_windows_console() -> None:
    """A frozen GUI-subsystem exe starts without stdout; reuse the launching console if any."""
    if sys.platform != "win32" or sys.stdout is not None:
        return
    import ctypes

    if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
        sys.stdout = open("CONOUT$", "w", encoding="utf-8")
        sys.stderr = open("CONOUT$", "w", encoding="utf-8")


def main(argv) -> int:
    _attach_windows_console()
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, file=sys.stderr if sys.stderr else sys.stdout)
        return 0 if argv else 2
    from app.scripting.runner import get_session, run_script

    session = get_session()
    session.progress_handler = text_progress
    try:
        run_script(argv[0], session=session, args=argv[1:])
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
