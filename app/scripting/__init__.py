"""Python scripting API of the ECM Tracker (Qt-free; also importable as ``ecm_tracker``).

Everything the GUI can do is a method of :class:`Session`, so a workflow can be scripted in the
built-in console / Script Editor, recorded as a macro, run headless with
``tracker --run script.py``, or embedded in any Python program::

    from ecm_tracker import Session

    session = Session()
    session.open_directory("data/experiment")
    session.set_roi_rectangle(60, 50, 240, 180)
    session.detect_corners()
    session.track()
    session.export_npy("data/experiment/coords.npy")
"""
from app.scripting.recorder import Recorder, format_call, format_value
from app.scripting.runner import (
    HeadlessApp,
    get_app,
    get_session,
    in_gui,
    make_namespace,
    run_script,
    set_current_session,
)
from app.scripting.session import (
    OperationCancelled,
    Session,
    SessionError,
    StaleResultError,
)

__all__ = [
    "HeadlessApp",
    "OperationCancelled",
    "Recorder",
    "Session",
    "SessionError",
    "StaleResultError",
    "format_call",
    "format_value",
    "get_app",
    "get_session",
    "in_gui",
    "make_namespace",
    "run_script",
    "set_current_session",
]
