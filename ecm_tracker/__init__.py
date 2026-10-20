"""ECM Tracker as a Python library.

``import ecm_tracker`` gives the same scripting API the application's Python console and Script
Editor use, without starting the GUI (PySide6 is not imported)::

    import ecm_tracker as et

    session = et.Session()                      # headless, built-in parameter defaults
    session.open_directory("data/experiment")
    session.set_frame_range(0, 40)
    session.set_roi_rectangle(60, 50, 240, 180)
    session.detect_corners(maxCorners=300)
    session.track(win_size=31)
    session.cleanup(fb_mean=0.5, drop_left_image=True)
    session.export_npy("data/experiment/coords.npy")

Recorded macros start with ``session = get_session()``: inside the app that is the GUI's
session, in a plain program a new headless one. Run a script file with :func:`run_script`, or
from a shell with ``tracker --run script.py`` / ``python -m ecm_tracker script.py``.
See ``help(ecm_tracker.Session)`` for every command.
"""
from app.core.cleanup import Metrics, Thresholds
from app.core.result import TrackerResult
from app.core.roi import ROI
from app.scripting import *  # noqa: F401,F403 -- the public scripting API
from app.scripting import __all__ as _scripting_all

__all__ = [*_scripting_all, "Metrics", "ROI", "Thresholds", "TrackerResult"]
