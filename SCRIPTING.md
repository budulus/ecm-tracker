# Scripting ECM Tracker with Python

Everything you can do with the toolbar and menus is also a Python command on one object, the
**session**. The same API is used in four ways:

| Where | How | `session` is |
|---|---|---|
| **Python Console** (dock under the image) | *Scripting → Console*, `Ctrl+Shift+P` | the open project — commands update the GUI |
| **Script Editor** (tabbed `.py` editor) | *Scripting → Editor*, `Ctrl+Shift+E`; `F5` run, `F9` run selection, `Shift+F5` stop | the open project |
| **Headless** (no window) | `tracker --run script.py [args]` or `python -m ecm_tracker script.py` | a fresh session |
| **Your own program** | `import ecm_tracker as et; s = et.Session()` | whatever you create |

`tracker --script script.py` starts the GUI and runs the script inside it (output goes to the
console).

## Recording macros

Press **Record** (toolbar *SCRIPT* group, `Ctrl+Shift+R`). The Script Editor opens a live
*Macro* tab and the status bar shows **● REC**. Every action that changes the project (from a
button, a dialog, the console, a script or a plugin) is appended as one line of Python. Press
**Stop Rec** to turn the tab into an ordinary script you can edit, run (`F5`) and save.

```python
# ECM Tracker macro - recorded 2026-10-20 14:03
from ecm_tracker import get_session

session = get_session()

session.open_directory(r"C:\data\experiment")
session.set_frame_range(0, 40)
session.set_roi_rectangle(60.12, 50.46, 240.0, 180.0)
session.detect_corners(maxCorners=500, qualityLevel=0.01, minDistance=7.0, blockSize=7, useHarrisDetector=False, k=0.04)
session.track(win_size=21, max_level=3, max_iter=30, epsilon=0.01, flags=0, min_eig_threshold=0.0001)
session.cleanup(fb_mean=0.5, drop_left_image=True)
session.remove_points([17, 42])
session.export_npy(r"C:\data\experiment\coords.npy")
```

Things to know about recording:

* **Replays are exact.** Detection and tracking are recorded with *every* parameter, so a
  macro doesn't depend on the saved defaults of the computer that replays it. Mouse-drawn ROIs
  and seed points are rounded to 0.01 px *before* they are applied, so the script holds exactly
  the values the GUI used.
* Dragging a frame-range slider records one line (the final range), not one line per step.
* Navigation (current frame, zoom, pan) and marker display settings are not recorded. They
  don't change results.
* A recorded macro starts with `session = get_session()`. Inside the app that's the open
  project, and in a plain Python program it's a new headless session. So the same file runs in
  the editor, with `tracker --run`, and with plain `python`.
* Opening a plugin is recorded as `app.launch_plugin("...")`. `app` only exists in the GUI. A
  headless run stops there with a clear error, so delete those lines from batch scripts.

## The API at a glance

`help(session)` in the console, or *Script Editor → Help → Session API Reference*, lists every
command with its documentation.

```python
# load
session.open_directory(folder)            # natural filename order
session.open_files([...])                 # natural filename order
session.load_sequence(paths, source_dir)  # order kept verbatim
session.open_image_pair(ref, dst, translation=(dx, dy), ...)
session.open_pair_setup(path) / session.save_pair_setup(path)

# frames (global indices: 0 = first loaded image)
session.set_frame_range(reference, last)  # moving the reference clears ROI + seeds
session.set_current_frame(i)

# region of interest
session.set_roi([(x, y), ...]) / session.set_roi_rectangle(x0, y0, x1, y1)
session.set_roi_circle(cx, cy, radius) / session.clear_roi()

# seeds (keyword arguments update the stored parameters first)
session.detect_corners(maxCorners=300, qualityLevel=0.01, minDistance=7.0)
session.detect_grid(spacing_x=20, spacing_y=20)
session.set_seeds(points) / add_seed(x, y) / add_seeds(points) / remove_seeds(indices) / clear_seeds()

# tracking
session.track(win_size=31, max_level=3)   # forward + backward pyramidal LK
session.clear_tracking()

# cleanup (each keyword enables one filter; undo with undo_cleanup())
session.cleanup(fb_mean=0.5, fb_max=2.0, fw_failures=0, distance=15, drop_left_image=True, drop_left_roi=True)
session.cleanup_mask(...)                 # preview the keep-mask without applying it
session.remove_points(indices) / session.undo_cleanup()

# results
session.coords()          # (frames, points, 2), row 0 = reference frame
session.valid()           # (frames, points) bool
session.point_ids()       # column of each point in the full result
session.metrics()         # per-point FB error, failure counts, max step, ...
session.exportable_mask() # active points valid on every frame (what exports write)
session.export_npy(path) / session.export_csv(path)
session.save_trackers(path) / session.load_trackers(path)

# parameters
session.corner_params / grid_params / lk_params           # read (copies)
session.set_corner_params(...) / set_grid_params(...) / set_lk_params(...)
```

Rules the kernel enforces (the same ones the GUI enforces):

* Changing an input to tracking (range, ROI, seeds) **discards** the tracking result. The GUI
  asks first; a script doesn't.
* Invalid values raise errors instead of being clamped. For example, an unknown parameter
  raises `TypeError`, an out-of-range value raises `ValueError`, and running a step too early
  raises `SessionError` (e.g. `track()` without seeds).
* `cleanup()` and `remove_points()` only ever *remove* points from the active set. Each one can
  be undone with `undo_cleanup()`.
* Long commands report progress (a cancellable dialog in the GUI, a one-line counter
  headless). Cancelling raises `OperationCancelled`.

## GUI-only helpers: `app`

Inside the app, `app` gives scripts the window: `app.launch_plugin(id)`, `app.plugin(id)`
(the running plugin instance), `app.plugins()`, `app.fit_view()`, `app.zoom_in()`,
`app.save_view("view.png")` (the canvas with its overlays), `app.message(text)`,
`app.process_events()`, plus `app.window` / `app.canvas` for advanced use.

## Running long scripts in the GUI

Scripts run on the GUI thread, so the window repaints and the **Stop** button / `Ctrl+C`
work. While a script runs, the main window's controls are locked so a click can't change the
project under the script. `print()` output goes to the editor's output pane (or the console),
and `input()` opens a small dialog.

## Using it as a library

```python
import ecm_tracker as et

session = et.Session()                    # headless, built-in parameter defaults
session.open_directory("data/experiment")
session.set_roi_circle(240, 180, 150)
session.detect_corners(maxCorners=300)
session.track()
session.cleanup(fb_mean=0.5, drop_left_image=True)
xy = session.coords()
```

`import ecm_tracker` never imports Qt, so it works on servers and in notebooks. From another
project, install this one (`uv add --editable <path>` or `pip install -e <path>`). Listen to
state changes with `session.add_listener(callback)`. Record programmatically with
`rec = session.start_recording()` … `session.stop_recording().script()`.

Ready-made scripts are in `examples/scripts/`; in the editor they're under *File → Open Example*:
`basic_workflow.py`, `batch_folders.py` (one recipe over many experiment folders),
`parameter_sweep.py` (pick the LK window by FB error) and `use_as_library.py`.
