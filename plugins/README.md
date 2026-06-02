# Writing ECM Tracker plugins

A plugin is a self-contained folder under `plugins/` that adds a window to the **Plugins**
menu. Plugins are for **post-processing, export, and visualization** — they read the tracked
data, draw on the canvas, capture mouse input, and (at most) apply a safe point filter. They
never touch the core tracking code.

**You only need to learn one object:** the `PluginContext`, handed to your plugin as
`self.ctx`. Its full reference (every method, with docstrings) lives in
[`app/plugins/api.py`](../app/plugins/api.py) — skim that file and you know the whole API.

---

## Install / run

Drop a folder into `plugins/` and restart the app (or **Plugins → Reload Plugins**). The app
scans `plugins/*/`, imports each as a Python package, and lists the ones that expose a plugin.
Already part of this repo as worked examples:

- `plugins/affine_zones/` — draw zones, fit a RANSAC affine per zone (mouse capture + overlay + compute + `apply_keep_mask`)
- `plugins/mts_uniaxial/` — sync MTS force/displacement sensor data with the image sequence (data access + image loading + frame-range control + export)

Copy the one closest to what you want and edit it.

---

## Minimal plugin

```
plugins/my_plugin/
└── __init__.py
```

```python
# plugins/my_plugin/__init__.py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from app.plugins import TrackerPlugin

class MyPlugin(TrackerPlugin):
    NAME = "My Plugin"                 # shown in the Plugins menu
    DESCRIPTION = "Says how many points are tracked."

    def launch(self):                  # called when the menu item is clicked
        w = QLabel(f"{self.ctx.n_active} points tracked", parent=self.ctx.window)
        w.setWindowFlags(Qt.Window)
        w.setWindowTitle(self.NAME)
        w.show()
        return w                       # return the window so it stays alive

PLUGIN = MyPlugin                      # how the app finds your plugin
```

That's a working plugin. Split into multiple modules (`__init__.py` + `mywindow.py`) once it
grows — the examples do this. Need an extra dependency? Add it to `pyproject.toml` and run
`uv sync` (plugins share the app's environment).

---

## The three capabilities

### 1. Read the tracked data

```python
ctx.has_result                       # is there a tracking result?
coords = ctx.coords()                # (frames, points, 2) float32, kept points, cut-indexed
ids    = ctx.point_indices()         # original ids of those points
img    = ctx.frame_bgr(ctx.current_index)   # current frame as an (H,W,3) BGR array
ctx.roi, ctx.roi_contains(x, y), ctx.roi_mask()
ctx.metrics()                        # per-point FB error, failures, max step, …
ctx.reference_index, ctx.last_index, ctx.current_index
```

`coords` is indexed by **cut** index on axis 0 — `coords[0]` is the reference frame. Use
`ctx.cut_to_global` / `ctx.global_to_cut` to relate it to folder positions. Treat everything you
read as **read-only**.

### 2. Draw an overlay on the canvas

```python
def paint(painter, ctx):             # called every repaint, in SCREEN space
    pt = ctx.image_to_screen(x, y)   # convert image coords → screen
    painter.drawEllipse(pt, 3, 3)

ctx.add_overlay(paint)               # register
ctx.request_redraw()                 # ask the canvas to repaint
...
ctx.remove_overlay(paint)            # in your window's closeEvent
```

Overlays paint in screen space, so marker sizes stay constant under zoom — convert every image
coordinate with `ctx.image_to_screen`. Connect to `ctx.signals` to repaint reactively (see below).

### 3. Capture the mouse

```python
from app.plugins import CanvasInteraction

class PickTool(CanvasInteraction):
    def on_press(self, image_pt, event):   # image coordinates
        print(image_pt.x(), image_pt.y())

ctx.begin_canvas_interaction(PickTool())   # left-clicks now come to you
...
ctx.end_canvas_interaction()               # restore normal behavior (also do this on close)
```

While an interaction is active the ROI/Pan tools are switched off; middle/right-drag still pans.

---

## Reacting to changes — `ctx.signals`

Don't poll. Connect to the signal hub and refresh when something changes:

```python
ctx.signals.frame_changed.connect(self._redraw)   # int = new global frame index
ctx.signals.result_changed.connect(self._reload)  # tracking produced/discarded
ctx.signals.mask_changed.connect(self._reload)     # cleanup / filter applied
ctx.signals.roi_changed.connect(self._reload)
ctx.signals.sequence_changed.connect(self._reload)
```

## Persisting settings

```python
ctx.save_settings({"format": "csv"})   # stored under a key unique to your plugin
ctx.get_settings()                       # -> {"format": "csv"} (or {} if none)
```

## Filtering points (the only state you may change)

```python
ctx.apply_keep_mask(keep_bool_array)   # length P or n_active; undoable in the Cleanup dialog
```

This only ever *removes* points from the active set (it can't resurrect filtered ones) and is
recorded on the cleanup undo stack.

---

## Conventions & gotchas

- **Indices are global** in the API (e.g. `frame_bgr(i)`), but **`coords` is cut-indexed**
  (`coords[0]` = reference). Convert with `ctx.cut_to_global` / `ctx.global_to_cut`.
- Parent your windows to `ctx.window` and set the `Qt.Window` flag so they float independently.
- **Always undo your canvas hooks** in `closeEvent`: `remove_overlay` and
  `end_canvas_interaction`. The example plugins show the pattern.
- A crash in an overlay just removes that overlay; a crash in `launch()` shows a dialog. Neither
  takes down the app — but check the message and fix it.
- Keep overlay painting cheap (it runs on the UI thread each repaint); cache heavy computation
  and recompute only on the relevant signal.
