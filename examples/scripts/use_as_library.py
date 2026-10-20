"""ECM Tracker as a library inside your own Python program (no GUI is started).

Run from a shell, e.g.:   uv run python examples/scripts/use_as_library.py <image folder>
or from another project after installing this one:  pip install -e <path to ecm_tracker>
"""
import sys

import numpy as np

import ecm_tracker as et

folder = sys.argv[1] if len(sys.argv) > 1 else input("Image folder:")

session = et.Session()  # headless; built-in parameter defaults
session.open_directory(folder)
session.set_frame_range(0, session.n_images - 1)
h, w = session.image_size()
session.set_roi_circle(w / 2, h / 2, 0.4 * min(w, h))
session.detect_corners(maxCorners=300)
session.track()
session.cleanup(fb_mean=0.5, drop_left_image=True)

coords = session.coords()  # (frames, points, 2), row 0 = reference frame
valid = session.valid()  # (frames, points) bool
displacement = coords[-1] - coords[0]
print(session.summary())
print("mean displacement (px):", np.round(displacement[valid[-1]].mean(axis=0), 3))
