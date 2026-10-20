"""Basic workflow: ROI -> corner seeds -> track -> cleanup -> export.

In the app:  open an image folder, then run this script in the Script Editor (F5).
Headless:    tracker --run examples/scripts/basic_workflow.py <image folder>
"""
import os
import sys

from ecm_tracker import get_session

session = get_session()  # the app's open project, or a new headless session
if len(sys.argv) > 1:
    session.open_directory(sys.argv[1])
if not session.has_sequence:
    raise SystemExit("Open an image folder first (or pass one on the command line).")

# A centred ROI covering 70 % of the frame.
h, w = session.image_size()
session.set_roi_rectangle(0.15 * w, 0.15 * h, 0.85 * w, 0.85 * h)

n = session.detect_corners(maxCorners=400, qualityLevel=0.01, minDistance=7.0)
print(f"{n} seed points")

session.track(win_size=21, max_level=3)
removed = session.cleanup(fb_mean=0.5, fw_failures=0, drop_left_image=True)
print(f"cleanup removed {removed} points, {session.n_active} remain")

coords_path, _, shape = session.export_npy(os.path.join(session.source_dir, "coords.npy"))
print(f"wrote {coords_path}: {shape[1]} points x {shape[0]} frames")
