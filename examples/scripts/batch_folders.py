"""Batch processing: track every experiment folder below a root folder with one recipe.

Each sub-folder of ROOT that contains images is opened in its own headless Session, tracked
with identical settings, and gets a trackers.npz + coords.csv written next to its images.

Headless:  tracker --run examples/scripts/batch_folders.py <root folder>
In the app: run it from the Script Editor; it asks for the root folder (the GUI project is
            not touched, every experiment gets a fresh session).
"""
import os
import sys

from ecm_tracker import Session, SessionError

root = sys.argv[1] if len(sys.argv) > 1 else input("Root folder with one sub-folder per experiment:")

for name in sorted(os.listdir(root)):
    folder = os.path.join(root, name)
    if not os.path.isdir(folder):
        continue
    s = Session()
    try:
        s.open_directory(folder)
    except (SessionError, OSError, ValueError) as exc:
        print(f"skip {name}: {exc}")
        continue
    h, w = s.image_size()
    s.set_roi_rectangle(0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h)
    s.detect_grid(spacing_x=20, spacing_y=20)
    s.track(win_size=31)
    s.cleanup(fb_max=1.0, drop_left_image=True)
    s.save_trackers(os.path.join(folder, "trackers.npz"))
    if s.exportable_mask().any():
        s.export_csv(os.path.join(folder, "coords.csv"))
    print(f"{name}: {s.n_images} frames, {s.n_active}/{s.n_points} points kept")
