"""Parameter sweep: compare the LK window size by its forward-backward (FB) error.

Needs a sequence with seed points (open a folder, define an ROI, detect corners), then run
this in the Script Editor. The best window size is applied at the end.
"""
import numpy as np

from ecm_tracker import get_session

session = get_session()
if session.n_seeds == 0:
    raise SystemExit("Detect or add seed points first.")

best = None
for win in (11, 15, 21, 31, 41):
    session.track(win_size=win)
    fb = session.metrics().fb_mean
    lost = int((~np.isfinite(fb)).sum())
    median = float(np.median(fb[np.isfinite(fb)])) if lost < len(fb) else float("inf")
    print(f"win_size {win:3d}: median FB error {median:.4f} px, lost {lost}/{len(fb)} points")
    if best is None or (lost, median) < best[1:]:
        best = (win, lost, median)

print(f"best: win_size={best[0]}")
session.track(win_size=best[0])
