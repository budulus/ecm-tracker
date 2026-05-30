"""Generate the tracking parity fixture consumed by `crates/core/tests/tracking_parity.rs`.

Writes synthetic frames plus the *Python* tracker's output (the reference the Rust port
must reproduce). Run from the repo root with the app's environment, e.g.:

    uv run python rust/crates/core/tests/fixtures/gen_fixture.py

Both implementations call the same OpenCV `calcOpticalFlowPyrLK` on identical uint8 frames
with identical seed points and LK params, so their coordinates should agree to sub-pixel
tolerance. Seeds are dumped (not re-detected in Rust) so this isolates *tracking* parity.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))  # -> tracker/
sys.path.insert(0, ROOT)

from app.core.image_sequence import ImageSequence  # noqa: E402
from app.core.feature_detection import shi_tomasi, DEFAULT_SHI_TOMASI  # noqa: E402
from app.core.roi import ROI  # noqa: E402
from app.core.tracking import track, DEFAULT_LK  # noqa: E402
from app.core.cleanup import compute_metrics, build_mask, Thresholds, BandFilter, BAND_CAP  # noqa: E402
from tests.synthetic import make_sequence  # noqa: E402

FRAMES_DIR = os.path.join(HERE, "frames")
REF_INDEX, LAST_INDEX = 0, 11

# 1. Synthetic sequence with known translation (dx=2, dy=1 per frame).
paths = make_sequence(FRAMES_DIR, n_frames=12, w=320, h=240, dx=2.0, dy=1.0, seed=0)
seq = ImageSequence(paths)

# 2. Seed Shi-Tomasi corners on the reference frame inside an inset ROI (keeps points
#    away from the edges so the translation doesn't push them out of frame).
roi = ROI([(40, 40), (280, 40), (280, 200), (40, 200)])
gray0 = seq.load_gray(REF_INDEX)
seeds = shi_tomasi(gray0, roi.mask(240, 320), dict(DEFAULT_SHI_TOMASI)).astype(np.float32)

# 3. Run the reference (Python) tracker.
res = track(seq, REF_INDEX, LAST_INDEX, seeds, dict(DEFAULT_LK))
assert res is not None

# 4. Dump everything the Rust test needs.
np.save(os.path.join(HERE, "seeds.npy"), seeds)
np.save(os.path.join(HERE, "coords_fw.npy"), res.coords_fw)
np.save(os.path.join(HERE, "coords_bw.npy"), res.coords_bw)
np.save(os.path.join(HERE, "status_fw.npy"), res.status_fw)
np.save(os.path.join(HERE, "status_bw.npy"), res.status_bw)
np.save(os.path.join(HERE, "fb_mean.npy"), res.fb_mean_error)
np.save(os.path.join(HERE, "fb_max.npy"), res.fb_max_error)

# 5. Cleanup metrics + a representative build_mask, for cleanup_parity.rs.
IMAGE_SIZE = (240, 320)  # (height, width) — matches make_sequence(w=320, h=240)
metrics = compute_metrics(res, roi, IMAGE_SIZE)
np.save(os.path.join(HERE, "m_fail_fw.npy"), metrics.fail_count_fw.astype(np.int64))
np.save(os.path.join(HERE, "m_fail_bw.npy"), metrics.fail_count_bw.astype(np.int64))
np.save(os.path.join(HERE, "m_max_err.npy"), metrics.max_err_fw)
np.save(os.path.join(HERE, "m_mean_err.npy"), metrics.mean_err_fw)
np.save(os.path.join(HERE, "m_max_step.npy"), metrics.max_step)
np.save(os.path.join(HERE, "m_left_image.npy"), metrics.left_image.astype(np.uint8))
np.save(os.path.join(HERE, "m_left_roi.npy"), metrics.left_roi.astype(np.uint8))

thr = Thresholds()
thr.fb_mean = BandFilter(enabled=True, hi=0.5, cap=BAND_CAP)
thr.distance = BandFilter(enabled=True, hi=5.0, cap=BAND_CAP)
thr.drop_left_image = True
np.save(os.path.join(HERE, "mask_demo.npy"), build_mask(metrics, thr).astype(np.uint8))

meta = {
    "reference_index": REF_INDEX,
    "last_index": LAST_INDEX,
    "n_frames": int(res.n_frames),
    "n_points": int(res.n_points),
    "lk": dict(DEFAULT_LK),
    "translation_per_frame": [2.0, 1.0],
}
with open(os.path.join(HERE, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"fixture: {res.n_frames} frames, {res.n_points} points -> {HERE}")
