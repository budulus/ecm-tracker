"""Generate a synthetic image sequence with known per-frame translation.

A feature visible at frame position (px, py) in frame 0 appears at (px + dx, py + dy) in
frame 1, etc. This gives a ground-truth motion to validate tracking against.
"""
import os

import cv2
import numpy as np


def make_sequence(out_dir, n_frames=12, w=320, h=240, dx=2.0, dy=1.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    margin = int(max(abs(dx), abs(dy)) * n_frames) + 20
    bw, bh = w + 2 * margin, h + 2 * margin
    base = np.full((bh, bw, 3), 40, np.uint8)
    for _ in range(600):
        x = int(rng.integers(0, bw))
        y = int(rng.integers(0, bh))
        sz = int(rng.integers(4, 20))
        color = int(rng.integers(60, 255))
        cv2.rectangle(base, (x, y), (x + sz, y + sz), (color, color, color), -1)

    paths = []
    for t in range(n_frames):
        m = np.float32([[1, 0, margin + dx * t], [0, 1, margin + dy * t]])
        frame = cv2.warpAffine(base, m, (w, h))
        path = os.path.join(out_dir, f"img_{t}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tracker_synth"
    p = make_sequence(out)
    print(f"wrote {len(p)} frames to {out}")
