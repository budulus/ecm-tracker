"""The plugin's dependency-chain state machine — the data-integrity backbone.

Steps run strictly in order. Editing a step invalidates everything downstream of it: those
fields reset and their on-disk artifacts are deleted, so the user can never reach a half-stale
state. ``invalidate_from(step)`` resets fields for ``step .. EXPORT`` and returns the artifact
filenames to wipe; the window (via ``project_io``) performs the disk deletes in the same op.

Note the ordering: REFERENCE (3) is *upstream* of TRACK (4). Changing the reference re-scopes the
range and clears the ROI, so it invalidates tracking + export — but it is **not** downstream of
the ROI/tracking. Qt-free and disk-free (returns filenames, never touches the filesystem).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from .parsers import ImageLog, Sensor


class Step(IntEnum):
    LOAD = 0       # root + images folder + sensor file resolved & parsed; sequence loaded
    CHANNEL = 1    # force-channel selection + temporal offset
    CROP = 2       # first/last sensor index of the experiment window
    REFERENCE = 3  # reference algorithm + detected reference + zeroing; range set in the core app
    TRACK = 4      # ROI + tracking done in the core app (reference must be set first)
    EXPORT = 5     # tracked-coordinate + aligned-data export


#: Artifact filenames owned by each step (deleted when that step is invalidated). ``manifest.json``
#: is intentionally absent — it is the always-rewritten master index, removed only on a full reset.
STEP_ARTIFACTS = {
    Step.LOAD: ["image_log.csv", "sensor_raw.csv"],
    Step.CHANNEL: ["sync_params.json"],
    Step.CROP: ["crop.json"],
    Step.REFERENCE: ["reference.json"],
    Step.TRACK: [],
    Step.EXPORT: ["tracked_coords.npy", "point_indices.npy", "aligned_data.csv", "measures.csv"],
}

# Defaults for the resettable input fields, applied step-by-step in invalidate_from.
_DEFAULTS = {
    Step.CHANNEL: dict(force_channel="average", offset_ms=0.0),
    Step.CROP: dict(crop_start=None, crop_end=None),
    Step.REFERENCE: dict(
        ref_algorithm=None, ref_sensor_index=None, ref_image_global=None,
        last_image_global=None, zero_disp=None, zero_force=None,
    ),
}


@dataclass
class MtsProjectState:
    # LOAD
    root: Optional[str] = None
    images_dir: Optional[str] = None
    sensor_file: Optional[str] = None
    log_path: Optional[str] = None
    image_log: Optional[ImageLog] = None
    sensor: Optional[Sensor] = None
    ordered_paths: Optional[List[str]] = None  # absolute, in acquisition-log order
    # CHANNEL
    force_channel: str = "average"  # "A" | "B" | "average"
    offset_ms: float = 0.0
    # CROP (sensor-sample indices, inclusive)
    crop_start: Optional[int] = None
    crop_end: Optional[int] = None
    # REFERENCE
    ref_algorithm: Optional[str] = None
    ref_sensor_index: Optional[int] = None   # absolute sensor index
    ref_image_global: Optional[int] = None
    last_image_global: Optional[int] = None
    zero_disp: Optional[float] = None
    zero_force: Optional[float] = None
    # MATERIAL — reference cross-section + constitutive flag. These are *parameters*, not a Step:
    # they never gate the workflow and never enter invalidate_from (editing them must not wipe
    # tracking). They only feed the stress/kinematics plots and the measures CSV, and persist in
    # the manifest. Width and thickness are always in millimetres.
    material_width: float = 10.0
    material_thickness: float = 0.5
    incompressible: bool = True
    # progress: highest fully-completed step, or -1 if nothing is done yet
    completed_through: int = -1

    @property
    def reference_area_mm2(self) -> float:
        """Reference cross-section A₀ = width × thickness (mm²)."""
        return self.material_width * self.material_thickness

    def done(self, step: Step) -> bool:
        """True if ``step`` (and everything before it) is complete."""
        return self.completed_through >= int(step)

    def invalidate_from(self, step: Step) -> List[str]:
        """Reset fields for ``step .. EXPORT`` and lower progress; return artifact filenames to wipe.

        ``step == LOAD`` is a full reset (clears the loaded files too). The caller updates the
        just-edited step's fields *after* this returns, then sets ``completed_through``.
        """
        step = Step(step)
        if step <= Step.LOAD:
            self.root = self.images_dir = self.sensor_file = self.log_path = None
            self.image_log = self.sensor = self.ordered_paths = None
        for s, defaults in _DEFAULTS.items():
            if step <= s:
                for k, v in defaults.items():
                    setattr(self, k, v)
        self.completed_through = min(self.completed_through, int(step) - 1)

        files: List[str] = []
        for s in Step:
            if s >= step:
                files.extend(STEP_ARTIFACTS.get(s, []))
        return files
