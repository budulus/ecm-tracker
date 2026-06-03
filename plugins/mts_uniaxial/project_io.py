"""Persist the plugin's project as a folder of individual, inspectable files (no monolithic DB).

Each step writes its artifact(s) immediately, plus a master ``manifest.json`` index, so work
resumes after closing the app. ``load_project`` re-parses the original raw files (their paths are
in the manifest) and layers the saved per-step parameters on top. The core tracking result is
persisted as ``trackers.npz`` (the TRACK artifact, written via the app's tracker IO); the plugin
restores it into the app on resume, so progress resumes through TRACK when that file is present.

Qt-free. The default project folder is ``<root>/mts_uniaxial_project/``.
"""
from __future__ import annotations

import csv
import json
import os
from typing import List, Optional

import numpy as np

from .parsers import parse_image_log, parse_sensor, resolve_image_paths
from .state import MtsProjectState, Step
from .sync import OFFSET_CONVENTION

SCHEMA_VERSION = 1
PROJECT_DIRNAME = "mts_uniaxial_project"
MANIFEST = "manifest.json"


def project_dir(root: str) -> str:
    return os.path.join(root, PROJECT_DIRNAME)


def _ensure(project_path: str) -> None:
    os.makedirs(project_path, exist_ok=True)


def _read_json(project_path: str, name: str) -> Optional[dict]:
    try:
        with open(os.path.join(project_path, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(project_path: str, name: str, data: dict) -> None:
    _ensure(project_path)
    with open(os.path.join(project_path, name), "w") as f:
        json.dump(data, f, indent=2)


def wipe_files(project_path: str, filenames: List[str]) -> None:
    """Delete the given artifact files if present (downstream-invalidation wipe)."""
    for name in filenames:
        path = os.path.join(project_path, name)
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- manifest


def _manifest_dict(state: MtsProjectState) -> dict:
    sensor = state.sensor
    image_log = state.image_log
    return {
        "schema_version": SCHEMA_VERSION,
        "root": state.root,
        "images_dir": state.images_dir,
        "sensor_file": state.sensor_file,
        "log_path": state.log_path,
        "completed_through": state.completed_through,
        "n_images": image_log.n_images if image_log else None,
        "n_samples": sensor.n_samples if sensor else None,
        "clamp_labels": list(sensor.clamp_labels) if sensor else None,
        "units": sensor.units if sensor else None,
        "sensor_warnings": sensor.warnings if sensor else None,
        "n_skipped": sensor.n_skipped if sensor else None,
        "offset_convention": OFFSET_CONVENTION,
        "force_channel": state.force_channel,
        "offset_ms": state.offset_ms,
        "crop_start": state.crop_start,
        "crop_end": state.crop_end,
        "ref_algorithm": state.ref_algorithm,
        "ref_sensor_index": state.ref_sensor_index,
        "ref_image_global": state.ref_image_global,
        "last_image_global": state.last_image_global,
        "zero_disp": state.zero_disp,
        "zero_force": state.zero_force,
        "material_width": state.material_width,
        "material_thickness": state.material_thickness,
        "incompressible": state.incompressible,
        "stages": {},  # reserved for future post-processing stages
    }


def save_manifest(state: MtsProjectState) -> None:
    _write_json(project_dir(state.root), MANIFEST, _manifest_dict(state))


# --------------------------------------------------------------------------- per-step saves


def save_load(state: MtsProjectState) -> None:
    """Write the LOAD snapshots (image log + raw sensor) and the manifest."""
    pdir = project_dir(state.root)
    _ensure(pdir)
    with open(os.path.join(pdir, "image_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "time_ms", "path"])
        for name, t, path in zip(state.image_log.filenames, state.image_log.time_ms, state.ordered_paths):
            w.writerow([name, f"{t:.3f}", path])
    s = state.sensor
    with open(os.path.join(pdir, "sensor_raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "disp_a", "force_a", "disp_b", "force_b"])
        for i in range(s.n_samples):
            w.writerow([s.time_s[i], s.disp_a[i], s.force_a[i], s.disp_b[i], s.force_b[i]])
    save_manifest(state)


def save_channel(state: MtsProjectState) -> None:
    _write_json(project_dir(state.root), "sync_params.json",
                {"force_channel": state.force_channel, "offset_ms": state.offset_ms})
    save_manifest(state)


def save_crop(state: MtsProjectState) -> None:
    _write_json(project_dir(state.root), "crop.json",
                {"crop_start": state.crop_start, "crop_end": state.crop_end})
    save_manifest(state)


def save_reference(state: MtsProjectState) -> None:
    _write_json(project_dir(state.root), "reference.json", {
        "ref_algorithm": state.ref_algorithm,
        "ref_sensor_index": state.ref_sensor_index,
        "ref_image_global": state.ref_image_global,
        "last_image_global": state.last_image_global,
        "zero_disp": state.zero_disp,
        "zero_force": state.zero_force,
    })
    save_manifest(state)


def save_export(state: MtsProjectState, coords: np.ndarray, point_ids: np.ndarray,
                aligned_rows: List[list]) -> List[str]:
    """Write the EXPORT artifacts; return the absolute paths written."""
    pdir = project_dir(state.root)
    _ensure(pdir)
    coords_path = os.path.join(pdir, "tracked_coords.npy")
    ids_path = os.path.join(pdir, "point_indices.npy")
    aligned_path = os.path.join(pdir, "aligned_data.csv")
    np.save(coords_path, coords.astype(np.float32))
    np.save(ids_path, point_ids.astype(np.int64))
    with open(aligned_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_global", "image_time_ms", "displacement", "force", "in_range"])
        for row in aligned_rows:
            g, t, d, force, in_range = row
            w.writerow([int(g), f"{t:.3f}", f"{d:.6g}", f"{force:.6g}", int(in_range)])
    save_manifest(state)
    return [coords_path, ids_path, aligned_path]


def save_measures(state: MtsProjectState, header: List[str], rows: List[list]) -> str:
    """Write the per-frame derived-measures CSV (``measures.csv``); return its absolute path.

    ``rows`` are pre-formatted (the window decides column selection + number formatting). The file
    is owned by the EXPORT step, so it is wiped alongside the aligned data when tracking changes.
    """
    pdir = project_dir(state.root)
    _ensure(pdir)
    path = os.path.join(pdir, "measures.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    save_manifest(state)
    return path


# --------------------------------------------------------------------------- resume


def load_project(root: str) -> Optional[MtsProjectState]:
    """Reconstruct a :class:`MtsProjectState` from ``<root>/mts_uniaxial_project/``.

    Returns ``None`` if there is no project or the original raw files can no longer be parsed.
    Resumable progress reaches TRACK when ``trackers.npz`` is present (the plugin reinstalls that
    saved result into the app on resume); otherwise it caps at REFERENCE.
    """
    pdir = project_dir(root)
    manifest = _read_json(pdir, MANIFEST)
    if manifest is None:
        return None

    images_dir = manifest.get("images_dir")
    sensor_file = manifest.get("sensor_file")
    log_path = manifest.get("log_path")
    try:
        image_log = parse_image_log(log_path)
        sensor = parse_sensor(sensor_file)
        ordered = resolve_image_paths(image_log, images_dir)
    except (OSError, ValueError, TypeError):
        return None

    st = MtsProjectState()
    st.root = manifest.get("root", root)
    st.images_dir, st.sensor_file, st.log_path = images_dir, sensor_file, log_path
    st.image_log, st.sensor, st.ordered_paths = image_log, sensor, ordered
    st.completed_through = int(Step.LOAD)
    stored = int(manifest.get("completed_through", Step.LOAD))

    # Material parameters are not step-gated — restore them unconditionally (old manifests that
    # predate them fall back to the defaults). Guard the coercions so a malformed or explicit-null
    # value degrades to "start fresh" (return None) rather than aborting the whole resume.
    try:
        st.material_width = float(manifest.get("material_width", 10.0))
        st.material_thickness = float(manifest.get("material_thickness", 0.5))
        st.incompressible = bool(manifest.get("incompressible", True))
    except (ValueError, TypeError):
        return None

    sp = _read_json(pdir, "sync_params.json")
    if stored >= Step.CHANNEL and sp:
        st.force_channel = sp.get("force_channel", "average")
        st.offset_ms = float(sp.get("offset_ms", 0.0))
        st.completed_through = int(Step.CHANNEL)

    cp = _read_json(pdir, "crop.json")
    if st.done(Step.CHANNEL) and stored >= Step.CROP and cp:
        cs, ce = cp.get("crop_start"), cp.get("crop_end")
        if cs is not None and ce is not None and 0 <= cs <= ce < sensor.n_samples:
            st.crop_start, st.crop_end = int(cs), int(ce)
            st.completed_through = int(Step.CROP)

    rf = _read_json(pdir, "reference.json")
    if st.done(Step.CROP) and stored >= Step.REFERENCE and rf:
        st.ref_algorithm = rf.get("ref_algorithm")
        st.ref_sensor_index = rf.get("ref_sensor_index")
        st.ref_image_global = rf.get("ref_image_global")
        st.last_image_global = rf.get("last_image_global")
        st.zero_disp = rf.get("zero_disp")
        st.zero_force = rf.get("zero_force")
        if (st.ref_image_global is not None and st.last_image_global is not None
                and 0 <= st.ref_image_global <= st.last_image_global < image_log.n_images):
            st.completed_through = int(Step.REFERENCE)

    # The core tracking result is reinstalled by the plugin (it needs the app handle); here we
    # only record that a saved trackers.npz is available so resume reaches TRACK.
    if st.done(Step.REFERENCE) and stored >= Step.TRACK and os.path.isfile(
        os.path.join(pdir, "trackers.npz")
    ):
        st.completed_through = int(Step.TRACK)

    return st
