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
import hashlib
import uuid
from typing import List, Optional

import numpy as np

from app.plugins.analysis import atomic_open, atomic_save_npy

from .parsers import parse_image_log, parse_sensor, resolve_image_paths
from .state import MtsProjectState, Step
from .sync import FORCE_CHANNELS, OFFSET_CONVENTION

SCHEMA_VERSION = 2
PROJECT_DIRNAME = "mts_uniaxial_project"
MANIFEST = "manifest.json"


def _reject_nonfinite(token):
    raise ValueError(f"Non-finite JSON number {token!r} is not supported")


def project_dir(root: str) -> str:
    return os.path.join(root, PROJECT_DIRNAME)


def _ensure(project_path: str) -> None:
    os.makedirs(project_path, exist_ok=True)


def _read_json(project_path: str, name: str) -> Optional[dict]:
    try:
        with open(os.path.join(project_path, name), encoding="utf-8") as f:
            return json.load(f, parse_constant=_reject_nonfinite)
    except (OSError, ValueError):
        return None


def _write_json(project_path: str, name: str, data: dict) -> None:
    _ensure(project_path)
    with atomic_open(os.path.join(project_path, name)) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def wipe_files(project_path: str, filenames: List[str]) -> List[str]:
    """Delete artifacts and return human-readable failures; missing files are successful no-ops."""
    failures = []
    for name in filenames:
        path = os.path.join(project_path, name)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            failures.append(f"{name}: {exc}")
    return failures


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
        "sensor_skipped_rows": sensor.skipped_rows if sensor else [],
        "image_log_skipped_rows": image_log.skipped_rows if image_log else [],
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
        "raw_fingerprints": state.raw_fingerprints,
        "relative_paths": {k: os.path.relpath(getattr(state, k), state.root) for k in ("images_dir", "sensor_file", "log_path") if getattr(state, k)},
        "export_generation": state.export_generation,
        "reference_parameters": state.reference_parameters,
        "force_basis": state.force_basis,
        "loading_axis_deg": state.loading_axis_deg,
        "stages": {},  # reserved for future post-processing stages
    }


def save_manifest(state: MtsProjectState) -> None:
    _write_json(project_dir(state.root), MANIFEST, _manifest_dict(state))


# --------------------------------------------------------------------------- per-step saves


def save_load(state: MtsProjectState) -> None:
    """Write the LOAD snapshots (image log + raw sensor) and the manifest."""
    state.raw_fingerprints = {k: _fingerprint(getattr(state, k)) for k in ("sensor_file", "log_path")}
    pdir = project_dir(state.root)
    _ensure(pdir)
    with atomic_open(os.path.join(pdir, "image_log.csv"), newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "time_ms", "path"])
        for name, t, path in zip(state.image_log.filenames, state.image_log.time_ms, state.ordered_paths):
            w.writerow([name, f"{t:.3f}", path])
    s = state.sensor
    with atomic_open(os.path.join(pdir, "sensor_raw.csv"), newline="") as f:
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
    generation = "export-" + uuid.uuid4().hex
    pdir = os.path.join(pdir, generation)
    _ensure(pdir)
    coords_path = os.path.join(pdir, "tracked_coords.npy")
    ids_path = os.path.join(pdir, "point_indices.npy")
    aligned_path = os.path.join(pdir, "aligned_data.csv")
    previous_step = state.completed_through
    previous_generation = state.export_generation
    try:
        atomic_save_npy(coords_path, coords.astype(np.float32))
        atomic_save_npy(ids_path, point_ids.astype(np.int64))
        with atomic_open(aligned_path, newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_global", "image_time_ms", "displacement", "force", "in_range"])
            for row in aligned_rows:
                g, t, d, force, in_range = row
                w.writerow([int(g), f"{t:.3f}", f"{d:.6g}", f"{force:.6g}", int(in_range)])
        state.completed_through = int(Step.EXPORT)
        state.export_generation = generation
        save_manifest(state)
    except Exception:
        state.completed_through = previous_step
        state.export_generation = previous_generation
        raise
    return [coords_path, ids_path, aligned_path]


def save_measures(state: MtsProjectState, header: List[str], rows: List[list]) -> str:
    """Write the per-frame derived-measures CSV (``measures.csv``); return its absolute path.

    ``rows`` are pre-formatted (the window decides column selection + number formatting). The file
    is owned by the EXPORT step, so it is wiped alongside the aligned data when tracking changes.
    """
    pdir = project_dir(state.root)
    _ensure(pdir)
    path = os.path.join(pdir, "measures.csv")
    with atomic_open(path, newline="") as f:
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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in (1, SCHEMA_VERSION):
        return None

    def source_path(key):
        relative = manifest.get("relative_paths", {}).get(key)
        if relative is not None:
            return os.path.normpath(os.path.join(root, relative))
        old = manifest.get(key)
        old_root = manifest.get("root")
        if old and old_root:
            relocated = os.path.normpath(os.path.join(root, os.path.relpath(old, old_root)))
            if os.path.exists(relocated):
                return relocated
        return old
    images_dir, sensor_file, log_path = (source_path(k) for k in ("images_dir", "sensor_file", "log_path"))
    fingerprints = manifest.get("raw_fingerprints", {})
    if not isinstance(fingerprints, dict):
        return None
    try:
        if fingerprints and any(_fingerprint(path) != fingerprints.get(key)
                                for key, path in (("sensor_file", sensor_file), ("log_path", log_path))):
            return None  # Never combine changed raw measurements with saved scientific state.
    except (OSError, TypeError):
        return None
    try:
        image_log = parse_image_log(log_path)
        sensor = parse_sensor(sensor_file)
        ordered = resolve_image_paths(image_log, images_dir)
    except (OSError, ValueError, TypeError):
        return None

    st = MtsProjectState()
    # The selected folder is authoritative. A copied/moved project must never keep writing to the
    # old absolute root recorded in its manifest.
    st.root = root
    st.raw_fingerprints = fingerprints
    st.reference_parameters = manifest.get("reference_parameters", {})
    if not isinstance(st.reference_parameters, dict):
        return None
    try:
        for key in ("search_start", "search_end"):
            if key in st.reference_parameters and type(st.reference_parameters[key]) is not int:
                return None
        if not np.isfinite(float(st.reference_parameters.get("preforce", 0))):
            return None
    except (TypeError, ValueError):
        return None
    st.force_basis = manifest.get("force_basis", "reference_change")
    st.loading_axis_deg = manifest.get("loading_axis_deg")
    if st.force_basis not in ("reference_change", "absolute"):
        return None
    try:
        if st.loading_axis_deg is not None:
            st.loading_axis_deg = float(st.loading_axis_deg)
            if not np.isfinite(st.loading_axis_deg):
                return None
    except (ValueError, TypeError):
        return None
    st.images_dir, st.sensor_file, st.log_path = images_dir, sensor_file, log_path
    st.image_log, st.sensor, st.ordered_paths = image_log, sensor, ordered
    st.completed_through = int(Step.LOAD)
    try:
        stored = int(manifest.get("completed_through", Step.LOAD))
    except (TypeError, ValueError):
        return None
    stored = max(int(Step.LOAD), min(stored, int(Step.EXPORT)))
    if not fingerprints:
        sensor.warnings.append("Legacy project has no raw-data provenance: reselect reference and rerun tracking.")
        stored = min(stored, int(Step.CROP))  # Legacy provenance cannot validate saved reference/tracks.

    # Material parameters are not step-gated — restore them unconditionally (old manifests that
    # predate them fall back to the defaults). Guard the coercions so a malformed or explicit-null
    # value degrades to "start fresh" (return None) rather than aborting the whole resume.
    try:
        st.material_width = float(manifest.get("material_width", 10.0))
        st.material_thickness = float(manifest.get("material_thickness", 0.5))
        incompressible = manifest.get("incompressible", True)
        if (
            not np.isfinite(st.material_width)
            or not np.isfinite(st.material_thickness)
            or st.material_width <= 0
            or st.material_thickness <= 0
            or not isinstance(incompressible, bool)
        ):
            return None
        st.incompressible = incompressible
    except (ValueError, TypeError):
        return None

    sp = manifest if manifest.get("schema_version") == 2 else _read_json(pdir, "sync_params.json")
    if stored >= Step.CHANNEL and isinstance(sp, dict):
        try:
            channel = sp.get("force_channel", "average")
            offset = float(sp.get("offset_ms", 0.0))
        except (TypeError, ValueError):
            channel, offset = None, np.nan
        if channel in FORCE_CHANNELS and np.isfinite(offset):
            st.force_channel = channel
            st.offset_ms = offset
            st.completed_through = int(Step.CHANNEL)

    cp = manifest if manifest.get("schema_version") == 2 else _read_json(pdir, "crop.json")
    if st.done(Step.CHANNEL) and stored >= Step.CROP and isinstance(cp, dict):
        try:
            cs, ce = int(cp["crop_start"]), int(cp["crop_end"])
        except (KeyError, TypeError, ValueError):
            cs = ce = -1
        if 0 <= cs <= ce < sensor.n_samples:
            st.crop_start, st.crop_end = cs, ce
            st.completed_through = int(Step.CROP)

    rf = manifest if manifest.get("schema_version") == 2 else _read_json(pdir, "reference.json")
    if st.done(Step.CROP) and stored >= Step.REFERENCE and isinstance(rf, dict):
        try:
            ref_image = int(rf["ref_image_global"])
            last_image = int(rf["last_image_global"])
            sensor_index = rf.get("ref_sensor_index")
            sensor_index = int(sensor_index) if sensor_index is not None else None
            zero_disp = float(rf["zero_disp"])
            zero_force = float(rf["zero_force"])
        except (KeyError, TypeError, ValueError):
            ref_image = last_image = -1
            sensor_index = None
            zero_disp = zero_force = np.nan
        valid_sensor = sensor_index is None or 0 <= sensor_index < sensor.n_samples
        if (
            0 <= ref_image <= last_image < image_log.n_images
            and valid_sensor
            and np.isfinite(zero_disp)
            and np.isfinite(zero_force)
        ):
            st.ref_algorithm = rf.get("ref_algorithm")
            st.ref_sensor_index = sensor_index
            st.ref_image_global = ref_image
            st.last_image_global = last_image
            st.zero_disp = zero_disp
            st.zero_force = zero_force
            st.completed_through = int(Step.REFERENCE)

    # The core tracking result is reinstalled by the plugin (it needs the app handle); here we
    # only record that a saved trackers.npz is available so resume reaches TRACK.
    if st.done(Step.REFERENCE) and stored >= Step.TRACK and os.path.isfile(
        os.path.join(pdir, "trackers.npz")
    ):
        st.completed_through = int(Step.TRACK)
        st.export_generation = manifest.get("export_generation")
        if st.export_generation:
            try:
                paths = export_paths(st)
                if stored >= Step.EXPORT and all(os.path.isfile(path) for path in paths):
                    st.completed_through = int(Step.EXPORT)
                else:
                    st.export_generation = None
            except (ValueError, TypeError, AttributeError):
                st.export_generation = None

    return st

def _fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_paths(state):
    """Resolve the authoritative committed export generation, never partial files."""
    if not state.export_generation or not state.export_generation.startswith("export-") or "/" in state.export_generation or "\\" in state.export_generation:
        raise ValueError("No valid committed export generation")
    base = os.path.join(project_dir(state.root), state.export_generation)
    return [os.path.join(base, name) for name in ("tracked_coords.npy", "point_indices.npy", "aligned_data.csv")]
