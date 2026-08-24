"""Save/load a complete tracker session to a single self-contained ``.npz`` file.

A saved tracker bundles the reference seed points and (if tracking has been run) the full
``TrackerResult`` arrays plus the active cleanup mask, so a session can be reloaded and
overlaid back onto the same image sequence. Numeric arrays go into the ``.npz`` directly;
all scalar/structured metadata (indices, ROI corners, parameter dicts) is JSON-encoded
into a single ``meta`` entry stored as a unicode array — so loading never needs
``allow_pickle``.

Kept Qt-free and free of ``app.models`` (dependency rule gui -> models -> core): callers
pass and receive plain primitives, exactly like :mod:`app.core.export`. Reconstructing a
``TrackerResult`` from the returned arrays is the GUI layer's job.
"""
import json
import os
from zipfile import BadZipFile

import numpy as np

from app.core.atomic_io import atomic_open

FORMAT = "ecmtracker-trackers"
VERSION = 2
SUPPORTED_VERSIONS = {1, VERSION}

# The eight TrackerResult arrays, with their on-disk dtypes (mirrors tracker_result.py).
RESULT_ARRAY_DTYPES = {
    "coords_fw": np.float32,
    "status_fw": np.uint8,
    "err_fw": np.float32,
    "coords_bw": np.float32,
    "status_bw": np.uint8,
    "err_bw": np.float32,
    "fb_mean_error": np.float32,
    "fb_max_error": np.float32,
}


def _reject_nonfinite(token):
    raise ValueError(f"Non-finite JSON number {token!r} is not supported")


def save_trackers(
    path,
    *,
    total_images,
    reference_index,
    last_index,
    current_index,
    features,
    roi_corners,
    active_mask,
    result_arrays,
    win_size,
    lk_params,
    shi_tomasi_params,
    grid_params,
    sequence_fingerprint=None,
):
    """Write a tracker session to ``path`` (a single ``.npz``); return the path written.

    ``features`` is the ``(P, 2)`` reference-frame seed array (required). ``result_arrays``
    is a dict of the eight :class:`TrackerResult` arrays, or ``None`` when tracking hasn't
    been run; ``active_mask`` is the ``(P,)`` keep mask, or ``None`` in that case.
    ``roi_corners`` is a list of ``(x, y)`` tuples or ``None``.
    ``sequence_fingerprint`` is the validated ordered image-content identity and is required for
    v2 files, preventing same-length sessions from being applied to a different experiment.
    """
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path += ".npz"

    features = np.asarray(features, dtype=np.float32).reshape(-1, 2)
    has_result = result_arrays is not None
    if not (
        isinstance(sequence_fingerprint, str)
        and len(sequence_fingerprint) == 64
        and all(c in "0123456789abcdef" for c in sequence_fingerprint.lower())
    ):
        raise ValueError("A validated 64-character sequence fingerprint is required")

    meta = {
        "format": FORMAT,
        "version": VERSION,
        "total_images": int(total_images),
        "reference_index": int(reference_index),
        "last_index": int(last_index),
        "current_index": int(current_index),
        "has_result": bool(has_result),
        "has_roi": roi_corners is not None,
        "roi_corners": (
            [[float(x), float(y)] for x, y in roi_corners]
            if roi_corners is not None
            else None
        ),
        "win_size": int(win_size),
        "lk_params": dict(lk_params),
        "shi_tomasi_params": dict(shi_tomasi_params),
        "grid_params": dict(grid_params),
        "sequence_fingerprint": sequence_fingerprint,
    }

    arrays = {"features": features, "meta": np.array(json.dumps(meta, allow_nan=False))}
    if has_result:
        arrays["active_mask"] = np.asarray(active_mask, dtype=bool).reshape(-1)
        for name, dtype in RESULT_ARRAY_DTYPES.items():
            arrays[name] = np.asarray(result_arrays[name], dtype=dtype)

    with atomic_open(path, "wb") as f:
        np.savez(f, **arrays)
    return path


def load_trackers(path):
    """Read a tracker session written by :func:`save_trackers`.

    Returns a dict with keys: ``total_images``, ``reference_index``, ``last_index``,
    ``current_index``, ``has_result``, ``features``, ``roi_corners`` (or None),
    ``active_mask`` (or None), ``result_arrays`` (dict or None), ``win_size``,
    ``lk_params``, ``shi_tomasi_params``, ``grid_params``.

    Raises :class:`ValueError` (with a human-readable message) if the file cannot be read,
    is not a tracker file, has an unknown version, or is internally inconsistent.
    """
    try:
        loaded = np.load(path, allow_pickle=False)
        if not hasattr(loaded, "files"):
            # np.load returned a bare ndarray (a plain .npy, not our .npz bundle).
            raise ValueError("Not a tracker file (expected a .npz bundle).")

        with loaded as data:
            if "meta" not in data.files:
                raise ValueError("Not a tracker file (no metadata).")
            raw_meta = np.asarray(data["meta"])
            if raw_meta.shape != ():
                raise ValueError("Tracker file metadata must be a scalar JSON value.")
            try:
                meta = json.loads(str(raw_meta.item()), parse_constant=_reject_nonfinite)
            except (TypeError, ValueError) as exc:
                raise ValueError("Tracker file metadata is corrupt.") from exc
            return _parse(meta, data)
    except ValueError:
        raise
    except (OSError, EOFError, BadZipFile, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Could not read tracker file: {exc}") from exc


def _parse(meta, data):
    if not isinstance(meta, dict):
        raise ValueError("Tracker file metadata must be a JSON object.")
    if meta.get("format") != FORMAT:
        raise ValueError("This file is not an ECM Tracker tracker file.")
    try:
        file_version = int(meta.get("version", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Tracker file version is invalid.") from exc
    if file_version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported tracker file version: {meta.get('version')}.")

    if "features" not in data.files:
        raise ValueError("Tracker file is missing reference points.")
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("Reference points have an unexpected shape.")
    if features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Reference points must be non-empty and finite.")
    p = features.shape[0]

    try:
        total_images = int(meta["total_images"])
        reference_index = int(meta["reference_index"])
        last_index = int(meta["last_index"])
        current_index = int(meta.get("current_index", reference_index))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tracker file frame metadata is missing or invalid.") from exc
    if total_images <= 0:
        raise ValueError("Tracker file has an invalid image count.")
    if not 0 <= reference_index <= last_index < total_images:
        raise ValueError("Tracker file contains an invalid reference/last frame range.")
    if not 0 <= current_index < total_images:
        raise ValueError("Tracker file contains an invalid current frame index.")
    n = last_index - reference_index + 1

    raw_has_result = meta.get("has_result", False)
    if not isinstance(raw_has_result, bool):
        raise ValueError("Tracker file has an invalid has_result flag.")
    has_result = raw_has_result
    result_arrays = None
    active_mask = None
    if has_result:
        missing = [k for k in RESULT_ARRAY_DTYPES if k not in data.files]
        if missing or "active_mask" not in data.files:
            raise ValueError("Tracker file claims a result but is missing tracked arrays.")
        expected_shapes = {
            "coords_fw": (n, p, 2),
            "status_fw": (n, p),
            "err_fw": (n, p),
            "coords_bw": (n, p, 2),
            "status_bw": (n, p),
            "err_bw": (n, p),
            "fb_mean_error": (p,),
            "fb_max_error": (p,),
        }
        result_arrays = {}
        for name, dtype in RESULT_ARRAY_DTYPES.items():
            raw_array = np.asarray(data[name])
            if raw_array.shape != expected_shapes[name]:
                raise ValueError(
                    f"Tracked array {name!r} has shape {tuple(raw_array.shape)}; "
                    f"expected {expected_shapes[name]}."
                )
            if name.startswith("status_") and not np.isin(raw_array, (0, 1)).all():
                raise ValueError(f"Tracked status array {name!r} contains values other than 0/1.")
            array = np.asarray(raw_array, dtype=dtype)
            if name.startswith("coords_") and not np.isfinite(array).all():
                raise ValueError(f"Tracked coordinate array {name!r} contains NaN or infinity.")
            if (
                file_version >= 2
                and (name.startswith("err_") or name.startswith("fb_"))
                and np.isnan(array).any()
            ):
                raise ValueError(f"Tracked error array {name!r} contains NaN.")
            result_arrays[name] = array

        raw_mask = np.asarray(data["active_mask"])
        if raw_mask.shape != (p,):
            raise ValueError("Active mask length doesn't match the number of points.")
        if raw_mask.dtype != np.bool_ and not np.isin(raw_mask, (0, 1)).all():
            raise ValueError("Active mask contains values other than boolean/0/1.")
        active_mask = raw_mask.astype(bool, copy=False)

        if file_version == 1:
            result_arrays = _migrate_v1_result(result_arrays)

    roi_corners = meta.get("roi_corners")
    if roi_corners is not None:
        try:
            roi_array = np.asarray(roi_corners, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Tracker ROI metadata is invalid.") from exc
        if (
            roi_array.ndim != 2
            or roi_array.shape[1] != 2
            or roi_array.shape[0] < 3
            or not np.isfinite(roi_array).all()
        ):
            raise ValueError("Tracker ROI must contain at least three finite (x, y) corners.")
        roi_corners = [(float(x), float(y)) for x, y in roi_array]

    def parameter_dict(name):
        value = meta.get(name, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"Tracker parameter section {name!r} must be an object.")
        return dict(value)

    fingerprint = meta.get("sequence_fingerprint")
    if fingerprint is not None:
        if not (
            isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(c in "0123456789abcdef" for c in fingerprint.lower())
        ):
            raise ValueError("Tracker sequence fingerprint is invalid.")
    if file_version >= 2 and fingerprint is None:
        raise ValueError("Tracker file is missing its required sequence fingerprint.")

    try:
        win_size = int(meta.get("win_size", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Tracker win_size metadata is invalid.") from exc
    if win_size < 0:
        raise ValueError("Tracker win_size metadata cannot be negative.")

    return {
        "total_images": total_images,
        "reference_index": reference_index,
        "last_index": last_index,
        "current_index": current_index,
        "has_result": has_result,
        "features": features,
        "roi_corners": roi_corners,
        "active_mask": active_mask,
        "result_arrays": result_arrays,
        "win_size": win_size,
        "lk_params": parameter_dict("lk_params"),
        "shi_tomasi_params": parameter_dict("shi_tomasi_params"),
        "grid_params": parameter_dict("grid_params"),
        "sequence_fingerprint": fingerprint,
        "file_version": file_version,
    }


def _migrate_v1_result(arrays):
    """Upgrade legacy transition-local statuses to v2's cumulative/frozen-track semantics."""
    migrated = {name: np.array(value, copy=True) for name, value in arrays.items()}
    cf, sf, ef = migrated["coords_fw"], migrated["status_fw"], migrated["err_fw"]
    cb, sb, eb = migrated["coords_bw"], migrated["status_bw"], migrated["err_bw"]
    ef[~np.isfinite(ef)] = np.inf
    eb[~np.isfinite(eb)] = np.inf

    sf[:] = np.minimum.accumulate(sf, axis=0)
    for t in range(1, sf.shape[0]):
        dead = sf[t] == 0
        cf[t, dead] = cf[t - 1, dead]
        ef[t, dead] = np.inf

    # Backward tracking progresses last -> reference. A dead forward endpoint is not a valid seed.
    sb[-1] &= sf[-1]
    sb[:] = np.minimum.accumulate(sb[::-1], axis=0)[::-1]
    for t in range(sb.shape[0] - 2, -1, -1):
        dead = sb[t] == 0
        cb[t, dead] = cb[t + 1, dead]
        eb[t, dead] = np.inf

    diff = np.linalg.norm(cf - cb, axis=2)
    valid = (sf == 1) & (sb == 1)
    if valid.shape[0] > 1:
        valid[-1] = False  # identical backward seed carries no FB information
    mean = np.full(sf.shape[1], np.inf, dtype=np.float32)
    maximum = np.full(sf.shape[1], np.inf, dtype=np.float32)
    if valid.shape[0] == 1:
        mean.fill(0.0)
        maximum.fill(0.0)
    else:
        for point in range(sf.shape[1]):
            values = diff[valid[:, point], point]
            if values.size:
                mean[point] = float(values.mean())
                maximum[point] = float(values.max())
    migrated["fb_mean_error"] = mean
    migrated["fb_max_error"] = maximum
    return migrated
