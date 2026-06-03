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

import numpy as np

FORMAT = "ecmtracker-trackers"
VERSION = 1

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
):
    """Write a tracker session to ``path`` (a single ``.npz``); return the path written.

    ``features`` is the ``(P, 2)`` reference-frame seed array (required). ``result_arrays``
    is a dict of the eight :class:`TrackerResult` arrays, or ``None`` when tracking hasn't
    been run; ``active_mask`` is the ``(P,)`` keep mask, or ``None`` in that case.
    ``roi_corners`` is a list of ``(x, y)`` tuples or ``None``.
    """
    if not path.endswith(".npz"):
        path += ".npz"

    features = np.asarray(features, dtype=np.float32).reshape(-1, 2)
    has_result = result_arrays is not None

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
    }

    arrays = {"features": features, "meta": np.array(json.dumps(meta))}
    if has_result:
        arrays["active_mask"] = np.asarray(active_mask, dtype=bool).reshape(-1)
        for name, dtype in RESULT_ARRAY_DTYPES.items():
            arrays[name] = np.asarray(result_arrays[name], dtype=dtype)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **arrays)
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
    except OSError as exc:
        raise ValueError(f"Could not open tracker file: {exc}") from exc

    if not hasattr(loaded, "files"):
        # np.load returned a bare ndarray (a plain .npy, not our .npz bundle).
        raise ValueError("Not a tracker file (expected a .npz bundle).")

    with loaded as data:
        if "meta" not in data.files:
            raise ValueError("Not a tracker file (no metadata).")
        try:
            meta = json.loads(str(data["meta"]))
        except ValueError as exc:  # JSONDecodeError is a ValueError subclass
            raise ValueError("Tracker file metadata is corrupt.") from exc
        return _parse(meta, data)


def _parse(meta, data):
    if meta.get("format") != FORMAT:
        raise ValueError("This file is not an ECM Tracker tracker file.")
    if int(meta.get("version", 0)) != VERSION:
        raise ValueError(f"Unsupported tracker file version: {meta.get('version')}.")

    if "features" not in data.files:
        raise ValueError("Tracker file is missing reference points.")
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("Reference points have an unexpected shape.")
    p = features.shape[0]

    reference_index = int(meta["reference_index"])
    last_index = int(meta["last_index"])
    n = last_index - reference_index + 1

    has_result = bool(meta.get("has_result"))
    result_arrays = None
    active_mask = None
    if has_result:
        missing = [k for k in RESULT_ARRAY_DTYPES if k not in data.files]
        if missing or "active_mask" not in data.files:
            raise ValueError("Tracker file claims a result but is missing tracked arrays.")
        result_arrays = {k: np.asarray(data[k]) for k in RESULT_ARRAY_DTYPES}
        active_mask = np.asarray(data["active_mask"], dtype=bool).reshape(-1)
        cf = result_arrays["coords_fw"]
        if cf.shape != (n, p, 2):
            raise ValueError(
                f"Tracked coords shape {tuple(cf.shape)} doesn't match "
                f"{n} frames x {p} points."
            )
        if active_mask.shape != (p,):
            raise ValueError("Active mask length doesn't match the number of points.")

    roi_corners = meta.get("roi_corners")
    if roi_corners is not None:
        roi_corners = [(float(x), float(y)) for x, y in roi_corners]

    return {
        "total_images": int(meta["total_images"]),
        "reference_index": reference_index,
        "last_index": last_index,
        "current_index": int(meta.get("current_index", reference_index)),
        "has_result": has_result,
        "features": features,
        "roi_corners": roi_corners,
        "active_mask": active_mask,
        "result_arrays": result_arrays,
        "win_size": int(meta.get("win_size", 0)),
        "lk_params": dict(meta.get("lk_params") or {}),
        "shi_tomasi_params": dict(meta.get("shi_tomasi_params") or {}),
        "grid_params": dict(meta.get("grid_params") or {}),
    }
