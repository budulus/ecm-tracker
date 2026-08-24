"""Parse the two raw experiment files into structured, numpy-backed records.

Both files are tab-delimited with multi-row headers. Parsing is header-aware (it locates the
column-names row and classifies columns by name) with a positional fallback, so it survives
small layout shifts without hard-coding column indices. Qt-free and side-effect-free so it can
be unit-tested headlessly.

* ``parse_image_log`` reads the acquisition log (``VDCCam.log``): one row per image, in
  acquisition order, with an elapsed-milliseconds timestamp.
* ``parse_sensor`` reads the MTS sensor export (``specimen.dat``): time plus displacement
  (``Weg``) and force (``Kraft``) for two clamps (``Achse-N``).
"""
from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Files seen in the wild are ASCII/UTF-8; fall back to latin-1 so a stray byte never aborts a load.
_ENCODINGS = ("utf-8-sig", "latin-1")


def _read_rows(path: str) -> List[List[str]]:
    """Read a tab-delimited text file into a list of cell-lists (no trimming of empty cells)."""
    text = None
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 decodes any byte stream
        raise IOError(f"Could not decode {path}")
    rows = []
    for line in text.splitlines():
        rows.append(line.split("\t"))
    return rows


def _is_float(cell: str) -> bool:
    try:
        return math.isfinite(float(cell))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- image log


@dataclass
class ImageLog:
    """One row per image, in acquisition order (NOT filename order)."""

    filenames: List[str]
    time_ms: np.ndarray  # (M,) float64, elapsed milliseconds from acquisition start
    log_path: str
    header_rows: List[List[str]] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.filenames)


def parse_image_log(log_path: str) -> ImageLog:
    """Parse ``VDCCam.log`` → :class:`ImageLog`.

    Heuristic: the column-names row is the first whose first cell is ``File``; the timestamp
    column is the one whose name contains ``ms`` (``Time [ms]``). Data rows follow in file order
    and that order *is* the image sequence. Falls back to (filename = col 0, time = last column)
    if the header can't be recognised.
    """
    rows = _read_rows(log_path)
    if not rows:
        raise ValueError(f"Image log is empty: {log_path}")

    header_idx = next(
        (i for i, r in enumerate(rows) if r and r[0].strip().lower() == "file"), None
    )
    if header_idx is not None:
        names = [c.strip() for c in rows[header_idx]]
        file_col = 0
        ms_col = next((i for i, n in enumerate(names) if "ms" in n.lower()), len(names) - 1)
        data_start = header_idx + 1
        header_rows = rows[: header_idx + 1]
    else:  # positional fallback
        file_col, ms_col, data_start = 0, -1, 3
        header_rows = rows[:3]

    filenames: List[str] = []
    times: List[float] = []
    for r in rows[data_start:]:
        if len(r) <= max(file_col, ms_col):
            continue
        name = r[file_col].strip()
        if not name or not _is_float(r[ms_col]):
            continue
        filenames.append(name)
        times.append(float(r[ms_col]))

    if not filenames:
        raise ValueError(f"No image rows found in {log_path}")

    time_ms = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(time_ms) < 0):
        raise ValueError(f"Image timestamps are not monotonic in {log_path}")

    return ImageLog(
        filenames=filenames,
        time_ms=time_ms,
        log_path=log_path,
        header_rows=header_rows,
    )


def resolve_image_paths(image_log: ImageLog, images_dir: str) -> List[str]:
    """Resolve logged filenames against ``images_dir`` to absolute paths, in log order.

    Raises ``ValueError`` if any logged image is missing on disk — a missing frame would desync
    the image timeline from the loaded sequence, which must never happen silently.
    """
    paths = [os.path.join(images_dir, name) for name in image_log.filenames]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ValueError(
            f"{len(missing)} image(s) listed in the log are missing from {images_dir}; "
            f"first missing: {os.path.basename(missing[0])}"
        )
    return paths


# --------------------------------------------------------------------------- sensor file


@dataclass
class Sensor:
    """Two-clamp MTS stream. Displacement is ``Weg`` (mm), force is ``Kraft`` (N)."""

    time_s: np.ndarray   # (K,) float64, elapsed seconds
    disp_a: np.ndarray   # (K,) clamp A displacement
    force_a: np.ndarray  # (K,) clamp A force
    disp_b: np.ndarray   # (K,) clamp B displacement
    force_b: np.ndarray  # (K,) clamp B force
    clamp_labels: Tuple[str, str]  # e.g. ("Achse-2", "Achse-4"), for UI labelling
    units: dict = field(default_factory=dict)
    header_rows: List[List[str]] = field(default_factory=list)
    n_skipped: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.time_s.shape[0]


_ACHSE_RE = re.compile(r"achse[-\s]*([0-9]+)", re.IGNORECASE)


def _classify_sensor_columns(names: List[str]) -> Optional[dict]:
    """From the column-names row, return ``{"time", "disp_a", "force_a", "disp_b", "force_b",
    "labels"}`` of column indices, or ``None`` if the name-based heuristic can't resolve two
    complete clamps."""
    disp_cols, force_cols = [], []
    for i, n in enumerate(names):
        low = n.lower()
        if "weg" in low:
            disp_cols.append(i)
        elif "kraft" in low:
            force_cols.append(i)
    if len(disp_cols) < 2 or len(force_cols) < 2:
        return None

    # Group displacement+force by the Achse-N token so column order doesn't matter.
    def achse(idx: int) -> Optional[int]:
        m = _ACHSE_RE.search(names[idx])
        return int(m.group(1)) if m else None

    clamps: dict = {}
    for i in disp_cols:
        k = achse(i)
        if k is not None:
            clamps.setdefault(k, {})["disp"] = i
    for i in force_cols:
        k = achse(i)
        if k is not None:
            clamps.setdefault(k, {})["force"] = i
    complete = {k: v for k, v in clamps.items() if "disp" in v and "force" in v}
    if len(complete) < 2:
        return None

    a, b = sorted(complete)[:2]  # clamp A = lowest Achse number
    return {
        "time": 0,
        "disp_a": complete[a]["disp"],
        "force_a": complete[a]["force"],
        "disp_b": complete[b]["disp"],
        "force_b": complete[b]["force"],
        "labels": (f"Achse-{a}", f"Achse-{b}"),
    }


def parse_sensor(dat_path: str) -> Sensor:
    """Parse ``specimen.dat`` → :class:`Sensor`.

    Heuristic: the column-names row is the first whose first cell is ``Time``; the next row holds
    units (kept for the manifest, not parsed as data). Columns are classified by name —
    ``Weg`` = displacement, ``Kraft`` = force — and paired into two clamps by their ``Achse-N``
    token. Falls back to fixed positions (time, dispA, forceA, dispB, forceB) when the header
    can't be recognised. Non-numeric data rows are skipped and counted.
    """
    rows = _read_rows(dat_path)
    if not rows:
        raise ValueError(f"Sensor file is empty: {dat_path}")

    header_idx = next(
        (i for i, r in enumerate(rows) if r and r[0].strip().lower() == "time"), None
    )
    labels = ("Clamp A", "Clamp B")
    units: dict = {}
    warnings: List[str] = []

    if header_idx is not None:
        names = [c.strip() for c in rows[header_idx]]
        cols = _classify_sensor_columns(names)
        if cols is not None:
            labels = cols["labels"]
            idx = (cols["time"], cols["disp_a"], cols["force_a"], cols["disp_b"], cols["force_b"])
        else:
            warnings.append("Could not classify sensor columns by name; used fixed positions.")
            idx = (0, 1, 2, 3, 4)
        # The row after the header is the units row IF it isn't itself numeric data.
        if header_idx + 1 < len(rows) and not _is_float(rows[header_idx + 1][min(1, len(rows[header_idx + 1]) - 1)]):
            unit_cells = [c.strip() for c in rows[header_idx + 1]]
            units = {
                "time": unit_cells[idx[0]] if idx[0] < len(unit_cells) else "",
                "disp": unit_cells[idx[1]] if idx[1] < len(unit_cells) else "",
                "force": unit_cells[idx[2]] if idx[2] < len(unit_cells) else "",
            }
            data_start = header_idx + 2
        else:
            data_start = header_idx + 1
        header_rows = rows[:data_start]
    else:  # positional fallback (no recognisable header): start at the first all-numeric row
        warnings.append("No 'Time' header row found; classified columns by position.")
        idx = (0, 1, 2, 3, 4)
        need0 = max(idx)
        data_start = next(
            (i for i, r in enumerate(rows) if len(r) > need0 and all(_is_float(r[c]) for c in idx)),
            len(rows),
        )
        header_rows = rows[:data_start]

    need = max(idx)
    cols_data: List[List[float]] = [[], [], [], [], []]
    n_skipped = 0
    for r in rows[data_start:]:
        if len(r) <= need or not all(_is_float(r[c]) for c in idx):
            if any(c.strip() for c in r):  # ignore truly blank lines
                n_skipped += 1
            continue
        for j, c in enumerate(idx):
            cols_data[j].append(float(r[c]))

    if not cols_data[0]:
        raise ValueError(f"No numeric sensor rows found in {dat_path}")

    time_s = np.asarray(cols_data[0], dtype=np.float64)
    disp_a = np.asarray(cols_data[1], dtype=np.float64)
    force_a = np.asarray(cols_data[2], dtype=np.float64)
    disp_b = np.asarray(cols_data[3], dtype=np.float64)
    force_b = np.asarray(cols_data[4], dtype=np.float64)

    # np.interp requires increasing x; sort by time if the export isn't monotonic.
    if np.any(np.diff(time_s) < 0):
        warnings.append("Sensor time was not monotonic; rows were sorted by time.")
        order = np.argsort(time_s, kind="stable")
        time_s, disp_a, force_a, disp_b, force_b = (
            time_s[order], disp_a[order], force_a[order], disp_b[order], force_b[order]
        )

    # np.interp requires a strictly increasing x-axis. Average duplicate timestamp rows rather
    # than allowing their implementation-dependent ordering to affect aligned measurements.
    if np.any(np.diff(time_s) == 0):
        warnings.append("Duplicate sensor timestamps were averaged.")
        unique_time, inverse = np.unique(time_s, return_inverse=True)

        def averaged(values):
            sums = np.bincount(inverse, weights=values)
            counts = np.bincount(inverse)
            return sums / counts

        time_s = unique_time
        disp_a, force_a, disp_b, force_b = (
            averaged(disp_a),
            averaged(force_a),
            averaged(disp_b),
            averaged(force_b),
        )

    return Sensor(
        time_s=time_s,
        disp_a=disp_a,
        force_a=force_a,
        disp_b=disp_b,
        force_b=force_b,
        clamp_labels=labels,
        units=units,
        header_rows=header_rows,
        n_skipped=n_skipped,
        warnings=warnings,
    )
