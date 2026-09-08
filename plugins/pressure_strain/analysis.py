"""Qt-free pressure parsing, frame synchronization, strain fitting, and CSV export."""
from __future__ import annotations

import csv
import os
import json
import hashlib
from dataclasses import dataclass, field

import numpy as np

from app.plugins.analysis import fit_affine, principal_stretches
from app.plugins.analysis import atomic_open


STRAIN_MODES = ("epsilon_1", "epsilon_2", "mean")


@dataclass(frozen=True)
class PressureData:
    """Parsed pressure samples after duplicate elapsed-time rows have been consolidated."""

    source_path: str
    elapsed_s: np.ndarray
    pressure_mbar: np.ndarray
    source_rows: int
    source_sha256: str = ""


@dataclass(frozen=True)
class AlignedPressure:
    """Pressure and image timing for every global frame in the loaded sequence."""

    image_mtime_s: np.ndarray
    image_elapsed_s: np.ndarray
    pressure_mbar: np.ndarray
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StrainSeries:
    """Principal linear strains for every cut-indexed tracked frame."""

    epsilon_1: np.ndarray
    epsilon_2: np.ndarray
    mean: np.ndarray
    n_valid_points: np.ndarray

    def values(self, mode: str) -> np.ndarray:
        if mode not in STRAIN_MODES:
            raise ValueError(f"Unknown strain mode {mode!r}")
        return getattr(self, mode)


@dataclass(frozen=True)
class PlotSeries:
    """The exact pressure/strain rows shown in the current plot and written on export."""

    global_indices: np.ndarray
    image_elapsed_s: np.ndarray
    pressure_mbar: np.ndarray
    strain: np.ndarray
    n_valid_points: np.ndarray
    strain_mode: str
    pressure_zeroed: bool
    provenance: dict = field(default_factory=dict)


def parse_pressure_csv(path: str) -> PressureData:
    """Parse ``Elapsed_s`` and ``Raw_mbar`` from a pressure logger CSV.

    Elapsed time must be finite and nondecreasing. Duplicate time groups are averaged because
    interpolation needs a strictly increasing x-axis. The first and final groups deliberately use
    the literal first/final row values so the documented frame endpoint correspondence stays exact.
    """
    elapsed = []
    pressure = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = [name for name in ("Elapsed_s", "Raw_mbar") if name not in fields]
            if missing:
                raise ValueError("Missing required CSV column(s): " + ", ".join(missing))
            for row_number, row in enumerate(reader, start=2):
                try:
                    t = float(row["Elapsed_s"])
                    p = float(row["Raw_mbar"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {row_number} has a non-numeric Elapsed_s or Raw_mbar value"
                    ) from exc
                if not (np.isfinite(t) and np.isfinite(p)):
                    raise ValueError(
                        f"Row {row_number} has a non-finite Elapsed_s or Raw_mbar value"
                    )
                elapsed.append(t)
                pressure.append(p)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read pressure CSV: {exc}") from exc

    if len(elapsed) < 2:
        raise ValueError("Pressure CSV must contain at least two data rows")
    elapsed_array = np.asarray(elapsed, dtype=np.float64)
    pressure_array = np.asarray(pressure, dtype=np.float64)
    if np.any(np.diff(elapsed_array) < 0):
        raise ValueError("Elapsed_s must be nondecreasing in CSV row order")
    if elapsed_array[-1] <= elapsed_array[0]:
        raise ValueError("Elapsed_s must have a positive first-to-last time span")

    unique_time, starts, counts = np.unique(
        elapsed_array, return_index=True, return_counts=True
    )
    consolidated = np.empty(unique_time.size, dtype=np.float64)
    for group, (start, count) in enumerate(zip(starts, counts)):
        values = pressure_array[start : start + count]
        if start == 0:
            consolidated[group] = values[0]
        elif start + count == pressure_array.size:
            consolidated[group] = values[-1]
        else:
            consolidated[group] = float(values.mean())

    with open(path, "rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return PressureData(
        source_path=os.path.abspath(path),
        elapsed_s=unique_time,
        pressure_mbar=consolidated,
        source_rows=len(elapsed),
        source_sha256=source_sha256,
    )


def image_modified_times(paths) -> np.ndarray:
    """Return modification timestamps in seconds for ordered global-frame paths."""
    timestamps = []
    for index, path in enumerate(paths):
        try:
            timestamps.append(os.stat(path).st_mtime_ns / 1_000_000_000.0)
        except OSError as exc:
            raise ValueError(f"Could not read modification time for frame {index}: {exc}") from exc
    return np.asarray(timestamps, dtype=np.float64)


def align_pressure_to_images(
    pressure: PressureData, image_mtime_s: np.ndarray
) -> AlignedPressure:
    """Endpoint-anchor pressure time to image modification time and interpolate every frame."""
    image_mtime_s = np.asarray(image_mtime_s, dtype=np.float64)
    if image_mtime_s.ndim != 1 or image_mtime_s.size < 2:
        raise ValueError("At least two ordered image timestamps are required")
    if not np.isfinite(image_mtime_s).all():
        raise ValueError("Image modification timestamps must be finite")
    if np.any(np.diff(image_mtime_s) < 0):
        raise ValueError(
            "Image modification times go backwards in loaded frame order; "
            "pressure alignment would be ambiguous"
        )
    image_span = image_mtime_s[-1] - image_mtime_s[0]
    if image_span <= 0:
        raise ValueError("Image modification timestamps must have a positive total span")

    phase = (image_mtime_s - image_mtime_s[0]) / image_span
    elapsed = pressure.elapsed_s
    query = elapsed[0] + phase * (elapsed[-1] - elapsed[0])
    values = np.interp(query, elapsed, pressure.pressure_mbar)
    # Be explicit about the two contractual anchors even if interpolation implementation details
    # change later.
    values[0] = pressure.pressure_mbar[0]
    values[-1] = pressure.pressure_mbar[-1]
    return AlignedPressure(
        image_mtime_s=image_mtime_s.copy(),
        image_elapsed_s=image_mtime_s - image_mtime_s[0],
        pressure_mbar=values,
        provenance={"clock_model": "endpoint_mtime_affine_assumption",
                    "pressure_source_sha256": pressure.source_sha256,
                    "image_mtime_s": image_mtime_s.tolist(),
                    "pressure_elapsed_anchors_s": [float(elapsed[0]), float(elapsed[-1])],
                    "pressure_time_per_image_time": float((elapsed[-1] - elapsed[0]) / image_span)},
    )


def compute_principal_strains(coords, status=None) -> StrainSeries:
    """Fit one homogeneous affine map per cut frame and return principal linear strains."""
    from app.plugins.analysis import compute_series
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3:
        raise ValueError("coords must have shape (frames, points, 2)")
    series = compute_series(coords, np.arange(coords.shape[0], dtype=float), status)
    return StrainSeries(series.eps_1, series.eps_2,
                        (series.eps_1 + series.eps_2) / 2, series.n_points)


def build_plot_series(
    aligned: AlignedPressure,
    strains: StrainSeries,
    reference_index: int,
    last_index: int,
    strain_mode: str,
    zero_pressure: bool,
) -> PlotSeries:
    """Slice globally aligned pressure into the tracked cut range without mixing index systems."""
    if strain_mode not in STRAIN_MODES:
        raise ValueError(f"Unknown strain mode {strain_mode!r}")
    if not (0 <= reference_index <= last_index < aligned.pressure_mbar.size):
        raise ValueError("Tracked global frame range is outside the aligned pressure data")
    frame_count = last_index - reference_index + 1
    selected = strains.values(strain_mode)
    if selected.size != frame_count or strains.n_valid_points.size != frame_count:
        raise ValueError("Tracked strain frames do not match the current reference/last range")

    sl = slice(reference_index, last_index + 1)
    plotted_pressure = aligned.pressure_mbar[sl].copy()
    if zero_pressure:
        plotted_pressure -= plotted_pressure[0]
    return PlotSeries(
        global_indices=np.arange(reference_index, last_index + 1, dtype=np.int64),
        image_elapsed_s=aligned.image_elapsed_s[sl].copy(),
        pressure_mbar=plotted_pressure,
        strain=selected.copy(),
        n_valid_points=strains.n_valid_points.copy(),
        strain_mode=strain_mode,
        pressure_zeroed=bool(zero_pressure),
        provenance={**aligned.provenance, "reference_global": reference_index,
                    "last_global": last_index, "reference_pressure_mbar": float(aligned.pressure_mbar[reference_index])},
    )


def export_plot_csv(path: str, series: PlotSeries, frame_paths) -> None:
    """Atomically export exactly the rows and transforms represented by ``series``."""
    frame_paths = tuple(frame_paths)
    if series.global_indices.size and int(series.global_indices[-1]) >= len(frame_paths):
        raise ValueError("Plot frame indices are outside the current image sequence")
    with atomic_open(path, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_global",
                "image_filename",
                "image_elapsed_s",
                "pressure_mbar",
                "strain",
                "strain_mode",
                "pressure_zeroed",
                "n_valid_points",
                "provenance_json",
            ]
        )
        for i, global_index in enumerate(series.global_indices):
            strain = series.strain[i]
            writer.writerow(
                [
                    int(global_index),
                    os.path.basename(frame_paths[int(global_index)]),
                    f"{series.image_elapsed_s[i]:.9g}",
                    f"{series.pressure_mbar[i]:.12g}",
                    "" if not np.isfinite(strain) else f"{strain:.12g}",
                    series.strain_mode,
                    int(series.pressure_zeroed),
                    int(series.n_valid_points[i]),
                    json.dumps(series.provenance, separators=(",", ":"), allow_nan=False) if i == 0 else "",
                ]
            )
