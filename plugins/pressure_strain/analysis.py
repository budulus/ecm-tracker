"""Qt-free pressure parsing, frame synchronization, strain fitting, and CSV export."""
from __future__ import annotations

import csv
import os
import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.plugins.analysis import fit_affine, principal_stretches
from app.plugins.analysis import atomic_open


STRAIN_MODES = ("epsilon_1", "epsilon_2", "mean")

LOGGER_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# Camera file names that carry the capture time: epoch milliseconds, 13 digits, optionally
# zero-padded up to 16 (e.g. Image_0001791895394319.bmp); the 17-digit form is local wall-clock
# time YYYYMMDDHHMMSSmmm.
_EPOCH_MS_NAME = re.compile(r"^Image_(\d{13,16})\.[A-Za-z0-9]+$")
_LOCAL_TIME_NAME = re.compile(r"^Image_(\d{8})(\d{6})(\d{3})\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class PressureData:
    """Parsed pressure samples on the PC clock (epoch seconds), strictly increasing in time.

    ``time_s`` is the logger's own sample clock (``Arduino_ms``) shifted onto the PC clock by the
    median of ``Timestamp - Arduino_ms``. ``clock_residual_s`` is the largest deviation of a single
    row's ``Timestamp`` from that model (PC timestamp jitter plus any oscillator drift).
    """

    source_path: str
    time_s: np.ndarray
    pressure_mbar: np.ndarray
    source_rows: int
    source_sha256: str = ""
    clock_offset_s: float = 0.0
    clock_residual_s: float = 0.0


@dataclass(frozen=True)
class AlignedPressure:
    """Pressure and image timing for every global frame in the loaded sequence.

    Frames captured outside the pressure log's time range get NaN pressure.
    """

    image_time_s: np.ndarray
    image_elapsed_s: np.ndarray
    pressure_mbar: np.ndarray
    provenance: dict = field(default_factory=dict)

    @property
    def covered(self) -> np.ndarray:
        """Boolean mask of frames that lie inside the pressure log's time range."""
        return np.isfinite(self.pressure_mbar)


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
    """Parse ``Timestamp``, ``Arduino_ms`` and ``Raw_mbar`` from a pressure logger CSV.

    ``Timestamp`` is the PC's local wall-clock time when the row was received
    (``YYYY-MM-DD HH:MM:SS.fff``), interpreted in this computer's time zone. It jitters because
    serial rows arrive in bursts, so sample times come from the logger's own ``Arduino_ms`` clock,
    which must be strictly increasing (a logger restart inside one file is rejected). That clock
    is shifted onto the PC clock by one constant offset, the median of ``Timestamp - Arduino_ms``.
    Oscillator drift is not modelled; on real logs it stayed below ~0.1 s over 8.5 minutes.
    """
    pc_time = []
    logger_s = []
    pressure = []
    required = ("Timestamp", "Arduino_ms", "Raw_mbar")
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = [name for name in required if name not in fields]
            if missing:
                raise ValueError("Missing required CSV column(s): " + ", ".join(missing))
            for row_number, row in enumerate(reader, start=2):
                try:
                    t_pc = datetime.strptime(row["Timestamp"], LOGGER_TIMESTAMP_FORMAT).timestamp()
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {row_number} has a Timestamp not in {LOGGER_TIMESTAMP_FORMAT} format"
                    ) from exc
                try:
                    t_logger = float(row["Arduino_ms"]) / 1000.0
                    p = float(row["Raw_mbar"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {row_number} has a non-numeric Arduino_ms or Raw_mbar value"
                    ) from exc
                if not (np.isfinite(t_logger) and np.isfinite(p)):
                    raise ValueError(
                        f"Row {row_number} has a non-finite Arduino_ms or Raw_mbar value"
                    )
                pc_time.append(t_pc)
                logger_s.append(t_logger)
                pressure.append(p)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read pressure CSV: {exc}") from exc

    if len(logger_s) < 2:
        raise ValueError("Pressure CSV must contain at least two data rows")
    pc_array = np.asarray(pc_time, dtype=np.float64)
    logger_array = np.asarray(logger_s, dtype=np.float64)
    if np.any(np.diff(logger_array) <= 0):
        raise ValueError(
            "Arduino_ms must be strictly increasing in CSV row order "
            "(was the logger restarted inside this file?)"
        )
    difference = pc_array - logger_array
    offset = float(np.median(difference))

    with open(path, "rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return PressureData(
        source_path=os.path.abspath(path),
        time_s=logger_array + offset,
        pressure_mbar=np.asarray(pressure, dtype=np.float64),
        source_rows=len(logger_s),
        source_sha256=source_sha256,
        clock_offset_s=offset,
        clock_residual_s=float(np.abs(difference - offset).max()),
    )


def image_capture_times(paths) -> np.ndarray:
    """Return each frame's capture time (epoch seconds) parsed from its file name.

    Recognised names are ``Image_<epoch ms>.<ext>`` (13 digits, may be zero-padded to 16) and
    ``Image_YYYYMMDDHHMMSSmmm.<ext>`` (local wall-clock time, interpreted in this computer's time
    zone like the logger's ``Timestamp``). File modification time is deliberately not used: it is
    the save or copy time, not the capture time. Any other name is rejected.
    """
    times = []
    for index, path in enumerate(paths):
        name = os.path.basename(path)
        epoch = _EPOCH_MS_NAME.match(name)
        local = _LOCAL_TIME_NAME.match(name)
        if epoch:
            times.append(int(epoch.group(1)) / 1000.0)
        elif local:
            try:
                stamp = datetime.strptime(local.group(1) + local.group(2), "%Y%m%d%H%M%S")
            except ValueError as exc:
                raise ValueError(f"Frame {index} ({name}) has an invalid date/time in its name") from exc
            times.append(stamp.timestamp() + int(local.group(3)) / 1000.0)
        else:
            raise ValueError(
                f"Frame {index} ({name}) has no capture time in its file name. Expected "
                "Image_<epoch ms>.<ext> or Image_YYYYMMDDHHMMSSmmm.<ext>."
            )
    return np.asarray(times, dtype=np.float64)


def align_pressure_to_images(
    pressure: PressureData, image_time_s: np.ndarray
) -> AlignedPressure:
    """Interpolate pressure at each frame's real capture time; NaN outside the log's time range."""
    image_time_s = np.asarray(image_time_s, dtype=np.float64)
    if image_time_s.ndim != 1 or image_time_s.size < 2:
        raise ValueError("At least two ordered image timestamps are required")
    if not np.isfinite(image_time_s).all():
        raise ValueError("Image capture times must be finite")
    if np.any(np.diff(image_time_s) < 0):
        raise ValueError(
            "Image capture times go backwards in loaded frame order; "
            "pressure alignment would be ambiguous"
        )
    if image_time_s[-1] <= image_time_s[0]:
        raise ValueError("Image capture times must have a positive total span")

    values = np.interp(
        image_time_s, pressure.time_s, pressure.pressure_mbar, left=np.nan, right=np.nan
    )
    if not np.isfinite(values).any():
        raise ValueError(
            "No image lies inside the pressure log's time range "
            f"(images {_clock_text(image_time_s[0])}-{_clock_text(image_time_s[-1])}, "
            f"log {_clock_text(pressure.time_s[0])}-{_clock_text(pressure.time_s[-1])}). "
            "Check that the CSV belongs to this image sequence."
        )
    return AlignedPressure(
        image_time_s=image_time_s.copy(),
        image_elapsed_s=image_time_s - image_time_s[0],
        pressure_mbar=values,
        provenance={"clock_model": "filename_capture_time_vs_logger_clock_constant_offset",
                    "pressure_source_sha256": pressure.source_sha256,
                    "pressure_clock_offset_s": pressure.clock_offset_s,
                    "pressure_clock_max_residual_s": pressure.clock_residual_s,
                    "image_time_first_last_s": [float(image_time_s[0]), float(image_time_s[-1])],
                    "pressure_time_first_last_s": [float(pressure.time_s[0]),
                                                   float(pressure.time_s[-1])]},
    )


def _clock_text(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%Y-%m-%d %H:%M:%S")


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
    outside = int((~aligned.covered[sl]).sum())
    if outside:
        raise ValueError(
            f"{outside} of {frame_count} tracked frames lie outside the pressure log's time range; "
            "move Reference/Last inside the logged frames"
        )
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
