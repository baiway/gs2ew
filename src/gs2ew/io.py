"""Reading GS2 output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_gs2_output(filepath: str | Path, chunks: dict | None = None) -> xr.Dataset:
    """Open a GS2 `.out.nc` file.

    Opened lazily so that large per-timestep arrays are read on demand. Pass
    `chunks` (e.g. ``{"t": 1}``) to go through dask as well; this is rarely
    needed now the velocity-resolved diagnostics are gone, as the remaining
    transfer arrays are tens of megabytes rather than tens of gigabytes.
    """
    return xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)


def detect_saturation_time(
    ds: xr.Dataset,
    window: float = 20.0,
    threshold: float = 0.1,
) -> float:
    """Estimate the end of the linear growth phase.

    Computes a rolling growth rate of ``log(phi2)`` over `window` time units
    and returns the first time after the growth-rate peak at which it falls
    below `threshold`.

    Returns NaN — rather than raising — whenever saturation cannot be
    determined: if `phi2` is missing, if the run is too short to span a single
    `window`, if `phi2` is non-positive (so the log is undefined), or if the
    run simply never saturates. Callers are expected to fall back to the start
    of the run.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from a `.out.nc` file).
    window : float, optional
        Time interval over which the growth rate is measured. Default 20.0.
    threshold : float, optional
        Growth rate below which the run is considered saturated. Default 0.1.

    Returns
    -------
    float
        The detected saturation time, or NaN if it could not be determined.
    """
    if "phi2" not in ds or "t" not in ds:
        return float("nan")

    t = np.asarray(ds["t"].values, dtype=float)
    phi2 = np.asarray(ds["phi2"].values, dtype=float)

    if t.size < 2:
        return float("nan")

    # log(phi2) linearises exponential growth. Non-positive or non-finite
    # entries (a crashed or not-yet-started run) make the log meaningless.
    with np.errstate(divide="ignore", invalid="ignore"):
        logphi2 = np.log(phi2)
    if not np.any(np.isfinite(logphi2)):
        return float("nan")

    # Convert the window from time units to a number of timesteps. GS2's
    # output cadence is uniform, so a single spacing is enough.
    dt = t[1] - t[0]
    if dt <= 0:
        return float("nan")
    idx = int(window / dt)
    if idx < 1 or idx >= t.size:
        # The run is shorter than one window, so no growth rate can be formed.
        return float("nan")

    growth_rate = np.full(t.size, np.nan)
    growth_rate[idx:] = (logphi2[idx:] - logphi2[:-idx]) / (t[idx:] - t[:-idx])

    if not np.any(np.isfinite(growth_rate)):
        return float("nan")

    # Search only after the peak growth rate, so early transients (which can
    # dip below the threshold before the instability takes off) don't trigger.
    max_growth_idx = int(np.nanargmax(growth_rate))
    tail = growth_rate[max_growth_idx:]
    saturated = t[max_growth_idx:][np.isfinite(tail) & (tail < threshold)]

    return float(saturated[0]) if saturated.size else float("nan")


__all__ = ["load_gs2_output", "detect_saturation_time"]
