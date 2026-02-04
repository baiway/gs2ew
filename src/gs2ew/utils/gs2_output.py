"""Utilities for reading GS2 output files."""

import numpy as np
import xarray as xr
from pathlib import Path


def load_gs2_output(filepath: str | Path) -> xr.Dataset:
    """Load a GS2 `.out.nc` file."""
    return xr.open_dataset(filepath, engine="netcdf4")


def detect_saturation_time(
    ds: xr.Dataset,
    window: float = 20.0,
    threshold: float = 0.1
) -> float:
    """Estimate the saturation time (end of linear growth phase) by
    detecting when the growth rate over some `window` drops below a
    certain `threshold`.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file)

    window : float, optional
        Time interval (in same units as time coordinate) for computing
        growth rate. Default is 20.0.

    threshold : float, optional
        Growth rate threshold (in inverse time units) below which
        saturation is considered to have occurred. Default is 0.1.

    Returns
    -------
    float
        Time at which saturation is detected. Returns NaN if saturation is
        not detected.
    """

    # Extract time and field data as NumPy arrays
    t = ds["t"].values
    phi2 = ds["phi2"].values
    logphi2 = np.log(phi2) # linearise

    # Convert window from time units to index
    idx = int(window / (t[1] - t[0]))

    # Compute rolling growth rate of points separated by `idx`
    # using finite differences
    growth_rate = np.full(len(t), np.nan)
    growth_rate[idx:] = (logphi2[idx:] - logphi2[:-idx]) / (t[idx:] - t[:-idx])

    # Find index of maximum growth rate (peak of linear growth phase)
    max_growth_idx = np.nanargmax(growth_rate)

    # Look for saturation only *after* the peak growth rate
    # (to avoid triggering on early transients)
    saturated_times = t[max_growth_idx:][growth_rate[max_growth_idx:] < threshold]
    if len(saturated_times) > 0:
        return float(saturated_times[0])
    return np.nan  # saturation not detected
