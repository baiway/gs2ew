"""Reading GS2 output files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


#: Suffix of GS2's main output file.
OUTPUT_SUFFIX = ".out.nc"


@dataclass(frozen=True)
class RunFiles:
    """The files of one GS2 run.

    `input_file` is the ``<name>.in`` next to the output, or None if there is
    none. The deck embedded in the output is the authoritative copy; this one
    is only checked against it.
    """

    run_dir: Path
    out_nc: Path
    input_file: Path | None

    @property
    def name(self) -> str:
        """The run name: the output's filename without ``.out.nc``."""
        return self.out_nc.name[: -len(OUTPUT_SUFFIX)]


def find_run(path: str | Path) -> RunFiles:
    """Locate a GS2 run from a run directory or its `.out.nc` file.

    A directory must contain exactly one ``*.out.nc``; anything else is
    ambiguous and raises rather than guessing.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist, or is a directory with no ``*.out.nc``.
    ValueError
        If `path` is a directory with several ``*.out.nc`` files, or a file
        that is not a ``.out.nc``.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    if path.is_dir():
        candidates = sorted(path.glob(f"*{OUTPUT_SUFFIX}"))
        if not candidates:
            raise FileNotFoundError(f"No *{OUTPUT_SUFFIX} file in {path}.")
        if len(candidates) > 1:
            names = "\n".join(f"  {c.name}" for c in candidates)
            raise ValueError(
                f"{path} holds {len(candidates)} *{OUTPUT_SUFFIX} files; "
                f"pass the one you mean:\n{names}"
            )
        out_nc = candidates[0]
    elif path.name.endswith(OUTPUT_SUFFIX):
        out_nc = path
    else:
        raise ValueError(f"{path} is neither a run directory nor a *{OUTPUT_SUFFIX} file.")

    name = out_nc.name[: -len(OUTPUT_SUFFIX)]
    input_file = out_nc.with_name(f"{name}.in")
    return RunFiles(
        run_dir=out_nc.parent,
        out_nc=out_nc,
        input_file=input_file if input_file.is_file() else None,
    )


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


__all__ = [
    "OUTPUT_SUFFIX",
    "RunFiles",
    "find_run",
    "load_gs2_output",
    "detect_saturation_time",
]
