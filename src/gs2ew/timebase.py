"""Resolving time windows and movie frames.

Every plot in `gs2ew` accepts the same `tstart`/`window` pair, and every movie
steps over the same kind of frame list. This module is the one implementation
of both, so the semantics cannot drift between a plot and the movie built from
it.

Window semantics, given `tstart` and `window`:

===================  ==================  ====================================
`tstart`             `window`            Result
===================  ==================  ====================================
None                 None                the last timestep, no averaging
None                 given               average over ``[t[-1] - window, t[-1]]``
given                None                average over ``[tstart, t[-1]]``
given                given               average over ``[tstart, tstart + window]``
``"saturation"``     either              as above, with `tstart` detected
===================  ==================  ====================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from gs2ew.io import detect_saturation_time

#: Fraction of the post-saturation interval skipped before averaging begins.
#: The detected saturation time marks where growth *stops*, which is typically
#: still too early to average from: the run is often still settling. Starting a
#: fifth of the way into the saturated interval puts the window well inside it.
DEFAULT_SATURATION_PAD = 0.20

# The saturation time is re-derived by every plot in a run. Reporting it each
# time buries the useful output, so an identical message is only printed once.
_last_reported: tuple | None = None


@dataclass(frozen=True)
class TimeWindow:
    """A resolved time selection.

    `averaging` distinguishes a genuine window from the single-timestep case,
    where taking a standard deviation or a mean is meaningless.
    """

    tstart: float
    tend: float
    averaging: bool
    title: str | None = None

    def mean(self, da: xr.DataArray) -> xr.DataArray:
        """Window average of `da`, or its single timestep when not averaging."""
        if not self.averaging:
            return da.isel(t=-1) if "t" in da.dims else da
        return da.sel(t=slice(self.tstart, self.tend)).mean(dim="t")

    def std(self, da: xr.DataArray) -> xr.DataArray | None:
        """Standard deviation over the window, or None when not averaging."""
        if not self.averaging:
            return None
        return da.sel(t=slice(self.tstart, self.tend)).std(dim="t")

    def attrs(self) -> dict[str, float | bool]:
        """The window as attributes, for the reduced data of a plot."""
        return {"tstart": self.tstart, "tend": self.tend, "averaging": self.averaging}

    @classmethod
    def from_attrs(cls, attrs: dict) -> TimeWindow:
        """Rebuild a window from :meth:`attrs`, as read back from a bundle.

        netCDF has no boolean type, so `averaging` may come back as
        ``"true"``/``"false"``.
        """
        averaging = attrs["averaging"]
        if isinstance(averaging, str):
            averaging = averaging == "true"
        tstart, tend = float(attrs["tstart"]), float(attrs["tend"])
        return cls(
            tstart=tstart,
            tend=tend,
            averaging=bool(averaging),
            title=_window_title(tstart, tend) if averaging else None,
        )


def _window_title(tstart: float, tend: float) -> str:
    return f"Averaged over t = [{tstart:.1f}, {tend:.1f}]"


def resolve_saturation_start(
    ds: xr.Dataset,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> float:
    """Detected saturation time, padded into the saturated interval.

    Returns ``tsat + saturation_pad * (t[-1] - tsat)``. Falls back to the first
    timestep, with a warning, if saturation cannot be detected (for example
    when `phi2` is absent, or the run never saturated).
    """
    t_values = np.asarray(ds["t"].values, dtype=float)

    global _last_reported

    tsat = detect_saturation_time(ds)
    if np.isnan(tsat):
        message = (
            "Warning: saturation time could not be detected; "
            "starting from the beginning of the run."
        )
        result = float(t_values[0])
    else:
        result = float(tsat + saturation_pad * (float(t_values[-1]) - tsat))
        message = (
            f"Detected saturation time tsat = {tsat:.2f}; starting from "
            f"tstart = {result:.2f} (saturation_pad = {saturation_pad:g})."
        )

    if not quiet and message != _last_reported:
        print(message)
        _last_reported = message

    return result


def resolve_window(
    ds: xr.Dataset,
    tstart: float | str | None = None,
    window: float | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> TimeWindow:
    """Resolve `tstart`/`window` into a :class:`TimeWindow`.

    See the module docstring for the full table of cases. `tstart` may be the
    string ``"saturation"`` to start from the detected saturation time, padded
    by `saturation_pad`.

    Raises
    ------
    ValueError
        If `tstart` is an unrecognised string.
    """
    t_values = np.asarray(ds["t"].values, dtype=float)

    if isinstance(tstart, str):
        if tstart != "saturation":
            raise ValueError(
                f"Unknown tstart {tstart!r}; expected a float or 'saturation'."
            )
        tstart = resolve_saturation_start(ds, saturation_pad, quiet=quiet)

    # A single timestep is only implied when neither bound was requested.
    if tstart is None and window is None:
        return TimeWindow(
            tstart=float(t_values[-1]),
            tend=float(t_values[-1]),
            averaging=False,
            title=None,
        )

    if tstart is None:
        tstart = float(t_values[-1]) - window
    tend = tstart + window if window is not None else float(t_values[-1])

    return TimeWindow(
        tstart=float(tstart),
        tend=float(tend),
        averaging=True,
        title=_window_title(tstart, tend),
    )


def frame_indices(
    ds: xr.Dataset,
    tstart: float | str | None = None,
    window: float | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    stride: int = 1,
    quiet: bool = False,
) -> np.ndarray:
    """Time indices to render as movie frames.

    Unlike :func:`resolve_window`, a `tstart` of None means "start at
    saturation" rather than "use the last timestep" — a movie of a single frame
    is not useful. Frames before `tstart` are dropped, and when `window` is
    given so are any frames whose full window ``[t, t + window]`` would run off
    the end of the data.

    `stride` keeps every nth surviving frame, for a quicker preview of a long
    run.

    Raises
    ------
    ValueError
        If no frames survive the filtering.
    """
    t_values = np.asarray(ds["t"].values, dtype=float)

    if tstart is None or (isinstance(tstart, str) and tstart == "saturation"):
        tstart = resolve_saturation_start(ds, saturation_pad, quiet=quiet)
    elif isinstance(tstart, str):
        raise ValueError(
            f"Unknown tstart {tstart!r}; expected a float, 'saturation' or None."
        )

    mask = t_values >= tstart

    # Partial windows at the end of the run would average over fewer timesteps
    # than the rest of the movie, so exclude them rather than silently
    # shortening the window.
    if window is not None:
        mask &= t_values + window <= float(t_values[-1])

    indices = np.nonzero(mask)[0][::stride]

    if len(indices) == 0:
        raise ValueError(
            f"No frames left after filtering (tstart = {tstart:.3g}, "
            f"window = {window}); the run spans t = [{t_values[0]:.3g}, "
            f"{t_values[-1]:.3g}]."
        )

    return indices


__all__ = [
    "DEFAULT_SATURATION_PAD",
    "TimeWindow",
    "resolve_saturation_start",
    "resolve_window",
    "frame_indices",
]
