"""Post-processing for the gyrokinetic code GS2.

The plotting entry points are re-exported here, so::

    import xarray as xr
    from gs2ew import plot_transfer_theta

    with xr.open_dataset("run.out.nc") as ds:
        plot_transfer_theta(ds, tstart="saturation", show_std=True)

Layout:

* :mod:`gs2ew.catalog` — which transfer variables GS2 wrote and what to call them
* :mod:`gs2ew.timebase` — resolving time windows and movie frames
* :mod:`gs2ew.geometry` — flux-surface-weighted theta integrals
* :mod:`gs2ew.movie` — the frame-loop and ffmpeg driver
* :mod:`gs2ew.io`, :mod:`gs2ew.metadata`, :mod:`gs2ew.weights` — reading files
* :mod:`gs2ew.plots` — the plotting routines themselves
"""

from gs2ew.catalog import Drive, available_drives, identity_residual
from gs2ew.geometry import theta_integral
from gs2ew.io import detect_saturation_time, load_gs2_output
from gs2ew.metadata import get_metadata, get_version, requires_version
from gs2ew.plots.fields import (
    plot_fields_by_mode,
    plot_fields_time_traces,
    plot_phi2_by_theta,
    plot_phi2_by_theta_movie,
)
from gs2ew.plots.transfer_spectral import (
    plot_transfer_spectral,
    plot_transfer_spectral_movie,
    plot_transfer_spectrum,
)
from gs2ew.plots.transfer_summary import plot_transfer_summary
from gs2ew.plots.transfer_theta import plot_transfer_theta, plot_transfer_theta_movie
from gs2ew.plots.transfer_time import plot_transfer_butterfly, plot_transfer_hovmoller
from gs2ew.plots.transfer_tp import (
    plot_transfer_trapped_passing,
    plot_transfer_trapped_passing_movie,
    plot_transfer_trapped_passing_spectral,
    plot_transfer_trapped_passing_spectral_movie,
)
from gs2ew.plots.weights import plot_weights, plot_wl_by_lambda
from gs2ew.timebase import TimeWindow, frame_indices, resolve_window

__all__ = [
    # Core
    "Drive",
    "TimeWindow",
    "available_drives",
    "detect_saturation_time",
    "frame_indices",
    "get_metadata",
    "get_version",
    "identity_residual",
    "load_gs2_output",
    "requires_version",
    "resolve_window",
    "theta_integral",
    # Fields
    "plot_fields_by_mode",
    "plot_fields_time_traces",
    "plot_phi2_by_theta",
    "plot_phi2_by_theta_movie",
    # Transfer
    "plot_transfer_butterfly",
    "plot_transfer_hovmoller",
    "plot_transfer_spectral",
    "plot_transfer_spectral_movie",
    "plot_transfer_spectrum",
    "plot_transfer_summary",
    "plot_transfer_theta",
    "plot_transfer_theta_movie",
    "plot_transfer_trapped_passing",
    "plot_transfer_trapped_passing_movie",
    "plot_transfer_trapped_passing_spectral",
    "plot_transfer_trapped_passing_spectral_movie",
    # Weights
    "plot_weights",
    "plot_wl_by_lambda",
]
