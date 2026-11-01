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
* :mod:`gs2ew.provenance` — code versions, input decks and source identity
* :mod:`gs2ew.bundle` — the reduced data and provenance saved with each plot
* :mod:`gs2ew.style` — how figures look, and the formats they are saved in
* :mod:`gs2ew.plots` — the plotting routines themselves
"""

from gs2ew.bundle import read_bundle, replot, reproduce
from gs2ew.catalog import Drive, available_drives, identity_residual
from gs2ew.geometry import theta_integral
from gs2ew.io import RunFiles, detect_saturation_time, find_run, load_gs2_output
from gs2ew.metadata import get_metadata, get_version, requires_version
from gs2ew.provenance import CodeVersion, gs2_version, gs2ew_version
from gs2ew.plots.fields import (
    compute_fields_by_mode,
    compute_fields_time_traces,
    compute_phi2_by_theta,
    plot_fields_by_mode,
    plot_fields_time_traces,
    plot_phi2_by_theta,
    plot_phi2_by_theta_movie,
    render_fields_by_mode,
    render_fields_time_traces,
    render_phi2_by_theta,
)
from gs2ew.plots.transfer_spectral import (
    compute_transfer_spectral,
    compute_transfer_spectrum,
    plot_transfer_spectral,
    plot_transfer_spectral_movie,
    plot_transfer_spectrum,
    render_transfer_spectral,
    render_transfer_spectrum,
)
from gs2ew.plots.transfer_summary import (
    compute_transfer_summary,
    plot_transfer_summary,
    render_transfer_summary,
)
from gs2ew.plots.transfer_surface import (
    compute_transfer_flux_surface,
    plot_transfer_flux_surface,
    render_transfer_flux_surface,
)
from gs2ew.plots.transfer_theta import (
    compute_transfer_theta,
    plot_transfer_theta,
    plot_transfer_theta_movie,
    render_transfer_theta,
)
from gs2ew.plots.transfer_time import (
    compute_transfer_butterfly,
    compute_transfer_hovmoller,
    plot_transfer_butterfly,
    plot_transfer_hovmoller,
    render_transfer_butterfly,
    render_transfer_hovmoller,
)
from gs2ew.plots.transfer_tp import (
    compute_transfer_trapped_passing,
    compute_transfer_trapped_passing_spectral,
    plot_transfer_trapped_passing,
    plot_transfer_trapped_passing_movie,
    plot_transfer_trapped_passing_spectral,
    plot_transfer_trapped_passing_spectral_movie,
    render_transfer_trapped_passing,
    render_transfer_trapped_passing_spectral,
)
from gs2ew.plots.weights import (
    compute_weights,
    plot_weights,
    plot_wl_by_lambda,
    render_weights,
    render_wl_by_lambda,
)
from gs2ew.style import FigureOptions, figure_options
from gs2ew.timebase import TimeWindow, frame_indices, resolve_window

__all__ = [
    # Core
    "CodeVersion",
    "Drive",
    "FigureOptions",
    "RunFiles",
    "TimeWindow",
    "available_drives",
    "detect_saturation_time",
    "figure_options",
    "find_run",
    "frame_indices",
    "get_metadata",
    "get_version",
    "gs2_version",
    "gs2ew_version",
    "identity_residual",
    "load_gs2_output",
    "read_bundle",
    "replot",
    "reproduce",
    "requires_version",
    "resolve_window",
    "theta_integral",
    # Fields
    "compute_fields_by_mode",
    "compute_fields_time_traces",
    "compute_phi2_by_theta",
    "plot_fields_by_mode",
    "plot_fields_time_traces",
    "plot_phi2_by_theta",
    "plot_phi2_by_theta_movie",
    "render_fields_by_mode",
    "render_fields_time_traces",
    "render_phi2_by_theta",
    # Transfer
    "compute_transfer_butterfly",
    "compute_transfer_flux_surface",
    "compute_transfer_hovmoller",
    "compute_transfer_spectral",
    "compute_transfer_spectrum",
    "compute_transfer_summary",
    "compute_transfer_theta",
    "compute_transfer_trapped_passing",
    "compute_transfer_trapped_passing_spectral",
    "plot_transfer_butterfly",
    "plot_transfer_flux_surface",
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
    "render_transfer_butterfly",
    "render_transfer_flux_surface",
    "render_transfer_hovmoller",
    "render_transfer_spectral",
    "render_transfer_spectrum",
    "render_transfer_summary",
    "render_transfer_theta",
    "render_transfer_trapped_passing",
    "render_transfer_trapped_passing_spectral",
    # Weights
    "compute_weights",
    "plot_weights",
    "plot_wl_by_lambda",
    "render_weights",
    "render_wl_by_lambda",
]
