"""Template analysis deck for gs2ew.

Copy this file, comment in or out the plots you want, then run:

    uv run examples/analysis.py results.out.nc outputs/

For the standard plot set with no editing, use the CLI instead:

    uv run gs2ew plot results.out.nc -o outputs/ --movies
"""

import argparse

import xarray as xr

from gs2ew import (
    plot_fields_by_mode,
    plot_fields_time_traces,
    plot_phi2_by_theta,
    plot_transfer_butterfly,
    plot_transfer_hovmoller,
    plot_transfer_spectral,
    plot_transfer_spectral_movie,
    plot_transfer_spectrum,
    plot_transfer_summary,
    plot_transfer_theta,
    plot_transfer_theta_movie,
    plot_transfer_trapped_passing,
    plot_transfer_trapped_passing_movie,
    plot_transfer_trapped_passing_spectral,
    plot_transfer_trapped_passing_spectral_movie,
)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run GS2 post-processing analysis.")
parser.add_argument("nc_file", help="Path to the GS2 .out.nc file")
parser.add_argument("output_dir", help="Directory where plots are saved")
args = parser.parse_args()

out = args.output_dir

with xr.open_dataset(args.nc_file) as ds:
    # ── Fields ────────────────────────────────────────────────────────────────

    plot_fields_time_traces(ds, output_dir=out)
    plot_fields_by_mode(ds, output_dir=out)
    plot_phi2_by_theta(ds, output_dir=out)

    # ── Transfer vs theta ─────────────────────────────────────────────────────

    # Averaged from 20% into the saturated interval to the end of the run, with
    # +/-1 std bands showing the temporal spread over that window.
    plot_transfer_theta(ds, tstart="saturation", show_std=True, output_dir=out)

    # `bmag_power=1` cancels the 1/bmag currently carried by GS2's free-energy
    # prefactor and leaves the kinetic-energy transfer alone. `div_by_bmag2=True`
    # is the other side of the same investigation (shorthand for bmag_power=-2).
    # plot_transfer_theta(ds, tstart="saturation", bmag_power=1, output_dir=out)

    # ── Transfer vs wavevector ────────────────────────────────────────────────

    # (theta, k) heatmaps and the theta-integrated 1D spectrum. `reduction` picks
    # which leg of the triad to resolve: kxtarget, kxsource or kysource.
    plot_transfer_spectral(ds, reduction="kxtarget", tstart="saturation",
                           klim=0.6, output_dir=out)
    plot_transfer_spectrum(ds, reduction="kxtarget", tstart="saturation",
                           klim=0.6, output_dir=out)

    # ── Time-resolved (static alternatives to a movie) ────────────────────────

    # The whole run in one figure: (t, theta) and (t, k) heatmaps.
    plot_transfer_hovmoller(ds, output_dir=out)
    plot_transfer_butterfly(ds, reduction="kxtarget", klim=0.6, output_dir=out)

    # ── Trapped vs passing ────────────────────────────────────────────────────

    # One panel per species, trapped and passing on the same axes.
    plot_transfer_trapped_passing(ds, tstart="saturation", show_std=True,
                                  output_dir=out)
    plot_transfer_trapped_passing_spectral(ds, quantity="U", field="phi",
                                           tstart="saturation", klim=0.6,
                                           output_dir=out)

    # ── Net budget ────────────────────────────────────────────────────────────

    plot_transfer_summary(ds, output_dir=out)

    # ── Movies ────────────────────────────────────────────────────────────────
    #
    # `stride` keeps every nth frame, which is worth using while you decide what
    # you want to look at. `normalise="movie"` divides each curve by a single
    # fixed factor (its peak over the whole movie), so small terms like U are
    # scaled into view while relative amplitudes and growth over time survive.

    # plot_transfer_trapped_passing_movie(ds, stride=2, fps=15, output_dir=out)
    # plot_transfer_theta_movie(ds, normalise="movie", stride=2, fps=15, output_dir=out)
    # plot_transfer_trapped_passing_spectral_movie(ds, klim=0.6, stride=2, fps=15,
    #                                              output_dir=out)
    # plot_transfer_spectral_movie(ds, klim=0.6, quantities=["U"], stride=2, fps=15,
    #                              output_dir=out)
