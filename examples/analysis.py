"""
Template analysis deck for gs2ew.

Copy this file, then run with:

    python analysis.py results.out.nc outputs/

or, if using uv:

    uv run analysis.py results.out.nc outputs/
"""

import argparse

import xarray as xr
from gs2ew.postprocess.fields import (
    plot_fields_by_mode,
    plot_fields_time_traces,
    plot_phi2_by_theta,
    plot_phi2_by_theta_movie,
)
from gs2ew.postprocess.transfer import (
    plot_transfer_by_theta,
    plot_transfer_by_theta_movie,
    plot_vel_transfer_by_theta_by_sign,
    plot_vel_transfer_by_theta_by_sign_movie,
    plot_vel_transfer_theta_lambda,
    plot_vel_transfer_theta_lambda_movie,
)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run GS2 post-processing analysis.")
parser.add_argument("nc_file", help="Path to the GS2 .out.nc file")
parser.add_argument("output_dir", help="Directory where plots are saved")
args = parser.parse_args()

# ── Run ───────────────────────────────────────────────────────────────────────

# Open with dask chunking along the time axis so that windowed averages over
# large velocity-resolved variables are computed one timestep at a time rather
# than loading the full window into memory.
ds = xr.open_dataset(args.nc_file, chunks={"t": 1})

with ds:
    # Field diagnostics
    plot_fields_time_traces(ds, output_dir=args.output_dir)
    plot_fields_by_mode(ds, output_dir=args.output_dir)

    # |phi|^2 vs theta, split into zonal (ky=0) and non-zonal (ky>0) parts
    plot_phi2_by_theta(ds, output_dir=args.output_dir)

    # Movie of |phi|^2 vs theta (needs time-resolved phi_t / write_phi_over_time)
    # plot_phi2_by_theta_movie(ds, output_dir=args.output_dir, fps=15)

    # ── Theta-resolved transfer (kinetic energy + free energy H, entropy S, U) ──

    # Transfer at the last timestep
    plot_transfer_by_theta(ds, output_dir=args.output_dir)

    # Transfer averaged over a time window, with ±1 std bands showing the
    # temporal spread over the window (adjust tstart and window to taste).
    plot_transfer_by_theta(
        ds, window=60.0, show_std=True, normalise=True, div_by_bmag2=True,
        output_dir=args.output_dir,
    )

    # Movie of the transfer over time, starting at t = 60. normalise="movie"
    # divides each curve by a single fixed factor (its peak over the whole
    # timeseries), so small terms like U are scaled into view while the relative
    # amplitudes between curves and their growth/decay over time are preserved.
    plot_transfer_by_theta_movie(
        ds, tstart=60.0, normalise="movie", fps=15, output_dir=args.output_dir
    )

    # ── Velocity-resolved transfer (needs write_*_transfer_velocity output) ──────

    # Velocity-resolved transfer vs theta, split by sign of v_parallel
    # plot_vel_transfer_by_theta_by_sign(ds, output_dir=args.output_dir)
    # plot_vel_transfer_by_theta_by_sign_movie(ds, window=100.0, output_dir=args.output_dir, fps=15)

    # Velocity-resolved transfer as 2D (theta, lambda) heatmaps, one row per drive
    # plot_vel_transfer_theta_lambda(ds, output_dir=args.output_dir)
    # plot_vel_transfer_theta_lambda_movie(ds, window=100.0, output_dir=args.output_dir, fps=15)
