"""
Template analysis deck for gs2ew.

Copy this file, then run with:

    python analysis.py results.out.nc outputs/

or, if using uv:

    uv run analysis.py results.out.nc outputs/
"""

import argparse

import xarray as xr
from gs2ew.postprocess.fields import plot_fields_by_mode, plot_fields_time_traces
from gs2ew.postprocess.transfer import plot_transfer_by_theta, plot_transfer_by_theta_movie, plot_vel_transfer_by_theta_by_sign_movie, plot_vel_transfer_theta_lambda_movie

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run GS2 post-processing analysis.")
parser.add_argument("nc_file", help="Path to the GS2 .out.nc file")
parser.add_argument("output_dir", help="Directory where plots are saved")
args = parser.parse_args()

# ── Run ───────────────────────────────────────────────────────────────────────

# Open lazily: data is read from disk only as each plot needs it,
# keeping memory usage bounded for large (tens of GB) files.
ds = xr.open_dataset(args.nc_file)

with ds:
    # Field diagnostics
    plot_fields_time_traces(ds, output_dir=args.output_dir)
    plot_fields_by_mode(ds, output_dir=args.output_dir)

    # Transfer at the last timestep
    plot_transfer_by_theta(ds, output_dir=args.output_dir)

    # Transfer averaged over a time window (adjust tstart and window to taste)
    plot_transfer_by_theta(ds, window=100.0, output_dir=args.output_dir)

    # Movie of the rolling-averaged transfer
    plot_transfer_by_theta_movie(ds, window=100.0, fps=15, output_dir=args.output_dir)

    # Plot velocity-resolved transfer by theta by sign
    plot_vel_transfer_by_theta_by_sign_movie(ds, window=100.0, output_dir=args.output_dir, fps=15)

    # Plot velocity-resolved transfer by theta and lambda (separate plots for each sign of
    # vpa)
    plot_vel_transfer_theta_lambda_movie(ds, window=100.0, output_dir=args.output_dir, fps=15)
