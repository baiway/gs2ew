"""
Template analysis deck for gs2ew.

Copy this file, fill in the settings below, then run with:

    python analysis.py

or, if using uv:

    uv run analysis.py
"""

import xarray as xr
from gs2ew.postprocess.fields import plot_fields_by_mode, plot_fields_time_traces
from gs2ew.postprocess.transfer import plot_transfer_by_theta, plot_transfer_by_theta_movie

# ── Settings ──────────────────────────────────────────────────────────────────

NC_FILE = "results.out.nc"   # path to GS2 .out.nc file
OUTPUT_DIR = "outputs"          # directory where plots are saved

# ── Run ───────────────────────────────────────────────────────────────────────

# Open lazily: data is read from disk only as each plot needs it,
# keeping memory usage bounded for large (tens of GB) files.
ds = xr.open_dataset(NC_FILE)

with ds:
    # Field diagnostics
    plot_fields_time_traces(ds, output_dir=OUTPUT_DIR)
    plot_fields_by_mode(ds, output_dir=OUTPUT_DIR)

    # Transfer at the last timestep
    plot_transfer_by_theta(ds, output_dir=OUTPUT_DIR)

    # Transfer averaged over a time window (adjust tstart and window to taste)
    plot_transfer_by_theta(ds, window=100.0, output_dir=OUTPUT_DIR)

    # Movie of the rolling-averaged transfer
    plot_transfer_by_theta_movie(ds, window=100.0, fps=15, output_dir=OUTPUT_DIR)
