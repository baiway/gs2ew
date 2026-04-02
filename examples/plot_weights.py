"""Example: plot velocity-space integration weights.

For GS2 runs produced before commit 452f7a8, w and wl are not written to
.out.nc. In that case they must be generated first with dump_grids:

    dump_grids <run_name>.in

or via run_dump_grids() below.
"""

import xarray as xr
from gs2ew.postprocess.weights import plot_weights
from gs2ew.utils.weights import run_dump_grids

NC_FILE    = "results.out.nc"
GRIDS_FILE = "results.grids.nc"
OUTPUT_DIR = "outputs"

# If dump_grids has not been run yet, uncomment the following to generate
# GRIDS_FILE automatically. Requires a compiled dump_grids binary.
#
# GRIDS_FILE = run_dump_grids(
#     executable="/path/to/gs2/bin/dump_grids",
#     input_file="/path/to/input.in",
# )

with xr.open_dataset(NC_FILE) as ds:
    # For new GS2 (>= commit 452f7a8), w and wl are in ds — omit grids_nc:
    #   plot_weights(ds, output_dir=OUTPUT_DIR)
    #
    # For older runs, pass the .grids.nc file produced by dump_grids:
    plot_weights(ds, grids_nc=GRIDS_FILE, output_dir=OUTPUT_DIR)
