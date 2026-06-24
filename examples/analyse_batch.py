"""Batch post-processing some simulations on ARCHER2.

Walks a project directory, finds every `*.out.nc` file (the `cbc` case
is skipped, as it has already been analysed), and for each one produces:

1. field time traces             — ``plot_fields_time_traces``
2. fields by mode                — ``plot_fields_by_mode``
3. theta-resolved transfer       — ``plot_transfer_by_theta`` averaged from the
                                    detected saturation time to the end of the
                                    run
4. theta-resolved transfer movie — `plot_transfer_by_theta_movie`

Outputs are written under `<output_root>/<case>`, mirroring each run's path
relative to the project root. This keeps the differently-located runs that
share an input-file name (the two ``kappa2p0`` cases, the six ``tri`` cases)
from colliding.

Usage
-----
    python analyse_batch.py [project_root] [output_root]

Both arguments are optional and default to the ARCHER2 layout below.

Can be called using the script below:
-------------------------------------

```
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=analysis
#SBATCH --output=analysis%j.log
#SBATCH --time=2:00:00                  # Time limit (HH:MM:SS). Movies dominate
                                        # the runtime; lower this once you know
                                        # how long the batch actually takes.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# `e281` is the Plasma Physics Consortium project code
#SBATCH --account=e281
#SBATCH --partition=serial
#SBATCH --qos=serial

# Paths
VENV_PATH="/work/e281/e281/bc1264/gk/gs2ew/.venv"
SCRIPT_PATH="/work/e281/e281/bc1264/gk/gs2ew/examples/analyse_batch.py"
PROJECT_ROOT="/work/e281/e281/bc1264/tobias-reproduce-v2"
OUTPUT_ROOT="${PROJECT_ROOT}/analysis"

# Activate Python virtual environment
source "${VENV_PATH}/bin/activate"

# Analyse every .out.nc under PROJECT_ROOT (cbc is skipped by the script),
# writing per-case outputs under OUTPUT_ROOT mirroring the run layout.
python "${SCRIPT_PATH}" "${PROJECT_ROOT}" "${OUTPUT_ROOT}"
```
"""

import sys
from pathlib import Path

# Use a non-interactive backend: this runs headless on a compute node. Must be
# set before any matplotlib.pyplot import, including those inside gs2ew.
import matplotlib

matplotlib.use("Agg")

import xarray as xr

from gs2ew.postprocess.fields import (
    plot_fields_by_mode,
    plot_fields_time_traces,
)
from gs2ew.postprocess.transfer import (
    plot_transfer_by_theta,
    plot_transfer_by_theta_movie,
)

# Defaults for the ARCHER2 layout; override on the command line if needed.
DEFAULT_ROOT = Path("/work/e281/e281/bc1264/tobias-reproduce-v2")

# Case directories to skip (already analysed).
SKIP_DIRS = {"cbc"}


def find_runs(root: Path) -> list[Path]:
    """Return every GS2 ``.out.nc`` under `root`, skipping `SKIP_DIRS`."""
    runs = []
    for nc in sorted(root.rglob("*.out.nc")):
        if any(part in SKIP_DIRS for part in nc.relative_to(root).parts):
            continue
        runs.append(nc)
    return runs


def analyse(nc_file: Path, output_dir: Path) -> None:
    """Run all diagnostics for a single ``.out.nc`` file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open lazily (chunked along t) so large velocity-resolved variables are
    # read one timestep at a time rather than loaded whole into memory.
    with xr.open_dataset(nc_file, chunks={"t": 1}) as ds:
        plot_fields_time_traces(ds, output_dir=output_dir)
        plot_fields_by_mode(ds, output_dir=output_dir)
        plot_transfer_by_theta(
            ds, tstart="saturation", show_std=True, normalise=True,
            output_dir=output_dir,
        )
        plot_transfer_by_theta_movie(ds, normalise="movie", output_dir=output_dir)


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT_ROOT
    output_root = Path(argv[2]) if len(argv) > 2 else root / "analysis"

    runs = find_runs(root)
    if not runs:
        print(f"No .out.nc files found under {root}")
        return 1

    print(f"Found {len(runs)} run(s) under {root}:")
    for nc in runs:
        print(f"  {nc.relative_to(root)}")

    failures = []
    for nc in runs:
        output_dir = output_root / nc.parent.relative_to(root)
        print(f"\n=== Analysing {nc.relative_to(root)} -> {output_dir} ===", flush=True)
        try:
            analyse(nc, output_dir)
        except Exception as exc:
            # One bad run shouldn't abort the whole batch; record and move on.
            print(f"  ERROR analysing {nc}: {exc!r}", flush=True)
            failures.append(nc)

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s):")
        for nc in failures:
            print(f"  {nc.relative_to(root)}")
        return 1

    print("\nAll runs analysed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
