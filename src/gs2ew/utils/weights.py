"""Utilities for loading GS2 velocity-space integration weights."""

import subprocess
from pathlib import Path

import xarray as xr


class MissingWeightsError(RuntimeError):
    """Raised when `w` and `wl` are absent from the dataset and no
    `grids_nc` fallback was provided.

    These variables were added to GS2's `.out.nc` output at commit
    `452f7a8`. For runs produced by older versions, generate them with the
    `dump_grids` program bundled with GS2::

        dump_grids <run_name>.in

    This writes `<run_name>.grids.nc`, which can then be passed to any
    function that accepts a `grids_nc` argument.
    """


def get_weights(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
) -> xr.Dataset:
    """Return a dataset containing the velocity weights `w` and `wl`.

    Checks `ds` first (GS2 >= commit `452f7a8`). If the variables are
    absent, falls back to loading them from `grids_nc` (produced by
    `dump_grids`). Raises :exc:`MissingWeightsError` if neither source is
    available.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc`).
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Only
        consulted if `w`/`wl` are absent from `ds`.

    Returns
    -------
    xarray.Dataset
        Dataset containing at least `w` and `wl`, with their coordinates.

    Raises
    ------
    MissingWeightsError
        If the weights cannot be found in `ds` or `grids_nc`.
    FileNotFoundError
        If `grids_nc` is provided but does not exist.
    """
    if "w" in ds and "wl" in ds:
        return ds[["w", "wl"]]

    if grids_nc is not None:
        grids_nc = Path(grids_nc)
        if not grids_nc.exists():
            raise FileNotFoundError(f"grids file not found: {grids_nc}")
        grids = xr.open_dataset(grids_nc)
        if "w" not in grids or "wl" not in grids:
            raise MissingWeightsError(
                f"'w' and/or 'wl' not found in {grids_nc}. "
                "Ensure the file was produced by 'dump_grids'."
            )
        # Load into memory before closing the file
        return grids[["w", "wl"]].load()

    # Neither source available — build a maximally helpful error message
    nc_path = ds.encoding.get("source", "")
    run_stem = Path(nc_path).name.replace(".out.nc", "") if nc_path else "<run_name>"

    raise MissingWeightsError(
        "'w' and 'wl' are not present in this dataset. They were added to "
        "GS2's .out.nc output at commit 452f7a8. For older runs, generate "
        "them by running:\n\n"
        f"    dump_grids {run_stem}.in\n\n"
        f"This produces '{run_stem}.grids.nc'. Then pass it to the plot "
        "function:\n\n"
        f"    plot_weights(ds, grids_nc='{run_stem}.grids.nc')\n\n"
        "If you need to run dump_grids from Python, use "
        "gs2ew.utils.weights.run_dump_grids()."
    )


def run_dump_grids(
    executable: str | Path,
    input_file: str | Path,
) -> Path:
    """Run the GS2 `dump_grids` program and return the path to the output
    `.grids.nc` file.

    `dump_grids` is a program bundled with GS2 that reads a `.in` input
    file and writes grid quantities (including the velocity weights `w` and
    `wl`) to `<run_name>.grids.nc`.

    Parameters
    ----------
    executable : str or Path
        Path to the compiled `dump_grids` binary
        (e.g. `~/scratch/gs2/bin/dump_grids`).
    input_file : str or Path
        Path to the GS2 `.in` input file for the run.

    Returns
    -------
    Path
        Path to the generated `<run_name>.grids.nc` file.

    Raises
    ------
    FileNotFoundError
        If `executable` or `input_file` do not exist.
    RuntimeError
        If `dump_grids` exits with a non-zero return code, or if the
        expected output file is not produced.
    """
    executable = Path(executable).expanduser().resolve()
    input_file = Path(input_file).expanduser().resolve()

    if not executable.exists():
        raise FileNotFoundError(f"dump_grids executable not found: {executable}")
    if not input_file.exists():
        raise FileNotFoundError(f"GS2 input file not found: {input_file}")

    # dump_grids writes output to the current directory using the run name
    # derived from the input file stem, so run it from the input file's directory.
    expected_output = input_file.parent / (input_file.stem + ".grids.nc")

    result = subprocess.run(
        [str(executable), str(input_file)],
        cwd=str(input_file.parent),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"dump_grids failed (exit code {result.returncode}):\n{result.stderr}"
        )

    if not expected_output.exists():
        raise RuntimeError(
            f"dump_grids exited successfully but expected output not found: "
            f"{expected_output}\nstdout:\n{result.stdout}"
        )

    return expected_output
