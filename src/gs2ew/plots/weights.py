"""Plot GS2 velocity-space integration weights."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.weights import get_weights


def compute_weights(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    quiet: bool = False,
) -> xr.Dataset:
    """The weights `w` and `wl`, and the energy grid, that the weights plots
    draw. Where they came from is recorded as the ``weights_source``
    attribute.

    Raises
    ------
    MissingWeightsError
        If the weights cannot be found in `ds` or `grids_nc`.
    """
    weights = get_weights(ds, grids_nc=grids_nc)
    source = "GS2 output" if "w" in ds and "wl" in ds else str(Path(grids_nc).resolve())
    return xr.Dataset(
        {"energy": ds["energy"], "w": weights["w"], "wl": weights["wl"]},
        attrs={"weights_source": source},
    )


def render_weights(reduced: xr.Dataset) -> Figure:
    """Draw `w` and `wl` from :func:`compute_weights`."""
    # energy has dims (species, egrid); w has dims (species, egrid)
    energy = reduced["energy"].values
    w = reduced["w"].values
    nspec = w.shape[0]

    # wl has dims (lambda, theta)
    theta = reduced["wl"].coords["theta"].values
    lam = reduced["wl"].coords["lambda"].values
    wl = reduced["wl"].values

    fig, (ax_w, ax_wl) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: energy weights — one curve per species
    for i in range(nspec):
        ax_w.plot(energy[i], w[i], linewidth=1.5, label=f"Species {i + 1}")
    ax_w.set_xlabel(r"$\varepsilon$", fontsize=12)
    ax_w.set_ylabel(r"$w(\varepsilon, s)$", fontsize=12)
    if nspec > 1:
        ax_w.legend()
    ax_w.grid(alpha=0.3)

    # Right: pitch-angle weights — 2D heatmap over (theta, lambda)
    # wl has shape (nlambda, ntheta), so pcolormesh(theta, lambda, wl) is correct
    pcm = ax_wl.pcolormesh(theta, lam, wl, shading="auto")
    fig.colorbar(pcm, ax=ax_wl)
    ax_wl.set_xlabel(r"$\theta$", fontsize=12)
    ax_wl.set_ylabel(r"$\lambda$", fontsize=12)
    ax_wl.set_title(r"$w_\lambda(\lambda, \theta)$")

    plt.tight_layout()
    return fig


def plot_weights(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Plot the velocity-space integration weights `w` (energy) and `wl`
    (pitch angle).

    `w` is plotted as one curve per species against the energy grid.
    `wl` is shown as a 2D heatmap over the (theta, lambda) grid.

    For GS2 runs produced before commit `452f7a8`, `w` and `wl` are not
    written to `.out.nc`. In that case, generate them first with
    `dump_grids` and pass the resulting file via `grids_nc`:

        from gs2ew.weights import run_dump_grids
        grids_nc = run_dump_grids("~/scratch/gs2/bin/dump_grids", "sim.in")
        plot_weights(ds, grids_nc=grids_nc)

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc`).
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Only used
        if `w`/`wl` are absent from `ds`.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is `"outputs"`.
    filename : str, optional
        Filename for the plot. If None, uses `"weights.png"`.
    save_data : bool, optional
        Write the plotted data and its provenance alongside the figure; see
        :mod:`gs2ew.bundle`. Default True.
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    MissingWeightsError
        If the weights cannot be found in `ds` or `grids_nc`.
    """
    return make_plot(
        "weights",
        ds,
        dict(grids_nc=grids_nc),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def render_wl_by_lambda(reduced: xr.Dataset, ncols: int = 6) -> Figure:
    """Draw `wl` against `lambda`, one panel per theta, from
    :func:`compute_weights`."""
    # wl has dims (lambda, theta); wl[:, i] is the lambda-profile at theta[i]
    lam = reduced["wl"].coords["lambda"].values
    theta = reduced["wl"].coords["theta"].values
    wl = reduced["wl"].values
    ntheta = len(theta)

    nrows = int(np.ceil(ntheta / ncols))

    # Fixed width; height scales with the number of rows so each subplot
    # stays a consistent size even for large theta grids.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, nrows * 2.5),
        sharex=True, sharey=True,
    )
    # Ensure axes is always a 2D array for uniform indexing
    axes = np.array(axes).reshape(nrows, ncols)

    for i, ax in enumerate(axes.flat):
        if i < ntheta:
            ax.plot(lam, wl[:, i], "x-", linewidth=0.8, markersize=4,
                    markeredgewidth=0.8)
            ax.set_title(rf"$\theta = {theta[i]:.2f}$", fontsize=8)
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    fig.supxlabel(r"$\lambda$", fontsize=12)
    fig.supylabel(r"$w_\lambda$", fontsize=12)
    plt.tight_layout()
    return fig


def plot_wl_by_lambda(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    ncols: int = 6,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Plot `wl` against `lambda` for each `theta` on a grid of subplots.

    Each subplot corresponds to one theta grid point and shows the pitch-angle
    weight `wl` as a function of `lambda`. Subplots are arranged in a grid
    with `ncols` columns; the figure width is fixed and the height grows with
    the number of rows, so individual subplots remain a consistent size
    regardless of how many theta points there are.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc`).
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Only used
        if `wl` is absent from `ds`.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is `"outputs"`.
    filename : str, optional
        Filename for the plot. If None, uses `"wl_by_lambda.png"`.
    ncols : int, optional
        Number of subplot columns. Default is 6.
    save_data : bool, optional
        Write the plotted data and its provenance alongside the figure; see
        :mod:`gs2ew.bundle`. Default True.
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    MissingWeightsError
        If the weights cannot be found in `ds` or `grids_nc`.
    """
    return make_plot(
        "wl_by_lambda",
        ds,
        dict(grids_nc=grids_nc),
        dict(ncols=ncols),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


register(
    PlotSpec("weights", compute_weights, render_weights, lambda reduced: "weights.png")
)
register(
    PlotSpec(
        "wl_by_lambda",
        compute_weights,
        render_wl_by_lambda,
        lambda reduced: "wl_by_lambda.png",
    )
)
