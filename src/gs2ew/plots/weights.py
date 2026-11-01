"""Plot GS2 velocity-space integration weights."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.plots import _common as common
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

    fig, axes = common.subplots(1, 2, aspect=0.8)
    ax_w, ax_wl = axes[0]

    # Left: energy weights — one curve per species
    for i in range(nspec):
        ax_w.plot(energy[i], w[i], label=f"species {i + 1}")
    ax_w.set_xlabel(r"$\varepsilon$")
    ax_w.set_ylabel(r"$w(\varepsilon, s)$")
    if nspec > 1:
        ax_w.legend()

    # Right: pitch-angle weights — 2D heatmap over (theta, lambda)
    # wl has shape (nlambda, ntheta), so pcolormesh(theta, lambda, wl) is correct.
    # The weights are positive, so a sequential map rather than the style's
    # diverging one.
    pcm = ax_wl.pcolormesh(
        theta, lam, wl, shading="auto", cmap="viridis", rasterized=True
    )
    fig.colorbar(pcm, ax=ax_wl)
    ax_wl.grid(False)
    ax_wl.set_xlabel(r"$\theta$")
    ax_wl.set_ylabel(r"$\lambda$")
    ax_wl.set_title(r"$w_\lambda(\lambda, \theta)$")
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
        Filename for the plot. If None, uses `"weights"`.
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

    # The style's width; height scales with the number of rows so each
    # subplot stays a consistent size even for large theta grids.
    fig, axes = common.subplots(nrows, ncols, aspect=0.8, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
        if i < ntheta:
            ax.plot(lam, wl[:, i], "x-", linewidth=0.8, markeredgewidth=0.8)
            ax.set_title(rf"$\theta = {theta[i]:.2f}$", fontsize="small")
        else:
            ax.set_visible(False)

    fig.supxlabel(r"$\lambda$")
    fig.supylabel(r"$w_\lambda$")
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
        Filename for the plot. If None, uses `"wl_by_lambda"`.
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
    PlotSpec("weights", compute_weights, render_weights, lambda reduced: "weights")
)
register(
    PlotSpec(
        "wl_by_lambda",
        compute_weights,
        render_wl_by_lambda,
        lambda reduced: "wl_by_lambda",
    )
)
