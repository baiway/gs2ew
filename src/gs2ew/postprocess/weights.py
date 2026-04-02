"""Plot GS2 velocity-space integration weights."""

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from gs2ew.utils.weights import get_weights


def plot_weights(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
) -> Path:
    """Plot the velocity-space integration weights `w` (energy) and `wl`
    (pitch angle).

    `w` is plotted as one curve per species against the energy grid.
    `wl` is shown as a 2D heatmap over the (theta, lambda) grid.

    For GS2 runs produced before commit `452f7a8`, `w` and `wl` are not
    written to `.out.nc`. In that case, generate them first with
    `dump_grids` and pass the resulting file via `grids_nc`:

        from gs2ew.utils.weights import run_dump_grids
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

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    MissingWeightsError
        If the weights cannot be found in `ds` or `grids_nc`.
    """
    weights = get_weights(ds, grids_nc=grids_nc)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "weights.png"

    # energy has dims (species, egrid); w has dims (species, egrid)
    energy = ds["energy"].values
    w = weights["w"].values
    nspec = w.shape[0]

    # wl has dims (lambda, theta)
    theta = weights["wl"].coords["theta"].values
    lam = weights["wl"].coords["lambda"].values
    wl = weights["wl"].values

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
    ax_wl.set_title("w_l(\lambda, \theta)")

    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")
    return output_path
