"""Create various plots of the kinetic energy and entropy transfer."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from gs2ew.utils.gs2_output import detect_saturation_time

def plot_transfer_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
) -> Path:
    """Plots the poloidal structure of each enabled transfer diagnostic.

    By default, plots the last time step. If `window` is provided, the
    diagnostics are averaged over a time window instead.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file)
    window : float, optional
        Duration of the averaging window. If provided without `tstart`,
        averages over the last `window` time units. If provided with
        `tstart`, averages from `tstart` over length `window`.
    tstart : float, optional
        Start time for the averaging window. Requires `window` to be set;
        raises ValueError if `window` is not provided.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is "outputs".
    filename : str, optional
        Filename for the plot. If None, uses "transfer_by_theta.png" (or
        "transfer_by_theta_averaged.png" when averaging).

    Returns
    -------
    Path
        Path to the saved figure file
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    # Determine enabled transfer diagnostics
    all_diags = [
        "kinetic_energy_transfer_theta",
        "entropy_transfer_phi_theta",
        "entropy_transfer_apar_theta",
        "entropy_transfer_bpar_theta",
    ]
    enabled_diagnostics = [d for d in all_diags if d in ds]

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values

    labels = {
        "kinetic_energy_transfer_theta": r"$T_v^\text{ZF}$",
        "entropy_transfer_phi_theta": r"$T_{S,\phi}^\text{ZF}$",
        "entropy_transfer_apar_theta": r"$T_{S,A_\parallel}^\text{ZF}$",
        "entropy_transfer_bpar_theta": r"$T_{S,B_\parallel}^\text{ZF}$"
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window

        if filename is None:
            filename = "transfer_by_theta_averaged.png"

        for diag in enabled_diagnostics:
            transfer_avg = ds[diag].sel(t=slice(tstart, tend)).mean(dim="t").values
            ax.plot(theta, transfer_avg, linewidth=1.5, label=labels[diag])

        ax.set_title(f"Averaged over t = [{tstart:.1f}, {tend:.1f}]")
    else:
        if filename is None:
            filename = "transfer_by_theta.png"

        for diag in enabled_diagnostics:
            transfer = ds[diag].isel(t=-1).values
            ax.plot(theta, transfer, linewidth=1.5, label=labels[diag])

    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("transfer", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
