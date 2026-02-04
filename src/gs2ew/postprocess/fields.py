"""Plot time traces and spectra of fields."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from gs2ew.utils.gs2_output import detect_saturation_time


def plot_fields_time_traces(
    ds: xr.Dataset,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    window: float = 20.0,
    threshold: float = 0.1,
) -> Path:
    """Plot time trace of each available field from (phi2, apar2, bpar2)
    and highlight the saturated region.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file)
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is "outputs".
    filename : str, optional
        Filename for the plot. If None, uses "field_time_traces.png".
    window : float, optional
        Time interval for computing growth rate. Default is 20.0.
    threshold : float, optional
        Growth rate threshold below which saturation is considered to
        have occurred. Default is 0.1.

    Returns
    -------
    Path
        Path to the saved figure file
    """
    # Determine enabled fields
    fields = [f for f in ["phi2", "apar2", "bpar2"] if f in ds]

    # Detect saturation time (just uses phi2)
    tsat = detect_saturation_time(ds, window=window, threshold=threshold)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default filename
    if filename is None:
        filename = "field_time_traces.png"

    # Extract time
    t = ds["t"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each field on log scale
    for field in fields:
        field_trace = ds[field].values
        ax.semilogy(t, field_trace, linewidth=1.5, label=field)

    # Highlight saturated region
    if not np.isnan(tsat):
        ax.axvspan(tsat, t.max(), alpha=0.2, color="green")
        ax.axvline(tsat, color="green", linestyle="--", linewidth=2,
                   label=f"Saturation (t = {tsat:.2f})")

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"field$^2$", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_fields_by_mode(
    ds: xr.Dataset,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
) -> Path:
    """Plot kx-ky spectra of each available field at the last time step.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file)
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is "outputs".
    filename : str, optional
        Filename for the plot. If None, uses "fields_by_mode.png".

    Returns
    -------
    Path
        Path to the saved figure file
    """
    # Determine enabled fields
    all_fields = ["phi2_by_mode", "apar2_by_mode", "bpar2_by_mode"]
    fields = [f for f in all_fields if f in ds]

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default filename
    if filename is None:
        filename = "fields_by_mode.png"

    # Extract kx and ky, shifting kx from FFT layout to monotonic
    kx = np.fft.fftshift(ds["kx"].values)
    ky = ds["ky"].values

    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Labels for each field
    labels = {
        "phi2_by_mode": r"$|\phi|^2$",
        "apar2_by_mode": r"$|A_\parallel|^2$",
        "bpar2_by_mode": r"$|B_\parallel|^2$",
    }

    for ax, field in zip(axes, all_fields):
        if field in fields:
            # Extract last time step and shift kx from FFT layout
            data = ds[field].isel(t=-1).values  # shape: (ky, kx)
            data_shifted = np.fft.fftshift(data, axes=1)

            # Plot using pcolormesh with log scale
            pcm = ax.pcolormesh(kx, ky, data_shifted, norm=LogNorm())
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel(r"$k_x \rho_\text{ref}$")
            ax.set_ylabel(r"$k_y \rho_\text{ref}$")
            ax.set_title(labels[field])
        else:
            ax.set_visible(False) # field not enabled - hide subplot

    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
