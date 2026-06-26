"""Plot time traces and spectra of fields."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from gs2ew.utils.gs2_output import detect_saturation_time
from gs2ew.postprocess.transfer import _resolve_frame_indices, _stitch_frames_to_movie


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

    print(f"Saved {output_path}")

    return output_path


def _get_phi_complex(ds: xr.Dataset) -> xr.DataArray:
    """Return the complex electrostatic potential.

    Prefers the time-resolved ``phi_t`` (written by GS2's
    ``write_phi_over_time``) over the single-snapshot ``phi``, so callers that
    need a time axis get one when it is available.

    Raises
    ------
    KeyError
        If neither ``phi_t`` nor ``phi`` is present.
    """
    if "phi_t" in ds:
        return ds["phi_t"]
    if "phi" in ds:
        return ds["phi"]
    raise KeyError(
        "No electrostatic potential field found: expected 'phi_t' "
        "(time-resolved) or 'phi' in the dataset."
    )


def _phi2_zonal_nonzonal(phi: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Split ``|phi|^2`` into its zonal and non-zonal parts as functions of theta.

    `phi` is the complex potential with a trailing real/imaginary ``ri``
    dimension plus ``ky``, ``kx`` and ``theta`` (and optionally ``t``).
    ``|phi|^2`` is summed over ``kx``; the zonal part is the ``ky = 0``
    component and the non-zonal part is the sum over all ``ky > 0`` modes.

    Note: only ``ky >= 0`` modes are stored, so the non-zonal sum counts each
    mode once (no factor of two for the conjugate ``-ky`` modes).

    Returns
    -------
    tuple of xarray.DataArray
        ``(zonal, non_zonal)``, each over ``theta`` (and ``t`` if `phi` has it).
    """
    # |phi|^2 = Re^2 + Im^2 over (..., ky, kx, theta)
    phi2 = phi.isel(ri=0) ** 2 + phi.isel(ri=1) ** 2
    ky = phi2["ky"]
    zonal = phi2.sel(ky=0.0).sum("kx")
    non_zonal = phi2.sel(ky=ky.where(ky > 0, drop=True)).sum(("ky", "kx"))
    return zonal, non_zonal


def _global_phi2_ylim(
    ds: xr.Dataset,
    frame_indices: np.ndarray,
) -> tuple[float, float]:
    """Return a padded (0, ymax) spanning the zonal and non-zonal phi2 curves
    over the given frames, so a movie's y-axis stays fixed across frames."""
    zonal, non_zonal = _phi2_zonal_nonzonal(_get_phi_complex(ds))
    z = zonal.isel(t=frame_indices).values
    n = non_zonal.isel(t=frame_indices).values
    hi = float(max(np.nanmax(z), np.nanmax(n)))
    pad = 0.05 * hi if hi > 0 else 1.0
    return 0.0, hi + pad


def plot_phi2_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    _quiet: bool = False,
) -> Path:
    """Plot the poloidal structure of ``|phi|^2``, split into its zonal
    (``ky = 0``) and non-zonal (``ky > 0``) parts.

    ``|phi|^2`` is computed from the complex potential and summed over ``kx``.
    If a time-resolved potential (``phi_t``) is available the last time step is
    plotted by default, or a window average if `window` is given; with only the
    single-snapshot ``phi`` the snapshot is plotted and `window`/`tstart` are
    not permitted.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file).
    window : float, optional
        Duration of the averaging window. If provided without `tstart`,
        averages over the last `window` time units. Requires a time-resolved
        potential.
    tstart : float, optional
        Start time for the averaging window. Requires `window` to be set.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is "outputs".
    filename : str, optional
        Filename for the plot. If None, uses "phi2_by_theta.png" (or
        "phi2_by_theta_averaged.png" when averaging).
    ylim : tuple of float, optional
        Fixed (ymin, ymax) for the y-axis. Used by the movie helper to keep the
        axis steady across frames; if None, matplotlib autoscales.
    tight_bbox : bool, optional
        If True (default), save with ``bbox_inches="tight"``. Movie frames pass
        False so every frame has identical pixel dimensions (required by the
        ffmpeg encoder).

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If `tstart` is given without `window`, or `window` is requested but the
        potential has no time dimension.
    KeyError
        If no potential field is present in `ds`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    zonal, non_zonal = _phi2_zonal_nonzonal(_get_phi_complex(ds))
    has_t = "t" in zonal.dims

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values

    if window is not None:
        if not has_t:
            raise ValueError(
                "Time averaging was requested but the potential has no time "
                "dimension; a time-resolved 'phi_t' (write_phi_over_time) is "
                "required."
            )
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window
        zonal = zonal.sel(t=slice(tstart, tend)).mean("t")
        non_zonal = non_zonal.sel(t=slice(tstart, tend)).mean("t")
        title = f"Averaged over t = [{tstart:.1f}, {tend:.1f}]"
        if filename is None:
            filename = "phi2_by_theta_averaged.png"
    else:
        if has_t:
            zonal = zonal.isel(t=-1)
            non_zonal = non_zonal.isel(t=-1)
        title = None
        if filename is None:
            filename = "phi2_by_theta.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, zonal.values, linewidth=1.5, label=r"zonal ($k_y = 0$)")
    ax.plot(theta, non_zonal.values, linewidth=1.5, linestyle="--",
            label=r"non-zonal ($k_y > 0$)")

    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$\sum_{k_x} |\phi|^2$", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)

    if not _quiet:
        print(f"Saved {output_path}")
    return output_path


def plot_phi2_by_theta_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Create a movie of the poloidal structure of ``|phi|^2`` (zonal and
    non-zonal parts) over time.

    Frames are saved individually to a subdirectory of `output_dir`, then
    stitched into a video. The y-axis is fixed across all frames so the curves
    do not rescale frame to frame.

    Requires a time-resolved potential ``phi_t`` (GS2's ``write_phi_over_time``);
    the single-snapshot ``phi`` has no time axis and cannot be animated.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset containing ``phi_t``. Should be opened lazily
        (``xr.open_dataset``) to keep memory bounded on large runs.
    window : float, optional
        Averaging window duration. If provided, each frame shows the rolling
        average over `window` time units. If omitted, frames are instantaneous.
    tstart : float, optional
        Start time for the movie; frames before this are skipped. Detected
        automatically via ``detect_saturation_time`` if not provided.
    output_dir : str or Path, optional
        Directory where the movie and frame images are saved. Default "outputs".
    filename : str, optional
        Filename for the output movie. If None, uses "phi2_by_theta_movie.mp4".
    fps : int, optional
        Frames per second. Default is 10.
    crf : int, optional
        libx264 Constant Rate Factor (0 = lossless, 51 = worst). Default 18.
    verbose : bool, optional
        If True, prints frame count and per-frame progress. Default is False.

    Returns
    -------
    Path
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If the potential has no time dimension, or no valid frames exist for
        the given `tstart` and `window`.
    KeyError
        If no potential field is present in `ds`.
    """
    if "t" not in _get_phi_complex(ds).dims:
        raise ValueError(
            "The phi2 movie requires a time-resolved potential with a 't' "
            "dimension (GS2 'phi_t', enabled via write_phi_over_time). This "
            "dataset only has the single-snapshot 'phi'."
        )

    frame_indices = _resolve_frame_indices(ds, tstart, window)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "phi2_by_theta_movie.mp4"

    frames_dir = output_dir / (Path(filename).stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_values = ds["t"].values

    # Fix the y-axis across all frames and save at a constant canvas size
    # (tight_bbox=False) as ffmpeg requires identical frame dimensions.
    ylim = _global_phi2_ylim(ds, frame_indices)

    n_frames = len(frame_indices)
    if verbose:
        print(f"Generating {n_frames} frames...")

    for frame_num, t_idx in enumerate(frame_indices):
        frame_filename = f"frame_{frame_num:06d}.png"
        if window is not None:
            plot_phi2_by_theta(
                ds,
                window=window,
                tstart=float(t_values[t_idx]),
                output_dir=frames_dir,
                filename=frame_filename,
                ylim=ylim,
                tight_bbox=False,
                _quiet=True,
            )
        else:
            plot_phi2_by_theta(
                ds.isel(t=slice(t_idx, t_idx + 1)),
                output_dir=frames_dir,
                filename=frame_filename,
                ylim=ylim,
                tight_bbox=False,
                _quiet=True,
            )
        if verbose:
            print(f"  Frame {frame_num + 1}/{n_frames}")

    output_path = output_dir / filename
    _stitch_frames_to_movie(frames_dir, output_path, fps=fps, crf=crf)

    print(f"Saved {output_path}")
    return output_path


def plot_fields_by_mode(
    ds: xr.Dataset,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
) -> Path:
    """Plot the kx and ky spectra of each available field at the last time step.

    Each enabled field (phi, apar, bpar) gets one column of two panels: the top
    shows the spectrum against ``kx`` (summed over ``ky``) and the bottom against
    ``ky`` (summed over ``kx``), both on a log y-scale. An electrostatic run
    (phi only) therefore gives a single 2x1 column; enabling ``apar`` and/or
    ``bpar`` adds further columns.

    The 1D spectra are taken from GS2's ``<field>2_by_kx`` and
    ``<field>2_by_ky`` outputs.

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
    # (field prefix, label); kept only if both 1D spectra are present.
    candidates = [
        ("phi", r"$|\phi|^2$"),
        ("apar", r"$|A_\parallel|^2$"),
        ("bpar", r"$|B_\parallel|^2$"),
    ]
    fields = [
        (f, lab) for f, lab in candidates
        if f"{f}2_by_kx" in ds and f"{f}2_by_ky" in ds
    ]
    if not fields:
        raise ValueError(
            "No field spectra found: expected '<field>2_by_kx' and "
            "'<field>2_by_ky' for at least one of phi/apar/bpar."
        )

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default filename
    if filename is None:
        filename = "fields_by_mode.png"

    # kx is stored in FFT layout; sort to a monotonic axis for a clean line
    # plot. ky is already monotonic and non-negative.
    kx = ds["kx"].values
    kx_order = np.argsort(kx)
    kx_sorted = kx[kx_order]
    ky = ds["ky"].values

    n_cols = len(fields)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4.5 * n_cols, 7.0), squeeze=False
    )

    for col, (field, lab) in enumerate(fields):
        kx_spec = ds[f"{field}2_by_kx"].isel(t=-1).values[kx_order]
        ky_spec = ds[f"{field}2_by_ky"].isel(t=-1).values

        ax_kx, ax_ky = axes[0, col], axes[1, col]
        ax_kx.semilogy(kx_sorted, kx_spec, marker=".", ms=4, lw=1.3)
        ax_ky.semilogy(ky, ky_spec, marker=".", ms=4, lw=1.3)

        ax_kx.set_title(lab)
        ax_kx.set_xlabel(r"$k_x \rho_\text{ref}$")
        ax_ky.set_xlabel(r"$k_y \rho_\text{ref}$")
        for ax in (ax_kx, ax_ky):
            ax.grid(alpha=0.3, which="both")

    axes[0, 0].set_ylabel("spectral power")
    axes[1, 0].set_ylabel("spectral power")

    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")

    return output_path
