"""Plot time traces and spectra of fields."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from gs2ew.io import detect_saturation_time
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, frame_indices, resolve_window


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

    return common.save_figure(fig, output_dir, filename)


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
    indices: np.ndarray,
) -> tuple[float, float]:
    """Return a padded (0, ymax) spanning the zonal and non-zonal phi2 curves
    over the given frames, so a movie's y-axis stays fixed across frames."""
    zonal, non_zonal = _phi2_zonal_nonzonal(_get_phi_complex(ds))
    z = zonal.isel(t=indices).values
    n = non_zonal.isel(t=indices).values
    hi = float(max(np.nanmax(z), np.nanmax(n)))
    pad = 0.05 * hi if hi > 0 else 1.0
    return 0.0, hi + pad


def plot_phi2_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    """Plot the poloidal structure of ``|phi|^2``, split into its zonal
    (``ky = 0``) and non-zonal (``ky > 0``) parts.

    ``|phi|^2`` is computed from the complex potential and summed over ``kx``.
    With a time-resolved potential (``phi_t``) the last timestep is plotted by
    default, or a window average when `window` or `tstart` is given. With only
    the single-snapshot ``phi``, that snapshot is plotted and averaging is
    rejected.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file).
    window : float, optional
        Length of the averaging window. See :mod:`gs2ew.timebase` for how
        `window` and `tstart` combine. Requires a time-resolved potential.
    tstart : float or str, optional
        Start of the averaging window, or ``"saturation"`` to start from the
        detected saturation time padded by `saturation_pad`.
    saturation_pad : float, optional
        Fraction of the post-saturation interval to skip when
        ``tstart="saturation"``. Default 0.20.
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
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If averaging is requested but the potential has no time dimension.
    KeyError
        If no potential field is present in `ds`.
    """
    zonal, non_zonal = _phi2_zonal_nonzonal(_get_phi_complex(ds))
    has_t = "t" in zonal.dims

    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    theta = ds["theta"].values

    if tw.averaging and not has_t:
        raise ValueError(
            "Time averaging was requested but the potential has no time "
            "dimension; a time-resolved 'phi_t' (write_phi_over_time) is "
            "required."
        )

    zonal = tw.mean(zonal) if has_t else zonal
    non_zonal = tw.mean(non_zonal) if has_t else non_zonal
    title = tw.title
    if filename is None:
        filename = common.default_filename("phi2_by_theta", tw.averaging)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, zonal.values, linewidth=1.5, color=common.QUANTITY_COLOURS["U"],
            label=r"zonal ($k_y = 0$)")
    ax.plot(theta, non_zonal.values, linewidth=1.5, linestyle="--",
            color=common.QUANTITY_COLOURS["H"], label=r"non-zonal ($k_y > 0$)")

    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$\sum_{k_x} |\phi|^2$", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return common.save_figure(
        fig, output_dir, filename, tight_bbox=tight_bbox, quiet=quiet
    )


def plot_phi2_by_theta_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str = "phi2_by_theta.mp4",
    stride: int = 1,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Animate :func:`plot_phi2_by_theta` over time.

    Requires a time-resolved potential ``phi_t`` (GS2's ``write_phi_over_time``);
    the single-snapshot ``phi`` has no time axis to animate over. The y-axis is
    fixed across all frames so the curves do not rescale frame to frame.

    Parameters
    ----------
    window : float, optional
        If given, each frame is a rolling average over `window` time units.
    tstart : float or str, optional
        First frame's time; None starts at the detected saturation time.
    stride : int, optional
        Keep every nth frame. Default 1.

    Returns
    -------
    Path
        Path to the saved movie.

    Raises
    ------
    ValueError
        If the potential has no time dimension, or no frames survive filtering.
    """
    if "t" not in _get_phi_complex(ds).dims:
        raise ValueError(
            "The phi2 movie requires a time-resolved potential with a 't' "
            "dimension (GS2 'phi_t', enabled via write_phi_over_time). This "
            "dataset only has the single-snapshot 'phi'."
        )

    indices = frame_indices(ds, tstart, window, saturation_pad, stride)
    ylim = _global_phi2_ylim(ds, indices)
    t_values = ds["t"].values

    def render(frame_path: Path, t_index: int) -> None:
        frame_ds = ds if window is not None else ds.isel(t=slice(t_index, t_index + 1))
        plot_phi2_by_theta(
            frame_ds,
            window=window,
            tstart=float(t_values[t_index]) if window is not None else None,
            output_dir=frame_path.parent,
            filename=frame_path.name,
            ylim=ylim,
            tight_bbox=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


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

    return common.save_figure(fig, output_dir, filename)
