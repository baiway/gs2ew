"""Plot time traces and spectra of fields."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.io import detect_saturation_time
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import (
    DEFAULT_SATURATION_PAD,
    TimeWindow,
    frame_indices,
    resolve_window,
)


def compute_fields_time_traces(
    ds: xr.Dataset,
    window: float = 20.0,
    threshold: float = 0.1,
    quiet: bool = False,
) -> xr.Dataset:
    """The traces :func:`plot_fields_time_traces` draws, with the detected
    saturation time (NaN if none) as the ``tsat`` attribute."""
    return xr.Dataset(
        {f: ds[f] for f in ["phi2", "apar2", "bpar2"] if f in ds},
        # Saturation detection just uses phi2.
        attrs={"tsat": detect_saturation_time(ds, window=window, threshold=threshold)},
    )


def render_fields_time_traces(reduced: xr.Dataset) -> Figure:
    """Draw the traces from :func:`compute_fields_time_traces`."""
    tsat = float(reduced.attrs["tsat"])
    t = reduced["t"].values

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each field on log scale
    for field in reduced.data_vars:
        ax.semilogy(t, reduced[field].values, linewidth=1.5, label=field)

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
    return fig


def plot_fields_time_traces(
    ds: xr.Dataset,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    window: float = 20.0,
    threshold: float = 0.1,
    save_data: bool = True,
    quiet: bool = False,
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
    save_data : bool, optional
        Write the plotted data and its provenance alongside the figure; see
        :mod:`gs2ew.bundle`. Default True.
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure file
    """
    return make_plot(
        "fields_time_traces",
        ds,
        dict(window=window, threshold=threshold),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


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


def compute_phi2_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> xr.Dataset:
    """The zonal and non-zonal profiles :func:`plot_phi2_by_theta` draws.

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

    if tw.averaging and not has_t:
        raise ValueError(
            "Time averaging was requested but the potential has no time "
            "dimension; a time-resolved 'phi_t' (write_phi_over_time) is "
            "required."
        )

    reduced = xr.Dataset(attrs=tw.attrs())
    for name, da, part in (
        ("phi2_zonal", zonal, "zonal (ky = 0)"),
        ("phi2_nonzonal", non_zonal, "non-zonal (ky > 0)"),
    ):
        if has_t:
            da = common.time_mean(tw, da)
        # Selecting ky = 0 leaves a scalar ky behind; the name says it.
        reduced[name] = da.reset_coords(drop=True)
        reduced[name].attrs = {
            "long_name": f"|phi|^2 summed over kx, {part} part",
            **({"cell_methods": "t: mean"} if tw.averaging else {}),
        }
    return reduced


def render_phi2_by_theta(
    reduced: xr.Dataset,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Draw the profiles from :func:`compute_phi2_by_theta`.

    `ylim` fixes the y-axis; the movie helper uses it to keep the axis steady
    across frames.
    """
    title = TimeWindow.from_attrs(reduced.attrs).title
    theta = reduced["theta"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, reduced["phi2_zonal"].values, linewidth=1.5,
            color=common.QUANTITY_COLOURS["U"], label=r"zonal ($k_y = 0$)")
    ax.plot(theta, reduced["phi2_nonzonal"].values, linewidth=1.5, linestyle="--",
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
    return fig


def plot_phi2_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    save_data: bool = True,
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
    ValueError
        If averaging is requested but the potential has no time dimension.
    KeyError
        If no potential field is present in `ds`.
    """
    return make_plot(
        "phi2_by_theta",
        ds,
        dict(window=window, tstart=tstart, saturation_pad=saturation_pad),
        dict(ylim=ylim),
        output_dir=output_dir,
        filename=filename,
        tight_bbox=tight_bbox,
        save_data=save_data,
        quiet=quiet,
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
            save_data=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


# (field prefix, label) for each field whose spectra may be present.
_SPECTRUM_FIELDS = [
    ("phi", r"$|\phi|^2$"),
    ("apar", r"$|A_\parallel|^2$"),
    ("bpar", r"$|B_\parallel|^2$"),
]


def compute_fields_by_mode(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> xr.Dataset:
    """The spectra :func:`plot_fields_by_mode` draws.

    GS2's ``<field>2_by_kx`` and ``<field>2_by_ky`` for each field that has
    both, at the last timestep or averaged over a window, with ``kx`` sorted
    out of FFT order.

    Raises
    ------
    ValueError
        If no field has both spectra.
    """
    fields = [
        f for f, _ in _SPECTRUM_FIELDS
        if f"{f}2_by_kx" in ds and f"{f}2_by_ky" in ds
    ]
    if not fields:
        raise ValueError(
            "No field spectra found: expected '<field>2_by_kx' and "
            "'<field>2_by_ky' for at least one of phi/apar/bpar."
        )

    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)
    reduced = xr.Dataset(attrs=tw.attrs())
    for field in fields:
        for name in (f"{field}2_by_kx", f"{field}2_by_ky"):
            reduced[name] = common.time_mean(tw, ds[name])
    # kx is stored in FFT layout; sort to a monotonic axis for a clean line
    # plot. ky is already monotonic and non-negative.
    return reduced.sortby("kx")


def render_fields_by_mode(reduced: xr.Dataset) -> Figure:
    """Draw the spectra from :func:`compute_fields_by_mode`."""
    fields = [
        (f, lab) for f, lab in _SPECTRUM_FIELDS if f"{f}2_by_kx" in reduced
    ]
    kx = reduced["kx"].values
    ky = reduced["ky"].values

    n_cols = len(fields)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4.5 * n_cols, 7.0), squeeze=False
    )

    for col, (field, lab) in enumerate(fields):
        ax_kx, ax_ky = axes[0, col], axes[1, col]
        ax_kx.semilogy(kx, reduced[f"{field}2_by_kx"].values, marker=".", ms=4, lw=1.3)
        ax_ky.semilogy(ky, reduced[f"{field}2_by_ky"].values, marker=".", ms=4, lw=1.3)

        ax_kx.set_title(lab)
        ax_kx.set_xlabel(r"$k_x \rho_\text{ref}$")
        ax_ky.set_xlabel(r"$k_y \rho_\text{ref}$")
        for ax in (ax_kx, ax_ky):
            ax.grid(alpha=0.3, which="both")

    axes[0, 0].set_ylabel("spectral power")
    axes[1, 0].set_ylabel("spectral power")

    title = TimeWindow.from_attrs(reduced.attrs).title
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_fields_by_mode(
    ds: xr.Dataset,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Plot the kx and ky spectra of each available field.

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
        Filename for the plot. If None, uses "fields_by_mode.png" (or
        "fields_by_mode_averaged.png" when averaging).
    window, tstart, saturation_pad : optional
        The time window, as for every other plot; see :mod:`gs2ew.timebase`.
        By default the last timestep is plotted.
    save_data : bool, optional
        Write the plotted data and its provenance alongside the figure; see
        :mod:`gs2ew.bundle`. Default True.
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure file
    """
    return make_plot(
        "fields_by_mode",
        ds,
        dict(window=window, tstart=tstart, saturation_pad=saturation_pad),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


register(
    PlotSpec(
        "fields_time_traces",
        compute_fields_time_traces,
        render_fields_time_traces,
        lambda reduced: "field_time_traces.png",
    )
)
register(
    PlotSpec(
        "phi2_by_theta",
        compute_phi2_by_theta,
        render_phi2_by_theta,
        lambda reduced: common.default_filename(
            "phi2_by_theta", TimeWindow.from_attrs(reduced.attrs).averaging
        ),
    )
)
register(
    PlotSpec(
        "fields_by_mode",
        compute_fields_by_mode,
        render_fields_by_mode,
        lambda reduced: common.default_filename(
            "fields_by_mode", TimeWindow.from_attrs(reduced.attrs).averaging
        ),
    )
)
