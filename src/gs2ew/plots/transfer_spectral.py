r"""Wavevector-resolved nonlinear transfer.

GS2 breaks each drive down by one leg of the three-wave triad:

* ``kxtarget`` — :math:`N_\mathbf{k}^{Q,f}(\theta, k_{xt})`, the target zonal
  mode being driven;
* ``kxsource`` — :math:`N_\mathbf{k}^{Q,f}(\theta, k_{xs})`, the source mode
  giving up the energy;
* ``kysource`` — :math:`N_\mathbf{k}^{Q,f}(\theta, k_{ys})`, the source
  binormal wavenumber, on an extended grid covering both signs.

Two views are offered: :func:`plot_transfer_spectral` shows the full
:math:`(\theta, k)` plane as a heatmap, and :func:`plot_transfer_spectrum`
integrates over :math:`\theta` to give a 1D spectrum.

Summing any of these over its wavevector axis reproduces the plain `_theta`
output to round-off, so no Jacobian-like weight belongs on the wavevector sum —
these are discrete sums over Fourier triads, not integrals over a coordinate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gs2ew import catalog
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, frame_indices, resolve_window


def _spectral_dim(reduction: str) -> str:
    """Name of the wavevector dimension carried by `reduction`."""
    try:
        _, dim = catalog.REDUCTIONS[reduction]
    except KeyError:
        raise ValueError(
            f"Unknown reduction {reduction!r}; expected one of "
            f"{[r for r in catalog.REDUCTIONS if r != 'theta']}."
        ) from None
    if dim is None:
        raise ValueError(
            "The 'theta' reduction has no wavevector axis; use "
            "gs2ew.plots.transfer_theta instead."
        )
    return dim



def plot_transfer_spectral(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    klim: float | None = None,
    vmax: float | None = None,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Heatmaps of :math:`N_\mathbf{k}^{Q,f}(\theta, k)` for one `reduction`.

    One panel per (quantity, field) pair: rows are the transferred quantities,
    columns the mediator fields. Colour limits are symmetric and shared across
    each row, so the zero point is always the colormap's neutral midpoint and
    fields are directly comparable within a quantity.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset.
    reduction : str, optional
        One of ``"kxtarget"`` (default), ``"kxsource"`` or ``"kysource"``.
    klim : float, optional
        Clip the wavevector axis to ``[-klim, klim]``. The spectra are usually
        concentrated near k = 0, so this is often worth setting.
    vmax : float, optional
        Fixed symmetric colour limit. Movie frames pass this so the colour
        scale does not shift between frames; otherwise each row picks its own.

    Other parameters are as :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    dim = _spectral_dim(reduction)
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    theta = ds["theta"].values

    if filename is None:
        filename = common.default_filename(
            f"transfer_spectral_{reduction}", tw.averaging
        )

    # Reduce every drive up front so the colour limits can be fixed before
    # anything is drawn.
    panels: dict[Drive, np.ndarray] = {}
    k_values: np.ndarray | None = None
    for drive in drives:
        da = tw.mean(catalog.load(ds, drive, reduction, scaling, ke_variant))
        if klim is not None:
            da = da.sel({dim: slice(-klim, klim)})
        if k_values is None:
            k_values = np.asarray(da[dim].values)
        # Panels are drawn as (theta, k), so transpose out of the stored
        # (k, theta) ordering that xarray hands back.
        panels[drive] = np.asarray(da.transpose("theta", dim).values)

    fig = common.heatmap_grid(
        panels,
        x=k_values,
        y=theta,
        xlabel=catalog.K_LABELS[dim],
        ylabel=r"$\theta$",
        scaling=scaling,
        vmax=vmax,
    )

    if tw.title is not None:
        fig.suptitle(tw.title)

    return common.save_figure(
        fig, output_dir, filename, tight_bbox=tight_bbox, quiet=quiet
    )


def plot_transfer_spectrum(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    klim: float | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    r"""1D spectrum :math:`\int \mathrm{d}\theta\, J\, N_\mathbf{k}^{Q,f}(\theta, k)`.

    The poloidal structure is integrated out with the flux-surface Jacobian,
    leaving the net transfer as a function of the triad wavevector. Faceted one
    column per mediator field, matching :func:`plot_transfer_theta`.

    Parameters
    ----------
    jacobian : bool, optional
        Weight the theta integral by `jacob`. Default True. Set False to
        integrate the bare profile instead.

    Other parameters are as :func:`plot_transfer_spectral`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    dim = _spectral_dim(reduction)
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    columns = catalog.field_columns(drives)

    if filename is None:
        filename = common.default_filename(
            f"transfer_spectrum_{reduction}", tw.averaging
        )

    fig, axes = plt.subplots(
        1, len(columns), figsize=(5 * len(columns), 4.5), sharey=True, squeeze=False
    )

    for ax, (field, col_drives) in zip(axes[0], columns):
        for drive in col_drives:
            da = catalog.load(ds, drive, reduction, scaling, ke_variant)
            spectrum = theta_integral(tw.mean(da), ds, jacobian=jacobian)
            if klim is not None:
                spectrum = spectrum.sel({dim: slice(-klim, klim)})
            ax.plot(
                np.asarray(spectrum[dim].values),
                np.asarray(spectrum.values),
                linewidth=1.5,
                color=common.drive_colour(drive),
                linestyle=common.drive_linestyle(drive),
                label=drive.label_short(scaling),
            )

        if field is not None:
            ax.set_title(f"${catalog.FIELDS[field]}$")
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
        ax.set_xlabel(catalog.K_LABELS[dim], fontsize=12)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel(r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$", fontsize=12)

    handles: dict[str, object] = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    axes[0, 0].legend(handles.values(), handles.keys(), loc="best")

    if tw.title is not None:
        fig.suptitle(tw.title)
    fig.tight_layout()

    return common.save_figure(
        fig, output_dir, filename, tight_bbox=tight_bbox, quiet=quiet
    )


def plot_transfer_spectral_movie(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    klim: float | None = None,
    stride: int = 1,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    r"""Animate :func:`plot_transfer_spectral` over time.

    The colour scale is fixed across the whole movie — from the largest
    absolute value any panel reaches at any frame — so a brightening panel
    means the transfer genuinely grew, not that the scale rescaled under it.

    Parameters are as :func:`plot_transfer_spectral`, plus `stride`, `fps`,
    `crf` and `verbose` as in
    :func:`gs2ew.plots.transfer_theta.plot_transfer_theta_movie`.
    """
    dim = _spectral_dim(reduction)
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    if filename is None:
        filename = f"transfer_spectral_{reduction}.mp4"

    clip = (lambda da: da.sel({dim: slice(-klim, klim)})) if klim is not None else None
    values = common.frame_values(
        ds, drives, reduction, indices, scaling, ke_variant, collapse=clip
    )
    vmax = common.symmetric_limit(*values.values())

    t_values = ds["t"].values

    def render(frame_path: Path, t_index: int) -> None:
        frame_ds = ds if window is not None else ds.isel(t=slice(t_index, t_index + 1))
        plot_transfer_spectral(
            frame_ds,
            reduction=reduction,
            window=window,
            tstart=float(t_values[t_index]) if window is not None else None,
            output_dir=frame_path.parent,
            filename=frame_path.name,
            bmag_power=scaling.field_power,
            kinetic_bmag_power=scaling.kinetic_power,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
            ke_variant=ke_variant,
            klim=klim,
            vmax=vmax,
            tight_bbox=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


__all__ = [
    "plot_transfer_spectral",
    "plot_transfer_spectrum",
    "plot_transfer_spectral_movie",
]
