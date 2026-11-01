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

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew import catalog
from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow, frame_indices, resolve_window


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


def _clip(da: xr.DataArray, dim: str, klim: float | None) -> xr.DataArray:
    """`da` restricted to ``[-klim, klim]`` along `dim`, if `klim` is given."""
    return da.sel({dim: slice(-klim, klim)}) if klim is not None else da


def compute_transfer_spectral(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    r"""The :math:`(\theta, k)` planes :func:`plot_transfer_spectral` draws.

    One variable per drive over the full wavevector range, averaged over the
    window (or at the last timestep). `klim` is applied when rendering.
    """
    dim = _spectral_dim(reduction)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    reduced = xr.Dataset(attrs={**tw.attrs(), "reduction": reduction})
    for drive in drives:
        name = drive.varname(reduction)
        reduced[name] = common.time_mean(tw, catalog.load(ds, drive, reduction))
        reduced[name].attrs.update(common.drive_attrs(drive))
    return reduced


def render_transfer_spectral(
    reduced: xr.Dataset,
    klim: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Draw the planes from :func:`compute_transfer_spectral`.

    `klim` clips the wavevector axis to ``[-klim, klim]``; `vmax` fixes the
    symmetric colour limit.
    """
    dim = _spectral_dim(reduced.attrs["reduction"])

    # Reduce every drive up front so the colour limits can be fixed before
    # anything is drawn.
    panels: dict[Drive, np.ndarray] = {}
    k_values: np.ndarray | None = None
    for name, drive in common.drives_in(reduced):
        da = _clip(reduced[name], dim, klim)
        if k_values is None:
            k_values = np.asarray(da[dim].values)
        # Panels are drawn as (theta, k), so transpose out of the stored
        # (k, theta) ordering that xarray hands back.
        panels[drive] = np.asarray(da.transpose("theta", dim).values)

    return common.heatmap_grid(
        panels,
        x=k_values,
        y=reduced["theta"].values,
        xlabel=catalog.K_LABELS[dim],
        ylabel=r"$\theta$",
        vmax=vmax,
    )


def plot_transfer_spectral(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    klim: float | None = None,
    vmax: float | None = None,
    tight_bbox: bool = False,
    save_data: bool = True,
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
        concentrated near k = 0, so this is often worth setting. The saved
        data keeps the full range.
    vmax : float, optional
        Fixed symmetric colour limit. Movie frames pass this so the colour
        scale does not shift between frames; otherwise each row picks its own.

    Other parameters are as :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "transfer_spectral",
        ds,
        dict(
            reduction=reduction,
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(klim=klim, vmax=vmax),
        output_dir=output_dir,
        filename=filename,
        tight_bbox=tight_bbox,
        save_data=save_data,
        quiet=quiet,
    )


def compute_transfer_spectrum(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """The theta-integrated spectra :func:`plot_transfer_spectrum` draws."""
    dim = _spectral_dim(reduction)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    reduced = xr.Dataset(attrs={**tw.attrs(), "reduction": reduction})
    for drive in drives:
        da = catalog.load(ds, drive, reduction)
        spectrum = theta_integral(common.time_mean(tw, da), ds, jacobian=jacobian)
        spectrum.attrs = common.derived_attrs(
            da,
            common.theta_integral_note(jacobian),
            "t: mean" if tw.averaging else None,
        )
        spectrum.attrs.update(common.drive_attrs(drive))
        reduced[drive.varname(reduction)] = spectrum
    return reduced


def render_transfer_spectrum(
    reduced: xr.Dataset,
    klim: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Draw the spectra from :func:`compute_transfer_spectrum`."""
    dim = _spectral_dim(reduced.attrs["reduction"])
    names = {drive: name for name, drive in common.drives_in(reduced)}
    columns = catalog.field_columns(list(names))

    fig, axes = common.subplots(1, len(columns), aspect=1.2, sharey=True)

    for ax, (field, col_drives) in zip(axes[0], columns):
        for drive in col_drives:
            spectrum = _clip(reduced[names[drive]], dim, klim)
            ax.plot(
                np.asarray(spectrum[dim].values),
                np.asarray(spectrum.values),
                color=common.drive_colour(drive),
                linestyle=common.drive_linestyle(drive),
                label=drive.label_short(),
            )

        if field is not None:
            ax.set_title(f"${catalog.FIELDS[field]}$")
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
        ax.set_xlabel(catalog.K_LABELS[dim])
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0, 0].set_ylabel(r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$")
    common.shared_legend(fig, axes[0])
    return fig


def plot_transfer_spectrum(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    klim: float | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = False,
    save_data: bool = True,
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
    return make_plot(
        "transfer_spectrum",
        ds,
        dict(
            reduction=reduction,
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            jacobian=jacobian,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(klim=klim, ylim=ylim),
        output_dir=output_dir,
        filename=filename,
        tight_bbox=tight_bbox,
        save_data=save_data,
        quiet=quiet,
    )


def plot_transfer_spectral_movie(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
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
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drives = catalog.available_drives(
        ds, reduction, quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    if filename is None:
        filename = f"transfer_spectral_{reduction}.mp4"

    clip = (lambda da: da.sel({dim: slice(-klim, klim)})) if klim is not None else None
    values = common.frame_values(
        ds, drives, reduction, indices, collapse=clip
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
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
            klim=klim,
            vmax=vmax,
            save_data=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


def _filename(stem: str):
    """Default filename for a spectral plot, marking its reduction and window."""
    return lambda reduced: common.default_filename(
        f"{stem}_{reduced.attrs['reduction']}",
        TimeWindow.from_attrs(reduced.attrs).averaging,
    )


register(
    PlotSpec(
        "transfer_spectral",
        compute_transfer_spectral,
        render_transfer_spectral,
        _filename("transfer_spectral"),
    )
)
register(
    PlotSpec(
        "transfer_spectrum",
        compute_transfer_spectrum,
        render_transfer_spectrum,
        _filename("transfer_spectrum"),
    )
)


__all__ = [
    "compute_transfer_spectral",
    "compute_transfer_spectrum",
    "plot_transfer_spectral",
    "plot_transfer_spectrum",
    "plot_transfer_spectral_movie",
    "render_transfer_spectral",
    "render_transfer_spectrum",
]
