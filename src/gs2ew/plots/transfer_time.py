r"""Time-resolved views of the nonlinear transfer.

Both plots here trade a movie for a single static figure. A Hovmoeller diagram
shows the whole time evolution at once, which is usually easier to read than
watching it go past: you can point at a feature, measure when it appeared, and
put it in a paper.

* :func:`plot_transfer_hovmoller` — :math:`N_\mathbf{k}^{Q,f}(t, \theta)`,
  how the poloidal structure develops.
* :func:`plot_transfer_butterfly` — :math:`N_\mathbf{k}^{Q,f}(t, k_{xt})` with
  :math:`\theta` integrated out, which target scales get driven and when.
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
from gs2ew.plots import _common as common


def _time_slice(ds: xr.Dataset, tstart: float | None, tend: float | None) -> xr.Dataset:
    """Restrict `ds` to ``[tstart, tend]``, treating None as open-ended."""
    if tstart is None and tend is None:
        return ds
    return ds.sel(t=slice(tstart, tend))


_TIME_LABEL = r"$t\, (a/v_\mathrm{th})$"


def _time_attrs(sub: xr.Dataset) -> dict[str, float]:
    """The time range a time-resolved plot covers, for its reduced data."""
    t = sub["t"].values
    return {"tstart": float(t[0]), "tend": float(t[-1])}


def compute_transfer_hovmoller(
    ds: xr.Dataset,
    tstart: float | None = None,
    tend: float | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    r"""The :math:`(t, \theta)` histories :func:`plot_transfer_hovmoller` draws."""
    sub = _time_slice(ds, tstart, tend)

    drives = catalog.available_drives(
        sub, "theta", quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    reduced = xr.Dataset(attrs=_time_attrs(sub))
    for drive in drives:
        name = drive.varname("theta")
        reduced[name] = catalog.load(sub, drive, "theta")
        reduced[name].attrs.update(common.drive_attrs(drive))
    return reduced


def render_transfer_hovmoller(
    reduced: xr.Dataset,
    share: str = "row",
    percentile: float = 99.5,
) -> Figure:
    """Draw the histories from :func:`compute_transfer_hovmoller`."""
    panels = {
        drive: np.asarray(reduced[name].transpose("theta", "t").values)
        for name, drive in common.drives_in(reduced)
    }
    return common.heatmap_grid(
        panels,
        x=np.asarray(reduced["t"].values),
        y=np.asarray(reduced["theta"].values),
        xlabel=_TIME_LABEL,
        ylabel=r"$\theta$",
        share=share,
        percentile=percentile,
    )


def plot_transfer_hovmoller(
    ds: xr.Dataset,
    tstart: float | None = None,
    tend: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    share: str = "row",
    percentile: float = 99.5,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Hovmoeller diagram of :math:`N_\mathbf{k}^{Q,f}(t, \theta)`.

    One panel per drive on a quantity-by-field grid, with time on the x-axis
    and :math:`\theta` on the y-axis. Shows at a glance when the transfer
    switches on, whether its poloidal structure drifts, and how bursty it is —
    all things a movie makes you watch for.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset.
    tstart, tend : float, optional
        Restrict the time axis. Both default to the full run.
    filename : str, optional
        Output filename. Defaults to ``transfer_hovmoller.png``.
    share : str, optional
        Colour-scale sharing: ``"row"`` (default) gives each quantity its own
        symmetric limit; ``"all"`` puts every panel on one scale.
    percentile : float, optional
        Clip the colour limit to this percentile of ``|value|`` rather than the
        maximum. Default 99.5, because the burst at the end of the linear phase
        is typically far larger than the saturated transfer and would otherwise
        wash the rest of the run out. Pass 100 for a true maximum, or set
        `tstart` past the overshoot.

    Other parameters are as
    :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "transfer_hovmoller",
        ds,
        dict(
            tstart=tstart,
            tend=tend,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(share=share, percentile=percentile),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


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
            "plot_transfer_hovmoller instead."
        )
    return dim


def compute_transfer_butterfly(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    tstart: float | None = None,
    tend: float | None = None,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """The theta-integrated :math:`(t, k)` histories
    :func:`plot_transfer_butterfly` draws, over the full wavevector range."""
    _spectral_dim(reduction)
    sub = _time_slice(ds, tstart, tend)

    drives = catalog.available_drives(
        sub, reduction, quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    reduced = xr.Dataset(attrs={**_time_attrs(sub), "reduction": reduction})
    for drive in drives:
        da = catalog.load(sub, drive, reduction)
        integrated = theta_integral(da, ds, jacobian=jacobian)
        integrated.attrs = common.derived_attrs(
            da, common.theta_integral_note(jacobian)
        )
        integrated.attrs.update(common.drive_attrs(drive))
        reduced[drive.varname(reduction)] = integrated
    return reduced


def render_transfer_butterfly(
    reduced: xr.Dataset,
    klim: float | None = None,
    share: str = "row",
    percentile: float = 99.5,
) -> Figure:
    """Draw the histories from :func:`compute_transfer_butterfly`.

    `klim` clips the wavevector axis to ``[-klim, klim]``.
    """
    dim = _spectral_dim(reduced.attrs["reduction"])

    panels: dict[Drive, np.ndarray] = {}
    k_values: np.ndarray | None = None
    for name, drive in common.drives_in(reduced):
        da = reduced[name]
        if klim is not None:
            da = da.sel({dim: slice(-klim, klim)})
        if k_values is None:
            k_values = np.asarray(da[dim].values)
        panels[drive] = np.asarray(da.transpose(dim, "t").values)

    return common.heatmap_grid(
        panels,
        x=np.asarray(reduced["t"].values),
        y=k_values,
        xlabel=_TIME_LABEL,
        ylabel=catalog.K_LABELS[dim],
        cbar_label=r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$",
        share=share,
        percentile=percentile,
    )


def plot_transfer_butterfly(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    tstart: float | None = None,
    tend: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    klim: float | None = None,
    share: str = "row",
    percentile: float = 99.5,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Butterfly diagram of :math:`\int \mathrm{d}\theta\, J\, N_\mathbf{k}^{Q,f}(t, k)`.

    Time on the x-axis, triad wavevector on the y-axis, poloidal structure
    integrated out with the flux-surface Jacobian. This is where a zonal flow
    condensing onto ever-longer wavelengths shows up as a feature migrating
    toward :math:`k = 0`.

    Parameters
    ----------
    reduction : str, optional
        ``"kxtarget"`` (default), ``"kxsource"`` or ``"kysource"``.
    jacobian : bool, optional
        Weight the theta integral by `jacob`. Default True.
    klim : float, optional
        Clip the wavevector axis to ``[-klim, klim]``. The saved data keeps
        the full range.

    Other parameters are as :func:`plot_transfer_hovmoller`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "transfer_butterfly",
        ds,
        dict(
            reduction=reduction,
            tstart=tstart,
            tend=tend,
            jacobian=jacobian,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(klim=klim, share=share, percentile=percentile),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


register(
    PlotSpec(
        "transfer_hovmoller",
        compute_transfer_hovmoller,
        render_transfer_hovmoller,
        lambda reduced: "transfer_hovmoller.png",
    )
)
register(
    PlotSpec(
        "transfer_butterfly",
        compute_transfer_butterfly,
        render_transfer_butterfly,
        lambda reduced: f"transfer_butterfly_{reduced.attrs['reduction']}.png",
    )
)


__all__ = [
    "compute_transfer_butterfly",
    "compute_transfer_hovmoller",
    "plot_transfer_butterfly",
    "plot_transfer_hovmoller",
    "render_transfer_butterfly",
    "render_transfer_hovmoller",
]
