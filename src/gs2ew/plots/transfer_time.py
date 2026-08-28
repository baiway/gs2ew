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

from gs2ew import catalog
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.plots import _common as common


def _time_slice(ds: xr.Dataset, tstart: float | None, tend: float | None) -> xr.Dataset:
    """Restrict `ds` to ``[tstart, tend]``, treating None as open-ended."""
    if tstart is None and tend is None:
        return ds
    return ds.sel(t=slice(tstart, tend))


def plot_transfer_hovmoller(
    ds: xr.Dataset,
    tstart: float | None = None,
    tend: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str = "transfer_hovmoller.png",
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    share: str = "row",
    percentile: float = 99.5,
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
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    sub = _time_slice(ds, tstart, tend)

    drives = catalog.available_drives(
        sub, "theta", quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    panels = {
        drive: np.asarray(
            catalog.load(sub, drive, "theta", scaling, ke_variant)
            .transpose("theta", "t")
            .values
        )
        for drive in drives
    }

    fig = common.heatmap_grid(
        panels,
        x=np.asarray(sub["t"].values),
        y=np.asarray(sub["theta"].values),
        xlabel=r"$t\, (a/v_\mathrm{th})$",
        ylabel=r"$\theta$",
        scaling=scaling,
        share=share,
        percentile=percentile,
    )

    return common.save_figure(fig, output_dir, filename, quiet=quiet)


def plot_transfer_butterfly(
    ds: xr.Dataset,
    reduction: str = "kxtarget",
    tstart: float | None = None,
    tend: float | None = None,
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
    share: str = "row",
    percentile: float = 99.5,
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
        Clip the wavevector axis to ``[-klim, klim]``.

    Other parameters are as :func:`plot_transfer_hovmoller`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
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

    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    sub = _time_slice(ds, tstart, tend)

    drives = catalog.available_drives(
        sub, reduction, quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError(f"No '{reduction}' transfer diagnostics found in dataset.")

    if filename is None:
        filename = f"transfer_butterfly_{reduction}.png"

    panels: dict[Drive, np.ndarray] = {}
    k_values: np.ndarray | None = None
    for drive in drives:
        da = catalog.load(sub, drive, reduction, scaling, ke_variant)
        da = theta_integral(da, ds, jacobian=jacobian)
        if klim is not None:
            da = da.sel({dim: slice(-klim, klim)})
        if k_values is None:
            k_values = np.asarray(da[dim].values)
        panels[drive] = np.asarray(da.transpose(dim, "t").values)

    fig = common.heatmap_grid(
        panels,
        x=np.asarray(sub["t"].values),
        y=k_values,
        xlabel=r"$t\, (a/v_\mathrm{th})$",
        ylabel=catalog.K_LABELS[dim],
        scaling=scaling,
        cbar_label=r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$",
        share=share,
        percentile=percentile,
    )

    return common.save_figure(fig, output_dir, filename, quiet=quiet)


__all__ = ["plot_transfer_hovmoller", "plot_transfer_butterfly"]
