r"""One-glance summary of where the nonlinear transfer goes.

Reduces everything to net numbers: the flux-surface integral
:math:`\int \mathrm{d}\theta\, J\, N_\mathbf{k}^{Q,f}`, averaged over a time
window, broken down by mediator field and — where the trapped/passing
diagnostics are enabled — by species and particle population.

The poloidal profiles show *shape*; this shows *budget*. It is the plot to
check first when asking which field or which population actually matters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gs2ew import catalog
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, resolve_window

#: Trapped bars are hatched so the two populations differ by more than colour.
TP_HATCHES: dict[int, str] = {1: "", 2: "///"}


def plot_transfer_summary(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str = "transfer_summary.png",
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    show_std: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Bar chart of the net transfer, by field and by particle population.

    The left panel groups the flux-surface-integrated transfer by mediator
    field, one bar per transferred quantity. The right panel — drawn only when
    `*_trapped_passing` output is present — splits the same integral by species
    and by trapped/passing, summed over :math:`k_{xt}`.

    Unlike the other plots, `tstart` defaults to ``"saturation"``: a net number
    from a window straddling the linear phase is not a meaningful budget.

    Parameters
    ----------
    jacobian : bool, optional
        Weight the theta integral by `jacob`. Default True.
    show_std : bool, optional
        Draw error bars showing +/-1 standard deviation of the integrated
        transfer over the averaging window. Default True.

    Other parameters are as
    :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    tp_drives = catalog.available_drives(
        ds, "trapped_passing", quantities, fields, include_kinetic=False
    )

    n_panels = 2 if tp_drives else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.0 * n_panels, 4.8), squeeze=False
    )

    _by_field(axes[0][0], ds, tw, drives, scaling, ke_variant, jacobian, show_std)
    if tp_drives:
        _by_population(axes[0][1], ds, tw, tp_drives, scaling, jacobian)

    if tw.title is not None:
        fig.suptitle(tw.title)
    fig.tight_layout()

    return common.save_figure(fig, output_dir, filename, quiet=quiet)


def _integrated(
    ds: xr.Dataset,
    tw,
    da: xr.DataArray,
    jacobian: bool,
) -> tuple[float, float]:
    """(mean, std) of the theta-integrated series over the window."""
    integrated = theta_integral(da, ds, jacobian=jacobian)
    mean = float(tw.mean(integrated).values)
    std = tw.std(integrated)
    return mean, (float(std.values) if std is not None else 0.0)


def _by_field(
    ax,
    ds: xr.Dataset,
    tw,
    drives: list[Drive],
    scaling,
    ke_variant: str | None,
    jacobian: bool,
    show_std: bool,
) -> None:
    """Grouped bars: x = mediator field, colour = transferred quantity."""
    quantities = [
        q
        for q in list(catalog.QUANTITIES) + [catalog.KINETIC]
        if any(d.quantity == q for d in drives)
    ]
    groups = [f for f in catalog.FIELDS if any(d.field == f for d in drives)]
    # The kinetic-energy transfer has no mediator field; give it its own group
    # rather than pretending it belongs to one.
    has_kinetic = catalog.KINETIC in quantities
    labels = [f"${catalog.FIELDS[f]}$" for f in groups] + (
        ["kinetic"] if has_kinetic else []
    )

    x = np.arange(len(labels))
    width = 0.8 / max(len(quantities), 1)

    for i, quantity in enumerate(quantities):
        heights, errors = [], []
        for field in groups:
            drive = Drive(quantity, field)
            if quantity == catalog.KINETIC or drive not in drives:
                heights.append(np.nan)
                errors.append(0.0)
                continue
            mean, std = _integrated(
                ds, tw, catalog.load(ds, drive, "theta", scaling), jacobian
            )
            heights.append(mean)
            errors.append(std)

        if has_kinetic:
            if quantity == catalog.KINETIC:
                mean, std = _integrated(
                    ds,
                    tw,
                    catalog.load(ds, Drive(catalog.KINETIC), "theta", scaling, ke_variant),
                    jacobian,
                )
                heights.append(mean)
                errors.append(std)
            else:
                heights.append(np.nan)
                errors.append(0.0)

        ax.bar(
            x + (i - (len(quantities) - 1) / 2) * width,
            heights,
            width,
            yerr=errors if show_std and tw.averaging else None,
            capsize=3,
            color=common.QUANTITY_COLOURS[quantity],
            label=Drive(quantity, groups[0] if groups else None).label_short(scaling),
            error_kw={"linewidth": 1.0, "ecolor": "0.3"},
        )

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("mediator field", fontsize=11)
    ax.set_ylabel(r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$", fontsize=11)
    ax.set_title("net transfer by field")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)


def _by_population(
    ax,
    ds: xr.Dataset,
    tw,
    drives: list[Drive],
    scaling,
    jacobian: bool,
) -> None:
    """Grouped bars: x = species/population, colour = quantity, hatch = trapped."""
    species = common.species_labels(ds)
    tp_values = [int(v) for v in np.asarray(ds["tp"].values)]
    quantities = [q for q in catalog.QUANTITIES if any(d.quantity == q for d in drives)]

    labels = [
        f"{name}\n{catalog.TP_LABELS[tp]}" for name in species for tp in tp_values
    ]
    x = np.arange(len(labels))
    width = 0.8 / max(len(quantities), 1)

    for i, quantity in enumerate(quantities):
        heights = []
        for j in range(len(species)):
            for tp in tp_values:
                # Sum the mediator fields: this panel asks which particles
                # carry the transfer, not which field mediates it.
                total = 0.0
                for drive in (d for d in drives if d.quantity == quantity):
                    da = catalog.load(ds, drive, "trapped_passing", scaling)
                    da = da.isel(species=j).sel(tp=tp).sum("kxt_shift")
                    mean, _ = _integrated(ds, tw, da, jacobian)
                    total += mean
                heights.append(total)

        offset = (i - (len(quantities) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            color=common.QUANTITY_COLOURS[quantity],
            label=Drive(quantity, "phi").label_short(scaling),
        )
        # Hatch the trapped bars; they alternate with passing along x.
        for k, bar in enumerate(bars):
            bar.set_hatch(TP_HATCHES[tp_values[k % len(tp_values)]])

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlabel("species and population", fontsize=11)
    ax.set_ylabel(r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$", fontsize=11)
    ax.set_title("net transfer by population (all fields)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)


__all__ = ["TP_HATCHES", "plot_transfer_summary"]
