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
from matplotlib.figure import Figure

from gs2ew import catalog
from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow, resolve_window

#: Trapped bars are hatched so the two populations differ by more than colour.
TP_HATCHES: dict[int, str] = {1: "", 2: "///"}


def _population_name(quantity: str) -> str:
    """Name of the reduced variable holding `quantity`'s population totals."""
    stem, _ = catalog.QUANTITIES[quantity]
    return f"{stem}_transfer_trapped_passing"


def compute_transfer_summary(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """The net transfers :func:`plot_transfer_summary` draws.

    For each drive, the theta-integrated transfer averaged over the window,
    and its standard deviation over the window as ``<name>_std``. Where
    trapped/passing output is present, also each quantity's integrated
    transfer by species and population, summed over the mediator fields.
    """
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    tp_drives = catalog.available_drives(
        ds, "trapped_passing", quantities, fields, include_kinetic=False
    )

    note = common.theta_integral_note(jacobian)
    cell_methods = "t: mean" if tw.averaging else None
    reduced = xr.Dataset(attrs=tw.attrs())
    for drive in drives:
        da = catalog.load(ds, drive, "theta")
        mean, std = _integrated(ds, tw, da, jacobian)
        name = drive.varname("theta")
        reduced[name] = (
            (),
            mean,
            {**common.derived_attrs(da, note, cell_methods), **common.drive_attrs(drive)},
        )
        if tw.averaging:
            reduced[f"{name}_std"] = (
                (),
                std,
                common.derived_attrs(
                    da, f"{note} (standard deviation over t)", "t: standard_deviation"
                ),
            )

    if tp_drives:
        reduced["type_of_species"] = ds["type_of_species"]
        species = range(int(ds.sizes["species"]))
        tp_values = [int(v) for v in np.asarray(ds["tp"].values)]
        for quantity in catalog.QUANTITIES:
            q_drives = [d for d in tp_drives if d.quantity == quantity]
            if not q_drives:
                continue
            totals = np.empty((len(species), len(tp_values)))
            for j in species:
                for k, tp in enumerate(tp_values):
                    # Sum the mediator fields: this panel asks which particles
                    # carry the transfer, not which field mediates it.
                    total = 0.0
                    for drive in q_drives:
                        da = catalog.load(ds, drive, "trapped_passing")
                        da = da.isel(species=j).sel(tp=tp).sum("kxt_shift")
                        mean, _ = _integrated(ds, tw, da, jacobian)
                        total += mean
                    totals[j, k] = total
            source = catalog.load(ds, q_drives[0], "trapped_passing")
            reduced[_population_name(quantity)] = (
                ("species", "tp"),
                totals,
                common.derived_attrs(
                    source,
                    f"summed over kxt_shift and the mediator fields, {note}",
                    cell_methods,
                ),
            )
        reduced = reduced.assign_coords(species=ds["species"], tp=ds["tp"])

    return reduced


def render_transfer_summary(reduced: xr.Dataset, show_std: bool = True) -> Figure:
    """Draw the net transfers from :func:`compute_transfer_summary`."""
    tw = TimeWindow.from_attrs(reduced.attrs)
    has_tp = any(_population_name(q) in reduced for q in catalog.QUANTITIES)

    n_panels = 2 if has_tp else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.0 * n_panels, 4.8), squeeze=False
    )

    _by_field(axes[0][0], reduced, show_std and tw.averaging)
    if has_tp:
        _by_population(axes[0][1], reduced)

    if tw.title is not None:
        fig.suptitle(tw.title)
    fig.tight_layout()
    return fig


def plot_transfer_summary(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    show_std: bool = True,
    save_data: bool = True,
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
    filename : str, optional
        Output filename. Defaults to ``transfer_summary.png``.
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
    return make_plot(
        "transfer_summary",
        ds,
        dict(
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            jacobian=jacobian,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(show_std=show_std),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def _integrated(
    ds: xr.Dataset,
    tw: TimeWindow,
    da: xr.DataArray,
    jacobian: bool,
) -> tuple[float, float]:
    """(mean, std) of the theta-integrated series over the window."""
    integrated = theta_integral(da, ds, jacobian=jacobian)
    mean = float(tw.mean(integrated).values)
    std = tw.std(integrated)
    return mean, (float(std.values) if std is not None else 0.0)


def _by_field(ax, reduced: xr.Dataset, show_std: bool) -> None:
    """Grouped bars: x = mediator field, colour = transferred quantity."""
    names = {drive: name for name, drive in common.drives_in(reduced)}
    drives = list(names)
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

    def bar(drive: Drive) -> tuple[float, float]:
        if drive not in names:
            return np.nan, 0.0
        name = names[drive]
        std = reduced[f"{name}_std"] if f"{name}_std" in reduced else None
        return float(reduced[name]), (float(std) if std is not None else 0.0)

    x = np.arange(len(labels))
    width = 0.8 / max(len(quantities), 1)

    for i, quantity in enumerate(quantities):
        heights, errors = [], []
        for field in groups:
            mean, std = (
                (np.nan, 0.0)
                if quantity == catalog.KINETIC
                else bar(Drive(quantity, field))
            )
            heights.append(mean)
            errors.append(std)

        if has_kinetic:
            mean, std = (
                bar(Drive(catalog.KINETIC))
                if quantity == catalog.KINETIC
                else (np.nan, 0.0)
            )
            heights.append(mean)
            errors.append(std)

        ax.bar(
            x + (i - (len(quantities) - 1) / 2) * width,
            heights,
            width,
            yerr=errors if show_std else None,
            capsize=3,
            color=common.QUANTITY_COLOURS[quantity],
            label=Drive(quantity, groups[0] if groups else None).label_short(),
            error_kw={"linewidth": 1.0, "ecolor": "0.3"},
        )

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("mediator field", fontsize=11)
    ax.set_ylabel(r"$\int \mathrm{d}\theta\, J\, N_\mathbf{k}$", fontsize=11)
    ax.set_title("net transfer by field")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)


def _by_population(ax, reduced: xr.Dataset) -> None:
    """Grouped bars: x = species/population, colour = quantity, hatch = trapped."""
    species = common.species_labels(reduced)
    tp_values = [int(v) for v in np.asarray(reduced["tp"].values)]
    quantities = [q for q in catalog.QUANTITIES if _population_name(q) in reduced]

    labels = [
        f"{name}\n{catalog.TP_LABELS[tp]}" for name in species for tp in tp_values
    ]
    x = np.arange(len(labels))
    width = 0.8 / max(len(quantities), 1)

    for i, quantity in enumerate(quantities):
        # Species-major, matching the labels.
        heights = list(np.asarray(reduced[_population_name(quantity)].values).ravel())

        offset = (i - (len(quantities) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            color=common.QUANTITY_COLOURS[quantity],
            label=Drive(quantity, "phi").label_short(),
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


register(
    PlotSpec(
        "transfer_summary",
        compute_transfer_summary,
        render_transfer_summary,
        lambda reduced: "transfer_summary.png",
    )
)


__all__ = [
    "TP_HATCHES",
    "compute_transfer_summary",
    "plot_transfer_summary",
    "render_transfer_summary",
]
