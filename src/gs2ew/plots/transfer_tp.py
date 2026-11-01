r"""Nonlinear transfer split by trapped and passing particles.

GS2 writes ``<quantity>_transfer_<field>_trapped_passing`` with dimensions
``(t, species, tp, theta, kxt_shift)``, where ``tp = 1`` is passing and
``tp = 2`` is trapped (marginally trapped `wfb` particles count as passing
when ``wfbbc_option = 'passing'``, and as trapped otherwise).
These replaced the old fully velocity-resolved arrays: they answer the
question that mattered — whether trapped or passing particles carry the
transfer — at a small fraction of the size.

The velocity-space integration weights ``w`` and ``wl`` are already applied
inside GS2, so nothing further is needed here; summing over species, `tp` and
`kxt_shift` reproduces the plain `_theta` output to round-off.

Two views: :func:`plot_transfer_trapped_passing` sums over :math:`k_{xt}` and
shows the poloidal profiles, one panel per species with trapped and passing on
the same axes; :func:`plot_transfer_trapped_passing_spectral` keeps
:math:`k_{xt}` and shows the :math:`(\theta, k_{xt})` plane, one row per
population.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gs2ew import catalog
from gs2ew.catalog import Drive
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, frame_indices, resolve_window

#: Passing particles solid, trapped dashed — a secondary encoding so the two
#: populations are distinguishable without relying on colour.
TP_LINESTYLES: dict[int, str] = {1: "-", 2: "--"}


def _tp_values(ds: xr.Dataset) -> list[int]:
    """The `tp` coordinate values present, as plain ints."""
    return [int(v) for v in np.asarray(ds["tp"].values)]


def plot_transfer_trapped_passing(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    normalise: bool = False,
    norm_factors: dict[tuple[Drive, int, int], float] | None = None,
    show_std: bool = False,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Poloidal transfer from trapped and passing particles, per species.

    Summed over :math:`k_{xt}`. Each panel is one species (columns) and one
    mediator field (rows); within a panel, trapped and passing curves for the
    same species share the axes so their shapes and magnitudes can be compared
    directly. Colour identifies the transferred quantity, line style the
    population.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset containing `*_trapped_passing` variables.
    quantities : list of str, optional
        Restrict to these quantities. Defaults to all present; pass
        ``["U"]`` for an uncluttered two-curve-per-panel view.
    normalise : bool, optional
        Divide each curve by its peak absolute value. Default False.
    norm_factors : dict, optional
        Divisors keyed by ``(drive, species_index, tp)``, used instead of each
        curve's own peak when `normalise` is True. The movie helper passes
        movie-wide factors here.
    show_std : bool, optional
        Shade each curve with a +/-1 standard deviation band over the window.

    Other parameters are as
    :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.

    Raises
    ------
    ValueError
        If `ds` has no trapped/passing transfer diagnostics.
    """
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, "trapped_passing", quantities, fields, include_kinetic=False
    )
    if not drives:
        raise ValueError(
            "No trapped/passing transfer diagnostics found in dataset "
            "(enable write_*_transfer_trapped_passing in GS2)."
        )

    field_rows = [f for f in catalog.FIELDS if any(d.field == f for d in drives)]
    species = common.species_labels(ds)
    tp_values = _tp_values(ds)
    theta = ds["theta"].values

    if filename is None:
        filename = common.default_filename("transfer_trapped_passing", tw.averaging)

    fig, axes = plt.subplots(
        len(field_rows),
        len(species),
        figsize=(5.2 * len(species), 4.0 * len(field_rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for i, field in enumerate(field_rows):
        for j, species_label in enumerate(species):
            ax = axes[i][j]

            for drive in (d for d in drives if d.field == field):
                da = catalog.load(ds, drive, "trapped_passing")
                da = da.isel(species=j).sum("kxt_shift")

                for tp in tp_values:
                    values = np.asarray(tw.mean(da.sel(tp=tp)).values)
                    std = tw.std(da.sel(tp=tp)) if (show_std and tw.averaging) else None
                    std = np.asarray(std.values) if std is not None else None

                    if normalise:
                        key = (drive, j, tp)
                        factor = (
                            norm_factors.get(key) if norm_factors is not None else None
                        )
                        if std is not None:
                            divisor = (
                                float(np.nanmax(np.abs(values)))
                                if factor is None
                                else factor
                            )
                            if divisor > 0:
                                std = std / divisor
                        values = common.normalise(values, factor)

                    ax.plot(
                        theta,
                        values,
                        linewidth=1.5,
                        color=common.drive_colour(drive),
                        linestyle=TP_LINESTYLES[tp],
                        label=f"{drive.label_short()} {catalog.TP_LABELS[tp]}",
                    )
                    if std is not None:
                        ax.fill_between(
                            theta,
                            values - std,
                            values + std,
                            color=common.drive_colour(drive),
                            alpha=0.15,
                            linewidth=0,
                        )

            ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
            ax.grid(alpha=0.3)
            if ylim is not None:
                ax.set_ylim(ylim)
            if i == 0:
                ax.set_title(species_label)
            if i == len(field_rows) - 1:
                ax.set_xlabel(r"$\theta$", fontsize=12)
            if j == 0:
                ax.set_ylabel(
                    rf"${catalog.FIELDS[field]}$" + "\n"
                    + ("normalised transfer" if normalise else "transfer"),
                    fontsize=11,
                )

    # A panel can carry six curves (three quantities x two populations), so an
    # in-axes legend would sit on top of the data. Put one below the figure.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(labels), 3),
        fontsize=9,
        frameon=False,
    )

    if tw.title is not None:
        fig.suptitle(tw.title)
    # Leave room at the bottom for the figure-level legend.
    legend_rows = -(-len(labels) // 3)
    fig.tight_layout(rect=(0, 0.04 + 0.035 * legend_rows, 1, 1))

    return common.save_figure(
        fig, output_dir, filename, tight_bbox=tight_bbox, quiet=quiet
    )


def plot_transfer_trapped_passing_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str = "transfer_trapped_passing.mp4",
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    normalise: bool | str = False,
    stride: int = 1,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Animate :func:`plot_transfer_trapped_passing` over time.

    `normalise` behaves as in
    :func:`gs2ew.plots.transfer_theta.plot_transfer_theta_movie`: False, per
    ``"frame"``, or movie-wide ``"movie"``.
    """
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drives = catalog.available_drives(
        ds, "trapped_passing", quantities, fields, include_kinetic=False
    )
    if not drives:
        raise ValueError("No trapped/passing transfer diagnostics found in dataset.")

    tp_values = _tp_values(ds)
    n_species = int(ds.sizes["species"])

    # Per-curve extrema over every frame, so neither the y-axis nor the
    # normalisation shifts as the movie plays.
    curves: dict[tuple[Drive, int, int], np.ndarray] = {}
    for drive in drives:
        da = catalog.load(ds, drive, "trapped_passing")
        da = da.isel(t=indices).sum("kxt_shift")
        for j in range(n_species):
            for tp in tp_values:
                curves[(drive, j, tp)] = np.asarray(
                    da.isel(species=j).sel(tp=tp).values
                )

    norm_mode = ("movie" if normalise == "movie" else "frame") if normalise else None
    if norm_mode is None:
        frame_norm, norm_factors = False, None
        ylim = common.padded_limits(list(curves.values()))
    else:
        frame_norm = True
        norm_factors = (
            {k: float(np.nanmax(np.abs(v))) for k, v in curves.items()}
            if norm_mode == "movie"
            else None
        )
        ylim = (-1.05, 1.05)

    t_values = ds["t"].values

    def render(frame_path: Path, t_index: int) -> None:
        frame_ds = ds if window is not None else ds.isel(t=slice(t_index, t_index + 1))
        plot_transfer_trapped_passing(
            frame_ds,
            window=window,
            tstart=float(t_values[t_index]) if window is not None else None,
            output_dir=frame_path.parent,
            filename=frame_path.name,
            quantities=quantities,
            fields=fields,
            normalise=frame_norm,
            norm_factors=norm_factors,
            ylim=ylim,
            tight_bbox=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


def plot_transfer_trapped_passing_spectral(
    ds: xr.Dataset,
    quantity: str = "U",
    field: str = "phi",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    klim: float | None = None,
    vmax: float | None = None,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Heatmaps of :math:`N_\mathbf{k}^{Q,f}(\theta, k_{xt})` per population.

    Rows are passing and trapped, columns are species, for a single
    (quantity, field) pair. All panels share one symmetric colour scale, so a
    darker panel really does carry more transfer — which is the comparison this
    plot exists to make.

    Parameters
    ----------
    quantity : str, optional
        Transferred quantity: ``"U"`` (default), ``"H"``, ``"S"`` or
        ``"U_flr"``. (``"U_field"`` has no trapped/passing output.)
    field : str, optional
        Mediator field. Default ``"phi"``.
    vmax : float, optional
        Fixed symmetric colour limit; movie frames pass this.

    Other parameters are as :func:`plot_transfer_trapped_passing`.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drive = Drive(quantity, field)
    da = catalog.load(ds, drive, "trapped_passing")
    da = tw.mean(da)
    if klim is not None:
        da = da.sel(kxt_shift=slice(-klim, klim))

    species = common.species_labels(ds)
    tp_values = _tp_values(ds)
    theta = ds["theta"].values
    k_values = np.asarray(da["kxt_shift"].values)

    if filename is None:
        filename = common.default_filename(
            f"transfer_tp_spectral_{quantity}_{field}", tw.averaging
        )

    limit = vmax if vmax is not None else common.symmetric_limit(np.asarray(da.values))

    fig, axes = plt.subplots(
        len(tp_values),
        len(species),
        figsize=(4.8 * len(species), 3.8 * len(tp_values)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    mesh = None
    for i, tp in enumerate(tp_values):
        for j, species_label in enumerate(species):
            ax = axes[i][j]
            mesh = ax.pcolormesh(
                k_values,
                theta,
                np.asarray(da.isel(species=j).sel(tp=tp).values),
                cmap=common.DIVERGING_CMAP,
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            if i == 0:
                ax.set_title(species_label)
            if i == len(tp_values) - 1:
                ax.set_xlabel(catalog.K_LABELS["kxt_shift"], fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    f"{catalog.TP_LABELS[tp]}\n" + r"$\theta$", fontsize=11
                )

    fig.colorbar(mesh, ax=axes, label=drive.label(), shrink=0.85)

    title = drive.label()
    if tw.title is not None:
        title = f"{title} — {tw.title}"
    fig.suptitle(title)

    return common.save_figure(
        fig, output_dir, filename, tight_bbox=tight_bbox, quiet=quiet
    )


def plot_transfer_trapped_passing_spectral_movie(
    ds: xr.Dataset,
    quantity: str = "U",
    field: str = "phi",
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    klim: float | None = None,
    stride: int = 1,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Animate :func:`plot_transfer_trapped_passing_spectral` over time."""
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drive = Drive(quantity, field)
    da = catalog.load(ds, drive, "trapped_passing").isel(t=indices)
    if klim is not None:
        da = da.sel(kxt_shift=slice(-klim, klim))
    vmax = common.symmetric_limit(np.asarray(da.values))

    if filename is None:
        filename = f"transfer_tp_spectral_{quantity}_{field}.mp4"

    t_values = ds["t"].values

    def render(frame_path: Path, t_index: int) -> None:
        frame_ds = ds if window is not None else ds.isel(t=slice(t_index, t_index + 1))
        plot_transfer_trapped_passing_spectral(
            frame_ds,
            quantity=quantity,
            field=field,
            window=window,
            tstart=float(t_values[t_index]) if window is not None else None,
            output_dir=frame_path.parent,
            filename=frame_path.name,
            klim=klim,
            vmax=vmax,
            tight_bbox=False,
            quiet=True,
        )

    return render_movie(
        render, indices, Path(output_dir) / filename, fps=fps, crf=crf, verbose=verbose
    )


__all__ = [
    "TP_LINESTYLES",
    "plot_transfer_trapped_passing",
    "plot_transfer_trapped_passing_movie",
    "plot_transfer_trapped_passing_spectral",
    "plot_transfer_trapped_passing_spectral_movie",
]
