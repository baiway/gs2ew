r"""Poloidal structure of the nonlinear transfer.

Plots :math:`N_\mathbf{k}^{Q,f}(\theta)` — the nonlinear drive of the zonal
modes for each transferred quantity Q and mediator field f — faceted into one
column per field, plus the kinetic-energy transfer :math:`N_\mathbf{k}^\mathrm{KE}`
as a common dashed reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew import catalog
from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.catalog import Drive
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow, frame_indices, resolve_window


def compute_transfer_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    show_std: bool = False,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """The theta profiles :func:`plot_transfer_theta` draws.

    One variable per drive, named as in the GS2 output and marked with its
    quantity and field, averaged over the window (or at the last timestep).
    With `show_std` and a window, each also gets a ``<name>_std``.

    Raises
    ------
    ValueError
        If `ds` contains no transfer diagnostics.
    """
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    reduced = xr.Dataset(attrs=tw.attrs())
    for drive in drives:
        da = catalog.load(ds, drive, "theta")
        name = drive.varname("theta")
        reduced[name] = common.time_mean(tw, da)
        reduced[name].attrs.update(common.drive_attrs(drive))
        std = common.time_std(tw, da) if show_std else None
        if std is not None:
            reduced[f"{name}_std"] = std
            reduced[name].attrs["ancillary_variables"] = f"{name}_std"
    return reduced


def render_transfer_theta(
    reduced: xr.Dataset,
    normalise: bool = False,
    norm_factors: dict[str, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw the profiles from :func:`compute_transfer_theta`.

    `norm_factors` maps variable names to the divisors used in place of each
    curve's own peak when `normalise` is True. `figsize` overrides the size
    the style gives the figure.
    """
    names = {drive: name for name, drive in common.drives_in(reduced)}
    columns = catalog.field_columns(list(names))
    theta = reduced["theta"].values

    fig, axes = common.subplots(1, len(columns), aspect=1.3, sharey=True)
    if figsize is not None:
        fig.set_size_inches(figsize)

    for ax, (field, col_drives) in zip(axes[0], columns):
        for drive in col_drives:
            name = names[drive]
            values = np.asarray(reduced[name].values)
            std = (
                np.asarray(reduced[f"{name}_std"].values)
                if f"{name}_std" in reduced
                else None
            )

            if normalise:
                factor = norm_factors.get(name) if norm_factors is not None else None
                # Scale the band by the same divisor as the curve it belongs to.
                if std is not None:
                    divisor = (
                        float(np.nanmax(np.abs(values))) if factor is None else factor
                    )
                    if divisor > 0:
                        std = std / divisor
                values = common.normalise(values, factor)

            ax.plot(
                theta,
                values,
                color=common.drive_colour(drive),
                linestyle=common.drive_linestyle(drive),
                label=drive.label_short(),
            )
            if std is not None:
                ax.fill_between(
                    theta,
                    values - std,
                    values + std,
                    color=common.drive_colour(drive),
                    alpha=0.2,
                    linewidth=0,
                )

        if field is not None:
            ax.set_title(f"${catalog.FIELDS[field]}$")
        ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel(r"$\theta$")

    axes[0, 0].set_ylabel("normalised transfer" if normalise else "transfer")

    # One legend for the whole figure. The columns share a curve set apart from
    # the kinetic-energy transfer, so collect handles across all of them and
    # drop duplicates rather than repeating a legend in every panel.
    common.shared_legend(fig, axes[0])
    return fig


def plot_transfer_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    normalise: bool = False,
    norm_factors: dict[Drive | str, float] | None = None,
    show_std: bool = False,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    tight_bbox: bool = False,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Plot the poloidal structure of each transfer diagnostic in `ds`.

    Each mediator field present gets its own column, sharing a y-axis, with the
    fluctuation-energy (U), free-energy (H) and entropy (S) drives overlaid.
    The kinetic-energy transfer has no field decomposition and is purely
    electrostatic, so it is drawn dashed on the ``phi`` column only.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset.
    window : float, optional
        Length of the averaging window. See :mod:`gs2ew.timebase` for how
        `window` and `tstart` combine; with neither, the last timestep is
        plotted.
    tstart : float or str, optional
        Start of the averaging window, or ``"saturation"`` to start from the
        detected saturation time padded by `saturation_pad`.
    saturation_pad : float, optional
        Fraction of the post-saturation interval to skip when
        ``tstart="saturation"``. Default 0.20.
    output_dir : str or Path, optional
        Directory to save into. Default "outputs".
    filename : str, optional
        Output filename. Defaults to ``transfer_theta[_averaged]``, saved in
        each of the formats set by :func:`gs2ew.style.figure_options`; a
        suffix such as ``.png`` saves that format only.
    normalise : bool, optional
        Divide each curve by its peak absolute value so the poloidal shapes can
        be compared regardless of amplitude. Default False.
    norm_factors : dict, optional
        Per-drive divisors, keyed by :class:`~gs2ew.catalog.Drive` or variable
        name, used instead of each curve's own peak when `normalise` is True.
        The movie helper passes movie-wide factors here.
    show_std : bool, optional
        Shade each curve with a +/-1 standard deviation band over the averaging
        window. Ignored when not averaging. Default False.
    quantities, fields : list of str, optional
        Restrict to these quantities (keys of
        :data:`~gs2ew.catalog.QUANTITIES`) or mediator fields. None means
        everything present.
    include_kinetic : bool, optional
        Include the kinetic-energy transfer. Default True.
    ylim : tuple of float, optional
        Fixed y-limits. Movie frames pass this so the axis stays steady.
    figsize : tuple of float, optional
        Figure size in inches. By default the style's width, with square
        panels.
    tight_bbox : bool, optional
        Crop the saved figure to its contents. Default False, so the figure
        is the size it was drawn at.
    save_data : bool, optional
        Write the plotted data and its provenance alongside the figure; see
        :mod:`gs2ew.bundle`. Default True.
    quiet : bool, optional
        Suppress the "Saved ..." message. Default False.

    Returns
    -------
    Path
        Path to the saved figure.

    Raises
    ------
    ValueError
        If `ds` contains no transfer diagnostics.
    """
    if norm_factors is not None:
        norm_factors = {
            key.varname("theta") if isinstance(key, Drive) else key: value
            for key, value in norm_factors.items()
        }
    return make_plot(
        "transfer_theta",
        ds,
        dict(
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            show_std=show_std,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(normalise=normalise, norm_factors=norm_factors, ylim=ylim, figsize=figsize),
        output_dir=output_dir,
        filename=filename,
        tight_bbox=tight_bbox,
        save_data=save_data,
        quiet=quiet,
    )


def plot_transfer_theta_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str = "transfer_theta.mp4",
    normalise: bool | str = False,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    stride: int = 1,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Animate :func:`plot_transfer_theta` over time.

    Parameters
    ----------
    window : float, optional
        If given, each frame shows a rolling average over `window` time units
        starting at that frame's time; otherwise frames are instantaneous.
    tstart : float or str, optional
        First frame's time. Unlike the static plot, None means "start at the
        detected saturation time" (padded by `saturation_pad`).
    normalise : bool or str, optional
        * False — no normalisation.
        * True or ``"frame"`` — each frame divides every curve by that frame's
          own peak. Shows shape, but discards relative amplitude and time
          evolution.
        * ``"movie"`` — each curve is divided by one fixed factor (its peak
          over all frames), preserving both while scaling small terms into view.
    stride : int, optional
        Keep every nth frame. Default 1.

    Other parameters are as :func:`plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved movie.
    """
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    values = common.frame_values(ds, drives, "theta", indices)

    norm_mode = ("movie" if normalise == "movie" else "frame") if normalise else None
    if norm_mode is None:
        frame_norm, norm_factors = False, None
        ylim = common.padded_limits(values)
    else:
        frame_norm = True
        # Movie-wide uses one fixed divisor per curve; per-frame self-normalises.
        norm_factors = common.peak_factors(values) if norm_mode == "movie" else None
        ylim = (-1.05, 1.05)

    t_values = ds["t"].values

    def render(frame_path: Path, t_index: int) -> None:
        # A windowed frame needs the whole dataset (it averages forward from
        # this time); an instantaneous frame is given a one-timestep slice.
        frame_ds = ds if window is not None else ds.isel(t=slice(t_index, t_index + 1))
        plot_transfer_theta(
            frame_ds,
            window=window,
            tstart=float(t_values[t_index]) if window is not None else None,
            output_dir=frame_path.parent,
            filename=frame_path.name,
            normalise=frame_norm,
            norm_factors=norm_factors,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
            ylim=ylim,
            save_data=False,
            quiet=True,
        )

    return render_movie(
        render,
        indices,
        Path(output_dir) / filename,
        fps=fps,
        crf=crf,
        verbose=verbose,
    )


register(
    PlotSpec(
        "transfer_theta",
        compute_transfer_theta,
        render_transfer_theta,
        lambda reduced: common.default_filename(
            "transfer_theta", TimeWindow.from_attrs(reduced.attrs).averaging
        ),
    )
)


__all__ = [
    "compute_transfer_theta",
    "plot_transfer_theta",
    "plot_transfer_theta_movie",
    "render_transfer_theta",
]
