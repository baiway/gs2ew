r"""Poloidal structure of the nonlinear transfer.

Plots :math:`N_\mathbf{k}^{Q,f}(\theta)` — the nonlinear drive of the zonal
modes for each transferred quantity Q and mediator field f — faceted into one
column per field, plus the kinetic-energy transfer :math:`T_v^\mathrm{ZF}` as a
common dashed reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gs2ew import catalog
from gs2ew.catalog import BmagScaling, Drive
from gs2ew.movie import render_movie
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow, frame_indices, resolve_window


def plot_transfer_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    normalise: bool = False,
    norm_factors: dict[Drive, float] | None = None,
    show_std: bool = False,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    tight_bbox: bool = True,
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
        Output filename. Defaults to ``transfer_theta[_averaged].png``.
    div_by_bmag2 : bool, optional
        Divide the field drives (H, S, U) by ``bmag**2``. Shorthand for
        ``bmag_power=-2``; see :class:`~gs2ew.catalog.BmagScaling`.
    bmag_power : float, optional
        Multiply the field drives by ``bmag**bmag_power``. Use ``1`` to cancel
        the ``1/bmag`` currently carried by GS2's free-energy prefactor.
    kinetic_bmag_power : float, optional
        Multiply the kinetic-energy transfer by ``bmag**kinetic_bmag_power``.
        Default 0 (unchanged).
    normalise : bool, optional
        Divide each curve by its peak absolute value so the poloidal shapes can
        be compared regardless of amplitude. Default False.
    norm_factors : dict, optional
        Per-drive divisors used instead of each curve's own peak when
        `normalise` is True. The movie helper passes movie-wide factors here.
    show_std : bool, optional
        Shade each curve with a +/-1 standard deviation band over the averaging
        window. Ignored when not averaging. Default False.
    quantities, fields : list of str, optional
        Restrict to these quantities (``"U"``, ``"H"``, ``"S"``) or mediator
        fields. None means everything present.
    include_kinetic : bool, optional
        Include the kinetic-energy transfer. Default True.
    ke_variant : str, optional
        Variant tag for the kinetic-energy variable name; see
        :meth:`~gs2ew.catalog.Drive.varname`.
    ylim : tuple of float, optional
        Fixed y-limits. Movie frames pass this so the axis stays steady.
    figsize : tuple of float, optional
        Figure size in inches. Defaults to ``(5 * n_columns, 5)``.
    tight_bbox : bool, optional
        Save with a tight bounding box. Movie frames pass False so every frame
        has identical pixel dimensions.
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
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    columns = catalog.field_columns(drives)
    theta = ds["theta"].values

    if filename is None:
        filename = common.default_filename("transfer_theta", tw.averaging)
    if figsize is None:
        figsize = (5 * len(columns), 5)

    fig, axes = plt.subplots(
        1, len(columns), figsize=figsize, sharey=True, squeeze=False
    )

    for ax, (field, col_drives) in zip(axes[0], columns):
        for drive in col_drives:
            da = catalog.load(ds, drive, "theta", scaling, ke_variant)
            values = np.asarray(tw.mean(da).values)
            std = tw.std(da) if (show_std and tw.averaging) else None
            std = np.asarray(std.values) if std is not None else None

            if normalise:
                factor = norm_factors.get(drive) if norm_factors is not None else None
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
                linewidth=1.5,
                color=common.drive_colour(drive),
                linestyle=common.drive_linestyle(drive),
                label=drive.label_short(scaling),
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
        ax.set_xlabel(r"$\theta$", fontsize=12)
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel(
        "normalised transfer" if normalise else "transfer", fontsize=12
    )

    # One legend for the whole figure. The columns share a curve set apart from
    # the kinetic-energy transfer, so collect handles across all of them and
    # drop duplicates rather than repeating a legend in every panel.
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


def plot_transfer_theta_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str = "transfer_theta.mp4",
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
    normalise: bool | str = False,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
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
    scaling = catalog.resolve_scaling(div_by_bmag2, bmag_power, kinetic_bmag_power)
    indices = frame_indices(ds, tstart, window, saturation_pad, stride)

    drives = catalog.available_drives(
        ds, "theta", quantities, fields, include_kinetic, ke_variant
    )
    if not drives:
        raise ValueError("No transfer diagnostics found in dataset.")

    values = common.frame_values(ds, drives, "theta", indices, scaling, ke_variant)

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
            bmag_power=scaling.field_power,
            kinetic_bmag_power=scaling.kinetic_power,
            normalise=frame_norm,
            norm_factors=norm_factors,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
            ke_variant=ke_variant,
            ylim=ylim,
            tight_bbox=False,
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


__all__ = ["plot_transfer_theta", "plot_transfer_theta_movie"]
