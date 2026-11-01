r"""The poloidal structure of the transfer, drawn on the flux surface.

Maps :math:`N_\mathbf{k}^{Q,f}(\theta)` onto the simulated flux surface in the
poloidal plane, as a band coloured by the transfer, so where on the surface
the zonal modes are driven can be read off against the shape of the surface.
The surface comes from GS2's ``rplot`` and ``zplot``, and the band's width is
along the surface's normal, from ``rprime`` and ``zprime``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.plots import _common as common
from gs2ew.plots.transfer_theta import compute_transfer_theta
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow

#: GS2's geometry along the surface: R and Z, in units of the minor radius,
#: and their derivatives with respect to the flux-surface label.
GEOMETRY = ("rplot", "zplot", "rprime", "zprime")


def compute_transfer_flux_surface(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """The theta profiles :func:`plot_transfer_flux_surface` draws, with the
    shape of the surface.

    As :func:`~gs2ew.plots.transfer_theta.compute_transfer_theta`, over one
    poloidal turn (:math:`|\\theta| \\le \\pi`), plus GS2's ``rplot``,
    ``zplot``, ``rprime`` and ``zprime``.

    Raises
    ------
    ValueError
        If `ds` has no transfer diagnostics or no flux-surface geometry.
    """
    missing = [name for name in GEOMETRY if name not in ds]
    if missing:
        raise ValueError(
            f"No flux-surface geometry in dataset: {', '.join(missing)} missing."
        )

    reduced = compute_transfer_theta(
        ds,
        window=window,
        tstart=tstart,
        saturation_pad=saturation_pad,
        quantities=quantities,
        fields=fields,
        include_kinetic=include_kinetic,
        quiet=quiet,
    )
    for name in GEOMETRY:
        reduced[name] = ds[name]
    # With nperiod > 1 the theta grid goes round the surface more than once.
    turn = np.pi * (1 + 1e-9)
    return reduced.sel(theta=slice(-turn, turn))


def reference_surfaces(
    r: np.ndarray, z: np.ndarray, r_prime: np.ndarray
) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], tuple[float, float]]:
    """Nested surfaces to draw around the simulated one, for orientation.

    GS2 knows only the simulated surface, so the others are that surface
    scaled about its centre to half-widths of 0.1, 0.2, ..., 1 (the last is
    the plasma boundary, at the minor radius), with the centre moved by the
    Shafranov shift taken from ``rprime``. They are a sketch, not an
    equilibrium.

    Returns
    -------
    surfaces : list of (half-width, R, Z)
    axis : (R, Z) of the sketched magnetic axis
    """
    half_width = (r.max() - r.min()) / 2
    r_centre, z_centre = (r.max() + r.min()) / 2, (z.max() + z.min()) / 2
    # The outboard and inboard edges move together by the shift.
    shift = (r_prime[np.argmax(r)] + r_prime[np.argmin(r)]) / 2

    surfaces = []
    for width in np.linspace(0.1, 1.0, 10):
        if abs(width - half_width) < 0.05:
            continue
        scale = width / half_width
        centre = r_centre + shift * (width - half_width)
        surfaces.append(
            (width, centre + (r - r_centre) * scale, z_centre + (z - z_centre) * scale)
        )
    return surfaces, (r_centre - shift * half_width, z_centre)


def render_transfer_flux_surface(
    reduced: xr.Dataset,
    normalise: bool = True,
    band: float = 0.05,
    surfaces: bool = True,
    ncols: int = 3,
) -> Figure:
    """Draw the transfer from :func:`compute_transfer_flux_surface` on the
    flux surface.

    One panel per drive, at most `ncols` to a row. With `normalise`, each drive is
    divided by its peak absolute value, so the panels show where on the
    surface each one acts; without, they share a colour scale. `band` is the
    half-width of the coloured band in the flux-surface label, and `surfaces`
    draws the sketched nested surfaces of :func:`reference_surfaces`.
    """
    drives = common.drives_in(reduced)
    r, z, r_prime, z_prime = (np.asarray(reduced[name].values) for name in GEOMETRY)
    nested, axis = reference_surfaces(r, z, r_prime)

    # As few rows as `ncols` allows, then as few columns as fill them, so
    # four drives make two rows of two rather than three and one.
    nrows = -(-len(drives) // ncols)
    ncols = -(-len(drives) // nrows)
    # The boundary is the simulated surface scaled up, so has its shape. The
    # colorbar and labels take some width, and the panels keep equal axes,
    # so the figure is that much shorter than the panels' shape alone.
    aspect = 0.85 * (z.max() - z.min()) / (r.max() - r.min())
    fig, axes = common.subplots(nrows, ncols, aspect=aspect, sharex=True, sharey=True)

    values = {name: np.asarray(reduced[name].values) for name, _ in drives}
    if normalise:
        values = {name: common.normalise(v, None) for name, v in values.items()}
        limit = 1.0
    else:
        limit = common.symmetric_limit(*values.values())

    edges_r = np.stack([r - band * r_prime, r + band * r_prime])
    edges_z = np.stack([z - band * z_prime, z + band * z_prime])

    mesh = None
    for ax, (name, drive) in zip(axes.flat, drives):
        if surfaces:
            for width, sr, sz in nested:
                boundary = width == 1.0
                ax.plot(
                    sr,
                    sz,
                    color="k" if boundary else "0.75",
                    linewidth=0.8 if boundary else 0.5,
                )
            ax.plot(*axis, "k.", markersize=2)
        mesh = ax.pcolormesh(
            edges_r,
            edges_z,
            np.stack([values[name], values[name]]),
            shading="gouraud",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        ax.set_title(drive.label())
        ax.set_aspect("equal")
        ax.grid(False)

    for ax in axes.flat[len(drives):]:
        ax.set_visible(False)
    # Under the lowest panel in each column, which may not be in the last row.
    for column in axes.T:
        lowest = [ax for ax in column if ax.get_visible()][-1]
        lowest.set_xlabel(r"$R/a$")
        lowest.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Z/a$")

    fig.colorbar(
        mesh,
        ax=axes,
        label="normalised transfer" if normalise else "transfer",
        shrink=0.85,
    )
    return fig


def plot_transfer_flux_surface(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | str | None = None,
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    normalise: bool = True,
    band: float = 0.05,
    surfaces: bool = True,
    ncols: int = 3,
    tight_bbox: bool = False,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Draw :math:`N_\mathbf{k}^{Q,f}(\theta)` on the flux surface.

    Each drive gets a panel showing the poloidal cross-section, with the
    simulated flux surface drawn as a band coloured by the transfer (over one
    poloidal turn), and sketched nested surfaces and plasma boundary for
    orientation.

    Parameters
    ----------
    normalise : bool, optional
        Divide each drive by its peak absolute value. Default True.
    band : float, optional
        Half-width of the coloured band in the flux-surface label, for
        visibility only. Default 0.05.
    surfaces : bool, optional
        Sketch nested flux surfaces, the plasma boundary and the magnetic
        axis; see :func:`reference_surfaces`. Default True.
    ncols : int, optional
        Most panels per row. Default 3.
    filename : str, optional
        Output filename. Defaults to ``transfer_flux_surface[_averaged]``.

    Other parameters are as :func:`gs2ew.plots.transfer_theta.plot_transfer_theta`.

    Returns
    -------
    Path
        Path to the saved figure.

    Raises
    ------
    ValueError
        If `ds` has no transfer diagnostics or no flux-surface geometry.
    """
    return make_plot(
        "transfer_flux_surface",
        ds,
        dict(
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(normalise=normalise, band=band, surfaces=surfaces, ncols=ncols),
        output_dir=output_dir,
        filename=filename,
        tight_bbox=tight_bbox,
        save_data=save_data,
        quiet=quiet,
    )


register(
    PlotSpec(
        "transfer_flux_surface",
        compute_transfer_flux_surface,
        render_transfer_flux_surface,
        lambda reduced: common.default_filename(
            "transfer_flux_surface", TimeWindow.from_attrs(reduced.attrs).averaging
        ),
    )
)


__all__ = [
    "GEOMETRY",
    "compute_transfer_flux_surface",
    "plot_transfer_flux_surface",
    "reference_surfaces",
    "render_transfer_flux_surface",
]
