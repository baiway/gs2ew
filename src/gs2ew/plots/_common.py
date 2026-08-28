"""Helpers shared by the plotting modules."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew import catalog
from gs2ew.catalog import BmagScaling, Drive

#: Fixed colour per transferred quantity, from the Okabe-Ito colourblind-safe
#: palette. Assigned by quantity rather than taken from a cycler, so U is the
#: same blue whether or not H and S happen to be present in a given run.
#: Checked for CVD separation against a light surface (worst adjacent pair
#: dE 11.0 deutan, 25.4 normal).
QUANTITY_COLOURS: dict[str, str] = {
    "U": "#0072B2",   # blue
    "H": "#D55E00",   # vermillion
    "S": "#009E73",   # bluish green
    "KE": "#CC79A7",  # reddish purple
}

#: The kinetic-energy transfer is drawn dashed so it is distinguishable from
#: the field-mediated drives by more than colour alone.
QUANTITY_LINESTYLES: dict[str, str] = {"KE": "--"}


def drive_colour(drive: Drive) -> str:
    """Fixed plot colour for `drive`, keyed on its transferred quantity."""
    return QUANTITY_COLOURS[drive.quantity]


def drive_linestyle(drive: Drive) -> str:
    """Line style for `drive`; solid unless the quantity overrides it."""
    return QUANTITY_LINESTYLES.get(drive.quantity, "-")


#: Diverging colormap for signed quantities. Every transfer heatmap uses this
#: with symmetric limits, so white always means zero net transfer.
DIVERGING_CMAP = "RdBu_r"


def save_figure(
    fig: Figure,
    output_dir: str | Path,
    filename: str,
    dpi: int = 150,
    tight_bbox: bool = True,
    quiet: bool = False,
) -> Path:
    """Save `fig` into `output_dir` and close it.

    `tight_bbox` must be False for movie frames: a tight bounding box crops to
    the drawn content, so frames would come out at different pixel sizes and
    ffmpeg would refuse to encode them.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)

    if not quiet:
        print(f"Saved {output_path}")
    return output_path


def frame_values(
    ds: xr.Dataset,
    drives: list[Drive],
    reduction: str,
    indices: np.ndarray,
    scaling: BmagScaling | None = None,
    ke_variant: str | None = None,
    collapse: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> dict[Drive, np.ndarray]:
    """Instantaneous values for each drive at every frame in `indices`.

    Used to fix a movie's axes and normalisation up front, so neither jumps
    from frame to frame. `collapse` reduces the dimensions a single frame does
    not show (for example summing over `kxt_shift`).

    Windowed movies average within each frame, and an average is bounded by the
    instantaneous values it is taken over, so limits derived from these values
    always contain the rendered frames.
    """
    out: dict[Drive, np.ndarray] = {}
    for drive in drives:
        da = catalog.load(ds, drive, reduction, scaling, ke_variant).isel(t=indices)
        if collapse is not None:
            da = collapse(da)
        out[drive] = np.asarray(da.values)
    return out


def padded_limits(
    arrays: list[np.ndarray] | dict[Drive, np.ndarray],
    pad_fraction: float = 0.05,
) -> tuple[float, float]:
    """Padded (ymin, ymax) spanning every value in `arrays`."""
    values = arrays.values() if isinstance(arrays, dict) else arrays
    flat = np.concatenate([np.asarray(a).ravel() for a in values])
    lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
    pad = pad_fraction * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def peak_factors(values: dict[Drive, np.ndarray]) -> dict[Drive, float]:
    """Peak absolute value of each drive, for movie-wide normalisation.

    Dividing every frame by these fixed factors — rather than by each frame's
    own peak — keeps both the relative amplitude between curves and their
    growth over time, while still scaling small terms into view.
    """
    return {drive: float(np.nanmax(np.abs(v))) for drive, v in values.items()}


def symmetric_limit(*arrays: np.ndarray, percentile: float = 100.0) -> float:
    """Colour limit for a centred diverging scale.

    By default the largest absolute value across `arrays`. A `percentile` below
    100 clips to that percentile of ``|value|`` instead, which matters for
    time-resolved plots: the burst at the end of the linear phase can be orders
    of magnitude above the saturated transfer and would otherwise flatten the
    rest of the figure to white.
    """
    flat = np.concatenate([np.abs(np.asarray(a)).ravel() for a in arrays])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    peak = float(np.max(flat) if percentile >= 100 else np.percentile(flat, percentile))
    # A completely flat (all-zero) field would otherwise give vmin == vmax.
    return peak if peak > 0 else 1.0


def normalise(values: np.ndarray, factor: float | None) -> np.ndarray:
    """Divide `values` by `factor`, or by its own peak when `factor` is None.

    A zero divisor (an identically zero curve) is left alone rather than
    producing NaNs.
    """
    divisor = float(np.nanmax(np.abs(values))) if factor is None else factor
    return values / divisor if divisor > 0 else values


def drive_grid(drives: list[Drive]) -> tuple[list[str], list[str | None]]:
    """Row (quantity) and column (field) keys for a grid of drives, in order.

    Rows follow the catalogue's quantity order with the kinetic-energy
    transfer last; columns follow GS2's field order. A dataset with no
    field-mediated drives at all gets a single unnamed column.
    """
    rows = [
        q
        for q in list(catalog.QUANTITIES) + [catalog.KINETIC]
        if any(d.quantity == q for d in drives)
    ]
    cols = [f for f in catalog.FIELDS if any(d.field == f for d in drives)]
    return rows, cols or [None]


def heatmap_grid(
    panels: dict[Drive, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    scaling: BmagScaling | None = None,
    cbar_label: str = "transfer",
    vmax: float | None = None,
    share: str = "row",
    percentile: float = 100.0,
    panel_size: tuple[float, float] = (4.6, 3.6),
):
    """Lay out one diverging heatmap per drive on a quantity-by-field grid.

    Every panel uses a symmetric colour scale so the colormap's neutral
    midpoint always means zero net transfer. `share` controls how the scale is
    chosen: ``"row"`` gives each quantity its own limit (fields comparable
    within a quantity), ``"all"`` gives every panel the same limit (everything
    comparable, but small quantities wash out).

    `percentile` clips the colour limit to that percentile of ``|value|``
    rather than its maximum; see :func:`symmetric_limit`.

    `panels` maps each drive to a ``(len(y), len(x))`` array. The
    kinetic-energy drive, having no mediator field, takes the first column of
    its own row and the remaining columns of that row are hidden.

    Returns
    -------
    matplotlib.figure.Figure
        The populated figure. The caller saves and closes it.
    """
    import matplotlib.pyplot as plt

    drives = list(panels)
    rows, cols = drive_grid(drives)

    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(panel_size[0] * len(cols), panel_size[1] * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
        # Constrained layout places the colorbars without overlapping the
        # axes, which tight_layout cannot do for a colorbar spanning a row.
        layout="constrained",
    )

    global_limit = (
        symmetric_limit(*panels.values(), percentile=percentile)
        if share == "all"
        else None
    )

    by_key = {(d.quantity, d.field): v for d, v in panels.items()}

    for i, quantity in enumerate(rows):
        row_values = [v for (q, _), v in by_key.items() if q == quantity]
        if vmax is not None:
            limit = vmax
        elif global_limit is not None:
            limit = global_limit
        else:
            limit = symmetric_limit(*row_values, percentile=percentile)

        mesh = None
        for j, field in enumerate(cols):
            ax = axes[i][j]
            # The kinetic-energy transfer has no mediator field, so its row
            # holds one panel rather than the same data repeated per column.
            is_kinetic = quantity == catalog.KINETIC
            key = (quantity, None if is_kinetic else field)
            if key not in by_key or (is_kinetic and j > 0):
                ax.set_visible(False)
                continue

            mesh = ax.pcolormesh(
                x,
                y,
                by_key[key],
                cmap=DIVERGING_CMAP,
                vmin=-limit,
                vmax=limit,
                shading="auto",
                rasterized=True,
            )
            ax.set_title(Drive(quantity, field).label(scaling), fontsize=11)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=11)

        if mesh is not None:
            fig.colorbar(mesh, ax=axes[i], label=cbar_label, shrink=0.85)

    return fig


def default_filename(stem: str, window_averaged: bool, suffix: str = ".png") -> str:
    """Filename for a plot, marking whether it is a window average."""
    return f"{stem}_averaged{suffix}" if window_averaged else f"{stem}{suffix}"


def species_labels(ds: xr.Dataset) -> list[str]:
    """Human-readable label for each species, e.g. ``"ion 1"`` / ``"electron"``.

    GS2's `type_of_species` uses 1 for ions and 2 for electrons; anything else
    is labelled by index. Ions are numbered only when there is more than one.
    """
    n_spec = int(ds.sizes["species"])
    types = (
        np.asarray(ds["type_of_species"].values, dtype=int).tolist()
        if "type_of_species" in ds
        else [0] * n_spec
    )

    n_ions = sum(1 for t in types if t == 1)
    labels, ion_index = [], 0
    for i, kind in enumerate(types):
        if kind == 1:
            ion_index += 1
            labels.append(f"ion {ion_index}" if n_ions > 1 else "ion")
        elif kind == 2:
            labels.append("electron")
        else:
            labels.append(f"species {i + 1}")
    return labels


__all__ = [
    "DIVERGING_CMAP",
    "QUANTITY_COLOURS",
    "QUANTITY_LINESTYLES",
    "drive_colour",
    "drive_linestyle",
    "save_figure",
    "frame_values",
    "padded_limits",
    "peak_factors",
    "symmetric_limit",
    "drive_grid",
    "heatmap_grid",
    "normalise",
    "default_filename",
    "species_labels",
]
