"""Comparing runs: how a result depends on one input parameter.

For a convergence test or a parameter scan. Each run is reduced as the
single-run plots reduce it, over its own saturated window, and the results
are set against the parameter, which is read from each run's own input deck:

* :func:`plot_compare_scalars`: the field amplitude, heat flux and each
  drive's net transfer against the parameter;
* :func:`plot_compare_theta`: each drive's :math:`N_\\mathbf{k}^{Q,f}(\\theta)`,
  one line per run;
* :func:`plot_compare_traces`: each field's amplitude against time, one line
  per run, for whether and when each saturates;
* :func:`plot_compare_spectra`: each field's kx and ky spectra, one line per
  run, for whether the fields decay at large k.

The runs are given as a list of datasets. :func:`find_runs` and
:func:`comparable` pick them out of a scan: the runs whose input decks differ
only in the parameter, so in a one-at-a-time scan the template's run and the
runs varying that parameter, not those varying another.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import f90nml
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew import catalog, provenance
from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.catalog import Drive
from gs2ew.geometry import theta_integral
from gs2ew.io import OUTPUT_SUFFIX, RunFiles, detect_saturation_time, find_run
from gs2ew.plots import _common as common
from gs2ew.timebase import DEFAULT_SATURATION_PAD, TimeWindow, resolve_window

#: The fields whose traces and spectra are compared, with their labels.
FIELDS = {
    "phi": r"$|\delta\phi|^2$",
    "apar": r"$|\delta A_\parallel|^2$",
    "bpar": r"$|\delta B_\parallel|^2$",
}

#: The time traces compared, when every run has them, with their labels.
SCALARS = {
    "phi2": r"$\langle|\delta\phi|^2\rangle$",
    "heat_flux_tot": r"$Q$",
}


def find_runs(paths: Sequence[str | Path]) -> list[RunFiles]:
    """The runs at `paths`: each a run directory or ``.out.nc``, or a
    directory searched for them, as a scan's is."""
    runs = []
    for path in map(Path, paths):
        if path.is_file() or len(list(path.glob(f"*{OUTPUT_SUFFIX}"))) == 1:
            runs.append(find_run(path))
        elif path.is_dir():
            runs += [find_run(nc) for nc in sorted(path.rglob(f"*{OUTPUT_SUFFIX}"))]
        else:
            raise FileNotFoundError(f"No such run or directory: {path}")
    return runs


def _decks(ds: xr.Dataset) -> list[f90nml.Namelist]:
    """The input decks GS2 embedded in `ds`: as given, then with every default
    filled in."""
    found = []
    for used in (False, True):
        text = provenance.input_deck(ds, used=used)
        if text is not None:
            found.append(f90nml.reads(text))
    return found


def _resolve(deck: f90nml.Namelist, key: str) -> str | None:
    """`key` as ``"namelist.key"`` in `deck`, or None if it is not there.

    Raises
    ------
    ValueError
        If a bare key is in several of `deck`'s namelists.
    """
    if "." in key:
        group, name = key.lower().split(".", 1)
        return f"{group}.{name}" if name in deck.get(group, {}) else None
    name = key.lower()
    groups = [group for group in deck if name in deck[group]]
    if len(groups) > 1:
        raise ValueError(
            f"{key!r} is in several namelists ({', '.join(groups)}); give it as "
            f"one of {', '.join(f'{g}.{name}' for g in groups)}."
        )
    return f"{groups[0]}.{name}" if groups else None


def parameter(ds: xr.Dataset, key: str) -> tuple[str, object]:
    """``(namelist.key, value)`` of `key` in the run `ds`: from the input
    deck GS2 embedded, or, for a value left at its default, from the deck with
    the defaults filled in.

    Raises
    ------
    ValueError
        If neither deck has `key`, or a bare `key` is ambiguous.
    """
    for deck in _decks(ds):
        resolved = _resolve(deck, key)
        if resolved is not None:
            group, name = resolved.split(".")
            return resolved, deck[group][name]
    raise ValueError(f"{key!r} is not in the input deck of {run_label(ds)}.")


def comparable(
    datasets: Sequence[xr.Dataset], key: str, ignore: Sequence[str] = ()
) -> list[int]:
    """Indices of the largest set of `datasets` whose input decks differ only
    in `key`, and in `ignore` and the settings of a restart.

    Raises
    ------
    ValueError
        If no run has `key`, no two runs compare, or two sets are equally
        large.
    """
    skipped = set(provenance.RESTART_KEYS)
    for name in (key, *ignore):
        found = set()
        for ds in datasets:
            try:
                found.add(parameter(ds, name)[0])
            except ValueError as exc:
                if "several namelists" in str(exc):
                    raise
        if name == key and not found:
            raise ValueError(f"{key!r} is not in the input deck of any run.")
        skipped |= found

    groups: dict[str, list[int]] = {}
    for i, ds in enumerate(datasets):
        decks = _decks(ds)
        if not decks:
            continue
        settings = sorted(
            (f"{group}.{name}", repr(value))
            for group in decks[0]
            for name, value in decks[0][group].items()
            if f"{group}.{name}" not in skipped
        )
        groups.setdefault(repr(settings), []).append(i)

    largest = sorted(groups.values(), key=len, reverse=True)
    if not largest or len(largest[0]) < 2:
        raise ValueError(f"No two runs differ only in {key!r}.")
    if len(largest) > 1 and len(largest[1]) == len(largest[0]):
        sets = " and ".join(
            "(" + ", ".join(run_label(datasets[i]) for i in g) + ")"
            for g in largest
            if len(g) == len(largest[0])
        )
        raise ValueError(
            f"Equally many runs differ only in {key!r} in each of {sets}; "
            "give the runs to compare."
        )
    return largest[0]


def in_order(datasets: Sequence[xr.Dataset], key: str) -> list[xr.Dataset]:
    """`datasets` in order of their value of `key`."""

    def order(ds: xr.Dataset):
        value = parameter(ds, key)[1]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return (0, value, "") if number else (1, 0, str(value))

    return sorted(datasets, key=order)


def run_label(ds: xr.Dataset) -> str:
    """The name of the run `ds` was read from: its output's, less ``.out.nc``."""
    source = Path(str(ds.encoding.get("source", "")))
    return source.name[: -len(OUTPUT_SUFFIX)] if source.name.endswith(OUTPUT_SUFFIX) else "run"


def run_labels(datasets: Sequence[xr.Dataset]) -> list[str]:
    """A name for each of `datasets` that tells them apart: the runs' own,
    or where those repeat, as runs in separate directories often do, their
    directories'."""
    names = [run_label(ds) for ds in datasets]
    if len(set(names)) == len(names):
        return names
    directories = [Path(str(ds.encoding.get("source", ""))).parent.name for ds in datasets]
    if len(set(directories)) == len(directories):
        return directories
    return [f"{d}/{n}" for d, n in zip(directories, names)]


def _run_axis(
    datasets: Sequence[xr.Dataset], key: str, windows: Sequence[TimeWindow]
) -> xr.Dataset:
    """The ``run`` dimension: each run's name, value of `key` and window."""
    resolved = {parameter(ds, key)[0] for ds in datasets}
    if len(resolved) > 1:
        raise ValueError(f"{key!r} is in different namelists in different runs: {resolved}.")
    values = [parameter(ds, key)[1] for ds in datasets]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    (name,) = resolved
    return xr.Dataset(
        {
            "tstart": ("run", [tw.tstart for tw in windows], {"long_name": "start of the run's window"}),
            "tend": ("run", [tw.tend for tw in windows], {"long_name": "end of the run's window"}),
        },
        coords={
            "run": np.arange(len(datasets)),
            "run_name": ("run", run_labels(datasets)),
            "value": (
                "run",
                np.array(values, dtype=float if numeric else object),
                {"long_name": name},
            ),
        },
        attrs={"parameter": name, "averaging": all(tw.averaging for tw in windows)},
    )


def _drives(
    datasets: Sequence[xr.Dataset],
    quantities: list[str] | None,
    fields: list[str] | None,
    include_kinetic: bool,
) -> list[list[Drive]]:
    """Each run's drives, and (last) the drives any run has, in catalogue
    order. A run can lack a drive another has: GS2 switches the transfer
    diagnostics off on a core count that splits (kx, ky), for one."""
    each = [
        catalog.available_drives(ds, "theta", quantities, fields, include_kinetic)
        for ds in datasets
    ]
    order = list(catalog.QUANTITIES) + [catalog.KINETIC]
    fields_order = [*catalog.FIELDS, None]
    union = sorted(
        {drive for drives in each for drive in drives},
        key=lambda d: (order.index(d.quantity), fields_order.index(d.field)),
    )
    return [*each, union]


_warned: set[str] = set()


def _windows(
    datasets: Sequence[xr.Dataset],
    tstart: float | str | None,
    window: float | None,
    saturation_pad: float,
    quiet: bool,
) -> list[TimeWindow]:
    """Each run's window, warning of runs with almost nothing in theirs."""
    windows = []
    for ds, label in zip(datasets, run_labels(datasets)):
        tw = resolve_window(ds, tstart, window, saturation_pad, quiet=quiet)
        steps = int(((ds["t"] >= tw.tstart) & (ds["t"] <= tw.tend)).sum())
        message = f"Warning: {label} has {steps} timestep(s) in its window."
        # Both comparison plots resolve the same windows; say it once.
        if tw.averaging and steps < 3 and not quiet and message not in _warned:
            _warned.add(message)
            print(message)
        windows.append(tw)
    return windows


def compute_compare_scalars(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    jacobian: bool = True,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """What :func:`plot_compare_scalars` draws, for each run of `datasets`.

    Over each run's own window: the mean and standard deviation of
    ``phi2`` and ``heat_flux_tot``, where every run has them, and of each
    drive's theta-integrated transfer, for the drives any run has (NaN for
    a run without it).

    Raises
    ------
    ValueError
        If the runs have nothing to compare.
    """
    windows = _windows(datasets, tstart, window, saturation_pad, quiet)
    reduced = _run_axis(datasets, vary, windows)
    cell_methods = "t: mean" if reduced.attrs["averaging"] else None

    def add(name: str, series: list[xr.DataArray | None], attrs: dict) -> None:
        means, stds = [], []
        for tw, s in zip(windows, series):
            std = None if s is None else tw.std(s)
            means.append(np.nan if s is None else float(tw.mean(s).values))
            stds.append(np.nan if s is None else 0.0 if std is None else float(std.values))
        reduced[name] = ("run", means, attrs)
        reduced[f"{name}_std"] = (
            "run",
            stds,
            {"long_name": f"standard deviation of {name} over each run's window"},
        )

    for name in SCALARS:
        if all(name in ds for ds in datasets):
            add(
                name,
                [ds[name] for ds in datasets],
                common.derived_attrs(datasets[0][name], "over each run's window", cell_methods),
            )

    note = common.theta_integral_note(jacobian)
    *each, union = _drives(datasets, quantities, fields, include_kinetic)
    for drive in union:
        series = [
            theta_integral(catalog.load(ds, drive, "theta"), ds, jacobian=jacobian)
            if drive in drives
            else None
            for ds, drives in zip(datasets, each)
        ]
        source = catalog.load(datasets[[drive in d for d in each].index(True)], drive, "theta")
        add(
            drive.varname("theta"),
            series,
            {**common.derived_attrs(source, note, cell_methods), **common.drive_attrs(drive)},
        )

    if len(reduced.data_vars) == 2:  # only the windows
        raise ValueError("The runs have no field, flux or transfer output in common.")
    return reduced


def compute_compare_theta(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    quiet: bool = False,
) -> xr.Dataset:
    """What :func:`plot_compare_theta` draws: each drive's theta profile,
    averaged over each run's window, for the drives any run has.

    Runs may have different theta grids (a scan over ``ntheta``, say), so the
    profiles share the union of them, and are NaN where a run has no point.

    Raises
    ------
    ValueError
        If the runs have no transfer diagnostics.
    """
    windows = _windows(datasets, tstart, window, saturation_pad, quiet)
    reduced = _run_axis(datasets, vary, windows)
    *each, drives = _drives(datasets, quantities, fields, include_kinetic)
    if not drives:
        raise ValueError("The runs have no transfer diagnostics.")

    grids = [np.asarray(ds["theta"].values) for ds in datasets]
    theta = np.unique(np.concatenate(grids))
    reduced = reduced.assign_coords(theta=("theta", theta, datasets[0]["theta"].attrs))
    cell_methods = "t: mean" if reduced.attrs["averaging"] else None
    for drive in drives:
        profiles = np.full((len(datasets), len(theta)), np.nan)
        for i, (ds, tw, grid) in enumerate(zip(datasets, windows, grids)):
            if drive in each[i]:
                profile = tw.mean(catalog.load(ds, drive, "theta"))
                profiles[i, np.searchsorted(theta, grid)] = profile.values
        source = catalog.load(datasets[[drive in d for d in each].index(True)], drive, "theta")
        reduced[drive.varname("theta")] = (
            ("run", "theta"),
            profiles,
            {
                **common.derived_attrs(source, "over each run's window", cell_methods),
                **common.drive_attrs(drive),
            },
        )
    return reduced


def _on_grid(
    arrays: Sequence[xr.DataArray | None], dim: str, grid: np.ndarray
) -> np.ndarray:
    """Each of `arrays` on `grid`, the union of their `dim` coordinates: NaN
    where an array has no point, or for a run without the array."""
    out = np.full((len(arrays), len(grid)), np.nan)
    for i, array in enumerate(arrays):
        if array is not None:
            out[i, np.searchsorted(grid, array[dim].values)] = array.values
    return out


def _union(datasets: Sequence[xr.Dataset], dim: str) -> np.ndarray:
    return np.unique(np.concatenate([ds[dim].values for ds in datasets if dim in ds.coords]))


def compute_compare_traces(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> xr.Dataset:
    """What :func:`plot_compare_traces` draws: each field's ``<field>2``
    against time, for each run, on the union of the runs' times, with each
    run's detected saturation time and window.

    Raises
    ------
    ValueError
        If no run has a field's trace.
    """
    windows = _windows(datasets, tstart, window, saturation_pad, quiet)
    reduced = _run_axis(datasets, vary, windows)
    reduced["tsat"] = (
        "run",
        [detect_saturation_time(ds) for ds in datasets],
        {"long_name": "detected saturation time (NaN if none)"},
    )
    t = _union(datasets, "t")
    reduced = reduced.assign_coords(t=("t", t, datasets[0]["t"].attrs))
    for field in FIELDS:
        name = f"{field}2"
        traces = [ds.get(name) for ds in datasets]
        source = next((a for a in traces if a is not None), None)
        if source is not None:
            reduced[name] = (("run", "t"), _on_grid(traces, "t", t), dict(source.attrs))
    if not any(f"{field}2" in reduced for field in FIELDS):
        raise ValueError("No run has a field's time trace (phi2, apar2 or bpar2).")
    return reduced


def compute_compare_spectra(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    quiet: bool = False,
) -> xr.Dataset:
    """What :func:`plot_compare_spectra` draws: each field's
    ``<field>2_by_kx`` and ``<field>2_by_ky``, averaged over each run's
    window, on the union of the runs' kx and ky (which differ in a scan over
    ``nx``, ``ny`` or the box size).

    Raises
    ------
    ValueError
        If no run has a field's spectra.
    """
    windows = _windows(datasets, tstart, window, saturation_pad, quiet)
    reduced = _run_axis(datasets, vary, windows)
    cell_methods = "t: mean" if reduced.attrs["averaging"] else None
    grids = {k: _union(datasets, k) for k in ("kx", "ky")}
    reduced = reduced.assign_coords(
        {k: (k, grid, datasets[0][k].attrs) for k, grid in grids.items()}
    )
    for field in FIELDS:
        for k, grid in grids.items():
            name = f"{field}2_by_{k}"
            spectra = [tw.mean(ds[name]) if name in ds else None for ds, tw in zip(datasets, windows)]
            source = next((ds[name] for ds in datasets if name in ds), None)
            if source is not None:
                reduced[name] = (
                    ("run", k),
                    _on_grid(spectra, k, grid),
                    common.derived_attrs(source, "over each run's window", cell_methods),
                )
    if not any(f"{field}2_by_kx" in reduced for field in FIELDS):
        raise ValueError("No run has a field's spectra (phi2_by_kx and phi2_by_ky, say).")
    return reduced


def _run_colours(n: int) -> np.ndarray:
    """A colour for each of `n` runs, in order of the parameter."""
    return plt.get_cmap("viridis")(np.linspace(0.0, 0.85, n))


def _run_lines(ax, x: np.ndarray, rows: np.ndarray, reduced: xr.Dataset, **style) -> None:
    """A line per run of `rows` against `x`, skipping each run's missing
    points, labelled with its value of the parameter."""
    parameter = _parameter_label(reduced)
    for i, (value, colour) in enumerate(zip(reduced["value"].values, _run_colours(len(rows)))):
        have = ~np.isnan(rows[i])
        ax.plot(
            x[have],
            rows[i][have],
            color=colour,
            label=f"{parameter} = {_value_label(value)}",
            **style,
        )


def _parameter_label(reduced: xr.Dataset) -> str:
    return reduced.attrs["parameter"].split(".")[1]


def _value_label(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_compare_scalars(reduced: xr.Dataset, show_std: bool = True) -> Figure:
    """Draw the results of :func:`compute_compare_scalars`: a panel each for
    the field amplitude and heat flux, and one for the net transfers through
    each mediator field, sharing the parameter axis."""
    scalars = [name for name in SCALARS if name in reduced]
    names = {drive: name for name, drive in common.drives_in(reduced)}
    columns = catalog.field_columns(list(names))
    fig, axes = common.subplots(len(scalars) + len(columns), 1, aspect=0.35, sharex=True)

    values = reduced["value"].values
    numeric = values.dtype.kind in "fiu"
    x = values if numeric else np.arange(len(values))

    def draw(ax, name: str, **style) -> None:
        y = reduced[name].values
        error = reduced[f"{name}_std"].values if show_std else None
        ax.errorbar(x, y, yerr=error, marker="o", capsize=2, **style)

    for ax, name in zip(axes[:, 0], scalars):
        draw(ax, name, color="k")
        ax.set_ylabel(SCALARS[name])
    for ax, (field, drives) in zip(axes[len(scalars) :, 0], columns):
        for drive in drives:
            draw(
                ax,
                names[drive],
                color=common.drive_colour(drive),
                linestyle=common.drive_linestyle(drive),
                label=drive.label_short(),
            )
        ax.axhline(0.0, color="0.5", linewidth=0.6)
        mediator = "" if field is None else f",{catalog.FIELDS[field]}"
        ax.set_ylabel(rf"$\int \mathrm{{d}}\theta\, J\, N_\mathbf{{k}}^{{Q{mediator}}}$")
    if columns:
        common.shared_legend(fig, axes[len(scalars) :, 0])

    bottom = axes[-1, 0]
    bottom.set_xlabel(_parameter_label(reduced))
    if not numeric:
        bottom.set_xticks(x, [_value_label(v) for v in values])
    return fig


def render_compare_theta(reduced: xr.Dataset, normalise: bool = False) -> Figure:
    """Draw the results of :func:`compute_compare_theta`: a panel per drive,
    by quantity (rows) and mediator field (columns), with a line per run
    coloured in order of the parameter. The kinetic-energy transfer, which
    has no field, takes the first column; with one field, the panels go two
    to a row. With `normalise`, each run's
    profile is divided by its peak absolute value."""
    drives = common.drives_in(reduced)
    rows, cols = common.drive_grid([drive for _, drive in drives])
    if len(cols) > 1:
        places = {
            drive: (rows.index(drive.quantity), 0 if drive.is_kinetic else cols.index(drive.field))
            for _, drive in drives
        }
    else:
        # One field (an electrostatic run, say): two to a row, not one tall column.
        places = {drive: divmod(i, 2) for i, (_, drive) in enumerate(drives)}
    nrows = max(r for r, _ in places.values()) + 1
    ncols = max(c for _, c in places.values()) + 1
    fig, axes = common.subplots(nrows, ncols, aspect=0.6, sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)

    theta = reduced["theta"].values
    for name, drive in drives:
        ax = axes[places[drive]]
        ax.set_visible(True)
        profiles = reduced[name].values
        if normalise:
            profiles = np.array([common.normalise(p, None) for p in profiles])
        _run_lines(ax, theta / np.pi, profiles, reduced)
        ax.axhline(0.0, color="0.5", linewidth=0.6)
        ax.set_title(drive.label())

    # In the grid a column is one field, so its panels share an axis below the
    # lowest; panels that only wrapped into a column each get their own.
    for column in axes.T:
        shown = [ax for ax in column if ax.get_visible()]
        for ax in shown if len(cols) == 1 else shown[-1:]:
            ax.set_xlabel(r"$\theta/\pi$")
            ax.xaxis.set_tick_params(labelbottom=True)
    common.shared_legend(fig, [ax for ax in axes.flat if ax.get_visible()][:1])
    return fig


def render_compare_traces(reduced: xr.Dataset, windows: bool = True) -> Figure:
    """Draw the results of :func:`compute_compare_traces`: a panel per field
    on a log scale, a line per run. With `windows`, a dotted line marks the
    start of each run's window, in its colour."""
    fields = [f for f in FIELDS if f"{f}2" in reduced]
    fig, axes = common.subplots(len(fields), 1, aspect=0.4, sharex=True)
    t = reduced["t"].values
    for ax, field in zip(axes[:, 0], fields):
        _run_lines(ax, t, reduced[f"{field}2"].values, reduced)
        ax.set_yscale("log")
        ax.set_ylabel(FIELDS[field])
        if windows and reduced.attrs["averaging"]:
            for start, colour in zip(reduced["tstart"].values, _run_colours(reduced.sizes["run"])):
                ax.axvline(start, color=colour, linestyle=":", linewidth=0.8)
    axes[-1, 0].set_xlabel(r"$t$")
    common.shared_legend(fig, axes[:1, 0])
    return fig


def render_compare_spectra(reduced: xr.Dataset) -> Figure:
    """Draw the results of :func:`compute_compare_spectra`: for each field, its
    kx spectrum above its ky spectrum (side by side for a single field), on
    log scales, a line per run."""
    fields = [f for f in FIELDS if f"{f}2_by_kx" in reduced]
    single = len(fields) == 1
    fig, axes = common.subplots(*((1, 2) if single else (2, len(fields))), aspect=0.75)
    if single:
        axes = axes.T
    labels = {"kx": r"$k_x \rho_\mathrm{ref}$", "ky": r"$k_y \rho_\mathrm{ref}$"}
    for column, field in enumerate(fields):
        for row, k in enumerate(("kx", "ky")):
            ax = axes[row, column]
            _run_lines(ax, reduced[k].values, reduced[f"{field}2_by_{k}"].values, reduced, marker=".")
            ax.set_yscale("log")
            ax.set_xlabel(labels[k])
        for row in range(2 if single else 1):
            axes[row, column].set_title(FIELDS[field])
    for ax in axes[:, 0] if not single else axes[:1, 0]:
        ax.set_ylabel("spectral power")
    common.shared_legend(fig, axes[:1, 0])
    return fig


def _filename(stem: str):
    return lambda reduced: f"compare_{_parameter_label(reduced)}_{stem}"


def plot_compare_scalars(
    datasets: Sequence[xr.Dataset],
    vary: str,
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
    """Draw the field amplitude, heat flux and net transfers of each run
    against `vary`, a key of the runs' input decks.

    Each run is averaged over its own window, chosen as by `tstart`,
    `window` and `saturation_pad`. Error bars are the standard deviation over
    the window. The runs are taken in the order given; see :func:`in_order`.
    The filename defaults to ``compare_<vary>_scalars``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "compare_scalars",
        list(datasets),
        dict(
            vary=vary,
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


def plot_compare_theta(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    normalise: bool = False,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    r"""Draw each drive's :math:`N_\mathbf{k}^{Q,f}(\theta)` with a line per
    run, coloured in order of `vary`, a key of the runs' input decks.

    Options are as :func:`plot_compare_scalars`. With `normalise`, each
    profile is divided by its peak absolute value, to compare shapes. The
    filename defaults to ``compare_<vary>_theta``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "compare_theta",
        list(datasets),
        dict(
            vary=vary,
            window=window,
            tstart=tstart,
            saturation_pad=saturation_pad,
            quantities=quantities,
            fields=fields,
            include_kinetic=include_kinetic,
        ),
        dict(normalise=normalise),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def plot_compare_traces(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    windows: bool = True,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Draw each field's amplitude against time, with a line per run
    coloured in order of `vary`, a key of the runs' input decks, and the
    start of each run's window marked (unless `windows` is False).

    Window options are as :func:`plot_compare_scalars`. The filename
    defaults to ``compare_<vary>_traces``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "compare_traces",
        list(datasets),
        dict(vary=vary, window=window, tstart=tstart, saturation_pad=saturation_pad),
        dict(windows=windows),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def plot_compare_spectra(
    datasets: Sequence[xr.Dataset],
    vary: str,
    window: float | None = None,
    tstart: float | str | None = "saturation",
    saturation_pad: float = DEFAULT_SATURATION_PAD,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Draw each field's kx and ky spectra, averaged over each run's window,
    with a line per run coloured in order of `vary`, a key of the runs'
    input decks.

    Window options are as :func:`plot_compare_scalars`. The filename
    defaults to ``compare_<vary>_spectra``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "compare_spectra",
        list(datasets),
        dict(vary=vary, window=window, tstart=tstart, saturation_pad=saturation_pad),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


register(
    PlotSpec(
        "compare_traces",
        compute_compare_traces,
        render_compare_traces,
        _filename("traces"),
    )
)
register(
    PlotSpec(
        "compare_spectra",
        compute_compare_spectra,
        render_compare_spectra,
        _filename("spectra"),
    )
)
register(
    PlotSpec(
        "compare_scalars",
        compute_compare_scalars,
        render_compare_scalars,
        _filename("scalars"),
    )
)
register(
    PlotSpec(
        "compare_theta",
        compute_compare_theta,
        render_compare_theta,
        _filename("theta"),
    )
)


__all__ = [
    "FIELDS",
    "SCALARS",
    "comparable",
    "compute_compare_scalars",
    "compute_compare_spectra",
    "compute_compare_theta",
    "compute_compare_traces",
    "find_runs",
    "in_order",
    "parameter",
    "plot_compare_scalars",
    "plot_compare_spectra",
    "plot_compare_theta",
    "plot_compare_traces",
    "render_compare_scalars",
    "render_compare_spectra",
    "render_compare_theta",
    "render_compare_traces",
    "run_label",
    "run_labels",
]
