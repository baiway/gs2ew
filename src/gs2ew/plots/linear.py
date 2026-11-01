"""Linear runs: growth rates and frequencies against ky.

Each run gives the complex frequency GS2 last wrote (``omega_average``,
averaged over ``navg`` steps) for each of its ky, and the runs are grouped
into a line for each value of a parameter, read from each run's own input
deck:

* :func:`plot_linear_spectrum`: the growth rate and real frequency against
  ky, a line for each value;
* :func:`plot_linear_gamma_max`: the largest growth rate, and the ky and
  frequency it has, against the parameter;
* :func:`plot_linear_modes`: each line's eigenfunctions along the field line,
  at its fastest-growing ky (or a ky given), for the mode's parity and extent.

Runs that didn't converge (see :func:`gs2ew.linear.stopped`) are drawn with
open markers, and left out of the largest growth rate. The runs are
typically those of a scan written by :func:`gs2ew.linear.write`, one ky each,
but a ``grid_option = 'range'`` run, with several, will do.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import ticker
from matplotlib.figure import Figure

from gs2ew.bundle import PlotSpec, make_plot, register
from gs2ew.linear import converged as is_converged
from gs2ew.linear import stopped
from gs2ew.plots import _common as common
from gs2ew.plots.compare import _run_colours, _value_label, parameter, run_labels

#: The fields whose eigenfunctions are drawn, with their labels.
FIELDS = {
    "phi": r"$\delta\phi$",
    "apar": r"$\delta A_\parallel$",
    "bpar": r"$\delta B_\parallel$",
}

LABELS = {
    "gamma": r"$\gamma\, L_\mathrm{ref}/v_\mathrm{ref}$",
    "omega": r"$\omega\, L_\mathrm{ref}/v_\mathrm{ref}$",
    "ky": r"$k_y \rho_\mathrm{ref}$",
}


def _values(datasets: Sequence[xr.Dataset], vary: str | None) -> tuple[str, list]:
    """The resolved name of `vary` and each run's value of it; ``""`` and
    NaN for each without `vary`."""
    if vary is None:
        return "", [np.nan] * len(datasets)
    found = [parameter(ds, vary) for ds in datasets]
    names = {name for name, _ in found}
    if len(names) > 1:
        raise ValueError(f"{vary!r} is in different namelists in different runs: {names}.")
    return names.pop(), [value for _, value in found]


def _frequencies(ds: xr.Dataset, label: str) -> xr.DataArray:
    """The last ``omega_average`` of `ds`, by ky, as (ky, ri)."""
    if "omega_average" not in ds:
        raise ValueError(
            f"{label} has no omega_average: set write_omega and write_omavg "
            "in gs2_diagnostics_knobs."
        )
    if ds.sizes["kx"] != 1:
        raise ValueError(
            f"{label} has {ds.sizes['kx']} kx; a spectrum is drawn from runs "
            "of one theta0 each."
        )
    return ds["omega_average"].isel(t=-1, kx=0)


def _positions(
    datasets: Sequence[xr.Dataset], vary: str | None
) -> tuple[str, np.ndarray, np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    """The parameter's name, its values in order, the union of the runs' ky,
    and where each (value, ky) comes from: (run, its ky index).

    Raises
    ------
    ValueError
        If two runs have the same value and ky, so differ in something else.
    """
    name, values = _values(datasets, vary)
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if numeric:
        line_values = np.unique(np.array(values, dtype=float))
        if np.isnan(line_values).all():
            line_values = np.array([np.nan])
    else:
        line_values = np.array(sorted({str(v) for v in values}), dtype=object)
    ky = np.unique(np.concatenate([ds["ky"].values for ds in datasets]))

    labels = run_labels(datasets)
    where: dict[tuple[int, int], tuple[int, int]] = {}
    for run, (ds, value) in enumerate(zip(datasets, values)):
        row = 0 if numeric and np.isnan(value) else int(
            np.flatnonzero(line_values == (value if numeric else str(value)))[0]
        )
        for k, ky_value in enumerate(ds["ky"].values):
            column = int(np.searchsorted(ky, ky_value))
            if (row, column) in where:
                other = labels[where[(row, column)][0]]
                given = f" and {vary}" if vary else ""
                raise ValueError(
                    f"{other} and {labels[run]} are both at ky = {ky_value:g}"
                    + (f" with {vary} = {_value_label(value)}" if vary else "")
                    + f": give runs that differ only in ky{given}."
                )
            where[(row, column)] = (run, k)
    return name, line_values, ky, where


def compute_linear_spectrum(
    datasets: Sequence[xr.Dataset], vary: str | None = None, quiet: bool = False
) -> xr.Dataset:
    """What :func:`plot_linear_spectrum` draws: the growth rate ``gamma`` and
    real frequency ``omega`` of each run at each of its ky, by value of
    `vary` (a key of the runs' input decks) and ky, with whether each run
    ``converged``, why it ``stopped`` and its ``run_name``. Missing points
    are NaN, or empty strings.
    """
    datasets = list(datasets)
    labels = run_labels(datasets)
    name, line_values, ky, where = _positions(datasets, vary)
    shape = (len(line_values), len(ky))
    gamma, omega, converged = (np.full(shape, np.nan) for _ in range(3))
    why = np.full(shape, "", dtype=object)
    names = np.full(shape, "", dtype=object)

    reasons = [stopped(ds) for ds in datasets]
    frequencies = [_frequencies(ds, label) for ds, label in zip(datasets, labels)]
    for (row, column), (run, k) in where.items():
        frequency = frequencies[run].isel(ky=k).values
        omega[row, column], gamma[row, column] = frequency[0], frequency[1]
        converged[row, column] = is_converged(reasons[run])
        why[row, column] = reasons[run]
        names[row, column] = labels[run]

    source = datasets[0]["omega_average"]
    units = {"units": source.attrs["units"]} if "units" in source.attrs else {}
    dims = ("value", "ky")
    return xr.Dataset(
        {
            "gamma": (dims, gamma, {"long_name": "growth rate", **units}),
            "omega": (dims, omega, {"long_name": "real frequency", **units}),
            "converged": (dims, converged, {"long_name": "1 if GS2 stopped the run on its frequency converging"}),
            "stopped": (dims, why, {"long_name": "why the run stopped"}),
            "run_name": (dims, names, {"long_name": "the run"}),
        },
        coords={
            "value": ("value", line_values, {"long_name": name}),
            "ky": ("ky", ky, dict(datasets[0]["ky"].attrs)),
        },
        attrs={
            "parameter": name,
            "frequency_source": "omega_average at the run's last output, averaged over navg steps",
        },
    )


def compute_linear_gamma_max(
    datasets: Sequence[xr.Dataset], vary: str, quiet: bool = False
) -> xr.Dataset:
    """What :func:`plot_linear_gamma_max` draws: for each value of `vary`,
    the largest growth rate of the runs that converged (``gamma_max``), its
    ky (``ky_max``) and real frequency (``omega_max``), and how many runs
    did not converge (``unconverged``).

    Raises
    ------
    ValueError
        If `vary` doesn't take numbers.
    """
    spectrum = compute_linear_spectrum(datasets, vary, quiet=quiet)
    if spectrum["value"].dtype.kind not in "fiu":
        raise ValueError(f"{vary!r} doesn't take numbers, so gamma_max can't be drawn against it.")
    good = spectrum["converged"].values == 1
    gamma = np.where(good, spectrum["gamma"].values, -np.inf)
    peak = gamma.argmax(axis=1)
    rows = np.arange(spectrum.sizes["value"])
    found = good.any(axis=1)

    def at_peak(values: np.ndarray) -> np.ndarray:
        return np.where(found, values[rows, peak], np.nan)

    ky = np.broadcast_to(spectrum["ky"].values, good.shape)
    have = ~np.isnan(spectrum["gamma"].values)
    units = {k: v for k, v in spectrum["gamma"].attrs.items() if k == "units"}
    return xr.Dataset(
        {
            "gamma_max": ("value", at_peak(spectrum["gamma"].values), {"long_name": "largest converged growth rate", **units}),
            "ky_max": ("value", at_peak(ky), {"long_name": "ky of gamma_max", **spectrum["ky"].attrs}),
            "omega_max": ("value", at_peak(spectrum["omega"].values), {"long_name": "real frequency at gamma_max", **units}),
            "unconverged": ("value", (have & ~good).sum(axis=1), {"long_name": "runs that did not converge"}),
        },
        coords={"value": spectrum["value"]},
        attrs=dict(spectrum.attrs),
    )


def compute_linear_modes(
    datasets: Sequence[xr.Dataset],
    vary: str | None = None,
    ky: float | None = None,
    quiet: bool = False,
) -> xr.Dataset:
    """What :func:`plot_linear_modes` draws: for each value of `vary`, the
    final fields along the field line of the run at its fastest-growing
    converged ky, or the ky nearest `ky`.

    Each field is turned by the phase that makes ``phi`` real and positive
    at its peak, so their parities can be read off and compared, and scaled
    to a peak of 1. They are stored by ``point`` along each run's own theta
    (``theta``, NaN-padded), as runs may differ in ``nperiod``.
    """
    datasets = list(datasets)
    spectrum = compute_linear_spectrum(datasets, vary, quiet=quiet)
    _, _, _, where = _positions(datasets, vary)
    ky_values = spectrum["ky"].values

    chosen = []
    for row in range(spectrum.sizes["value"]):
        have = ~np.isnan(spectrum["gamma"].values[row])
        if ky is not None:
            candidates = np.flatnonzero(have)
            column = candidates[np.abs(ky_values[candidates] - ky).argmin()]
        else:
            good = have & (spectrum["converged"].values[row] == 1)
            if not good.any():
                continue
            column = int(np.where(good, spectrum["gamma"].values[row], -np.inf).argmax())
        chosen.append((row, column, *where[(row, int(column))]))
    if not chosen:
        raise ValueError("No line has a converged run to draw the eigenfunctions of.")

    fields = [f for f in FIELDS if any(f in datasets[run] for _, _, run, _ in chosen)]
    if "phi" not in fields:
        raise ValueError("The runs have no final fields: set write_fields in gs2_diagnostics_knobs.")
    npoints = max(datasets[run].sizes["theta"] for _, _, run, _ in chosen)
    theta = np.full((len(chosen), npoints), np.nan)
    shapes = {f: np.full((len(chosen), npoints, 2), np.nan) for f in fields}
    for i, (_, _, run, k) in enumerate(chosen):
        ds = datasets[run]
        theta[i, : ds.sizes["theta"]] = ds["theta"].values
        complex_fields = {
            f: ds[f].isel(ky=k, kx=0, ri=0).values + 1j * ds[f].isel(ky=k, kx=0, ri=1).values
            for f in fields
            if f in ds
        }
        phi = complex_fields["phi"]
        peak = phi[np.abs(phi).argmax()]
        turn = np.conj(peak) / abs(peak) if peak != 0 else 1.0
        for f, values in complex_fields.items():
            size = np.abs(values).max()
            if size > 0:
                shaped = values * turn / size
                shapes[f][i, : len(values)] = np.stack([shaped.real, shaped.imag], axis=-1)

    rows = [row for row, _, _, _ in chosen]
    columns = [column for _, column, _, _ in chosen]
    dims = ("value", "point")
    reduced = xr.Dataset(
        {
            "theta": (dims, theta, dict(datasets[chosen[0][2]]["theta"].attrs)),
            **{
                f: (
                    (*dims, "ri"),
                    shapes[f],
                    {
                        "long_name": f"{FIELDS[f]} turned to make phi real at its peak, and scaled to a peak of 1",
                    },
                )
                for f in fields
            },
            "ky": ("value", ky_values[columns], dict(spectrum["ky"].attrs)),
            "gamma": ("value", spectrum["gamma"].values[rows, columns], dict(spectrum["gamma"].attrs)),
            "omega": ("value", spectrum["omega"].values[rows, columns], dict(spectrum["omega"].attrs)),
            "converged": ("value", spectrum["converged"].values[rows, columns]),
            "run_name": ("value", spectrum["run_name"].values[rows, columns]),
        },
        coords={"value": spectrum["value"].isel(value=rows), "ri": ("ri", [0, 1])},
        attrs={**spectrum.attrs, "ky_asked": "fastest-growing" if ky is None else float(ky)},
    )
    return reduced


def _line_label(reduced: xr.Dataset, value: object) -> str | None:
    parameter = reduced.attrs["parameter"]
    if not parameter:
        return None
    return f"{parameter.split('.')[-1]} = {_value_label(value)}"


class _PlainLogFormatter(ticker.LogFormatter):
    """Labels the ticks matplotlib's would, as 0.2 rather than 2e-01."""

    def _num_to_string(self, x, vmin, vmax):
        return f"{x:g}"


def render_linear_spectrum(reduced: xr.Dataset) -> Figure:
    """Draw the results of :func:`compute_linear_spectrum`: the growth rate
    above the real frequency, against ky on a log scale, a line for each
    value, with runs that didn't converge as open markers."""
    fig, axes = common.subplots(2, 1, aspect=0.45, sharex=True)
    ky = reduced["ky"].values
    values = reduced["value"].values
    converged = reduced["converged"].values == 1
    for ax, name in zip(axes[:, 0], ("gamma", "omega")):
        rows = reduced[name].values
        for i, colour in enumerate(_run_colours(len(values))):
            have = ~np.isnan(rows[i])
            ax.plot(ky[have], rows[i][have], color=colour, label=_line_label(reduced, values[i]))
            good = have & converged[i]
            ax.plot(ky[good], rows[i][good], "o", color=colour, markersize=3)
            bad = have & ~converged[i]
            ax.plot(ky[bad], rows[i][bad], "o", color=colour, markersize=3, markerfacecolor="none")
        ax.axhline(0.0, color="0.5", linewidth=0.6)
        ax.set_ylabel(LABELS[name])
    axes[-1, 0].set_xscale("log")
    # Plain numbers, on the minor ticks too over a decade or two.
    numbers = _PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
    axes[-1, 0].xaxis.set_major_formatter(numbers)
    axes[-1, 0].xaxis.set_minor_formatter(numbers)
    axes[-1, 0].set_xlabel(LABELS["ky"])
    if (~np.isnan(reduced["gamma"].values) & ~converged).any():
        axes[0, 0].plot([], [], "o", color="0.4", markersize=3, markerfacecolor="none", label="not converged")
    if reduced.attrs["parameter"] or not converged.all():
        common.shared_legend(fig, axes[:1, 0])
    return fig


def render_linear_gamma_max(reduced: xr.Dataset) -> Figure:
    """Draw the results of :func:`compute_linear_gamma_max`: the largest
    growth rate, its ky and its real frequency against the parameter."""
    fig, axes = common.subplots(3, 1, aspect=0.3, sharex=True)
    x = reduced["value"].values
    for ax, name, label in zip(
        axes[:, 0],
        ("gamma_max", "ky_max", "omega_max"),
        (r"$\gamma_\mathrm{max}\, L_\mathrm{ref}/v_\mathrm{ref}$", LABELS["ky"], LABELS["omega"]),
    ):
        ax.plot(x, reduced[name].values, "o-", color="k", markersize=3)
        ax.set_ylabel(label)
    axes[0, 0].axhline(0.0, color="0.5", linewidth=0.6)
    axes[2, 0].axhline(0.0, color="0.5", linewidth=0.6)
    axes[-1, 0].set_xlabel(reduced.attrs["parameter"].split(".")[-1])
    return fig


def render_linear_modes(reduced: xr.Dataset) -> Figure:
    """Draw the results of :func:`compute_linear_modes`: a panel for each
    field against theta, its real part solid and imaginary part dotted, a
    line for each value."""
    fields = [f for f in FIELDS if f in reduced]
    fig, axes = common.subplots(len(fields), 1, aspect=0.4, sharex=True)
    theta = reduced["theta"].values
    colours = _run_colours(reduced.sizes["value"])
    for ax, field in zip(axes[:, 0], fields):
        shapes = reduced[field].values
        for i, colour in enumerate(colours):
            have = ~np.isnan(theta[i]) & ~np.isnan(shapes[i, :, 0])
            label = _line_label(reduced, reduced["value"].values[i])
            if label is not None:
                label += rf", $k_y \rho = {reduced['ky'].values[i]:g}$"
            ax.plot(theta[i][have] / np.pi, shapes[i, have, 0], color=colour, label=label)
            ax.plot(theta[i][have] / np.pi, shapes[i, have, 1], color=colour, linestyle=":")
        ax.axhline(0.0, color="0.5", linewidth=0.6)
        ax.set_ylabel(FIELDS[field])
    axes[0, 0].plot([], [], color="0.4", label="real part")
    axes[0, 0].plot([], [], color="0.4", linestyle=":", label="imaginary part")
    axes[-1, 0].set_xlabel(r"$\theta/\pi$")
    common.shared_legend(fig, axes[:1, 0])
    return fig


def _filename(stem: str):
    def name(reduced: xr.Dataset) -> str:
        parameter = reduced.attrs["parameter"].split(".")[-1]
        return f"linear_{parameter}_{stem}" if parameter else f"linear_{stem}"

    return name


def plot_linear_spectrum(
    datasets: Sequence[xr.Dataset],
    vary: str | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Draw the growth rate and real frequency of the linear runs
    `datasets` against ky, a line for each value of `vary`, a key of their
    input decks (or one line without it). The filename defaults to
    ``linear_<vary>_spectrum``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "linear_spectrum",
        list(datasets),
        dict(vary=vary),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def plot_linear_gamma_max(
    datasets: Sequence[xr.Dataset],
    vary: str,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Draw the largest converged growth rate of the linear runs `datasets`
    for each value of `vary`, with its ky and real frequency. The filename
    defaults to ``linear_<vary>_gamma_max``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "linear_gamma_max",
        list(datasets),
        dict(vary=vary),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


def plot_linear_modes(
    datasets: Sequence[xr.Dataset],
    vary: str | None = None,
    ky: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Draw the eigenfunctions of the linear runs `datasets` along the field
    line, for each value of `vary` at its fastest-growing ky, or the ky
    nearest `ky`. The filename defaults to ``linear_<vary>_modes``.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    return make_plot(
        "linear_modes",
        list(datasets),
        dict(vary=vary, ky=ky),
        output_dir=output_dir,
        filename=filename,
        save_data=save_data,
        quiet=quiet,
    )


register(PlotSpec("linear_spectrum", compute_linear_spectrum, render_linear_spectrum, _filename("spectrum")))
register(PlotSpec("linear_gamma_max", compute_linear_gamma_max, render_linear_gamma_max, _filename("gamma_max")))
register(PlotSpec("linear_modes", compute_linear_modes, render_linear_modes, _filename("modes")))


__all__ = [
    "FIELDS",
    "compute_linear_gamma_max",
    "compute_linear_modes",
    "compute_linear_spectrum",
    "plot_linear_gamma_max",
    "plot_linear_modes",
    "plot_linear_spectrum",
    "render_linear_gamma_max",
    "render_linear_modes",
    "render_linear_spectrum",
]
