"""Plot bundles: the reduced data behind a figure, with its provenance.

Every static plot is made in two stages. ``compute`` reduces the GS2 output to
the few arrays the figure draws (a time average, a theta integral, ...) and
``render`` draws them. :func:`make_plot` runs both, and writes the reduced
data next to the figure as a self-describing netCDF *bundle*::

    transfer_theta_averaged.png   the figure
    transfer_theta_averaged.nc    its data, recipe and provenance

The figure is always rendered from the bundle's contents, never from the raw
output, so the bundle is guaranteed to hold what the figure shows. A bundle
records:

* the reduced arrays, with GS2's ``units`` and ``long_name`` and CF
  ``cell_methods`` describing any reduction over time;
* the *recipe* — the plot's name and every option given to ``compute`` and
  ``render`` — as JSON in the ``gs2ew_recipe`` attribute;
* the GS2 and gs2ew versions, commits, branches and dirty states, and the
  versions of Python and the numerical libraries;
* the GS2 output's path, size and UUID, and the input deck it was run with.

From a bundle, :func:`replot` redraws the figure in seconds without the raw
output, and :func:`reproduce` repeats the whole analysis from the raw output,
after checking it is the same run.

Each output directory also gets the run's input deck and a ``reproduce.sh``
listing the exact, pinned command for every bundle in it; see
:func:`write_run_files`.
"""

from __future__ import annotations

import inspect
import json
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from gs2ew import provenance
from gs2ew.io import OUTPUT_SUFFIX, find_run, load_gs2_output
from gs2ew.provenance import UNKNOWN, CodeVersion

#: Incremented if the layout of a bundle changes incompatibly.
BUNDLE_FORMAT = 1

#: Name of the generated script listing how to reproduce each bundle.
REPRODUCE_SCRIPT = "reproduce.sh"

#: Subdirectory :func:`reproduce` writes into by default, so a reproduction
#: never overwrites the original it is meant to be compared against.
REPRODUCED_DIR = "reproduced"


@dataclass(frozen=True)
class PlotSpec:
    """How to make one kind of plot.

    `compute` takes the GS2 dataset and returns the reduced data; it must also
    accept ``quiet``. `render` takes that reduced data and returns a figure.
    `filename` picks the default output filename from the reduced data (for
    example to mark a window average).
    """

    name: str
    compute: Callable[..., xr.Dataset]
    render: Callable[..., Figure]
    filename: Callable[[xr.Dataset], str]


#: Every registered plot, by name. Populated as the plot modules are imported.
PLOTS: dict[str, PlotSpec] = {}


def register(spec: PlotSpec) -> PlotSpec:
    """Add `spec` to :data:`PLOTS`."""
    if spec.name in PLOTS and PLOTS[spec.name] != spec:
        raise ValueError(f"A plot called {spec.name!r} is already registered.")
    PLOTS[spec.name] = spec
    return spec


def get_spec(name: str) -> PlotSpec:
    """The registered :class:`PlotSpec` called `name`."""
    # The plot modules register themselves on import.
    import gs2ew.plots  # noqa: F401

    try:
        return PLOTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown plot {name!r}; known plots are {sorted(PLOTS)}."
        ) from None


# --- Making a plot ------------------------------------------------------------


def make_plot(
    name: str,
    ds: xr.Dataset,
    compute_options: dict[str, Any],
    render_options: dict[str, Any] | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    dpi: int = 150,
    tight_bbox: bool = True,
    save_data: bool = True,
    quiet: bool = False,
) -> Path:
    """Compute, render and save the plot called `name`.

    The figure is rendered from the reduced data that is written to the
    bundle, so the two cannot disagree. With `save_data` False only the figure
    is written (movie frames use this).

    Returns
    -------
    Path
        Path to the saved figure. Its bundle has the same stem and a ``.nc``
        suffix.
    """
    spec = get_spec(name)
    render_options = dict(render_options or {})

    reduced = spec.compute(ds, **compute_options, quiet=quiet).load()
    if filename is None:
        filename = spec.filename(reduced)

    recipe = {
        "plot": name,
        "compute": compute_options,
        "render": render_options,
        "save": {"filename": filename, "dpi": dpi, "tight_bbox": tight_bbox},
    }

    fig = spec.render(reduced, **render_options)
    path = _save_figure(fig, Path(output_dir) / filename, dpi, tight_bbox, quiet)

    if save_data:
        write_bundle(reduced, ds, recipe, path.with_suffix(".nc"), quiet=quiet)
        write_run_files(ds, path.parent, quiet=True)

    return path


def _save_figure(
    fig: Figure, path: Path, dpi: int, tight_bbox: bool, quiet: bool
) -> Path:
    from gs2ew.plots._common import save_figure

    return save_figure(
        fig, path.parent, path.name, dpi=dpi, tight_bbox=tight_bbox, quiet=quiet
    )


# --- Writing and reading bundles ----------------------------------------------


def _jsonable(value: Any) -> Any:
    """`value` with tuples, numpy scalars and Paths turned into JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _netcdf_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """`attrs` made storable as netCDF attributes.

    netCDF has no boolean or null attribute types, so booleans are stored as
    ``"true"``/``"false"`` and None is dropped.
    """
    out = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            value = "true" if value else "false"
        out[key] = value
    return out


# Versions already warned about, so a plot set warns once rather than per plot.
_dirty_warned: set[tuple[str, str]] = set()


def _warn_if_dirty(version: CodeVersion) -> None:
    key = (version.name, version.commit)
    if version.dirty and key not in _dirty_warned:
        _dirty_warned.add(key)
        print(
            f"Warning: {version.name} {version.version} ({version.commit[:9]}) "
            "had uncommitted changes; its commit does not pin this result."
        )


# Edited input decks already warned about, so each is reported once.
_deck_warned: set[Path] = set()


def _input_deck(ds: xr.Dataset) -> tuple[str | None, str]:
    """The input deck to record and where it came from.

    The deck GS2 embedded in its output is what the run read, so it decides.
    But GS2 stores it lossily (tabs become spaces, trailing whitespace is
    dropped), so when the ``.in`` next to the output is the same deck, that
    file is recorded verbatim instead. Without an embedded deck the ``.in`` is
    used anyway, marked as unverified.
    """
    deck = provenance.input_deck(ds)
    run = _source_run(ds)
    on_disk = run.input_file if run is not None else None

    if deck is not None:
        if on_disk is None:
            return deck, "embedded in the GS2 output"
        if provenance.deck_matches(deck, on_disk):
            return (
                on_disk.read_text(),
                f"{on_disk.name} from the run directory, verified against the "
                "deck embedded in the GS2 output",
            )
        if on_disk not in _deck_warned:
            _deck_warned.add(on_disk)
            print(
                f"Warning: {on_disk} differs from the input deck embedded in "
                "the output; recording the embedded one, which the run read."
            )
        return deck, (
            f"embedded in the GS2 output ({on_disk.name} in the run directory "
            "has been edited since)"
        )

    if on_disk is not None:
        return (
            on_disk.read_text(),
            f"{on_disk.name} from the run directory; unverified, as GS2 did "
            "not embed the deck in its output",
        )
    return None, "none found"


def _source_run(ds: xr.Dataset):
    """The :class:`~gs2ew.io.RunFiles` `ds` was read from, or None."""
    source = ds.encoding.get("source")
    if not source or not str(source).endswith(OUTPUT_SUFFIX):
        return None
    try:
        return find_run(source)
    except (FileNotFoundError, ValueError):
        return None


def provenance_attrs(ds: xr.Dataset) -> dict[str, str]:
    """Every provenance attribute a bundle made from `ds` carries."""
    gs2 = provenance.gs2_version(ds)
    gs2ew = provenance.gs2ew_version()
    _warn_if_dirty(gs2)
    _warn_if_dirty(gs2ew)

    attrs = {
        **gs2.attrs(),
        **gs2ew.attrs(),
        **provenance.source_identity(ds),
        "dependency_versions": json.dumps(provenance.dependency_versions()),
    }
    return attrs


def write_bundle(
    reduced: xr.Dataset,
    ds: xr.Dataset,
    recipe: dict[str, Any],
    path: str | Path,
    quiet: bool = False,
) -> Path:
    """Write `reduced`, with `recipe` and the provenance of `ds`, to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = reduced.copy()
    out.attrs = _netcdf_attrs(dict(reduced.attrs))
    for name in out.variables:
        out[name].attrs = _netcdf_attrs(dict(out[name].attrs))

    deck, deck_origin = _input_deck(ds)
    if deck is not None:
        out["input_file"] = ((), deck, {"long_name": "GS2 input file"})
    used = provenance.input_deck(ds, used=True)
    if used is not None:
        out["used_input_file"] = (
            (),
            used,
            {"long_name": "GS2 input file with every default filled in"},
        )

    gs2ew = provenance.gs2ew_version()
    out.attrs.update(
        {
            "title": f"gs2ew {recipe['plot']} plot data",
            "gs2ew_bundle_format": BUNDLE_FORMAT,
            "gs2ew_plot": recipe["plot"],
            "gs2ew_recipe": json.dumps(_jsonable(recipe)),
            "gs2ew_invocation": shlex.join(sys.argv),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_file_origin": deck_origin,
            **provenance_attrs(ds),
            "replot_command": f"gs2ew replot {shlex.quote(path.name)}",
            "reproduce_command": reproduce_command(path.name, gs2ew),
        }
    )

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in out.data_vars.items()
        if var.dtype.kind in "fiuc"
    }
    # CF advises against fill values on coordinates; they are never missing.
    encoding.update({name: {"_FillValue": None} for name in out.coords})
    out.to_netcdf(path, engine="netcdf4", encoding=encoding)

    if not quiet:
        print(f"Saved {path}")
    return path


def reproduce_command(bundle_name: str, gs2ew: CodeVersion | None = None) -> str:
    """The pinned command that repeats the analysis behind `bundle_name`.

    Pinned to gs2ew's exact commit through ``uvx``, so it runs the same code
    regardless of what is installed. Without a known commit it falls back to
    whatever ``gs2ew`` is on the path.
    """
    gs2ew = gs2ew or provenance.gs2ew_version()
    command = f"gs2ew reproduce {shlex.quote(bundle_name)}"
    if gs2ew.commit == UNKNOWN:
        return command
    source = f"git+{gs2ew.repository}@{gs2ew.commit}"
    return f"uvx --from {shlex.quote(source)} {command}"


def read_bundle(path: str | Path) -> xr.Dataset:
    """Load a bundle fully into memory, so the file is not held open.

    Raises
    ------
    ValueError
        If `path` is not a gs2ew bundle.
    """
    with xr.open_dataset(path, engine="netcdf4") as ds:
        bundle = ds.load()
    if "gs2ew_plot" not in bundle.attrs:
        raise ValueError(f"{path} is not a gs2ew plot bundle.")
    return bundle


def recipe(bundle: xr.Dataset) -> dict[str, Any]:
    """The recipe recorded in `bundle`."""
    return json.loads(bundle.attrs["gs2ew_recipe"])


def reduced_data(bundle: xr.Dataset) -> xr.Dataset:
    """`bundle` without the input decks, i.e. what ``compute`` returned."""
    return bundle.drop_vars(
        [v for v in ("input_file", "used_input_file") if v in bundle]
    )


def data_differences(a: str | Path, b: str | Path) -> dict[str, float]:
    """Max absolute difference of each variable that differs between bundles.

    Compares the reduced data and coordinates of the bundles at `a` and `b`,
    ignoring attributes and the input decks. A variable present in only one,
    or with a different shape, differs by infinity. An empty result means the
    data are identical, NaNs included.
    """
    data_a, data_b = reduced_data(read_bundle(a)), reduced_data(read_bundle(b))
    differences = {}
    for name in sorted(set(data_a.variables) | set(data_b.variables)):
        if name not in data_a.variables or name not in data_b.variables:
            differences[name] = float("inf")
            continue
        va, vb = data_a[name].values, data_b[name].values
        if va.shape != vb.shape:
            differences[name] = float("inf")
        elif va.dtype.kind in "fc" or vb.dtype.kind in "fc":
            if not np.array_equal(va, vb, equal_nan=True):
                # NaNs in different places are a difference no number measures.
                same_nans = np.array_equal(np.isnan(va), np.isnan(vb))
                differences[name] = (
                    float(np.nanmax(np.abs(va - vb))) if same_nans else float("inf")
                )
        elif not np.array_equal(va, vb):
            differences[name] = float("inf")
    return differences


# --- Using bundles ------------------------------------------------------------


def replot(
    path: str | Path,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    quiet: bool = False,
    **render_overrides: Any,
) -> Path:
    """Redraw the figure in the bundle at `path` from its reduced data.

    Needs no GS2 output, so takes seconds. By default the figure is written
    where the recipe says, next to the bundle; with no overrides it is
    identical to the original.

    `render_overrides` replace the recorded render options (e.g.
    ``normalise=True``); the plot's ``render_<plot>`` function lists them.
    Since the recipe in the original bundle would then no longer describe the
    figure, a bundle with the updated recipe is written alongside the new
    figure, so every figure keeps a bundle that describes it exactly.

    Raises
    ------
    ValueError
        If a render override is not an option of this plot.
    """
    path = Path(path)
    bundle = read_bundle(path)
    spec = get_spec(bundle.attrs["gs2ew_plot"])
    rec = recipe(bundle)

    known = set(inspect.signature(spec.render).parameters) - {"reduced"}
    unknown = set(render_overrides) - known
    if unknown:
        raise ValueError(
            f"{spec.name} has no render option {', '.join(sorted(unknown))}; "
            f"it has {', '.join(sorted(known)) or 'none'}."
        )

    render_options = {**rec["render"], **render_overrides}
    save = rec["save"]
    output_dir = Path(output_dir) if output_dir is not None else path.parent
    filename = filename or save["filename"]

    fig = spec.render(reduced_data(bundle), **render_options)
    figure = _save_figure(
        fig, output_dir / filename, save["dpi"], save["tight_bbox"], quiet
    )

    target = figure.with_suffix(".nc")
    if render_overrides or target.resolve() != path.resolve():
        updated = bundle.copy()
        rec["render"] = _jsonable(render_options)
        rec["save"]["filename"] = filename
        gs2ew = provenance.gs2ew_version()
        updated.attrs.update(
            {
                "gs2ew_recipe": json.dumps(rec),
                "replot_command": f"gs2ew replot {shlex.quote(target.name)}",
                # The render code may differ from the analysis code.
                **{
                    key.replace("gs2ew_", "gs2ew_render_", 1): value
                    for key, value in gs2ew.attrs().items()
                },
            }
        )
        updated.to_netcdf(target, engine="netcdf4")
        write_reproduce_script(target.parent)
        if "input_file" in bundle:
            # Replotting into a new directory: bring the input deck along.
            source = Path(bundle.attrs.get("source_path", UNKNOWN)).name
            name = source[: -len(OUTPUT_SUFFIX)] if source.endswith(OUTPUT_SUFFIX) else "input"
            deck = target.parent / f"{name}.in"
            if not deck.exists():
                deck.write_text(bundle["input_file"].item())
        if not quiet:
            print(f"Saved {target}")

    return figure


class SourceMismatchError(RuntimeError):
    """The GS2 output given to :func:`reproduce` is not the one recorded."""


def reproduce(
    path: str | Path,
    run: str | Path | None = None,
    output_dir: str | Path | None = None,
    quiet: bool = False,
) -> Path:
    """Repeat the analysis recorded in the bundle at `path` from raw output.

    `run` is the run directory or ``.out.nc`` to read; by default the path
    recorded in the bundle. Before anything is computed, the output's UUID is
    checked against the recorded one, so a moved or renamed run is fine but a
    different run is refused.

    Results go to a ``reproduced/`` directory next to the bundle unless
    `output_dir` is given, so the original is kept for comparison.

    Raises
    ------
    SourceMismatchError
        If the output's UUID differs from the recorded one.
    """
    path = Path(path)
    bundle = read_bundle(path)
    rec = recipe(bundle)

    if run is None:
        recorded = bundle.attrs.get("source_path", UNKNOWN)
        if recorded == UNKNOWN:
            raise ValueError(
                f"{path.name} does not record where its GS2 output was; "
                "pass the run directory."
            )
        run = recorded
    out_nc = find_run(run).out_nc

    if output_dir is None:
        output_dir = path.parent / REPRODUCED_DIR

    with load_gs2_output(out_nc) as ds:
        expected = bundle.attrs.get("source_id", UNKNOWN)
        actual = str(ds.attrs.get("id", UNKNOWN))
        if expected != UNKNOWN and actual != expected:
            raise SourceMismatchError(
                f"{out_nc} is GS2 run {actual}, but {path.name} was made from "
                f"run {expected}."
            )
        if expected == UNKNOWN:
            print(
                "Warning: the recorded GS2 output has no run UUID, so it cannot "
                "be checked that this is the same run."
            )

        return make_plot(
            rec["plot"],
            ds,
            rec["compute"],
            rec["render"],
            output_dir=output_dir,
            filename=rec["save"]["filename"],
            dpi=rec["save"]["dpi"],
            tight_bbox=rec["save"]["tight_bbox"],
            quiet=quiet,
        )


# --- Per-directory files ------------------------------------------------------


def write_run_files(
    ds: xr.Dataset, output_dir: str | Path, quiet: bool = False
) -> list[Path]:
    """Write the run's input deck and ``reproduce.sh`` into `output_dir`.

    The deck is written as ``<run name>.in``. ``reproduce.sh`` is regenerated
    from every bundle in `output_dir`, so it stays complete however many
    times plots are added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    deck, _ = _input_deck(ds)
    if deck is not None:
        run = _source_run(ds)
        name = run.name if run is not None else "input"
        deck_path = output_dir / f"{name}.in"
        deck_path.write_text(deck)
        written.append(deck_path)

    written.append(write_reproduce_script(output_dir))

    if not quiet:
        for p in written:
            print(f"Saved {p}")
    return written


def write_reproduce_script(directory: str | Path) -> Path:
    """(Re)write ``reproduce.sh`` in `directory` from the bundles there."""
    script = Path(directory) / REPRODUCE_SCRIPT
    script.write_text(reproduce_script(directory))
    script.chmod(0o755)
    return script


def _bundles_in(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """(path, global attributes) for every bundle directly in `directory`."""
    found = []
    for nc in sorted(directory.glob("*.nc")):
        try:
            with xr.open_dataset(nc, engine="netcdf4") as ds:
                attrs = dict(ds.attrs)
        except (OSError, ValueError):
            continue
        if "gs2ew_plot" in attrs:
            found.append((nc, attrs))
    return found


def reproduce_script(directory: str | Path) -> str:
    """The text of ``reproduce.sh`` for the bundles in `directory`."""
    directory = Path(directory)
    lines = [
        "#!/bin/sh",
        "# Generated by gs2ew.",
        "#",
        "# Each command below repeats the analysis behind one figure from the raw",
        "# GS2 output, after checking it is the same run, and writes the result",
        f"# to ./{REPRODUCED_DIR}/ for comparison. If the run has moved, add",
        "# `--run <run directory>`.",
        "#",
        "# To redraw a figure from its bundle alone, without the raw output:",
        "#   gs2ew replot <bundle>.nc",
        'cd "$(dirname "$0")"',
        "",
    ]

    for nc, attrs in _bundles_in(directory):
        lines.append(f"# {nc.name}: {attrs.get('gs2ew_plot')}")
        for code in ("gs2", "gs2ew"):
            version = attrs.get(f"{code}_version", UNKNOWN)
            commit = attrs.get(f"{code}_commit", UNKNOWN)
            lines.append(f"#   {code} {version} ({commit})")
            if attrs.get(f"{code}_dirty") == "true":
                lines.append(
                    f"#   WARNING: {code} had uncommitted changes, so its commit "
                    "does not pin this result."
                )
        lines.append(f"#   source: {attrs.get('source_path', UNKNOWN)}")
        lines.append(str(attrs.get("reproduce_command")))
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "BUNDLE_FORMAT",
    "PLOTS",
    "PlotSpec",
    "SourceMismatchError",
    "data_differences",
    "get_spec",
    "make_plot",
    "provenance_attrs",
    "read_bundle",
    "recipe",
    "reduced_data",
    "register",
    "replot",
    "reproduce",
    "reproduce_command",
    "reproduce_script",
    "write_bundle",
    "write_reproduce_script",
    "write_run_files",
]
