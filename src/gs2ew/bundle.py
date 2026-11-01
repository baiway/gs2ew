"""Plot bundles: the reduced data behind a figure, with its provenance.

Every static plot is made in two stages. ``compute`` reduces the GS2 output to
the few arrays the figure draws (a time average, a theta integral, ...) and
``render`` draws them. :func:`make_plot` runs both, and writes the reduced
data next to the figure as a self-describing netCDF *bundle*::

    transfer_theta_averaged.pdf   the figure, for print
    transfer_theta_averaged.png   the same figure, for viewing
    transfer_theta_averaged.nc    its data, recipe and provenance

The figure is always rendered from the bundle's contents, never from the raw
output, so the bundle is guaranteed to hold what the figure shows. A bundle
records:

* the reduced arrays, with GS2's ``units`` and ``long_name`` and CF
  ``cell_methods`` describing any reduction over time;
* the *recipe* — the plot's name, every option given to ``compute`` and
  ``render``, and the figure's style and formats (see :mod:`gs2ew.style`) —
  as JSON in the ``gs2ew_recipe`` attribute;
* the GS2 and gs2ew versions, commits, branches and dirty states, and the
  versions of Python and the numerical libraries;
* the GS2 output's path, size and UUID, and the input deck it was run with.

A plot comparing several runs (see :mod:`gs2ew.plots.compare`) is made from a
list of datasets: its bundle keeps each run's input deck along a ``run``
dimension, and each run's path, UUID and GS2 version in ``gs2ew_sources``.

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
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure

from gs2ew import provenance, style
from gs2ew.io import OUTPUT_SUFFIX, find_run, load_gs2_output
from gs2ew.provenance import UNKNOWN, CodeVersion
from gs2ew.style import FigureOptions
from gs2ew.timebase import TimeWindow

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

    `compute` takes the GS2 dataset, or a list of them for a plot comparing
    runs, and returns the reduced data; it must also accept ``quiet``. `render` takes that reduced data and returns a figure.
    `filename` picks the default output filename, without a suffix, from the
    reduced data (for example to mark a window average).
    """

    name: str
    compute: Callable[..., xr.Dataset]
    render: Callable[..., Figure]
    filename: Callable[[xr.Dataset], str]


#: The GS2 output a plot is made from: one run's dataset, or a list of them,
#: one per run, for a plot comparing runs.
Sources = xr.Dataset | Sequence[xr.Dataset]


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
    ds: Sources,
    compute_options: dict[str, Any],
    render_options: dict[str, Any] | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    tight_bbox: bool = False,
    save_data: bool = True,
    quiet: bool = False,
    options: FigureOptions | None = None,
) -> Path:
    """Compute, render and save the plot called `name`.

    The figure is rendered from the reduced data that is written to the
    bundle, so the two cannot disagree. With `save_data` False only the figure
    is written (movie frames use this).

    The figure is saved in each of `options`' formats (by default those set
    by :func:`gs2ew.style.figure_options`), unless `filename` ends in one, in
    which case only that format is saved.

    Returns
    -------
    Path
        Path to the saved figure, in the first format. Its bundle has the same
        stem and a ``.nc`` suffix.
    """
    spec = get_spec(name)
    render_options = dict(render_options or {})
    options = options or style.current()

    reduced = spec.compute(ds, **compute_options, quiet=quiet).load()
    stem, options = _split_filename(filename or spec.filename(reduced), options)

    recipe = {
        "plot": name,
        "compute": compute_options,
        "render": render_options,
        "figure": _figure_recipe(options),
        "save": {"filename": stem, "tight_bbox": tight_bbox},
    }

    path = _draw(
        spec, reduced, render_options, options, Path(output_dir) / stem, tight_bbox, quiet
    )

    if save_data:
        write_bundle(reduced, ds, recipe, path.with_suffix(".nc"), quiet=quiet)
        write_run_files(ds, path.parent, quiet=True)

    return path


#: File types matplotlib can save, which a filename's suffix may name.
_FORMATS = set(FigureCanvasBase.get_supported_filetypes())


def _split_filename(filename: str, options: FigureOptions) -> tuple[str, FigureOptions]:
    """`filename` without a format suffix, and the formats to save it in."""
    path = Path(filename)
    suffix = path.suffix.lstrip(".").lower()
    if suffix in _FORMATS:
        return str(path.with_suffix("")), replace(options, formats=(suffix,))
    return filename, options


def _figure_recipe(options: FigureOptions) -> dict[str, Any]:
    return {name: getattr(options, name) for name in FigureOptions.names()}


def _recorded_options(rec: dict[str, Any]) -> tuple[str, FigureOptions]:
    """The filename stem and figure options a recipe records.

    Bundles from before figure options were recorded name a single PNG and
    its resolution in their ``save`` section.
    """
    save = rec["save"]
    if "figure" in rec:
        return save["filename"], FigureOptions(**rec["figure"])
    return _split_filename(save["filename"], FigureOptions(dpi=save.get("dpi", 150)))


def _draw(
    spec: PlotSpec,
    reduced: xr.Dataset,
    render_options: dict[str, Any],
    options: FigureOptions,
    stem: Path,
    tight_bbox: bool,
    quiet: bool,
    embedded_style: str | None = None,
) -> Path:
    """Render `reduced` in `options`' style and save it in each format.

    Titles the figure with the averaging window, if there is one and
    `options` asks for it. Returns the path in the first format.
    """
    from gs2ew.plots._common import save_figure

    with style.context(options, embedded_style):
        fig = spec.render(reduced, **render_options)
        title = _window_title(reduced.attrs)
        if options.title and title is not None:
            # A legend above the panels takes the place a title would, so the
            # title heads the legend instead.
            if fig.legends:
                fig.legends[0].set_title(title)
            else:
                fig.suptitle(title)
        paths = [
            save_figure(
                fig,
                stem.parent,
                f"{stem.name}.{fmt}",
                dpi=options.dpi,
                tight_bbox=tight_bbox,
                close=False,
                quiet=quiet,
            )
            for fmt in options.formats
        ]
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths[0]


def _window_title(attrs: dict[str, Any]) -> str | None:
    """The averaging window, for a figure title; None if not averaged."""
    if not {"tstart", "tend", "averaging"} <= set(attrs):
        return None
    return TimeWindow.from_attrs(attrs).title


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
        if provenance.deck_restarted(deck, on_disk):
            return deck, (
                f"embedded in the GS2 output ({on_disk.name} in the run directory "
                "has since been set to restart)"
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


def provenance_attrs(ds: Sources) -> dict[str, str]:
    """Every provenance attribute a bundle made from `ds` carries.

    For several runs, each one's GS2 version and identity go in
    ``gs2ew_sources``, as a JSON list in the order of the runs; the GS2
    version attributes are those the runs share, or ``"mixed"``.
    """
    gs2ew = provenance.gs2ew_version()
    _warn_if_dirty(gs2ew)
    common = {
        **gs2ew.attrs(),
        "dependency_versions": json.dumps(provenance.dependency_versions()),
    }
    if isinstance(ds, xr.Dataset):
        gs2 = provenance.gs2_version(ds)
        _warn_if_dirty(gs2)
        return {**gs2.attrs(), **provenance.source_identity(ds), **common}

    sources = []
    for run in ds:
        gs2 = provenance.gs2_version(run)
        _warn_if_dirty(gs2)
        sources.append({**gs2.attrs(), **provenance.source_identity(run)})
    shared = {
        key: value if all(source[key] == value for source in sources) else "mixed"
        for key, value in sources[0].items()
        if key.startswith("gs2_")
    }
    return {**shared, "gs2ew_sources": json.dumps(sources), **common}


def _record_decks(out: xr.Dataset, ds: Sources) -> str:
    """Add the input decks of `ds` to `out`, and say where they came from.

    One run's go in scalar ``input_file`` and ``used_input_file``; several
    runs' along the ``run`` dimension, if every run has one.
    """
    runs = [ds] if isinstance(ds, xr.Dataset) else list(ds)
    decks = [_input_deck(run) for run in runs]
    used = [provenance.input_deck(run, used=True) for run in runs]
    dims: tuple[str, ...] = () if isinstance(ds, xr.Dataset) else ("run",)

    def value(texts: list[str | None]):
        return texts[0] if not dims else np.array(texts, dtype=object)

    if all(deck is not None for deck, _ in decks):
        out["input_file"] = (dims, value([d for d, _ in decks]), {"long_name": "GS2 input file"})
    if all(u is not None for u in used):
        out["used_input_file"] = (
            dims,
            value(used),
            {"long_name": "GS2 input file with every default filled in"},
        )
    if not dims:
        return decks[0][1]
    return "; ".join(f"run {i}: {origin}" for i, (_, origin) in enumerate(decks))


def write_bundle(
    reduced: xr.Dataset,
    ds: Sources,
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

    deck_origin = _record_decks(out, ds)

    style_text = style.style_file(recipe["figure"]["style"])
    if style_text is not None:
        out.attrs["gs2ew_style_file"] = style_text

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
    **overrides: Any,
) -> Path:
    """Redraw the figure in the bundle at `path` from its reduced data.

    Needs no GS2 output, so takes seconds. By default the figure is written
    where the recipe says, next to the bundle; with no overrides it is
    identical to the original.

    `overrides` replace recorded render options (e.g. ``normalise=True``;
    the plot's ``render_<plot>`` function lists them) or figure options
    (``style``, ``width``, ``title``, ``formats``, ``dpi``; see
    :class:`gs2ew.style.FigureOptions`). Since the recipe in the original
    bundle would then no longer describe the figure, a bundle with the updated
    recipe is written alongside the new figure, so every figure keeps a bundle
    that describes it exactly.

    Raises
    ------
    ValueError
        If an override is neither a render nor a figure option of this plot.
    """
    path = Path(path)
    bundle = read_bundle(path)
    spec = get_spec(bundle.attrs["gs2ew_plot"])
    rec = recipe(bundle)

    figure_names = set(FigureOptions.names())
    render_names = set(inspect.signature(spec.render).parameters) - {"reduced"}
    unknown = set(overrides) - render_names - figure_names
    if unknown:
        raise ValueError(
            f"{spec.name} has no render option {', '.join(sorted(unknown))}; "
            f"it has {', '.join(sorted(render_names)) or 'none'}, and every "
            f"plot has the figure options {', '.join(FigureOptions.names())}."
        )

    render_options = {
        **rec["render"],
        **{k: v for k, v in overrides.items() if k in render_names},
    }
    stem, options = _recorded_options(rec)
    options = replace(
        options, **{k: v for k, v in overrides.items() if k in figure_names}
    )
    if filename is not None:
        stem, options = _split_filename(filename, options)
    output_dir = Path(output_dir) if output_dir is not None else path.parent

    # A style file of the user's is redrawn from the copy in the bundle,
    # unless a new style was asked for.
    embedded = None if "style" in overrides else bundle.attrs.get("gs2ew_style_file")

    figure = _draw(
        spec,
        reduced_data(bundle),
        render_options,
        options,
        output_dir / stem,
        rec["save"].get("tight_bbox", False),
        quiet,
        embedded_style=embedded,
    )

    target = figure.with_suffix(".nc")
    if overrides or target.resolve() != path.resolve():
        updated = bundle.copy()
        rec["render"] = _jsonable(render_options)
        rec["figure"] = _figure_recipe(options)
        rec["save"] = {**rec["save"], "filename": stem}
        rec["save"].pop("dpi", None)
        if "style" in overrides:
            updated.attrs.pop("gs2ew_style_file", None)
            text = style.style_file(options.style)
            if text is not None:
                updated.attrs["gs2ew_style_file"] = text
        gs2ew = provenance.gs2ew_version()
        updated.attrs.update(
            {
                "gs2ew_recipe": json.dumps(_jsonable(rec)),
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
            # Replotting into a new directory: bring the input decks along.
            for name, text in _bundle_decks(bundle):
                deck = target.parent / f"{name}.in"
                if not deck.exists():
                    deck.write_text(text)
        if not quiet:
            print(f"Saved {target}")

    return figure


def _bundle_decks(bundle: xr.Dataset) -> list[tuple[str, str]]:
    """(run name, input deck) for each run recorded in `bundle`."""
    if bundle["input_file"].ndim == 0:
        paths = [bundle.attrs.get("source_path", UNKNOWN)]
        texts = [bundle["input_file"].item()]
    else:
        paths = [s.get("source_path", UNKNOWN) for s in _sources(bundle)]
        texts = [str(t) for t in bundle["input_file"].values]
    return [(_run_name(path), text) for path, text in zip(paths, texts)]


def _run_name(source_path: str) -> str:
    name = Path(source_path).name
    return name[: -len(OUTPUT_SUFFIX)] if name.endswith(OUTPUT_SUFFIX) else "input"


def _sources(bundle: xr.Dataset | dict[str, Any]) -> list[dict[str, str]] | None:
    """The runs a bundle comparing several was made from, or None."""
    attrs = bundle.attrs if isinstance(bundle, xr.Dataset) else bundle
    text = attrs.get("gs2ew_sources")
    return None if text is None else json.loads(text)


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
    different run is refused. A bundle comparing several runs reads each from
    its recorded path, and takes no `run`.

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
    if output_dir is None:
        output_dir = path.parent / REPRODUCED_DIR
    stem, options = _recorded_options(rec)
    remake = dict(
        output_dir=output_dir,
        filename=stem,
        tight_bbox=rec["save"].get("tight_bbox", False),
        quiet=quiet,
        options=options,
    )

    sources = _sources(bundle)
    if sources is not None:
        if run is not None:
            raise ValueError(
                f"{path.name} compares several runs, each read from where it was."
            )
        with ExitStack() as stack:
            datasets = []
            for source in sources:
                out_nc = find_run(source["source_path"]).out_nc
                ds = stack.enter_context(load_gs2_output(out_nc))
                _check_source(source.get("source_id", UNKNOWN), ds, out_nc, path)
                datasets.append(ds)
            return make_plot(rec["plot"], datasets, rec["compute"], rec["render"], **remake)

    if run is None:
        recorded = bundle.attrs.get("source_path", UNKNOWN)
        if recorded == UNKNOWN:
            raise ValueError(
                f"{path.name} does not record where its GS2 output was; "
                "pass the run directory."
            )
        run = recorded
    out_nc = find_run(run).out_nc

    with load_gs2_output(out_nc) as ds:
        _check_source(bundle.attrs.get("source_id", UNKNOWN), ds, out_nc, path)
        return make_plot(rec["plot"], ds, rec["compute"], rec["render"], **remake)


def _check_source(expected: str, ds: xr.Dataset, out_nc: Path, bundle: Path) -> None:
    """Refuse `ds` unless it is the GS2 run with UUID `expected`."""
    actual = str(ds.attrs.get("id", UNKNOWN))
    if expected != UNKNOWN and actual != expected:
        raise SourceMismatchError(
            f"{out_nc} is GS2 run {actual}, but {bundle.name} was made from "
            f"run {expected}."
        )
    if expected == UNKNOWN:
        print(
            f"Warning: {out_nc} has no recorded run UUID, so it cannot be "
            "checked that this is the same run."
        )


# --- Per-directory files ------------------------------------------------------


def write_run_files(
    ds: Sources, output_dir: str | Path, quiet: bool = False
) -> list[Path]:
    """Write the runs' input decks and ``reproduce.sh`` into `output_dir`.

    Each deck is written as ``<run name>.in``. ``reproduce.sh`` is regenerated
    from every bundle in `output_dir`, so it stays complete however many
    times plots are added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for run in [ds] if isinstance(ds, xr.Dataset) else ds:
        deck, _ = _input_deck(run)
        if deck is not None:
            files = _source_run(run)
            name = files.name if files is not None else "input"
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
        sources = _sources(attrs)
        if sources is None:
            lines.append(f"#   source: {attrs.get('source_path', UNKNOWN)}")
        for source in sources or []:
            lines.append(f"#   source: {source.get('source_path', UNKNOWN)}")
        lines.append(str(attrs.get("reproduce_command")))
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "BUNDLE_FORMAT",
    "PLOTS",
    "PlotSpec",
    "SourceMismatchError",
    "Sources",
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
