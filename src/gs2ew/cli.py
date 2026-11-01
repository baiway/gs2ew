"""Command-line interface.

Six subcommands::

    gs2ew info  RUN                        # what is in the file
    gs2ew plot  RUN                        # the standard plot set, in RUN/gs2ew/
    gs2ew batch project/                   # every .out.nc under a directory
    gs2ew replot    RUN/gs2ew/*.nc         # redraw figures from their bundles
    gs2ew reproduce RUN/gs2ew/*.nc         # repeat the analysis from raw output
    gs2ew scan write|submit|status ...     # write and run a parameter scan

RUN is a run directory holding one ``.out.nc``, or the ``.out.nc`` itself.
`plot` and `batch` take the same options, so a batch job produces exactly what
a single run would. Anything more bespoke than this is better written as a
short script against the Python API — see `examples/analysis.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

# A non-interactive backend: this is routinely run headless on a compute node.
# Must be selected before pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

import xarray as xr

from gs2ew import bundle, catalog, provenance, scan, style
from gs2ew.io import OUTPUT_SUFFIX, RunFiles, find_run, load_gs2_output
from gs2ew.metadata import get_metadata
from gs2ew.provenance import CodeVersion

#: Directory a run's plots go in by default, inside the run directory.
DEFAULT_OUTPUT_DIR = "gs2ew"

#: The standard plot set, in the order it is drawn. `--only` takes these names.
STANDARD_PLOTS = (
    "fields_time_traces",
    "fields_by_mode",
    "phi2_by_theta",
    "transfer_theta",
    "transfer_flux_surface",
    "transfer_summary",
    "transfer_hovmoller",
    "transfer_spectral",
    "transfer_spectrum",
    "transfer_butterfly",
    "transfer_trapped_passing",
    "transfer_trapped_passing_spectral",
)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by `plot` and `batch`."""
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=f"directory for plots (default: {DEFAULT_OUTPUT_DIR}/ in the run directory)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=STANDARD_PLOTS,
        metavar="PLOT",
        help=f"draw only these plots (and their movies); one of {', '.join(STANDARD_PLOTS)}",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="save the figures only, without their data bundles",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to plot unless gs2ew and GS2 were both built from clean, "
        "committed code",
    )
    parser.add_argument(
        "--tstart",
        default="saturation",
        help="window start: a time, or 'saturation' (default), or 'none' for "
        "the last timestep only",
    )
    parser.add_argument("--window", type=float, default=None, help="averaging window")
    parser.add_argument(
        "--saturation-pad",
        type=float,
        default=0.20,
        help="fraction of the post-saturation interval to skip (default 0.20)",
    )
    parser.add_argument(
        "--quantities",
        nargs="*",
        default=None,
        choices=list(catalog.QUANTITIES),
        help="restrict to these transferred quantities (default: all present)",
    )
    parser.add_argument(
        "--no-kinetic",
        action="store_true",
        help="omit the kinetic-energy transfer",
    )
    parser.add_argument(
        "--klim", type=float, default=None, help="clip wavevector axes to +/-KLIM"
    )
    parser.add_argument(
        "--style",
        default=None,
        help="a style to apply on top of gs2ew's print style: a .mplstyle file "
        "or a matplotlib style name",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="figure width in inches (default: the style's, 6.0 for print)",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        default=None,
        metavar="FORMAT",
        help=f"file types to save (default: {' '.join(style.DEFAULT_FORMATS)})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"resolution of PNGs and of heatmaps in PDFs (default: {style.DEFAULT_DPI})",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="leave the averaging window out of figure titles, for a caption to give",
    )
    parser.add_argument(
        "--movies", action="store_true", help="also render the movies (slow)"
    )
    parser.add_argument("--fps", type=int, default=10, help="movie frame rate")
    parser.add_argument(
        "--stride", type=int, default=1, help="keep every nth movie frame"
    )


def _tstart(value: str) -> float | str | None:
    """Parse the --tstart option into what the plot functions expect."""
    if value.lower() in {"none", "last"}:
        return None
    if value == "saturation":
        return "saturation"
    try:
        return float(value)
    except ValueError:
        raise SystemExit(
            f"--tstart must be a number, 'saturation' or 'none' (got {value!r})"
        ) from None


def _find_run(path: str) -> RunFiles:
    """:func:`~gs2ew.io.find_run`, reporting a bad path as a usage error."""
    try:
        return find_run(path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"gs2ew: {exc}") from None


def cmd_info(args: argparse.Namespace) -> int:
    """Print build metadata and the transfer diagnostics present."""
    run = _find_run(args.run)
    with load_gs2_output(run.out_nc) as ds:
        print(f"=== {run.out_nc} ===")
        for key, value in get_metadata(ds).items():
            print(f"  {key}: {value}")

        t = ds["t"].values
        print(f"\n  t: [{t[0]:.3g}, {t[-1]:.3g}] over {len(t)} steps")
        print(f"  species: {ds.sizes.get('species', 0)}, theta: {ds.sizes.get('theta', 0)}")

        print("\n=== Transfer diagnostics ===")
        for reduction in list(catalog.REDUCTIONS) + ["trapped_passing"]:
            drives = catalog.available_drives(ds, reduction)
            names = ", ".join(
                f"{d.quantity}/{d.field or '-'}" for d in drives
            )
            print(f"  {reduction:<16} {names or '(none)'}")

        print("\n=== Identity checks ===")
        for identity in catalog.IDENTITIES:
            for field in catalog.FIELDS:
                residual = catalog.identity_residual(ds, field, identity=identity)
                if residual == residual:  # not NaN
                    print(
                        f"  {identity:<20} {field:<5} "
                        f"max relative residual {residual:.2e}"
                    )
    return 0


def _plot_all(ds: xr.Dataset, args: argparse.Namespace, output_dir: Path) -> None:
    """Render the standard plot set, or the `--only` subset, for one dataset."""
    from gs2ew.plots import fields, transfer_spectral, transfer_summary
    from gs2ew.plots import transfer_surface, transfer_theta, transfer_time, transfer_tp

    window_opts = dict(
        tstart=_tstart(args.tstart),
        window=args.window,
        saturation_pad=args.saturation_pad,
    )
    # The trapped/passing diagnostics have no kinetic-energy counterpart, so
    # they take the quantity selection but not the kinetic options.
    select = dict(quantities=args.quantities, include_kinetic=not args.no_kinetic)
    tp_select = dict(quantities=args.quantities)
    # Each static plot saves its data alongside; movies have none to save.
    out = dict(output_dir=output_dir, save_data=not args.no_data)

    plots: list[tuple[str, Callable[[], object]]] = [
        ("fields_time_traces", partial(fields.plot_fields_time_traces, ds, **out)),
        ("fields_by_mode", partial(fields.plot_fields_by_mode, ds, **out)),
        ("phi2_by_theta", partial(fields.plot_phi2_by_theta, ds, **out)),
        (
            "transfer_theta",
            partial(
                transfer_theta.plot_transfer_theta,
                ds,
                show_std=True,
                **out,
                **window_opts,
                **select,
            ),
        ),
        (
            "transfer_summary",
            partial(
                transfer_summary.plot_transfer_summary, ds, **out, **window_opts, **select
            ),
        ),
        (
            "transfer_hovmoller",
            partial(transfer_time.plot_transfer_hovmoller, ds, **out, **select),
        ),
    ]

    for reduction in ("kxtarget", "kysource"):
        spectral = dict(reduction=reduction, klim=args.klim, **out, **select)
        plots += [
            (
                "transfer_spectral",
                partial(
                    transfer_spectral.plot_transfer_spectral, ds, **spectral, **window_opts
                ),
            ),
            (
                "transfer_spectrum",
                partial(
                    transfer_spectral.plot_transfer_spectrum, ds, **spectral, **window_opts
                ),
            ),
            (
                "transfer_butterfly",
                partial(transfer_time.plot_transfer_butterfly, ds, **spectral),
            ),
        ]

    # GS2 writes the surface's shape for most geometries, but not all.
    if all(name in ds for name in transfer_surface.GEOMETRY):
        plots.insert(
            4,
            (
                "transfer_flux_surface",
                partial(
                    transfer_surface.plot_transfer_flux_surface,
                    ds,
                    **out,
                    **window_opts,
                    **select,
                ),
            ),
        )

    # Trapped/passing output is optional in GS2, so only plot it if it is there.
    has_tp = bool(
        catalog.available_drives(
            ds, "trapped_passing", args.quantities, include_kinetic=False
        )
    )
    if has_tp:
        plots += [
            (
                "transfer_trapped_passing",
                partial(
                    transfer_tp.plot_transfer_trapped_passing,
                    ds,
                    show_std=True,
                    **out,
                    **window_opts,
                    **tp_select,
                ),
            ),
            (
                "transfer_trapped_passing_spectral",
                partial(
                    transfer_tp.plot_transfer_trapped_passing_spectral,
                    ds,
                    klim=args.klim,
                    **out,
                    **window_opts,
                ),
            ),
        ]

    if args.movies:
        movie_opts = dict(fps=args.fps, stride=args.stride, output_dir=output_dir)
        plots += [
            (
                "transfer_theta",
                partial(
                    transfer_theta.plot_transfer_theta_movie,
                    ds,
                    normalise="movie",
                    **movie_opts,
                    **select,
                ),
            ),
            (
                "transfer_spectral",
                partial(
                    transfer_spectral.plot_transfer_spectral_movie,
                    ds,
                    klim=args.klim,
                    **movie_opts,
                    **select,
                ),
            ),
        ]
        if has_tp:
            plots += [
                (
                    "transfer_trapped_passing",
                    partial(
                        transfer_tp.plot_transfer_trapped_passing_movie,
                        ds,
                        **movie_opts,
                        **tp_select,
                    ),
                ),
                (
                    "transfer_trapped_passing_spectral",
                    partial(
                        transfer_tp.plot_transfer_trapped_passing_spectral_movie,
                        ds,
                        klim=args.klim,
                        **movie_opts,
                    ),
                ),
            ]

    for name, plot in plots:
        if args.only is None or name in args.only:
            plot()


def _output_dir(run: RunFiles, base: Path | None = None) -> Path:
    """Where a run's plots go: `base`, or ``gs2ew/`` in the run directory.

    A directory holding several outputs gets a subdirectory per run, so their
    plots cannot overwrite each other.
    """
    output_dir = base if base is not None else run.run_dir / DEFAULT_OUTPUT_DIR
    if len(list(run.run_dir.glob(f"*{OUTPUT_SUFFIX}"))) > 1:
        output_dir /= run.name
    return output_dir


def _unclean(version: CodeVersion) -> str | None:
    """Why `version` fails `--require-clean`, or None if it passes."""
    if version.dirty is False:
        return None
    state = (
        "had uncommitted changes"
        if version.dirty
        else "has no record of whether it had uncommitted changes"
    )
    return (
        f"--require-clean: {version.name} {version.version} "
        f"({version.commit[:9]}) {state}."
    )


def _plot_run(run: RunFiles, args: argparse.Namespace, output_dir: Path) -> int:
    """Plot one run, refusing an unclean GS2 build under `--require-clean`."""
    with load_gs2_output(run.out_nc) as ds:
        if args.require_clean:
            reason = _unclean(provenance.gs2_version(ds))
            if reason is not None:
                print(f"{run.out_nc}: {reason}")
                return 1
        with style.figure_options(**_figure_options(args)):
            _plot_all(ds, args, output_dir)
    return 0


def _figure_options(args: argparse.Namespace) -> dict[str, object]:
    """The figure options given on the command line."""
    if args.style is not None:
        try:
            style.resolve(args.style)
        except ValueError as exc:
            raise SystemExit(f"gs2ew: {exc}") from None
    return dict(
        style=args.style,
        width=args.width,
        formats=args.formats,
        dpi=args.dpi,
        title=False if args.no_title else None,
    )


def _gs2ew_unclean(args: argparse.Namespace) -> bool:
    """Report and return True if `--require-clean` rules out this gs2ew."""
    if not args.require_clean:
        return False
    reason = _unclean(provenance.gs2ew_version())
    if reason is not None:
        print(reason)
    return reason is not None


def cmd_plot(args: argparse.Namespace) -> int:
    """Render the standard plot set for a single run."""
    if _gs2ew_unclean(args):
        return 1
    run = _find_run(args.run)
    # An explicit -o is used as given.
    output_dir = Path(args.output_dir) if args.output_dir is not None else _output_dir(run)
    return _plot_run(run, args, output_dir)


def cmd_batch(args: argparse.Namespace) -> int:
    """Render the standard plot set for every `.out.nc` under a directory.

    Each run's plots go in its own ``gs2ew/`` directory, or with `-o`, in a
    tree mirroring each run's path relative to `root`, so runs that share an
    input-file name do not collide. One failing run is reported and skipped
    rather than aborting the batch.
    """
    if _gs2ew_unclean(args):
        return 1
    root = Path(args.root)
    skip = set(args.skip or [])

    runs = [
        nc
        for nc in sorted(root.rglob(f"*{OUTPUT_SUFFIX}"))
        if not any(part in skip for part in nc.relative_to(root).parts)
    ]
    if not runs:
        print(f"No {OUTPUT_SUFFIX} files found under {root}")
        return 1

    print(f"Found {len(runs)} run(s) under {root}")

    failures = []
    for nc in runs:
        run = find_run(nc)
        base = (
            Path(args.output_dir) / nc.parent.relative_to(root)
            if args.output_dir is not None
            else None
        )
        output_dir = _output_dir(run, base)
        print(f"\n=== {nc.relative_to(root)} -> {output_dir} ===", flush=True)
        try:
            if _plot_run(run, args, output_dir) != 0:
                failures.append(nc)
        except Exception as exc:
            print(f"  ERROR: {exc!r}", flush=True)
            failures.append(nc)

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s):")
        for nc in failures:
            print(f"  {nc.relative_to(root)}")
        return 1

    print("\nAll runs analysed successfully.")
    return 0


def _render_override(text: str) -> tuple[str, object]:
    """Parse ``KEY=VALUE`` for `replot --set`; VALUE is JSON if it parses."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def cmd_replot(args: argparse.Namespace) -> int:
    """Redraw figures from their bundles, without the raw output."""
    overrides = dict(args.set or [])
    for path in args.bundles:
        try:
            bundle.replot(path, output_dir=args.output_dir, **overrides)
        except ValueError as exc:
            raise SystemExit(f"gs2ew: {path}: {exc}") from None
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Repeat the analysis behind each bundle and compare with the original."""
    status = 0
    for path in args.bundles:
        figure = bundle.reproduce(path, run=args.run, output_dir=args.output_dir)
        differences = bundle.data_differences(path, figure.with_suffix(".nc"))
        if not differences:
            print(f"{path}: reproduced data identical to the original")
            continue
        status = 1
        print(f"{path}: reproduced data DIFFERS from the original:")
        for name, diff in differences.items():
            print(f"  {name}: max |difference| {diff:.3g}")
    return status


def _scalar(text: str) -> object:
    """An input-file value from the command line: an integer, a real, a
    logical (true/false, .true./.false.) or else a string."""
    for parse in (int, float):
        try:
            return parse(text)
        except ValueError:
            pass
    logical = text.lower().strip(".")
    if logical in {"true", "false"}:
        return logical == "true"
    return text


def _vary(text: str) -> tuple[str, list[object]]:
    """Parse ``KEY=V1,V2,...`` for `scan write --vary`."""
    key, sep, values = text.partition("=")
    if not sep or not key or not values:
        raise argparse.ArgumentTypeError(f"expected KEY=V1,V2,..., got {text!r}")
    return key, [_scalar(v) for v in values.split(",")]


#: How `scan write` combines its --vary parameters.
SCAN_MODES = {
    "one-at-a-time": scan.one_at_a_time,
    "product": scan.product,
    "zip": scan.zipped,
}


def cmd_scan_write(args: argparse.Namespace) -> int:
    """Write a run directory for each point of a scan."""
    points = SCAN_MODES[args.mode](dict(args.vary))
    machine = None if args.machine == "none" else args.machine
    try:
        run_dirs = scan.write(args.template, points, args.directory, machine=machine)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"gs2ew: {exc}") from None
    for run_dir in run_dirs:
        print(run_dir)
    print(f"Wrote {len(run_dirs)} run(s); submit with: gs2ew scan submit {args.directory}")
    return 0


def cmd_scan_submit(args: argparse.Namespace) -> int:
    """Submit the runs of a scan that have not been yet."""
    sbatch_args = args.sbatch_args
    if sbatch_args and sbatch_args[0] == "--":
        sbatch_args = sbatch_args[1:]
    try:
        outcomes = scan.submit(args.directory, sbatch_args, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"gs2ew: {exc}") from None
    width = max((len(name) for name, _ in outcomes), default=0)
    for name, outcome in outcomes:
        print(f"{name:<{width}}  {outcome}")
    return 1 if any(o.startswith("sbatch failed") for _, o in outcomes) else 0


def cmd_scan_status(args: argparse.Namespace) -> int:
    """Report where each run of a scan has got to."""
    try:
        states = scan.status(args.directory)
    except FileNotFoundError as exc:
        raise SystemExit(f"gs2ew: {exc}") from None
    width = max((len(name) for name, _ in states), default=0)
    for name, state in states:
        print(f"{name:<{width}}  {state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gs2ew", description="Post-processing for GS2 output."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_help = "a run directory holding one .out.nc, or the .out.nc itself"

    info = sub.add_parser("info", help="show metadata and available diagnostics")
    info.add_argument("run", help=run_help)
    info.set_defaults(func=cmd_info)

    plot = sub.add_parser("plot", help="render the standard plot set")
    plot.add_argument("run", help=run_help)
    _add_common_options(plot)
    plot.set_defaults(func=cmd_plot)

    batch = sub.add_parser("batch", help="render plots for every run under a directory")
    batch.add_argument("root", help="directory to search for .out.nc files")
    batch.add_argument(
        "--skip", nargs="*", default=[], help="path components to skip"
    )
    _add_common_options(batch)
    batch.set_defaults(func=cmd_batch)

    replot = sub.add_parser(
        "replot", help="redraw figures from their data bundles, without the raw output"
    )
    replot.add_argument("bundles", nargs="+", help="bundle .nc files")
    replot.add_argument(
        "-o", "--output-dir", default=None, help="directory for the figures (default: beside each bundle)"
    )
    replot.add_argument(
        "--set",
        action="append",
        type=_render_override,
        metavar="KEY=VALUE",
        help="change a render option, e.g. --set normalise=true, or a figure "
        "option (style, width, title, formats, dpi), e.g. --set formats=pdf; "
        "VALUE is read as JSON if it can be. A bundle recording the change is "
        "written with the figure",
    )
    replot.set_defaults(func=cmd_replot)

    reproduce = sub.add_parser(
        "reproduce",
        help="repeat the analysis behind bundles from the raw output, and compare",
    )
    reproduce.add_argument("bundles", nargs="+", help="bundle .nc files")
    reproduce.add_argument(
        "--run", default=None, help="the run, if it has moved since the bundle was made"
    )
    reproduce.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=f"directory for the results (default: {bundle.REPRODUCED_DIR}/ beside each bundle)",
    )
    reproduce.set_defaults(func=cmd_reproduce)

    scan_parser = sub.add_parser("scan", help="write, submit and follow a parameter scan")
    scan_sub = scan_parser.add_subparsers(dest="scan_command", required=True)

    write = scan_sub.add_parser(
        "write", help="write a run directory for each point of a scan"
    )
    write.add_argument("template", help="the GS2 input file each run starts from")
    write.add_argument("directory", help="where to write the scan; must be new or empty")
    write.add_argument(
        "--vary",
        action="append",
        type=_vary,
        required=True,
        metavar="KEY=V1,V2,...",
        help="a parameter and its values, e.g. --vary ntheta=32,48; qualify "
        "KEY by its namelist (species_parameters_2.tprim) if the bare name is "
        "ambiguous or not in the template. Repeat for more parameters",
    )
    write.add_argument(
        "--mode",
        choices=list(SCAN_MODES),
        default="one-at-a-time",
        help="the template plus each value on its own (default); every "
        "combination of the values (product); or the i-th values together (zip)",
    )
    write.add_argument(
        "--machine",
        default="archer2",
        help=f"the Slurm template to copy into each run: {', '.join(scan.machines())}, "
        "a file of your own, or none (default: archer2)",
    )
    write.set_defaults(func=cmd_scan_write)

    submit = scan_sub.add_parser(
        "submit", help="sbatch every run without output or a job yet"
    )
    submit.add_argument("directory", help="the scan directory")
    submit.add_argument(
        "--dry-run", action="store_true", help="print the sbatch commands without running them"
    )
    submit.add_argument(
        "sbatch_args",
        nargs="*",
        metavar="-- SBATCH_ARGS",
        help="options for sbatch, after --, e.g. -- --time=02:00:00 --nodes=2",
    )
    submit.set_defaults(func=cmd_scan_submit)

    status = scan_sub.add_parser("status", help="where each run of a scan has got to")
    status.add_argument("directory", help="the scan directory")
    status.set_defaults(func=cmd_scan_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
