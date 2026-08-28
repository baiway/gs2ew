"""Command-line interface.

Three subcommands::

    gs2ew info  run.out.nc                 # what is in the file
    gs2ew plot  run.out.nc -o outputs/     # the standard plot set
    gs2ew batch project/ -o analysis/      # every .out.nc under a directory

`plot` and `batch` take the same options, so a batch job produces exactly what
a single run would. Anything more bespoke than this is better written as a
short script against the Python API — see `examples/analysis.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A non-interactive backend: this is routinely run headless on a compute node.
# Must be selected before pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

import xarray as xr

from gs2ew import catalog
from gs2ew.io import load_gs2_output
from gs2ew.metadata import get_metadata


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by `plot` and `batch`."""
    parser.add_argument(
        "-o", "--output-dir", default="outputs", help="directory for plots"
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
        "--bmag-power",
        type=float,
        default=None,
        help="multiply the field drives by bmag**POWER (e.g. 1 to cancel the "
        "1/bmag in GS2's free-energy prefactor)",
    )
    parser.add_argument(
        "--kinetic-bmag-power",
        type=float,
        default=0.0,
        help="multiply the kinetic-energy transfer by bmag**POWER",
    )
    parser.add_argument(
        "--klim", type=float, default=None, help="clip wavevector axes to +/-KLIM"
    )
    parser.add_argument(
        "--ke-variant",
        default=None,
        help="variant tag in the kinetic-energy variable name (e.g. 'new'), "
        "for GS2 branches that write more than one",
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


def cmd_info(args: argparse.Namespace) -> int:
    """Print build metadata and the transfer diagnostics present."""
    with load_gs2_output(args.nc_file) as ds:
        print(f"=== {args.nc_file} ===")
        for key, value in get_metadata(ds).items():
            print(f"  {key}: {value}")

        t = ds["t"].values
        print(f"\n  t: [{t[0]:.3g}, {t[-1]:.3g}] over {len(t)} steps")
        print(f"  species: {ds.sizes.get('species', 0)}, theta: {ds.sizes.get('theta', 0)}")

        print("\n=== Transfer diagnostics ===")
        for reduction in list(catalog.REDUCTIONS) + ["trapped_passing"]:
            drives = catalog.available_drives(
                ds, reduction, ke_variant=args.ke_variant
            )
            names = ", ".join(
                f"{d.quantity}/{d.field or '-'}" for d in drives
            )
            print(f"  {reduction:<16} {names or '(none)'}")

        print("\n=== Identity check (H = U - S) ===")
        for field in catalog.FIELDS:
            residual = catalog.identity_residual(ds, field)
            if residual == residual:  # not NaN
                print(f"  {field:<5} max relative residual {residual:.2e}")
    return 0


def _plot_all(ds: xr.Dataset, args: argparse.Namespace, output_dir: Path) -> None:
    """Render the standard plot set for one dataset."""
    from gs2ew.plots import fields, transfer_spectral, transfer_summary
    from gs2ew.plots import transfer_theta, transfer_time, transfer_tp

    window_opts = dict(
        tstart=_tstart(args.tstart),
        window=args.window,
        saturation_pad=args.saturation_pad,
    )
    scale_opts = dict(
        bmag_power=args.bmag_power,
        kinetic_bmag_power=args.kinetic_bmag_power,
    )

    fields.plot_fields_time_traces(ds, output_dir=output_dir)
    fields.plot_fields_by_mode(ds, output_dir=output_dir)
    fields.plot_phi2_by_theta(ds, output_dir=output_dir)

    transfer_theta.plot_transfer_theta(
        ds, output_dir=output_dir, show_std=True, **window_opts, **scale_opts
    )
    transfer_summary.plot_transfer_summary(
        ds, output_dir=output_dir, **window_opts, **scale_opts
    )
    transfer_time.plot_transfer_hovmoller(ds, output_dir=output_dir, **scale_opts)

    for reduction in ("kxtarget", "kysource"):
        transfer_spectral.plot_transfer_spectral(
            ds,
            reduction=reduction,
            output_dir=output_dir,
            klim=args.klim,
            ke_variant=args.ke_variant,
            **window_opts,
            **scale_opts,
        )
        transfer_spectral.plot_transfer_spectrum(
            ds,
            reduction=reduction,
            output_dir=output_dir,
            klim=args.klim,
            ke_variant=args.ke_variant,
            **window_opts,
            **scale_opts,
        )
        transfer_time.plot_transfer_butterfly(
            ds,
            reduction=reduction,
            output_dir=output_dir,
            klim=args.klim,
            ke_variant=args.ke_variant,
            **scale_opts,
        )

    # Trapped/passing output is optional in GS2, so only plot it if it is there.
    if catalog.available_drives(ds, "trapped_passing", include_kinetic=False):
        transfer_tp.plot_transfer_trapped_passing(
            ds,
            output_dir=output_dir,
            show_std=True,
            **window_opts,
            bmag_power=args.bmag_power,
        )
        transfer_tp.plot_transfer_trapped_passing_spectral(
            ds,
            output_dir=output_dir,
            klim=args.klim,
            **window_opts,
            bmag_power=args.bmag_power,
        )

    if args.movies:
        movie_opts = dict(fps=args.fps, stride=args.stride, output_dir=output_dir)
        transfer_theta.plot_transfer_theta_movie(
            ds, normalise="movie", **movie_opts, **scale_opts
        )
        transfer_spectral.plot_transfer_spectral_movie(
            ds, klim=args.klim, **movie_opts, **scale_opts
        )
        if catalog.available_drives(ds, "trapped_passing", include_kinetic=False):
            transfer_tp.plot_transfer_trapped_passing_movie(
                ds, **movie_opts, bmag_power=args.bmag_power
            )
            transfer_tp.plot_transfer_trapped_passing_spectral_movie(
                ds, klim=args.klim, **movie_opts, bmag_power=args.bmag_power
            )


def cmd_plot(args: argparse.Namespace) -> int:
    """Render the standard plot set for a single run."""
    output_dir = Path(args.output_dir)
    with load_gs2_output(args.nc_file) as ds:
        _plot_all(ds, args, output_dir)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Render the standard plot set for every `.out.nc` under a directory.

    Outputs mirror each run's path relative to `root`, so runs that share an
    input-file name do not collide. One failing run is reported and skipped
    rather than aborting the batch.
    """
    root = Path(args.root)
    output_root = Path(args.output_dir)
    skip = set(args.skip or [])

    runs = [
        nc
        for nc in sorted(root.rglob("*.out.nc"))
        if not any(part in skip for part in nc.relative_to(root).parts)
    ]
    if not runs:
        print(f"No .out.nc files found under {root}")
        return 1

    print(f"Found {len(runs)} run(s) under {root}")

    failures = []
    for nc in runs:
        output_dir = output_root / nc.parent.relative_to(root)
        print(f"\n=== {nc.relative_to(root)} -> {output_dir} ===", flush=True)
        try:
            with load_gs2_output(nc) as ds:
                _plot_all(ds, args, output_dir)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gs2ew", description="Post-processing for GS2 output."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show metadata and available diagnostics")
    info.add_argument("nc_file", help="path to a GS2 .out.nc file")
    info.add_argument("--ke-variant", default=None)
    info.set_defaults(func=cmd_info)

    plot = sub.add_parser("plot", help="render the standard plot set")
    plot.add_argument("nc_file", help="path to a GS2 .out.nc file")
    _add_common_options(plot)
    plot.set_defaults(func=cmd_plot)

    batch = sub.add_parser("batch", help="render plots for every run under a directory")
    batch.add_argument("root", help="directory to search for .out.nc files")
    batch.add_argument(
        "--skip", nargs="*", default=[], help="path components to skip"
    )
    _add_common_options(batch)
    batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
