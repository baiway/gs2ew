#!/usr/bin/env python3
"""Generate a plot set for one or more `bmag` powers.

Written for the open question of which power of B0 belongs in the GS2 transfer
prefactors. One output directory per power, plus an overlay figure comparing
the poloidal *shapes* directly — a scalar multiple cannot change a shape, so if
no power makes the field drive and the kinetic-energy transfer coincide, the
discrepancy is not just a missing B0 factor.

Run with the project's environment:

    uv run examples/bmag_scan.py run.out.nc outputs/scan

or, with the venv active and the file marked executable:

    ./bmag_scan.py run.out.nc outputs/scan

Everything in the CONFIGURATION block below can also be overridden on the
command line; run with --help to see how.
"""

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Edit these, or override any of them on the command line.

# Transferred quantities to plot: any of "U", "H", "S".
QUANTITIES = ["U"]

# Include the kinetic-energy transfer as a reference curve.
INCLUDE_KINETIC = True

# One output directory per power. The field drives (H, S, U) are multiplied by
# bmag**power; the kinetic-energy transfer is not.
#   1  cancels the 1/bmag currently in GS2's free-energy prefactor
#   0  leaves the drives exactly as GS2 wrote them
#  -2  is the old `div_by_bmag2` option
BMAG_POWERS = [1.0]

# Applied to the kinetic-energy transfer only, independently of the above.
KINETIC_BMAG_POWER = 0.0

# Averaging window. TSTART may be a number, "saturation", or None for the last
# timestep only. See gs2ew.timebase for how the two combine.
TSTART = "saturation"
WINDOW = None
SATURATION_PAD = 0.20

# Wavevector reductions to plot: any of "kxtarget", "kxsource", "kysource".
REDUCTIONS = ["kxtarget"]

# Clip wavevector axes to +/-KLIM. None keeps the full range.
KLIM = 0.6

# Movies are much slower than the static plots. STRIDE keeps every nth frame.
MOVIES = False
FPS = 10
STRIDE = 1

# ─────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path

# Headless backend, so this works unchanged on a compute node. Must be set
# before pyplot is imported, including via gs2ew.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gs2ew import catalog
from gs2ew.catalog import Drive
from gs2ew.plots import _common as common
from gs2ew.plots import (
    fields,
    transfer_spectral,
    transfer_summary,
    transfer_theta,
    transfer_time,
    transfer_tp,
)
from gs2ew.timebase import resolve_window

# Okabe-Ito hues in a fixed order, for the overlay figure. Assigned by bmag
# power so a given power keeps its colour however many are plotted.
POWER_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"]


def power_tag(power: float) -> str:
    """Directory-safe label for a bmag power, e.g. ``bmag_power_+1``."""
    text = str(int(power)) if float(power).is_integer() else f"{power:g}"
    return f"bmag_power_{'+' if power > 0 else ''}{text}"


def plot_set(ds: xr.Dataset, args: argparse.Namespace, power: float, out: Path) -> None:
    """Render the full plot set for a single bmag power."""
    window = dict(
        tstart=args.tstart, window=args.window, saturation_pad=args.saturation_pad
    )
    scale = dict(bmag_power=power, kinetic_bmag_power=args.kinetic_bmag_power)
    select = dict(quantities=args.quantities, include_kinetic=args.include_kinetic)

    fields.plot_fields_time_traces(ds, output_dir=out)
    fields.plot_fields_by_mode(ds, output_dir=out)

    transfer_theta.plot_transfer_theta(
        ds, output_dir=out, show_std=True, **window, **scale, **select
    )
    transfer_summary.plot_transfer_summary(ds, output_dir=out, **window, **scale, **select)
    transfer_time.plot_transfer_hovmoller(ds, output_dir=out, **scale, **select)

    for reduction in args.reductions:
        transfer_spectral.plot_transfer_spectral(
            ds, reduction=reduction, output_dir=out, klim=args.klim,
            **window, **scale, **select,
        )
        transfer_spectral.plot_transfer_spectrum(
            ds, reduction=reduction, output_dir=out, klim=args.klim,
            **window, **scale, **select,
        )
        transfer_time.plot_transfer_butterfly(
            ds, reduction=reduction, output_dir=out, klim=args.klim, **scale, **select,
        )

    # The trapped/passing diagnostics are optional in GS2 and have no
    # kinetic-energy counterpart, so they take neither kinetic option.
    if catalog.available_drives(ds, "trapped_passing", args.quantities,
                                include_kinetic=False):
        transfer_tp.plot_transfer_trapped_passing(
            ds, output_dir=out, show_std=True, quantities=args.quantities,
            bmag_power=power, **window,
        )
        for quantity in args.quantities or list(catalog.QUANTITIES):
            transfer_tp.plot_transfer_trapped_passing_spectral(
                ds, quantity=quantity, output_dir=out, klim=args.klim,
                bmag_power=power, **window,
            )

    if args.movies:
        movie = dict(fps=args.fps, stride=args.stride, output_dir=out)
        transfer_theta.plot_transfer_theta_movie(
            ds, normalise="movie", **movie, **scale, **select
        )
        transfer_spectral.plot_transfer_spectral_movie(
            ds, klim=args.klim, **movie, **scale, **select
        )
        if catalog.available_drives(ds, "trapped_passing", args.quantities,
                                    include_kinetic=False):
            transfer_tp.plot_transfer_trapped_passing_movie(
                ds, quantities=args.quantities, bmag_power=power, **movie
            )


def plot_shape_comparison(
    ds: xr.Dataset,
    args: argparse.Namespace,
    output_dir: Path,
    field: str = "phi",
    quantity: str = "U",
) -> Path | None:
    """Overlay the poloidal shape of one drive at each bmag power, plus T_v^ZF.

    Every curve is divided by its own peak absolute value, so only *shape* is
    compared. A power of B0 rescales a profile pointwise in theta, so the
    curves do move apart; what this asks is whether any of them lands on the
    kinetic-energy transfer. If none does, no single power of B0 reconciles
    the two and the difference has poloidal structure of its own.
    """
    drive = Drive(quantity, field)
    if drive.varname() not in ds:
        print(f"  (no {drive.varname()}; skipping shape comparison)")
        return None

    tw = resolve_window(ds, args.tstart, args.window, args.saturation_pad, quiet=True)
    theta = ds["theta"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, power in enumerate(args.bmag_powers):
        scaling = catalog.resolve_scaling(bmag_power=power)
        values = np.asarray(tw.mean(catalog.load(ds, drive, scaling=scaling)).values)
        ax.plot(
            theta,
            common.normalise(values, None),
            linewidth=1.6,
            color=POWER_COLOURS[i % len(POWER_COLOURS)],
            label=drive.label_short(scaling),
        )

    ke = Drive(catalog.KINETIC)
    if ke.varname() in ds:
        ke_scaling = catalog.resolve_scaling(
            kinetic_bmag_power=args.kinetic_bmag_power
        )
        values = np.asarray(tw.mean(catalog.load(ds, ke, scaling=ke_scaling)).values)
        ax.plot(
            theta,
            common.normalise(values, None),
            linewidth=1.6,
            linestyle="--",
            color=common.QUANTITY_COLOURS["KE"],
            label=ke.label(ke_scaling),
        )

    ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("normalised to own peak", fontsize=12)
    ax.set_title(
        f"Poloidal shape vs $B_0$ power (${catalog.FIELDS[field]}$)"
        + (f" — {tw.title}" if tw.title else "")
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return common.save_figure(fig, output_dir, f"shape_comparison_{quantity}_{field}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nc_file", help="path to a GS2 .out.nc file")
    parser.add_argument("output_root", help="directory to write the plot sets into")
    parser.add_argument(
        "--bmag-powers", type=float, nargs="+", default=BMAG_POWERS,
        help="one plot set per power; try '-1 0 1' to compare",
    )
    parser.add_argument(
        "--kinetic-bmag-power", type=float, default=KINETIC_BMAG_POWER,
        help="applied to the kinetic-energy transfer only",
    )
    parser.add_argument(
        "--quantities", nargs="*", default=QUANTITIES, choices=list(catalog.QUANTITIES),
        help="transferred quantities to plot",
    )
    parser.add_argument(
        "--no-kinetic", dest="include_kinetic", action="store_false",
        default=INCLUDE_KINETIC, help="omit the kinetic-energy transfer",
    )
    parser.add_argument(
        "--tstart", default=TSTART,
        help="window start: a number, 'saturation', or 'none' for the last timestep",
    )
    parser.add_argument("--window", type=float, default=WINDOW, help="averaging window")
    parser.add_argument(
        "--saturation-pad", type=float, default=SATURATION_PAD,
        help="fraction of the post-saturation interval to skip",
    )
    parser.add_argument(
        "--reductions", nargs="*", default=REDUCTIONS,
        choices=[r for r in catalog.REDUCTIONS if r != "theta"],
        help="wavevector reductions to plot",
    )
    parser.add_argument("--klim", type=float, default=KLIM, help="clip k axes to +/-KLIM")
    parser.add_argument(
        "--movies", action="store_true", default=MOVIES, help="also render movies (slow)"
    )
    parser.add_argument("--fps", type=int, default=FPS, help="movie frame rate")
    parser.add_argument("--stride", type=int, default=STRIDE, help="keep every nth frame")
    parser.add_argument(
        "--no-comparison", dest="comparison", action="store_false", default=True,
        help="skip the shape-comparison overlay",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if isinstance(args.tstart, str):
        if args.tstart.lower() in {"none", "last"}:
            args.tstart = None
        elif args.tstart != "saturation":
            args.tstart = float(args.tstart)

    output_root = Path(args.output_root)

    with xr.open_dataset(args.nc_file) as ds:
        for power in args.bmag_powers:
            out = output_root / power_tag(power)
            print(f"\n=== bmag_power = {power:g} -> {out} ===", flush=True)
            plot_set(ds, args, power, out)

        # One overlay across all powers, so it lives above the per-power dirs.
        if args.comparison and len(args.bmag_powers) >= 1:
            print(f"\n=== shape comparison -> {output_root} ===", flush=True)
            for quantity in args.quantities or ["U"]:
                for field in catalog.FIELDS:
                    plot_shape_comparison(ds, args, output_root, field, quantity)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
