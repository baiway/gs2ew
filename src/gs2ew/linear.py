"""Linear scans: a growth-rate spectrum for each point of a parameter scan.

Each (point, ky) is its own GS2 run with ``grid_option = 'single'``, so each
stops as soon as its frequency has converged and can have its own settings,
and a spectrum is refined by adding runs. The runs of a job go through GS2's
``multigs2``, which splits the job's cores into batches and runs a share of
the runs on each, one after another::

    lin/
      scan.json              as for any scan, plus what makes it linear
      template.in
      job.slurm              the machine's Slurm template, run on a list
      jobs.csv
      runs_00.list           written by `submit`: the runs of one job
      beta_0p01-aky_0p1/
        beta_0p01-aky_0p1.in

Every run is the template made linear (see :func:`linear_settings`), so a
nonlinear input file will do as the template. The parallel domain
(``nperiod``) is the template's for every ky: a growth rate depends on it
until the mode fits, at high ky as well as low, so it is checked by a scan
over ``nperiod`` rather than chosen per ky.
"""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import f90nml
import numpy as np
import xarray as xr

from gs2ew import provenance, scan
from gs2ew.io import OUTPUT_SUFFIX

#: The key a linear scan varies within each point.
KY_KEY = "kt_grids_single_parameters.aky"

#: The default ky: from 0.05 to 2, evenly spaced in log(ky).
DEFAULT_KY = "0.05:2:20"

#: The default time GS2 averages the frequency over before testing it for
#: convergence. Two modes with similar growth rates beat with periods of tens
#: of a/v_th, and a window shorter than that can pass the test with a growth
#: rate of the wrong sign.
DEFAULT_OMEGA_WINDOW = 25.0

#: Time a job allows for starting up, beyond its runs' own.
JOB_STARTUP = 60.0

#: The share of a run's time that GS2 keeps back to finish cleanly.
MARGIN_FRACTION = 0.05

#: GS2's convergence tolerance, for a template whose own can never be met.
DEFAULT_OMEGATOL = 1e-3

#: Diagnostics a linear run has no use for.
_NONLINEAR_DIAGNOSTICS = (*scan._TRANSFER_SWITCHES, "write_kinetic_energy_transfer")

#: Written by :func:`submit`, numbered for each job.
LIST_PATTERN = "runs_{:02d}.list"


def ky_values(spec: str) -> list[float]:
    """The ky in `spec`: a comma-separated list, or ``MIN:MAX:N`` for N values
    evenly spaced in log(ky), rounded to three significant figures.

    Raises
    ------
    ValueError
        If `spec` is neither, or a ky is not positive.
    """
    if ":" in spec:
        try:
            low, high, count = spec.split(":")
            low, high, n = float(low), float(high), int(count)
        except ValueError:
            raise ValueError(f"Not MIN:MAX:N: {spec!r}") from None
        if n < 1 or low <= 0 or high < low:
            raise ValueError(f"Not 0 < MIN <= MAX with N >= 1: {spec!r}")
        values = np.geomspace(low, high, n) if n > 1 else np.array([low])
        values = [float(f"{v:.3g}") for v in values]
    else:
        values = [float(v) for v in spec.split(",") if v.strip()]
    if not values or min(values) <= 0:
        raise ValueError(f"ky must be positive: {spec!r}")
    return sorted(set(values))


def box_ky(deck: f90nml.Namelist) -> list[float]:
    """The nonzero ky of the box in `deck`'s ``kt_grids_box_parameters``,
    which a linear scan for that box should include.

    ky runs in steps of ``1/y0`` (``-y0`` for negative ``y0``, ``2 pi / ly``
    with ``ly``) up to ``(ny - 1) // 3`` of them, as GS2's dealiasing leaves.

    Raises
    ------
    ValueError
        If `deck` has no box with ``ny``.
    """
    box = deck.get("kt_grids_box_parameters", {})
    if "ny" not in box:
        raise ValueError("The template has no kt_grids_box_parameters with ny.")
    if box.get("ly"):
        step = 2 * math.pi / box["ly"]
    else:
        y0 = box.get("y0", 2.0)
        step = -y0 if y0 < 0 else 1 / y0
    return [float(f"{step * n:.6g}") for n in range(1, (box["ny"] - 1) // 3 + 1)]


def linear_settings(
    deck: f90nml.Namelist,
    omega_window: float = DEFAULT_OMEGA_WINDOW,
    omegatol: float | None = None,
    max_sim_time: float | None = None,
    theta0: float = 0.0,
) -> dict[str, object]:
    """What to change in `deck` to make it a linear run of one ky.

    The nonlinear term goes off, the grid becomes ``'single'`` at `theta0`,
    and GS2 writes the frequency (``write_omega``, ``write_omavg``) and the
    final fields, averages the frequency over `omega_window` (as ``navg``,
    in steps) and stops when it has converged, to within `omegatol` if
    given; or if `deck`'s own can't be met, as a nonlinear run's negative
    one can't, :data:`DEFAULT_OMEGATOL`. What would cost a linear run time or
    space for nothing is turned off where `deck` turns it on: the transfer
    diagnostics, restart files and saved response matrices.
    """
    settings: dict[str, object] = {
        "kt_grids_knobs.grid_option": "single",
        "kt_grids_single_parameters.theta0": float(theta0),
        "gs2_diagnostics_knobs.write_omega": True,
        "gs2_diagnostics_knobs.write_omavg": True,
        "gs2_diagnostics_knobs.write_fields": True,
        "gs2_diagnostics_knobs.exit_when_converged": True,
    }
    delt = deck.get("knobs", {}).get("delt", 0.1)
    settings["gs2_diagnostics_knobs.navg"] = max(1, round(omega_window / delt))
    if omegatol is None and deck.get("gs2_diagnostics_knobs", {}).get("omegatol", 1) <= 0:
        omegatol = DEFAULT_OMEGATOL
    if omegatol is not None:
        settings["gs2_diagnostics_knobs.omegatol"] = float(omegatol)
    if max_sim_time is not None:
        settings["knobs.max_sim_time"] = float(max_sim_time)

    def off(group: str, name: str) -> None:
        if deck.get(group, {}).get(name):
            settings[f"{group}.{name}"] = False

    if str(deck.get("nonlinear_terms_knobs", {}).get("nonlinear_mode", "off")).lower() != "off":
        settings["nonlinear_terms_knobs.nonlinear_mode"] = "off"
    for name in _NONLINEAR_DIAGNOSTICS:
        off("gs2_diagnostics_knobs", name)
    off("gs2_diagnostics_knobs", "save_for_restart")
    off("fields_knobs", "dump_response")
    off("fields_knobs", "read_response")
    return settings


def write(
    template: str | Path,
    points: Sequence[scan.Point],
    ky: Sequence[float],
    directory: str | Path,
    machine: str | Path | None = "archer2",
    omega_window: float = DEFAULT_OMEGA_WINDOW,
    omegatol: float | None = None,
    max_sim_time: float | None = None,
    theta0: float = 0.0,
    settings: scan.Point | None = None,
) -> list[Path]:
    """Write a linear run for each of `points` at each of `ky`.

    `points` are as :func:`gs2ew.scan.write` takes them; a point repeating
    the template's values is dropped if the template itself (``{}``) is among
    them. Each run is the template made linear by :func:`linear_settings`,
    with any further `settings` of your own, which are recorded once in
    ``scan.json``. `machine`'s Slurm template is copied into `directory` for
    :func:`submit`.

    Returns
    -------
    list of Path
        The run directories.
    """
    template = Path(template)
    directory = Path(directory)
    script = scan._job_script(machine)
    deck = f90nml.read(template)
    fixed = linear_settings(deck, omega_window, omegatol, max_sim_time, theta0)
    for key, value in (settings or {}).items():
        fixed[scan.resolve_key(scan._apply(deck, fixed), key)] = value
    base = scan._apply(deck, fixed)
    if any(key.lower().split(".")[-1] == "aky" for point in points for key in point):
        raise ValueError("ky is given with `ky`, not in the points.")
    if not ky:
        raise ValueError("No ky to run at.")

    resolved = [
        {scan.resolve_key(base, key): value for key, value in point.items()} for point in points
    ]
    if any(not point for point in resolved):
        resolved = [
            point
            for point in resolved
            if not point
            or not all(scan._template_value(base, k) == v for k, v in point.items())
        ]
    _check_window(base)
    runs = scan.combine(resolved, [{KY_KEY: float(k)} for k in ky])
    run_dirs = scan.write(template, runs, directory, machine=None, settings=fixed)

    if script is not None:
        (directory / scan.JOB_SCRIPT).write_text(script.read_text())
    manifest = scan.load(directory)
    manifest["machine"] = None if machine is None else str(machine)
    manifest["linear"] = {
        "ky": [float(k) for k in ky],
        "omega_window": omega_window,
        "theta0": float(theta0),
    }
    (directory / scan.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    nperiod = base.get("theta_grid_parameters", {}).get("nperiod", 2)
    if not any("theta_grid_parameters.nperiod" in point for point in resolved):
        print(
            f"Note: every run has nperiod = {nperiod}. Check the growth rates "
            "don't change with it, with a scan over nperiod."
        )
    return run_dirs


def _check_window(deck: f90nml.Namelist) -> None:
    """Warn if a run would stop before GS2 could test its frequency."""
    knobs = deck.get("knobs", {})
    window = deck["gs2_diagnostics_knobs"]["navg"] * knobs.get("delt", 0.1)
    max_sim_time = knobs.get("max_sim_time")
    if max_sim_time is not None and max_sim_time < 4 * window:
        print(
            f"Warning: max_sim_time = {max_sim_time:g} leaves little time to "
            f"converge after the first {window:g} a/v_th that the frequency is "
            "averaged over; allow at least four times that."
        )


def _pending(directory: Path, names: Sequence[str]) -> list[str]:
    jobs = scan._read_jobs(directory)
    return [
        name
        for name in names
        if name not in jobs and not any((directory / name).glob(f"*{OUTPUT_SUFFIX}"))
    ]


def submit(
    directory: str | Path,
    jobs: int = 1,
    cores_per_run: int = 4,
    sbatch_args: Sequence[str] = (),
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Submit the runs of a linear scan that have neither output nor a job.

    They are dealt in turn to `jobs` jobs in order of ky, so each job gets a
    mix of the slow low-ky runs and the quick high-ky ones, and so does each
    batch, as ``multigs2`` deals a job's runs to its batches in turn too. Each job runs ``job.slurm``
    on a list of its runs through ``multigs2``, in as many batches as it has
    `cores_per_run`, up to one per run.

    A batch runs its share one after another, so when the job's time limit
    is known each run's ``avail_cpu_time`` is set to its share of it: a run
    that is too slow stops itself, and the rest still run.

    Returns
    -------
    list of (run, outcome)
        The job ID, the command (with `dry_run`), or why it was not submitted.
    """
    directory = Path(directory)
    manifest = scan.load(directory)
    if "linear" not in manifest:
        raise ValueError(f"{directory} is not a linear scan; submit it with `gs2ew scan submit`.")
    if manifest["machine"] is None:
        raise ValueError(f"{directory} was written without a job script.")
    if not dry_run and shutil.which("sbatch") is None:
        raise FileNotFoundError("sbatch not found: submit from a Slurm login node.")
    if jobs < 1 or cores_per_run < 1:
        raise ValueError("jobs and cores_per_run must be at least 1.")

    names = scan._names(manifest)
    pending = _pending(directory, names)
    outcomes = {name: "skipped: has output or a job" for name in names if name not in pending}
    if not pending:
        return [(name, outcomes[name]) for name in names]

    options = scan._sbatch_options(directory / scan.JOB_SCRIPT, sbatch_args)
    cores, _ = scan._cores(options)
    if cores is None:
        raise ValueError(
            "Can't tell how many cores a job has: give --ntasks, or --nodes "
            "with --ntasks-per-node, after --."
        )
    if cores < cores_per_run:
        raise ValueError(f"A job of {cores} cores can't give a run {cores_per_run}.")
    limit = scan._slurm_seconds(options["time"]) if "time" in options else None

    first = next(i for i in range(len(names) + 1) if not (directory / LIST_PATTERN.format(i)).exists())
    # Dealt in order of ky, as are a job's runs to its batches, so the slow
    # low-ky runs are spread out.
    ky = {run["name"]: run["values"].get(KY_KEY, 0.0) for run in manifest["runs"]}
    ordered = sorted(pending, key=lambda name: ky[name])
    groups = [ordered[i :: jobs] for i in range(min(jobs, len(ordered)))]
    for number, group in enumerate(groups, start=first):
        nbatch = min(cores // cores_per_run, len(group))
        per_batch = math.ceil(len(group) / nbatch)
        listing = LIST_PATTERN.format(number)
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={directory.resolve().name}-{Path(listing).stem}",
            *sbatch_args,
            scan.JOB_SCRIPT,
            listing,
            "--nbatch",
            str(nbatch),
        ]
        budget = None if limit is None else (limit - JOB_STARTUP) / per_batch
        described = f"{len(group)} runs in {nbatch} batches of {cores // nbatch} cores"
        if budget is not None:
            described += f", {budget:.0f} s each"
        if dry_run:
            for name in group:
                outcomes[name] = f"(cd {directory} && {shlex.join(command)}) [{described}]"
            continue

        (directory / listing).write_text("".join(f"{name}/{name}.in\n" for name in group))
        if budget is not None:
            for name in group:
                _set_time(directory / name / f"{name}.in", budget)
        result = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            (directory / listing).unlink()
            for name in group:
                outcomes[name] = f"sbatch failed: {result.stderr.strip()}"
            continue
        job = result.stdout.strip().split(";")[0]
        for name in group:
            scan._record_job(directory, name, job, cores // nbatch)
            outcomes[name] = f"job {job} ({listing}: {described})"
    return [(name, outcomes[name]) for name in names]


def _set_time(path: Path, budget: float) -> None:
    """Give the run at `path` `budget` seconds: GS2 stops once it has used
    ``avail_cpu_time`` less ``margin_cpu_time``."""
    deck = f90nml.read(path)
    changes = {
        "knobs.avail_cpu_time": float(round(budget)),
        "knobs.margin_cpu_time": float(round(MARGIN_FRACTION * budget)),
    }
    changes = {k: scan._like_template(deck, k, v) for k, v in changes.items()}
    partial = path.with_name(f"{path.name}.partial")
    f90nml.patch(str(path), scan._nested(changes), str(partial))
    partial.replace(path)


#: GS2's exit codes that say why a linear run stopped short.
_EXIT_STOPS = {
    scan.TIME_LIMIT_EXIT: "reached avail_cpu_time",
    -6: "numerically unstable",
}


def stopped(ds: xr.Dataset) -> str:
    """Why the linear run `ds` stopped: ``"converged"`` if GS2 stopped it
    early, its frequency having converged, or what stopped it first.

    GS2's ``.exit_reason`` can't say: a converged run is reported as having
    completed its ``nstep`` steps. So the run counts as converged if it
    stopped before ``max_sim_time`` and ``nstep``, and ``.exit_reason``
    gives no other reason. An output without an ``.exit_reason`` beside it
    (copied without one, or still running) is judged by the output alone, as
    ``"converged (no .exit_reason)"`` if it stopped early.
    """
    settings: dict[str, float] = {"max_sim_time": math.inf, "nstep": math.inf, "delt": 0.1}
    for deck in reversed(provenance.input_decks(ds)):
        for name in settings:
            if name in deck.get("knobs", {}):
                settings[name] = float(deck["knobs"][name])

    source = Path(str(ds.encoding.get("source", "")))
    unconfirmed = ""
    if source.name.endswith(OUTPUT_SUFFIX):
        exit_reason = source.with_name(source.name[: -len(OUTPUT_SUFFIX)] + ".exit_reason")
        if not exit_reason.is_file():
            unconfirmed = " (no .exit_reason)"
        reasons = scan._exit_reasons(source.parent, exit_reason.stem, None) or {}
        for code, why in _EXIT_STOPS.items():
            if code in reasons:
                return why

    t_end = float(ds["t"][-1])
    delt = settings["delt"]
    if t_end >= settings["max_sim_time"] - 1.5 * delt:
        return "reached max_sim_time"
    if t_end / delt >= settings["nstep"] - 1:
        return "reached nstep"
    return "converged" + unconfirmed


def converged(why: str) -> bool:
    """Whether a run that :func:`stopped` for `why` converged."""
    return why.startswith("converged")


def status(directory: str | Path) -> list[tuple[str, str]]:
    """Where each run of a linear scan has got to, as
    :func:`gs2ew.scan.status` says, but with how each finished run stopped
    read from its output (see :func:`stopped`)."""
    directory = Path(directory)
    states = []
    for name, state in scan.status(directory):
        if state.startswith("finished"):
            outputs = sorted((directory / name).glob(f"*{OUTPUT_SUFFIX}"))
            if outputs:
                with xr.open_dataset(outputs[0], engine="netcdf4") as ds:
                    state = f"finished: {stopped(ds)}"
        states.append((name, state))
    return states


def scanned_key(directory: str | Path) -> str | None:
    """The one key a linear scan in `directory` varies besides ky, if it
    varies exactly one; None otherwise."""
    try:
        manifest = scan.load(directory)
    except FileNotFoundError:
        return None
    keys = {key for run in manifest["runs"] for key in run["values"]} - {KY_KEY}
    return keys.pop() if len(keys) == 1 else None


def summary(spectrum: xr.Dataset) -> list[str]:
    """Notes on a spectrum from
    :func:`gs2ew.plots.linear.compute_linear_spectrum`: each line's peak,
    and any that peaks at the end of its ky or hasn't fallen by its
    largest ky, or has runs that didn't converge."""
    lines = []
    parameter = spectrum.attrs["parameter"].split(".")[-1]
    for i in range(spectrum.sizes["value"]):
        row = spectrum.isel(value=i)
        label = f"{parameter} = {row['value'].item():g}" if parameter else "spectrum"
        have = ~np.isnan(row["gamma"].values)
        good = have & (row["converged"].values == 1)
        ky = row["ky"].values
        if not good.any():
            lines.append(f"{label}: no converged runs.")
            continue
        gamma = np.where(good, row["gamma"].values, np.nan)
        peak = int(np.nanargmax(gamma))
        note = f"{label}: gamma_max = {gamma[peak]:.4g} at ky = {ky[peak]:g}"
        if (have & ~good).any():
            note += f"; {int((have & ~good).sum())} run(s) not converged"
        lines.append(note + ".")
        scanned = ky[have]
        if gamma[peak] > 0 and ky[peak] == scanned.min():
            lines.append("  The peak is at the lowest ky: run lower ky.")
        if gamma[peak] > 0 and ky[peak] == scanned.max():
            lines.append("  The peak is at the highest ky: run higher ky.")
        top = gamma[have][-1]
        if gamma[peak] > 0 and not np.isnan(top) and top > 0.05 * gamma[peak]:
            lines.append(
                f"  gamma at the highest ky, {scanned.max():g}, is still "
                f"{top / gamma[peak]:.0%} of the peak."
            )
    return lines


__all__ = [
    "DEFAULT_KY",
    "DEFAULT_OMEGATOL",
    "DEFAULT_OMEGA_WINDOW",
    "KY_KEY",
    "box_ky",
    "converged",
    "ky_values",
    "linear_settings",
    "scanned_key",
    "status",
    "stopped",
    "submit",
    "summary",
    "write",
]
