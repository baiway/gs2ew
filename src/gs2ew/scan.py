"""Parameter scans: GS2 input files written from a template, and their jobs.

A scan is a template input file and a list of points, each a set of changes
to it. Every point gets a run directory holding its input file, named after
the directory, and a copy of a Slurm template::

    scan/
      scan.json             what was written, from what, by which gs2ew
      template.in           the template, as it was
      jobs.csv              the jobs submitted, written by `submit`
      ntheta_32/
        ntheta_32.in        the template with ntheta = 32, comments and all
        job.slurm
        restart/  response/ if the input file names them

The points are built with :func:`product`, :func:`zipped` and
:func:`one_at_a_time`, which can be combined with :func:`combine`::

    from gs2ew import scan

    points = scan.one_at_a_time({"ntheta": [32, 96], "nx": [128, 256]})
    scan.write("NL_cbc.in", points, "convergence/", machine="archer2")

A key is a name from the input file, qualified by its namelist
(``species_parameters_2.tprim``) where the bare name is ambiguous or not in the
template. A value that must follow the scanned ones, such as ``tripri`` with
``tri``, is given by a `derive` function. Not the pressure gradient, though: set
``auto_set_beta_prime = .true.`` in the template and GS2 works it out from
``beta`` and the species itself (:func:`write` warns if a scan needs it).

Nothing is changed but the values asked for, so a run's input file differs
from the template only in those lines. Inline comments on those lines are
kept, so they may no longer be true.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import f90nml

from gs2ew.io import OUTPUT_SUFFIX
from gs2ew.provenance import gs2ew_version

TEMPLATES_DIR = Path(__file__).parent / "templates"

#: What :func:`write` puts in the scan directory.
MANIFEST = "scan.json"
TEMPLATE_INPUT = "template.in"
JOB_SCRIPT = "job.slurm"
#: Written by :func:`submit`: one row per job.
JOBS = "jobs.csv"

#: The run directory of a point that changes nothing.
BASE_NAME = "base"

#: Directories GS2 writes into but does not create, as (namelist, key).
_DIRECTORY_KEYS = (("init_g_knobs", "restart_dir"), ("fields_knobs", "response_dir"))

#: What the pressure gradient depends on, and the geometry inputs that give
#: it. With ``auto_set_beta_prime``, GS2 sets the inputs from the species and
#: ``beta``, unless the input file gives them itself.
_PRESSURE_KEYS = {"beta", "tprim", "fprim", "dens", "temp"}
_PRESSURE_INPUTS = ("beta_prime_input", "alpha_input", "p_prime_input", "invlp_input")

#: A point: values keyed by ``"namelist.key"``, or by a bare key before
#: :func:`write` resolves it against the template.
Point = dict[str, object]


def machines() -> list[str]:
    """Names of the Slurm templates shipped with gs2ew."""
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.slurm"))


def product(parameters: Mapping[str, Sequence]) -> list[Point]:
    """Every combination of the values of `parameters`."""
    keys = list(parameters)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(parameters[k] for k in keys))
    ]


def zipped(parameters: Mapping[str, Sequence]) -> list[Point]:
    """The i-th values of every parameter together, for values that are
    paired by hand (``tri`` with ``tripri``, say).

    Raises
    ------
    ValueError
        If the parameters have different numbers of values.
    """
    lengths = {k: len(v) for k, v in parameters.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"zipped parameters need equally many values: {lengths}")
    keys = list(parameters)
    return [dict(zip(keys, values)) for values in zip(*(parameters[k] for k in keys))]


def one_at_a_time(parameters: Mapping[str, Sequence]) -> list[Point]:
    """The template itself, then each value of each parameter on its own, as a
    convergence test wants."""
    return [{}] + [{key: value} for key, values in parameters.items() for value in values]


def combine(*scans: Sequence[Point]) -> list[Point]:
    """Every combination of a point from each of `scans`: a shaping scan,
    zipped, over each of a list of betas, say."""
    return [
        {key: value for point in points for key, value in point.items()}
        for points in itertools.product(*scans)
    ]


def resolve_key(deck: f90nml.Namelist, key: str) -> str:
    """`key` as ``"namelist.key"``, finding the namelist in `deck` if not given.

    A qualified key is taken as it is, so it may add a value the template
    leaves at its default.

    Raises
    ------
    ValueError
        If a bare key is in none of `deck`'s namelists, or in more than one.
    """
    if "." in key:
        group, name = key.lower().split(".", 1)
        return f"{group}.{name}"
    name = key.lower()
    groups = [group for group in deck if name in deck[group]]
    if not groups:
        raise ValueError(
            f"{key!r} is not in the template; give it with its namelist, "
            f"as NAMELIST.{key}."
        )
    if len(groups) > 1:
        raise ValueError(
            f"{key!r} is in several namelists ({', '.join(groups)}); give it "
            f"as one of {', '.join(f'{g}.{name}' for g in groups)}."
        )
    return f"{groups[0]}.{name}"


def write(
    template: str | Path,
    points: Sequence[Point],
    directory: str | Path,
    machine: str | Path | None = "archer2",
    derive: Callable[[Point, f90nml.Namelist], Point] | None = None,
) -> list[Path]:
    """Write a run directory for each of `points` under `directory`.

    Parameters
    ----------
    template : str or Path
        The GS2 input file each run starts from.
    points : list of dict
        The changes to make for each run; see :func:`product` and friends.
    directory : str or Path
        Where to write the scan. It must not exist, or be empty.
    machine : str or Path, optional
        Which Slurm template to copy into each run directory: one of
        :func:`machines`, or the path to one of your own, which is run as
        ``sbatch job.slurm RUN.in`` from the run directory. None for no job
        script. Default ``"archer2"``.
    derive : callable, optional
        ``derive(point, deck)`` returns further values to set, given a point
        (with qualified keys) and the template with that point applied. These
        are written like the point's own, and recorded as derived.

    Returns
    -------
    list of Path
        The run directories, in the order of `points`. A point that repeats
        the template's values is skipped if the template itself (``{}``) is
        one of `points`.

    Raises
    ------
    FileExistsError
        If `directory` exists and is not empty.
    ValueError
        If a key cannot be resolved, or two points would share a directory.
    """
    template = Path(template)
    directory = Path(directory)
    script = _job_script(machine)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} already exists and is not empty.")

    deck = f90nml.read(template)
    runs = []
    for point in points:
        values = {}
        for key, value in point.items():
            resolved = resolve_key(deck, key)
            values[resolved] = _like_template(deck, resolved, value)
        applied = _apply(deck, values)
        derived = {}
        if derive is not None:
            for key, value in derive(dict(values), applied).items():
                resolved = resolve_key(deck, key)
                if resolved in values:
                    raise ValueError(f"derive() sets {resolved}, which is scanned.")
                derived[resolved] = _like_template(deck, resolved, value)
        runs.append((values, derived))

    # With the template among the points, a point repeating its values would
    # run it twice: a convergence scan's list of ntheta may well include the
    # template's own.
    if any(not values for values, _ in runs):
        kept = []
        for values, derived in runs:
            if values and all(_template_value(deck, k) == v for k, v in values.items()):
                described = ", ".join(f"{k} = {v}" for k, v in values.items())
                print(f"Skipping {described}: the template's own, run as {BASE_NAME}/.")
                continue
            kept.append((values, derived))
        runs = kept

    names = _run_names([values for values, _ in runs])
    _check_restart_reads(deck)
    _check_pressure_gradient(
        deck,
        scanned={key for values, _ in runs for key in values},
        derived={key for _, derived in runs for key in derived},
    )

    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, directory / TEMPLATE_INPUT)
    run_dirs = []
    for name, (values, derived) in zip(names, runs):
        run_dir = directory / name
        run_dir.mkdir()
        changes = {**values, **derived}
        f90nml.patch(str(template), _nested(changes), str(run_dir / f"{name}.in"))
        for subdirectory in _directories_named(_apply(deck, changes)):
            (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
        if script is not None:
            shutil.copy(script, run_dir / JOB_SCRIPT)
        run_dirs.append(run_dir)

    manifest = {
        "template": str(template.resolve()),
        "machine": None if machine is None else str(machine),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        **gs2ew_version().attrs(),
        "runs": [
            {"name": name, "values": values, "derived": derived}
            for name, (values, derived) in zip(names, runs)
        ],
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dirs


def load(directory: str | Path) -> dict:
    """The manifest :func:`write` left in `directory`."""
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"{directory} is not a scan: it has no {MANIFEST}.")
    return json.loads(path.read_text())


def submit(
    directory: str | Path,
    sbatch_args: Sequence[str] = (),
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Submit each run in the scan that has neither output nor a job yet.

    Each is submitted from its run directory as ``sbatch --job-name=SCAN-RUN
    [sbatch_args] job.slurm RUN.in``, and its job ID added to ``jobs.csv``.
    With `dry_run`, the commands are returned without being run.

    Returns
    -------
    list of (run, outcome)
        The job ID, the command (with `dry_run`), or why it was skipped.
    """
    directory = Path(directory)
    manifest = load(directory)
    if manifest["machine"] is None:
        raise ValueError(f"{directory} was written without a job script.")
    jobs = _read_jobs(directory)
    if not dry_run and shutil.which("sbatch") is None:
        raise FileNotFoundError("sbatch not found: submit from a Slurm login node.")

    outcomes = []
    short = {}  # (avail_cpu_time, time limit) -> runs it is too long for
    for run in manifest["runs"]:
        name = run["name"]
        run_dir = directory / name
        if any(run_dir.glob(f"*{OUTPUT_SUFFIX}")):
            outcomes.append((name, "skipped: has output"))
            continue
        if name in jobs:
            outcomes.append((name, f"skipped: submitted as job {jobs[name]}"))
            continue
        too_long = _too_long(run_dir, name, sbatch_args)
        if too_long is not None:
            short.setdefault(too_long, []).append(name)
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={directory.resolve().name}-{name}",
            *sbatch_args,
            JOB_SCRIPT,
            f"{name}.in",
        ]
        if dry_run:
            outcomes.append((name, f"(cd {run_dir} && {shlex.join(command)})"))
            continue
        result = subprocess.run(
            command, cwd=run_dir, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            outcomes.append((name, f"sbatch failed: {result.stderr.strip()}"))
            continue
        # --parsable prints "jobid" or "jobid;cluster".
        job = result.stdout.strip().split(";")[0]
        _record_job(directory, name, job)
        outcomes.append((name, f"job {job}"))

    for (available, limit), names in short.items():
        stated = "is not set" if available is None else f"is {available:g} s"
        print(
            f"Warning: avail_cpu_time {stated} but the job's time limit is "
            f"{limit} s, so Slurm will stop GS2 before it writes its restart "
            f"files: {', '.join(names)}."
        )
    return outcomes


def status(directory: str | Path) -> list[tuple[str, str]]:
    """Where each run in the scan has got to.

    A run is finished once GS2 has written its ``.exit_reason``. Before that,
    the job's state comes from ``squeue`` if it is available.
    """
    directory = Path(directory)
    manifest = load(directory)
    jobs = _read_jobs(directory)
    queue = _queue_states(list(jobs.values()))

    states = []
    for run in manifest["runs"]:
        name = run["name"]
        exit_reason = directory / name / f"{name}.exit_reason"
        job = jobs.get(name)
        if exit_reason.is_file():
            reason = "; ".join(
                line.strip() for line in exit_reason.read_text().splitlines() if line.strip()
            )
            states.append((name, f"finished: {reason}"))
        elif job is None:
            states.append((name, "not submitted"))
        elif queue is None:
            states.append((name, f"job {job}: no exit reason yet"))
        elif job in queue:
            states.append((name, f"job {job}: {queue[job].lower()}"))
        else:
            states.append((name, f"job {job}: ended without an exit reason"))
    return states


def _job_script(machine: str | Path | None) -> Path | None:
    if machine is None:
        return None
    if str(machine) in machines():
        return TEMPLATES_DIR / f"{machine}.slurm"
    if Path(machine).expanduser().is_file():
        return Path(machine).expanduser()
    raise ValueError(
        f"Unknown machine {str(machine)!r}: not a file, nor one of "
        f"{', '.join(machines())}."
    )


def _template_value(deck: f90nml.Namelist, key: str) -> object:
    group, name = key.split(".")
    return deck.get(group, {}).get(name)


def _like_template(deck: f90nml.Namelist, key: str, value: object) -> object:
    """`value` as the type the template gives `key`, so ``beta = 0`` is still
    written as a real and ``ntheta = 32.0`` as an integer."""
    current = _template_value(deck, key)
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if (
        isinstance(current, int)
        and not isinstance(current, bool)
        and isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)
    return value


def _nested(values: Point) -> f90nml.Namelist:
    nested: dict[str, dict[str, object]] = {}
    for key, value in values.items():
        group, name = key.split(".")
        nested.setdefault(group, {})[name] = value
    return f90nml.Namelist(nested)


def _apply(deck: f90nml.Namelist, values: Point) -> f90nml.Namelist:
    applied = copy.deepcopy(deck)
    for key, value in values.items():
        group, name = key.split(".")
        applied.setdefault(group, f90nml.Namelist())[name] = value
    return applied


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        text = str(int(value)) if value.is_integer() else repr(value)
    else:
        text = str(value)
    return re.sub(r"[\s/]+", "-", text.replace(".", "p"))


def _run_names(points: Sequence[Point]) -> list[str]:
    """``key_value`` for each scanned value, joined by ``-``; ``base`` for a
    point that changes nothing. Keys go by their bare name, with the species
    number added where two namelists share one (``tprim1``, ``tprim2``)."""
    keys = list(dict.fromkeys(key for point in points for key in point))
    bare = [key.split(".")[1] for key in keys]
    short = {}
    for key, name in zip(keys, bare):
        if bare.count(name) == 1:
            short[key] = name
        else:
            group = key.split(".")[0]
            index = re.search(r"_(\d+)$", group)
            short[key] = f"{name}{index.group(1)}" if index else key.replace(".", "_")

    names = [
        "-".join(f"{short[k]}_{_format_value(v)}" for k, v in point.items())
        or BASE_NAME
        for point in points
    ]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Several points would share a directory: {', '.join(repeated)}.")
    return names


def _directories_named(deck: f90nml.Namelist) -> list[str]:
    """The relative directories `deck` tells GS2 to write into."""
    found = []
    for group, key in _DIRECTORY_KEYS:
        value = deck.get(group, {}).get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            found.append(value)
    return found


def _check_restart_reads(deck: f90nml.Namelist) -> None:
    """Warn about settings that read files a fresh run directory lacks."""
    if deck.get("fields_knobs", {}).get("read_response"):
        print(
            "Warning: the template sets read_response = .true., but the run "
            "directories have no response matrices to read."
        )
    if str(deck.get("init_g_knobs", {}).get("ginit_option", "")).lower() == "many":
        print(
            "Warning: the template sets ginit_option = 'many', a restart, but "
            "the run directories have no restart files."
        )


def _check_pressure_gradient(
    deck: f90nml.Namelist, scanned: set[str], derived: set[str]
) -> None:
    """Warn if the geometry's pressure gradient won't follow a scanned beta or
    species gradient."""
    if not any(key.split(".")[1] in _PRESSURE_KEYS for key in scanned):
        return
    if any(key.split(".")[1] in _PRESSURE_INPUTS for key in scanned | derived):
        return
    if not deck.get("theta_grid_parameters", {}).get("auto_set_beta_prime"):
        print(
            "Warning: beta or a species gradient is scanned, but the template "
            "does not set auto_set_beta_prime = .true. in theta_grid_parameters, "
            "so the geometry's pressure gradient stays as the template gives it."
        )
        return
    fixed = [key for key in _PRESSURE_INPUTS if key in deck.get("theta_grid_eik_knobs", {})]
    if fixed:
        print(
            f"Warning: the template sets auto_set_beta_prime, but also "
            f"{', '.join(fixed)} in theta_grid_eik_knobs, which GS2 uses in place "
            "of the values auto_set_beta_prime would give."
        )


def _slurm_seconds(text: str) -> int | None:
    """A Slurm time limit in seconds: minutes, MM:SS, HH:MM:SS, D-HH,
    D-HH:MM or D-HH:MM:SS."""
    days, sep, rest = text.rpartition("-")
    try:
        parts = [int(p) for p in rest.split(":")]
        days_ = int(days) if sep else 0
    except ValueError:
        return None
    if sep:  # D-HH[:MM[:SS]]
        parts += [0] * (3 - len(parts))
        hours, minutes, seconds = parts
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        hours, minutes, seconds = parts
    return ((days_ * 24 + hours) * 60 + minutes) * 60 + seconds


def _time_limit(script: Path, sbatch_args: Sequence[str]) -> int | None:
    """The job's time limit in seconds: from `sbatch_args` if given there,
    otherwise from the script's ``#SBATCH`` lines."""
    limit = None
    pattern = re.compile(r"^#SBATCH\s+(?:--time[=\s]|-t\s*)(\S+)")
    for line in script.read_text().splitlines():
        match = pattern.match(line)
        if match:
            limit = match.group(1)
    args = list(sbatch_args)
    for i, arg in enumerate(args):
        if arg.startswith("--time="):
            limit = arg.split("=", 1)[1]
        elif arg in ("--time", "-t") and i + 1 < len(args):
            limit = args[i + 1]
        elif arg.startswith("-t") and len(arg) > 2:
            limit = arg[2:]
    return None if limit is None else _slurm_seconds(limit)


def _too_long(
    run_dir: Path, name: str, sbatch_args: Sequence[str]
) -> tuple[float | None, int] | None:
    """``(avail_cpu_time, time limit)`` if GS2 would be killed before stopping
    itself, or None.

    GS2 stops cleanly, writing its restart files, once it has used
    ``avail_cpu_time`` less ``margin_cpu_time``; a job killed by Slurm first
    leaves nothing to restart from.
    """
    limit = _time_limit(run_dir / JOB_SCRIPT, sbatch_args)
    if limit is None:
        return None
    available = f90nml.read(run_dir / f"{name}.in").get("knobs", {}).get("avail_cpu_time")
    if available is None:
        return None, limit
    return (float(available), limit) if float(available) > limit else None


def _read_jobs(directory: Path) -> dict[str, str]:
    """The latest job submitted for each run."""
    path = directory / JOBS
    if not path.is_file():
        return {}
    with path.open(newline="") as f:
        return {row["run"]: row["job"] for row in csv.DictReader(f)}


def _record_job(directory: Path, name: str, job: str) -> None:
    path = directory / JOBS
    new = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["run", "job", "submitted"])
        writer.writerow(
            [name, job, datetime.now(timezone.utc).isoformat(timespec="seconds")]
        )


def _queue_states(jobs: list[str]) -> dict[str, str] | None:
    """The state of each of your jobs still in the queue, or None if
    ``squeue`` can't say."""
    if not jobs or shutil.which("squeue") is None:
        return None
    # All your jobs rather than `jobs`: squeue fails outright if one of those
    # has been forgotten.
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--format=%i %T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return dict(line.split()[:2] for line in result.stdout.splitlines() if line.strip())


__all__ = [
    "BASE_NAME",
    "JOBS",
    "JOB_SCRIPT",
    "MANIFEST",
    "TEMPLATE_INPUT",
    "combine",
    "load",
    "machines",
    "one_at_a_time",
    "product",
    "resolve_key",
    "status",
    "submit",
    "write",
    "zipped",
]
