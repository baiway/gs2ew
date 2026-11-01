"""Parameter scans: GS2 input files written from a template, and their jobs.

A scan is a template input file and a list of points, each a set of changes
to it. Every point gets a run directory holding its input file, named after
the directory, and a copy of a Slurm template::

    scan/
      scan.json             what was written, from what, by which gs2ew
      template.in           the template, as it was
      jobs.csv              the jobs submitted, by `submit` and `restart`
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
import math
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
from gs2ew.provenance import RESTART_KEYS, gs2ew_version

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

#: The diagnostics GS2 switches off unless every process holds whole (kx,
#: ky) planes (``diagnostics_configuration.f90``). The kinetic-energy
#: transfer is not among them.
_TRANSFER_SWITCHES = (
    "write_entropy_transfer",
    "write_entropy_transfer_trapped_passing",
    "write_fluctuation_energy_transfer",
    "write_fluctuation_energy_transfer_trapped_passing",
    "write_fluctuation_energy_transfer_split",
    "write_free_energy_transfer",
    "write_free_energy_transfer_trapped_passing",
)

#: GS2's names for the dimensions of its distribution-function layout.
_DIMENSIONS = {"x": "ntheta0", "y": "naky", "l": "nlambda", "e": "negrid", "s": "nspec"}

#: The code GS2 writes to ``.exit_reason`` when it stops at avail_cpu_time.
TIME_LIMIT_EXIT = -1

#: What :func:`restart` sets in a run's input file, besides ``read_response``
#: for a run that saved its response matrices.
RESTART_SETTINGS = {
    "init_g_knobs.ginit_option": "many",
    "gs2_diagnostics_knobs.append_old": True,
    "knobs.delt_option": "check_restart",
}
assert set(RESTART_SETTINGS) < RESTART_KEYS

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
    ingen: str | Path | None = None,
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
    ingen : str or Path, optional
        GS2's ``ingen``, run on each input file for the ``.report`` giving
        its grid sizes exactly, which the core-count checks then use; see
        :func:`submit`. Without it, they use an estimate.

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

    if _writes_transfer(deck):
        _check_cores_early(run_dirs, ingen)

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
    check_cores: bool = True,
) -> list[tuple[str, str]]:
    """Submit each run in the scan that has neither output nor a job yet.

    Each is submitted from its run directory as ``sbatch --job-name=SCAN-RUN
    [sbatch_args] job.slurm RUN.in``, and its job ID added to ``jobs.csv``.
    With `dry_run`, the commands are returned without being run.

    The core count asked for, by the job script or `sbatch_args`, is the
    most a run gets. A run that writes the transfer diagnostics, which can't
    work with (kx, ky) split across processes, gets the largest count up to
    it that keeps them whole, unless `check_cores` is False.

    Returns
    -------
    list of (run, outcome)
        The job ID, the command (with `dry_run`), or why it was not submitted.
    """
    directory = Path(directory)
    manifest, jobs = _open_scan(directory, dry_run)
    outcomes = {}
    pending = []
    for name in _names(manifest):
        if any((directory / name).glob(f"*{OUTPUT_SUFFIX}")):
            outcomes[name] = "skipped: has output"
        elif name in jobs:
            outcomes[name] = f"skipped: submitted as job {jobs[name]['job']}"
        else:
            pending.append(name)
    outcomes.update(_submit_runs(directory, pending, sbatch_args, dry_run, check_cores))
    return [(name, outcomes[name]) for name in _names(manifest)]


def restart(
    directory: str | Path,
    runs: Sequence[str] | None = None,
    sbatch_args: Sequence[str] = (),
    dry_run: bool = False,
    check_cores: bool = True,
) -> list[tuple[str, str]]:
    """Restart the runs GS2 stopped at ``avail_cpu_time``, or `runs`, whatever
    stopped them.

    Each run's input file is set to restart, in place: ``ginit_option =
    'many'`` to start from its restart files, ``append_old`` to add to its
    output, ``delt_option = 'check_restart'`` to carry on with the timestep it
    had reached, and ``read_response`` if it saved its response matrices.
    Then it is submitted as by :func:`submit`.

    A run can only be restarted if it set ``save_for_restart`` and its
    restart files are there. It runs on as many cores as before unless
    `sbatch_args` ask for others, which with ``save_many`` (restart files
    one per process) it can't. Each restart runs another
    ``nstep`` steps; ``max_sim_time`` stops a run at a given time instead.

    Returns
    -------
    list of (run, outcome)
        As :func:`submit`, for the runs considered.
    """
    directory = Path(directory)
    manifest, jobs = _open_scan(directory, dry_run)
    names = _names(manifest)
    if runs is not None:
        unknown = sorted(set(runs) - set(names))
        if unknown:
            raise ValueError(f"Not runs of this scan: {', '.join(unknown)}.")
    queue = _queue_states([row["job"] for row in jobs.values()])

    outcomes = {}
    pending = []
    for name in names:
        if runs is not None and name not in runs:
            continue
        job = jobs.get(name)
        if job is None:
            outcomes[name] = "skipped: never submitted"
            continue
        if queue is not None and job["job"] in queue:
            outcomes[name] = f"skipped: job {job['job']} is {queue[job['job']].lower()}"
            continue
        if runs is None:
            reasons = _exit_reasons(directory / name, name, job)
            if reasons is None:
                outcomes[name] = "skipped: no exit reason since its last job"
                continue
            if TIME_LIMIT_EXIT not in reasons:
                outcomes[name] = f"skipped: {'; '.join(reasons.values())}"
                continue
        problem = _cannot_restart(directory / name, name)
        if problem is not None:
            outcomes[name] = f"cannot restart: {problem}"
            continue
        pending.append(name)

    outcomes.update(
        _submit_runs(directory, pending, sbatch_args, dry_run, check_cores, restarting=jobs)
    )
    return [(name, outcomes[name]) for name in names if name in outcomes]


def status(directory: str | Path) -> list[tuple[str, str]]:
    """Where each run in the scan has got to.

    A run has finished once GS2 has written its ``.exit_reason`` (one older
    than the run's latest job is from before a restart). Until then, the
    job's state comes from ``squeue`` if it is available.
    """
    directory = Path(directory)
    manifest = load(directory)
    jobs = _read_jobs(directory)
    queue = _queue_states([row["job"] for row in jobs.values()])

    states = []
    for name in _names(manifest):
        job = jobs.get(name)
        number = None if job is None else job["job"]
        reasons = _exit_reasons(directory / name, name, job)
        if queue is not None and number in queue:
            state = f"job {number}: {queue[number].lower()}"
        elif reasons is not None:
            state = f"finished: {'; '.join(reasons.values())}"
            if TIME_LIMIT_EXIT in reasons:
                state += " (can be restarted)"
        elif job is None:
            state = "not submitted"
        elif queue is None:
            state = f"job {number}: no exit reason yet"
        else:
            state = f"job {number}: ended without an exit reason"
        states.append((name, state))
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
    """Warn about a restart, which a fresh run directory has no files for.

    (``read_response`` is fine: GS2 calculates the response matrices if it
    finds none.)
    """
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


#: sbatch's short options that gs2ew reads, by their long names.
_SHORT_OPTIONS = {"-t": "time", "-n": "ntasks", "-N": "nodes"}

#: The sbatch options that set a job's core count.
_RESOURCE_OPTIONS = {"nodes", "ntasks", "ntasks-per-node"}


def _option_groups(tokens: Sequence[str]) -> list[tuple[str | None, list[str]]]:
    """`tokens` grouped into options, as (long name, its tokens); anything
    else is (None, [token])."""
    groups = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name, taken = None, [token]
        if token.startswith("--"):
            name, sep, _ = token[2:].partition("=")
            if not sep and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                taken.append(tokens[i + 1])
        elif token[:2] in _SHORT_OPTIONS:
            name = _SHORT_OPTIONS[token[:2]]
            if len(token) == 2 and i + 1 < len(tokens):
                taken.append(tokens[i + 1])
        groups.append((name, taken))
        i += len(taken)
    return groups


def _sbatch_options(script: Path, sbatch_args: Sequence[str]) -> dict[str, str]:
    """The options the job is submitted with: the script's ``#SBATCH`` lines,
    overridden by `sbatch_args`. Short forms are given their long names."""
    tokens = []
    for line in script.read_text().splitlines():
        if line.startswith("#SBATCH"):
            tokens += shlex.split(line[len("#SBATCH"):], comments=True)
    options = {}
    for name, taken in _option_groups([*tokens, *sbatch_args]):
        if name is None:
            continue
        if len(taken) == 2:
            options[name] = taken[1]
        elif taken[0].startswith("--"):
            options[name] = taken[0].partition("=")[2]
        else:
            options[name] = taken[0][2:]
    return options


def _without_resources(sbatch_args: Sequence[str]) -> list[str]:
    return [
        token
        for name, taken in _option_groups(sbatch_args)
        if name not in _RESOURCE_OPTIONS
        for token in taken
    ]


def _cores(options: Mapping[str, str]) -> tuple[int | None, int | None]:
    """The job's core count and cores per node, where `options` say."""

    def number(name: str) -> int | None:
        try:
            return int(options[name])
        except (KeyError, ValueError):  # absent, or a range such as 2-4
            return None

    ntasks, nodes, per_node = number("ntasks"), number("nodes"), number("ntasks-per-node")
    if ntasks is not None:
        return ntasks, per_node
    if nodes is not None and per_node is not None:
        return nodes * per_node, per_node
    return None, per_node


def _too_long(
    deck: f90nml.Namelist, options: Mapping[str, str]
) -> tuple[float | None, int] | None:
    """``(avail_cpu_time, time limit)`` if GS2 would be killed before stopping
    itself, or None.

    GS2 stops cleanly, writing its restart files, once it has used
    ``avail_cpu_time`` less ``margin_cpu_time``; a job killed by Slurm first
    leaves nothing to restart from.
    """
    limit = _slurm_seconds(options["time"]) if "time" in options else None
    if limit is None:
        return None
    available = deck.get("knobs", {}).get("avail_cpu_time")
    if available is None:
        return None, limit
    return (float(available), limit) if float(available) > limit else None


def _layout_sizes(
    deck: f90nml.Namelist, report: Path
) -> tuple[str, dict[str, int | None], bool]:
    """GS2's layout and the sizes of its dimensions, keyed by layout letter,
    and whether they are exact.

    Exact from the ``.report`` GS2 writes when it starts. Before that, they
    are worked out from the input: nlambda is the passing pitch angles
    (``npassing``, or ``2 * ngauss``) plus one trapped pitch angle per bounce
    point, which GS2 bounds by ``ntheta / 2 + 1``, the number an up-down
    symmetric surface has.
    """
    layout = str(deck.get("layouts_knobs", {}).get("layout", "lxyes")).lower()
    if report.is_file():
        text = report.read_text()
        found = dict(re.findall(r"^\s*(nlambda|negrid|ntheta0|naky|nspec)=\s*(\d+)", text, re.MULTILINE))
        written = re.search(r"^Layout = (\w+)", text, re.MULTILINE)
        if len(found) == len(_DIMENSIONS) and written:
            sizes = {letter: int(found[name]) for letter, name in _DIMENSIONS.items()}
            return written.group(1), sizes, True

    le = deck.get("le_grids_knobs", {})
    npassing = le.get("npassing", -1)
    passing = npassing if npassing >= 0 else 2 * le.get("ngauss", 5)
    ntheta = deck.get("theta_grid_parameters", {}).get("ntheta")
    trapped = le.get("trapped_particles", True) and ntheta is not None
    negrid = le.get("negrid", -10)
    box = deck.get("kt_grids_box_parameters", {})
    sizes = {
        "x": 2 * ((box["nx"] - 1) // 3) + 1 if "nx" in box else None,
        "y": (box["ny"] - 1) // 3 + 1 if "ny" in box else None,
        "l": passing + ntheta // 2 + 1 if trapped else passing,
        "e": negrid if negrid > 0 else le.get("nesub", 8) + le.get("nesuper", 2),
        "s": deck.get("species_knobs", {}).get("nspec", 2),
    }
    return layout, sizes, False


def _planes_whole(layout: str, sizes: Mapping[str, int | None], cores: int) -> bool:
    """Whether each of `cores` processes holds whole (kx, ky) planes.

    GS2 gives each process a contiguous block of ``ceil(N / cores)`` of the
    N points of its layout (``gs2_layouts.fpp``), so the block must be a
    whole number of the stretches that run over kx and ky; in practice,
    `cores` must divide the product of the dimensions outside them.
    """
    last = max(layout.index("x"), layout.index("y"))
    inner = [sizes[c] for c in layout[: last + 1]]
    outer = math.prod(sizes[c] for c in layout[last + 1 :])
    if None in inner:
        return outer % cores == 0
    stretch = math.prod(inner)
    return -(-stretch * outer // cores) % stretch == 0


def _splits_evenly(layout: str, sizes: Mapping[str, int | None], cores: int) -> bool:
    """Whether `cores` splits GS2's layout evenly, dimension by dimension from
    the outside in: what ``ingen`` calls a sweetspot.

    For nlambda = 48, negrid = 16 and nspec = 2, that is 1, 2, 4, ..., 32,
    then 32 times each divisor of 48; not 3, say, which divides the whole but
    not nspec.
    """
    remaining = cores
    for letter in reversed(layout):
        if remaining <= 1:
            return True
        size = sizes[letter]
        if size is None:
            return False
        if remaining < size:
            return size % remaining == 0
        if remaining % size:
            return False
        remaining //= size
    return remaining <= 1


def _sweetspots(report: Path) -> set[int] | None:
    """The core counts ``ingen``'s report recommends for a nonlinear run: in
    both its xxf and its yxf sweetspots, the layouts of the nonlinear term's
    transforms. None without a report that lists them."""
    if not report.is_file():
        return None
    found: dict[str, set[int]] = {}
    current = None
    for line in report.read_text().splitlines():
        header = re.match(r"(xxf|yxf) sweetspot #proc", line)
        npe = re.match(r"\s*npe =\s*(\d+)", line)
        if header:
            current = found.setdefault(header.group(1), set())
        elif npe and current is not None:
            current.add(int(npe.group(1)))
        elif line.strip():
            current = None
    if len(found) < 2:
        return None
    return found["xxf"] & found["yxf"]


def _choose_cores(
    deck: f90nml.Namelist, report: Path, target: int
) -> tuple[int, str] | None:
    """The core count to run on instead of `target`, and why; None if
    `target` will do.

    The count must keep (kx, ky) whole on each process if the run writes the
    transfer diagnostics, and for a nonlinear run should be a sweetspot,
    which keeps the transforms of the nonlinear term cheap: from ``ingen``'s
    report, or worked out the same way without one. The largest count up to
    `target` that is both is chosen.
    """
    transfer = _writes_transfer(deck)
    nonlinear = str(deck.get("nonlinear_terms_knobs", {}).get("nonlinear_mode", "off")).lower() == "on"
    if not (transfer or nonlinear):
        return None
    layout, sizes, exact = _layout_sizes(deck, report)
    spots = _sweetspots(report)

    def whole(c: int) -> bool:
        return not transfer or _planes_whole(layout, sizes, c)

    def sweet(c: int) -> bool:
        if not nonlinear:
            return True
        if spots is not None:
            return c in spots
        # Beyond splitting nlambda, negrid and nspec, the transforms' own
        # layouts decide, which only ingen knows; so leave those counts be.
        return c > outer_product or _splits_evenly(layout, sizes, c)

    last = max(layout.index("x"), layout.index("y"))
    outer = layout[last + 1 :]
    known = all(sizes[c] is not None for c in outer)
    outer_product = math.prod(sizes[c] for c in outer) if known else 0

    if whole(target) and sweet(target):
        return None
    # One core is both, unless a report says otherwise.
    cores = next((c for c in range(target - 1, 0, -1) if whole(c) and sweet(c)), None)
    if cores is None:
        return None

    if not whole(target):
        product = " * ".join(_DIMENSIONS[c] for c in outer)
        if known:
            product += f" = {outer_product}"
        estimate = "" if exact else f", estimating nlambda as {sizes['l']}"
        why = (
            f"{target} cores would split the (kx, ky) planes the transfer "
            f"diagnostics need, not dividing {product}{estimate}"
        )
    elif spots is not None:
        why = f"{target} cores isn't one of ingen's xxf and yxf sweetspots"
    else:
        why = f"{target} cores doesn't split GS2's layout evenly (a sweetspot)"
    return cores, why


def _resource_args(
    cores: int, per_node: int | None, options: Mapping[str, str]
) -> tuple[list[str], str]:
    """sbatch options asking for `cores`, and a description of them.

    With a most per node, on the fewest nodes that share them evenly, so
    full nodes where `cores` allows.
    """
    if per_node is None:
        return [f"--ntasks={cores}"], f"{cores} cores"
    nodes = next(n for n in range(-(-cores // per_node), cores + 1) if cores % n == 0)
    args = [f"--nodes={nodes}", f"--ntasks-per-node={cores // nodes}"]
    if "ntasks" in options:
        args.append(f"--ntasks={cores}")
    shape = f"{nodes} nodes" if cores // nodes == per_node else f"{nodes} nodes of {cores // nodes}"
    return args, f"{cores} cores ({shape})"


def _writes_transfer(deck: f90nml.Namelist) -> bool:
    diagnostics = deck.get("gs2_diagnostics_knobs", {})
    return any(diagnostics.get(name) for name in _TRANSFER_SWITCHES)


def _run_ingen(ingen: str | Path, run_dir: Path, name: str) -> bool:
    """Run ``ingen`` on the run's input for its ``.report``, and whether that
    worked. Everything else ``ingen`` writes is removed: GS2 writes its own
    when it runs, and an empty ``.exit_reason`` would read as a finished run.
    """
    report = run_dir / f"{name}.report"
    before = set(run_dir.iterdir())
    try:
        result = subprocess.run(
            [str(ingen), f"{name}.in"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        worked = result.returncode == 0 and report.is_file()
    except (OSError, subprocess.SubprocessError):
        worked = False
    for path in set(run_dir.iterdir()) - before:
        if path.is_file() and (path != report or not worked):
            path.unlink()
    return worked


def _check_cores_early(run_dirs: Sequence[Path], ingen: str | Path | None) -> None:
    """Get each run's grid sizes from `ingen` if given, and say which runs
    the job script's core count won't do for."""
    failed = []
    if ingen is not None:
        failed = [d.name for d in run_dirs if not _run_ingen(ingen, d, d.name)]
    if ingen is None or failed:
        which = "all runs" if ingen is None else ", ".join(failed)
        why = "without ingen" if ingen is None else "as ingen failed"
        print(
            f"Note: {why}, core counts will be checked against an estimate of "
            f"nlambda for {which}: an upper bound, which strong shaping can "
            "bring down."
        )

    for run_dir in run_dirs:
        script = run_dir / JOB_SCRIPT
        if not script.is_file():
            continue
        options = _sbatch_options(script, ())
        cores, per_node = _cores(options)
        if cores is None:
            continue
        deck = f90nml.read(run_dir / f"{run_dir.name}.in")
        choice = _choose_cores(deck, run_dir / f"{run_dir.name}.report", cores)
        if choice is not None:
            _, described = _resource_args(choice[0], per_node, options)
            print(f"Note: {run_dir.name} will run on {described}: {choice[1]}.")


def _cannot_restart(run_dir: Path, name: str) -> str | None:
    """Why the run in `run_dir` can't be restarted, or None if it can."""
    deck = f90nml.read(run_dir / f"{name}.in")
    if not deck.get("gs2_diagnostics_knobs", {}).get("save_for_restart"):
        return "it doesn't set save_for_restart, so saved nothing to restart from"
    init = deck.get("init_g_knobs", {})
    restart_file = Path(init.get("restart_file", f"{name}.nc"))
    where = run_dir / restart_file.parent / init.get("restart_dir", "./")
    if not any(where.glob(f"{restart_file.name}*")):
        return f"no restart files {restart_file.name}* in {where.relative_to(run_dir)}/"
    return None


def _set_restart(path: Path, deck: f90nml.Namelist) -> None:
    """Set the input file at `path`, read as `deck`, to restart."""
    changes = dict(RESTART_SETTINGS)
    if deck.get("fields_knobs", {}).get("dump_response"):
        changes["fields_knobs.read_response"] = True
    needed = {k: v for k, v in changes.items() if _template_value(deck, k) != v}
    if needed:
        partial = path.with_name(f"{path.name}.partial")
        f90nml.patch(str(path), _nested(needed), str(partial))
        partial.replace(path)


def _exit_reasons(run_dir: Path, name: str, job: Mapping[str, str] | None) -> dict[int, str] | None:
    """GS2's exit reasons for the run, by code, or None if it hasn't written
    any since `job` was submitted."""
    path = run_dir / f"{name}.exit_reason"
    if not path.is_file():
        return None
    try:
        submitted = datetime.fromisoformat(job["submitted"]) if job else None
    except ValueError:  # an edited jobs.csv
        submitted = None
    if submitted is not None:
        written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if written < submitted:
            return None
    reasons = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(-?\d+)\s+(.*\S)", line)
        if match:
            reasons[int(match.group(1))] = match.group(2)
    return reasons or None


def _open_scan(directory: Path, dry_run: bool) -> tuple[dict, dict[str, dict[str, str]]]:
    """The scan's manifest and jobs, once it's clear jobs can be submitted."""
    manifest = load(directory)
    if manifest["machine"] is None:
        raise ValueError(f"{directory} was written without a job script.")
    if not dry_run and shutil.which("sbatch") is None:
        raise FileNotFoundError("sbatch not found: submit from a Slurm login node.")
    return manifest, _read_jobs(directory)


def _names(manifest: Mapping) -> list[str]:
    return [run["name"] for run in manifest["runs"]]


def _submit_runs(
    directory: Path,
    names: Sequence[str],
    sbatch_args: Sequence[str],
    dry_run: bool,
    check_cores: bool,
    restarting: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Submit each of `names`, after the checks every submission gets.

    With `restarting`, the jobs that last ran them, each is set to restart
    first.
    """
    asked = any(name in _RESOURCE_OPTIONS for name, _ in _option_groups(sbatch_args))
    outcomes = {}
    short = {}  # (avail_cpu_time, time limit) -> runs it is too long for
    for name in names:
        run_dir = directory / name
        deck = f90nml.read(run_dir / f"{name}.in")
        args = list(sbatch_args)
        options = _sbatch_options(run_dir / JOB_SCRIPT, args)
        cores, per_node = _cores(options)
        before = restarting[name].get("cores") if restarting is not None else None

        # A restart goes on as many cores as before, unless told otherwise.
        if before and not asked and cores is not None:
            cores = int(before)
            args = _without_resources(args) + _resource_args(cores, per_node, options)[0]
        chosen = None
        if check_cores and cores is not None:
            choice = _choose_cores(deck, run_dir / f"{name}.report", cores)
            if choice is not None:
                cores = choice[0]
                extra, described = _resource_args(cores, per_node, options)
                args = _without_resources(args) + extra
                chosen = f" on {described}, as {choice[1]}"
        if restarting is not None:
            saved_many = deck.get("gs2_diagnostics_knobs", {}).get("save_many")
            if saved_many and before and cores is not None and int(before) != cores:
                outcomes[name] = (
                    f"cannot restart: its restart files are per process (save_many), "
                    f"from {before} cores, not {cores}"
                )
                continue
            if not dry_run:
                _set_restart(run_dir / f"{name}.in", deck)

        too_long = _too_long(deck, options)
        if too_long is not None:
            short.setdefault(too_long, []).append(name)
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={directory.resolve().name}-{name}",
            *args,
            JOB_SCRIPT,
            f"{name}.in",
        ]
        if dry_run:
            outcomes[name] = f"(cd {run_dir} && {shlex.join(command)})" + (chosen or "")
            continue
        result = subprocess.run(
            command, cwd=run_dir, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            outcomes[name] = f"sbatch failed: {result.stderr.strip()}"
            continue
        # --parsable prints "jobid" or "jobid;cluster".
        job = result.stdout.strip().split(";")[0]
        _record_job(directory, name, job, cores)
        outcomes[name] = f"job {job}" + (chosen or "")

    for (available, limit), affected in short.items():
        stated = "is not set" if available is None else f"is {available:g} s"
        print(
            f"Warning: avail_cpu_time {stated} but the job's time limit is "
            f"{limit} s, so Slurm will stop GS2 before it writes its restart "
            f"files: {', '.join(affected)}."
        )
    return outcomes


def _read_jobs(directory: Path) -> dict[str, dict[str, str]]:
    """The latest job submitted for each run: its ``job``, when it was
    ``submitted`` and on how many ``cores``."""
    path = directory / JOBS
    if not path.is_file():
        return {}
    with path.open(newline="") as f:
        return {row["run"]: row for row in csv.DictReader(f)}


def _record_job(directory: Path, name: str, job: str, cores: int | None) -> None:
    path = directory / JOBS
    new = not path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["run", "job", "submitted", "cores"])
        writer.writerow(
            [
                name,
                job,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "" if cores is None else cores,
            ]
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
    "restart",
    "status",
    "submit",
    "write",
    "zipped",
]
