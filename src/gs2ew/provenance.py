"""Recording where a result came from.

Everything needed to say which code, which inputs and which data produced a
plot: the GS2 and gs2ew versions (each with its commit, branch and whether the
working tree had uncommitted changes), the input deck GS2 embedded in its
output, the identity of the output file itself, and the versions of the
numerical libraries in between.

Fields that cannot be determined are reported as ``"unknown"`` (or ``None`` for
the dirty flag) rather than raising: old GS2 output lacks most of this
metadata, and a missing record should never stop a plot being made.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import f90nml
import numpy as np
import xarray as xr

from gs2ew.metadata import _extract_build_config_fields

UNKNOWN = "unknown"

GS2EW_REPOSITORY = "https://github.com/baiway/gs2ew"
GS2_REPOSITORY = "https://bitbucket.org/gyrokinetics/gs2"

# Paths (relative to the gs2ew repository root) whose uncommitted changes can
# alter a result. Edits to the tests, examples or docs cannot, so they do not
# make a checkout count as dirty.
_GS2EW_SOURCE_PATHS = ("src", "pyproject.toml", "uv.lock")

# Libraries whose versions can change the numbers or the rendering.
_DEPENDENCIES = ("numpy", "xarray", "matplotlib", "netCDF4", "dask")


@dataclass(frozen=True)
class CodeVersion:
    """The version of one piece of code, as precisely as it can be pinned.

    `dirty` is True when the code was built or run with uncommitted changes,
    False when it was not, and None when that cannot be determined.
    """

    name: str
    repository: str
    version: str = UNKNOWN
    commit: str = UNKNOWN
    branch: str = UNKNOWN
    dirty: bool | None = None

    @property
    def commit_url(self) -> str:
        """Link to the commit in the repository's web interface, or ``""``."""
        if self.commit == UNKNOWN:
            return ""
        # Bitbucket and GitHub spell the path differently.
        segment = "commits" if "bitbucket.org" in self.repository else "commit"
        return f"{self.repository}/{segment}/{self.commit}"

    def attrs(self) -> dict[str, str]:
        """Flat ``<name>_<field>`` string attributes, for a netCDF header."""
        dirty = UNKNOWN if self.dirty is None else str(self.dirty).lower()
        return {
            f"{self.name}_version": self.version,
            f"{self.name}_commit": self.commit,
            f"{self.name}_branch": self.branch,
            f"{self.name}_dirty": dirty,
            f"{self.name}_repository": self.repository,
            f"{self.name}_commit_url": self.commit_url,
        }


def gs2_version(ds: xr.Dataset) -> CodeVersion:
    """The GS2 build that wrote `ds`.

    Taken from the ``software_version`` attribute and the ``build_config``
    record GS2 writes into every output file. ``GIT_STATE`` is ``"clean"``
    for a build from a clean working tree and ``"modified"`` otherwise.
    """
    build = _extract_build_config_fields(ds)
    state = build.get("git_state")
    return CodeVersion(
        name="gs2",
        repository=GS2_REPOSITORY,
        version=str(ds.attrs.get("software_version", UNKNOWN)),
        commit=build.get("git_hash_full", UNKNOWN),
        branch=build.get("branch", UNKNOWN),
        dirty=None if state is None else state != "clean",
    )


def gs2ew_version() -> CodeVersion:
    """The gs2ew installation doing the analysis.

    A pip or uv install from git records the exact commit it was built from
    (PEP 610 ``direct_url.json``), and such an install cannot have local
    edits. An editable install, or a source checkout on ``sys.path``, is
    asked directly with ``git``; only uncommitted changes under ``src/``,
    ``pyproject.toml`` and ``uv.lock`` count as dirty.
    """
    try:
        version = importlib_metadata.version("gs2ew")
    except importlib_metadata.PackageNotFoundError:
        version = UNKNOWN

    direct_url = _read_direct_url()
    if direct_url is not None and "vcs_info" in direct_url:
        return _from_vcs_install(direct_url, version)

    root = _editable_root(direct_url)
    if root is None and direct_url is None:
        # Not installed at all (e.g. `src/` on PYTHONPATH): try the checkout
        # this file lives in.
        root = Path(__file__).resolve().parents[2]

    if root is not None and (root / "src" / "gs2ew").is_dir():
        state = _git_state(root)
        if state is not None:
            commit, branch, dirty = state
            return CodeVersion(
                name="gs2ew",
                repository=GS2EW_REPOSITORY,
                version=version,
                commit=commit,
                branch=branch,
                dirty=dirty,
            )

    return CodeVersion(name="gs2ew", repository=GS2EW_REPOSITORY, version=version)


def _read_direct_url() -> dict | None:
    """The installed distribution's PEP 610 ``direct_url.json``, if any."""
    try:
        text = importlib_metadata.distribution("gs2ew").read_text("direct_url.json")
    except importlib_metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _from_vcs_install(direct_url: dict, version: str) -> CodeVersion:
    """A :class:`CodeVersion` for a pip/uv install straight from git."""
    vcs = direct_url["vcs_info"]
    return CodeVersion(
        name="gs2ew",
        repository=GS2EW_REPOSITORY,
        version=version,
        commit=vcs.get("commit_id", UNKNOWN),
        branch=vcs.get("requested_revision", UNKNOWN),
        # Built from a fetched commit, so there is nothing uncommitted.
        dirty=False,
    )


def _editable_root(direct_url: dict | None) -> Path | None:
    """The source directory of an editable install, or None.

    A non-editable install from a local directory is a snapshot taken at
    install time; that directory's current git state says nothing reliable
    about it, so it gets None too.
    """
    if direct_url is None:
        return None
    if not direct_url.get("dir_info", {}).get("editable", False):
        return None
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


def _git_state(root: Path) -> tuple[str, str, bool] | None:
    """(commit, branch, dirty) for the git checkout at `root`, or None."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        status = git("status", "--porcelain", "--", *_GS2EW_SOURCE_PATHS)
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit, branch, bool(status)


def dependency_versions() -> dict[str, str]:
    """Versions of Python and the libraries that shape the numbers and plots."""
    versions = {"python": platform.python_version()}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = UNKNOWN
    return versions


def _decode_text(da: xr.DataArray) -> list[str]:
    """Lines of text from a netCDF character variable.

    GS2 stores text as a ``(lines, chars)`` array of single bytes padded with
    spaces, or as a scalar byte string. Trailing padding is stripped.
    """
    values = np.asarray(da.values)
    if values.ndim == 0:
        raw = values.item()
        text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
        return text.splitlines()

    if values.dtype.kind == "S" and values.dtype.itemsize == 1 and values.ndim == 2:
        rows = (b"".join(row.tolist()) for row in values)
        return [row.decode(errors="replace").rstrip() for row in rows]

    return [
        (v.decode(errors="replace") if isinstance(v, bytes) else str(v)).rstrip()
        for v in values.ravel()
    ]


def input_deck(ds: xr.Dataset, used: bool = False) -> str | None:
    """The input deck GS2 embedded in `ds`, or None if it has none.

    This is the deck the run actually read, so it is the authoritative record
    even if the ``.in`` file next to the output has since been edited. With
    `used`, return ``used_input_file`` instead: every namelist with its
    defaults filled in.
    """
    name = "used_input_file" if used else "input_file"
    if name not in ds:
        return None
    lines = _decode_text(ds[name])
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


@functools.lru_cache(maxsize=4096)
def _parse_deck(text: str) -> f90nml.Namelist:
    return f90nml.reads(text)


def input_decks(ds: xr.Dataset) -> list[f90nml.Namelist]:
    """The input decks GS2 embedded in `ds`, parsed: as given, then with
    every default filled in, as far as `ds` has them.

    Parsed once for each deck and shared, so not to be changed.
    """
    found = []
    for used in (False, True):
        text = input_deck(ds, used=used)
        if text is not None:
            found.append(_parse_deck(text))
    return found


def _normalise_deck(text: str) -> list[str]:
    # GS2 stores each line in 200 characters, padded with spaces, and turns
    # each tab into a single space.
    lines = [line.replace("\t", " ")[:200].rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def deck_matches(embedded: str, path: str | Path) -> bool:
    """Whether the input file at `path` is the deck embedded in the output.

    Allows for how GS2 stores the deck: lines truncated to 200 characters,
    tabs turned into single spaces, and trailing whitespace and blank lines
    dropped.
    """
    return _normalise_deck(embedded) == _normalise_deck(Path(path).read_text())


#: Input settings that only say how a run starts, which a restart changes in
#: the input file in place (see :func:`gs2ew.scan.restart`). GS2 keeps the
#: first deck in an output it appends to, so these are the only differences
#: a restarted run's input file should show.
RESTART_KEYS = frozenset(
    {
        "init_g_knobs.ginit_option",
        "gs2_diagnostics_knobs.append_old",
        "knobs.delt_option",
        "fields_knobs.read_response",
    }
)


def deck_restarted(embedded: str, path: str | Path) -> bool:
    """Whether the input file at `path` is the deck embedded in the output
    with only its :data:`RESTART_KEYS` changed: the run has been restarted."""
    try:
        first = f90nml.reads("\n".join(_normalise_deck(embedded)))
        now = f90nml.read(str(path))
    except (ValueError, IndexError, StopIteration):  # not a namelist f90nml can read
        return False

    def settings(deck: f90nml.Namelist) -> dict[str, object]:
        return {
            f"{group}.{key}": value
            for group in deck
            for key, value in deck[group].items()
            if f"{group}.{key}" not in RESTART_KEYS
        }

    return settings(first) == settings(now)


def source_identity(ds: xr.Dataset) -> dict[str, str]:
    """What identifies the GS2 output `ds` was read from.

    GS2 stamps every output file with a UUID (the ``id`` attribute), which is
    a far cheaper fingerprint than hashing a file of hundreds of gigabytes.
    The path and size are recorded too, as a human-readable pointer.
    """
    source = ds.encoding.get("source", "")
    path = str(Path(source).resolve()) if source else UNKNOWN
    try:
        size = str(os.stat(source).st_size) if source else UNKNOWN
    except OSError:
        size = UNKNOWN

    run_command = UNKNOWN
    if "run_command" in ds:
        run_command = " ".join(_decode_text(ds["run_command"])) or UNKNOWN

    return {
        "source_path": path,
        "source_size_bytes": size,
        "source_id": str(ds.attrs.get("id", UNKNOWN)),
        "source_date_created": str(ds.attrs.get("date_created", UNKNOWN)),
        "source_run_command": run_command,
    }


__all__ = [
    "UNKNOWN",
    "GS2EW_REPOSITORY",
    "GS2_REPOSITORY",
    "CodeVersion",
    "gs2_version",
    "gs2ew_version",
    "dependency_versions",
    "input_deck",
    "input_decks",
    "deck_matches",
    "RESTART_KEYS",
    "deck_restarted",
    "source_identity",
]
