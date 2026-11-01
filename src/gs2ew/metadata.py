"""Utilities for reading and comparing GS2 build metadata from `.out.nc` files."""

import re
from dataclasses import dataclass, field

import numpy as np
import xarray as xr


@dataclass(order=True) # allows for e.g. `version < required` comparisons
class GS2Version:
    """Parsed representation of a GS2 `git describe` version string.

    The version string has the format `{tag}-{commits_since_tag}-g{short_hash}`
    (e.g. `8.2.1-1171-g003c87eb7`), or just `{tag}` when built exactly on
    a release tag.

    Instances are orderable: comparison is by `(major, minor, patch,
    commits_since_tag)`.
    
    Attributes
    ----------
    major, minor, patch : int
        Semver components of the most recent release tag.
    commits_since_tag : int
        Number of commits on top of the tag. Zero for exact tag builds.
    commit_hash : str
        Short git hash (without leading `g`). Empty for exact tag builds.
    branch : str
        Git branch the code was built from, if available from `build_config`.
    git_state : str
        `"clean"` or `"dirty"`, if available from `build_config`.
    """

    major: int
    minor: int
    patch: int
    commits_since_tag: int
    commit_hash: str = field(compare=False, default="")
    branch: str = field(compare=False, default="")
    git_state: str = field(compare=False, default="")

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.commits_since_tag:
            base += f"-{self.commits_since_tag}-g{self.commit_hash}"
        extras = []
        if self.branch:
            extras.append(self.branch)
        if self.git_state:
            extras.append(self.git_state)
        if extras:
            base += f" ({', '.join(extras)})"
        return base


def _parse_version_string(version_str: str) -> GS2Version:
    """Parse a `git describe`-style version string into a class `GS2Version`.

    Accepts both long form (`8.2.1-1171-g003c87eb7`) and exact-tag form
    (`8.2.1`).
    """
    parts = version_str.strip().split("-")
    if len(parts) == 3:
        tag, commits, hash_part = parts
        major, minor, patch = (int(x) for x in tag.split("."))
        return GS2Version(
            major=major,
            minor=minor,
            patch=patch,
            commits_since_tag=int(commits),
            commit_hash=hash_part.lstrip("g"),
        )
    elif len(parts) == 1:
        major, minor, patch = (int(x) for x in parts[0].split("."))
        return GS2Version(major=major, minor=minor, patch=patch, commits_since_tag=0)
    else:
        raise ValueError(f"Cannot parse GS2 version string: {version_str!r}")


def _extract_build_config_fields(ds: xr.Dataset) -> dict[str, str]:
    """Pull `GIT_BRANCH`, `GIT_STATE`, and the full `GIT_HASH` out of
    `build_config`."""
    if "build_config" not in ds:
        return {}

    raw = ds["build_config"].values.item()
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode()

    result = {}
    for pattern, key in [
        (r'GIT_BRANCH="([^"]+)"', "branch"),
        (r'GIT_STATE="([^"]+)"', "git_state"),
        (r'GIT_HASH="([^"]+)"', "git_hash_full"),
    ]:
        m = re.search(pattern, raw)
        if m:
            result[key] = m.group(1)
    return result


def get_version(ds: xr.Dataset) -> GS2Version:
    """Return the class `GS2Version` recorded in a GS2 dataset.

    Reads `ds.attrs['software_version']` and adds the branch and state
    information from `build_config` where available.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file).

    Raises
    ------
    KeyError
        If `software_version` is not present in the dataset attributes.
    """
    version_str = ds.attrs.get("software_version")
    if version_str is None:
        raise KeyError(
            "Dataset has no 'software_version' attribute. "
            "This may not be a GS2 output file, or it was produced by a very old version."
        )

    version = _parse_version_string(version_str)
    extra = _extract_build_config_fields(ds)
    version.branch = extra.get("branch", "")
    version.git_state = extra.get("git_state", "")
    return version


def get_metadata(ds: xr.Dataset) -> dict[str, str]:
    """Return a summary of the GS2 build metadata stored in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file).

    Returns
    -------
    dict
        Keys: `software_version`, `date_created`, `netcdf_version`,
        and any fields extractable from `build_config` (`branch`,
        `git_state`, `git_hash_full`).
    """
    meta: dict[str, str] = {
        "software_version": ds.attrs.get("software_version", "unknown"),
        "date_created": ds.attrs.get("date_created", "unknown"),
        "netcdf_version": ds.attrs.get("netcdf_version", "unknown"),
    }
    meta.update(_extract_build_config_fields(ds))
    return meta


def requires_version(ds: xr.Dataset, min_version: str) -> None:
    """Raise `RuntimeError` if the dataset was produced by a GS2 version
    older than `min_version`.

    Intended as a guard at the top of plot functions that rely on diagnostics
    added in a specific GS2 release::

        requires_version(ds, "8.2.1-1171-g003c87eb7")

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file).
    min_version : str
        Minimum acceptable version string in `git describe` format.

    Raises
    ------
    RuntimeError
        If the dataset version is older than `min_version`.
    """
    actual = get_version(ds)
    required = _parse_version_string(min_version)
    if actual < required:
        raise RuntimeError(
            f"This diagnostic requires GS2 >= {min_version}, "
            f"but this dataset was produced with {ds.attrs.get('software_version')}."
        )
