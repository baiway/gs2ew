"""Shared fixtures.

Most tests run against a small synthetic dataset built here rather than a real
GS2 file, so they are fast and exercise field combinations (all three fields,
or none) that no single output file has. The tests marked `slow` check the
real thing; they skip when the file is not present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gs2ew import catalog

# Real GS2 output used by the `slow` tests. Override with GS2EW_TEST_NC.
REAL_NC = Path(
    os.environ.get(
        "GS2EW_TEST_NC",
        "/Users/bailey/analysis/aug2026-nonlinear-beta-scan/scan/beta_0p0140"
        "/beta_0p0140.out.nc",
    )
)


def make_dataset(
    fields: tuple[str, ...] = ("phi", "apar", "bpar"),
    reductions: tuple[str, ...] = ("theta", "kxsource", "kxtarget", "kysource"),
    trapped_passing: bool = True,
    kinetic: bool = True,
    ke_variant: str | None = None,
    nt: int = 24,
    ntheta: int = 9,
    nk: int = 7,
    nspec: int = 2,
    seed: int = 0,
) -> xr.Dataset:
    """Build a synthetic dataset shaped like GS2's transfer output.

    The three quantities are constructed so ``H = U - S`` holds exactly, as it
    does in real output, and the wavevector reductions sum to the plain
    `_theta` variable.
    """
    rng = np.random.default_rng(seed)

    # Long enough to span detect_saturation_time's default 20-unit window.
    t = np.linspace(0.0, 100.0, nt)
    theta = np.linspace(-np.pi, np.pi, ntheta)
    coords = {
        "t": t,
        "theta": theta,
        "kxs_shift": np.linspace(-1.0, 1.0, nk),
        "kxt_shift": np.linspace(-1.0, 1.0, nk),
        "kys_ext": np.linspace(-1.0, 1.0, nk),
        "tp": np.array([1, 2]),
        "species": np.arange(nspec, dtype=float),
    }

    data: dict[str, xr.DataArray] = {
        # phi2 grows exponentially then flattens at t = 50, so saturation
        # detection has a knee to find.
        "phi2": ("t", np.exp(0.4 * np.minimum(t, 50.0))),
        "bmag": ("theta", 1.0 + 0.3 * np.cos(theta)),
        "jacob": ("theta", 1.0 + 0.2 * np.cos(theta)),
        "type_of_species": ("species", np.array([1, 2][:nspec])),
    }

    def spectral(dim: str) -> np.ndarray:
        return rng.normal(size=(nt, ntheta, nk))

    for field in fields:
        for reduction in reductions:
            _, dim = catalog.REDUCTIONS[reduction]
            dims = ("t", "theta") if dim is None else ("t", "theta", dim)
            u = rng.normal(size=tuple(len(coords[d]) for d in dims))
            s = rng.normal(size=u.shape)
            for stem, values in (
                ("fluctuation_energy", u),
                ("entropy", s),
                # H = U - S is the identity GS2's stored sign convention gives.
                ("free_energy", u - s),
            ):
                suffix, _ = catalog.REDUCTIONS[reduction]
                data[f"{stem}_transfer_{field}_theta{suffix}"] = (dims, values)

        if trapped_passing:
            dims = ("t", "species", "tp", "theta", "kxt_shift")
            shape = tuple(len(coords[d]) for d in dims)
            u = rng.normal(size=shape)
            s = rng.normal(size=shape)
            for stem, values in (
                ("fluctuation_energy", u),
                ("entropy", s),
                ("free_energy", u - s),
            ):
                data[f"{stem}_transfer_{field}_trapped_passing"] = (dims, values)

    if kinetic:
        tag = f"_{ke_variant}" if ke_variant else ""
        for reduction in reductions:
            suffix, dim = catalog.REDUCTIONS[reduction]
            dims = ("t", "theta") if dim is None else ("t", "theta", dim)
            data[f"kinetic_energy_transfer{tag}_theta{suffix}"] = (
                dims,
                rng.normal(size=tuple(len(coords[d]) for d in dims)),
            )

    return xr.Dataset(data, coords=coords)


@pytest.fixture
def ds() -> xr.Dataset:
    """A synthetic dataset with all three fields and every diagnostic."""
    return make_dataset()


@pytest.fixture
def ds_es() -> xr.Dataset:
    """A synthetic electrostatic dataset: phi only."""
    return make_dataset(fields=("phi",))


@pytest.fixture
def real_ds():
    """A real GS2 output file, or a skip if it is not available."""
    if not REAL_NC.exists():
        pytest.skip(f"GS2 output not found at {REAL_NC}")
    with xr.open_dataset(REAL_NC) as dataset:
        yield dataset
