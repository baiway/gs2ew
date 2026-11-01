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

# A spread of real outputs, old and new, for checking that provenance degrades
# gracefully when GS2 recorded less. Each skips if absent.
_ANALYSIS = Path("/Users/bailey/analysis")
REAL_FILES: dict[str, Path] = {
    # Newest layout, clean GS2 build, embedded input deck and run command.
    "nonlinear-8.2.1-2129": _ANALYSIS
    / "aug2026-nonlinear-beta-scan/scan/beta_0p0000/beta_0p0000.out.nc",
    # Small linear run from a GS2 build with uncommitted changes.
    "linear-8.2.1-2130-modified": _ANALYSIS
    / "05-linear-beta-scan/claude-assisted-analysis/refine_beta0p012/runs"
    / "b0p0120_ky0p100.out.nc",
    # Older transfer layout (velocity-resolved era).
    "nonlinear-8.2.1-1755": _ANALYSIS
    / "2026-06-18-EPS-scramble/2026-06-18-EPS-scramble"
    / "01-electrostatic-higher-res-test/NL_cbc_higher_res.out.nc",
    # GS2 8.2.0: an input deck but no build_config.
    "reduced-8.2.0": _ANALYSIS
    / "tobias-shaping-paper/cbc/tprim6p0"
    / "00_nx128_ny64_ntheta32_npassing15_negrid16_cfl0p5/baseline/reduced_output"
    / "reduced_gs2_output.out.nc",
    # No version metadata at all.
    "no-metadata": _ANALYSIS / "collab/16-cbc-long-time/cbc-unshaped.out.nc",
}

#: The input deck embedded in synthetic datasets.
SYNTHETIC_DECK = """&knobs
 delt = 0.05
 nstep = 1000
/

&theta_grid_parameters
 ntheta = 8
/
"""

#: GS2's commit and state as recorded in synthetic datasets.
SYNTHETIC_GS2_HASH = "5f2ef277b36dbda90f9386769027521ebd17edbc"


def char_array(lines: list[str], width: int = 200) -> np.ndarray:
    """Text as GS2 stores it: a (lines, width) array of space-padded bytes."""
    padded = b"".join(line.ljust(width)[:width].encode() for line in lines)
    return np.frombuffer(padded, dtype="S1").reshape(len(lines), width)


def make_dataset(
    fields: tuple[str, ...] = ("phi", "apar", "bpar"),
    reductions: tuple[str, ...] = ("theta", "kxsource", "kxtarget", "kysource"),
    trapped_passing: bool = True,
    kinetic: bool = True,
    split: bool = True,
    metadata: bool = True,
    gs2_state: str = "clean",
    potential: bool = False,
    weights: bool = False,
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

    With `split`, the field-only and FLR parts of U are added too. Like GS2's,
    they sum to U only in the `theta` and `kxtarget` reductions; in the
    source-resolved reductions they are independent of U. Only the FLR part
    has trapped/passing output.

    With `potential`, the time-resolved potential ``phi_t`` is included, and
    with `weights`, the velocity-space weights ``w`` and ``wl``.

    With `metadata`, the dataset carries the provenance GS2 records: version
    attributes, a ``build_config`` with git state `gs2_state`, a run UUID,
    and :data:`SYNTHETIC_DECK` as the embedded input deck.
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
        # GS2 stores kx in FFT order: zero, the positive modes, the negative.
        "kx": np.array([0.0, 0.1, 0.2, -0.2, -0.1]),
        "ky": np.array([0.0, 0.1, 0.2, 0.3]),
    }

    data: dict[str, xr.DataArray] = {
        # phi2 grows exponentially then flattens at t = 50, so saturation
        # detection has a knee to find.
        "phi2": ("t", np.exp(0.4 * np.minimum(t, 50.0))),
        "bmag": ("theta", 1.0 + 0.3 * np.cos(theta)),
        "jacob": ("theta", 1.0 + 0.2 * np.cos(theta)),
        # A circular surface of half-width 0.5 at R = 3, as GS2 writes it.
        "rplot": ("theta", 3.0 + 0.5 * np.cos(theta)),
        "zplot": ("theta", 0.5 * np.sin(theta)),
        "rprime": ("theta", np.cos(theta)),
        "zprime": ("theta", np.sin(theta)),
        "type_of_species": ("species", np.array([1, 2][:nspec])),
    }

    def spectral(dim: str) -> np.ndarray:
        return rng.normal(size=(nt, ntheta, nk))

    for field in fields:
        for k in ("kx", "ky"):
            data[f"{field}2_by_{k}"] = (
                ("t", k),
                rng.uniform(1.0, 2.0, size=(nt, len(coords[k]))),
                {"long_name": f"|{field}|^2 summed over the other k", "units": "GS2 units"},
            )

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
                data[f"{stem}_transfer_{field}_theta{suffix}"] = (
                    dims,
                    values,
                    {"units": "GS2 units (to be specified)"},
                )

            if split:
                u_field = rng.normal(size=u.shape)
                u_flr = (
                    u - u_field
                    if reduction in ("theta", "kxtarget")
                    else rng.normal(size=u.shape)
                )
                suffix, _ = catalog.REDUCTIONS[reduction]
                for stem, values in (
                    ("fluctuation_energy_field", u_field),
                    ("fluctuation_energy_flr", u_flr),
                ):
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
            if split:
                data[f"fluctuation_energy_flr_transfer_{field}_trapped_passing"] = (
                    dims,
                    rng.normal(size=shape),
                )

    if potential:
        shape = (nt, len(coords["ky"]), len(coords["kx"]), ntheta, 2)
        data["phi_t"] = (("t", "ky", "kx", "theta", "ri"), rng.normal(size=shape))

    if weights:
        negrid, nlambda = 5, 6
        coords["lambda"] = np.linspace(0.0, 1.0, nlambda)
        data["energy"] = (
            ("species", "egrid"),
            np.tile(np.linspace(0.1, 4.0, negrid), (nspec, 1)),
        )
        data["w"] = (("species", "egrid"), rng.uniform(size=(nspec, negrid)))
        data["wl"] = (("lambda", "theta"), rng.uniform(size=(nlambda, ntheta)))

    if kinetic:
        for reduction in reductions:
            suffix, dim = catalog.REDUCTIONS[reduction]
            dims = ("t", "theta") if dim is None else ("t", "theta", dim)
            data[f"kinetic_energy_transfer_theta{suffix}"] = (
                dims,
                rng.normal(size=tuple(len(coords[d]) for d in dims)),
            )

    attrs: dict[str, str] = {}
    if metadata:
        deck = SYNTHETIC_DECK.splitlines()
        data["input_file"] = (("nlines", "char200"), char_array(deck))
        data["used_input_file"] = (
            ("nused_lines", "char200"),
            char_array(deck + ["&layouts_knobs", " layout = 'lxyes'", "/"]),
        )
        data["build_config"] = (
            (),
            np.bytes_(
                f'GIT_HASH="{SYNTHETIC_GS2_HASH}"\n'
                'GIT_VERSION="8.2.1-2129-g5f2ef277b"\n'
                'GIT_BRANCH="fix/kinetic-energy-transfer-prefactor"\n'
                f'GIT_STATE="{gs2_state}"\n'
            ),
        )
        data["run_command"] = ((), np.bytes_("gs2 synthetic.in"))
        attrs = {
            "software_version": "8.2.1-2129-g5f2ef277b",
            "id": "00000000-0000-4000-8000-000000000000",
            "date_created": "2026-08-21T08:05:28.316Z+01:00",
        }

    return xr.Dataset(data, coords=coords, attrs=attrs)


@pytest.fixture
def ds() -> xr.Dataset:
    """A synthetic dataset with all three fields and every diagnostic."""
    return make_dataset()


@pytest.fixture
def ds_es() -> xr.Dataset:
    """A synthetic electrostatic dataset: phi only."""
    return make_dataset(fields=("phi",))


@pytest.fixture(params=list(REAL_FILES), ids=list(REAL_FILES))
def real_file(request) -> Path:
    """Each of :data:`REAL_FILES` in turn, or a skip if it is not available."""
    path = REAL_FILES[request.param]
    if not path.exists():
        pytest.skip(f"GS2 output not found at {path}")
    return path


@pytest.fixture
def real_ds():
    """A real GS2 output file, or a skip if it is not available."""
    if not REAL_NC.exists():
        pytest.skip(f"GS2 output not found at {REAL_NC}")
    with xr.open_dataset(REAL_NC) as dataset:
        yield dataset
