# gs2ew

Hodge-podge of SLURM and post-processing scripts for the gyrokinetic code [GS2](https://bitbucket.org/gyrokinetics/gs2). These scripts are for my own personal use during my PhD and are in no way "official" or "recommeded" by the GS2 team.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

### Install `uv`

On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install `gs2ew`

Clone the repository and install:
```bash
git clone https://github.com/baiway/gs2ew.git
cd gs2ew
uv sync
```

This will create a virtual environment and install all dependencies. To activate the environment, run:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Command line

```bash
uv run gs2ew info run.out.nc                 # what diagnostics the file contains
uv run gs2ew plot run.out.nc -o outputs/     # the standard plot set
uv run gs2ew plot run.out.nc -o outputs/ --movies
uv run gs2ew batch project/ -o analysis/ --skip cbc
```

`batch` walks a directory for `*.out.nc`, mirrors each run's path into the
output root (so runs sharing an input-file name don't collide), and keeps going
if one run fails. It selects a headless matplotlib backend, so it works on a
compute node. See [`examples/analyse_batch.py`](examples/analyse_batch.py) for
the SLURM script.

### Python

Every plot takes an open dataset:

```python
import xarray as xr
from gs2ew import plot_transfer_theta, plot_transfer_trapped_passing

with xr.open_dataset("run.out.nc") as ds:
    plot_transfer_theta(ds, tstart="saturation", show_std=True, output_dir="outputs")
    plot_transfer_trapped_passing(ds, tstart="saturation", output_dir="outputs")
```

[`examples/analysis.py`](examples/analysis.py) is a template deck covering every
plot; copy it and comment out what you don't want.

## Transfer diagnostics

GS2's `diagnostics_free_energy_transfer` module writes the nonlinear drive of
the zonal modes for three quantities, each decomposed by the *mediator* field
that carries the three-wave coupling:

| Quantity | Symbol | netCDF stem |
| --- | --- | --- |
| Fluctuation energy | $N_\mathbf{k}^{U,f}$ | `fluctuation_energy_transfer_<field>_*` |
| Free energy | $N_\mathbf{k}^{H,f}$ | `free_energy_transfer_<field>_*` |
| Entropy | $N_\mathbf{k}^{S,f}$ | `entropy_transfer_<field>_*` |
| Kinetic energy | $T_v^\mathrm{ZF}$ | `kinetic_energy_transfer_*` |

`<field>` is one of `phi`, `apar`, `bpar` (labelled $\delta\phi$,
$\delta A_\parallel$, $\delta B_\parallel$). The kinetic-energy transfer is a
separate, purely electrostatic diagnostic with no field decomposition.

All three quantities are written directly; `U` is **not** derived here. They
satisfy $H = U - S$ in the stored sign convention, which `gs2ew info` checks.

With `write_fluctuation_energy_transfer_split`, GS2 also writes U's field-only
and FLR parts (`fluctuation_energy_field_*` and `fluctuation_energy_flr_*`,
quantity keys `U_field` and `U_flr`); see GS2's
`diagnostics_free_energy_transfer` for their definitions. `gs2ew info` checks
that they sum to U where they should (`theta` and `kxtarget` only).

Each quantity comes in five reductions:

| Reduction | Dimensions | Meaning |
| --- | --- | --- |
| `theta` | `(t, theta)` | summed over the whole triad |
| `kxtarget` | `(t, theta, kxt_shift)` | resolved by the target zonal mode |
| `kxsource` | `(t, theta, kxs_shift)` | resolved by the source radial wavenumber |
| `kysource` | `(t, theta, kys_ext)` | resolved by the source binormal wavenumber |
| `trapped_passing` | `(t, species, tp, theta, kxt_shift)` | split by species and by trapped (`tp=2`) / passing (`tp=1`) |

The `trapped_passing` output replaced the old fully velocity-resolved
`*_velocity` arrays, which GS2 no longer writes. GS2 applies the velocity-space
weights `w` and `wl` itself, so nothing further is needed on the analysis side.

### Which plot for what

| Function | Shows |
| --- | --- |
| `plot_transfer_theta` | $N_\mathbf{k}^{Q,f}(\theta)$, one column per field |
| `plot_transfer_spectral` | $(\theta, k)$ heatmaps for one reduction |
| `plot_transfer_spectrum` | $\int\mathrm{d}\theta\,J\,N_\mathbf{k}(k)$, 1D spectrum |
| `plot_transfer_trapped_passing` | $\theta$ profiles, trapped vs passing, per species |
| `plot_transfer_trapped_passing_spectral` | $(\theta, k_{xt})$ per population |
| `plot_transfer_hovmoller` | $(t, \theta)$ — the whole run in one figure |
| `plot_transfer_butterfly` | $(t, k)$ — which scales get driven, and when |
| `plot_transfer_summary` | net budget by field and by population |

Each of `plot_transfer_theta`, `plot_transfer_spectral`,
`plot_transfer_trapped_passing` and `plot_transfer_trapped_passing_spectral`
has a `_movie` counterpart. The Hovmöller and butterfly plots are the static
alternative: they show the same time evolution in a figure you can point at.

### Time windows

Every plot takes the same `tstart` / `window` pair:

| `tstart` | `window` | Result |
| --- | --- | --- |
| — | — | the last timestep |
| — | given | average over `[t[-1] - window, t[-1]]` |
| given | — | average over `[tstart, t[-1]]` |
| given | given | average over `[tstart, tstart + window]` |
| `"saturation"` | either | as above, with `tstart` detected |

`tstart="saturation"` starts a fraction `saturation_pad` (default `0.20`) into
the saturated interval — `tsat + 0.20 * (t[-1] - tsat)` — because the detected
saturation time itself is usually still too early to average from.

Movies default `tstart` to `"saturation"` rather than the last timestep.

### Targeted GS2 version

The `experimental/electromagnetic-entropy-transfer-field-flr-split` branch.
Its transfer diagnostics have not been tested yet.

## Development

```bash
uv run --extra test pytest              # everything
uv run --extra test pytest -m "not slow"  # skip the tests needing a real .out.nc
```

The `slow` tests read a real GS2 output file; point `GS2EW_TEST_NC` at one, or
they skip. Everything else runs against a synthetic dataset built in
`tests/conftest.py`.

### Layout

| Module | Responsibility |
| --- | --- |
| `gs2ew.catalog` | which transfer variables exist, what they're called, how they're labelled |
| `gs2ew.timebase` | time windows, saturation detection, movie frame lists |
| `gs2ew.geometry` | flux-surface-weighted $\theta$ integrals |
| `gs2ew.movie` | the frame loop and ffmpeg encoder |
| `gs2ew.io`, `.metadata`, `.weights` | reading files |
| `gs2ew.plots.*` | the plots themselves |
| `gs2ew.cli` | the `gs2ew` command |

Adding a field, quantity or reduction is a table entry in `catalog.py`; the
plots discover whatever is present in a given file.
