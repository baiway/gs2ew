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

A template analysis deck is provided in
[`examples/analysis.py`](examples/analysis.py). It takes the input `.out.nc`
file and the output directory as command-line arguments. Copy it, comment in
or out the plots you want, then run:

```bash
python examples/analysis.py results.out.nc outputs/
```

or, without activating the virtual environment:

```bash
uv run examples/analysis.py results.out.nc outputs/
```

Plots are saved to the output directory you pass. Movie functions also save
individual frames to a `<name>_frames/` subdirectory alongside the `.mp4`
file.

## Transfer diagnostics

GS2 writes three related nonlinear-transfer quantities, plotted by the
functions in [`gs2ew.postprocess.transfer`](src/gs2ew/postprocess/transfer.py):

| Quantity | Symbol | netCDF variable |
| --- | --- | --- |
| Free energy | $N_\mathbf{k}^{H,f}$ | `free_energy_transfer_<field>_{theta,velocity}` |
| Entropy | $N_\mathbf{k}^{S,f}$ | `entropy_transfer_<field>_{theta,velocity}` |
| Kinetic energy | $T_v^\text{ZF}$ | `kinetic_energy_transfer_theta` |

Here `<field>` is one of `phi`, `apar`, `bpar`. The fluctuation energy
$N_\mathbf{k}^{U,f}$ is not written to file; it is derived as `U = H + S`.

> **Note on naming:** what older GS2 versions called the "entropy" transfer is
> now the **free energy** transfer (`free_energy_transfer_*`). The
> `entropy_transfer_*` variables are a genuinely separate, newer output.

By default the transfer plots multiply the free-energy, entropy and U drives
by `-1` (`fix_transfer_sign=True`) to correct a sign error in the GS2
implementation; the kinetic-energy transfer is left untouched.
