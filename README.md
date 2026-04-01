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

You can specify the `.out.nc` file, the plots to produce, and the output directory in
a Python file. A template is provided in [`examples/analysis.py`](examples/analysis.py).

Copy it, set `NC_FILE` and `OUTPUT_DIR` at the top, comment in or out the
plots you want, then run:

```bash
python analysis.py
```

or, without activating the virtual environment:

```bash
uv run analysis.py
```

Plots are saved to `OUTPUT_DIR` (default: `outputs/`). Movie functions also
save individual frames to a `<name>_frames/` subdirectory alongside the
`.mp4` file.
