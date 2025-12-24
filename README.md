# gs2ew
Hodge-podge of SLURM and post-processing scripts for the gyrokinetic code [GS2](https://bitbucket.org/gyrokinetics/gs2).

## Installation
This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

### Install `uv`
On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, install via pip:
```bash
pip install uv
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

