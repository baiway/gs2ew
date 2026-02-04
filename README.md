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

## To-do
- [x] Implement basic check for saturation. See [`detect_saturation_time`](https://github.com/baiway/gs2ew/blob/main/src/gs2ew/utils/gs2_output.py)
- [x] Plot time traces for each enabled field. See [`plot_fields_time_traces`](https://github.com/baiway/gs2ew/blob/main/src/gs2ew/postprocess/fields.py)
- [x] Plot $k_x$-$k_y$ spectra of each enabled field for last time step. See [`plot_fields_by_mode`](https://github.com/baiway/gs2ew/blob/main/src/gs2ew/postprocess/fields.py)
- [x] Plot poloidal structure of each enabled transfer diagnostic
- [ ] Plot time traces of the transfer for each target $k_x$
