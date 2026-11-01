<h1 align="center">
  <picture>
    <source srcset="docs/images/gs2ew-logo-dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/images/gs2ew-logo-light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/images/gs2ew-logo-light.svg" alt="gs2ew logo: a steaming bowl of stew" width="600">
  </picture>
</h1>

Hodge-podge of SLURM and post-processing scripts for the gyrokinetic code
[GS2](https://bitbucket.org/gyrokinetics/gs2). These scripts are for my own personal
use during my PhD and are in no way "official" or "recommended" by the GS2 team.

## Quick start

You need [`uv`](https://docs.astral.sh/uv/), which installs Python 3.10 or newer
if you don't already have it.

```bash
git clone https://github.com/baiway/gs2ew.git
cd gs2ew
uv sync
```

## Usage

A template analysis deck is provided in
[`examples/analysis.py`](examples/analysis.py). It takes the input `.out.nc`
file and the output directory as command-line arguments. Copy it, comment in
or out the plots you want, then run:

```bash
uv run examples/analysis.py results.out.nc outputs/
```

Plots are saved to the output directory you pass. Movie functions also save
individual frames to a `<name>_frames/` subdirectory alongside the `.mp4`
file.
