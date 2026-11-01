<h1 align="center">
  <picture>
    <source srcset="docs/images/gs2ew-logo-dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/images/gs2ew-logo-light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/images/gs2ew-logo-light.svg" alt="gs2ew logo: a steaming bowl of stew" width="360">
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

```bash
uv run gs2ew info RUN     # what diagnostics the file contains
uv run gs2ew plot RUN     # the standard plot set, in RUN/gs2ew/
```

`RUN` is a run directory holding one `*.out.nc`, or the `*.out.nc` itself. For
plots tailored to a run, copy the template deck in
[`examples/analysis.py`](examples/analysis.py) and comment in or out the plots
you want. Each figure is saved as a PDF and a PNG at the size it will be
printed, with the data it was drawn from and the versions of GS2 and gs2ew that
made it, so it can be restyled, redrawn or reproduced later.

## Documentation

- [Usage](docs/usage.md): the command line, the Python API and time windows
- [Figures](docs/figures.md): formats, the print style, and restyling a figure
  for a thesis or journal
- [Plot data and provenance](docs/reproducibility.md): what is saved with each
  figure, and `replot` / `reproduce`
- [Transfer diagnostics](docs/transfer-diagnostics.md): what GS2 writes, and
  which plot shows what
- [Parameter scans](docs/scans.md): writing, submitting and following them
- [Development](docs/development.md): tests and code layout

[`extras/`](extras/) holds the environment files I source before compiling
GS2 on each machine (`gs2-<system>.sh`). The Slurm templates for running it
on ARCHER2 and Viking, which `gs2ew scan` uses, are in
[`src/gs2ew/templates/`](src/gs2ew/templates/).
