# Usage

## Command line

```bash
uv run gs2ew info RUN                        # what diagnostics the file contains
uv run gs2ew plot RUN                        # the standard plot set, in RUN/gs2ew/
uv run gs2ew plot RUN --only transfer_theta transfer_summary
uv run gs2ew plot RUN -o outputs/ --movies
uv run gs2ew batch project/ --skip cbc       # every run under project/
uv run gs2ew compare scan/ --vary ntheta     # runs that differ in one parameter
```

`RUN` is a run directory holding one `*.out.nc`, or the `*.out.nc` itself.
Plots go in `gs2ew/` inside the run directory unless `-o` is given.

`batch` walks a directory for `*.out.nc` and keeps going if one run fails. With
`-o`, it mirrors each run's path into the output root, so runs sharing an
input-file name don't collide. It selects a headless matplotlib backend, so it
works on a compute node. See [`examples/analyse_batch.py`](../examples/analyse_batch.py)
for the SLURM script.

## Python

Every plot takes an open dataset:

```python
import xarray as xr
from gs2ew import plot_transfer_theta, plot_transfer_trapped_passing

with xr.open_dataset("run.out.nc") as ds:
    plot_transfer_theta(ds, tstart="saturation", show_std=True, output_dir="outputs")
    plot_transfer_trapped_passing(ds, tstart="saturation", output_dir="outputs")
```

[`examples/analysis.py`](../examples/analysis.py) is a template deck covering every
plot; copy it and comment out what you don't want.

Each `plot_*` is a `compute_*` step, which reduces the output to the data the
figure shows, followed by a `render_*` step that draws it. Call them yourself to
work with the reduced data directly.

## Time windows

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

See [Transfer diagnostics](transfer-diagnostics.md) for which plot shows what,
[Figures](figures.md) for formats and styles, and
[Plot data and provenance](reproducibility.md) for what is saved alongside
each figure.
