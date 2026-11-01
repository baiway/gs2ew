# Parameter scans

```bash
gs2ew scan write NL_cbc.in convergence/ \
    --vary ntheta=32,64,96 --vary nx=128,256 --vary negrid=8,24
gs2ew scan submit convergence/ --dry-run -- --time=12:00:00   # check first
gs2ew scan submit convergence/ -- --time=12:00:00
gs2ew scan status convergence/
gs2ew scan restart convergence/                              # runs that hit their time limit
```

`write` gives each point its own directory, holding the template with only
the scanned values changed (comments and all), named after the directory, and
a copy of a Slurm template from `src/gs2ew/templates/` (`--machine`, default
`archer2`). `scan.json` records what was written, from what, and by which
gs2ew. The job scripts find their run from the directory they are submitted
in, so a scan can be written locally and copied to the cluster.

`--mode` decides how the `--vary` parameters combine:

| Mode | Points |
| --- | --- |
| `one-at-a-time` (default) | the template (`base/`), then each value on its own |
| `product` | every combination |
| `zip` | the i-th values together, for values paired by hand |

One-at-a-time grows with the number of values rather than their product, so
a large scan has to be asked for. A value equal to the template's is run once,
as `base/`.

A key is the name in the input file. Give its namelist too
(`species_parameters_2.tprim`) when the bare name is in several namelists, or
in none because the template leaves it at its default.

`submit` runs `sbatch` in every run directory without output or a job yet,
records job IDs in `jobs.csv`, and passes anything after `--` to `sbatch`.
`status` reports each run's exit reason, or its job's state.

## Core counts

The transfer diagnostics need every process to hold whole $(k_x, k_y)$
planes. GS2 hands each process an equal, contiguous share of its layout, so
for `layout = 'xyles'` the core count must divide `nlambda * negrid * nspec`.
`nlambda` counts the pitch angles: the passing ones (`npassing`, or
`2 * ngauss`) and one trapped one per bounce point, of which there are up to
`ntheta / 2 + 1`. So a scan over `ntheta` changes the core counts that work:
for the CBC with `npassing = 15`, ntheta 48, 64 and 96 need core counts
dividing 1280, 1536 and 2048.

A nonlinear run should also be on a sweetspot: a count that splits GS2's
layouts evenly, dimension by dimension, which keeps the transforms of the
nonlinear term cheap. `ingen` lists them in its report; without one, gs2ew
works them out the same way. For nlambda = 48 that is 1, 2, 4, ..., 32, then
32 times each divisor of 48: so 96, but not 48, though 48 divides 1536.

The core count asked for, by the job script or with `-- --nodes=N`, is the
most a run gets: `submit` gives each run the largest count up to it that
keeps the planes whole and is a sweetspot, on the fewest nodes that share it
evenly, and says when that's fewer. A restart goes on as many cores as
before. `--no-core-check` uses the count asked for regardless.

GS2's own check (in `gs2_layouts.fpp`) looks to let some split counts
through, such as 640 for nlambda = 48, keeping the diagnostics on when the
planes are split; not yet confirmed by a run.

`write` runs GS2's `ingen` on each input (`--ingen`, or `$GS2_INGEN`, or
`ingen` on `PATH`) for the exact `nlambda` in its `.report`. Without it,
`nlambda` is estimated by the upper bound, which strong shaping can bring
down (κ = 2 has 44 rather than 48 for ntheta = 64).

## Restarts

`restart` restarts the runs GS2 stopped at `avail_cpu_time` (or those given
with `--runs`, whatever stopped them). It sets each one's input file, in place,
to restart:

| Setting | So that GS2 |
| --- | --- |
| `ginit_option = 'many'` | starts from the restart files |
| `append_old = .true.` | adds to the existing `.out.nc` |
| `delt_option = 'check_restart'` | carries on with the timestep it had reached |
| `read_response = .true.` | reuses the response matrices, if it saved them |

and submits it as `submit` would. The run needs `save_for_restart = .true.`
and its restart files in `restart_dir`; with `save_many`, they are one file
per process, so the run must go on the same core count. Each restart runs
another `nstep` steps, so set `max_sim_time` to stop at a given time.

GS2 keeps the first input deck in an output it appends to, so gs2ew doesn't
count these settings as edits to the input file.

## Comparing runs

```bash
gs2ew compare convergence/ --vary ntheta
```

draws, in `convergence/gs2ew/`, with a line per run:

| Figure | For |
| --- | --- |
| `compare_ntheta_traces` | each field against time, and where each run's window starts: did it saturate? |
| `compare_ntheta_spectra` | each field's kx and ky spectra: do the fields decay at large k? |
| `compare_ntheta_scalars` | $\langle|\delta\phi|^2\rangle$, heat flux and net transfers against `ntheta` |
| `compare_ntheta_theta` | each drive's $N_\mathbf{k}(\theta)$ |

Each run is averaged over its own saturated window, and runs on different
grids share the union of them. The value of `ntheta` comes from each run's own input deck,
so `compare` works on any runs, not only a scan's.

Given a directory, it compares the largest set of runs whose decks differ only
in `--vary`: in a one-at-a-time scan, `base/` and the runs over that
parameter. `--ignore tripri` lets another parameter differ too, for one paired
with the varied one. Their bundles record every run, and `reproduce` checks
each one's UUID.

## From Python

Points are lists of dicts, so scans combine:

```python
from gs2ew import scan

shaping = scan.zipped({"tri": [0.2, 0.4], "tripri": [0.21, 0.44]})
points = scan.combine(shaping, scan.product({"beta": [0.0, 0.01]}))
scan.write("NL_cbc.in", points, "shaping/", machine="viking")
```

`derive(point, deck)` sets values that follow the scanned ones, given the
template with the point applied; these are recorded as derived.

## Checks

`write` and `submit` warn about mistakes that waste a run:

- A restart (`ginit_option = 'many'`) in the template, which a fresh
  directory has nothing for. (`read_response` is fine: GS2 calculates the
  response matrices if it finds none.)
- A scan over `beta` or a species gradient without
  `auto_set_beta_prime = .true.` in `theta_grid_parameters`, or with
  `beta_prime_input` (or `alpha_input`, ...) still set, which GS2 then uses
  instead. Let GS2 work out the pressure gradient rather than setting it.
- `avail_cpu_time` longer than the job's time limit, so Slurm would kill GS2
  before it writes its restart files.
