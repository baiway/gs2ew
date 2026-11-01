# Parameter scans

```bash
gs2ew scan write NL_cbc.in convergence/ \
    --vary ntheta=32,64,96 --vary nx=128,256 --vary negrid=8,24
gs2ew scan submit convergence/ --dry-run -- --time=12:00:00   # check first
gs2ew scan submit convergence/ -- --time=12:00:00
gs2ew scan status convergence/
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

- `read_response = .true.` or a restart (`ginit_option = 'many'`) in the
  template, which a fresh directory has nothing for.
- A scan over `beta` or a species gradient without
  `auto_set_beta_prime = .true.` in `theta_grid_parameters`, or with
  `beta_prime_input` (or `alpha_input`, ...) still set, which GS2 then uses
  instead. Let GS2 work out the pressure gradient rather than setting it.
- `avail_cpu_time` longer than the job's time limit, so Slurm would kill GS2
  before it writes its restart files.

Not checked: the transfer diagnostics switch themselves off unless the core
count divides `nlambda * negrid * nspec`, and `nlambda` follows `ntheta`. A
convergence scan over either can need a different core count per run.
