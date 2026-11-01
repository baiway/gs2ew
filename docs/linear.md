# Linear scans

A growth-rate spectrum for each point of a parameter scan:

```bash
gs2ew linear write NL_cbc.in lin/ --vary beta=0,0.005,0.01 --ky 0.05:2:20 --set nperiod=4
gs2ew linear submit lin/ --jobs 4 --dry-run -- --qos=short --time=00:20:00 --nodes=4
gs2ew linear submit lin/ --jobs 4 -- --qos=short --time=00:20:00 --nodes=4
gs2ew linear status lin/
gs2ew linear plot lin/
```

`write` runs each point at each ky as its own GS2 run, with
`grid_option = 'single'`. A run stops as soon as its frequency has converged,
so most take minutes, and a spectrum is refined by adding runs. The template
can be a nonlinear input file. `write` turns off the nonlinear term, the
transfer diagnostics, restart files and saved response matrices, and turns on
what a spectrum needs (`write_omega`, `write_omavg`, `write_fields`). The
changes are recorded in `scan.json`. `--vary` and `--mode` work as for
[`scan write`](scans.md), and `--set KEY=VALUE` sets a value in every run.

`--ky` takes a list, or `MIN:MAX:N` for N values evenly spaced in log(ky).
`--box-ky` adds the template box's own ky, so the spectrum lines up with
the nonlinear run it is for. `--theta0` sets the ballooning angle (default 0).

`submit` deals the runs in order of ky to `--jobs` jobs, so each job gets a
mix of slow low-ky runs and quick high-ky ones. Each job runs its list through
GS2's `multigs2`, which splits the job's cores into one batch per
`--cores-per-run` (default 4) and runs a share of the list on each, one after
another. With the job's time limit known, each run's `avail_cpu_time` is set
to its share of it, so a slow run stops itself rather than the job being
killed. The job script looks for `multigs2` beside `$GS2_EXEC`, or at
`$MULTIGS2_EXEC`. On ARCHER2, `short` is a QoS of the `standard` partition:
20 minutes, up to 32 nodes, and 4 jobs running at once.

## Convergence

GS2 averages the frequency over `navg` steps and stops when the frequency
settles to within `omegatol`. Two modes with similar growth rates beat with
periods of tens of a/v_th, and a short window can pass the test with a growth
rate of the wrong sign. So `write` sets `navg` to cover `--omega-window`,
default 25 a/v_th. Leave `max_sim_time` at least four times that.

GS2's `.exit_reason` says "nstep timesteps completed" for a converged run, so
`status` and `plot` count a run as converged if it stopped before
`max_sim_time` and `nstep` and gives no other reason. A run that didn't
converge is drawn with an open marker and left out of γ_max.

`nperiod` is the same for every ky. A mode extends along the field line well
beyond the 1/(ky ŝ) estimate: at high ky through the electrons, and at finite
beta through A∥. Check it with a scan over `nperiod` itself:

```bash
gs2ew linear write NL_cbc.in nperiod/ --vary nperiod=2,4,8 --mode product
gs2ew linear plot nperiod/
```

## Plots

`plot` takes a scan directory, or any runs, and draws a line for each value of
the parameter the scan varies, or of `--vary`:

| Figure | Shows |
| --- | --- |
| `linear_<key>_spectrum` | γ and ω against ky |
| `linear_<key>_gamma_max` | γ_max, its ky and its ω against the parameter |
| `linear_<key>_modes` | φ, A∥ and B∥ along the field line at each line's fastest-growing ky (or `--ky`) |

The eigenfunctions are turned so that φ is real at its peak, so parities can
be read off: φ even and A∥ odd for ITG and KBM, and the reverse for tearing
modes. It also prints each line's γ_max, and says where a peak is at the end
of the ky range or γ hasn't fallen by the largest ky. Runs that differ in
more than ky and the parameter are left out, as `compare` does; a
`grid_option = 'range'` run, with several ky, will do too.
