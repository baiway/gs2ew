# Plot data and provenance

Next to each figure, gs2ew writes a netCDF *bundle* with the same name, holding:

- the data the figure was drawn from, with GS2's units and how it was reduced
- every option used to make it, including its style and formats
- the GS2 and gs2ew versions, commits, branches, and whether either had
  uncommitted changes
- the GS2 output's path and run UUID, and the input deck the run read

Each output directory also gets the run's input deck and a `reproduce.sh`.

```bash
gs2ew replot RUN/gs2ew/transfer_theta_averaged.nc     # redraw, no raw output needed
gs2ew replot RUN/gs2ew/transfer_theta_averaged.nc --set width=3.4    # restyled
gs2ew reproduce RUN/gs2ew/*.nc                        # redo the analysis and compare
```

`reproduce` checks the run UUID first, writes to `reproduced/`, and reports
whether the data match. `reproduce.sh` pins each command to gs2ew's exact commit
with `uvx`. Pass `--require-clean` to `plot` to refuse to run from uncommitted
gs2ew or GS2 code, and `--no-data` to skip the bundles. Movies save no data.
