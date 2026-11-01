# Development

```bash
uv run --extra test pytest              # everything
uv run --extra test pytest -m "not slow"  # skip the tests needing a real .out.nc
```

The `slow` tests read a real GS2 output file; point `GS2EW_TEST_NC` at one, or
they skip. Everything else runs against a synthetic dataset built in
`tests/conftest.py`.

## Layout

| Module | Responsibility |
| --- | --- |
| `gs2ew.catalog` | which transfer variables exist, what they're called, how they're labelled |
| `gs2ew.timebase` | time windows, saturation detection, movie frame lists |
| `gs2ew.geometry` | flux-surface-weighted $\theta$ integrals |
| `gs2ew.movie` | the frame loop and ffmpeg encoder |
| `gs2ew.io`, `.metadata`, `.weights` | reading files |
| `gs2ew.provenance` | code versions, input decks and run identity |
| `gs2ew.bundle` | the data and provenance saved with each plot, `replot`, `reproduce` |
| `gs2ew.style`, `styles/` | how figures look and the formats they're saved in |
| `gs2ew.plots.*` | the plots themselves; `plots.compare` compares runs |
| `gs2ew.scan`, `templates/` | writing, submitting and following parameter scans |
| `gs2ew.cli` | the `gs2ew` command |

Adding a field, quantity or reduction is a table entry in `catalog.py`; the
plots discover whatever is present in a given file.
