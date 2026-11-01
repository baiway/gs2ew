# Figures

Each figure is saved as a PDF, for print, and a PNG, for viewing. Both are
drawn at the size they will be printed, 6 in wide by default, so include them
at their natural width (`\includegraphics{fig.pdf}`, with no `width=`) and the
text stays the size it was set at. Heatmaps are rasterised inside the PDF, so
it stays small and quick to open while the axes and text stay vector. PDFs
carry no creation date, so redrawing one gives an identical file.

The default style is gs2ew's `print` style
([`src/gs2ew/styles/print.mplstyle`](../src/gs2ew/styles/print.mplstyle)): set
in XCharter, to match a Charter document, with a light grid, colourblind-safe
line colours and a blue–red colour map for signed data. XCharter is taken from
TeX Live if it is installed; otherwise the text is set in DejaVu Serif.

## Changing how figures look

| `plot` / `batch` | `replot --set` | |
| --- | --- | --- |
| `--style FILE` | `style=FILE` | an `.mplstyle` applied on top of the print style |
| `--width INCHES` | `width=INCHES` | figure width; each plot sets its own height |
| `--no-title` | `title=false` | leave the averaging window to the caption |
| `--format pdf png` | `formats=pdf` | file types to save |
| `--dpi N` | `dpi=N` | resolution of PNGs and of heatmaps in PDFs |

Restyling needs only the bundle, so it takes seconds. For a single-column
journal figure:

```bash
gs2ew replot RUN/gs2ew/transfer_theta_averaged.nc -o paper/ \
    --set style=journal.mplstyle --set width=3.4 --set title=false
```

A style file sets only what it changes; anything else comes from the print
style:

```
font.size:       8
xtick.labelsize: 7
ytick.labelsize: 7
legend.fontsize: 7
axes.grid:       False
image.cmap:      seismic
```

Each transferred quantity takes its colour from its place in the style's
colour cycle (`axes.prop_cycle`), in the order U, H, S, KE, U_field, U_flr, and
signed heatmaps use `image.cmap`. The bundle keeps a copy of the style file,
so the figure can be redrawn after the file has moved.

From Python, `figure_options` sets the same options for a block of plots:

```python
from gs2ew import figure_options, plot_transfer_theta

with figure_options(width=3.4, title=False, formats=["pdf"]):
    plot_transfer_theta(ds, tstart="saturation")
```

A `filename` ending in a format, such as `filename="theta.png"`, saves that
format only.
