# Transfer diagnostics

GS2's `diagnostics_free_energy_transfer` module writes the nonlinear drive of
the zonal modes for several quantities, each decomposed by the *mediator* field
that carries the three-wave coupling:

| Quantity | Key | Symbol | netCDF stem |
| --- | --- | --- | --- |
| Fluctuation energy | `U` | $N_\mathbf{k}^{U,f}$ | `fluctuation_energy_transfer_<field>_*` |
| Free energy | `H` | $N_\mathbf{k}^{H,f}$ | `free_energy_transfer_<field>_*` |
| Entropy | `S` | $N_\mathbf{k}^{S,f}$ | `entropy_transfer_<field>_*` |
| Fluctuation energy, field-only part | `U_field` | $N_\mathbf{k}^{U_\mathrm{field},f}$ | `fluctuation_energy_field_transfer_<field>_*` |
| Fluctuation energy, FLR part | `U_flr` | $N_\mathbf{k}^{U_\mathrm{FLR},f}$ | `fluctuation_energy_flr_transfer_<field>_*` |
| Kinetic energy | `KE` | $N_\mathbf{k}^\mathrm{KE}$ | `kinetic_energy_transfer_*` |

`<field>` is one of `phi`, `apar`, `bpar` (labelled $\delta\phi$,
$\delta A_\parallel$, $\delta B_\parallel$). The kinetic-energy transfer is a
separate, purely electrostatic diagnostic with no field decomposition.

U, H and S are all written directly; `U` is **not** derived here. They
satisfy $H = U - S$ in the stored sign convention, which `gs2ew info` checks.

The field-only and FLR parts of U are only written with
`write_fluctuation_energy_transfer_split`, and only when `phi` is evolved; see
GS2's `diagnostics_free_energy_transfer` for their definitions. `gs2ew info`
checks that they sum to U where they should (`theta` and `kxtarget` only).

The quantities come in up to five reductions:

| Reduction | Dimensions | Meaning |
| --- | --- | --- |
| `theta` | `(t, theta)` | summed over the whole triad |
| `kxtarget` | `(t, theta, kxt_shift)` | resolved by the target zonal mode |
| `kxsource` | `(t, theta, kxs_shift)` | resolved by the source radial wavenumber |
| `kysource` | `(t, theta, kys_ext)` | resolved by the source binormal wavenumber |
| `trapped_passing` | `(t, species, tp, theta, kxt_shift)` | split by species and by trapped (`tp=2`) / passing (`tp=1`) |

Every quantity has the first four. Only U, H, S and the FLR part of U have
`trapped_passing`:

- U, H and S each have their own switch, such as
  `write_free_energy_transfer_trapped_passing`.
- The FLR part of U gets it whenever the split is on.
- The field-only part of U has no velocity-space resolution, so it has no
  `trapped_passing` output.
- The kinetic-energy transfer has no `trapped_passing` output either.

The `trapped_passing` output replaced the old fully velocity-resolved
`*_velocity` arrays, which GS2 no longer writes. GS2 applies the velocity-space
weights `w` and `wl` itself, so nothing further is needed on the analysis side.

## Which plot for what

| Function | Shows |
| --- | --- |
| `plot_transfer_theta` | $N_\mathbf{k}^{Q,f}(\theta)$, one column per field |
| `plot_transfer_flux_surface` | the same, drawn on the flux surface in the poloidal plane |
| `plot_transfer_spectral` | $(\theta, k)$ heatmaps for one reduction |
| `plot_transfer_spectrum` | $\int\mathrm{d}\theta\,J\,N_\mathbf{k}(k)$, 1D spectrum |
| `plot_transfer_trapped_passing` | $\theta$ profiles, trapped vs passing, per species |
| `plot_transfer_trapped_passing_spectral` | $(\theta, k_{xt})$ per population |
| `plot_transfer_hovmoller` | $(t, \theta)$ — the whole run in one figure |
| `plot_transfer_butterfly` | $(t, k)$ — which scales get driven, and when |
| `plot_transfer_summary` | net budget by field and by population |

Each of `plot_transfer_theta`, `plot_transfer_spectral`,
`plot_transfer_trapped_passing` and `plot_transfer_trapped_passing_spectral`
has a `_movie` counterpart. The Hovmöller and butterfly plots are the static
alternative: they show the same time evolution in a figure you can point at.

`plot_transfer_flux_surface` draws the simulated surface from GS2's `rplot`
and `zplot`. GS2 describes only that surface, so the nested surfaces and
boundary around it are the same shape scaled to other radii, for orientation
only, and the width of the coloured band is for visibility.

## Targeted GS2 version

The `experimental/electromagnetic-entropy-transfer-field-flr-split` branch.
Its transfer diagnostics have not been tested yet.
