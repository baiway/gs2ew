"""Registry of GS2's nonlinear-transfer netCDF variables.

GS2 writes the nonlinear drive of the zonal modes for three related
quantities, each broken down by the *mediator* field that carries the
coupling and by one of several reductions over the triad wavevectors:

    <quantity>_transfer_<field>_theta[_<reduction>]
    <quantity>_transfer_<field>_trapped_passing

with ``quantity`` in ``{free_energy, entropy, fluctuation_energy}`` and
``field`` in ``{phi, apar, bpar}``. The kinetic-energy transfer has no field
decomposition and is written as ``kinetic_energy_transfer_theta[_<reduction>]``.

The three quantities satisfy :math:`H = U - S` in the stored sign convention
(GS2 folds the sign of the entropy integrand into the written array), which
:func:`identity_residual` checks.

This module is the single place that knows those names. Everything else works
in terms of :class:`Drive` objects and asks the catalogue for the data, so
adding a field, quantity or reduction is a table entry rather than another
``if name in ds`` branch scattered across the plot code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

# Mediator fields, in the order GS2 declares them, with their LaTeX symbols.
# The `delta` matches the notation used in GS2's diagnostic documentation.
FIELDS: dict[str, str] = {
    "phi": r"\delta\phi",
    "apar": r"\delta A_\parallel",
    "bpar": r"\delta B_\parallel",
}

# Transferred quantities: plot key -> (netCDF stem, human-readable name).
# `U` is written directly by GS2 (commit 17a7177ea); older analysis code
# derived it as `H + S`, which is no longer necessary.
QUANTITIES: dict[str, tuple[str, str]] = {
    "U": ("fluctuation_energy", "fluctuation energy"),
    "H": ("free_energy", "free energy"),
    "S": ("entropy", "entropy"),
}

# The kinetic-energy transfer is a separate diagnostic with no mediator-field
# decomposition, so it gets its own quantity key and its own label.
KINETIC = "KE"
KINETIC_LABEL = r"$T_v^\mathrm{ZF}$"

# Reductions over the triad wavevectors: name -> (variable suffix, extra dim).
# `theta` is the fully summed profile; the others resolve one wavevector leg.
# `trapped_passing` is handled separately as it replaces the `_theta` infix.
REDUCTIONS: dict[str, tuple[str, str | None]] = {
    "theta": ("", None),
    "kxsource": ("_kxsource", "kxs_shift"),
    "kxtarget": ("_kxtarget", "kxt_shift"),
    "kysource": ("_kysource", "kys_ext"),
}

# Axis labels for each reduction's spectral dimension.
K_LABELS: dict[str, str] = {
    "kxs_shift": r"$k_{xs}\rho_\mathrm{ref}$",
    "kxt_shift": r"$k_{xt}\rho_\mathrm{ref}$",
    "kys_ext": r"$k_{ys}\rho_\mathrm{ref}$",
}

# The `tp` axis is 1-based in the file: 1 = passing, 2 = trapped (marginally
# trapped `wfb` particles are counted as trapped).
TP_LABELS: dict[int, str] = {1: "passing", 2: "trapped"}


@dataclass(frozen=True)
class Drive:
    """One nonlinear drive: a transferred quantity and its mediator field.

    `quantity` is a key of :data:`QUANTITIES` or :data:`KINETIC`; `field` is a
    key of :data:`FIELDS`, or None for the kinetic-energy transfer.
    """

    quantity: str
    field: str | None = None

    @property
    def is_kinetic(self) -> bool:
        return self.quantity == KINETIC

    def varname(self, reduction: str = "theta") -> str:
        """netCDF variable name for this drive under `reduction`."""
        if reduction == "trapped_passing":
            if self.is_kinetic:
                raise ValueError(
                    "The kinetic-energy transfer has no trapped/passing output."
                )
            stem, _ = QUANTITIES[self.quantity]
            return f"{stem}_transfer_{self.field}_trapped_passing"

        try:
            suffix, _ = REDUCTIONS[reduction]
        except KeyError:
            raise ValueError(
                f"Unknown reduction {reduction!r}; expected one of "
                f"{sorted(REDUCTIONS)} or 'trapped_passing'."
            ) from None

        if self.is_kinetic:
            return f"kinetic_energy_transfer_theta{suffix}"

        stem, _ = QUANTITIES[self.quantity]
        return f"{stem}_transfer_{self.field}_theta{suffix}"

    def label(self) -> str:
        r"""Full LaTeX label, e.g. ``$N_\mathbf{k}^{U,\delta\phi}$``."""
        if self.is_kinetic:
            return KINETIC_LABEL
        sym = FIELDS[self.field]
        return rf"$N_\mathbf{{k}}^{{{self.quantity},{sym}}}$"

    def label_short(self) -> str:
        r"""Field-free label, e.g. ``$N_\mathbf{k}^{U}$``.

        Used as the legend entry in plots faceted by field, where the mediator
        field is already named in the column title.
        """
        if self.is_kinetic:
            return KINETIC_LABEL
        return rf"$N_\mathbf{{k}}^{{{self.quantity}}}$"


def available_drives(
    ds: xr.Dataset,
    reduction: str = "theta",
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
) -> list[Drive]:
    """Return every drive present in `ds` for the given `reduction`.

    Drives are ordered U, H, S within each field, and fields in GS2's
    declaration order, so U takes the first colour in a shared colour cycle.
    The kinetic-energy transfer, when present and requested, comes last.

    `quantities` and `fields` restrict the search; None means "all".
    """
    want_q = list(QUANTITIES) if quantities is None else list(quantities)
    want_f = list(FIELDS) if fields is None else list(fields)

    drives = [
        Drive(q, f)
        for f in want_f
        for q in want_q
        if f in FIELDS and q in QUANTITIES and Drive(q, f).varname(reduction) in ds
    ]

    if include_kinetic and reduction != "trapped_passing":
        ke = Drive(KINETIC)
        if ke.varname(reduction) in ds:
            drives.append(ke)

    return drives


def load(ds: xr.Dataset, drive: Drive, reduction: str = "theta") -> xr.DataArray:
    """Return the data for `drive`.

    The returned array keeps its netCDF dimensions (`t`, `theta`, and the
    reduction's spectral or trapped/passing axes).

    Raises
    ------
    KeyError
        If the variable is not present in `ds`.
    """
    name = drive.varname(reduction)
    if name not in ds:
        raise KeyError(f"{name!r} not found in dataset.")
    return ds[name]


def field_columns(drives: list[Drive]) -> list[tuple[str | None, list[Drive]]]:
    """Group `drives` into one column per mediator field, preserving order.

    The kinetic-energy transfer has no field of its own; it is appended to the
    `phi` column (it is a purely electrostatic diagnostic) so it appears as a
    common reference without creating a column of its own. If there is no
    `phi` column it becomes its own trailing column.
    """
    columns: dict[str | None, list[Drive]] = {}
    for drive in drives:
        if drive.is_kinetic:
            continue
        columns.setdefault(drive.field, []).append(drive)

    kinetic = [d for d in drives if d.is_kinetic]
    if kinetic:
        if "phi" in columns:
            columns["phi"].extend(kinetic)
        elif columns:
            # No phi column: attach to the first field rather than inventing one.
            columns[next(iter(columns))].extend(kinetic)
        else:
            columns[None] = kinetic

    return list(columns.items())


def identity_residual(ds: xr.Dataset, field: str, reduction: str = "theta") -> float:
    """Max relative residual of the identity :math:`H = U - S` for `field`.

    GS2 derives all three quantities from the same triad integrand, so this
    should be at round-off (order 1e-15). A large value means the three
    outputs did not come from the same run or the sign convention has changed.

    Returns NaN if any of the three variables is absent.
    """
    try:
        H = load(ds, Drive("H", field), reduction).values
        S = load(ds, Drive("S", field), reduction).values
        U = load(ds, Drive("U", field), reduction).values
    except KeyError:
        return float("nan")

    scale = np.nanmax(np.abs(H))
    if scale == 0:
        return 0.0
    return float(np.nanmax(np.abs(H - (U - S))) / scale)


__all__ = [
    "FIELDS",
    "QUANTITIES",
    "KINETIC",
    "KINETIC_LABEL",
    "REDUCTIONS",
    "K_LABELS",
    "TP_LABELS",
    "Drive",
    "available_drives",
    "load",
    "field_columns",
    "identity_residual",
    "replace",
]
