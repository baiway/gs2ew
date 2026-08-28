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

    def varname(self, reduction: str = "theta", ke_variant: str | None = None) -> str:
        """netCDF variable name for this drive under `reduction`.

        `ke_variant` inserts a variant tag into the kinetic-energy name, as on
        GS2's `experimental/compare-KE-transfer-prefactors` branch which writes
        both `kinetic_energy_transfer_new_theta` and `..._old_theta`. The
        `fix/kinetic-energy-transfer-prefactor` branch we target writes the
        plain name, so `ke_variant` is None by default.
        """
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
            tag = f"_{ke_variant}" if ke_variant else ""
            return f"kinetic_energy_transfer{tag}_theta{suffix}"

        stem, _ = QUANTITIES[self.quantity]
        return f"{stem}_transfer_{self.field}_theta{suffix}"

    def label(self, scaling: "BmagScaling | None" = None) -> str:
        r"""Full LaTeX label, e.g. ``$N_\mathbf{k}^{U,\delta\phi}$``."""
        if self.is_kinetic:
            return _with_bmag(KINETIC_LABEL, scaling.kinetic_power if scaling else 0.0)
        sym = FIELDS[self.field]
        base = rf"$N_\mathbf{{k}}^{{{self.quantity},{sym}}}$"
        return _with_bmag(base, scaling.field_power if scaling else 0.0)

    def label_short(self, scaling: "BmagScaling | None" = None) -> str:
        r"""Field-free label, e.g. ``$N_\mathbf{k}^{U}$``.

        Used as the legend entry in plots faceted by field, where the mediator
        field is already named in the column title.
        """
        if self.is_kinetic:
            return _with_bmag(KINETIC_LABEL, scaling.kinetic_power if scaling else 0.0)
        base = rf"$N_\mathbf{{k}}^{{{self.quantity}}}$"
        return _with_bmag(base, scaling.field_power if scaling else 0.0)


def _with_bmag(label: str, power: float) -> str:
    """Append a ``B_0`` factor to a ``$...$`` label to reflect `power`."""
    if power == 0:
        return label
    if power == 1:
        factor = r"B_0"
    elif power == -1:
        factor = r"/B_0"
    elif power > 0:
        factor = rf"B_0^{{{_fmt_power(power)}}}"
    else:
        factor = rf"/B_0^{{{_fmt_power(-power)}}}"
    return label[:-1] + factor + "$"


def _fmt_power(power: float) -> str:
    return str(int(power)) if float(power).is_integer() else f"{power:g}"


@dataclass(frozen=True)
class BmagScaling:
    r"""Ad-hoc ``bmag`` powers applied to the transfer diagnostics.

    Scratch machinery for the ongoing investigation into which powers of
    :math:`B_0` belong in the GS2 prefactors. The field-mediated drives
    (H, S, U) currently carry a ``1/bmag`` from
    ``diagnostics_free_energy_transfer``, and the kinetic-energy transfer
    carries ``1/bmag**2`` from ``diagnostics_kinetic_energy_transfer``; what
    is certain is that the two differ by ``bmag**2``.

    `field_power` multiplies H/S/U by ``bmag**field_power``, `kinetic_power`
    multiplies the kinetic-energy transfer by ``bmag**kinetic_power``. Setting
    ``field_power=1`` cancels the ``1/bmag`` in the current GS2 implementation
    and leaves the kinetic-energy transfer alone.

    This whole mechanism is expected to be removed once the correct prefactors
    are settled upstream.
    """

    field_power: float = 0.0
    kinetic_power: float = 0.0

    @property
    def active(self) -> bool:
        return self.field_power != 0 or self.kinetic_power != 0

    def power_for(self, drive: Drive) -> float:
        return self.kinetic_power if drive.is_kinetic else self.field_power


def resolve_scaling(
    div_by_bmag2: bool = False,
    bmag_power: float | None = None,
    kinetic_bmag_power: float = 0.0,
) -> BmagScaling:
    """Build a :class:`BmagScaling` from the plot functions' keyword options.

    `div_by_bmag2` is the long-standing shorthand for dividing the field
    drives by ``bmag**2``, i.e. ``bmag_power=-2``. An explicit `bmag_power`
    takes precedence over it.
    """
    if bmag_power is None:
        bmag_power = -2.0 if div_by_bmag2 else 0.0
    elif div_by_bmag2:
        raise ValueError(
            "Pass either `div_by_bmag2` or `bmag_power`, not both "
            "(`div_by_bmag2=True` is shorthand for `bmag_power=-2`)."
        )
    return BmagScaling(field_power=bmag_power, kinetic_power=kinetic_bmag_power)


def available_drives(
    ds: xr.Dataset,
    reduction: str = "theta",
    quantities: list[str] | None = None,
    fields: list[str] | None = None,
    include_kinetic: bool = True,
    ke_variant: str | None = None,
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
        if ke.varname(reduction, ke_variant) in ds:
            drives.append(ke)

    return drives


def load(
    ds: xr.Dataset,
    drive: Drive,
    reduction: str = "theta",
    scaling: BmagScaling | None = None,
    ke_variant: str | None = None,
) -> xr.DataArray:
    """Return the data for `drive`, with any `scaling` applied.

    The returned array keeps its netCDF dimensions (`t`, `theta`, and the
    reduction's spectral or trapped/passing axes).

    Raises
    ------
    KeyError
        If the variable is not present in `ds`.
    """
    name = drive.varname(reduction, ke_variant)
    if name not in ds:
        raise KeyError(f"{name!r} not found in dataset.")

    da = ds[name]

    if scaling is not None:
        power = scaling.power_for(drive)
        if power != 0:
            if "bmag" not in ds:
                raise ValueError(
                    "bmag scaling was requested but `bmag` is not in the dataset."
                )
            # bmag is a function of theta, so this reshapes the poloidal
            # profile rather than rescaling it uniformly.
            da = da * ds["bmag"] ** power

    return da.rename(name)


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
    "BmagScaling",
    "resolve_scaling",
    "available_drives",
    "load",
    "field_columns",
    "identity_residual",
    "replace",
]
