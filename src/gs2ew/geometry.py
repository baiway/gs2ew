r"""Flux-surface geometry helpers.

The only thing needed here is integrating a poloidal profile properly.
:math:`\theta` is a continuous coordinate, so summing a profile over the
`theta` grid points is not the flux-surface integral — the grid is uniform in
:math:`\theta` but the physical volume element is not. GS2 writes the Jacobian
as ``jacob`` (:math:`1/(\nabla\rho\cdot\nabla\psi\,\, \mathbf{b}\cdot\nabla\theta\, B)`),
and the integral is

.. math:: \int \mathrm{d}\theta \; J(\theta) \, f(\theta).

The wavevector reductions (`kxsource`, `kxtarget`, `kysource`) need no such
weighting: they are discrete sums over Fourier triads, and GS2 itself forms the
summed `_theta` output by plain-summing them. Summing them here plainly
reproduces the `_theta` variable to round-off.
"""

from __future__ import annotations

import xarray as xr


def theta_weights(ds: xr.Dataset) -> xr.DataArray:
    """The Jacobian :math:`J(\\theta)` used to weight poloidal integrals.

    Raises
    ------
    KeyError
        If `jacob` is not present in the dataset.
    """
    if "jacob" not in ds:
        raise KeyError(
            "`jacob` is not in the dataset, so a flux-surface-weighted theta "
            "integral cannot be formed. Pass `jacobian=False` to sum over the "
            "theta grid points instead."
        )
    return ds["jacob"]


def theta_integral(
    da: xr.DataArray,
    ds: xr.Dataset,
    jacobian: bool = True,
    normalise: bool = False,
) -> xr.DataArray:
    r"""Integrate `da` over :math:`\theta`.

    Parameters
    ----------
    da : xarray.DataArray
        Array with a `theta` dimension.
    ds : xarray.Dataset
        Dataset supplying `jacob`.
    jacobian : bool, optional
        If True (default), weight by the Jacobian, giving the flux-surface
        integral :math:`\int \mathrm{d}\theta\, J f`. If False, integrate the
        bare profile :math:`\int \mathrm{d}\theta\, f`, which is what a plain
        sum over grid points approximates.
    normalise : bool, optional
        If True, divide by :math:`\int \mathrm{d}\theta\, J` to give a
        flux-surface *average* rather than an integral. Default False.

    Returns
    -------
    xarray.DataArray
        `da` with the `theta` dimension integrated out.
    """
    if "theta" not in da.dims:
        raise ValueError(f"`da` has no 'theta' dimension (dims: {da.dims}).")

    weight = theta_weights(ds) if jacobian else xr.ones_like(ds["theta"])

    # The theta grid is uniform and spans a full poloidal turn with both
    # endpoints present, so trapezoidal integration is the right rule.
    integral = (da * weight).integrate("theta")

    if normalise:
        integral = integral / weight.integrate("theta")

    return integral


__all__ = ["theta_weights", "theta_integral"]
