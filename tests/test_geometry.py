"""Flux-surface-weighted theta integration."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gs2ew.geometry import theta_integral, theta_weights


def test_weights_come_from_jacob(ds):
    assert np.allclose(theta_weights(ds).values, ds["jacob"].values)


def test_missing_jacob_is_explained(ds):
    with pytest.raises(KeyError, match="jacobian=False"):
        theta_weights(ds.drop_vars("jacob"))


def test_integral_uses_the_jacobian(ds):
    da = ds["fluctuation_energy_transfer_phi_theta"]
    weighted = theta_integral(da, ds)
    bare = theta_integral(da, ds, jacobian=False)
    expected = (da * ds["jacob"]).integrate("theta")
    assert np.allclose(weighted.values, expected.values)
    assert not np.allclose(weighted.values, bare.values)


def test_normalise_gives_a_flux_surface_average(ds):
    """A constant profile averages to its own value whatever the Jacobian."""
    constant = xr.DataArray(
        np.full(ds.sizes["theta"], 3.0), coords={"theta": ds["theta"]}, dims="theta"
    )
    assert theta_integral(constant, ds, normalise=True).values == pytest.approx(3.0)


def test_theta_dimension_is_consumed(ds):
    da = ds["fluctuation_energy_transfer_phi_theta_kxtarget"]
    assert set(theta_integral(da, ds).dims) == {"t", "kxt_shift"}


def test_requires_a_theta_dimension(ds):
    with pytest.raises(ValueError, match="no 'theta' dimension"):
        theta_integral(ds["phi2"], ds)
