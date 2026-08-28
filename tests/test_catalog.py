"""The transfer-variable catalogue."""

from __future__ import annotations

import numpy as np
import pytest

from gs2ew import catalog
from gs2ew.catalog import BmagScaling, Drive

from conftest import make_dataset


class TestVarname:
    def test_field_reductions(self):
        assert (
            Drive("H", "apar").varname("kxtarget")
            == "free_energy_transfer_apar_theta_kxtarget"
        )
        assert Drive("U", "phi").varname() == "fluctuation_energy_transfer_phi_theta"
        assert (
            Drive("S", "bpar").varname("trapped_passing")
            == "entropy_transfer_bpar_trapped_passing"
        )

    def test_kinetic(self):
        assert Drive("KE").varname() == "kinetic_energy_transfer_theta"
        assert (
            Drive("KE").varname("kysource") == "kinetic_energy_transfer_theta_kysource"
        )

    def test_kinetic_variant(self):
        """GS2's prefactor-comparison branch writes new/old variants."""
        assert (
            Drive("KE").varname("theta", ke_variant="new")
            == "kinetic_energy_transfer_new_theta"
        )

    def test_kinetic_has_no_trapped_passing(self):
        with pytest.raises(ValueError, match="no trapped/passing"):
            Drive("KE").varname("trapped_passing")

    def test_unknown_reduction(self):
        with pytest.raises(ValueError, match="Unknown reduction"):
            Drive("H", "phi").varname("nonsense")


class TestAvailableDrives:
    def test_finds_everything(self, ds):
        drives = catalog.available_drives(ds)
        assert len(drives) == 10  # 3 quantities x 3 fields, plus kinetic
        assert Drive("KE") in drives

    def test_electrostatic_run(self, ds_es):
        drives = catalog.available_drives(ds_es)
        assert {d.field for d in drives} == {"phi", None}

    def test_u_comes_first(self, ds):
        """U leads so it takes the first colour slot in each column."""
        assert catalog.available_drives(ds)[0].quantity == "U"

    def test_filters(self, ds):
        drives = catalog.available_drives(ds, quantities=["H"], fields=["apar"])
        assert drives == [Drive("H", "apar"), Drive("KE")]

    def test_exclude_kinetic(self, ds):
        assert Drive("KE") not in catalog.available_drives(ds, include_kinetic=False)

    def test_missing_diagnostics(self):
        empty = make_dataset(fields=(), kinetic=False, trapped_passing=False)
        assert catalog.available_drives(empty) == []

    def test_variant_lookup(self):
        variant_ds = make_dataset(ke_variant="new")
        assert Drive("KE") not in catalog.available_drives(variant_ds)
        assert Drive("KE") in catalog.available_drives(variant_ds, ke_variant="new")


class TestFieldColumns:
    def test_kinetic_joins_phi_column(self, ds):
        columns = dict(catalog.field_columns(catalog.available_drives(ds)))
        assert Drive("KE") in columns["phi"]
        assert Drive("KE") not in columns["apar"]

    def test_kinetic_alone(self):
        only_ke = make_dataset(fields=(), trapped_passing=False)
        columns = catalog.field_columns(catalog.available_drives(only_ke))
        assert columns == [(None, [Drive("KE")])]


class TestLabels:
    def test_full_label(self):
        assert Drive("U", "phi").label() == r"$N_\mathbf{k}^{U,\delta\phi}$"

    def test_short_label_drops_the_field(self):
        assert Drive("H", "bpar").label_short() == r"$N_\mathbf{k}^{H}$"

    def test_kinetic_label_ignores_field_scaling(self):
        scaling = BmagScaling(field_power=1.0)
        assert Drive("KE").label(scaling) == catalog.KINETIC_LABEL

    @pytest.mark.parametrize(
        "power, expected",
        [(1, "B_0"), (-1, "/B_0"), (2, "B_0^{2}"), (-2, "/B_0^{2}")],
    )
    def test_bmag_suffix(self, power, expected):
        label = Drive("U", "phi").label(BmagScaling(field_power=power))
        assert label.endswith(expected + "$")


class TestScaling:
    def test_div_by_bmag2_is_shorthand(self):
        assert catalog.resolve_scaling(div_by_bmag2=True).field_power == -2

    def test_explicit_power(self):
        scaling = catalog.resolve_scaling(bmag_power=1, kinetic_bmag_power=-2)
        assert (scaling.field_power, scaling.kinetic_power) == (1, -2)

    def test_conflicting_options_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            catalog.resolve_scaling(div_by_bmag2=True, bmag_power=1)

    def test_applied_to_fields_only(self, ds):
        scaling = catalog.resolve_scaling(bmag_power=1)
        scaled = catalog.load(ds, Drive("U", "phi"), scaling=scaling)
        raw = catalog.load(ds, Drive("U", "phi"))
        assert np.allclose((scaled / raw).values, ds["bmag"].values)

        ke_scaled = catalog.load(ds, Drive("KE"), scaling=scaling)
        assert (ke_scaled.values == catalog.load(ds, Drive("KE")).values).all()


class TestIdentity:
    def test_holds_for_synthetic_data(self, ds):
        assert catalog.identity_residual(ds, "phi") < 1e-12

    def test_nan_when_absent(self, ds_es):
        assert catalog.identity_residual(ds_es, "apar") != catalog.identity_residual(
            ds_es, "apar"
        )  # NaN != NaN


def test_load_missing_variable_raises(ds_es):
    with pytest.raises(KeyError, match="free_energy_transfer_apar_theta"):
        catalog.load(ds_es, Drive("H", "apar"))
