"""The transfer-variable catalogue."""

from __future__ import annotations

import pytest

from gs2ew import catalog
from gs2ew.catalog import Drive

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
