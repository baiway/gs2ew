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

    def test_split_parts(self):
        assert (
            Drive("U_field", "phi").varname()
            == "fluctuation_energy_field_transfer_phi_theta"
        )
        assert (
            Drive("U_flr", "bpar").varname("trapped_passing")
            == "fluctuation_energy_flr_transfer_bpar_trapped_passing"
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
        assert len(drives) == 16  # 5 quantities x 3 fields, plus kinetic
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

    def test_field_only_part_has_no_trapped_passing(self, ds):
        quantities = {
            d.quantity for d in catalog.available_drives(ds, "trapped_passing")
        }
        assert "U_flr" in quantities
        assert "U_field" not in quantities

    def test_split_is_optional(self):
        unsplit = make_dataset(split=False)
        quantities = {d.quantity for d in catalog.available_drives(unsplit)}
        assert quantities == {"U", "H", "S", "KE"}

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

    def test_split_labels_use_symbols(self):
        assert (
            Drive("U_flr", "phi").label()
            == r"$N_\mathbf{k}^{U_\mathrm{FLR},\delta\phi}$"
        )
        assert (
            Drive("U_field", "apar").label_short()
            == r"$N_\mathbf{k}^{U_\mathrm{field}}$"
        )

    def test_kinetic_label(self):
        kinetic = Drive(catalog.KINETIC)
        assert kinetic.label() == kinetic.label_short() == r"$N_\mathbf{k}^\mathrm{KE}$"


class TestIdentity:
    def test_holds_for_synthetic_data(self, ds):
        assert catalog.identity_residual(ds, "phi") < 1e-12

    @pytest.mark.parametrize("reduction", ["theta", "kxtarget"])
    def test_split_holds_where_summed_over_sources(self, ds, reduction):
        residual = catalog.identity_residual(
            ds, "apar", reduction, identity="U = U_field + U_flr"
        )
        assert residual < 1e-12

    @pytest.mark.parametrize("reduction", ["kxsource", "kysource", "trapped_passing"])
    def test_split_refused_where_it_does_not_hold(self, ds, reduction):
        with pytest.raises(ValueError, match="does not hold"):
            catalog.identity_residual(
                ds, "phi", reduction, identity="U = U_field + U_flr"
            )

    def test_split_nan_when_not_written(self):
        unsplit = make_dataset(split=False)
        residual = catalog.identity_residual(
            unsplit, "phi", identity="U = U_field + U_flr"
        )
        assert residual != residual  # NaN

    def test_unknown_identity(self, ds):
        with pytest.raises(ValueError, match="Unknown identity"):
            catalog.identity_residual(ds, "phi", identity="E = mc^2")

    def test_nan_when_absent(self, ds_es):
        assert catalog.identity_residual(ds_es, "apar") != catalog.identity_residual(
            ds_es, "apar"
        )  # NaN != NaN


def test_load_missing_variable_raises(ds_es):
    with pytest.raises(KeyError, match="free_energy_transfer_apar_theta"):
        catalog.load(ds_es, Drive("H", "apar"))
