"""Plot smoke tests.

These check that every plot runs, writes a file, and degrades gracefully when
a diagnostic is missing — not that the picture looks right. The `slow` tests
run the same paths against real GS2 output.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from gs2ew import bundle, catalog
from gs2ew.catalog import Drive
from gs2ew.plots import fields, transfer_spectral, transfer_summary
from gs2ew.plots import transfer_theta, transfer_time, transfer_tp

from conftest import make_dataset


def assert_replots_identically(figure):
    """Redrawing from the bundle alone gives a byte-identical figure."""
    original = figure.read_bytes()
    figure.unlink()
    assert bundle.replot(figure.with_suffix(".nc"), quiet=True).read_bytes() == original


class TestTransferTheta:
    def test_writes_a_file(self, ds, tmp_path):
        path = transfer_theta.plot_transfer_theta(ds, output_dir=tmp_path, quiet=True)
        assert path.exists() and path.stat().st_size > 0

    def test_averaged_filename(self, ds, tmp_path):
        path = transfer_theta.plot_transfer_theta(
            ds, window=30.0, output_dir=tmp_path, quiet=True
        )
        assert path.name == "transfer_theta_averaged.png"

    @pytest.mark.parametrize("kwargs", [
        {"show_std": True, "window": 30.0},
        {"normalise": True},
        {"quantities": ["U"]},
        {"quantities": ["U", "U_field", "U_flr"]},
        {"fields": ["apar"]},
        {"include_kinetic": False},
    ])
    def test_options(self, ds, tmp_path, kwargs):
        assert transfer_theta.plot_transfer_theta(
            ds, output_dir=tmp_path, quiet=True, **kwargs
        ).exists()

    def test_electrostatic_run(self, ds_es, tmp_path):
        assert transfer_theta.plot_transfer_theta(
            ds_es, output_dir=tmp_path, quiet=True
        ).exists()

    def test_no_diagnostics_raises(self, tmp_path):
        empty = make_dataset(fields=(), kinetic=False, trapped_passing=False)
        with pytest.raises(ValueError, match="No transfer diagnostics"):
            transfer_theta.plot_transfer_theta(empty, output_dir=tmp_path, quiet=True)

    @pytest.mark.parametrize("kwargs", [
        {},
        {"show_std": True, "window": 30.0},
        {"show_std": True, "window": 30.0, "normalise": True},
        {"quantities": ["U", "U_field", "U_flr"], "fields": ["phi"]},
        {"include_kinetic": False, "ylim": (-3, 3), "figsize": (8, 3)},
    ])
    def test_replots_from_its_bundle(self, ds, tmp_path, kwargs):
        figure = transfer_theta.plot_transfer_theta(
            ds, output_dir=tmp_path, quiet=True, **kwargs
        )
        assert_replots_identically(figure)

    def test_reduced_data(self, ds):
        reduced = transfer_theta.compute_transfer_theta(
            ds, window=30.0, show_std=True, quiet=True
        )
        u = reduced["fluctuation_energy_transfer_phi_theta"]
        assert u.dims == ("theta",)
        assert u.attrs["quantity"] == "U" and u.attrs["field"] == "phi"
        assert u.attrs["units"] == "GS2 units (to be specified)"
        assert u.attrs["cell_methods"] == "t: mean"
        std = reduced["fluctuation_energy_transfer_phi_theta_std"]
        assert std.attrs["cell_methods"] == "t: standard_deviation"
        assert "field" not in reduced["kinetic_energy_transfer_theta"].attrs
        assert reduced.attrs["averaging"] is True

    def test_last_timestep_has_no_std(self, ds):
        reduced = transfer_theta.compute_transfer_theta(ds, show_std=True, quiet=True)
        assert not any(name.endswith("_std") for name in reduced)
        assert "t" not in reduced.coords
        assert reduced.attrs["tend"] == float(ds["t"][-1])

    def test_recipe(self, ds, tmp_path):
        figure = transfer_theta.plot_transfer_theta(
            ds, tstart="saturation", show_std=True, output_dir=tmp_path, quiet=True
        )
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["plot"] == "transfer_theta"
        assert rec["compute"]["tstart"] == "saturation"
        assert rec["compute"]["show_std"] is True
        assert rec["render"]["normalise"] is False

    def test_norm_factors_recorded_by_variable(self, ds, tmp_path):
        figure = transfer_theta.plot_transfer_theta(
            ds,
            normalise=True,
            norm_factors={Drive("U", "phi"): 2.0},
            fields=["phi"],
            output_dir=tmp_path,
            quiet=True,
        )
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["render"]["norm_factors"] == {
            "fluctuation_energy_transfer_phi_theta": 2.0
        }
        assert_replots_identically(figure)

    def test_no_data(self, ds, tmp_path):
        figure = transfer_theta.plot_transfer_theta(
            ds, output_dir=tmp_path, save_data=False, quiet=True
        )
        assert figure.exists()
        assert not list(tmp_path.glob("*.nc"))


class TestFieldsByMode:
    def test_writes_figure_and_bundle(self, ds, tmp_path):
        figure = fields.plot_fields_by_mode(ds, output_dir=tmp_path, quiet=True)
        assert figure.name == "fields_by_mode.png"
        assert figure.with_suffix(".nc").exists()

    def test_kx_sorted(self, ds):
        reduced = fields.compute_fields_by_mode(ds, quiet=True)
        kx = reduced["kx"].values
        assert (kx[1:] > kx[:-1]).all()
        # Sorting moves the data with its coordinate.
        expected = ds["phi2_by_kx"].isel(t=-1).sel(kx=kx).values
        assert (reduced["phi2_by_kx"].values == expected).all()

    def test_averaged(self, ds, tmp_path):
        figure = fields.plot_fields_by_mode(
            ds, tstart="saturation", output_dir=tmp_path, quiet=True
        )
        assert figure.name == "fields_by_mode_averaged.png"
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b["apar2_by_ky"].attrs["cell_methods"] == "t: mean"
        assert b["apar2_by_ky"].attrs["units"] == "GS2 units"

    @pytest.mark.parametrize("kwargs", [{}, {"window": 30.0}])
    def test_replots_from_its_bundle(self, ds_es, tmp_path, kwargs):
        figure = fields.plot_fields_by_mode(
            ds_es, output_dir=tmp_path, quiet=True, **kwargs
        )
        assert_replots_identically(figure)

    def test_no_spectra_raises(self, ds, tmp_path):
        without = ds.drop_vars([v for v in ds if "2_by_k" in v])
        with pytest.raises(ValueError, match="No field spectra"):
            fields.plot_fields_by_mode(without, output_dir=tmp_path, quiet=True)


class TestSpectral:
    @pytest.mark.parametrize("reduction", ["kxsource", "kxtarget", "kysource"])
    def test_each_reduction(self, ds, tmp_path, reduction):
        assert transfer_spectral.plot_transfer_spectral(
            ds, reduction=reduction, output_dir=tmp_path, quiet=True
        ).exists()
        assert transfer_spectral.plot_transfer_spectrum(
            ds, reduction=reduction, output_dir=tmp_path, quiet=True
        ).exists()

    def test_theta_reduction_rejected(self, ds, tmp_path):
        with pytest.raises(ValueError, match="no wavevector axis"):
            transfer_spectral.plot_transfer_spectral(
                ds, reduction="theta", output_dir=tmp_path, quiet=True
            )

    def test_unknown_reduction_rejected(self, ds, tmp_path):
        with pytest.raises(ValueError, match="Unknown reduction"):
            transfer_spectral.plot_transfer_spectral(
                ds, reduction="kztarget", output_dir=tmp_path, quiet=True
            )

    def test_klim_clips(self, ds, tmp_path):
        assert transfer_spectral.plot_transfer_spectral(
            ds, klim=0.5, output_dir=tmp_path, quiet=True
        ).exists()


class TestTrappedPassing:
    def test_writes_a_file(self, ds, tmp_path):
        assert transfer_tp.plot_transfer_trapped_passing(
            ds, output_dir=tmp_path, quiet=True
        ).exists()

    def test_spectral_view(self, ds, tmp_path):
        assert transfer_tp.plot_transfer_trapped_passing_spectral(
            ds, quantity="H", field="apar", output_dir=tmp_path, quiet=True
        ).exists()

    def test_flr_part(self, ds, tmp_path):
        assert transfer_tp.plot_transfer_trapped_passing_spectral(
            ds, quantity="U_flr", output_dir=tmp_path, quiet=True
        ).exists()

    def test_missing_diagnostic_is_explained(self, tmp_path):
        without = make_dataset(trapped_passing=False)
        with pytest.raises(ValueError, match="write_.*_trapped_passing"):
            transfer_tp.plot_transfer_trapped_passing(
                without, output_dir=tmp_path, quiet=True
            )


class TestTimeResolved:
    def test_hovmoller(self, ds, tmp_path):
        assert transfer_time.plot_transfer_hovmoller(
            ds, output_dir=tmp_path, quiet=True
        ).exists()

    def test_butterfly(self, ds, tmp_path):
        assert transfer_time.plot_transfer_butterfly(
            ds, klim=0.5, output_dir=tmp_path, quiet=True
        ).exists()

    def test_time_slice(self, ds, tmp_path):
        assert transfer_time.plot_transfer_hovmoller(
            ds, tstart=20.0, tend=80.0, output_dir=tmp_path, quiet=True
        ).exists()


class TestSummary:
    def test_with_trapped_passing(self, ds, tmp_path):
        assert transfer_summary.plot_transfer_summary(
            ds, output_dir=tmp_path, quiet=True
        ).exists()

    def test_without_trapped_passing(self, tmp_path):
        without = make_dataset(trapped_passing=False)
        assert transfer_summary.plot_transfer_summary(
            without, output_dir=tmp_path, quiet=True
        ).exists()


class TestFields:
    def test_phi2_by_theta_needs_a_potential(self, ds, tmp_path):
        with pytest.raises(KeyError, match="No electrostatic potential"):
            fields.plot_phi2_by_theta(ds, output_dir=tmp_path, quiet=True)

    def test_movie_needs_a_time_axis(self, ds, tmp_path):
        with pytest.raises(KeyError):
            fields.plot_phi2_by_theta_movie(ds, output_dir=tmp_path)


class TestMovies:
    def test_theta_movie(self, ds, tmp_path):
        path = transfer_theta.plot_transfer_theta_movie(
            ds, tstart=60.0, stride=4, fps=4, output_dir=tmp_path
        )
        assert path.exists() and path.stat().st_size > 0
        # Frames are not individually reproducible plots.
        assert not list(tmp_path.rglob("*.nc"))

    def test_trapped_passing_movie(self, ds, tmp_path):
        assert transfer_tp.plot_transfer_trapped_passing_movie(
            ds, tstart=60.0, stride=4, fps=4, quantities=["U"], output_dir=tmp_path
        ).exists()

    def test_stale_frames_are_cleared(self, ds, tmp_path):
        """A shorter re-run must not inherit the previous run's tail frames."""
        transfer_theta.plot_transfer_theta_movie(
            ds, tstart=20.0, stride=2, fps=4, output_dir=tmp_path
        )
        transfer_theta.plot_transfer_theta_movie(
            ds, tstart=80.0, stride=2, fps=4, output_dir=tmp_path
        )
        frames = list((tmp_path / "transfer_theta_frames").glob("frame_*.png"))
        expected = len(
            [t for t in ds["t"].values if t >= 80.0][::2]
        )
        assert len(frames) == expected


@pytest.mark.slow
class TestRealOutput:
    """The same paths against a real GS2 file."""

    def test_identity_holds(self, real_ds):
        for field in catalog.FIELDS:
            residual = catalog.identity_residual(real_ds, field)
            if residual == residual:  # skip NaN (field not present)
                assert residual < 1e-10

    def test_reductions_sum_to_the_theta_output(self, real_ds):
        """GS2 forms `_theta` by plain-summing the triad legs, with no weight."""
        import numpy as np

        drive = catalog.Drive("U", "phi")
        total = catalog.load(real_ds, drive).isel(t=-1).values
        for reduction in ("kxsource", "kxtarget", "kysource"):
            _, dim = catalog.REDUCTIONS[reduction]
            summed = catalog.load(real_ds, drive, reduction).isel(t=-1).sum(dim).values
            assert np.allclose(summed, total, rtol=1e-10)

    def test_trapped_passing_sums_to_the_theta_output(self, real_ds):
        import numpy as np

        drive = catalog.Drive("H", "phi")
        total = catalog.load(real_ds, drive).isel(t=-1).values
        summed = (
            catalog.load(real_ds, drive, "trapped_passing")
            .isel(t=-1)
            .sum(["species", "tp", "kxt_shift"])
            .values
        )
        assert np.allclose(summed, total, rtol=1e-10)

    def test_standard_plots_run(self, real_ds, tmp_path):
        transfer_theta.plot_transfer_theta(
            real_ds, tstart="saturation", show_std=True, output_dir=tmp_path, quiet=True
        )
        transfer_tp.plot_transfer_trapped_passing(
            real_ds, tstart="saturation", output_dir=tmp_path, quiet=True
        )
        transfer_spectral.plot_transfer_spectral(
            real_ds, klim=0.5, output_dir=tmp_path, quiet=True
        )
        transfer_time.plot_transfer_hovmoller(real_ds, output_dir=tmp_path, quiet=True)
        transfer_summary.plot_transfer_summary(real_ds, output_dir=tmp_path, quiet=True)
        assert len(list(tmp_path.glob("*.png"))) == 5
