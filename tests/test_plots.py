"""Plot smoke tests.

These check that every plot runs, writes a file, and degrades gracefully when
a diagnostic is missing — not that the picture looks right. The `slow` tests
run the same paths against real GS2 output.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gs2ew import bundle, catalog
from gs2ew.catalog import Drive
from gs2ew.plots import compare, fields, transfer_spectral, transfer_summary
from gs2ew.plots import transfer_surface, transfer_theta, transfer_time, transfer_tp, weights

from conftest import SYNTHETIC_DECK, make_dataset


def assert_replots_identically(figure):
    """Redrawing from the bundle alone gives a byte-identical figure."""
    original = figure.read_bytes()
    figure.unlink()
    assert bundle.replot(figure.with_suffix(".nc"), quiet=True).read_bytes() == original


#: (plot function, keyword arguments) covering every registered plot, run on
#: a synthetic dataset with everything in it.
BUNDLED_PLOTS = {
    "fields_time_traces": (fields.plot_fields_time_traces, {}),
    "fields_by_mode": (fields.plot_fields_by_mode, {}),
    "phi2_by_theta": (fields.plot_phi2_by_theta, {"window": 30.0, "ylim": (0, 50)}),
    "transfer_theta": (transfer_theta.plot_transfer_theta, {"show_std": True, "window": 30.0}),
    "transfer_summary": (transfer_summary.plot_transfer_summary, {}),
    "transfer_flux_surface": (
        transfer_surface.plot_transfer_flux_surface,
        {"window": 30.0, "quantities": ["U", "H"], "ncols": 2},
    ),
    "transfer_hovmoller": (
        transfer_time.plot_transfer_hovmoller,
        {"tstart": 20.0, "share": "all"},
    ),
    "transfer_butterfly": (
        transfer_time.plot_transfer_butterfly,
        {"reduction": "kysource", "klim": 0.5},
    ),
    "transfer_spectral": (
        transfer_spectral.plot_transfer_spectral,
        {"reduction": "kxsource", "klim": 0.5, "tstart": "saturation"},
    ),
    "transfer_spectrum": (
        transfer_spectral.plot_transfer_spectrum,
        {"klim": 0.5, "jacobian": False},
    ),
    "transfer_trapped_passing": (
        transfer_tp.plot_transfer_trapped_passing,
        {"show_std": True, "window": 30.0, "normalise": True},
    ),
    "transfer_trapped_passing_spectral": (
        transfer_tp.plot_transfer_trapped_passing_spectral,
        {"quantity": "U_flr", "field": "apar", "klim": 0.5},
    ),
    "weights": (weights.plot_weights, {}),
    "wl_by_lambda": (weights.plot_wl_by_lambda, {"ncols": 4}),
}


def compared_with_a_finer_run(plot):
    """`plot`, comparing the dataset given with one at ntheta = 16 on a finer
    theta grid, so it fits the single-dataset tests below."""

    def compared(ds, **kwargs):
        finer = make_dataset(
            ntheta=17, deck=SYNTHETIC_DECK.replace("ntheta = 8", "ntheta = 16"), seed=1
        )
        return plot([ds, finer], "ntheta", **kwargs)

    return compared


BUNDLED_PLOTS["compare_scalars"] = (
    compared_with_a_finer_run(compare.plot_compare_scalars),
    {"window": 30.0},
)
BUNDLED_PLOTS["compare_traces"] = (
    compared_with_a_finer_run(compare.plot_compare_traces),
    {"tstart": 30.0},
)
BUNDLED_PLOTS["compare_spectra"] = (
    compared_with_a_finer_run(compare.plot_compare_spectra),
    {"window": 30.0},
)
BUNDLED_PLOTS["compare_theta"] = (
    compared_with_a_finer_run(compare.plot_compare_theta),
    {"quantities": ["U", "S"], "normalise": True},
)


class TestBundles:
    """Every plot saves a bundle that redraws it exactly."""

    def test_every_plot_is_covered(self):
        assert set(BUNDLED_PLOTS) == set(bundle.PLOTS) - {"test_profile"}

    @pytest.mark.parametrize("name", list(BUNDLED_PLOTS))
    def test_replots_from_its_bundle(self, name, tmp_path):
        ds = make_dataset(potential=True, weights=True)
        plot, kwargs = BUNDLED_PLOTS[name]
        figure = plot(ds, output_dir=tmp_path, quiet=True, **kwargs)
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b.attrs["gs2ew_plot"] == name
        assert_replots_identically(figure)

    @pytest.mark.parametrize("name", list(BUNDLED_PLOTS))
    def test_no_data(self, name, tmp_path):
        ds = make_dataset(potential=True, weights=True)
        plot, kwargs = BUNDLED_PLOTS[name]
        plot(ds, output_dir=tmp_path, save_data=False, quiet=True, **kwargs)
        assert not list(tmp_path.glob("*.nc"))

    def test_klim_only_affects_the_figure(self, ds, tmp_path):
        figure = transfer_spectral.plot_transfer_spectral(
            ds, klim=0.5, output_dir=tmp_path, quiet=True
        )
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b.sizes["kxt_shift"] == ds.sizes["kxt_shift"]
        assert bundle.recipe(b)["render"]["klim"] == 0.5

    def test_summary_matches_the_theta_integral(self, ds):
        from gs2ew.geometry import theta_integral

        reduced = transfer_summary.compute_transfer_summary(ds, window=30.0, quiet=True)
        name = "free_energy_transfer_apar_theta"
        series = theta_integral(ds[name], ds).sel(t=slice(70.0, 100.0))
        assert float(reduced[name]) == pytest.approx(float(series.mean("t")))
        assert float(reduced[f"{name}_std"]) == pytest.approx(float(series.std("t")))
        totals = reduced["fluctuation_energy_transfer_trapped_passing"]
        assert totals.dims == ("species", "tp")

    def test_trapped_passing_norm_factors_recorded_by_key(self, ds, tmp_path):
        figure = transfer_tp.plot_transfer_trapped_passing(
            ds,
            normalise=True,
            norm_factors={(Drive("U", "phi"), 0, 2): 3.0},
            quantities=["U"],
            output_dir=tmp_path,
            quiet=True,
        )
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["render"]["norm_factors"] == {
            "fluctuation_energy_transfer_phi_trapped_passing:0:2": 3.0
        }

    def test_phi2_by_theta_drops_the_ky_coordinate(self):
        ds = make_dataset(potential=True)
        reduced = fields.compute_phi2_by_theta(ds, quiet=True)
        assert set(reduced.coords) == {"theta"}

    def test_time_traces_record_saturation(self, ds):
        from gs2ew.io import detect_saturation_time

        reduced = fields.compute_fields_time_traces(ds, window=10.0, quiet=True)
        assert reduced.attrs["tsat"] == detect_saturation_time(ds, window=10.0)


class TestTransferTheta:
    def test_writes_a_file(self, ds, tmp_path):
        path = transfer_theta.plot_transfer_theta(ds, output_dir=tmp_path, quiet=True)
        assert path.exists() and path.stat().st_size > 0

    def test_averaged_filename(self, ds, tmp_path):
        path = transfer_theta.plot_transfer_theta(
            ds, window=30.0, output_dir=tmp_path, quiet=True
        )
        assert path.name == "transfer_theta_averaged.pdf"
        assert path.with_suffix(".png").exists()

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


class TestFluxSurface:
    def test_needs_geometry(self, ds, tmp_path):
        with pytest.raises(ValueError, match="rplot, zplot"):
            transfer_surface.plot_transfer_flux_surface(
                ds.drop_vars(["rplot", "zplot"]), output_dir=tmp_path, quiet=True
            )

    def test_one_poloidal_turn(self, ds):
        """With nperiod > 1, only |theta| <= pi is drawn."""
        reduced = transfer_surface.compute_transfer_flux_surface(
            ds.assign_coords(theta=2 * ds["theta"]), quiet=True
        )
        assert abs(reduced["theta"]).max() <= np.pi
        assert reduced.sizes["theta"] < ds.sizes["theta"]

    def test_four_drives_make_two_rows_of_two(self, ds):
        reduced = transfer_surface.compute_transfer_flux_surface(
            ds, quantities=["U", "H", "S"], fields=["phi"], quiet=True
        )
        fig = transfer_surface.render_transfer_flux_surface(reduced)
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        assert panels[0].get_gridspec().get_geometry() == (2, 2)
        assert all(ax.get_visible() for ax in panels)

    def test_reference_surfaces_of_a_circle(self, ds):
        theta = ds["theta"].values
        surfaces, axis = transfer_surface.reference_surfaces(
            ds["rplot"].values, ds["zplot"].values, ds["rprime"].values
        )
        width, r, z = surfaces[-1]
        assert width == 1.0
        np.testing.assert_allclose(r, 3.0 + np.cos(theta), atol=1e-12)
        np.testing.assert_allclose(z, np.sin(theta), atol=1e-12)
        np.testing.assert_allclose(axis, (3.0, 0.0), atol=1e-12)
        # The simulated surface itself is not repeated.
        assert 0.5 not in [w for w, _, _ in surfaces]


class TestFieldsByMode:
    def test_writes_figure_and_bundle(self, ds, tmp_path):
        figure = fields.plot_fields_by_mode(ds, output_dir=tmp_path, quiet=True)
        assert figure.name == "fields_by_mode.pdf"
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
        assert figure.name == "fields_by_mode_averaged.pdf"
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
