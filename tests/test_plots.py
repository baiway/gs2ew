"""Plot smoke tests.

These check that every plot runs, writes a file, and degrades gracefully when
a diagnostic is missing — not that the picture looks right. The `slow` tests
run the same paths against real GS2 output.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from gs2ew import catalog
from gs2ew.plots import fields, transfer_spectral, transfer_summary
from gs2ew.plots import transfer_theta, transfer_time, transfer_tp

from conftest import make_dataset


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
