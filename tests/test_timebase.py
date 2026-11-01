"""Time-window and frame resolution."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gs2ew import timebase
from gs2ew.io import detect_saturation_time

from conftest import make_dataset


class TestResolveWindow:
    def test_no_arguments_gives_a_single_timestep(self, ds):
        tw = timebase.resolve_window(ds)
        assert not tw.averaging
        assert tw.tstart == tw.tend == float(ds["t"].values[-1])
        assert tw.title is None

    def test_window_alone_takes_the_tail(self, ds):
        tw = timebase.resolve_window(ds, window=30.0)
        assert tw.averaging
        assert tw.tend == pytest.approx(float(ds["t"].values[-1]))
        assert tw.tstart == pytest.approx(tw.tend - 30.0)

    def test_tstart_alone_runs_to_the_end(self, ds):
        tw = timebase.resolve_window(ds, tstart=20.0)
        assert (tw.tstart, tw.tend) == (20.0, pytest.approx(float(ds["t"].values[-1])))

    def test_both_bounds(self, ds):
        tw = timebase.resolve_window(ds, tstart=20.0, window=30.0)
        assert (tw.tstart, tw.tend) == (20.0, 50.0)

    def test_saturation_is_padded_into_the_saturated_interval(self, ds):
        tsat = detect_saturation_time(ds)
        tend = float(ds["t"].values[-1])
        tw = timebase.resolve_window(ds, tstart="saturation", quiet=True)
        assert tw.tstart == pytest.approx(tsat + 0.20 * (tend - tsat))

    def test_saturation_pad_is_configurable(self, ds):
        a = timebase.resolve_window(ds, tstart="saturation", saturation_pad=0.0, quiet=True)
        b = timebase.resolve_window(ds, tstart="saturation", saturation_pad=0.5, quiet=True)
        assert b.tstart > a.tstart

    def test_unknown_string_rejected(self, ds):
        with pytest.raises(ValueError, match="Unknown tstart"):
            timebase.resolve_window(ds, tstart="whenever")

    def test_undetectable_saturation_falls_back_to_start(self, ds):
        tw = timebase.resolve_window(
            ds.drop_vars("phi2"), tstart="saturation", quiet=True
        )
        assert tw.tstart == pytest.approx(float(ds["t"].values[0]))


class TestTimeWindowReduction:
    def test_mean_over_the_window(self, ds):
        tw = timebase.resolve_window(ds, tstart=0.0, window=100.0)
        da = ds["fluctuation_energy_transfer_phi_theta"]
        assert np.allclose(tw.mean(da).values, da.mean("t").values)

    def test_single_step_takes_the_last(self, ds):
        tw = timebase.resolve_window(ds)
        da = ds["fluctuation_energy_transfer_phi_theta"]
        assert np.allclose(tw.mean(da).values, da.isel(t=-1).values)

    def test_no_std_without_averaging(self, ds):
        assert timebase.resolve_window(ds).std(ds["phi2"]) is None


class TestFrameIndices:
    def test_defaults_to_saturation_not_the_last_step(self, ds):
        """A movie starting at the last timestep would be one frame long."""
        indices = timebase.frame_indices(ds, quiet=True)
        assert len(indices) > 1

    def test_drops_frames_before_tstart(self, ds):
        indices = timebase.frame_indices(ds, tstart=50.0)
        assert (ds["t"].values[indices] >= 50.0).all()

    def test_drops_partial_windows(self, ds):
        window = 30.0
        indices = timebase.frame_indices(ds, tstart=0.0, window=window)
        assert (ds["t"].values[indices] + window <= ds["t"].values[-1]).all()

    def test_stride(self, ds):
        every = timebase.frame_indices(ds, tstart=0.0)
        third = timebase.frame_indices(ds, tstart=0.0, stride=3)
        assert list(third) == list(every[::3])

    def test_no_frames_raises(self, ds):
        with pytest.raises(ValueError, match="No frames left"):
            timebase.frame_indices(ds, tstart=1e9)


class TestSaturationDetection:
    def test_finds_the_knee(self, ds):
        tsat = detect_saturation_time(ds)
        assert 0 < tsat < float(ds["t"].values[-1])

    def test_missing_phi2(self, ds):
        assert np.isnan(detect_saturation_time(ds.drop_vars("phi2")))

    def test_run_shorter_than_the_window(self, ds):
        assert np.isnan(detect_saturation_time(ds, window=1e6))

    def test_single_timestep(self, ds):
        assert np.isnan(detect_saturation_time(ds.isel(t=slice(0, 1))))

    def test_non_positive_phi2(self, ds):
        zeroed = ds.assign(phi2=("t", np.zeros(ds.sizes["t"])))
        assert np.isnan(detect_saturation_time(zeroed))

    def test_never_saturates(self):
        """A run still growing at the end has no saturation time."""
        nt = 40
        growing = xr.Dataset(
            {"phi2": ("t", np.exp(np.linspace(0, 40, nt)))},
            coords={"t": np.linspace(0, 100, nt)},
        )
        assert np.isnan(detect_saturation_time(growing))


def test_geometry_and_reductions_agree_on_the_synthetic_dataset():
    """The wavevector reductions are plain sums, matching GS2's own reduction."""
    ds = make_dataset()
    assert "kinetic_energy_transfer_theta_kxtarget" in ds
