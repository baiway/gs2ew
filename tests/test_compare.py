"""Comparing runs that differ in one input parameter."""

from __future__ import annotations

import json
from contextlib import ExitStack

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gs2ew import bundle, cli
from gs2ew.io import load_gs2_output
from gs2ew.plots import compare

from conftest import SYNTHETIC_DECK, make_dataset


def make_run(root, name, ntheta=8, grid=9, delt=0.05, seed=0, **kwargs):
    """A run directory whose deck has `ntheta` and `delt`, on a theta grid of
    `grid` points."""
    deck = SYNTHETIC_DECK.replace("ntheta = 8", f"ntheta = {ntheta}").replace(
        "delt = 0.05", f"delt = {delt}"
    )
    run_dir = root / name
    run_dir.mkdir(parents=True)
    make_dataset(
        ntheta=grid,
        deck=deck,
        seed=seed,
        run_id=f"{seed:08d}-0000-4000-8000-000000000000",
        **kwargs,
    ).to_netcdf(run_dir / f"{name}.out.nc")
    (run_dir / f"{name}.in").write_text(deck)
    return run_dir


@pytest.fixture
def scan_dir(tmp_path):
    """A one-at-a-time scan: base, two over ntheta, one over delt."""
    root = tmp_path / "scan"
    make_run(root, "base", seed=1)
    make_run(root, "ntheta_16", ntheta=16, grid=17, seed=2)
    make_run(root, "ntheta_4", ntheta=4, grid=5, seed=3)
    make_run(root, "delt_0p1", delt=0.1, seed=4)
    return root


@pytest.fixture
def datasets(scan_dir):
    with ExitStack() as stack:
        yield [
            stack.enter_context(load_gs2_output(run.out_nc))
            for run in compare.find_runs([scan_dir])
        ]


def by_name(datasets, *names):
    found = {compare.run_label(ds): ds for ds in datasets}
    return [found[name] for name in names]


class TestFindingRuns:
    def test_searches_a_directory(self, scan_dir):
        names = [run.name for run in compare.find_runs([scan_dir])]
        assert names == ["base", "delt_0p1", "ntheta_16", "ntheta_4"]

    def test_run_directories(self, scan_dir):
        runs = compare.find_runs([scan_dir / "base", scan_dir / "ntheta_4" / "ntheta_4.out.nc"])
        assert [run.name for run in runs] == ["base", "ntheta_4"]

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare.find_runs([tmp_path / "nowhere"])

    def test_parameter(self, datasets):
        (base,) = by_name(datasets, "base")
        assert compare.parameter(base, "ntheta") == ("theta_grid_parameters.ntheta", 8)
        assert compare.parameter(base, "knobs.delt") == ("knobs.delt", 0.05)

    def test_parameter_left_at_its_default(self, datasets):
        """Only in the deck GS2 fills the defaults into."""
        (base,) = by_name(datasets, "base")
        assert compare.parameter(base, "layout") == ("layouts_knobs.layout", "lxyes")

    def test_parameter_missing(self, datasets):
        with pytest.raises(ValueError, match="not in the input deck of base"):
            compare.parameter(by_name(datasets, "base")[0], "negrid")

    def test_comparable_runs_differ_only_in_the_parameter(self, datasets):
        chosen = [compare.run_label(datasets[i]) for i in compare.comparable(datasets, "ntheta")]
        assert sorted(chosen) == ["base", "ntheta_16", "ntheta_4"]
        chosen = [compare.run_label(datasets[i]) for i in compare.comparable(datasets, "delt")]
        assert sorted(chosen) == ["base", "delt_0p1"]

    def test_ignored_parameters_may_differ(self, datasets):
        assert len(compare.comparable(datasets, "ntheta", ignore=["delt"])) == 4

    def test_nothing_to_compare(self, datasets):
        with pytest.raises(ValueError, match="No two runs"):
            compare.comparable(by_name(datasets, "ntheta_4", "delt_0p1"), "ntheta")

    def test_equally_many(self, scan_dir):
        # Over delt: base with delt_0p1, and ntheta_16 with this.
        make_run(scan_dir, "ntheta_16_delt_0p1", ntheta=16, grid=17, delt=0.1, seed=6)
        with ExitStack() as stack:
            runs = [
                stack.enter_context(load_gs2_output(r.out_nc))
                for r in compare.find_runs([scan_dir])
            ]
            pairs = by_name(runs, "base", "delt_0p1", "ntheta_16", "ntheta_16_delt_0p1")
            with pytest.raises(ValueError, match="Equally many"):
                compare.comparable(pairs, "delt")

    def test_in_order(self, datasets):
        ordered = compare.in_order(by_name(datasets, "ntheta_16", "base", "ntheta_4"), "ntheta")
        assert [compare.run_label(ds) for ds in ordered] == ["ntheta_4", "base", "ntheta_16"]


class TestCompute:
    def test_scalars(self, datasets):
        runs = by_name(datasets, "ntheta_4", "base", "ntheta_16")
        reduced = compare.compute_compare_scalars(runs, "ntheta", quiet=True)
        assert list(reduced["value"].values) == [4.0, 8.0, 16.0]
        assert list(reduced["run_name"].values) == ["ntheta_4", "base", "ntheta_16"]
        assert reduced.attrs["parameter"] == "theta_grid_parameters.ntheta"
        assert "phi2" in reduced and "phi2_std" in reduced
        assert "heat_flux_tot" not in reduced  # the synthetic runs write none
        assert "fluctuation_energy_transfer_phi_theta" in reduced
        assert reduced["kinetic_energy_transfer_theta"].attrs["quantity"] == "KE"
        # Each run's own window.
        tw = compare.resolve_window(runs[0], "saturation", None, 0.2, quiet=True)
        assert reduced["tstart"].values[0] == tw.tstart

    def test_drives_any_run_has(self, scan_dir):
        make_run(scan_dir, "ntheta_32", ntheta=32, fields=("phi",), seed=5)
        with ExitStack() as stack:
            runs = [
                stack.enter_context(load_gs2_output(scan_dir / n / f"{n}.out.nc"))
                for n in ("base", "ntheta_32")
            ]
            reduced = compare.compute_compare_scalars(runs, "ntheta", quiet=True)
        assert not np.isnan(reduced["free_energy_transfer_phi_theta"].values).any()
        apar = reduced["free_energy_transfer_apar_theta"].values
        assert not np.isnan(apar[0]) and np.isnan(apar[1])

    def test_warns_of_runs_with_almost_nothing_in_their_window(self, scan_dir, capsys):
        make_run(scan_dir, "ntheta_32", ntheta=32, nt=2, seed=5)
        with ExitStack() as stack:
            runs = [
                stack.enter_context(load_gs2_output(scan_dir / n / f"{n}.out.nc"))
                for n in ("base", "ntheta_32")
            ]
            compare.compute_compare_scalars(runs, "ntheta", tstart=0.0)
        assert "Warning: ntheta_32 has 2 timestep(s) in its window." in capsys.readouterr().out

    def test_profiles_share_the_union_of_theta_grids(self, datasets):
        runs = by_name(datasets, "ntheta_4", "ntheta_16")
        reduced = compare.compute_compare_theta(runs, "ntheta", quiet=True)
        profile = reduced["entropy_transfer_phi_theta"].values
        assert reduced.sizes["theta"] == 17
        assert np.isnan(profile[0]).sum() == 12  # 5 of the 17 points
        assert not np.isnan(profile[1]).any()


class TestFields:
    def test_traces_share_the_union_of_times(self, scan_dir):
        make_run(scan_dir, "ntheta_32", ntheta=32, nt=12, seed=5)
        with ExitStack() as stack:
            runs = [
                stack.enter_context(load_gs2_output(scan_dir / n / f"{n}.out.nc"))
                for n in ("base", "ntheta_32")
            ]
            reduced = compare.compute_compare_traces(runs, "ntheta", quiet=True)
        assert reduced.sizes["t"] == len(np.union1d(runs[0]["t"], runs[1]["t"]))
        assert (~np.isnan(reduced["phi2"].values)).sum(axis=1).tolist() == [24, 12]
        assert "apar2" not in reduced  # the synthetic runs write only phi2
        assert np.isfinite(reduced["tsat"].values).all()

    def test_spectra_averaged_over_each_window(self, datasets):
        runs = by_name(datasets, "ntheta_4", "base")
        reduced = compare.compute_compare_spectra(runs, "ntheta", tstart=50.0, quiet=True)
        expected = runs[1]["phi2_by_ky"].sel(t=slice(50.0, None)).mean("t").values
        np.testing.assert_allclose(reduced["phi2_by_ky"].values[1], expected)
        assert reduced["phi2_by_kx"].dims == ("run", "kx")
        assert (np.diff(reduced["kx"].values) > 0).all()

    def test_one_field_side_by_side(self, tmp_path):
        root = tmp_path / "es"
        for i, ntheta in enumerate((8, 16)):
            make_run(root, f"ntheta_{ntheta}", ntheta=ntheta, fields=("phi",), seed=i)
        with ExitStack() as stack:
            runs = [stack.enter_context(load_gs2_output(r.out_nc)) for r in compare.find_runs([root])]
            reduced = compare.compute_compare_spectra(runs, "ntheta", quiet=True)
        assert "apar2_by_kx" not in reduced
        fig = compare.render_compare_spectra(reduced)
        assert (fig.axes[0].get_gridspec().nrows, fig.axes[0].get_gridspec().ncols) == (1, 2)

    def test_no_traces(self, datasets):
        runs = [ds.drop_vars("phi2") for ds in by_name(datasets, "base", "ntheta_4")]
        with pytest.raises(ValueError, match="time trace"):
            compare.compute_compare_traces(runs, "ntheta", tstart=None, quiet=True)


class TestPlots:
    def test_bundles_record_every_run(self, datasets, tmp_path):
        runs = compare.in_order(by_name(datasets, "base", "ntheta_4", "ntheta_16"), "ntheta")
        figure = compare.plot_compare_theta(runs, "ntheta", output_dir=tmp_path / "out", quiet=True)
        assert figure.name == "compare_ntheta_theta.pdf"

        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b["input_file"].dims == ("run",)
        assert "ntheta = 4" in str(b["input_file"].values[0])
        sources = json.loads(b.attrs["gs2ew_sources"])
        assert [s["source_path"].split("/")[-1] for s in sources] == [
            "ntheta_4.out.nc", "base.out.nc", "ntheta_16.out.nc"
        ]
        assert b.attrs["gs2_version"] == "8.2.1-2129-g5f2ef277b"
        assert "source_path" not in b.attrs
        assert sorted(p.name for p in figure.parent.glob("*.in")) == [
            "base.in", "ntheta_16.in", "ntheta_4.in"
        ]
        script = (figure.parent / bundle.REPRODUCE_SCRIPT).read_text()
        assert script.count("#   source: ") == 3

    def test_replot_and_reproduce(self, datasets, tmp_path):
        runs = by_name(datasets, "ntheta_4", "base", "ntheta_16")
        figure = compare.plot_compare_scalars(runs, "ntheta", output_dir=tmp_path / "out", quiet=True)
        path = figure.with_suffix(".nc")

        replotted = bundle.replot(path, show_std=False, output_dir=tmp_path / "re", quiet=True)
        assert replotted.exists()
        assert sorted(p.name for p in replotted.parent.glob("*.in")) == [
            "base.in", "ntheta_16.in", "ntheta_4.in"
        ]

        again = bundle.reproduce(path, quiet=True)
        assert bundle.data_differences(path, again.with_suffix(".nc")) == {}
        with pytest.raises(ValueError, match="several runs"):
            bundle.reproduce(path, run=tmp_path, quiet=True)

    def test_reproduce_checks_every_run(self, datasets, scan_dir, tmp_path):
        runs = by_name(datasets, "base", "ntheta_4")
        figure = compare.plot_compare_scalars(runs, "ntheta", output_dir=tmp_path / "out", quiet=True)
        for ds in datasets:
            ds.close()
        target = scan_dir / "ntheta_4"
        (target / "ntheta_4.out.nc").unlink()
        make_run(scan_dir, "other", ntheta=4, grid=5, seed=9)
        (scan_dir / "other" / "other.out.nc").rename(target / "ntheta_4.out.nc")
        with pytest.raises(bundle.SourceMismatchError, match="ntheta_4.out.nc"):
            bundle.reproduce(figure.with_suffix(".nc"), quiet=True)

    def test_one_field_goes_two_to_a_row(self, tmp_path):
        root = tmp_path / "es"
        for i, ntheta in enumerate((8, 16)):
            make_run(root, f"ntheta_{ntheta}", ntheta=ntheta, fields=("phi",), split=False, seed=i)
        with ExitStack() as stack:
            runs = [stack.enter_context(load_gs2_output(r.out_nc)) for r in compare.find_runs([root])]
            reduced = compare.compute_compare_theta(runs, "ntheta", quiet=True)
        fig = compare.render_compare_theta(reduced)
        shown = [ax for ax in fig.axes if ax.get_visible()]
        assert len(shown) == 4  # U, H, S and KE
        assert fig.axes[0].get_gridspec().ncols == 2
        # Wrapped, not a column of one field: every panel has its own axis.
        assert all(ax.get_xlabel() == r"$\theta/\pi$" for ax in shown)


def test_cli(scan_dir, capsys):
    assert cli.main(["compare", str(scan_dir), "--vary", "ntheta"]) == 0
    out = capsys.readouterr().out
    assert "Comparing 3 runs by ntheta: ntheta_4, base, ntheta_16" in out
    assert "Leaving out, as they differ in more than ntheta: delt_0p1" in out
    written = {p.name for p in (scan_dir / "gs2ew").iterdir()}
    assert {
        "compare_ntheta_traces.pdf",
        "compare_ntheta_spectra.pdf",
        "compare_ntheta_scalars.pdf",
        "compare_ntheta_theta.nc",
    } <= written


def test_cli_bad_key(scan_dir):
    with pytest.raises(SystemExit, match="not in the input deck of any run"):
        cli.main(["compare", str(scan_dir), "--vary", "negrid"])
