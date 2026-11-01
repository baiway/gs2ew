"""The command-line interface, run against synthetic run directories."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from gs2ew import bundle, cli, provenance
from gs2ew.provenance import CodeVersion

from conftest import SYNTHETIC_DECK, make_dataset

PILOT = ["--only", "transfer_theta", "fields_by_mode"]


def make_run(directory, name="cbc", **kwargs):
    """A run directory holding a synthetic output and its input deck."""
    directory.mkdir(parents=True, exist_ok=True)
    make_dataset(**kwargs).to_netcdf(directory / f"{name}.out.nc")
    (directory / f"{name}.in").write_text(SYNTHETIC_DECK)
    return directory


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "run")


@pytest.fixture
def clean_gs2ew(monkeypatch):
    """Pretend gs2ew is a clean checkout, whatever state this one is in."""
    monkeypatch.setattr(
        provenance,
        "gs2ew_version",
        lambda: CodeVersion("gs2ew", provenance.GS2EW_REPOSITORY, "0.1", "abc123", "main", False),
    )


class TestPlot:
    def test_default_output_dir(self, run):
        assert cli.main(["plot", str(run), *PILOT]) == 0
        out = run / "gs2ew"
        assert sorted(p.name for p in out.glob("*.png")) == [
            "fields_by_mode.png",
            "transfer_theta_averaged.png",
        ]
        assert (out / "transfer_theta_averaged.nc").exists()
        assert (out / "cbc.in").read_text() == SYNTHETIC_DECK
        assert (out / bundle.REPRODUCE_SCRIPT).exists()

    def test_output_file_and_dir(self, run, tmp_path):
        out = tmp_path / "elsewhere"
        cli.main(["plot", str(run / "cbc.out.nc"), "-o", str(out), *PILOT])
        assert (out / "transfer_theta_averaged.png").exists()

    def test_only(self, run):
        cli.main(["plot", str(run), "--only", "fields_by_mode"])
        assert [p.name for p in (run / "gs2ew").glob("*.png")] == ["fields_by_mode.png"]

    def test_window_options_reach_the_recipe(self, run):
        cli.main(["plot", str(run), "--tstart", "40", "--window", "20", *PILOT])
        b = bundle.read_bundle(run / "gs2ew" / "transfer_theta_averaged.nc")
        compute = bundle.recipe(b)["compute"]
        assert compute["tstart"] == 40.0 and compute["window"] == 20.0
        assert compute["show_std"] is True

    def test_no_data(self, run):
        cli.main(["plot", str(run), "--no-data", *PILOT])
        assert not list((run / "gs2ew").glob("*.nc"))

    def test_ambiguous_run_dir(self, run):
        make_dataset().to_netcdf(run / "other.out.nc")
        with pytest.raises(SystemExit, match="holds 2"):
            cli.main(["plot", str(run), *PILOT])

    def test_several_outputs_get_a_directory_each(self, run):
        make_dataset().to_netcdf(run / "other.out.nc")
        cli.main(["plot", str(run / "other.out.nc"), *PILOT])
        assert (run / "gs2ew" / "other" / "fields_by_mode.png").exists()


class TestRequireClean:
    def test_clean(self, run, clean_gs2ew):
        assert cli.main(["plot", str(run), "--require-clean", *PILOT]) == 0

    def test_dirty_gs2(self, tmp_path, clean_gs2ew, capsys):
        run = make_run(tmp_path / "run", gs2_state="modified")
        assert cli.main(["plot", str(run), "--require-clean", *PILOT]) == 1
        assert "gs2 8.2.1-2129-g5f2ef277b" in capsys.readouterr().out
        assert not (run / "gs2ew").exists()

    def test_unknown_gs2_state(self, tmp_path, clean_gs2ew):
        run = make_run(tmp_path / "run", metadata=False)
        assert cli.main(["plot", str(run), "--require-clean", *PILOT]) == 1

    def test_dirty_gs2ew(self, run, monkeypatch, capsys):
        monkeypatch.setattr(
            provenance,
            "gs2ew_version",
            lambda: CodeVersion("gs2ew", provenance.GS2EW_REPOSITORY, dirty=True),
        )
        assert cli.main(["plot", str(run), "--require-clean", *PILOT]) == 1
        assert "uncommitted changes" in capsys.readouterr().out
        assert not (run / "gs2ew").exists()


class TestBatch:
    def test_each_run_gets_its_own_output(self, tmp_path):
        make_run(tmp_path / "scan" / "a")
        make_run(tmp_path / "scan" / "b")
        assert cli.main(["batch", str(tmp_path / "scan"), *PILOT]) == 0
        for name in ("a", "b"):
            assert (tmp_path / "scan" / name / "gs2ew" / "fields_by_mode.nc").exists()

    def test_dirty_run_is_a_failure(self, tmp_path, clean_gs2ew):
        make_run(tmp_path / "scan" / "a")
        make_run(tmp_path / "scan" / "b", gs2_state="modified")
        assert cli.main(["batch", str(tmp_path / "scan"), "--require-clean", *PILOT]) == 1
        assert (tmp_path / "scan" / "a" / "gs2ew").exists()
        assert not (tmp_path / "scan" / "b" / "gs2ew").exists()


class TestReplotAndReproduce:
    def test_replot_with_override(self, run):
        cli.main(["plot", str(run), *PILOT])
        nc = run / "gs2ew" / "transfer_theta_averaged.nc"
        out = run / "replotted"
        assert cli.main(
            ["replot", str(nc), "-o", str(out), "--set", "normalise=true",
             "--set", "ylim=[-1.1, 1.1]"]
        ) == 0
        render = bundle.recipe(bundle.read_bundle(out / nc.name))["render"]
        assert render["normalise"] is True and render["ylim"] == [-1.1, 1.1]

    def test_unknown_override(self, run):
        cli.main(["plot", str(run), *PILOT])
        nc = run / "gs2ew" / "fields_by_mode.nc"
        with pytest.raises(SystemExit, match="fields_by_mode has no render option normalise"):
            cli.main(["replot", str(nc), "--set", "normalise=true"])

    def test_bad_override(self, run):
        with pytest.raises(SystemExit):
            cli.main(["replot", "x.nc", "--set", "normalise"])

    def test_reproduce(self, run, capsys):
        cli.main(["plot", str(run), *PILOT])
        nc = run / "gs2ew" / "transfer_theta_averaged.nc"
        assert cli.main(["reproduce", str(nc)]) == 0
        assert "identical to the original" in capsys.readouterr().out
        assert (run / "gs2ew" / bundle.REPRODUCED_DIR / nc.name).exists()

    def test_reproduce_reports_differences(self, run, tmp_path, capsys):
        cli.main(["plot", str(run), *PILOT])
        nc = run / "gs2ew" / "fields_by_mode.nc"
        # Same run UUID, different numbers: as if the output had been altered.
        moved = make_run(tmp_path / "moved", seed=1)
        assert cli.main(["reproduce", str(nc), "--run", str(moved)]) == 1
        assert "phi2_by_kx" in capsys.readouterr().out


def test_info_takes_a_run_directory(run, capsys):
    assert cli.main(["info", str(run)]) == 0
    assert "Identity checks" in capsys.readouterr().out
