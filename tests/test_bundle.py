"""Plot bundles: writing, reading, replotting and reproducing."""

from __future__ import annotations

import json
import stat

import matplotlib
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from gs2ew import bundle, provenance, style
from gs2ew.bundle import PlotSpec

from conftest import SYNTHETIC_DECK, make_dataset


def _compute(ds, tstart=None, scale=1.0, quiet=False):
    """Mean kinetic-energy transfer profile over t >= tstart."""
    da = ds["kinetic_energy_transfer_theta"]
    if tstart is not None:
        da = da.sel(t=slice(tstart, None))
    mean = scale * da.mean("t")
    mean.attrs = {"units": "GS2 units", "cell_methods": "t: mean"}
    return xr.Dataset(
        {"profile": mean},
        attrs={"tstart": float(da["t"][0]), "averaging": True, "unset": None},
    )


def _render(reduced, normalise=False, figsize=(4.0, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    values = reduced["profile"].values
    if normalise:
        values = values / abs(values).max()
    ax.plot(reduced["theta"].values, values)
    return fig


@pytest.fixture
def spec():
    """A minimal plot, registered for the duration of one test."""
    spec = bundle.register(
        PlotSpec("test_profile", _compute, _render, lambda reduced: "profile")
    )
    yield spec
    del bundle.PLOTS["test_profile"]


@pytest.fixture
def run_dir(tmp_path):
    """A run directory holding a synthetic GS2 output and its input deck."""
    run = tmp_path / "run"
    run.mkdir()
    make_dataset().to_netcdf(run / "cbc.out.nc")
    (run / "cbc.in").write_text(SYNTHETIC_DECK)
    return run


@pytest.fixture
def made(spec, run_dir, tmp_path):
    """A plot made through the full pipeline: (figure path, bundle path)."""
    out = tmp_path / "out"
    with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
        figure = bundle.make_plot(
            "test_profile",
            ds,
            {"tstart": 50.0, "scale": 2.0},
            {"figsize": (5, 3)},
            output_dir=out,
            quiet=True,
        )
    return figure, figure.with_suffix(".nc")


class TestMakePlot:
    def test_writes_figure_and_bundle(self, made):
        figure, nc = made
        assert figure.name == "profile.pdf" and figure.exists()
        assert figure.with_suffix(".png").exists()
        assert nc.exists()

    def test_pdf_has_no_creation_date(self, made):
        assert b"/CreationDate" not in made[0].read_bytes()

    def test_suffix_picks_one_format(self, spec, run_dir, tmp_path):
        out = tmp_path / "out"
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=out, filename="p.png", quiet=True
            )
        assert sorted(p.name for p in out.glob("p.*")) == ["p.nc", "p.png"]
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["save"]["filename"] == "p"
        assert rec["figure"]["formats"] == ["png"]

    def test_figure_options(self, spec, run_dir, tmp_path):
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds, style.figure_options(
            formats=["svg"], dpi=72
        ):
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path, quiet=True
            )
        assert figure.name == "profile.svg"
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["figure"]["formats"] == ["svg"] and rec["figure"]["dpi"] == 72

    def test_bundle_holds_the_reduced_data(self, made):
        b = bundle.read_bundle(made[1])
        assert b["profile"].dims == ("theta",)
        assert b["profile"].attrs["cell_methods"] == "t: mean"
        assert b["profile"].attrs["units"] == "GS2 units"

    def test_attrs_made_netcdf_safe(self, made):
        attrs = bundle.read_bundle(made[1]).attrs
        assert attrs["averaging"] == "true"
        assert "unset" not in attrs

    def test_recipe(self, made):
        rec = bundle.recipe(bundle.read_bundle(made[1]))
        assert rec["plot"] == "test_profile"
        assert rec["compute"] == {"tstart": 50.0, "scale": 2.0}
        assert rec["render"] == {"figsize": [5, 3]}
        assert rec["figure"] == {
            "style": "print",
            "width": None,
            "title": True,
            "formats": ["pdf", "png"],
            "dpi": 300,
        }
        assert rec["save"] == {"filename": "profile", "tight_bbox": False}

    def test_provenance(self, made, run_dir):
        attrs = bundle.read_bundle(made[1]).attrs
        assert attrs["gs2_version"] == "8.2.1-2129-g5f2ef277b"
        assert attrs["gs2_dirty"] == "false"
        assert attrs["gs2ew_commit"] == provenance.gs2ew_version().commit
        assert attrs["gs2ew_dirty"] in ("true", "false")
        assert attrs["source_path"] == str((run_dir / "cbc.out.nc").resolve())
        assert attrs["source_id"] == "00000000-0000-4000-8000-000000000000"
        assert "numpy" in json.loads(attrs["dependency_versions"])

    def test_input_decks(self, made):
        b = bundle.read_bundle(made[1])
        assert b["input_file"].item() == SYNTHETIC_DECK
        assert "layout = 'lxyes'" in b["used_input_file"].item()
        assert b.attrs["input_file_origin"].startswith("cbc.in from the run directory")

    def test_commands(self, made):
        attrs = bundle.read_bundle(made[1]).attrs
        commit = provenance.gs2ew_version().commit
        assert attrs["replot_command"] == "gs2ew replot profile.nc"
        assert attrs["reproduce_command"] == (
            f"uvx --from git+https://github.com/baiway/gs2ew@{commit} "
            "gs2ew reproduce profile.nc"
        )

    def test_no_data(self, spec, ds, tmp_path):
        figure = bundle.make_plot(
            "test_profile", ds, {}, output_dir=tmp_path, save_data=False, quiet=True
        )
        assert figure.exists()
        assert not figure.with_suffix(".nc").exists()

    def test_unknown_plot(self, ds, tmp_path):
        with pytest.raises(ValueError, match="Unknown plot"):
            bundle.make_plot("no_such_plot", ds, {}, output_dir=tmp_path)


class TestRunFiles:
    def test_input_deck_written(self, made):
        assert (made[0].parent / "cbc.in").read_text() == SYNTHETIC_DECK

    def test_reproduce_script(self, made):
        script = made[0].parent / bundle.REPRODUCE_SCRIPT
        assert script.stat().st_mode & stat.S_IXUSR
        text = script.read_text()
        assert "# profile.nc: test_profile" in text
        assert "gs2ew reproduce profile.nc" in text
        assert "gs2 8.2.1-2129-g5f2ef277b" in text

    def test_script_lists_every_bundle(self, spec, made, run_dir):
        out = made[0].parent
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
            bundle.make_plot(
                "test_profile", ds, {}, output_dir=out, filename="other.png", quiet=True
            )
        text = (out / bundle.REPRODUCE_SCRIPT).read_text()
        assert "gs2ew reproduce profile.nc" in text
        assert "gs2ew reproduce other.nc" in text

    def test_dirty_gs2_is_flagged(self, spec, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        make_dataset(gs2_state="modified").to_netcdf(run / "x.out.nc")
        with xr.open_dataset(run / "x.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path / "out", quiet=True
            )
        text = (figure.parent / bundle.REPRODUCE_SCRIPT).read_text()
        assert "WARNING: gs2 had uncommitted changes" in text

    def test_matching_deck_copied_verbatim(self, spec, run_dir, tmp_path):
        """GS2's embedded copy loses tabs and trailing spaces; the .in doesn't."""
        original = SYNTHETIC_DECK.replace(" nstep", "\tnstep") + "   \n"
        (run_dir / "cbc.in").write_text(original)
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path / "out", quiet=True
            )
        assert (figure.parent / "cbc.in").read_text() == original
        assert bundle.read_bundle(figure.with_suffix(".nc"))["input_file"].item() == original

    def test_embedded_deck_without_a_file_on_disk(self, spec, run_dir, tmp_path):
        (run_dir / "cbc.in").unlink()
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path / "out", quiet=True
            )
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b["input_file"].item() == SYNTHETIC_DECK
        assert b.attrs["input_file_origin"] == "embedded in the GS2 output"

    def test_deck_falls_back_to_run_directory(self, spec, tmp_path):
        """Old GS2 output has no embedded deck; use the .in, marked unverified."""
        run = tmp_path / "run"
        run.mkdir()
        make_dataset(metadata=False).to_netcdf(run / "old.out.nc")
        (run / "old.in").write_text("&knobs\n/\n")
        with xr.open_dataset(run / "old.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path / "out", quiet=True
            )
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b["input_file"].item() == "&knobs\n/\n"
        assert "unverified" in b.attrs["input_file_origin"]
        assert (figure.parent / "old.in").exists()

    def test_edited_deck_on_disk_warns(self, spec, run_dir, tmp_path, capsys):
        (run_dir / "cbc.in").write_text(SYNTHETIC_DECK.replace("1000", "2000"))
        with xr.open_dataset(run_dir / "cbc.out.nc") as ds:
            figure = bundle.make_plot(
                "test_profile", ds, {}, output_dir=tmp_path / "out", quiet=True
            )
        assert "differs from the input deck embedded" in capsys.readouterr().out
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert b["input_file"].item() == SYNTHETIC_DECK


class TestReplot:
    def test_identical_figure(self, made, tmp_path):
        figure, nc = made
        original = figure.read_bytes()
        figure.unlink()
        assert bundle.replot(nc, quiet=True).read_bytes() == original

    def test_overrides_write_a_matching_bundle(self, made, tmp_path):
        _, nc = made
        out = tmp_path / "replotted"
        figure = bundle.replot(nc, output_dir=out, normalise=True, quiet=True)
        rec = bundle.recipe(bundle.read_bundle(figure.with_suffix(".nc")))
        assert rec["render"] == {"figsize": [5, 3], "normalise": True}
        # The original is untouched.
        assert bundle.recipe(bundle.read_bundle(nc))["render"] == {"figsize": [5, 3]}
        # The input deck comes along to the new directory.
        assert (out / "cbc.in").read_text() == SYNTHETIC_DECK

    def test_unknown_override_names_the_options(self, made):
        _, nc = made
        with pytest.raises(ValueError, match="no render option klim; it has figsize, normalise"):
            bundle.replot(nc, klim=0.5, quiet=True)

    def test_new_bundle_listed_in_reproduce_script(self, made):
        _, nc = made
        bundle.replot(nc, filename="linear.png", normalise=True, quiet=True)
        text = (nc.parent / bundle.REPRODUCE_SCRIPT).read_text()
        assert "# linear.nc: test_profile" in text

    def test_restyle_from_a_style_file(self, made, tmp_path):
        _, nc = made
        style_file = tmp_path / "journal.mplstyle"
        style_file.write_text("axes.grid: False\nfont.size: 8\n")
        figure = bundle.replot(
            nc,
            output_dir=tmp_path / "journal",
            style=str(style_file),
            formats="pdf",
            quiet=True,
        )
        assert sorted(p.name for p in figure.parent.glob("profile.*")) == [
            "profile.nc",
            "profile.pdf",
        ]
        b = bundle.read_bundle(figure.with_suffix(".nc"))
        assert bundle.recipe(b)["figure"]["style"] == str(style_file)
        assert b.attrs["gs2ew_style_file"] == style_file.read_text()

        # The bundle keeps the style, so the figure survives the file moving.
        original = figure.read_bytes()
        style_file.unlink()
        again = bundle.replot(
            figure.with_suffix(".nc"), output_dir=tmp_path / "again", quiet=True
        )
        assert again.read_bytes() == original

    def test_unknown_style(self, made):
        with pytest.raises(ValueError, match="Unknown style 'nope'"):
            bundle.replot(made[1], style="nope", quiet=True)

    def test_bundle_from_before_figure_options(self, made, tmp_path):
        """Old recipes name one PNG and its resolution in `save`."""
        _, nc = made
        b = bundle.read_bundle(nc)
        rec = bundle.recipe(b)
        del rec["figure"]
        rec["save"] = {"filename": "profile.png", "dpi": 150, "tight_bbox": True}
        b.attrs["gs2ew_recipe"] = json.dumps(rec)
        old = tmp_path / "old" / "profile.nc"
        old.parent.mkdir()
        b.to_netcdf(old)

        figure = bundle.replot(old, quiet=True)
        assert figure == old.with_suffix(".png")
        assert not old.with_suffix(".pdf").exists()

    def test_rejects_other_netcdf(self, ds, tmp_path):
        path = tmp_path / "not_a_bundle.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="not a gs2ew plot bundle"):
            bundle.replot(path)


class TestReproduce:
    def test_identical_result(self, made):
        figure, nc = made
        reproduced = bundle.reproduce(nc, quiet=True)
        assert reproduced.parent == figure.parent / bundle.REPRODUCED_DIR
        assert reproduced.read_bytes() == figure.read_bytes()
        # The variable, not the Dataset, whose `created` time differs.
        xr.testing.assert_identical(
            bundle.read_bundle(reproduced.with_suffix(".nc"))["profile"],
            bundle.read_bundle(nc)["profile"],
        )

    def test_moved_run(self, made, run_dir, tmp_path):
        moved = run_dir.rename(tmp_path / "moved")
        assert bundle.reproduce(made[1], run=moved, quiet=True).exists()

    def test_different_run_refused(self, made, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        different = make_dataset()
        different.attrs["id"] = "11111111-1111-4111-8111-111111111111"
        different.to_netcdf(other / "cbc.out.nc")
        with pytest.raises(bundle.SourceMismatchError, match="11111111"):
            bundle.reproduce(made[1], run=other, quiet=True)

    def test_unrecorded_source_needs_a_run(self, spec, ds, tmp_path):
        figure = bundle.make_plot("test_profile", ds, {}, output_dir=tmp_path, quiet=True)
        with pytest.raises(ValueError, match="pass the run directory"):
            bundle.reproduce(figure.with_suffix(".nc"))


def test_reproduce_command_without_a_known_commit():
    unknown = provenance.CodeVersion("gs2ew", provenance.GS2EW_REPOSITORY)
    assert bundle.reproduce_command("a.nc", unknown) == "gs2ew reproduce a.nc"


def test_jsonable():
    import numpy as np
    from pathlib import Path

    assert bundle._jsonable({"a": (1, np.float64(2.5)), "p": Path("x")}) == {
        "a": [1, 2.5],
        "p": "x",
    }
