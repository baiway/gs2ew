"""Figure styles and options."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from gs2ew import style
from gs2ew.style import FigureOptions


class TestFigureOptions:
    @pytest.mark.parametrize("formats", ["pdf,png", ["pdf", ".PNG"], ("pdf", "png")])
    def test_formats_normalised(self, formats):
        assert FigureOptions(formats=formats).formats == ("pdf", "png")

    def test_defaults(self):
        assert style.current() == FigureOptions()
        assert style.current().formats == ("pdf", "png")

    def test_figure_options_nest_and_restore(self):
        with style.figure_options(width=3.4):
            with style.figure_options(formats="pdf", dpi=None):
                assert style.current() == FigureOptions(width=3.4, formats=("pdf",))
            assert style.current().formats == ("pdf", "png")
        assert style.current() == FigureOptions()


class TestResolve:
    def test_builtin(self):
        assert "print" in style.builtin_styles()
        assert style.resolve("print").endswith("print.mplstyle")

    def test_file_and_matplotlib_names(self, tmp_path):
        path = tmp_path / "mine.mplstyle"
        path.write_text("font.size: 8\n")
        assert style.resolve(str(path)) == str(path)
        assert style.resolve("ggplot") == "ggplot"

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown style 'nope'"):
            style.resolve("nope")

    def test_only_user_files_are_kept(self, tmp_path):
        path = tmp_path / "mine.mplstyle"
        path.write_text("font.size: 8\n")
        assert style.style_file(str(path)) == "font.size: 8\n"
        assert style.style_file("print") is None
        assert style.style_file("ggplot") is None


class TestContext:
    def test_print_style_ignores_local_rc(self):
        with plt.rc_context({"axes.grid": False, "font.size": 20}):
            with style.context(FigureOptions()):
                assert plt.rcParams["axes.grid"] is True
                assert plt.rcParams["font.size"] == 10
            assert plt.rcParams["font.size"] == 20

    def test_width(self):
        with style.context(FigureOptions(width=3.4)):
            assert plt.rcParams["figure.figsize"][0] == 3.4

    def test_user_style_goes_on_top(self, tmp_path):
        path = tmp_path / "mine.mplstyle"
        path.write_text("font.size: 8\n")
        with style.context(FigureOptions(style=str(path))):
            assert plt.rcParams["font.size"] == 8
            # Everything else is still the print style.
            assert plt.rcParams["axes.grid"] is True

    def test_embedded_style_wins(self, tmp_path):
        with style.context(FigureOptions(style="ggplot"), embedded="font.size: 7\n"):
            assert plt.rcParams["font.size"] == 7


@pytest.mark.skipif(
    not style.shutil.which("kpsewhich"), reason="needs TeX Live for XCharter"
)
def test_xcharter_embedded_as_truetype(tmp_path):
    """XCharter is used as TrueType, so PDFs label its fonts correctly."""
    from matplotlib import font_manager

    with style.context(FigureOptions()):
        path = font_manager.findfont("serif")
    if "XCharter" not in path:
        pytest.skip("XCharter is not installed")
    assert path.endswith(".ttf")
