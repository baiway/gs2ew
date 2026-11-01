"""How gs2ew figures look, as opposed to what they show.

Every figure is drawn inside :func:`context`, which starts from matplotlib's
defaults, applies gs2ew's ``print`` style, and then any style the caller
names. So a figure never depends on the local ``matplotlibrc``, and a style
of your own only needs to set what it changes. The ``print`` style draws each
figure at the size it is printed (6 in wide by default), and plots use
constrained layout and save without cropping, so the saved figure is exactly
that wide.

The options are part of each bundle's recipe, so a figure can be restyled
from its bundle alone::

    gs2ew replot fig.nc --set style=journal.mplstyle --set width=3.4

From Python, :func:`figure_options` sets them for a block of plots::

    with figure_options(style="journal.mplstyle", formats=["pdf"]):
        plot_transfer_theta(ds)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc_params_from_file

STYLES_DIR = Path(__file__).parent / "styles"

#: The style every figure starts from.
BASE_STYLE = "print"

#: Vector for print, raster for viewing.
DEFAULT_FORMATS = ("pdf", "png")

#: Resolution of PNGs, and of the rasterised heatmaps inside a PDF.
DEFAULT_DPI = 300


@dataclass(frozen=True)
class FigureOptions:
    """How a figure is drawn and saved.

    `style` is a gs2ew style name, a matplotlib style name, or the path to an
    ``.mplstyle`` file, applied on top of the ``print`` style. `width` is the
    figure width in inches (default: the style's ``figure.figsize``). `title`
    keeps the title naming the averaging window; leave it to the caption in a
    thesis or paper. `formats` are the file types to save; `dpi` is the
    resolution of PNGs and of heatmaps inside PDFs.
    """

    style: str = BASE_STYLE
    width: float | None = None
    title: bool = True
    formats: tuple[str, ...] = DEFAULT_FORMATS
    dpi: int = DEFAULT_DPI

    def __post_init__(self) -> None:
        formats = self.formats
        if isinstance(formats, str):
            formats = formats.split(",")
        object.__setattr__(
            self, "formats", tuple(f.strip().lstrip(".").lower() for f in formats)
        )
        object.__setattr__(self, "style", str(self.style))

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


_current: ContextVar[FigureOptions] = ContextVar(
    "gs2ew_figure_options", default=FigureOptions()  # noqa: B039 (frozen)
)


def current() -> FigureOptions:
    """The options gs2ew figures are drawn with here."""
    return _current.get()


@contextmanager
def figure_options(**changes) -> Iterator[FigureOptions]:
    """Draw and save every gs2ew figure in the block with `changes` applied.

    `changes` are fields of :class:`FigureOptions`; None leaves one as it is.
    """
    options = replace(current(), **{k: v for k, v in changes.items() if v is not None})
    token = _current.set(options)
    try:
        yield options
    finally:
        _current.reset(token)


def builtin_styles() -> list[str]:
    """Names of the styles shipped with gs2ew."""
    return sorted(p.stem for p in STYLES_DIR.glob("*.mplstyle"))


def resolve(style: str) -> str:
    """The matplotlib style specification for `style`.

    Raises
    ------
    ValueError
        If `style` is neither a known style nor an existing file.
    """
    if style in builtin_styles():
        return str(STYLES_DIR / f"{style}.mplstyle")
    if Path(style).expanduser().is_file():
        return str(Path(style).expanduser())
    if style in plt.style.available or style == "default":
        return style
    raise ValueError(
        f"Unknown style {style!r}: not a file, nor one of gs2ew's "
        f"({', '.join(builtin_styles())}) or matplotlib's styles."
    )


def style_file(style: str) -> str | None:
    """The text of `style`, if it is a style file of the user's.

    Bundles keep it, so a figure can be redrawn in the same style after the
    file has moved.
    """
    if style in builtin_styles() or not Path(style).expanduser().is_file():
        return None
    return Path(style).expanduser().read_text()


@contextmanager
def context(options: FigureOptions, embedded: str | None = None) -> Iterator[None]:
    """Apply `options`' style and width for the duration of the block.

    `embedded` is the text of a style file kept in a bundle, used in place of
    ``options.style`` so the figure is redrawn as it was first drawn.
    """
    _register_tex_fonts()
    styles = ["default", resolve(BASE_STYLE)]
    if embedded is not None:
        styles.append(_parse(embedded))
    elif options.style != BASE_STYLE:
        styles.append(resolve(options.style))

    with plt.style.context(styles):
        if options.width is not None:
            height = plt.rcParams["figure.figsize"][1]
            plt.rcParams["figure.figsize"] = (float(options.width), height)
        yield


def _parse(text: str) -> dict:
    """The rcParams set by the style file whose text is `text`."""
    with tempfile.NamedTemporaryFile("w", suffix=".mplstyle", delete=False) as f:
        f.write(text)
    try:
        return dict(rc_params_from_file(f.name, use_default_template=False))
    finally:
        Path(f.name).unlink()


#: Where gs2ew keeps its TrueType copies of XCharter. Bump the suffix if
#: :func:`_to_truetype` changes, so stale copies are not reused.
FONT_CACHE = Path(matplotlib.get_cachedir()) / "gs2ew-fonts-1"


@cache
def _register_tex_fonts() -> None:
    """Make TeX Live's XCharter available to matplotlib, if it is installed.

    matplotlib only searches the system's font directories, and TeX Live
    keeps its fonts in its own tree. XCharter is an OpenType font with CFF
    outlines, which matplotlib embeds in a PDF labelled as TrueType, making a
    malformed file that a journal's checks may reject. So it is converted to
    TrueType once, into :data:`FONT_CACHE`.
    """
    if any(f.name == "XCharter" for f in font_manager.fontManager.ttflist):
        return
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return
    try:
        found = subprocess.run(
            [kpsewhich, "XCharter-Roman.otf"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return
    if not found:
        return
    # Not the Slanted faces: matplotlib counts them as italic too, and could
    # set maths in one.
    for face in ("Roman", "Italic", "Bold", "BoldItalic"):
        source = Path(found).with_name(f"XCharter-{face}.otf")
        target = FONT_CACHE / f"XCharter-{face}.ttf"
        if not source.is_file():
            continue
        if not target.is_file():
            FONT_CACHE.mkdir(parents=True, exist_ok=True)
            _to_truetype(source, target)
        font_manager.fontManager.addfont(str(target))


def _to_truetype(source: Path, target: Path) -> None:
    """Convert the CFF-outline OpenType font `source` to TrueType `target`.

    Follows fontTools' ``otf2ttf`` example: the cubic outlines are
    approximated by quadratic ones to within 1 font unit, far below what
    print can show.
    """
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(source)
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    font["loca"] = newTable("loca")
    glyf = font["glyf"] = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(None)
        glyph_set[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf[name] = pen.glyph()
    for tag in ("CFF ", "VORG"):
        if tag in font:
            del font[tag]

    font["head"].glyphDataFormat = 0
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for attr in (
        "maxTwilightPoints",
        "maxStorage",
        "maxFunctionDefs",
        "maxInstructionDefs",
        "maxStackElements",
        "maxSizeOfInstructions",
        "maxComponentElements",
        "maxComponentDepth",
    ):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"

    # Written aside and renamed, so a concurrent plot never reads half a file.
    partial = target.with_suffix(".tmp")
    font.save(partial)
    partial.replace(target)


__all__ = [
    "BASE_STYLE",
    "DEFAULT_DPI",
    "DEFAULT_FORMATS",
    "FigureOptions",
    "builtin_styles",
    "context",
    "current",
    "figure_options",
    "resolve",
    "style_file",
]
