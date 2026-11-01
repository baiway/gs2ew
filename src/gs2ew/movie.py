"""Rendering plot sequences as movies.

Every movie in `gs2ew` is the same shape: pick a set of time indices, render
one figure per index into a frames directory, then hand the PNGs to ffmpeg.
Only the per-frame rendering differs, so that is the one thing
:func:`render_movie` takes as an argument.

Frames are kept on disk next to the movie rather than held in memory, which
also makes it easy to inspect or re-encode an individual frame afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from gs2ew.style import figure_options

#: Frames are drawn at the figure's printed size, about 6 in wide by default,
#: which this resolution makes about 1200 pixels.
FRAME_DPI = 200

#: Signature of a per-frame renderer: given the frame's output path and the
#: time index it should show, draw and save exactly one figure.
FrameRenderer = Callable[[Path, int], None]


def render_movie(
    render_frame: FrameRenderer,
    indices: np.ndarray,
    output_path: str | Path,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
    keep_frames: bool = True,
) -> Path:
    """Render one frame per entry of `indices` and encode them as an MP4.

    Parameters
    ----------
    render_frame : callable
        Called as ``render_frame(frame_path, t_index)`` for each frame. Must
        save a PNG at `frame_path`. Every frame must come out at identical
        pixel dimensions, so frame renderers save at the size the figure was
        drawn, rather than cropped to a tight bounding box.
    indices : numpy.ndarray
        Time indices to render, typically from
        :func:`gs2ew.timebase.frame_indices`.
    output_path : str or Path
        Path of the `.mp4` to write. Frames are written to a
        `<stem>_frames/` directory alongside it.
    fps : int, optional
        Frames per second. Default 10.
    crf : int, optional
        libx264 Constant Rate Factor (0 = lossless, 51 = worst). Default 18,
        which is visually near-lossless for line plots and much smaller than
        libx264's default of 23.
    verbose : bool, optional
        Print per-frame progress. Default False.
    keep_frames : bool, optional
        If False, delete the frames directory after encoding. Default True.

    Returns
    -------
    Path
        Path to the encoded movie.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_dir = output_path.parent / f"{output_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Stale frames from a previous, longer run would otherwise be picked up by
    # the glob below and appended to the end of the new movie.
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    n_frames = len(indices)
    if verbose:
        print(f"Rendering {n_frames} frames to {frames_dir}...")

    with figure_options(dpi=FRAME_DPI):
        for frame_num, t_index in enumerate(indices):
            frame_path = frames_dir / f"frame_{frame_num:06d}.png"
            render_frame(frame_path, int(t_index))
            if verbose:
                print(f"  frame {frame_num + 1}/{n_frames}", flush=True)

    _encode(frames_dir, output_path, fps=fps, crf=crf)

    if not keep_frames:
        for frame in frames_dir.glob("frame_*.png"):
            frame.unlink()
        frames_dir.rmdir()

    print(f"Saved {output_path}")
    return output_path


#: H.264 encodes in 16x16 macroblocks, so frame dimensions must be multiples
#: of 16. Matplotlib figure sizes rarely are.
MACRO_BLOCK = 16


def _pad_to_macroblock(frame: np.ndarray) -> np.ndarray:
    """Pad a frame with white up to the next multiple of :data:`MACRO_BLOCK`.

    Left to itself, imageio *resizes* frames to fit the macroblock grid, which
    resamples every pixel of an otherwise pixel-exact plot. Padding adds a few
    blank rows and columns instead and leaves the rendered content untouched.
    """
    height, width = frame.shape[:2]
    pad_h = -height % MACRO_BLOCK
    pad_w = -width % MACRO_BLOCK
    if pad_h == 0 and pad_w == 0:
        return frame

    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (frame.ndim - 2)
    # Matplotlib saves on a white canvas, so white padding is invisible.
    return np.pad(frame, pad_width, mode="constant", constant_values=255)


def _encode(frames_dir: Path, output_path: Path, fps: int, crf: int) -> None:
    """Stitch the PNGs in `frames_dir` into an H.264 MP4."""
    # The v2 API is the one with the ffmpeg writer plugin; v3's imread has
    # different semantics and warns if reached through the v2-style import.
    import imageio.v2 as imageio

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        # H.264 is universally supported by browsers, players and OSes.
        codec="libx264",
        # Required for H.264 compatibility: without it libx264 defaults to
        # yuv444p, which most players (QuickTime included) cannot decode.
        pixelformat="yuv420p",
        # Frames are padded to the macroblock grid below, so imageio does not
        # need to resize them.
        macro_block_size=1,
        # preset=slow buys better compression at the same quality for more CPU
        # time, a good trade when frame rendering dominates the runtime anyway.
        output_params=["-crf", str(crf), "-preset", "slow"],
    ) as writer:
        for frame_path in sorted(frames_dir.glob("frame_*.png")):
            writer.append_data(_pad_to_macroblock(imageio.imread(str(frame_path))))


__all__ = ["FrameRenderer", "render_movie"]
