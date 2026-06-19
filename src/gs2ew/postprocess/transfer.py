"""Create various plots of the nonlinear transfer diagnostics.

GS2 writes three closely related nonlinear-transfer quantities:

* **free energy** ``H`` — ``free_energy_transfer_<field>_{theta,velocity}``
* **entropy** ``S``     — ``entropy_transfer_<field>_{theta,velocity}``
* **kinetic energy**    — ``kinetic_energy_transfer_theta``

where ``<field>`` is one of ``phi``, ``apar`` or ``bpar``. The fluctuation
energy ``U`` is *not* written to file; it is derived here as ``U = H + S``.

Note on naming: what older GS2 versions called the "entropy" transfer is now
the **free energy** transfer (``free_energy_transfer_*``); the
``entropy_transfer_*`` variables are a genuinely separate, newer output. The
nonlinear drives are labelled :math:`N_\\mathbf{k}^{Q,f}` for quantity
``Q`` (H/S/U) and field ``f``.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from gs2ew.utils.gs2_output import detect_saturation_time as _detect_saturation_time
from gs2ew.utils.weights import get_weights


# LaTeX symbols for each field, used to build drive labels.
_FIELD_SYMS = {
    "phi": r"\phi",
    "apar": r"A_\parallel",
    "bpar": r"B_\parallel",
}

# Stored nonlinear-drive quantities and their label symbols. Fluctuation
# energy U is not stored; it is derived as U = H + S where both are present.
_STORED_QUANTITIES = (
    ("free_energy", "H"),
    ("entropy", "S"),
)

# Label for the kinetic-energy transfer (no field/velocity decomposition).
_KINETIC_LABEL = r"$T_v^\text{ZF}$"


def _drive_label(quantity_sym: str, field: str) -> str:
    """LaTeX label for a nonlinear drive, e.g. ``$N_\\mathbf{k}^{H,\\phi}$``.

    Fluctuation energy U gets an expanded label spelling out U = H + sum_s T_s S.
    """
    sym = _FIELD_SYMS[field]
    if quantity_sym == "U":
        return (
            rf"$U_\mathbf{{k}} = H_\mathbf{{k}} + \sum_s T_s S_{{s\mathbf{{k}}}}, "
            rf"\, N_\mathbf{{k}}^U$ (${sym}$)"
        )
    return rf"$N_\mathbf{{k}}^{{{quantity_sym},{sym}}}$"


def _global_theta_ylim(
    ds: xr.Dataset,
    frame_indices: np.ndarray,
    transfer_sign: int,
) -> tuple[float, float]:
    """Return padded (ymin, ymax) spanning every theta transfer curve over the
    given frames, so a movie's y-axis stays fixed instead of jumping.

    Uses instantaneous values at each frame's start index; for windowed movies
    these bound the (averaged) frame values, so the axis is never too small.
    """
    series: list[np.ndarray] = []

    if "kinetic_energy_transfer_theta" in ds:
        series.append(ds["kinetic_energy_transfer_theta"].isel(t=frame_indices).values)

    for field in _FIELD_SYMS:
        h_name = f"free_energy_transfer_{field}_theta"
        s_name = f"entropy_transfer_{field}_theta"
        H = ds[h_name].isel(t=frame_indices).values if h_name in ds else None
        S = ds[s_name].isel(t=frame_indices).values if s_name in ds else None
        if H is not None:
            series.append(transfer_sign * H)
        if S is not None:
            series.append(transfer_sign * S)
        if H is not None and S is not None:
            series.append(transfer_sign * (H + S))

    allv = np.concatenate([s.ravel() for s in series])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def _movie_norm_factors(
    ds: xr.Dataset,
    frame_indices: np.ndarray,
) -> dict[str, float]:
    """Per-curve divisors for movie-wide normalisation: each curve's peak
    absolute value over *all* frames, keyed by the curve's plot label.

    Dividing every frame by these fixed factors (rather than each frame's own
    peak) preserves both the relative amplitude between curves and their
    growth/decay over time, while still scaling small terms (e.g. U) into view.
    """
    factors: dict[str, float] = {}

    if "kinetic_energy_transfer_theta" in ds:
        v = ds["kinetic_energy_transfer_theta"].isel(t=frame_indices).values
        factors[_KINETIC_LABEL] = float(np.nanmax(np.abs(v)))

    for field in _FIELD_SYMS:
        h_name = f"free_energy_transfer_{field}_theta"
        s_name = f"entropy_transfer_{field}_theta"
        H = ds[h_name].isel(t=frame_indices).values if h_name in ds else None
        S = ds[s_name].isel(t=frame_indices).values if s_name in ds else None
        if H is not None:
            factors[_drive_label("H", field)] = float(np.nanmax(np.abs(H)))
        if S is not None:
            factors[_drive_label("S", field)] = float(np.nanmax(np.abs(S)))
        if H is not None and S is not None:
            factors[_drive_label("U", field)] = float(np.nanmax(np.abs(H + S)))

    return factors


def _resolve_frame_indices(
    ds: xr.Dataset,
    tstart: float | None,
    window: float | None,
) -> np.ndarray:
    """Resolve `tstart` and return the array of valid frame time indices.

    If `tstart` is None, attempts to detect the saturation time via
    `detect_saturation_time`. Falls back to `t[0]` with a warning if
    detection fails (e.g. `phi2` is absent from `ds`).

    Excludes timesteps before `tstart` and, when `window` is given, any
    timestep where a full window `[t, t + window]` would exceed the data.

    Raises `ValueError` if no valid frames remain after filtering.
    """
    t_values = ds["t"].values

    if tstart is None:
        try:
            tstart = _detect_saturation_time(ds)
            if np.isnan(tstart):
                raise ValueError
            print(f"Using detected saturation time tstart = {tstart:.2f}")
        except (KeyError, ValueError):
            print("Warning: saturation time could not be detected; starting movie from t[0].")
            tstart = float(t_values[0])

    # Build a boolean mask over all timesteps to select valid frame times.
    # Start with all timesteps included, then narrow down.
    mask = np.ones(len(t_values), dtype=bool)

    # Skip frames before the requested start time.
    mask &= t_values >= tstart

    # For averaged frames, only include timesteps where a full window
    # [t, t + window] fits within the data — partial windows are excluded.
    if window is not None:
        mask &= t_values + window <= float(t_values[-1])

    frame_indices = np.nonzero(mask)[0]

    if len(frame_indices) == 0:
        raise ValueError("No valid frames found for the given `tstart` and `window`.")

    return frame_indices


def _stitch_frames_to_movie(
    frames_dir: Path,
    output_path: Path,
    fps: int,
    crf: int,
) -> None:
    """Stitch PNG frames in `frames_dir` into an H.264 MP4 at `output_path`."""
    import imageio

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        # H.264: universally supported by browsers, media players, and OSes.
        codec="libx264",
        # yuv420p is required for H.264 compatibility. Without it, libx264
        # defaults to yuv444p, which most players (including QuickTime) cannot
        # decode. The matplotlib PNGs are RGB; ffmpeg converts them on the fly.
        pixelformat="yuv420p",
        # crf controls quality vs. file size (0 = lossless, 51 = worst).
        # preset=slow gets better compression at the same quality by spending
        # more CPU time — a worthwhile trade since frame generation dominates
        # the overall runtime.
        output_params=["-crf", str(crf), "-preset", "slow"],
    ) as writer:
        for frame_path in sorted(frames_dir.glob("frame_*.png")):
            writer.append_data(imageio.imread(str(frame_path)))


def plot_transfer_by_theta(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    normalise: bool = False,
    norm_factors: dict[str, float] | None = None,
    ylim: tuple[float, float] | None = None,
    tight_bbox: bool = True,
    _quiet: bool = False,
) -> Path:
    """Plots the poloidal structure of each enabled transfer diagnostic.

    Plots the kinetic-energy transfer alongside the free-energy (H), entropy
    (S) and derived fluctuation-energy (U = H + S) nonlinear drives, for each
    field (``phi``, ``apar``, ``bpar``) present in `ds`.

    By default, plots the last time step. If `window` is provided, the
    diagnostics are averaged over a time window instead.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file)
    window : float, optional
        Duration of the averaging window. If provided without `tstart`,
        averages over the last `window` time units. If provided with
        `tstart`, averages from `tstart` over length `window`.
    tstart : float, optional
        Start time for the averaging window. Requires `window` to be set;
        raises ValueError if `window` is not provided.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is "outputs".
    filename : str, optional
        Filename for the plot. If None, uses "transfer_by_theta.png" (or
        "transfer_by_theta_averaged.png" when averaging).
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation. Does not affect
        the kinetic-energy transfer.
    normalise : bool, optional
        If True, divide each curve by its peak absolute value so all curves
        span [-1, 1] and their poloidal structure can be compared regardless of
        amplitude (useful as the fluctuation energy U is typically much smaller
        than the other terms). Default is False.
    norm_factors : dict of str to float, optional
        Per-curve divisors keyed by plot label, used when ``normalise`` is True.
        If given, each curve is divided by its supplied factor instead of its
        own peak; this is how the movie helper applies a fixed, movie-wide
        normalisation. If None, each curve self-normalises to its own peak.
    ylim : tuple of float, optional
        Fixed (ymin, ymax) for the y-axis. Used by the movie helper to keep
        the axis steady across frames; if None, matplotlib autoscales.
    tight_bbox : bool, optional
        If True (default), save with ``bbox_inches="tight"``. Movie frames
        pass False so every frame has identical pixel dimensions (a
        requirement for the ffmpeg encoder).

    Returns
    -------
    Path
        Path to the saved figure file

    Raises
    ------
    ValueError
        If `tstart` is given without `window`, or if no transfer diagnostics
        are present in `ds`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    transfer_sign = -1 if fix_transfer_sign else 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window
        if filename is None:
            filename = "transfer_by_theta_averaged.png"
        title = f"Averaged over t = [{tstart:.1f}, {tend:.1f}]"

        def reduce(name):
            if name not in ds:
                return None
            return ds[name].sel(t=slice(tstart, tend)).mean(dim="t").values
    else:
        if filename is None:
            filename = "transfer_by_theta.png"
        title = None

        def reduce(name):
            if name not in ds:
                return None
            da = ds[name]
            return da.isel(t=-1).values if "t" in da.dims else da.values

    # Assemble curves as (label, values, linestyle). Kinetic energy keeps its
    # native sign; H/S/U are sign-corrected together.
    curves: list[tuple[str, np.ndarray, str]] = []

    ke = reduce("kinetic_energy_transfer_theta")
    if ke is not None:
        curves.append((_KINETIC_LABEL, ke, "-"))

    for field, sym in _FIELD_SYMS.items():
        H = reduce(f"free_energy_transfer_{field}_theta")
        S = reduce(f"entropy_transfer_{field}_theta")
        if H is not None:
            curves.append((_drive_label("H", field), transfer_sign * H, "-"))
        if S is not None:
            curves.append((_drive_label("S", field), transfer_sign * S, "-"))
        if H is not None and S is not None:
            curves.append((_drive_label("U", field), transfer_sign * (H + S), "--"))

    if not curves:
        raise ValueError("No transfer diagnostics found in dataset.")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values, linestyle in curves:
        if normalise:
            # Movie-wide divisor if supplied, else self-normalise to own peak.
            divisor = norm_factors.get(label, 0.0) if norm_factors is not None \
                else np.nanmax(np.abs(values))
            if divisor > 0:
                values = values / divisor
        ax.plot(theta, values, linewidth=1.5, linestyle=linestyle, label=label)

    if title is not None:
        ax.set_title(title)
    ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("normalised transfer" if normalise else "transfer", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)

    if not _quiet:
        print(f"Saved {output_path}")
    return output_path


def plot_transfer_by_theta_movie(
    ds: xr.Dataset,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    normalise: bool | str = False,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Creates a movie of the poloidal structure of each enabled transfer
    diagnostic over time.

    Frames are saved individually to a subdirectory of `output_dir`, then
    stitched into a video with ffmpeg. To avoid exhausting memory on large
    datasets (tens of gigabytes), `ds` should be opened lazily via
    ``xr.open_dataset`` (the default) rather than ``xr.load_dataset``, so
    that only the data required for each frame is read from disk at a time.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file). Must be lazily
        loaded to keep memory usage bounded.
    window : float, optional
        Duration of the averaging window. If provided, each frame shows
        the rolling average over `window` time units starting at that
        frame's time. If omitted, each frame shows the instantaneous
        transfer.
    tstart : float, optional
        Start time for the movie; frames before this time are skipped. If not
        provided, the saturation time is detected automatically via
        ``detect_saturation_time``. If saturation cannot be detected, falls
        back to the first timestep with a warning.
    output_dir : str or Path, optional
        Directory where the movie and frame images are saved.
        Default is "outputs".
    filename : str, optional
        Filename for the output movie. If None, uses
        "transfer_by_theta_movie.mp4".
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation. Does not affect
        the kinetic-energy transfer. Passed through to each frame's plot call.
    normalise : bool or str, optional
        Controls amplitude normalisation; the y-axis is fixed to [-1, 1] when
        enabled. Default is False (off).

        * ``False`` — no normalisation.
        * ``True`` or ``"frame"`` — per-frame: each frame divides every curve
          by *that frame's* own peak. Reveals structure but discards both
          relative amplitude between curves and growth/decay over time.
        * ``"movie"`` — movie-wide: each curve is divided by a single fixed
          divisor (its peak over all frames), so relative amplitude and
          time evolution are preserved while small terms are scaled into view.
    fps : int, optional
        Frames per second for the output movie. Default is 10.
    crf : int, optional
        Constant Rate Factor for libx264 (0 = lossless, 51 = worst quality).
        Lower values give higher quality at the cost of larger files. Default
        is 18, which is visually near-lossless for scientific plots and
        significantly smaller than the libx264 default of 23.
    verbose : bool, optional
        If True, prints the total frame count before rendering begins and
        logs progress after each frame. Default is False.

    Returns
    -------
    Path
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If no valid frames exist for the given parameters.
    """
    frame_indices = _resolve_frame_indices(ds, tstart, window)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "transfer_by_theta_movie.mp4"

    frames_dir = output_dir / (Path(filename).stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_values = ds["t"].values

    # Resolve the normalisation mode.
    norm_mode = ("movie" if normalise == "movie" else "frame") if normalise else None

    # Fix the y-axis across all frames so the curves don't jump around, and
    # save frames at a constant canvas size (tight_bbox=False) as ffmpeg
    # requires every frame to share the same pixel dimensions. When
    # normalising, every curve is bounded by [-1, 1], so use a fixed axis.
    if norm_mode is None:
        frame_normalise = False
        norm_factors = None
        transfer_sign = -1 if fix_transfer_sign else 1
        ylim = _global_theta_ylim(ds, frame_indices, transfer_sign)
    else:
        frame_normalise = True
        # Movie-wide: one fixed divisor per curve; per-frame: self-normalise.
        norm_factors = _movie_norm_factors(ds, frame_indices) if norm_mode == "movie" else None
        ylim = (-1.05, 1.05)

    n_frames = len(frame_indices)
    if verbose:
        print(f"Generating {n_frames} frames...")

    # Produce frames
    for frame_num, t_idx in enumerate(frame_indices):
        frame_filename = f"frame_{frame_num:06d}.png"
        if window is not None:
            plot_transfer_by_theta(
                ds,
                window=window,
                tstart=float(t_values[t_idx]),
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                normalise=frame_normalise,
                norm_factors=norm_factors,
                ylim=ylim,
                tight_bbox=False,
                _quiet=True,
            )
        else:
            # Pass a single-timestep lazy slice; isel(t=-1) inside
            # plot_transfer_by_theta will select the only timestep.
            plot_transfer_by_theta(
                ds.isel(t=slice(t_idx, t_idx + 1)),
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                normalise=frame_normalise,
                norm_factors=norm_factors,
                ylim=ylim,
                tight_bbox=False,
                _quiet=True,
            )
        if verbose:
            print(f"  Frame {frame_num + 1}/{n_frames}")

    # Stitch frames into a video.
    output_path = output_dir / filename
    _stitch_frames_to_movie(frames_dir, output_path, fps=fps, crf=crf)

    print(f"Saved {output_path}")
    return output_path


def plot_vel_transfer_by_theta_by_sign(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    tight_bbox: bool = True,
    _quiet: bool = False,
) -> Path:
    """Plot the poloidal structure of the velocity-resolved nonlinear drives,
    integrated over velocity space.

    Each velocity-resolved free-energy (H) and entropy (S) diagnostic in `ds`
    is weighted by the velocity-space integration weights `wl(lambda, theta)`
    and `w(species, egrid)` before summing over all non-theta dimensions
    except `sign`. The derived fluctuation energy U = H + S is also plotted.
    The two parallel-velocity directions (sign=1: vpa > 0, sign=2: vpa < 0)
    are plotted as separate curves (solid and dashed respectively).

    By default, plots the last time step. If `window` is provided, the
    diagnostics are averaged over a time window instead.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file). Expected to contain
        one or more velocity-resolved diagnostics, each with dims
        `(t, species, sign, lambda, egrid, theta, kxt_shift)`.
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Used to load
        `wl` and `w` if they are absent from `ds`.
    window : float, optional
        Duration of the averaging window. If provided without `tstart`,
        averages over the last `window` time units. If provided with
        `tstart`, averages from `tstart` over length `window`.
    tstart : float, optional
        Start time for the averaging window. Requires `window` to be set;
        raises ValueError if `window` is not provided.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is `"outputs"`.
    filename : str, optional
        Filename for the plot. If None, uses `"vel_transfer_by_theta.png"`
        (or `"vel_transfer_by_theta_averaged.png"` when averaging).
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation.
    tight_bbox : bool, optional
        If True (default), save with ``bbox_inches="tight"``. Movie frames
        pass False so every frame has identical pixel dimensions (a
        requirement for the ffmpeg encoder).

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If `tstart` is given without `window`, or if no velocity-resolved
        diagnostics are present in `ds`.
    MissingWeightsError
        If `wl` and `w` cannot be found in `ds` or `grids_nc`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    transfer_sign = -1 if fix_transfer_sign else 1

    weights = get_weights(ds, grids_nc=grids_nc)
    wl = weights["wl"]  # dims (lambda, theta)
    w = weights["w"]    # dims (species, egrid)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values
    sign_values = ds["sign"].values

    # sign=1 → v_∥ > 0 (solid), sign=2 → v_∥ < 0 (dashed)
    sign_styles = {sign_values[0]: "-", sign_values[1]: "--"}
    sign_labels = {
        sign_values[0]: r"$v_\parallel > 0$",
        sign_values[1]: r"$v_\parallel < 0$",
    }

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window
        if filename is None:
            filename = "vel_transfer_by_theta_averaged.png"
        title = f"Averaged over t = [{tstart:.1f}, {tend:.1f}]"

        def reduce(name):
            """Weighted (sign, theta) profile for `name`, or None if absent."""
            if name not in ds:
                return None
            da = ds[name].sel(t=slice(tstart, tend)).mean(dim="t")
            return (da * wl * w).sum(dim=["species", "lambda", "egrid", "kxt_shift"])
    else:
        if filename is None:
            filename = "vel_transfer_by_theta.png"
        title = None

        def reduce(name):
            if name not in ds:
                return None
            da = ds[name]
            da = da.isel(t=-1) if "t" in da.dims else da
            return (da * wl * w).sum(dim=["species", "lambda", "egrid", "kxt_shift"])

    # Assemble curves as (label, DataArray over (sign, theta)).
    curves: list[tuple[str, xr.DataArray]] = []
    for field, sym in _FIELD_SYMS.items():
        H = reduce(f"free_energy_transfer_{field}_velocity")
        S = reduce(f"entropy_transfer_{field}_velocity")
        if H is not None:
            curves.append((_drive_label("H", field), transfer_sign * H))
        if S is not None:
            curves.append((_drive_label("S", field), transfer_sign * S))
        if H is not None and S is not None:
            curves.append((_drive_label("U", field), transfer_sign * (H + S)))

    if not curves:
        raise ValueError("No velocity-resolved transfer diagnostics found in dataset.")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, transfer in curves:
        for s in sign_values:
            ax.plot(theta, transfer.sel(sign=s).values, linewidth=1.5,
                    linestyle=sign_styles[s],
                    label=f"{label}, {sign_labels[s]}")

    if title is not None:
        ax.set_title(title)
    ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("transfer", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)

    if not _quiet:
        print(f"Saved {output_path}")
    return output_path


def plot_vel_transfer_theta_lambda(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    window: float | None = None,
    tstart: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    tight_bbox: bool = True,
    _quiet: bool = False,
) -> Path:
    """Plot 2D heatmaps of the velocity-resolved nonlinear drives in
    lambda-theta space.

    For each velocity-resolved drive (free energy H, entropy S, and derived
    U = H + S, per field), shows a row of two subplots side by side — one for
    each parallel-velocity sign (sign=1: vpa > 0, sign=2: vpa < 0). The x-axis
    is `theta`, the y-axis is `lambda`. Each cell is weighted by
    `wl(lambda, theta)` and `w(species, egrid)` and summed over `species`,
    `egrid`, and `kxt_shift`, so that integrating along the lambda axis of
    either panel recovers the corresponding curve in
    `plot_vel_transfer_by_theta_by_sign`.

    A symmetric diverging colormap (`RdBu_r`) is used with a per-row shared
    colorbar, so the zero point is always centred.

    By default, plots the last time step. If `window` is provided, the
    diagnostics are averaged over a time window instead.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file). Expected to contain
        one or more velocity-resolved diagnostics, each with dims
        `(t, species, sign, lambda, egrid, theta, kxt_shift)`.
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Used to load
        `wl` and `w` if they are absent from `ds`.
    window : float, optional
        Duration of the averaging window. If provided without `tstart`,
        averages over the last `window` time units. If provided with
        `tstart`, averages from `tstart` over length `window`.
    tstart : float, optional
        Start time for the averaging window. Requires `window` to be set;
        raises ValueError if `window` is not provided.
    output_dir : str or Path, optional
        Directory where the plot will be saved. Default is `"outputs"`.
    filename : str, optional
        Filename for the plot. If None, uses `"vel_transfer_theta_lambda.png"`
        (or `"vel_transfer_theta_lambda_averaged.png"` when averaging).
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation.
    tight_bbox : bool, optional
        If True (default), save with ``bbox_inches="tight"``. Movie frames
        pass False so every frame has identical pixel dimensions (a
        requirement for the ffmpeg encoder).

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If `tstart` is given without `window`, or if no velocity-resolved
        diagnostics are present in `ds`.
    MissingWeightsError
        If `wl` and `w` cannot be found in `ds` or `grids_nc`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    transfer_sign = -1 if fix_transfer_sign else 1

    weights = get_weights(ds, grids_nc=grids_nc)
    wl = weights["wl"]  # dims (lambda, theta)
    w  = weights["w"]   # dims (species, egrid)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values
    lam   = ds["lambda"].values
    sign_values = ds["sign"].values

    sign_titles = {
        sign_values[0]: r"$v_\parallel > 0$",
        sign_values[1]: r"$v_\parallel < 0$",
    }

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window
        if filename is None:
            filename = "vel_transfer_theta_lambda_averaged.png"
        time_title = f"Averaged over t = [{tstart:.1f}, {tend:.1f}]"

        def reduce(name):
            """Weighted (sign, lambda, theta) field for `name`, or None."""
            if name not in ds:
                return None
            da = ds[name].sel(t=slice(tstart, tend)).mean(dim="t")
            return (da * wl * w).sum(dim=["species", "egrid", "kxt_shift"])
    else:
        if filename is None:
            filename = "vel_transfer_theta_lambda.png"
        time_title = None

        def reduce(name):
            if name not in ds:
                return None
            da = ds[name]
            da = da.isel(t=-1) if "t" in da.dims else da
            return (da * wl * w).sum(dim=["species", "egrid", "kxt_shift"])

    # One row per drive: (label, DataArray over (sign, lambda, theta)).
    rows: list[tuple[str, xr.DataArray]] = []
    for field, sym in _FIELD_SYMS.items():
        H = reduce(f"free_energy_transfer_{field}_velocity")
        S = reduce(f"entropy_transfer_{field}_velocity")
        if H is not None:
            rows.append((_drive_label("H", field), transfer_sign * H))
        if S is not None:
            rows.append((_drive_label("S", field), transfer_sign * S))
        if H is not None and S is not None:
            rows.append((_drive_label("U", field), transfer_sign * (H + S)))

    if not rows:
        raise ValueError("No velocity-resolved transfer diagnostics found in dataset.")

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows), sharey=True,
                             squeeze=False)

    for row, (label, transfer) in enumerate(rows):
        # Symmetric colour limits so zero is always centred.
        vmax = float(abs(transfer).max())
        vmin = -vmax

        row_axes = axes[row]
        images = []
        for ax, s in zip(row_axes, sign_values):
            data = transfer.sel(sign=s).values  # shape (lambda, theta)
            im = ax.pcolormesh(theta, lam, data, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                               shading="auto")
            images.append(im)
            ax.set_title(f"{label}, {sign_titles[s]}")
            ax.set_xlabel(r"$\theta$", fontsize=12)

        row_axes[0].set_ylabel(r"$\lambda$", fontsize=12)
        fig.colorbar(images[0], ax=row_axes, label="transfer", shrink=0.8)

    if time_title is not None:
        fig.suptitle(time_title, fontsize=12)

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)

    if not _quiet:
        print(f"Saved {output_path}")
    return output_path


def plot_vel_transfer_theta_lambda_movie(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    tstart: float | None = None,
    window: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Creates a movie of `plot_vel_transfer_theta_lambda` over time.

    Frames are saved individually to a subdirectory of `output_dir`, then
    stitched into a video. `ds` must contain a `t` dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset with a `t` dimension.
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Used to load
        `wl` and `w` if they are absent from `ds`.
    tstart : float, optional
        Start time for the movie. Detected automatically if not provided.
        Falls back to `t[0]` with a warning if detection fails.
    window : float, optional
        Averaging window duration. If provided, each frame shows the rolling
        average over `window` time units. If omitted, frames are instantaneous.
    output_dir : str or Path, optional
        Directory where the movie and frame images are saved.
        Default is `"outputs"`.
    filename : str, optional
        Filename for the output movie. If None, uses
        `"vel_transfer_theta_lambda_movie.mp4"`.
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation. Passed through to
        each frame's plot call.
    fps : int, optional
        Frames per second. Default is 10.
    crf : int, optional
        libx264 Constant Rate Factor (0 = lossless, 51 = worst). Default
        is 18.
    verbose : bool, optional
        If True, prints frame count and per-frame progress. Default is False.

    Returns
    -------
    Path
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If no valid frames exist for the given `tstart` and `window`.
    """
    frame_indices = _resolve_frame_indices(ds, tstart, window)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "vel_transfer_theta_lambda_movie.mp4"

    frames_dir = output_dir / (Path(filename).stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_values = ds["t"].values

    n_frames = len(frame_indices)
    if verbose:
        print(f"Generating {n_frames} frames...")

    for frame_num, t_idx in enumerate(frame_indices):
        frame_filename = f"frame_{frame_num:06d}.png"
        if window is not None:
            plot_vel_transfer_theta_lambda(
                ds,
                grids_nc=grids_nc,
                window=window,
                tstart=float(t_values[t_idx]),
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                tight_bbox=False,
                _quiet=True,
            )
        else:
            plot_vel_transfer_theta_lambda(
                ds.isel(t=t_idx),
                grids_nc=grids_nc,
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                tight_bbox=False,
                _quiet=True,
            )
        if verbose:
            print(f"  Frame {frame_num + 1}/{n_frames}")

    output_path = output_dir / filename
    _stitch_frames_to_movie(frames_dir, output_path, fps=fps, crf=crf)

    print(f"Saved {output_path}")
    return output_path


def plot_vel_transfer_by_theta_by_sign_movie(
    ds: xr.Dataset,
    grids_nc: str | Path | None = None,
    tstart: float | None = None,
    window: float | None = None,
    output_dir: str | Path = "outputs",
    filename: str | None = None,
    fix_transfer_sign: bool = True,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = False,
) -> Path:
    """Creates a movie of `plot_vel_transfer_by_theta_by_sign` over time.

    Frames are saved individually to a subdirectory of `output_dir`, then
    stitched into a video. `ds` must contain a `t` coordinate (i.e. the
    full time-trace file, not a single-snapshot slice).

    Parameters
    ----------
    ds : xarray.Dataset
        Velocity-resolved transfer dataset with a `t` coordinate.
    grids_nc : str or Path, optional
        Path to a `.grids.nc` file produced by `dump_grids`. Used to load
        `wl` and `w` if they are absent from `ds`.
    tstart : float, optional
        Start time for the movie; frames before this time are skipped. If not
        provided, the saturation time is detected automatically from `ds`
        (requires `phi2` to be present). Falls back to `t[0]` with a warning
        if detection fails.
    window : float, optional
        Duration of the averaging window. If provided, each frame shows the
        rolling average over `window` time units starting at that frame's
        time. If omitted, each frame shows the instantaneous transfer.
    output_dir : str or Path, optional
        Directory where the movie and frame images are saved.
        Default is `"outputs"`.
    filename : str, optional
        Filename for the output movie. If None, uses
        `"vel_transfer_by_theta_by_sign_movie.mp4"`.
    fix_transfer_sign : bool, optional
        If True (default), multiplies the free-energy, entropy and U drives by
        -1 to correct a sign error in the GS2 implementation. Passed through to
        each frame's plot call.
    fps : int, optional
        Frames per second. Default is 10.
    crf : int, optional
        libx264 Constant Rate Factor (0 = lossless, 51 = worst). Default
        is 18.
    verbose : bool, optional
        If True, prints frame count and per-frame progress. Default is False.

    Returns
    -------
    Path
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If no valid frames exist for the given `tstart` and `window`.
    """
    frame_indices = _resolve_frame_indices(ds, tstart, window)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "vel_transfer_by_theta_by_sign_movie.mp4"

    frames_dir = output_dir / (Path(filename).stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_values = ds["t"].values

    n_frames = len(frame_indices)
    if verbose:
        print(f"Generating {n_frames} frames...")

    for frame_num, t_idx in enumerate(frame_indices):
        frame_filename = f"frame_{frame_num:06d}.png"
        if window is not None:
            plot_vel_transfer_by_theta_by_sign(
                ds,
                grids_nc=grids_nc,
                window=window,
                tstart=float(t_values[t_idx]),
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                tight_bbox=False,
                _quiet=True,
            )
        else:
            plot_vel_transfer_by_theta_by_sign(
                ds.isel(t=t_idx),
                grids_nc=grids_nc,
                output_dir=frames_dir,
                filename=frame_filename,
                fix_transfer_sign=fix_transfer_sign,
                tight_bbox=False,
                _quiet=True,
            )
        if verbose:
            print(f"  Frame {frame_num + 1}/{n_frames}")

    output_path = output_dir / filename
    _stitch_frames_to_movie(frames_dir, output_path, fps=fps, crf=crf)

    print(f"Saved {output_path}")
    return output_path
