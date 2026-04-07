"""Create various plots of the kinetic energy and entropy transfer."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from gs2ew.utils.gs2_output import detect_saturation_time as _detect_saturation_time
from gs2ew.utils.weights import get_weights


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
    _quiet: bool = False,
) -> Path:
    """Plots the poloidal structure of each enabled transfer diagnostic.

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

    Returns
    -------
    Path
        Path to the saved figure file
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    # Determine enabled transfer diagnostics
    all_diags = [
        "kinetic_energy_transfer_theta",
        "entropy_transfer_phi_theta",
        "entropy_transfer_apar_theta",
        "entropy_transfer_bpar_theta",
    ]
    enabled_diagnostics = [d for d in all_diags if d in ds]

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theta = ds["theta"].values

    labels = {
        "kinetic_energy_transfer_theta": r"$T_v^\text{ZF}$",
        "entropy_transfer_phi_theta": r"$T_{S,\phi}^\text{ZF}$",
        "entropy_transfer_apar_theta": r"$T_{S,A_\parallel}^\text{ZF}$",
        "entropy_transfer_bpar_theta": r"$T_{S,B_\parallel}^\text{ZF}$"
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window

        if filename is None:
            filename = "transfer_by_theta_averaged.png"

        for diag in enabled_diagnostics:
            transfer_avg = ds[diag].sel(t=slice(tstart, tend)).mean(dim="t").values
            ax.plot(theta, transfer_avg, linewidth=1.5, label=labels[diag])

        ax.set_title(f"Averaged over t = [{tstart:.1f}, {tend:.1f}]")
    else:
        if filename is None:
            filename = "transfer_by_theta.png"

        for diag in enabled_diagnostics:
            transfer = ds[diag].isel(t=-1).values
            ax.plot(theta, transfer, linewidth=1.5, label=labels[diag])

    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("transfer", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    n_frames = len(frame_indices)
    if verbose:
        print(f"Generating {n_frames} frames...")

    # Produce frames
    for frame_num, t_idx in enumerate(frame_indices):
        frame_filename = f"frame_{frame_num:06d}.png"
        if window is not None:
            _ = plot_transfer_by_theta(
                ds,
                window=window,
                tstart=float(t_values[t_idx]),
                output_dir=frames_dir,
                filename=frame_filename,
                _quiet=True,
            )
        else:
            # Pass a single-timestep lazy slice; isel(t=-1) inside
            # plot_transfer_by_theta will select the only timestep.
            plot_transfer_by_theta(
                ds.isel(t=slice(t_idx, t_idx + 1)),
                output_dir=frames_dir,
                filename=frame_filename,
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
    _quiet: bool = False,
) -> Path:
    """Plot the poloidal structure of the velocity-resolved entropy transfer,
    integrated over velocity space.

    Each enabled velocity-resolved diagnostic in `ds` is weighted by the
    velocity-space integration weights `wl(lambda, theta)` and
    `w(species, egrid)` before summing over all non-theta dimensions except
    `sign`. The two parallel-velocity directions (sign=1: vpa > 0,
    sign=2: vpa < 0) are plotted as separate curves (solid and
    dashed respectively) on the same axis.

    By default, plots the last time step. If `window` is provided, the
    diagnostics are averaged over a time window instead.

    Parameters
    ----------
    ds : xarray.Dataset
        GS2 output dataset (loaded from `.out.nc` file). Expected to contain
        one or more of the diagnostics listed below, each with dims
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

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If `tstart` is given without `window`.
    MissingWeightsError
        If `wl` and `w` cannot be found in `ds` or `grids_nc`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    all_diags = [
        "entropy_transfer_phi_velocity",
        "entropy_transfer_apar_velocity",
        "entropy_transfer_bpar_velocity",
    ]
    enabled_diagnostics = [d for d in all_diags if d in ds]

    labels = {
        "entropy_transfer_phi_velocity":  r"$T_{S,\phi}^\text{ZF}$",
        "entropy_transfer_apar_velocity": r"$T_{S,A_\parallel}^\text{ZF}$",
        "entropy_transfer_bpar_velocity": r"$T_{S,B_\parallel}^\text{ZF}$",
    }

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

    fig, ax = plt.subplots(figsize=(10, 5))

    if window is not None:
        if tstart is None:
            tstart = float(ds["t"].values[-1]) - window
        tend = tstart + window

        if filename is None:
            filename = "vel_transfer_by_theta_averaged.png"

        for diag in enabled_diagnostics:
            for s in sign_values:
                # Average over the time window per diagnostic, then weight and
                # sum over velocity dimensions to get the theta-dependent transfer.
                transfer = (
                    ds[diag].sel(t=slice(tstart, tend), sign=s).mean(dim="t") * wl * w
                ).sum(dim=["species", "lambda", "egrid", "kxt_shift"])

                ax.plot(theta, transfer.values, linewidth=1.5,
                        linestyle=sign_styles[s],
                        label=f"{labels[diag]}, {sign_labels[s]}")

        ax.set_title(f"Averaged over t = [{tstart:.1f}, {tend:.1f}]")
    else:
        if filename is None:
            filename = "vel_transfer_by_theta.png"

        for diag in enabled_diagnostics:
            for s in sign_values:
                # Multiply by velocity-space weights, then sum over all non-theta
                # dimensions (excluding sign) to obtain the theta-dependent transfer.
                diag_da = ds[diag].isel(t=-1) if "t" in ds[diag].dims else ds[diag]
                transfer = (
                    diag_da.sel(sign=s) * wl * w
                ).sum(dim=["species", "lambda", "egrid", "kxt_shift"])

                ax.plot(theta, transfer.values, linewidth=1.5,
                        linestyle=sign_styles[s],
                        label=f"{labels[diag]}, {sign_labels[s]}")

    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("transfer", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    _quiet: bool = False,
) -> Path:
    """Plot 2D heatmaps of the entropy transfer in lambda-theta space.

    For each enabled velocity-resolved diagnostic, shows a row of two subplots
    side by side — one for each parallel-velocity sign (sign=1: vpa > 0,
    sign=2: vpa < 0). The x-axis is `theta`, the y-axis is `lambda`. Each
    cell is weighted by `wl(lambda, theta)` and `w(species, egrid)` and summed
    over `species`, `egrid`, and `kxt_shift`, so that integrating along the
    lambda axis of either panel recovers the corresponding curve in
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

    Returns
    -------
    Path
        Path to the saved figure file.

    Raises
    ------
    ValueError
        If `tstart` is given without `window`.
    MissingWeightsError
        If `wl` and `w` cannot be found in `ds` or `grids_nc`.
    """
    if tstart is not None and window is None:
        raise ValueError("`tstart` requires `window` to be specified.")

    all_diags = [
        "entropy_transfer_phi_velocity",
        "entropy_transfer_apar_velocity",
        "entropy_transfer_bpar_velocity",
    ]
    enabled_diagnostics = [d for d in all_diags if d in ds]

    diag_labels = {
        "entropy_transfer_phi_velocity":  r"$T_{S,\phi}^\text{ZF}$",
        "entropy_transfer_apar_velocity": r"$T_{S,A_\parallel}^\text{ZF}$",
        "entropy_transfer_bpar_velocity": r"$T_{S,B_\parallel}^\text{ZF}$",
    }

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

        def get_transfer(diag):
            return (
                ds[diag].sel(t=slice(tstart, tend)).mean(dim="t") * wl * w
            ).sum(dim=["species", "egrid", "kxt_shift"])
    else:
        if filename is None:
            filename = "vel_transfer_theta_lambda.png"

        time_title = None

        def get_transfer(diag):
            diag_da = ds[diag].isel(t=-1) if "t" in ds[diag].dims else ds[diag]
            return (diag_da * wl * w).sum(dim=["species", "egrid", "kxt_shift"])

    n_diags = len(enabled_diagnostics)
    fig, axes = plt.subplots(n_diags, 2, figsize=(12, 5 * n_diags), sharey=True,
                             squeeze=False)

    for row, diag in enumerate(enabled_diagnostics):
        # Compute weighted transfer: sum over species, egrid, kxt_shift.
        # Result has dims (sign, lambda, theta).
        transfer = get_transfer(diag)

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
            ax.set_title(f"{diag_labels[diag]}, {sign_titles[s]}")
            ax.set_xlabel(r"$\theta$", fontsize=12)

        row_axes[0].set_ylabel(r"$\lambda$", fontsize=12)
        fig.colorbar(images[0], ax=row_axes, label="transfer", shrink=0.8)

    if time_title is not None:
        fig.suptitle(time_title, fontsize=12)

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
                _quiet=True,
            )
        else:
            plot_vel_transfer_theta_lambda(
                ds.isel(t=t_idx),
                grids_nc=grids_nc,
                output_dir=frames_dir,
                filename=frame_filename,
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
                _quiet=True,
            )
        else:
            plot_vel_transfer_by_theta_by_sign(
                ds.isel(t=t_idx),
                grids_nc=grids_nc,
                output_dir=frames_dir,
                filename=frame_filename,
                _quiet=True,
            )
        if verbose:
            print(f"  Frame {frame_num + 1}/{n_frames}")

    output_path = output_dir / filename
    _stitch_frames_to_movie(frames_dir, output_path, fps=fps, crf=crf)

    print(f"Saved {output_path}")
    return output_path
