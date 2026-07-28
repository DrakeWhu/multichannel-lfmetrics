#!/usr/bin/env python3
"""Render Fig. 9-like composites using wavelength-averaged transverse B.

Read-only SUNRISE helper. It reads existing WarpX/openPMD field diagnostics and
tracked final-bunch trajectories, then writes a flat directory of PNG files plus
summary metadata. It does not modify simulation inputs, HDF5/openPMD files,
tracking NPZ files, or optimizer state.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfmetrics.field_slices import read_mesh_plane
from lfmetrics.trajectory_analysis import load_trajectory_npz

C_LIGHT = 299792458.0


def parse_steps(text: str | None) -> list[int] | None:
    if text is None or text.strip().lower() in {"", "all"}:
        return None
    steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate steps requested: {steps}")
    return steps


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("No valid values for weighted median")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def matrix_for_axes(plane, vertical_axis: str, horizontal_axis: str) -> np.ndarray:
    values = np.asarray(plane.values_si, dtype=float)
    labels = tuple(plane.axis_labels)
    if labels == (vertical_axis, horizontal_axis):
        return values
    if labels == (horizontal_axis, vertical_axis):
        return values.T
    raise ValueError(
        f"Unexpected axis order {labels}; expected {(vertical_axis, horizontal_axis)}"
    )


def read_plane(
    field_h5: Path,
    record: str,
    component: str,
    plane: str,
    coordinate_m: float,
    step: int,
):
    return read_mesh_plane(
        field_h5,
        record,
        component,
        plane=plane,
        coordinate_m=float(coordinate_m),
        method="linear",
        iteration=int(step),
    )


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def inferno_quiver(
    ax,
    x_m: np.ndarray,
    y_m: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bmag: np.ndarray,
    *,
    target_arrows: int,
    threshold_fraction: float,
    scale: float,
    width: float,
    alpha: float,
):
    ny, nx = bmag.shape
    stride = max(1, int(round(min(nx, ny) / max(1, target_arrows))))

    xq = x_m[::stride] * 1.0e6
    yq = y_m[::stride] * 1.0e6
    x_grid, y_grid = np.meshgrid(xq, yq)

    bxq = np.asarray(bx[::stride, ::stride], dtype=float)
    byq = np.asarray(by[::stride, ::stride], dtype=float)
    bmq = np.asarray(bmag[::stride, ::stride], dtype=float)

    bmax = finite_max(bmag)
    threshold = max(float(threshold_fraction) * bmax, 1.0e-30)
    visible = np.isfinite(bmq) & (bmq > threshold)
    safe = np.where(bmq > 0.0, bmq, 1.0)

    u = np.ma.masked_where(~visible, bxq / safe)
    v = np.ma.masked_where(~visible, byq / safe)
    colors = np.ma.masked_where(~visible, bmq / 1.0e3)

    q = ax.quiver(
        x_grid,
        y_grid,
        u,
        v,
        colors,
        cmap="inferno",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=width,
        headwidth=3.6,
        headlength=4.8,
        headaxislength=4.3,
        minlength=0.0,
        pivot="mid",
        alpha=alpha,
        zorder=5,
    )
    q.set_clim(0.0, max(0.25, finite_percentile(bmag / 1.0e3, 99.5)))
    return q


def trajectory_energy_mev(trajectory, frame_index: int, present: np.ndarray) -> np.ndarray:
    if hasattr(trajectory, "energy_MeV"):
        return np.asarray(trajectory.energy_MeV[frame_index, present], dtype=float)
    raise AttributeError(
        "trajectory object has no energy_MeV; cannot color tracked particles"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--trajectory-npz", required=True, type=Path)
    parser.add_argument("--resolved-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--lambda0-um", default=0.8, type=float)
    parser.add_argument("--lambda-samples", default=21, type=int)
    parser.add_argument("--b-background-cmap", default="viridis")
    parser.add_argument("--quiver-target-arrows", default=26, type=int)
    parser.add_argument("--quiver-threshold-fraction", default=0.06, type=float)
    parser.add_argument("--quiver-scale", default=0.9, type=float)
    parser.add_argument("--quiver-width", default=0.0032, type=float)
    parser.add_argument("--quiver-alpha", default=0.95, type=float)
    parser.add_argument("--particle-size", default=6.0, type=float)
    parser.add_argument("--dpi", default=180, type=int)
    args = parser.parse_args()

    case_dir = args.case_dir
    trajectory_npz = args.trajectory_npz
    resolved_parameters = args.resolved_parameters
    output_dir = args.output_dir
    zip_path = args.zip_path

    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    if not trajectory_npz.is_file():
        raise FileNotFoundError(trajectory_npz)
    if not resolved_parameters.is_file():
        raise FileNotFoundError(resolved_parameters)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output_dir: {output_dir}")
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite zip_path: {zip_path}")
    if args.lambda_samples < 3 or args.lambda_samples % 2 == 0:
        raise ValueError("--lambda-samples must be an odd integer >= 3")

    output_dir.mkdir(parents=True, exist_ok=False)

    trajectory = load_trajectory_npz(trajectory_npz)
    resolved = json.loads(resolved_parameters.read_text(encoding="utf-8"))

    field_dir = case_dir / "fields3D"
    available_steps = sorted(
        int(path.stem.rsplit("_", 1)[1]) for path in field_dir.glob("openpmd_*.h5")
    )
    requested_steps = parse_steps(args.steps)
    if requested_steps is None:
        steps = available_steps
    else:
        missing = sorted(set(requested_steps) - set(available_steps))
        if missing:
            raise FileNotFoundError(f"Missing field files for steps: {missing}")
        steps = requested_steps

    trajectory_steps = np.asarray(trajectory.steps, dtype=np.int64)
    lambda0_m = float(args.lambda0_um) * 1.0e-6

    frames = []

    for step in steps:
        frame_indices = np.flatnonzero(trajectory_steps == int(step))
        if frame_indices.size != 1:
            print(f"SKIP step={step}: not found uniquely in trajectory")
            continue
        frame_index = int(frame_indices[0])
        present = np.asarray(trajectory.present[frame_index], dtype=bool)
        if not np.any(present):
            print(f"SKIP step={step}: no tracked final-bunch IDs present")
            continue

        center_z_m = weighted_median(
            np.asarray(trajectory.z_m[frame_index, present], dtype=float),
            np.asarray(trajectory.weighting[frame_index, present], dtype=float),
        )
        field_h5 = field_dir / f"openpmd_{int(step):06d}.h5"

        # Panel (a): wavelength-averaged transverse B in xy.
        z_samples = np.linspace(
            center_z_m - 0.5 * lambda0_m,
            center_z_m + 0.5 * lambda0_m,
            int(args.lambda_samples),
        )
        bx_samples = []
        by_samples = []
        xy_reference = None
        for z_sample in z_samples:
            bx_plane = read_plane(field_h5, "B", "x", "xy", float(z_sample), int(step))
            by_plane = read_plane(field_h5, "B", "y", "xy", float(z_sample), int(step))
            if xy_reference is None:
                xy_reference = bx_plane
            bx_samples.append(matrix_for_axes(bx_plane, "y", "x"))
            by_samples.append(matrix_for_axes(by_plane, "y", "x"))

        x_xy = np.asarray(xy_reference.coordinate("x"), dtype=float)
        y_xy = np.asarray(xy_reference.coordinate("y"), dtype=float)
        bx_lambda = np.mean(np.stack(bx_samples, axis=0), axis=0)
        by_lambda = np.mean(np.stack(by_samples, axis=0), axis=0)
        bmag_lambda = np.hypot(bx_lambda, by_lambda)
        bmag_lambda_kT = bmag_lambda / 1.0e3
        b_vmax = max(0.25, finite_percentile(bmag_lambda_kT, 99.5))

        # Panels (b,c): Ez longitudinal planes.
        ez_xz_plane = read_plane(field_h5, "E", "z", "xz", 0.0, int(step))
        ez_yz_plane = read_plane(field_h5, "E", "z", "yz", 0.0, int(step))
        x_xz = np.asarray(ez_xz_plane.coordinate("x"), dtype=float)
        z_xz = np.asarray(ez_xz_plane.coordinate("z"), dtype=float)
        y_yz = np.asarray(ez_yz_plane.coordinate("y"), dtype=float)
        z_yz = np.asarray(ez_yz_plane.coordinate("z"), dtype=float)
        ez_xz = matrix_for_axes(ez_xz_plane, "x", "z")
        ez_yz = matrix_for_axes(ez_yz_plane, "y", "z")

        xi_xz_um = (z_xz - center_z_m) * 1.0e6
        xi_yz_um = (z_yz - center_z_m) * 1.0e6
        ez_values_TV_m = np.concatenate(
            [
                np.ravel(ez_xz[np.isfinite(ez_xz)]),
                np.ravel(ez_yz[np.isfinite(ez_yz)]),
            ]
        ) / 1.0e12
        ez_limit = max(0.5, finite_percentile(np.abs(ez_values_TV_m), 99.5))

        x_part_um = np.asarray(trajectory.x_m[frame_index, present], dtype=float) * 1.0e6
        y_part_um = np.asarray(trajectory.y_m[frame_index, present], dtype=float) * 1.0e6
        xi_part_um = (
            np.asarray(trajectory.z_m[frame_index, present], dtype=float) - center_z_m
        ) * 1.0e6
        energy_mev = trajectory_energy_mev(trajectory, frame_index, present)
        energy_vmax = max(0.1, finite_percentile(energy_mev, 99.5))

        time_s = float(ez_xz_plane.time_s)
        time_fs = time_s * 1.0e15

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18.8, 5.9),
            constrained_layout=True,
        )

        im0 = axes[0].imshow(
            bmag_lambda_kT,
            origin="lower",
            extent=[
                x_xy[0] * 1.0e6,
                x_xy[-1] * 1.0e6,
                y_xy[0] * 1.0e6,
                y_xy[-1] * 1.0e6,
            ],
            cmap=args.b_background_cmap,
            vmin=0.0,
            vmax=b_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        inferno_quiver(
            axes[0],
            x_xy,
            y_xy,
            bx_lambda,
            by_lambda,
            bmag_lambda,
            target_arrows=args.quiver_target_arrows,
            threshold_fraction=args.quiver_threshold_fraction,
            scale=args.quiver_scale,
            width=args.quiver_width,
            alpha=args.quiver_alpha,
        )
        axes[0].set_title(r"$|\langle B_\perp\rangle_\lambda|$ [kT]")
        axes[0].set_xlabel(r"$x$ [$\mu$m]")
        axes[0].set_ylabel(r"$y$ [$\mu$m]")

        im1 = axes[1].imshow(
            ez_xz / 1.0e12,
            origin="lower",
            extent=[
                xi_xz_um[0],
                xi_xz_um[-1],
                x_xz[0] * 1.0e6,
                x_xz[-1] * 1.0e6,
            ],
            cmap="RdBu_r",
            vmin=-ez_limit,
            vmax=ez_limit,
            interpolation="nearest",
            aspect="auto",
        )
        axes[1].scatter(
            xi_part_um,
            x_part_um,
            c=energy_mev,
            s=args.particle_size,
            cmap="turbo",
            vmin=0.0,
            vmax=energy_vmax,
            linewidths=0.0,
            alpha=0.95,
            zorder=5,
        )
        axes[1].set_title(r"$E_z$ [TV/m]")
        axes[1].set_xlabel(r"$\xi_{\rm grid}$ [$\mu$m]")
        axes[1].set_ylabel(r"$x$ [$\mu$m]")

        im2 = axes[2].imshow(
            ez_yz / 1.0e12,
            origin="lower",
            extent=[
                xi_yz_um[0],
                xi_yz_um[-1],
                y_yz[0] * 1.0e6,
                y_yz[-1] * 1.0e6,
            ],
            cmap="RdBu_r",
            vmin=-ez_limit,
            vmax=ez_limit,
            interpolation="nearest",
            aspect="auto",
        )
        scatter = axes[2].scatter(
            xi_part_um,
            y_part_um,
            c=energy_mev,
            s=args.particle_size,
            cmap="turbo",
            vmin=0.0,
            vmax=energy_vmax,
            linewidths=0.0,
            alpha=0.95,
            zorder=5,
        )
        axes[2].set_title(r"tracked bunch $E_{\rm kin}$ [MeV]")
        axes[2].set_xlabel(r"$\xi_{\rm grid}$ [$\mu$m]")
        axes[2].set_ylabel(r"$y$ [$\mu$m]")

        cb0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.08)
        cb0.set_label(r"$|\langle B_\perp\rangle_\lambda|$ [kT]")
        cb1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", fraction=0.05, pad=0.08)
        cb1.set_label(r"$E_z$ [TV/m]")
        cb2 = fig.colorbar(scatter, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.08)
        cb2.set_label(r"tracked bunch $E_{\rm kin}$ [MeV]")

        fig.suptitle(
            f"SOFT iter_022 c003 — publication composite with "
            f"$\\langle B_\\perp\\rangle_\\lambda$, step {step}, "
            f"t = {time_fs:.2f} fs",
            fontsize=16,
        )

        png_path = output_dir / f"soft_i022_c003_fig9_blambda_step{int(step):06d}.png"
        fig.savefig(png_path, dpi=args.dpi, facecolor="white")
        plt.close(fig)

        frame = {
            "step": int(step),
            "time_fs": time_fs,
            "center_z_um": center_z_m * 1.0e6,
            "blambda_p99_kT": finite_percentile(bmag_lambda_kT, 99.0),
            "blambda_p99_5_kT": finite_percentile(bmag_lambda_kT, 99.5),
            "blambda_max_kT": finite_max(bmag_lambda_kT),
            "ez_limit_TV_m": ez_limit,
            "energy_color_max_MeV": energy_vmax,
            "n_tracked_present": int(np.count_nonzero(present)),
            "png": png_path.name,
        }
        frames.append(frame)

        print(
            f"step={int(step):4d} "
            f"time_fs={time_fs:8.3f} "
            f"tracked={frame['n_tracked_present']:5d} "
            f"Blam_p99.5={frame['blambda_p99_5_kT']:7.3f} kT "
            f"Blam_max={frame['blambda_max_kT']:7.3f} kT "
            f"Ez_lim={ez_limit:6.3f} TV/m "
            f"Emax_plot={energy_vmax:7.3f} MeV"
        )

    summary = {
        "schema": "publication_fig9_blambda_composite_v1",
        "case_dir": str(case_dir),
        "trajectory_npz": str(trajectory_npz),
        "lambda0_m": lambda0_m,
        "lambda_samples": int(args.lambda_samples),
        "b_background_cmap": args.b_background_cmap,
        "quiver_cmap": "inferno",
        "ez_cmap": "RdBu_r",
        "particle_cmap": "turbo",
        "resolved_parameters": resolved,
        "n_generated_frames": len(frames),
        "frames": frames,
    }
    summary_path = output_dir / "publication_fig9_blambda_composites_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    files = sorted(output_dir.glob("*.png")) + [summary_path]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.name)

    print()
    print("===== OUTPUT =====")
    print("output_dir:", output_dir)
    print("zip_path:", zip_path)
    print("summary:", summary_path)
    print("n_frames:", len(frames))
    print("PUBLICATION_FIG9_BLAMBDA_COMPOSITES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
