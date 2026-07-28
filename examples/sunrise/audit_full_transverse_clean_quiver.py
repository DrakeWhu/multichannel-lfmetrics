#!/usr/bin/env python3
"""Clean full-transverse B/W/lambda-B quiver audit for SUNRISE publication plots.

This is a read-only plotting helper. It reads WarpX/openPMD field diagnostics
and a final-bunch trajectory NPZ, then writes one PNG per requested step plus
CSV/JSON summaries. The plots use scalar magnitude backgrounds and a single
monochrome quiver layer for vector direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfmetrics.field_slices import read_mesh_plane
from lfmetrics.trajectory_analysis import load_trajectory_npz


C_LIGHT = 299_792_458.0


def parse_steps(text: str) -> list[int]:
    steps: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            steps.append(int(chunk))
    if not steps:
        raise ValueError("At least one step is required")
    if len(set(steps)) != len(steps):
        raise ValueError(f"Duplicate steps are not allowed: {steps}")
    return steps


def wrap_period(angle_deg: float, period_deg: float) -> float:
    return (float(angle_deg) + 0.5 * period_deg) % period_deg - 0.5 * period_deg


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("No valid weighted values")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def to_yx_matrix(plane) -> np.ndarray:
    values = np.asarray(plane.values_si, dtype=float)
    if plane.axis_labels == ("y", "x"):
        return values
    if plane.axis_labels == ("x", "y"):
        return values.T
    raise ValueError(f"Unexpected xy axis order: {plane.axis_labels}")


def read_xy(field_h5: Path, record: str, component: str, z_m: float, step: int):
    return read_mesh_plane(
        field_h5,
        record,
        component,
        plane="xy",
        coordinate_m=float(z_m),
        method="linear",
        iteration=int(step),
    )


def percentile_stats(values: np.ndarray, scale: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values for percentile statistics")
    v = values / scale
    return {
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "p99_5": float(np.percentile(v, 99.5)),
        "max": float(np.max(v)),
    }


def mean_angle_deg(vx: np.ndarray, vy: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    theta = np.arctan2(vy, vx)
    unit = np.exp(1j * theta)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan")
    z = np.sum(weights[valid] * unit[valid]) / np.sum(weights[valid])
    return float(np.degrees(np.angle(z))), float(abs(z))


def clean_unit_quiver(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    mag: np.ndarray,
    *,
    stride: int,
    threshold_fraction: float,
    threshold_absolute: float,
    color: str,
    scale: float,
    width: float,
    alpha: float,
) -> int:
    xq = x[::stride]
    yq = y[::stride]
    vxq = vx[::stride, ::stride]
    vyq = vy[::stride, ::stride]
    magq = mag[::stride, ::stride]

    x_grid, y_grid = np.meshgrid(xq * 1.0e6, yq * 1.0e6)
    max_mag = float(np.nanmax(mag)) if np.size(mag) else 0.0
    threshold = max(threshold_fraction * max_mag, threshold_absolute)

    visible = np.isfinite(magq) & (magq > threshold)
    safe = np.where(magq > 0.0, magq, 1.0)

    u = np.ma.masked_where(~visible, vxq / safe)
    v = np.ma.masked_where(~visible, vyq / safe)

    ax.quiver(
        x_grid,
        y_grid,
        u,
        v,
        color=color,
        angles="xy",
        scale_units="width",
        scale=scale,
        width=width,
        headwidth=3.5,
        headlength=4.6,
        headaxislength=4.1,
        alpha=alpha,
        zorder=5,
    )
    return int(np.count_nonzero(visible))


def render_step(
    *,
    case_dir: Path,
    trajectory,
    trajectory_steps: np.ndarray,
    step: int,
    resolved: dict,
    output_dir: Path,
    lambda0_m: float,
    n_lambda_samples: int,
    quiver_stride: int,
    quiver_threshold_fraction: float,
    quiver_color: str,
    quiver_scale: float,
    quiver_width: float,
    quiver_alpha: float,
    dpi: int,
) -> tuple[dict, dict]:
    idx = np.flatnonzero(trajectory_steps == step)
    if idx.size != 1:
        raise ValueError(f"Step {step} not found uniquely in trajectory")
    frame_index = int(idx[0])
    present = trajectory.present[frame_index]
    if not np.any(present):
        raise ValueError(f"No tracked IDs present at step {step}")

    center_z = weighted_median(
        trajectory.z_m[frame_index, present],
        trajectory.weighting[frame_index, present],
    )

    field_h5 = case_dir / "fields3D" / f"openpmd_{step:06d}.h5"
    if not field_h5.is_file():
        raise FileNotFoundError(field_h5)

    planes = {}
    for record, component in (("B", "x"), ("B", "y"), ("E", "x"), ("E", "y")):
        planes[(record, component)] = read_xy(field_h5, record, component, center_z, step)

    x = np.asarray(planes[("B", "x")].coordinate("x"), dtype=float)
    y = np.asarray(planes[("B", "x")].coordinate("y"), dtype=float)

    bx = to_yx_matrix(planes[("B", "x")])
    by = to_yx_matrix(planes[("B", "y")])
    ex = to_yx_matrix(planes[("E", "x")])
    ey = to_yx_matrix(planes[("E", "y")])

    btotal = np.hypot(bx, by)
    wx = ex - C_LIGHT * by
    wy = ey + C_LIGHT * bx
    wperp = np.hypot(wx, wy)
    bresidual = wperp / C_LIGHT

    z_samples = np.linspace(center_z - 0.5 * lambda0_m, center_z + 0.5 * lambda0_m, n_lambda_samples)
    bx_samples = []
    by_samples = []
    for z_sample in z_samples:
        bx_plane = read_xy(field_h5, "B", "x", z_sample, step)
        by_plane = read_xy(field_h5, "B", "y", z_sample, step)
        bx_samples.append(to_yx_matrix(bx_plane))
        by_samples.append(to_yx_matrix(by_plane))

    bx_lambda = np.mean(np.stack(bx_samples, axis=0), axis=0)
    by_lambda = np.mean(np.stack(by_samples, axis=0), axis=0)
    blambda = np.hypot(bx_lambda, by_lambda)

    b_stats = percentile_stats(btotal, 1.0e3)
    w_stats = percentile_stats(wperp, 1.0e12)
    bres_stats = percentile_stats(bresidual, 1.0e3)
    blam_stats = percentile_stats(blambda, 1.0e3)

    b_angle, b_r = mean_angle_deg(bx, by, btotal)
    w_angle, w_r = mean_angle_deg(wx, wy, wperp)
    blam_angle, blam_r = mean_angle_deg(bx_lambda, by_lambda, blambda)

    polarization_angle_deg = float(resolved["polarization_angle_deg"])
    honeycomb_angle_deg = float(resolved["honeycomb_angle_deg"])
    expected_laser_b_axis = polarization_angle_deg + 90.0

    row = {
        "step": int(step),
        "time_fs": float(planes[("B", "x")].time_s * 1.0e15),
        "t_over_T_800nm": float(planes[("B", "x")].time_s / (lambda0_m / C_LIGHT)),
        "center_z_um": float(center_z * 1.0e6),
        "Btotal_p99_kT": b_stats["p99"],
        "Btotal_max_kT": b_stats["max"],
        "Wperp_p99_TV_m": w_stats["p99"],
        "Wperp_max_TV_m": w_stats["max"],
        "Bresidual_p99_kT": bres_stats["p99"],
        "Bresidual_max_kT": bres_stats["max"],
        "Blambda_p99_kT": blam_stats["p99"],
        "Blambda_max_kT": blam_stats["max"],
        "B_angle_deg": b_angle,
        "B_coherence_R": b_r,
        "W_angle_deg": w_angle,
        "W_coherence_R": w_r,
        "Blambda_angle_deg": blam_angle,
        "Blambda_coherence_R": blam_r,
        "delta_B_to_laserB_axis_deg": wrap_period(b_angle - expected_laser_b_axis, 180.0),
        "delta_B_to_hex_axis_deg": wrap_period(b_angle - honeycomb_angle_deg, 60.0),
        "delta_W_to_hex_axis_deg": wrap_period(w_angle - honeycomb_angle_deg, 60.0),
        "delta_Blambda_to_hex_axis_deg": wrap_period(blam_angle - honeycomb_angle_deg, 60.0),
    }

    b_vmax = max(5.0, b_stats["p99_5"])
    w_vmax = max(0.05, w_stats["p99_5"])
    blam_vmax = max(0.5, blam_stats["p99_5"])
    extent = [x[0] * 1.0e6, x[-1] * 1.0e6, y[0] * 1.0e6, y[-1] * 1.0e6]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)

    im0 = axes[0].imshow(
        btotal / 1.0e3,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=0.0,
        vmax=b_vmax,
        interpolation="nearest",
    )
    axes[0].set_title(rf"raw $|B_\perp|$ [kT], vmax={b_vmax:.1f}")
    n_b_arrows = clean_unit_quiver(
        axes[0],
        x,
        y,
        bx,
        by,
        btotal,
        stride=quiver_stride,
        threshold_fraction=quiver_threshold_fraction,
        threshold_absolute=2.0e3,
        color=quiver_color,
        scale=quiver_scale,
        width=quiver_width,
        alpha=quiver_alpha,
    )

    im1 = axes[1].imshow(
        wperp / 1.0e12,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=w_vmax,
        interpolation="nearest",
    )
    axes[1].set_title(rf"$|W_\perp|$ [TV/m], vmax={w_vmax:.2f}")
    n_w_arrows = clean_unit_quiver(
        axes[1],
        x,
        y,
        wx,
        wy,
        wperp,
        stride=quiver_stride,
        threshold_fraction=quiver_threshold_fraction,
        threshold_absolute=0.015e12,
        color=quiver_color,
        scale=quiver_scale,
        width=quiver_width,
        alpha=quiver_alpha,
    )

    im2 = axes[2].imshow(
        blambda / 1.0e3,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=blam_vmax,
        interpolation="nearest",
    )
    axes[2].set_title(rf"$|\langle B_\perp\rangle_\lambda|$ [kT], vmax={blam_vmax:.1f}")
    n_blam_arrows = clean_unit_quiver(
        axes[2],
        x,
        y,
        bx_lambda,
        by_lambda,
        blambda,
        stride=quiver_stride,
        threshold_fraction=quiver_threshold_fraction,
        threshold_absolute=0.1e3,
        color=quiver_color,
        scale=quiver_scale,
        width=quiver_width,
        alpha=quiver_alpha,
    )

    for ax in axes:
        ax.set_xlabel(r"$x$ [$\mu$m]")
        ax.set_ylabel(r"$y$ [$\mu$m]")
        ax.set_aspect("equal")
        ax.set_xlim(x[0] * 1.0e6, x[-1] * 1.0e6)
        ax.set_ylim(y[0] * 1.0e6, y[-1] * 1.0e6)

    fig.colorbar(im0, ax=axes[0], shrink=0.82)
    fig.colorbar(im1, ax=axes[1], shrink=0.82)
    fig.colorbar(im2, ax=axes[2], shrink=0.82)
    fig.suptitle(
        f"SOFT iter_022 c003 — clean full transverse fields, step {step}, t/T={row['t_over_T_800nm']:.2f}",
        fontsize=12,
    )

    png = output_dir / f"clean_full_transverse_B_W_Blambda_step{step:06d}.png"
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)

    row.update(
        {
            "png": str(png),
            "B_quiver_visible_arrows": n_b_arrows,
            "W_quiver_visible_arrows": n_w_arrows,
            "Blambda_quiver_visible_arrows": n_blam_arrows,
        }
    )

    frame = {
        "step": int(step),
        "time_fs": row["time_fs"],
        "t_over_T_800nm": row["t_over_T_800nm"],
        "center_z_um": row["center_z_um"],
        "png": str(png),
        "Btotal_kT": b_stats,
        "Wperp_TV_m": w_stats,
        "Bresidual_from_wake_kT": bres_stats,
        "Blambda_kT": blam_stats,
        "angles": {
            "B_angle_deg": b_angle,
            "B_coherence_R": b_r,
            "W_angle_deg": w_angle,
            "W_coherence_R": w_r,
            "Blambda_angle_deg": blam_angle,
            "Blambda_coherence_R": blam_r,
            "delta_B_to_laserB_axis_deg": row["delta_B_to_laserB_axis_deg"],
            "delta_B_to_hex_axis_deg": row["delta_B_to_hex_axis_deg"],
            "delta_W_to_hex_axis_deg": row["delta_W_to_hex_axis_deg"],
            "delta_Blambda_to_hex_axis_deg": row["delta_Blambda_to_hex_axis_deg"],
        },
        "quiver": {
            "stride": quiver_stride,
            "threshold_fraction": quiver_threshold_fraction,
            "color": quiver_color,
            "visible_arrows": {
                "B": n_b_arrows,
                "W": n_w_arrows,
                "Blambda": n_blam_arrows,
            },
        },
    }
    return row, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--resolved-parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--steps", default="200,500,600,700,800,900,1000")
    parser.add_argument("--lambda0-um", type=float, default=0.8)
    parser.add_argument("--lambda-samples", type=int, default=21)
    parser.add_argument("--quiver-stride", type=int, default=3)
    parser.add_argument("--quiver-threshold-fraction", type=float, default=0.02)
    parser.add_argument("--quiver-color", default="white")
    parser.add_argument("--quiver-scale", type=float, default=42.0)
    parser.add_argument("--quiver-width", type=float, default=0.0018)
    parser.add_argument("--quiver-alpha", type=float, default=0.95)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    if args.lambda_samples < 3 or args.lambda_samples % 2 == 0:
        raise ValueError("--lambda-samples must be an odd integer >= 3")
    if args.quiver_stride < 1:
        raise ValueError("--quiver-stride must be >= 1")
    if not (0.0 <= args.quiver_threshold_fraction < 1.0):
        raise ValueError("--quiver-threshold-fraction must be in [0,1)")

    for required in (args.case_dir, args.trajectory_npz, args.resolved_parameters):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.zip_path.exists():
        raise FileExistsError(args.zip_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    trajectory = load_trajectory_npz(args.trajectory_npz)
    trajectory_steps = np.asarray(trajectory.steps, dtype=np.int64)
    resolved = json.loads(args.resolved_parameters.read_text(encoding="utf-8"))
    lambda0_m = float(args.lambda0_um) * 1.0e-6

    rows: list[dict] = []
    frames: list[dict] = []
    for step in steps:
        row, frame = render_step(
            case_dir=args.case_dir,
            trajectory=trajectory,
            trajectory_steps=trajectory_steps,
            step=step,
            resolved=resolved,
            output_dir=args.output_dir,
            lambda0_m=lambda0_m,
            n_lambda_samples=args.lambda_samples,
            quiver_stride=args.quiver_stride,
            quiver_threshold_fraction=args.quiver_threshold_fraction,
            quiver_color=args.quiver_color,
            quiver_scale=args.quiver_scale,
            quiver_width=args.quiver_width,
            quiver_alpha=args.quiver_alpha,
            dpi=args.dpi,
        )
        rows.append(row)
        frames.append(frame)
        print(
            f"step={step:4d} "
            f"t/T={row['t_over_T_800nm']:7.3f} "
            f"Bp99={row['Btotal_p99_kT']:8.3f}kT "
            f"Wp99={row['Wperp_p99_TV_m']:7.4f}TV/m "
            f"Bres={row['Bresidual_p99_kT']:7.3f}kT "
            f"Blam={row['Blambda_p99_kT']:7.3f}kT "
            f"arrows(B/W/Blam)={row['B_quiver_visible_arrows']}/"
            f"{row['W_quiver_visible_arrows']}/"
            f"{row['Blambda_quiver_visible_arrows']}"
        )

    summary = {
        "schema": "multichannel_clean_full_transverse_quiver_audit_v1",
        "case_dir": str(args.case_dir),
        "trajectory_npz": str(args.trajectory_npz),
        "resolved_parameters": str(args.resolved_parameters),
        "lambda0_m": lambda0_m,
        "lambda_samples": args.lambda_samples,
        "quiver_stride": args.quiver_stride,
        "quiver_threshold_fraction": args.quiver_threshold_fraction,
        "quiver_color": args.quiver_color,
        "definitions": {
            "Btotal": "raw total sqrt(Bx^2+By^2)",
            "Wperp": "sqrt((Ex-cBy)^2 + (Ey+cBx)^2), beta=1, +z",
            "Bresidual_from_wake": "Wperp/c",
            "Blambda": "spatial wavelength average of Bx,By before magnitude",
        },
        "frames": frames,
    }

    json_path = args.output_dir / "clean_full_transverse_B_W_Blambda_audit.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "clean_full_transverse_B_W_Blambda_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pngs = sorted(args.output_dir.glob("*.png"))
    files = pngs + [csv_path, json_path]
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.name)

    print()
    print("===== OUTPUT =====")
    print("output_dir:", args.output_dir)
    print("zip_path:", args.zip_path)
    print("n_steps:", len(steps))
    print("n_png:", len(pngs))
    print("CLEAN_FULL_TRANSVERSE_QUIVER_AUDIT=PASS")


if __name__ == "__main__":
    main()
