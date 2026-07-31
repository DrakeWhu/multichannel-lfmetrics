#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfmetrics.field_slices import read_mesh_plane


@dataclass(frozen=True)
class SliceRequest:
    label: str
    case_dir: Path
    step: int
    xi_um: float


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_text_tuple(value) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim == 0:
        return (decode_text(array.item()),)
    return tuple(decode_text(item) for item in array.tolist())


def step_field_file(case_dir: Path, step: int) -> Path:
    path = case_dir / "fields3D" / f"openpmd_{step:06d}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Missing field file: {path}")
    return path


def iteration_key_for_file(path: Path, step: int) -> str:
    with h5py.File(path, "r") as h5:
        keys = sorted(h5["data"].keys(), key=lambda item: int(item))
    requested = str(int(step))
    if requested in keys:
        return requested
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"Iteration {requested} not found in {path}; available={keys}")


def axis_coordinates_m(path: Path, *, record_name: str, component_name: str, axis_name: str, iteration: int) -> np.ndarray:
    iteration_key = iteration_key_for_file(path, iteration)
    with h5py.File(path, "r") as h5:
        record = h5["data"][iteration_key]["fields"][record_name]
        component = record[component_name] if isinstance(record, h5py.Group) else record
        axis_labels = decode_text_tuple(record.attrs["axisLabels"])
        grid_spacing = np.asarray(record.attrs["gridSpacing"], dtype=float)
        grid_offset = np.asarray(record.attrs["gridGlobalOffset"], dtype=float)
        grid_unit_si = float(record.attrs.get("gridUnitSI", 1.0))
        if "position" in component.attrs:
            position = np.asarray(component.attrs["position"], dtype=float)
        elif "position" in record.attrs:
            position = np.asarray(record.attrs["position"], dtype=float)
        else:
            position = np.full(len(axis_labels), 0.5, dtype=float)
        if len(axis_labels) != len(component.shape):
            raise ValueError(f"axisLabels {axis_labels} do not match component shape {component.shape}")
        if axis_name not in axis_labels:
            raise KeyError(f"Axis {axis_name!r} not found in {axis_labels}")
        axis_index = axis_labels.index(axis_name)
        indices = np.arange(int(component.shape[axis_index]), dtype=float)
        return (grid_offset[axis_index] + (indices + position[axis_index]) * grid_spacing[axis_index]) * grid_unit_si


def plane_to_yx(plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid = np.asarray(plane.coordinate("x"), dtype=float)
    y_grid = np.asarray(plane.coordinate("y"), dtype=float)
    values = np.asarray(plane.values_si, dtype=float)
    labels = tuple(plane.axis_labels)
    if labels == ("y", "x"):
        matrix = values
    elif labels == ("x", "y"):
        matrix = values.T
    else:
        raise ValueError(f"Unexpected xy plane axis labels: {labels}")
    expected_shape = (y_grid.size, x_grid.size)
    if matrix.shape != expected_shape:
        raise ValueError(f"XY matrix shape {matrix.shape} does not match expected {expected_shape}")
    return x_grid, y_grid, matrix


def read_bxy(path: Path, step: int, z_m: float, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx_plane = read_mesh_plane(path, "B", "x", plane="xy", coordinate_m=float(z_m), method=method, iteration=int(step))
    by_plane = read_mesh_plane(path, "B", "y", plane="xy", coordinate_m=float(z_m), method=method, iteration=int(step))
    x_bx, y_bx, bx = plane_to_yx(bx_plane)
    x_by, y_by, by = plane_to_yx(by_plane)
    if not np.allclose(x_bx, x_by, rtol=1.0e-12, atol=1.0e-18):
        raise ValueError("B/x and B/y x grids do not match")
    if not np.allclose(y_bx, y_by, rtol=1.0e-12, atol=1.0e-18):
        raise ValueError("B/x and B/y y grids do not match")
    return x_bx, y_bx, bx, by


def finite_stats(values: np.ndarray, prefix: str, *, scale: float = 1.0, suffix: str = "") -> dict[str, float | int]:
    scaled = np.asarray(values, dtype=float).ravel() * scale
    scaled = scaled[np.isfinite(scaled)]
    tag = f"_{suffix}" if suffix else ""
    if scaled.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_p50{tag}": math.nan,
            f"{prefix}_p90{tag}": math.nan,
            f"{prefix}_p95{tag}": math.nan,
            f"{prefix}_p99{tag}": math.nan,
            f"{prefix}_max{tag}": math.nan,
        }
    return {
        f"{prefix}_n": int(scaled.size),
        f"{prefix}_p50{tag}": float(np.percentile(scaled, 50.0)),
        f"{prefix}_p90{tag}": float(np.percentile(scaled, 90.0)),
        f"{prefix}_p95{tag}": float(np.percentile(scaled, 95.0)),
        f"{prefix}_p99{tag}": float(np.percentile(scaled, 99.0)),
        f"{prefix}_max{tag}": float(np.max(scaled)),
    }


def finite_stats_kT(values_T: np.ndarray, prefix: str) -> dict[str, float | int]:
    return finite_stats(values_T, prefix, scale=1.0e-3, suffix="kT")


def finite_stats_plain(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    return finite_stats(values, prefix, scale=1.0, suffix="")


def mask_box(x_grid: np.ndarray, y_grid: np.ndarray, cx_um: float, cy_um: float, half_width_um: float) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(x_grid * 1.0e6 - cx_um) <= half_width_um, np.abs(y_grid * 1.0e6 - cy_um) <= half_width_um


def mask_radius(x_grid: np.ndarray, y_grid: np.ndarray, cx_um: float, cy_um: float, radius_um: float) -> np.ndarray:
    xx_um, yy_um = np.meshgrid(x_grid * 1.0e6, y_grid * 1.0e6, indexing="xy")
    return (xx_um - cx_um) ** 2 + (yy_um - cy_um) ** 2 <= radius_um ** 2


def top_location(x_grid: np.ndarray, y_grid: np.ndarray, values_T: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_T, dtype=float)
    if not np.any(np.isfinite(values)):
        return {"top_x_um": math.nan, "top_y_um": math.nan, "top_B_kT": math.nan}
    flat_index = int(np.nanargmax(values))
    y_index, x_index = np.unravel_index(flat_index, values.shape)
    return {"top_x_um": float(x_grid[x_index] * 1.0e6), "top_y_um": float(y_grid[y_index] * 1.0e6), "top_B_kT": float(values[y_index, x_index] / 1.0e3)}


def save_bfield_plot(path: Path, x_grid: np.ndarray, y_grid: np.ndarray, values_T: np.ndarray, request: SliceRequest, title_prefix: str) -> None:
    values_kT = np.asarray(values_T, dtype=float) / 1.0e3
    vmax = np.nanpercentile(values_kT, 99.5) if np.any(np.isfinite(values_kT)) else 1.0
    vmax = max(float(vmax), 1.0)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.pcolormesh(x_grid * 1.0e6, y_grid * 1.0e6, values_kT, shading="auto", vmin=0.0, vmax=vmax)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"{title_prefix}: {request.label}, step {request.step}, xi={request.xi_um:.2f} um")
    ax.set_aspect("equal", adjustable="box")
    ax.plot([0.0], [0.0], marker="+", markersize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|B_perp| [kT]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_request(request: SliceRequest, *, field_method: str, lambda_m: float, lambda_samples: int, box_half_width_um: float, radius_um: float, output_dir: Path) -> dict[str, object]:
    field_path = step_field_file(request.case_dir, request.step)
    z_grid = axis_coordinates_m(field_path, record_name="B", component_name="x", axis_name="z", iteration=request.step)
    z_front_m = float(np.max(z_grid))
    target_z_m = z_front_m + request.xi_um * 1.0e-6
    x_grid, y_grid, bx, by = read_bxy(field_path, request.step, target_z_m, field_method)
    b_inst = np.hypot(bx, by)

    bx_samples = []
    by_samples = []
    bmag_samples = []
    sample_failures = []
    for offset in np.linspace(-0.5 * lambda_m, 0.5 * lambda_m, lambda_samples):
        sample_z_m = target_z_m + float(offset)
        try:
            xs, ys, bxs, bys = read_bxy(field_path, request.step, sample_z_m, field_method)
            if not np.allclose(xs, x_grid, rtol=1.0e-12, atol=1.0e-18) or not np.allclose(ys, y_grid, rtol=1.0e-12, atol=1.0e-18):
                raise ValueError("lambda sample grid mismatch")
            bx_samples.append(bxs)
            by_samples.append(bys)
            bmag_samples.append(np.hypot(bxs, bys))
        except Exception as exc:
            sample_failures.append({"offset_m": float(offset), "z_m": sample_z_m, "error": repr(exc)})

    if bx_samples:
        bx_avg = np.mean(np.stack(bx_samples, axis=0), axis=0)
        by_avg = np.mean(np.stack(by_samples, axis=0), axis=0)
        b_lambda_vec = np.hypot(bx_avg, by_avg)
        b_lambda_mag = np.mean(np.stack(bmag_samples, axis=0), axis=0)
        coherence = np.divide(b_lambda_vec, b_lambda_mag, out=np.full_like(b_lambda_vec, np.nan), where=np.isfinite(b_lambda_mag) & (b_lambda_mag > 0.0))
    else:
        b_lambda_vec = np.full_like(b_inst, np.nan)
        b_lambda_mag = np.full_like(b_inst, np.nan)
        coherence = np.full_like(b_inst, np.nan)

    row: dict[str, object] = {
        "label": request.label,
        "case_dir": str(request.case_dir),
        "field_file": str(field_path),
        "step": int(request.step),
        "xi_um": float(request.xi_um),
        "z_front_cell_center_um": z_front_m * 1.0e6,
        "target_z_um": target_z_m * 1.0e6,
        "z_min_cell_center_um": float(np.min(z_grid) * 1.0e6),
        "z_max_cell_center_um": float(np.max(z_grid) * 1.0e6),
        "field_method": field_method,
        "lambda_m": float(lambda_m),
        "lambda_samples_requested": int(lambda_samples),
        "lambda_samples_used": int(len(bx_samples)),
        "lambda_sample_failures": int(len(sample_failures)),
        "box_half_width_um": float(box_half_width_um),
        "radius_um": float(radius_um),
    }

    axis_x, axis_y = mask_box(x_grid, y_grid, 0.0, 0.0, box_half_width_um)
    if np.any(axis_x) and np.any(axis_y):
        box_values = np.ix_(axis_y, axis_x)
        row.update(finite_stats_kT(b_inst[box_values], "instant_box_axis"))
        row.update(finite_stats_kT(b_lambda_vec[box_values], "lambda_vec_box_axis"))
        row.update(finite_stats_kT(b_lambda_mag[box_values], "lambda_mag_box_axis"))
        row.update(finite_stats_plain(coherence[box_values], "coherence_box_axis"))
    else:
        row["axis_box_error"] = "box does not intersect grid"

    radial_mask = mask_radius(x_grid, y_grid, 0.0, 0.0, radius_um)
    row.update(finite_stats_kT(b_inst[radial_mask], "instant_radius_axis"))
    row.update(finite_stats_kT(b_lambda_vec[radial_mask], "lambda_vec_radius_axis"))
    row.update(finite_stats_kT(b_lambda_mag[radial_mask], "lambda_mag_radius_axis"))
    row.update(finite_stats_plain(coherence[radial_mask], "coherence_radius_axis"))

    row.update(finite_stats_kT(b_inst, "instant_full_plane"))
    row.update(finite_stats_kT(b_lambda_vec, "lambda_vec_full_plane"))
    row.update(finite_stats_kT(b_lambda_mag, "lambda_mag_full_plane"))
    row.update(finite_stats_plain(coherence, "coherence_full_plane"))
    row.update({f"instant_{key}": value for key, value in top_location(x_grid, y_grid, b_inst).items()})
    row.update({f"lambda_vec_{key}": value for key, value in top_location(x_grid, y_grid, b_lambda_vec).items()})
    row.update({f"lambda_mag_{key}": value for key, value in top_location(x_grid, y_grid, b_lambda_mag).items()})

    if sample_failures:
        failure_path = output_dir / f"{request.label}_lambda_sample_failures.json"
        failure_path.write_text(json.dumps(sample_failures, indent=2) + "\n", encoding="utf-8")
        row["lambda_sample_failures_json"] = str(failure_path)

    instant_plot = output_dir / f"{request.label}_instant_Bperp_xy_step{request.step:06d}_xi{request.xi_um:+06.2f}um.png"
    save_bfield_plot(instant_plot, x_grid, y_grid, b_inst, request, "instant")
    row["instant_Bperp_plot"] = str(instant_plot)

    lambda_plot = output_dir / f"{request.label}_lambda_vec_Bperp_xy_step{request.step:06d}_xi{request.xi_um:+06.2f}um.png"
    save_bfield_plot(lambda_plot, x_grid, y_grid, b_lambda_vec, request, "lambda-vector")
    row["lambda_vec_Bperp_plot"] = str(lambda_plot)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "label", "step", "xi_um", "target_z_um",
        "instant_box_axis_p95_kT", "instant_box_axis_p99_kT", "instant_box_axis_max_kT",
        "lambda_vec_box_axis_p95_kT", "lambda_mag_box_axis_p95_kT",
        "coherence_box_axis_p50", "coherence_box_axis_p95",
        "instant_radius_axis_p95_kT", "instant_radius_axis_p99_kT", "instant_radius_axis_max_kT",
        "instant_full_plane_max_kT", "instant_top_B_kT", "instant_top_x_um", "instant_top_y_um",
        "lambda_samples_used", "lambda_sample_failures", "instant_Bperp_plot", "lambda_vec_Bperp_plot",
    ]
    ordered = preferred + [name for name in fieldnames if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in ordered})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe instantaneous and lambda-averaged transverse B field at selected xi slices.")
    parser.add_argument("--soft-case", type=Path, required=True)
    parser.add_argument("--hard-case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--soft-step", type=int, default=900)
    parser.add_argument("--soft-xi-um", type=float, default=-17.0)
    parser.add_argument("--hard-step", type=int, default=1000)
    parser.add_argument("--hard-xi-um", type=float, default=-17.5)
    parser.add_argument("--field-method", default="linear")
    parser.add_argument("--lambda-m", type=float, default=8.0e-7)
    parser.add_argument("--lambda-samples", type=int, default=21)
    parser.add_argument("--box-half-width-um", type=float, default=2.5)
    parser.add_argument("--radius-um", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    if args.lambda_samples < 3 or args.lambda_samples % 2 == 0:
        raise ValueError("--lambda-samples must be an odd integer >= 3")
    args.output_dir.mkdir(parents=True)

    requests = [
        SliceRequest("soft_s0900_xim17p0", args.soft_case, args.soft_step, args.soft_xi_um),
        SliceRequest("hard_s1000_xim17p5", args.hard_case, args.hard_step, args.hard_xi_um),
    ]
    rows = [
        analyze_request(request, field_method=args.field_method, lambda_m=args.lambda_m, lambda_samples=args.lambda_samples, box_half_width_um=args.box_half_width_um, radius_um=args.radius_um, output_dir=args.output_dir)
        for request in requests
    ]

    csv_path = args.output_dir / "bfield_slice_probe.csv"
    json_path = args.output_dir / "bfield_slice_probe_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"CSV={csv_path}")
    print(f"SUMMARY={json_path}")


if __name__ == "__main__":
    main()
