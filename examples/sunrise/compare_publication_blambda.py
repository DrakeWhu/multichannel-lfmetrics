#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfmetrics.beam_metrics import weighted_percentile
from lfmetrics.constants import E_CHARGE_C
from lfmetrics.field_slices import read_mesh_plane


LAMBDA_M_DEFAULT = 8.0e-7
LAMBDA_SAMPLES_DEFAULT = 21
FIELD_METHOD_DEFAULT = "linear"
BOX_HALF_WIDTH_M_DEFAULT = 2.5e-6


def step_map(case_dir: Path, subdir: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in (case_dir / subdir).glob("openpmd_*.h5"):
        match = re.match(r"openpmd_(\d+)\.h5$", path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def weighted_percentile_or_nan(values: np.ndarray, percentile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return math.nan
    result = weighted_percentile(values[mask], percentile, weights[mask])
    return math.nan if result is None else float(result)


def weighted_mean_or_nan(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_rms_radius_um(
    x_m: np.ndarray,
    y_m: np.ndarray,
    weights: np.ndarray,
    center_x_m: float,
    center_y_m: float,
) -> float:
    radius_sq = (np.asarray(x_m, dtype=float) - center_x_m) ** 2
    radius_sq += (np.asarray(y_m, dtype=float) - center_y_m) ** 2
    mean_radius_sq = weighted_mean_or_nan(radius_sq, weights)
    if not np.isfinite(mean_radius_sq):
        return math.nan
    return math.sqrt(mean_radius_sq) * 1.0e6


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan
    return float(np.percentile(array, percentile))


def plane_to_xy(plane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError(
            f"XY matrix shape {matrix.shape} does not match expected {expected_shape}"
        )
    return x_grid, y_grid, matrix


def read_lambda_averaged_bmag_xy(
    field_h5: Path,
    step: int,
    center_z_m: float,
    *,
    lambda_m: float,
    lambda_samples: int,
    field_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.linspace(-0.5 * lambda_m, 0.5 * lambda_m, lambda_samples)

    bx_matrices: list[np.ndarray] = []
    by_matrices: list[np.ndarray] = []
    reference_x: np.ndarray | None = None
    reference_y: np.ndarray | None = None

    for offset in offsets:
        coordinate_m = float(center_z_m + offset)
        bx_plane = read_mesh_plane(
            field_h5,
            "B",
            "x",
            plane="xy",
            coordinate_m=coordinate_m,
            method=field_method,
            iteration=int(step),
        )
        by_plane = read_mesh_plane(
            field_h5,
            "B",
            "y",
            plane="xy",
            coordinate_m=coordinate_m,
            method=field_method,
            iteration=int(step),
        )

        bx_x, bx_y, bx_matrix = plane_to_xy(bx_plane)
        by_x, by_y, by_matrix = plane_to_xy(by_plane)

        if not np.allclose(bx_x, by_x, rtol=1.0e-12, atol=1.0e-18):
            raise ValueError("B/x and B/y x grids do not match")
        if not np.allclose(bx_y, by_y, rtol=1.0e-12, atol=1.0e-18):
            raise ValueError("B/x and B/y y grids do not match")

        if reference_x is None:
            reference_x = bx_x
            reference_y = bx_y
        else:
            if not np.allclose(reference_x, bx_x, rtol=1.0e-12, atol=1.0e-18):
                raise ValueError("lambda-averaged B/x x grids do not match")
            if not np.allclose(reference_y, bx_y, rtol=1.0e-12, atol=1.0e-18):
                raise ValueError("lambda-averaged B/x y grids do not match")

        bx_matrices.append(bx_matrix)
        by_matrices.append(by_matrix)

    if reference_x is None or reference_y is None:
        raise ValueError("No planes were read for lambda average")

    bx_average = np.mean(np.stack(bx_matrices, axis=0), axis=0)
    by_average = np.mean(np.stack(by_matrices, axis=0), axis=0)
    return reference_x, reference_y, np.hypot(bx_average, by_average)


def analyze_case(
    *,
    label: str,
    case_dir: Path,
    trajectory_npz: Path,
    energy_threshold_MeV: float,
    lambda_m: float,
    lambda_samples: int,
    field_method: str,
    box_half_width_m: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    particle_files = step_map(case_dir, "3D")
    field_files = step_map(case_dir, "fields3D")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with np.load(trajectory_npz) as data:
        steps = np.asarray(data["step"], dtype=int)
        present = np.asarray(data["present"], dtype=bool)
        x_m = np.asarray(data["x_m"], dtype=float)
        y_m = np.asarray(data["y_m"], dtype=float)
        z_m = np.asarray(data["z_m"], dtype=float)
        weights_all = np.asarray(data["weighting"], dtype=float)
        energy_MeV = np.asarray(data["energy_MeV"], dtype=float)

    for frame_index, raw_step in enumerate(steps):
        step = int(raw_step)
        if step not in particle_files or step not in field_files:
            continue

        mask = np.asarray(present[frame_index], dtype=bool)
        if not np.any(mask):
            continue

        weights = weights_all[frame_index, mask]
        x = x_m[frame_index, mask]
        y = y_m[frame_index, mask]
        z = z_m[frame_index, mask]
        energy = energy_MeV[frame_index, mask]

        charge_pC = float(np.nansum(weights) * E_CHARGE_C * 1.0e12)
        center_x_m = weighted_percentile_or_nan(x, 50.0, weights)
        center_y_m = weighted_percentile_or_nan(y, 50.0, weights)
        center_z_m = weighted_percentile_or_nan(z, 50.0, weights)

        base: dict[str, object] = {
            "case": label,
            "step": step,
            "selection_energy_threshold_MeV": float(energy_threshold_MeV),
            "tracked_present": int(np.count_nonzero(mask)),
            "tracked_charge_pC": charge_pC,
            "center_x_um": center_x_m * 1.0e6,
            "center_y_um": center_y_m * 1.0e6,
            "center_z_um": center_z_m * 1.0e6,
            "tracked_rms_radius_um": weighted_rms_radius_um(
                x,
                y,
                weights,
                center_x_m,
                center_y_m,
            ),
            "tracked_energy_p50_MeV": weighted_percentile_or_nan(energy, 50.0, weights),
            "tracked_energy_p90_MeV": weighted_percentile_or_nan(energy, 90.0, weights),
            "tracked_energy_p95_MeV": weighted_percentile_or_nan(energy, 95.0, weights),
            "tracked_energy_max_MeV": float(np.nanmax(energy)) if energy.size else math.nan,
        }

        try:
            x_grid, y_grid, bmag_t = read_lambda_averaged_bmag_xy(
                field_files[step],
                step,
                center_z_m,
                lambda_m=lambda_m,
                lambda_samples=lambda_samples,
                field_method=field_method,
            )

            x_mask = np.abs(x_grid - center_x_m) <= box_half_width_m
            y_mask = np.abs(y_grid - center_y_m) <= box_half_width_m
            if not np.any(x_mask) or not np.any(y_mask):
                raise ValueError("central B box does not intersect field grid")

            central_box_kT = bmag_t[np.ix_(y_mask, x_mask)] / 1.0e3

            p50 = finite_percentile(central_box_kT, 50.0)
            p90 = finite_percentile(central_box_kT, 90.0)
            p95 = finite_percentile(central_box_kT, 95.0)
            p99 = finite_percentile(central_box_kT, 99.0)
            bmax = finite_percentile(central_box_kT, 100.0)

            row = dict(base)
            row.update(
                {
                    "B_lambda_box2p5um_p50_kT": p50,
                    "B_lambda_box2p5um_p90_kT": p90,
                    "B_lambda_box2p5um_p95_kT": p95,
                    "B_lambda_box2p5um_p99_kT": p99,
                    "B_lambda_box2p5um_max_kT": bmax,
                    "B_lambda_box2p5um_p95_kT_per_100pC": (
                        math.nan if charge_pC <= 0.0 else p95 / charge_pC * 100.0
                    ),
                    "B_lambda_box2p5um_p99_kT_per_100pC": (
                        math.nan if charge_pC <= 0.0 else p99 / charge_pC * 100.0
                    ),
                }
            )
            rows.append(row)

        except Exception as exc:
            failure = dict(base)
            failure["error"] = repr(exc)
            failures.append(failure)

    return rows, failures


FIELDNAMES = [
    "case",
    "step",
    "selection_energy_threshold_MeV",
    "tracked_present",
    "tracked_charge_pC",
    "center_x_um",
    "center_y_um",
    "center_z_um",
    "tracked_rms_radius_um",
    "tracked_energy_p50_MeV",
    "tracked_energy_p90_MeV",
    "tracked_energy_p95_MeV",
    "tracked_energy_max_MeV",
    "B_lambda_box2p5um_p50_kT",
    "B_lambda_box2p5um_p90_kT",
    "B_lambda_box2p5um_p95_kT",
    "B_lambda_box2p5um_p99_kT",
    "B_lambda_box2p5um_max_kT",
    "B_lambda_box2p5um_p95_kT_per_100pC",
    "B_lambda_box2p5um_p99_kT_per_100pC",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, math.nan)
        try:
            converted = float(value)
        except Exception:
            converted = math.nan
        if np.isfinite(converted):
            values.append(converted)
    return np.asarray(values, dtype=float)


def median_or_nan(rows: list[dict[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return math.nan if values.size == 0 else float(np.median(values))


def max_or_nan(rows: list[dict[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return math.nan if values.size == 0 else float(np.max(values))


def summarize_case(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    selected = sorted(
        [row for row in rows if row["case"] == label],
        key=lambda row: int(row["step"]),
    )
    if not selected:
        return {
            "n_frames": 0,
            "first_step": None,
            "last_step": None,
        }

    return {
        "n_frames": len(selected),
        "first_step": int(selected[0]["step"]),
        "last_step": int(selected[-1]["step"]),
        "charge_pC_median": median_or_nan(selected, "tracked_charge_pC"),
        "tracked_rms_radius_um_median": median_or_nan(selected, "tracked_rms_radius_um"),
        "energy_p50_MeV_final": selected[-1]["tracked_energy_p50_MeV"],
        "energy_p95_MeV_final": selected[-1]["tracked_energy_p95_MeV"],
        "B_lambda_box2p5um_p95_kT_median": median_or_nan(
            selected,
            "B_lambda_box2p5um_p95_kT",
        ),
        "B_lambda_box2p5um_p95_kT_max": max_or_nan(
            selected,
            "B_lambda_box2p5um_p95_kT",
        ),
        "B_lambda_box2p5um_p99_kT_median": median_or_nan(
            selected,
            "B_lambda_box2p5um_p99_kT",
        ),
        "B_lambda_box2p5um_p99_kT_max": max_or_nan(
            selected,
            "B_lambda_box2p5um_p99_kT",
        ),
        "B_lambda_box2p5um_p95_kT_per_100pC_median": median_or_nan(
            selected,
            "B_lambda_box2p5um_p95_kT_per_100pC",
        ),
        "B_lambda_box2p5um_p99_kT_per_100pC_median": median_or_nan(
            selected,
            "B_lambda_box2p5um_p99_kT_per_100pC",
        ),
    }


def plot_metric(rows: list[dict[str, object]], key: str, ylabel: str, filename: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for label in ("SOFT_i022_c003_E20", "HARD_i017_c000_E100"):
        selected = sorted(
            [row for row in rows if row["case"] == label],
            key=lambda row: int(row["step"]),
        )
        if not selected:
            continue
        x_values = [int(row["step"]) for row in selected]
        y_values = [float(row.get(key, math.nan)) for row in selected]
        ax.plot(x_values, y_values, marker="o", markersize=3.0, linewidth=1.2, label=label)

    ax.set_xlabel("WarpX step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare lambda-averaged transverse magnetic field metrics for SOFT and HARD publication cases."
    )
    parser.add_argument("--soft-case", type=Path, required=True)
    parser.add_argument("--hard-case", type=Path, required=True)
    parser.add_argument("--soft-traj", type=Path, required=True)
    parser.add_argument("--hard-traj", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-m", type=float, default=LAMBDA_M_DEFAULT)
    parser.add_argument("--lambda-samples", type=int, default=LAMBDA_SAMPLES_DEFAULT)
    parser.add_argument("--field-method", default=FIELD_METHOD_DEFAULT)
    parser.add_argument("--box-half-width-um", type=float, default=BOX_HALF_WIDTH_M_DEFAULT * 1.0e6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.lambda_samples < 3 or args.lambda_samples % 2 == 0:
        raise ValueError("--lambda-samples must be an odd integer >= 3")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    box_half_width_m = float(args.box_half_width_um) * 1.0e-6

    soft_rows, soft_failures = analyze_case(
        label="SOFT_i022_c003_E20",
        case_dir=args.soft_case,
        trajectory_npz=args.soft_traj,
        energy_threshold_MeV=20.0,
        lambda_m=args.lambda_m,
        lambda_samples=args.lambda_samples,
        field_method=args.field_method,
        box_half_width_m=box_half_width_m,
    )
    hard_rows, hard_failures = analyze_case(
        label="HARD_i017_c000_E100",
        case_dir=args.hard_case,
        trajectory_npz=args.hard_traj,
        energy_threshold_MeV=100.0,
        lambda_m=args.lambda_m,
        lambda_samples=args.lambda_samples,
        field_method=args.field_method,
        box_half_width_m=box_half_width_m,
    )

    rows = soft_rows + hard_rows
    failures = soft_failures + hard_failures
    if not rows:
        raise RuntimeError("no comparison rows generated")

    csv_path = args.output_dir / "hard_vs_soft_blambda_frame_metrics.csv"
    failures_path = args.output_dir / "hard_vs_soft_blambda_failures.json"
    summary_path = args.output_dir / "hard_vs_soft_blambda_summary.json"

    write_csv(csv_path, rows)
    failures_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")

    summary = {
        "schema": "hard_vs_soft_blambda_compare_v2",
        "soft_case": str(args.soft_case),
        "hard_case": str(args.hard_case),
        "soft_trajectory_npz": str(args.soft_traj),
        "hard_trajectory_npz": str(args.hard_traj),
        "lambda_average_m": float(args.lambda_m),
        "lambda_samples": int(args.lambda_samples),
        "field_method": str(args.field_method),
        "box_half_width_um": float(args.box_half_width_um),
        "csv": str(csv_path),
        "failures_json": str(failures_path),
        "n_rows": len(rows),
        "n_failures": len(failures),
        "case_summaries": {
            "SOFT_i022_c003_E20": summarize_case("SOFT_i022_c003_E20", rows),
            "HARD_i017_c000_E100": summarize_case("HARD_i017_c000_E100", rows),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_metric(
        rows,
        "B_lambda_box2p5um_p95_kT",
        "p95 |B_perp lambda-avg| [kT], central box",
        "blambda_p95_box2p5_vs_step.png",
        args.output_dir,
    )
    plot_metric(
        rows,
        "B_lambda_box2p5um_p99_kT",
        "p99 |B_perp lambda-avg| [kT], central box",
        "blambda_p99_box2p5_vs_step.png",
        args.output_dir,
    )
    plot_metric(
        rows,
        "B_lambda_box2p5um_p95_kT_per_100pC",
        "p95 |B_perp lambda-avg| per 100 pC [kT/100pC]",
        "blambda_p95_box2p5_per100pC_vs_step.png",
        args.output_dir,
    )
    plot_metric(
        rows,
        "tracked_charge_pC",
        "tracked charge present [pC]",
        "tracked_charge_vs_step.png",
        args.output_dir,
    )
    plot_metric(
        rows,
        "tracked_energy_p50_MeV",
        "tracked energy p50 [MeV]",
        "tracked_energy_p50_vs_step.png",
        args.output_dir,
    )
    plot_metric(
        rows,
        "tracked_rms_radius_um",
        "tracked transverse rms radius [um]",
        "tracked_rms_radius_vs_step.png",
        args.output_dir,
    )

    print(json.dumps(summary, indent=2))
    print(f"CSV={csv_path}")
    print(f"SUMMARY={summary_path}")
    print(f"FAILURES={failures_path}")


if __name__ == "__main__":
    main()
