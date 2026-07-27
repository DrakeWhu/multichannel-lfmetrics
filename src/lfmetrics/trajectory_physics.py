from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .beam_metrics import pz_MeV_c, weighted_mean, weighted_percentile, weighted_std
from .constants import E_CHARGE_C
from .trajectory_analysis import (
    DEFAULT_ENERGY_THRESHOLDS_MEV,
    TrajectoryData,
    load_trajectory_npz,
)


TRAJECTORY_PHYSICS_SCHEMA = "multichannel_trajectory_physics_v1"


@dataclass(frozen=True)
class TrajectoryPhysicsContext:
    plasma_start_m: float
    window_front_z0_m: float
    moving_window_velocity_m_s: float

    def validate(self) -> None:
        values = {
            "plasma_start_m": self.plasma_start_m,
            "window_front_z0_m": self.window_front_z0_m,
            "moving_window_velocity_m_s": self.moving_window_velocity_m_s,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.moving_window_velocity_m_s <= 0.0:
            raise ValueError("moving_window_velocity_m_s must be positive")

    def window_front_z_m(self, times_s: np.ndarray | float) -> np.ndarray:
        self.validate()
        return self.window_front_z0_m + self.moving_window_velocity_m_s * np.asarray(
            times_s, dtype=float
        )


def _first_true_index(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    any_true = np.any(mask, axis=0)
    index = np.argmax(mask, axis=0).astype(np.int64)
    index[~any_true] = -1
    return index


def _threshold_crossing_index(
    values: np.ndarray,
    present: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return _first_true_index(present & np.isfinite(values) & (values >= threshold))


def _first_sustained_forward_index(pz_si: np.ndarray, present: np.ndarray) -> np.ndarray:
    forward = present & np.isfinite(pz_si) & (pz_si > 0.0)
    sustained = np.logical_and.accumulate(forward[::-1], axis=0)[::-1]
    return _first_true_index(sustained)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    mask = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(w_arr)
        & (w_arr >= 0.0)
    )
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    w_arr = w_arr[mask]
    if x_arr.size < 2 or float(np.sum(w_arr)) <= 0.0:
        return None

    x_mean = weighted_mean(x_arr, w_arr)
    y_mean = weighted_mean(y_arr, w_arr)
    if x_mean is None or y_mean is None:
        return None

    wsum = float(np.sum(w_arr))
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    covariance = float(np.sum(w_arr * dx * dy) / wsum)
    variance_x = float(np.sum(w_arr * dx * dx) / wsum)
    variance_y = float(np.sum(w_arr * dy * dy) / wsum)
    denominator = float(np.sqrt(variance_x * variance_y))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return None
    return float(covariance / denominator)


def _physical_coordinates(
    data: TrajectoryData,
    context: TrajectoryPhysicsContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context.validate()
    time_fs = data.times_s * 1.0e15
    z_from_plasma_um = (data.z_m - context.plasma_start_m) * 1.0e6
    front_z_m = context.window_front_z_m(data.times_s)[:, None]
    xi_window_um = (data.z_m - front_z_m) * 1.0e6
    return time_fs, z_from_plasma_um, xi_window_um


def build_physical_particle_rows(
    data: TrajectoryData,
    context: TrajectoryPhysicsContext,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> list[dict[str, object]]:
    data.validate()
    context.validate()
    first_index = _first_true_index(data.present)
    if np.any(first_index < 0):
        raise ValueError("every tracked particle must appear in at least one frame")

    time_fs, z_from_plasma_um, xi_window_um = _physical_coordinates(data, context)
    pz_mev_c = pz_MeV_c(data.pz_si)
    forward_index = _first_sustained_forward_index(data.pz_si, data.present)
    crossing_indices = {
        float(threshold): _threshold_crossing_index(
            data.energy_MeV, data.present, float(threshold)
        )
        for threshold in energy_thresholds_MeV
    }

    rows: list[dict[str, object]] = []
    for particle_index, particle_id in enumerate(data.ids):
        first = int(first_index[particle_index])
        first_front_z_um = float(
            context.window_front_z_m(data.times_s[first]) * 1.0e6
        )
        row: dict[str, object] = {
            "id": int(particle_id),
            "first_seen_step": int(data.steps[first]),
            "first_seen_time_fs": float(time_fs[first]),
            "first_x_um": float(data.x_m[first, particle_index] * 1.0e6),
            "first_y_um": float(data.y_m[first, particle_index] * 1.0e6),
            "first_r_um": float(
                np.hypot(
                    data.x_m[first, particle_index],
                    data.y_m[first, particle_index],
                )
                * 1.0e6
            ),
            "first_z_um": float(data.z_m[first, particle_index] * 1.0e6),
            "first_z_from_plasma_start_um": float(
                z_from_plasma_um[first, particle_index]
            ),
            "window_front_z_at_first_seen_um": first_front_z_um,
            "first_xi_window_um": float(xi_window_um[first, particle_index]),
            "first_energy_MeV": float(data.energy_MeV[first, particle_index]),
            "first_pz_MeV_c": float(pz_mev_c[first, particle_index]),
            "final_energy_MeV": float(data.energy_MeV[-1, particle_index]),
            "final_pz_MeV_c": float(pz_mev_c[-1, particle_index]),
            "final_weighting": float(data.weighting[-1, particle_index]),
        }

        fwd = int(forward_index[particle_index])
        row["first_sustained_forward_step"] = None if fwd < 0 else int(data.steps[fwd])
        row["first_sustained_forward_time_fs"] = (
            None if fwd < 0 else float(time_fs[fwd])
        )
        row["first_sustained_forward_z_from_plasma_start_um"] = (
            None if fwd < 0 else float(z_from_plasma_um[fwd, particle_index])
        )
        row["first_sustained_forward_xi_window_um"] = (
            None if fwd < 0 else float(xi_window_um[fwd, particle_index])
        )

        for threshold, indices in crossing_indices.items():
            crossing = int(indices[particle_index])
            stem = f"first_E_ge_{threshold:g}_MeV"
            row[f"{stem}_step"] = (
                None if crossing < 0 else int(data.steps[crossing])
            )
            row[f"{stem}_time_fs"] = (
                None if crossing < 0 else float(time_fs[crossing])
            )
            row[f"{stem}_z_from_plasma_start_um"] = (
                None
                if crossing < 0
                else float(z_from_plasma_um[crossing, particle_index])
            )
            row[f"{stem}_xi_window_um"] = (
                None
                if crossing < 0
                else float(xi_window_um[crossing, particle_index])
            )
            row[f"{stem}_pz_MeV_c"] = (
                None if crossing < 0 else float(pz_mev_c[crossing, particle_index])
            )
        rows.append(row)
    return rows


def build_physical_threshold_rows(
    data: TrajectoryData,
    context: TrajectoryPhysicsContext,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> list[dict[str, object]]:
    time_fs, z_from_plasma_um, xi_window_um = _physical_coordinates(data, context)
    pz_mev_c = pz_MeV_c(data.pz_si)
    rows: list[dict[str, object]] = []
    for threshold in energy_thresholds_MeV:
        indices = _threshold_crossing_index(
            data.energy_MeV, data.present, float(threshold)
        )
        for particle_index, particle_id in enumerate(data.ids):
            crossing = int(indices[particle_index])
            rows.append(
                {
                    "id": int(particle_id),
                    "threshold_MeV": float(threshold),
                    "crossed": crossing >= 0,
                    "first_crossing_step": (
                        None if crossing < 0 else int(data.steps[crossing])
                    ),
                    "first_crossing_time_fs": (
                        None if crossing < 0 else float(time_fs[crossing])
                    ),
                    "z_from_plasma_start_at_crossing_um": (
                        None
                        if crossing < 0
                        else float(z_from_plasma_um[crossing, particle_index])
                    ),
                    "xi_window_at_crossing_um": (
                        None
                        if crossing < 0
                        else float(xi_window_um[crossing, particle_index])
                    ),
                    "pz_at_crossing_MeV_c": (
                        None
                        if crossing < 0
                        else float(pz_mev_c[crossing, particle_index])
                    ),
                }
            )
    return rows


def build_physical_cohort_rows(
    data: TrajectoryData,
    context: TrajectoryPhysicsContext,
) -> list[dict[str, object]]:
    first_index = _first_true_index(data.present)
    time_fs, z_from_plasma_um, xi_window_um = _physical_coordinates(data, context)
    first_steps = data.steps[first_index]
    final_weights = data.weighting[-1]
    final_energy = data.energy_MeV[-1]
    rows: list[dict[str, object]] = []

    for step in np.unique(first_steps):
        mask = first_steps == step
        particle_indices = np.flatnonzero(mask)
        frame_indices = first_index[mask]
        first_z = z_from_plasma_um[frame_indices, particle_indices]
        first_xi = xi_window_um[frame_indices, particle_indices]
        first_r = (
            np.hypot(
                data.x_m[frame_indices, particle_indices],
                data.y_m[frame_indices, particle_indices],
            )
            * 1.0e6
        )
        weights = final_weights[mask]
        energy = final_energy[mask]
        rows.append(
            {
                "first_seen_step": int(step),
                "first_seen_time_fs": float(time_fs[int(frame_indices[0])]),
                "n_particles": int(np.count_nonzero(mask)),
                "weight_sum": float(np.sum(weights)),
                "charge_pC": float(np.sum(weights) * E_CHARGE_C * 1.0e12),
                "first_z_from_plasma_start_mean_um": weighted_mean(
                    first_z, weights
                ),
                "first_z_from_plasma_start_std_um": weighted_std(first_z, weights),
                "first_xi_window_mean_um": weighted_mean(first_xi, weights),
                "first_r_mean_um": weighted_mean(first_r, weights),
                "final_energy_mean_MeV": weighted_mean(energy, weights),
                "final_energy_p50_MeV": weighted_percentile(energy, 50.0, weights),
                "final_energy_p90_MeV": weighted_percentile(energy, 90.0, weights),
                "corr_first_z_rel_final_energy": weighted_correlation(
                    first_z, energy, weights
                ),
                "corr_first_xi_final_energy": weighted_correlation(
                    first_xi, energy, weights
                ),
                "corr_first_r_final_energy": weighted_correlation(
                    first_r, energy, weights
                ),
            }
        )
    return rows


def _weighted_frame_mean(
    values: np.ndarray,
    weights: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        mask = (
            present[frame]
            & np.isfinite(values[frame])
            & np.isfinite(weights[frame])
        )
        mean = weighted_mean(values[frame, mask], weights[frame, mask])
        if mean is not None:
            result[frame] = mean
    return result


def write_physical_trajectory_plots(
    data: TrajectoryData,
    context: TrajectoryPhysicsContext,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    first_index = _first_true_index(data.present)
    columns = np.arange(data.n_particles)
    time_fs, z_from_plasma_um, xi_window_um = _physical_coordinates(data, context)
    first_steps = data.steps[first_index]
    first_z_rel = z_from_plasma_um[first_index, columns]
    first_xi = xi_window_um[first_index, columns]
    final_energy = data.energy_MeV[-1]
    pz_mev_c = pz_MeV_c(data.pz_si)
    paths: list[Path] = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    plt.figure()
    plt.scatter(first_z_rel, final_energy, s=4, alpha=0.35)
    plt.xlabel("First observed z - plasma start [um]")
    plt.ylabel("Final kinetic energy [MeV]")
    save("origin_z_from_plasma_vs_final_energy.png")

    plt.figure()
    plt.scatter(first_xi, final_energy, s=4, alpha=0.35)
    plt.xlabel("First observed xi_window = z - z_front [um]")
    plt.ylabel("Final kinetic energy [MeV]")
    save("origin_xi_window_vs_final_energy.png")

    plt.figure()
    for step in np.unique(first_steps):
        cohort = first_steps == step
        mean_energy = _weighted_frame_mean(
            data.energy_MeV[:, cohort],
            data.weighting[:, cohort],
            data.present[:, cohort],
        )
        plt.plot(
            time_fs,
            mean_energy,
            marker="o",
            markersize=2,
            label=f"first {int(step)}",
        )
    plt.xlabel("Time [fs]")
    plt.ylabel("Weighted mean kinetic energy [MeV]")
    plt.legend(fontsize="small")
    save("energy_evolution_time_by_cohort.png")

    plt.figure()
    for step in np.unique(first_steps):
        cohort = first_steps == step
        mean_pz = _weighted_frame_mean(
            pz_mev_c[:, cohort],
            data.weighting[:, cohort],
            data.present[:, cohort],
        )
        plt.plot(
            time_fs,
            mean_pz,
            marker="o",
            markersize=2,
            label=f"first {int(step)}",
        )
    plt.xlabel("Time [fs]")
    plt.ylabel("Weighted mean pz [MeV/c]")
    plt.legend(fontsize="small")
    save("pz_evolution_time_by_cohort.png")

    return paths


def analyze_trajectory_physics_file(
    trajectory_npz: str | Path,
    output_dir: str | Path,
    context: TrajectoryPhysicsContext,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> dict[str, object]:
    data = load_trajectory_npz(trajectory_npz)
    context.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    particle_rows = build_physical_particle_rows(
        data, context, energy_thresholds_MeV
    )
    threshold_rows = build_physical_threshold_rows(
        data, context, energy_thresholds_MeV
    )
    cohort_rows = build_physical_cohort_rows(data, context)
    _write_csv(output / "particle_physics.csv", particle_rows)
    _write_csv(output / "threshold_physics.csv", threshold_rows)
    _write_csv(output / "cohort_physics.csv", cohort_rows)
    plots = write_physical_trajectory_plots(data, context, output / "plots")

    first_index = _first_true_index(data.present)
    columns = np.arange(data.n_particles)
    _, z_from_plasma_um, xi_window_um = _physical_coordinates(data, context)
    first_z_rel = z_from_plasma_um[first_index, columns]
    first_xi = xi_window_um[first_index, columns]
    first_r = (
        np.hypot(
            data.x_m[first_index, columns],
            data.y_m[first_index, columns],
        )
        * 1.0e6
    )
    final_energy = data.energy_MeV[-1]
    final_weights = data.weighting[-1]

    summary: dict[str, object] = {
        "schema": TRAJECTORY_PHYSICS_SCHEMA,
        "source_trajectory_npz": str(Path(trajectory_npz).resolve(strict=False)),
        "n_particles": data.n_particles,
        "n_frames": data.n_frames,
        "context": {
            "plasma_start_m": float(context.plasma_start_m),
            "window_front_z0_m": float(context.window_front_z0_m),
            "moving_window_velocity_m_s": float(
                context.moving_window_velocity_m_s
            ),
            "xi_window_definition": (
                "z_particle - (window_front_z0 + moving_window_velocity * time)"
            ),
        },
        "energy_thresholds_MeV": [float(value) for value in energy_thresholds_MeV],
        "first_seen_is_not_capture_time": True,
        "weighted_origin": {
            "first_z_from_plasma_start_p10_um": weighted_percentile(
                first_z_rel, 10.0, final_weights
            ),
            "first_z_from_plasma_start_p50_um": weighted_percentile(
                first_z_rel, 50.0, final_weights
            ),
            "first_z_from_plasma_start_p90_um": weighted_percentile(
                first_z_rel, 90.0, final_weights
            ),
            "first_r_p10_um": weighted_percentile(first_r, 10.0, final_weights),
            "first_r_p50_um": weighted_percentile(first_r, 50.0, final_weights),
            "first_r_p90_um": weighted_percentile(first_r, 90.0, final_weights),
            "first_xi_window_p10_um": weighted_percentile(
                first_xi, 10.0, final_weights
            ),
            "first_xi_window_p50_um": weighted_percentile(
                first_xi, 50.0, final_weights
            ),
            "first_xi_window_p90_um": weighted_percentile(
                first_xi, 90.0, final_weights
            ),
        },
        "weighted_correlations": {
            "first_z_from_plasma_start_vs_final_energy": weighted_correlation(
                first_z_rel, final_energy, final_weights
            ),
            "first_xi_window_vs_final_energy": weighted_correlation(
                first_xi, final_energy, final_weights
            ),
            "first_r_vs_final_energy": weighted_correlation(
                first_r, final_energy, final_weights
            ),
        },
        "plots": [str(path.name) for path in plots],
    }
    (output / "trajectory_physics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
