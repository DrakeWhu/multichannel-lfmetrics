from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .beam_metrics import weighted_mean, weighted_percentile, weighted_std


TRAJECTORY_ANALYSIS_SCHEMA = "multichannel_trajectory_analysis_v1"
DEFAULT_ENERGY_THRESHOLDS_MEV = (1.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class TrajectoryData:
    ids: np.ndarray
    steps: np.ndarray
    times_s: np.ndarray
    present: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    px_si: np.ndarray
    py_si: np.ndarray
    pz_si: np.ndarray
    weighting: np.ndarray
    energy_MeV: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.steps.size)

    @property
    def n_particles(self) -> int:
        return int(self.ids.size)

    def validate(self) -> None:
        if self.ids.ndim != 1 or self.ids.size == 0:
            raise ValueError("id must be a non-empty one-dimensional array")
        if self.steps.ndim != 1 or self.steps.size == 0:
            raise ValueError("step must be a non-empty one-dimensional array")
        if self.times_s.shape != self.steps.shape:
            raise ValueError("time_s and step must have matching shapes")
        expected = (self.n_frames, self.n_particles)
        for name in (
            "present",
            "x_m",
            "y_m",
            "z_m",
            "px_si",
            "py_si",
            "pz_si",
            "weighting",
            "energy_MeV",
        ):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {expected}")
        if np.unique(self.ids).size != self.ids.size:
            raise ValueError("particle IDs must be unique")
        if not np.all(np.diff(self.steps) > 0):
            raise ValueError("diagnostic steps must be strictly increasing")
        if not np.all(self.present[-1]):
            raise ValueError("all tracked particles must be present in the final frame")


def load_trajectory_npz(path: str | Path) -> TrajectoryData:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "id", "step", "time_s", "present", "x_m", "y_m", "z_m",
            "px_si", "py_si", "pz_si", "weighting", "energy_MeV",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing trajectory records: {sorted(missing)}")
        result = TrajectoryData(
            ids=np.asarray(data["id"], dtype=np.uint64),
            steps=np.asarray(data["step"], dtype=np.int64),
            times_s=np.asarray(data["time_s"], dtype=float),
            present=np.asarray(data["present"], dtype=bool),
            x_m=np.asarray(data["x_m"], dtype=float),
            y_m=np.asarray(data["y_m"], dtype=float),
            z_m=np.asarray(data["z_m"], dtype=float),
            px_si=np.asarray(data["px_si"], dtype=float),
            py_si=np.asarray(data["py_si"], dtype=float),
            pz_si=np.asarray(data["pz_si"], dtype=float),
            weighting=np.asarray(data["weighting"], dtype=float),
            energy_MeV=np.asarray(data["energy_MeV"], dtype=float),
        )
    result.validate()
    return result


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


def _value_at_first(array: np.ndarray, first_index: np.ndarray) -> np.ndarray:
    columns = np.arange(first_index.size)
    return array[first_index, columns]


def build_particle_history_rows(
    data: TrajectoryData,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> list[dict[str, object]]:
    data.validate()
    first_index = _first_true_index(data.present)
    if np.any(first_index < 0):
        raise ValueError("every tracked particle must appear in at least one frame")
    forward_index = _first_sustained_forward_index(data.pz_si, data.present)
    crossing_indices = {
        float(threshold): _threshold_crossing_index(
            data.energy_MeV, data.present, float(threshold)
        )
        for threshold in energy_thresholds_MeV
    }
    columns = np.arange(data.n_particles)
    rows: list[dict[str, object]] = []
    for particle_index, particle_id in enumerate(data.ids):
        first = int(first_index[particle_index])
        row: dict[str, object] = {
            "id": int(particle_id),
            "first_seen_step": int(data.steps[first]),
            "first_seen_time_s": float(data.times_s[first]),
            "first_x_um": float(data.x_m[first, particle_index] * 1.0e6),
            "first_y_um": float(data.y_m[first, particle_index] * 1.0e6),
            "first_z_um": float(data.z_m[first, particle_index] * 1.0e6),
            "first_energy_MeV": float(data.energy_MeV[first, particle_index]),
            "first_pz_si": float(data.pz_si[first, particle_index]),
            "final_x_um": float(data.x_m[-1, particle_index] * 1.0e6),
            "final_y_um": float(data.y_m[-1, particle_index] * 1.0e6),
            "final_z_um": float(data.z_m[-1, particle_index] * 1.0e6),
            "final_energy_MeV": float(data.energy_MeV[-1, particle_index]),
            "final_pz_si": float(data.pz_si[-1, particle_index]),
            "final_weighting": float(data.weighting[-1, particle_index]),
        }
        fwd = int(forward_index[particle_index])
        row["first_sustained_forward_step"] = None if fwd < 0 else int(data.steps[fwd])
        for threshold, index in crossing_indices.items():
            crossing = int(index[particle_index])
            key = f"first_E_ge_{threshold:g}_MeV_step"
            row[key] = None if crossing < 0 else int(data.steps[crossing])
        rows.append(row)
    del columns
    return rows


def build_threshold_crossing_rows(
    data: TrajectoryData,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in energy_thresholds_MeV:
        index = _threshold_crossing_index(data.energy_MeV, data.present, float(threshold))
        for particle_index, particle_id in enumerate(data.ids):
            crossing = int(index[particle_index])
            rows.append({
                "id": int(particle_id),
                "threshold_MeV": float(threshold),
                "crossed": crossing >= 0,
                "first_crossing_step": None if crossing < 0 else int(data.steps[crossing]),
                "first_crossing_time_s": None if crossing < 0 else float(data.times_s[crossing]),
                "z_at_crossing_um": None if crossing < 0 else float(data.z_m[crossing, particle_index] * 1.0e6),
            })
    return rows


def build_cohort_summary_rows(data: TrajectoryData) -> list[dict[str, object]]:
    first_index = _first_true_index(data.present)
    first_steps = data.steps[first_index]
    final_weights = data.weighting[-1]
    final_energy = data.energy_MeV[-1]
    rows: list[dict[str, object]] = []
    for step in np.unique(first_steps):
        mask = first_steps == step
        cohort_index = np.flatnonzero(mask)
        cohort_first_index = first_index[mask]
        first_z = data.z_m[cohort_first_index, cohort_index] * 1.0e6
        first_r = np.sqrt(
            data.x_m[cohort_first_index, cohort_index] ** 2
            + data.y_m[cohort_first_index, cohort_index] ** 2
        ) * 1.0e6
        weights = final_weights[mask]
        energy = final_energy[mask]
        rows.append({
            "first_seen_step": int(step),
            "n_particles": int(np.count_nonzero(mask)),
            "weight_sum": float(np.sum(weights)),
            "first_z_mean_um": weighted_mean(first_z, weights),
            "first_z_std_um": weighted_std(first_z, weights),
            "first_r_mean_um": weighted_mean(first_r, weights),
            "final_energy_mean_MeV": weighted_mean(energy, weights),
            "final_energy_p10_MeV": weighted_percentile(energy, 10.0, weights),
            "final_energy_p50_MeV": weighted_percentile(energy, 50.0, weights),
            "final_energy_p90_MeV": weighted_percentile(energy, 90.0, weights),
            "final_energy_max_MeV": float(np.max(energy)),
        })
    return rows


def _weighted_frame_mean(values: np.ndarray, weights: np.ndarray, present: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    for frame in range(values.shape[0]):
        mask = present[frame] & np.isfinite(values[frame]) & np.isfinite(weights[frame])
        mean = weighted_mean(values[frame, mask], weights[frame, mask])
        if mean is not None:
            result[frame] = mean
    return result


def write_trajectory_plots(data: TrajectoryData, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    first_index = _first_true_index(data.present)
    columns = np.arange(data.n_particles)
    first_z_um = data.z_m[first_index, columns] * 1.0e6
    first_x_um = data.x_m[first_index, columns] * 1.0e6
    first_y_um = data.y_m[first_index, columns] * 1.0e6
    first_steps = data.steps[first_index]
    final_energy = data.energy_MeV[-1]
    paths: list[Path] = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    plt.figure()
    plt.scatter(first_z_um, final_energy, s=4, alpha=0.35)
    plt.xlabel("First observed z [um]")
    plt.ylabel("Final kinetic energy [MeV]")
    save("origin_z_vs_final_energy.png")

    plt.figure()
    scatter = plt.scatter(first_x_um, first_y_um, c=final_energy, s=5, alpha=0.5)
    plt.xlabel("First observed x [um]")
    plt.ylabel("First observed y [um]")
    plt.colorbar(scatter, label="Final kinetic energy [MeV]")
    save("origin_xy_vs_final_energy.png")

    plt.figure()
    plt.scatter(first_steps, final_energy, s=5, alpha=0.4)
    plt.xlabel("First observed diagnostic step")
    plt.ylabel("Final kinetic energy [MeV]")
    save("first_seen_step_vs_final_energy.png")

    plt.figure()
    for step in np.unique(first_steps):
        cohort = first_steps == step
        mean_energy = _weighted_frame_mean(
            data.energy_MeV[:, cohort], data.weighting[:, cohort], data.present[:, cohort]
        )
        plt.plot(data.steps, mean_energy, marker="o", markersize=2, label=f"first {int(step)}")
    plt.xlabel("Diagnostic step")
    plt.ylabel("Weighted mean kinetic energy [MeV]")
    plt.legend(fontsize="small")
    save("energy_evolution_by_cohort.png")

    plt.figure()
    for step in np.unique(first_steps):
        cohort = first_steps == step
        mean_pz = _weighted_frame_mean(
            data.pz_si[:, cohort], data.weighting[:, cohort], data.present[:, cohort]
        )
        plt.plot(data.steps, mean_pz, marker="o", markersize=2, label=f"first {int(step)}")
    plt.xlabel("Diagnostic step")
    plt.ylabel("Weighted mean pz [SI]")
    plt.legend(fontsize="small")
    save("pz_evolution_by_cohort.png")

    return paths


def analyze_trajectory_file(
    trajectory_npz: str | Path,
    output_dir: str | Path,
    energy_thresholds_MeV: Sequence[float] = DEFAULT_ENERGY_THRESHOLDS_MEV,
) -> dict[str, object]:
    data = load_trajectory_npz(trajectory_npz)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    particle_rows = build_particle_history_rows(data, energy_thresholds_MeV)
    threshold_rows = build_threshold_crossing_rows(data, energy_thresholds_MeV)
    cohort_rows = build_cohort_summary_rows(data)
    _write_csv(output / "particle_history.csv", particle_rows)
    _write_csv(output / "threshold_crossings.csv", threshold_rows)
    _write_csv(output / "cohort_summary.csv", cohort_rows)
    plots = write_trajectory_plots(data, output / "plots")

    first_index = _first_true_index(data.present)
    first_steps, counts = np.unique(data.steps[first_index], return_counts=True)
    summary: dict[str, object] = {
        "schema": TRAJECTORY_ANALYSIS_SCHEMA,
        "source_trajectory_npz": str(Path(trajectory_npz).resolve(strict=False)),
        "n_particles": data.n_particles,
        "n_frames": data.n_frames,
        "energy_thresholds_MeV": [float(value) for value in energy_thresholds_MeV],
        "first_seen_histogram": {
            str(int(step)): int(count) for step, count in zip(first_steps, counts)
        },
        "first_seen_is_not_capture_time": True,
        "plots": [str(path.name) for path in plots],
    }
    (output / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
