from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .beam_metrics import kinetic_energy_MeV, weighted_mean, weighted_percentile, weighted_std
from .constants import E_CHARGE_C
from .openpmd_h5 import (
    choose_species,
    last_iteration_key,
    list_openpmd_h5_files,
    parse_openpmd_step,
    read_particles_from_h5,
)


TRACKING_SCHEMA_VERSION = "multichannel_final_bunch_tracking_v1"


@dataclass(frozen=True)
class FinalBunchSelection:
    ids: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    px_si: np.ndarray
    py_si: np.ndarray
    pz_si: np.ndarray
    weighting: np.ndarray
    energy_MeV: np.ndarray
    source_h5: Path
    source_step: int
    species: str
    energy_threshold_MeV: float
    forward_only: bool


@dataclass(frozen=True)
class BunchTrackingResult:
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


def _read_openpmd_time_s(iteration_group: h5py.Group) -> float:
    raw_time = np.asarray(iteration_group.attrs.get("time", np.nan), dtype=float)
    raw_unit = np.asarray(iteration_group.attrs.get("timeUnitSI", 1.0), dtype=float)
    if raw_time.size != 1 or raw_unit.size != 1:
        raise ValueError("openPMD time and timeUnitSI must be scalar")
    return float(raw_time.reshape(-1)[0]) * float(raw_unit.reshape(-1)[0])


def read_particle_ids_from_h5(
    path: str | Path,
    *,
    species: str | None = None,
    iteration: str | int | None = None,
) -> tuple[int, str, np.ndarray, float]:
    """Read persistent WarpX particle IDs and iteration time from one openPMD file."""

    h5_path = Path(path)
    with h5py.File(h5_path, "r") as h5:
        iteration_key = str(iteration) if iteration is not None else last_iteration_key(h5)
        iteration_group = h5["data"][iteration_key]
        particles_root = iteration_group.get("particles")
        if particles_root is None:
            raise KeyError(f"No particles group found in {h5_path} at iteration {iteration_key}")

        selected_species = choose_species(
            available_species=sorted(particles_root.keys()),
            requested_species=species,
        )
        species_group = particles_root[selected_species]
        if "id" not in species_group:
            raise KeyError(
                f"Persistent particle ID record missing for species '{selected_species}' in {h5_path}"
            )

        ids = np.asarray(species_group["id"], dtype=np.uint64)
        time_s = _read_openpmd_time_s(iteration_group)

    if ids.ndim != 1:
        raise ValueError(f"Particle ID record must be one-dimensional, got {ids.shape}")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"Particle IDs are not unique in {h5_path}")

    return int(iteration_key), selected_species, ids, time_s


def select_final_bunch(
    h5_path: str | Path,
    *,
    species: str | None = "electrons",
    energy_threshold_MeV: float = 20.0,
    forward_only: bool = True,
) -> FinalBunchSelection:
    """Select a final-frame bunch and preserve its persistent particle IDs."""

    threshold = float(energy_threshold_MeV)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("energy_threshold_MeV must be finite and non-negative")

    source = Path(h5_path).resolve(strict=False)
    particles = read_particles_from_h5(source, species=species)
    step, selected_species, ids, _ = read_particle_ids_from_h5(
        source,
        species=particles.species,
        iteration=particles.step,
    )
    if ids.shape != (particles.size,):
        raise ValueError(
            f"Particle ID length {ids.shape} does not match particle data length {particles.size}"
        )
    if particles.weighting is None:
        raise ValueError("Particle weighting is required for final-bunch selection")

    energy = kinetic_energy_MeV(particles.px_si, particles.py_si, particles.pz_si)
    mask = np.isfinite(energy) & np.isfinite(particles.pz_si) & (energy >= threshold)
    if forward_only:
        mask &= particles.pz_si > 0.0

    selected_ids = ids[mask]
    order = np.argsort(selected_ids)

    def selected(values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[mask][order]

    selection = FinalBunchSelection(
        ids=selected_ids[order],
        x_m=selected(particles.x_m),
        y_m=selected(particles.y_m),
        z_m=selected(particles.z_m),
        px_si=selected(particles.px_si),
        py_si=selected(particles.py_si),
        pz_si=selected(particles.pz_si),
        weighting=selected(particles.weighting),
        energy_MeV=selected(energy),
        source_h5=source,
        source_step=step,
        species=selected_species,
        energy_threshold_MeV=threshold,
        forward_only=bool(forward_only),
    )

    if np.unique(selection.ids).size != selection.ids.size:
        raise ValueError("Selected final-bunch IDs are not unique")
    return selection


def write_final_bunch_selection(
    selection: FinalBunchSelection,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write selection IDs, complete final state and a JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    np.save(output / "final_bunch_ids.npy", selection.ids, allow_pickle=False)
    np.savez_compressed(
        output / "final_bunch_state.npz",
        id=selection.ids,
        x_m=selection.x_m,
        y_m=selection.y_m,
        z_m=selection.z_m,
        px_si=selection.px_si,
        py_si=selection.py_si,
        pz_si=selection.pz_si,
        weighting=selection.weighting,
        energy_MeV=selection.energy_MeV,
    )

    charge_pC = float(np.sum(selection.weighting) * E_CHARGE_C * 1.0e12)
    summary: dict[str, object] = {
        "schema": TRACKING_SCHEMA_VERSION,
        "source_h5": str(selection.source_h5),
        "source_step": int(selection.source_step),
        "species": selection.species,
        "selection": (
            f"energy_MeV >= {selection.energy_threshold_MeV:g}"
            + (" and pz_si > 0" if selection.forward_only else "")
        ),
        "energy_threshold_MeV": float(selection.energy_threshold_MeV),
        "forward_only": bool(selection.forward_only),
        "n_macroparticles": int(selection.ids.size),
        "weight_sum": float(np.sum(selection.weighting)),
        "charge_pC": charge_pC,
        "energy_min_MeV": (
            None if selection.energy_MeV.size == 0 else float(np.min(selection.energy_MeV))
        ),
        "energy_max_MeV": (
            None if selection.energy_MeV.size == 0 else float(np.max(selection.energy_MeV))
        ),
        "ids_unique": True,
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _sorted_diagnostic_files(
    case_dir: str | Path,
    diagnostics_dir: str,
) -> list[Path]:
    files = list_openpmd_h5_files(case_dir, diagnostics_dir=diagnostics_dir)
    parsed = [(parse_openpmd_step(path), path) for path in files]
    if any(step is None for step, _ in parsed):
        raise ValueError("All tracking diagnostic filenames must contain a numeric step")
    return [path for _, path in sorted(parsed, key=lambda item: int(item[0]))]


def backtrack_particle_ids(
    case_dir: str | Path,
    target_ids: np.ndarray,
    *,
    species: str | None = "electrons",
    diagnostics_dir: str = "3D",
) -> BunchTrackingResult:
    """Read all diagnostic frames and backtrack a fixed set of persistent IDs."""

    ids = np.asarray(target_ids, dtype=np.uint64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("target_ids must be a non-empty one-dimensional array")
    if np.unique(ids).size != ids.size:
        raise ValueError("target_ids must be unique")
    ids = np.sort(ids)

    files = _sorted_diagnostic_files(case_dir, diagnostics_dir)
    n_frames = len(files)
    n_particles = ids.size
    steps = np.empty(n_frames, dtype=np.int64)
    times_s = np.empty(n_frames, dtype=float)
    present = np.zeros((n_frames, n_particles), dtype=bool)
    arrays = {
        name: np.full((n_frames, n_particles), np.nan, dtype=float)
        for name in (
            "x_m",
            "y_m",
            "z_m",
            "px_si",
            "py_si",
            "pz_si",
            "weighting",
            "energy_MeV",
        )
    }

    for frame_index, h5_path in enumerate(files):
        particles = read_particles_from_h5(h5_path, species=species)
        step, selected_species, frame_ids, time_s = read_particle_ids_from_h5(
            h5_path,
            species=particles.species,
            iteration=particles.step,
        )
        if selected_species != particles.species:
            raise ValueError("Particle data and ID species do not match")
        if frame_ids.shape != (particles.size,):
            raise ValueError(f"Particle ID length mismatch in {h5_path}")
        if particles.weighting is None:
            raise ValueError(f"Particle weighting is required for tracking: {h5_path}")

        common_ids, target_index, frame_index_in_particles = np.intersect1d(
            ids,
            frame_ids,
            assume_unique=True,
            return_indices=True,
        )
        del common_ids
        present[frame_index, target_index] = True
        steps[frame_index] = step
        times_s[frame_index] = time_s

        px = particles.px_si[frame_index_in_particles]
        py = particles.py_si[frame_index_in_particles]
        pz = particles.pz_si[frame_index_in_particles]
        values = {
            "x_m": particles.x_m[frame_index_in_particles],
            "y_m": particles.y_m[frame_index_in_particles],
            "z_m": particles.z_m[frame_index_in_particles],
            "px_si": px,
            "py_si": py,
            "pz_si": pz,
            "weighting": particles.weighting[frame_index_in_particles],
            "energy_MeV": kinetic_energy_MeV(px, py, pz),
        }
        for name, selected_values in values.items():
            arrays[name][frame_index, target_index] = selected_values

    if not np.all(np.diff(steps) > 0):
        raise ValueError("Diagnostic steps must be strictly increasing")
    finite_times = np.isfinite(times_s)
    if np.count_nonzero(finite_times) > 1 and not np.all(np.diff(times_s[finite_times]) > 0):
        raise ValueError("Finite diagnostic times must be strictly increasing")
    if not np.all(present[-1]):
        raise ValueError("Not all target IDs are present in the final diagnostic frame")

    return BunchTrackingResult(
        ids=ids,
        steps=steps,
        times_s=times_s,
        present=present,
        **arrays,
    )


def _frame_summary_rows(result: BunchTrackingResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_index, step in enumerate(result.steps):
        mask = result.present[frame_index]
        weights = result.weighting[frame_index, mask]
        energy = result.energy_MeV[frame_index, mask]
        z_m = result.z_m[frame_index, mask]
        charge_pC = float(np.sum(weights) * E_CHARGE_C * 1.0e12)
        rows.append(
            {
                "step": int(step),
                "time_s": float(result.times_s[frame_index]),
                "time_fs": float(result.times_s[frame_index] * 1.0e15),
                "n_present": int(np.count_nonzero(mask)),
                "fraction_present": float(np.mean(mask)),
                "charge_present_pC": charge_pC,
                "energy_mean_MeV": weighted_mean(energy, weights),
                "energy_std_MeV": weighted_std(energy, weights),
                "energy_p10_MeV": weighted_percentile(energy, 10.0, weights),
                "energy_p50_MeV": weighted_percentile(energy, 50.0, weights),
                "energy_p90_MeV": weighted_percentile(energy, 90.0, weights),
                "energy_p95_MeV": weighted_percentile(energy, 95.0, weights),
                "energy_max_MeV": None if energy.size == 0 else float(np.max(energy)),
                "z_mean_um": (
                    None if z_m.size == 0 else weighted_mean(z_m, weights) * 1.0e6
                ),
                "z_std_um": None if z_m.size == 0 else weighted_std(z_m, weights) * 1.0e6,
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_tracking_outputs(
    result: BunchTrackingResult,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write trajectory arrays, frame summaries and first-observation metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output / "bunch_trajectories.npz",
        id=result.ids,
        step=result.steps,
        time_s=result.times_s,
        present=result.present,
        x_m=result.x_m,
        y_m=result.y_m,
        z_m=result.z_m,
        px_si=result.px_si,
        py_si=result.py_si,
        pz_si=result.pz_si,
        weighting=result.weighting,
        energy_MeV=result.energy_MeV,
    )

    frame_rows = _frame_summary_rows(result)
    _write_csv(output / "frame_summary.csv", frame_rows)

    first_frame_index = np.argmax(result.present, axis=0)
    origins = []
    for particle_index, particle_id in enumerate(result.ids):
        first = int(first_frame_index[particle_index])
        origins.append(
            {
                "id": int(particle_id),
                "first_seen_step": int(result.steps[first]),
                "first_seen_time_s": float(result.times_s[first]),
                "first_seen_time_fs": float(result.times_s[first] * 1.0e15),
                "first_x_um": float(result.x_m[first, particle_index] * 1.0e6),
                "first_y_um": float(result.y_m[first, particle_index] * 1.0e6),
                "first_z_um": float(result.z_m[first, particle_index] * 1.0e6),
                "first_energy_MeV": float(result.energy_MeV[first, particle_index]),
                "final_x_um": float(result.x_m[-1, particle_index] * 1.0e6),
                "final_y_um": float(result.y_m[-1, particle_index] * 1.0e6),
                "final_z_um": float(result.z_m[-1, particle_index] * 1.0e6),
                "final_energy_MeV": float(result.energy_MeV[-1, particle_index]),
                "final_weighting": float(result.weighting[-1, particle_index]),
            }
        )
    _write_csv(output / "particle_origin.csv", origins)

    unique_first, counts = np.unique(result.steps[first_frame_index], return_counts=True)
    summary: dict[str, object] = {
        "schema": TRACKING_SCHEMA_VERSION,
        "n_particles": int(result.ids.size),
        "n_frames": int(result.steps.size),
        "steps": [int(step) for step in result.steps],
        "first_seen_histogram": {
            str(int(step)): int(count) for step, count in zip(unique_first, counts)
        },
        "all_final_ids_present_at_final_step": bool(np.all(result.present[-1])),
        "first_seen_is_not_capture_time": True,
    }
    (output / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
