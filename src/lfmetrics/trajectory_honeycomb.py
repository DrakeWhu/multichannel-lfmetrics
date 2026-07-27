from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import E_CHARGE_C
from .honeycomb_geometry import HoneycombGeometry, read_honeycomb_geometry
from .trajectory_analysis import TrajectoryData, load_trajectory_npz


@dataclass(frozen=True)
class TrajectoryHoneycombClassification:
    source_trajectory_npz: str | None
    source_geometry_json: str | None
    trajectory: TrajectoryData
    rod_center_x_m: np.ndarray
    rod_center_y_m: np.ndarray
    void_center_x_m: np.ndarray
    void_center_y_m: np.ndarray
    void_sublattice: np.ndarray
    rod_distance_m: np.ndarray
    rod_clearance_m: np.ndarray
    void_distance_m: np.ndarray
    inside_plasma_bounds: np.ndarray
    inside_physical_rod: np.ndarray
    inside_interstitial_void: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.trajectory.n_frames, self.trajectory.n_particles)

    def validate(self) -> None:
        self.trajectory.validate()
        expected = self.shape
        for name in (
            "rod_center_x_m",
            "rod_center_y_m",
            "void_center_x_m",
            "void_center_y_m",
            "void_sublattice",
            "rod_distance_m",
            "rod_clearance_m",
            "void_distance_m",
            "inside_plasma_bounds",
            "inside_physical_rod",
            "inside_interstitial_void",
        ):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")

        absent = ~self.trajectory.present
        for name in (
            "rod_center_x_m",
            "rod_center_y_m",
            "void_center_x_m",
            "void_center_y_m",
            "rod_distance_m",
            "rod_clearance_m",
            "void_distance_m",
        ):
            if not np.all(np.isnan(getattr(self, name)[absent])):
                raise ValueError(f"{name} must be NaN where particles are absent")
        if not np.all(self.void_sublattice[absent] == ""):
            raise ValueError("void_sublattice must be empty where particles are absent")
        for name in (
            "inside_plasma_bounds",
            "inside_physical_rod",
            "inside_interstitial_void",
        ):
            if np.any(getattr(self, name)[absent]):
                raise ValueError(f"{name} must be false where particles are absent")

        present = self.trajectory.present
        if not np.all(np.isin(self.void_sublattice[present], ("A", "B"))):
            raise ValueError("present particles must have void sublattice A or B")
        if np.any(self.inside_physical_rod & self.inside_interstitial_void):
            raise ValueError("rod and interstitial classifications must be disjoint")
        classified = self.inside_physical_rod | self.inside_interstitial_void
        if np.any(classified != (present & self.inside_plasma_bounds)):
            raise ValueError(
                "inside-bounds present particles must be classified as rod or interstitial"
            )


def _nearest_void_center_and_sublattice_lab(
    geometry: HoneycombGeometry,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest honeycomb-dual center and its A/B sublattice."""

    x_rot, y_rot = geometry.lab_to_lattice(x_m, y_m)
    center_x_rot, center_y_rot = geometry.nearest_void_center_lattice(
        x_rot,
        y_rot,
    )

    fractional_row = (
        center_y_rot / geometry.lattice_height_m
        - np.floor(center_y_rot / geometry.lattice_height_m)
    )
    is_a = np.isclose(
        fractional_row,
        1.0 / 3.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    is_b = np.isclose(
        fractional_row,
        2.0 / 3.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    if not np.all(is_a | is_b):
        raise ValueError(
            "Nearest void centers do not belong to honeycomb sublattice A or B"
        )

    center_x_m, center_y_m = geometry.lattice_to_lab(
        center_x_rot,
        center_y_rot,
    )
    sublattice = np.where(is_a, "A", "B")
    return center_x_m, center_y_m, sublattice


def classify_trajectory_honeycomb(
    trajectory: TrajectoryData,
    geometry: HoneycombGeometry,
    *,
    source_trajectory_npz: str | Path | None = None,
    source_geometry_json: str | Path | None = None,
) -> TrajectoryHoneycombClassification:
    """Classify every present trajectory point relative to the exact honeycomb."""

    trajectory.validate()
    geometry.validate()

    present = np.asarray(trajectory.present, dtype=bool)
    finite_xy = np.isfinite(trajectory.x_m) & np.isfinite(trajectory.y_m)
    if np.any(present & ~finite_xy):
        raise ValueError("Present trajectory points must have finite x_m and y_m")

    shape = (trajectory.n_frames, trajectory.n_particles)
    float_arrays = {
        name: np.full(shape, np.nan, dtype=float)
        for name in (
            "rod_center_x_m",
            "rod_center_y_m",
            "void_center_x_m",
            "void_center_y_m",
            "rod_distance_m",
            "rod_clearance_m",
            "void_distance_m",
        )
    }
    void_sublattice = np.full(shape, "", dtype="<U1")
    inside_plasma_bounds = np.zeros(shape, dtype=bool)
    inside_physical_rod = np.zeros(shape, dtype=bool)
    inside_interstitial_void = np.zeros(shape, dtype=bool)

    x = np.asarray(trajectory.x_m[present], dtype=float)
    y = np.asarray(trajectory.y_m[present], dtype=float)

    rod_x, rod_y = geometry.nearest_rod_center_lab(x, y)
    void_x, void_y, sublattice = _nearest_void_center_and_sublattice_lab(
        geometry,
        x,
        y,
    )

    rod_distance = np.hypot(x - rod_x, y - rod_y)
    void_distance = np.hypot(x - void_x, y - void_y)
    in_bounds = geometry.inside_plasma_bounds(x, y)
    in_rod = in_bounds & (rod_distance <= geometry.cnt_radius_m)
    in_void = in_bounds & ~in_rod

    float_arrays["rod_center_x_m"][present] = rod_x
    float_arrays["rod_center_y_m"][present] = rod_y
    float_arrays["void_center_x_m"][present] = void_x
    float_arrays["void_center_y_m"][present] = void_y
    float_arrays["rod_distance_m"][present] = rod_distance
    float_arrays["rod_clearance_m"][present] = (
        rod_distance - geometry.cnt_radius_m
    )
    float_arrays["void_distance_m"][present] = void_distance
    void_sublattice[present] = sublattice
    inside_plasma_bounds[present] = in_bounds
    inside_physical_rod[present] = in_rod
    inside_interstitial_void[present] = in_void

    result = TrajectoryHoneycombClassification(
        source_trajectory_npz=(
            None
            if source_trajectory_npz is None
            else str(Path(source_trajectory_npz).resolve(strict=False))
        ),
        source_geometry_json=(
            None
            if source_geometry_json is None
            else str(Path(source_geometry_json).resolve(strict=False))
        ),
        trajectory=trajectory,
        void_sublattice=void_sublattice,
        inside_plasma_bounds=inside_plasma_bounds,
        inside_physical_rod=inside_physical_rod,
        inside_interstitial_void=inside_interstitial_void,
        **float_arrays,
    )
    result.validate()
    return result


def load_and_classify_trajectory_honeycomb(
    trajectory_npz: str | Path,
    resolved_parameters_json: str | Path,
) -> TrajectoryHoneycombClassification:
    trajectory_path = Path(trajectory_npz)
    geometry_path = Path(resolved_parameters_json)
    trajectory = load_trajectory_npz(trajectory_path)
    geometry = read_honeycomb_geometry(geometry_path)
    return classify_trajectory_honeycomb(
        trajectory,
        geometry,
        source_trajectory_npz=trajectory_path,
        source_geometry_json=geometry_path,
    )


def _first_true_index(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("present mask must be two-dimensional")
    any_true = np.any(mask, axis=0)
    index = np.argmax(mask, axis=0).astype(np.int64)
    index[~any_true] = -1
    return index


def build_particle_honeycomb_rows(
    result: TrajectoryHoneycombClassification,
) -> list[dict[str, object]]:
    """Build per-particle first-observation and final geometry rows."""

    result.validate()
    data = result.trajectory
    first_index = _first_true_index(data.present)
    if np.any(first_index < 0):
        raise ValueError("every tracked particle must appear in at least one frame")

    rows: list[dict[str, object]] = []
    for particle_index, particle_id in enumerate(data.ids):
        first = int(first_index[particle_index])
        final = data.n_frames - 1

        def geometry_fields(prefix: str, frame: int) -> dict[str, object]:
            return {
                f"{prefix}_x_um": float(data.x_m[frame, particle_index] * 1.0e6),
                f"{prefix}_y_um": float(data.y_m[frame, particle_index] * 1.0e6),
                f"{prefix}_rod_center_x_um": float(
                    result.rod_center_x_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_rod_center_y_um": float(
                    result.rod_center_y_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_rod_distance_um": float(
                    result.rod_distance_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_rod_clearance_um": float(
                    result.rod_clearance_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_void_center_x_um": float(
                    result.void_center_x_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_void_center_y_um": float(
                    result.void_center_y_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_void_distance_um": float(
                    result.void_distance_m[frame, particle_index] * 1.0e6
                ),
                f"{prefix}_void_sublattice": str(
                    result.void_sublattice[frame, particle_index]
                ),
                f"{prefix}_inside_plasma_bounds": bool(
                    result.inside_plasma_bounds[frame, particle_index]
                ),
                f"{prefix}_inside_physical_rod": bool(
                    result.inside_physical_rod[frame, particle_index]
                ),
                f"{prefix}_inside_interstitial_void": bool(
                    result.inside_interstitial_void[frame, particle_index]
                ),
            }

        row: dict[str, object] = {
            "id": int(particle_id),
            "first_seen_step": int(data.steps[first]),
            "first_seen_time_s": float(data.times_s[first]),
            "first_seen_is_not_capture_time": True,
            **geometry_fields("first_seen", first),
            "final_step": int(data.steps[final]),
            **geometry_fields("final", final),
            "final_energy_MeV": float(data.energy_MeV[final, particle_index]),
            "final_weighting": float(data.weighting[final, particle_index]),
        }
        rows.append(row)
    return rows


def build_frame_honeycomb_rows(
    result: TrajectoryHoneycombClassification,
) -> list[dict[str, object]]:
    """Build charge-weighted frame summaries for rods and interstitial voids."""

    result.validate()
    data = result.trajectory
    rows: list[dict[str, object]] = []

    for frame_index, step in enumerate(data.steps):
        present = data.present[frame_index]
        in_bounds = result.inside_plasma_bounds[frame_index]
        in_rod = result.inside_physical_rod[frame_index]
        in_void = result.inside_interstitial_void[frame_index]
        outside_bounds = present & ~in_bounds
        weights = data.weighting[frame_index]

        if np.any(present & ~np.isfinite(weights)):
            raise ValueError(
                f"Present particle weighting is non-finite at step {int(step)}"
            )

        def charge_pC(mask: np.ndarray) -> float:
            return float(np.sum(weights[mask]) * E_CHARGE_C * 1.0e12)

        n_present = int(np.count_nonzero(present))
        rows.append(
            {
                "step": int(step),
                "time_s": float(data.times_s[frame_index]),
                "time_fs": float(data.times_s[frame_index] * 1.0e15),
                "n_present": n_present,
                "charge_present_pC": charge_pC(present),
                "n_inside_physical_rod": int(np.count_nonzero(in_rod)),
                "charge_inside_physical_rod_pC": charge_pC(in_rod),
                "fraction_inside_physical_rod": (
                    float(np.count_nonzero(in_rod) / n_present)
                    if n_present
                    else float("nan")
                ),
                "n_inside_interstitial_void": int(np.count_nonzero(in_void)),
                "charge_inside_interstitial_void_pC": charge_pC(in_void),
                "fraction_inside_interstitial_void": (
                    float(np.count_nonzero(in_void) / n_present)
                    if n_present
                    else float("nan")
                ),
                "n_outside_plasma_bounds": int(np.count_nonzero(outside_bounds)),
                "charge_outside_plasma_bounds_pC": charge_pC(outside_bounds),
            }
        )
    return rows
