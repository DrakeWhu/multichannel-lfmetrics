from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .beam_metrics import kinetic_energy_MeV
from .openpmd_h5 import read_particles_from_h5
from .particles import ParticleData
from .tracking import read_particle_ids_from_h5


PlaneName = Literal["xy", "xz", "yz"]


@dataclass(frozen=True)
class PublicationParticleFrame:
    """One WarpX particle frame with persistent IDs and kinetic energy."""

    source_file: Path
    step: int
    time_s: float
    species: str
    ids: np.ndarray
    particles: ParticleData
    energy_MeV: np.ndarray

    @property
    def size(self) -> int:
        return int(self.ids.size)

    def validate(self) -> None:
        self.particles.validate()
        if self.step != self.particles.step:
            raise ValueError(
                f"Frame step {self.step} does not match particle step "
                f"{self.particles.step}"
            )
        if self.species != self.particles.species:
            raise ValueError(
                f"Frame species '{self.species}' does not match particle species "
                f"'{self.particles.species}'"
            )
        if not np.isfinite(self.time_s):
            raise ValueError("Particle frame time_s must be finite")

        ids = np.asarray(self.ids)
        if ids.ndim != 1:
            raise ValueError(f"Particle IDs must be one-dimensional, got {ids.shape}")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Particle IDs must use an integer dtype")
        if ids.shape != (self.particles.size,):
            raise ValueError(
                f"Particle ID shape {ids.shape} does not match "
                f"particle size {self.particles.size}"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError("Particle IDs must be unique")

        energy = np.asarray(self.energy_MeV, dtype=float)
        if energy.shape != ids.shape:
            raise ValueError(
                f"energy_MeV shape {energy.shape} does not match IDs {ids.shape}"
            )

        for name in ("x_m", "y_m", "z_m", "px_si", "py_si", "pz_si"):
            values = np.asarray(getattr(self.particles, name), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinite values")
        if not np.all(np.isfinite(energy)):
            raise ValueError("energy_MeV contains NaN or infinite values")

        if self.particles.weighting is not None:
            weights = np.asarray(self.particles.weighting, dtype=float)
            if not np.all(np.isfinite(weights)):
                raise ValueError("weighting contains NaN or infinite values")
            if np.any(weights < 0.0):
                raise ValueError("weighting contains negative values")


@dataclass(frozen=True)
class PublicationParticlePlaneSelection:
    """Background and tracked-bunch particles selected in one physical slab."""

    frame: PublicationParticleFrame
    plane: PlaneName
    horizontal_axis: str
    vertical_axis: str
    normal_axis: str
    coordinate_m: float
    slab_half_width_m: float
    in_slab_mask: np.ndarray
    tracked_in_frame_mask: np.ndarray
    background_mask: np.ndarray
    tracked_mask: np.ndarray
    missing_tracked_ids: np.ndarray

    def validate(self) -> None:
        self.frame.validate()
        expected = (self.frame.size,)
        for name in (
            "in_slab_mask",
            "tracked_in_frame_mask",
            "background_mask",
            "tracked_mask",
        ):
            mask = np.asarray(getattr(self, name))
            if mask.shape != expected or mask.dtype != np.bool_:
                raise ValueError(
                    f"{name} must be a boolean mask with shape {expected}"
                )

        if np.any(self.background_mask & self.tracked_mask):
            raise ValueError("background and tracked masks must be disjoint")
        if np.any(
            (self.background_mask | self.tracked_mask) != self.in_slab_mask
        ):
            raise ValueError("background and tracked masks must partition the slab")
        if np.any(self.tracked_mask & ~self.tracked_in_frame_mask):
            raise ValueError("tracked_mask contains particles outside the tracked IDs")

        missing = np.asarray(self.missing_tracked_ids)
        if missing.ndim != 1 or not np.issubdtype(missing.dtype, np.integer):
            raise ValueError("missing_tracked_ids must be a 1D integer array")

    @property
    def n_background(self) -> int:
        return int(np.count_nonzero(self.background_mask))

    @property
    def n_tracked(self) -> int:
        return int(np.count_nonzero(self.tracked_mask))

    @property
    def horizontal_m(self) -> np.ndarray:
        return np.asarray(
            getattr(self.frame.particles, f"{self.horizontal_axis}_m")
        )

    @property
    def vertical_m(self) -> np.ndarray:
        return np.asarray(
            getattr(self.frame.particles, f"{self.vertical_axis}_m")
        )

    @property
    def normal_m(self) -> np.ndarray:
        return np.asarray(getattr(self.frame.particles, f"{self.normal_axis}_m"))


_PLANE_AXES: dict[str, tuple[str, str, str]] = {
    "xy": ("x", "y", "z"),
    "xz": ("z", "x", "y"),
    "yz": ("z", "y", "x"),
}


def read_publication_particle_frame(
    path: str | Path,
    *,
    species: str | None = "electrons",
    iteration: str | int | None = None,
) -> PublicationParticleFrame:
    """Read one particle frame by reusing the validated WarpX HDF5 reader."""

    source = Path(path).resolve(strict=False)
    particles = read_particles_from_h5(
        source,
        species=species,
        iteration=iteration,
    )
    step, selected_species, ids, time_s = read_particle_ids_from_h5(
        source,
        species=particles.species,
        iteration=particles.step,
    )
    energy = kinetic_energy_MeV(
        particles.px_si,
        particles.py_si,
        particles.pz_si,
    )
    frame = PublicationParticleFrame(
        source_file=source,
        step=step,
        time_s=float(time_s),
        species=selected_species,
        ids=np.asarray(ids, dtype=np.uint64),
        particles=particles,
        energy_MeV=np.asarray(energy, dtype=float),
    )
    frame.validate()
    return frame


def read_tracked_particle_ids(path: str | Path) -> np.ndarray:
    """Load the fixed final-bunch persistent IDs from a NumPy file."""

    source = Path(path)
    values = np.load(source, allow_pickle=False)
    ids = np.asarray(values, dtype=np.uint64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("Tracked particle IDs must be a non-empty 1D array")
    if np.unique(ids).size != ids.size:
        raise ValueError("Tracked particle IDs must be unique")
    return ids


def select_publication_particle_plane(
    frame: PublicationParticleFrame,
    tracked_ids: np.ndarray,
    *,
    plane: PlaneName,
    coordinate_m: float,
    slab_half_width_m: float,
) -> PublicationParticlePlaneSelection:
    """Split one physical slab into untracked background and tracked bunch."""

    frame.validate()

    normalized_plane = str(plane).strip().lower()
    if normalized_plane not in _PLANE_AXES:
        raise ValueError("plane must be one of: xy, xz, yz")

    coordinate = float(coordinate_m)
    half_width = float(slab_half_width_m)
    if not np.isfinite(coordinate):
        raise ValueError("coordinate_m must be finite")
    if not np.isfinite(half_width) or half_width < 0.0:
        raise ValueError("slab_half_width_m must be finite and non-negative")

    tracked = np.asarray(tracked_ids, dtype=np.uint64)
    if tracked.ndim != 1 or tracked.size == 0:
        raise ValueError("tracked_ids must be a non-empty 1D array")
    if np.unique(tracked).size != tracked.size:
        raise ValueError("tracked_ids must be unique")

    horizontal_axis, vertical_axis, normal_axis = _PLANE_AXES[normalized_plane]
    normal_values = np.asarray(
        getattr(frame.particles, f"{normal_axis}_m"),
        dtype=float,
    )
    tolerance = max(
        1.0e-18,
        8.0 * np.finfo(float).eps * max(1.0, abs(coordinate)),
    )
    in_slab = np.abs(normal_values - coordinate) <= half_width + tolerance
    tracked_in_frame = np.isin(frame.ids, tracked, assume_unique=True)
    tracked_mask = in_slab & tracked_in_frame
    background_mask = in_slab & ~tracked_in_frame
    missing = tracked[~np.isin(tracked, frame.ids, assume_unique=True)]

    result = PublicationParticlePlaneSelection(
        frame=frame,
        plane=normalized_plane,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
        normal_axis=normal_axis,
        coordinate_m=coordinate,
        slab_half_width_m=half_width,
        in_slab_mask=np.asarray(in_slab, dtype=bool),
        tracked_in_frame_mask=np.asarray(tracked_in_frame, dtype=bool),
        background_mask=np.asarray(background_mask, dtype=bool),
        tracked_mask=np.asarray(tracked_mask, dtype=bool),
        missing_tracked_ids=np.asarray(missing, dtype=np.uint64),
    )
    result.validate()
    return result
