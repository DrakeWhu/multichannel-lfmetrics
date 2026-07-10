from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .particles import ParticleData


DEFAULT_SPECIES_PRIORITY = "electrons"


def list_openpmd_h5_files(
    case_dir: str | Path, diagnostics_dir: str = "3D"
) -> list[Path]:
    """Return sorted WarpX/openPMD HDF5 files for one case directory."""
    case_path = Path(case_dir)
    data_dir = case_path / diagnostics_dir

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Diagnostics directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in: {data_dir}")

    return files


def parse_openpmd_step(path: str | Path) -> int | None:
    """Parse step from names such as openpmd_005000.h5."""
    match = re.search(r"(\d+)(?=\.h5$)", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def last_openpmd_h5_file(case_dir: str | Path, diagnostics_dir: str = "3D") -> Path:
    """Return the last HDF5 file by parsed iteration, falling back to lexical order."""
    files = list_openpmd_h5_files(case_dir, diagnostics_dir=diagnostics_dir)

    def sort_key(path: Path) -> tuple[int, str]:
        parsed = parse_openpmd_step(path)
        return (-1 if parsed is None else parsed, path.name)

    return sorted(files, key=sort_key)[-1]


def _iteration_keys(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise KeyError("Missing openPMD 'data' group")
    keys = list(h5["data"].keys())
    if not keys:
        raise KeyError("openPMD 'data' group is empty")
    return sorted(keys, key=lambda item: int(item))


def last_iteration_key(h5: h5py.File) -> str:
    return _iteration_keys(h5)[-1]


def list_species_in_h5(
    path: str | Path, iteration: str | int | None = None
) -> list[str]:
    """List particle species available in one openPMD HDF5 file."""
    with h5py.File(path, "r") as h5:
        it_key = str(iteration) if iteration is not None else last_iteration_key(h5)
        particles = h5["data"][it_key].get("particles", None)
        if particles is None:
            return []
        return sorted(particles.keys())


def choose_species(
    available_species: Sequence[str],
    requested_species: str | None = None,
    priority: Sequence[str] = DEFAULT_SPECIES_PRIORITY,
) -> str:
    """Choose species explicitly or by a stable beam-oriented priority."""
    available = list(available_species)

    if requested_species:
        if requested_species not in available:
            raise KeyError(
                f"Requested species '{requested_species}' not found. "
                f"Available species: {available}"
            )
        return requested_species

    for candidate in priority:
        if candidate in available:
            return candidate

    for species in available:
        if "electron" in species.lower():
            return species

    if available:
        return available[0]

    raise KeyError("No particle species available")


def _read_component(
    group: h5py.Group, record_name: str, component_name: str
) -> np.ndarray:
    record = group[record_name]
    return np.asarray(record[component_name])


def _read_optional_position_offset(
    group: h5py.Group, component_name: str, length: int
) -> np.ndarray | float:
    """Read one optional openPMD ``positionOffset`` component.

    WarpX/openPMD HDF5 output can encode a constant component as an HDF5
    group whose scalar value is stored in its ``value`` attribute. Returning
    that value as a scalar avoids materializing a full-length zero array for
    every particle coordinate.
    """
    if "positionOffset" not in group:
        return 0.0

    offset = group["positionOffset"]
    if component_name not in offset:
        return 0.0

    component = offset[component_name]

    if isinstance(component, h5py.Group):
        if "value" not in component.attrs:
            raise ValueError(
                "constant positionOffset component "
                f"'{component_name}' is missing its 'value' attribute"
            )

        value = np.asarray(component.attrs["value"], dtype=float)
        if value.size != 1:
            raise ValueError(
                "constant positionOffset component "
                f"'{component_name}' has non-scalar value shape {value.shape}"
            )
        return float(value.reshape(-1)[0])

    values = np.asarray(component, dtype=float)
    if values.ndim != 1 or values.shape[0] != length:
        raise ValueError(
            f"positionOffset '{component_name}' shape {values.shape} "
            f"does not match particle length {length}"
        )
    return values


def _read_weighting(group: h5py.Group, length: int) -> np.ndarray | None:
    """Read optional particle weighting.

    WarpX/openPMD files commonly expose weighting either directly as a dataset
    or as a record with a SCALAR component. Both are accepted.
    """
    if "weighting" not in group:
        return None

    weighting = group["weighting"]

    if isinstance(weighting, h5py.Dataset):
        values = np.asarray(weighting, dtype=float)
    elif isinstance(weighting, h5py.Group) and "SCALAR" in weighting:
        values = np.asarray(weighting["SCALAR"], dtype=float)
    else:
        return None

    if values.shape[0] != length:
        raise ValueError(
            f"weighting length {values.shape[0]} does not match particle length {length}"
        )

    return values


def read_particles_from_h5(
    path: str | Path,
    species: str | None = None,
    iteration: str | int | None = None,
) -> ParticleData:
    """Read one particle species from one WarpX/openPMD HDF5 file."""
    h5_path = Path(path)

    with h5py.File(h5_path, "r") as h5:
        it_key = str(iteration) if iteration is not None else last_iteration_key(h5)
        step = int(it_key)

        iteration_group = h5["data"][it_key]
        particles_root = iteration_group.get("particles", None)
        if particles_root is None:
            raise KeyError(
                f"No particles group found in {h5_path} at iteration {it_key}"
            )

        selected_species = choose_species(
            available_species=sorted(particles_root.keys()),
            requested_species=species,
        )

        species_group = particles_root[selected_species]

        x = _read_component(species_group, "position", "x")
        y = _read_component(species_group, "position", "y")
        z = _read_component(species_group, "position", "z")

        length = int(x.shape[0])
        if y.shape[0] != length or z.shape[0] != length:
            raise ValueError("position components have inconsistent lengths")

        x = x + _read_optional_position_offset(species_group, "x", length)
        y = y + _read_optional_position_offset(species_group, "y", length)
        z = z + _read_optional_position_offset(species_group, "z", length)

        px = _read_component(species_group, "momentum", "x")
        py = _read_component(species_group, "momentum", "y")
        pz = _read_component(species_group, "momentum", "z")

        weighting = _read_weighting(species_group, length)

    particles = ParticleData(
        species=selected_species,
        step=step,
        x_m=np.asarray(x, dtype=float),
        y_m=np.asarray(y, dtype=float),
        z_m=np.asarray(z, dtype=float),
        px_si=np.asarray(px, dtype=float),
        py_si=np.asarray(py, dtype=float),
        pz_si=np.asarray(pz, dtype=float),
        weighting=weighting,
    )
    particles.validate()
    return particles


def read_particles_from_case(
    case_dir: str | Path,
    species: str | None = None,
    diagnostics_dir: str = "3D",
) -> ParticleData:
    """Read selected species from the last HDF5 output of a case directory."""
    h5_path = last_openpmd_h5_file(case_dir, diagnostics_dir=diagnostics_dir)
    return read_particles_from_h5(h5_path, species=species)
