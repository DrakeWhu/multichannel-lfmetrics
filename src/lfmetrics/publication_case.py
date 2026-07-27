from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PUBLICATION_CASE_AUDIT_SCHEMA = "multichannel_publication_case_audit_v1"


@dataclass(frozen=True)
class ComponentAudit:
    name: str
    hdf5_path: str
    shape: tuple[int, ...] | None
    dtype: str | None
    unit_si: float | None
    position: tuple[float, ...] | None
    constant: bool


@dataclass(frozen=True)
class RecordAudit:
    name: str
    hdf5_path: str
    record_type: str
    components: dict[str, ComponentAudit]
    axis_labels: tuple[str, ...] | None
    grid_spacing: tuple[float, ...] | None
    grid_global_offset: tuple[float, ...] | None
    grid_unit_si: float | None
    geometry: str | None
    geometry_parameters: str | None
    data_order: str | None


@dataclass(frozen=True)
class IterationAudit:
    step: int
    iteration_key: str
    source_file: str
    series_directory: str
    time_s: float | None
    field_records: dict[str, RecordAudit]
    particle_records: dict[str, dict[str, RecordAudit]]

    @property
    def particle_species(self) -> tuple[str, ...]:
        return tuple(sorted(self.particle_records))

    @property
    def has_fields(self) -> bool:
        return bool(self.field_records)

    @property
    def has_particles(self) -> bool:
        return bool(self.particle_records)


@dataclass(frozen=True)
class SeriesAudit:
    series_kind: str
    directory: str
    file_pattern: str | None
    files: tuple[str, ...]
    steps: tuple[int, ...]
    iterations: tuple[IterationAudit, ...]


@dataclass(frozen=True)
class PublicationContextFiles:
    input_py_candidates: tuple[str, ...]
    resolved_parameters_candidates: tuple[str, ...]
    final_bunch_ids_candidates: tuple[str, ...]
    bunch_trajectories_candidates: tuple[str, ...]


@dataclass(frozen=True)
class PublicationCaseAudit:
    schema: str
    case_dir: str
    context_files: PublicationContextFiles
    series: tuple[SeriesAudit, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _attribute_scalar(attrs: h5py.AttributeManager, name: str) -> float | None:
    if name not in attrs:
        return None
    value = np.asarray(attrs[name])
    if value.size != 1:
        raise ValueError(f"Attribute '{name}' must be scalar, got shape {value.shape}")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"Attribute '{name}' must be finite")
    return result


def _attribute_text(attrs: h5py.AttributeManager, name: str) -> str | None:
    if name not in attrs:
        return None
    value = np.asarray(attrs[name])
    if value.size != 1:
        raise ValueError(f"Attribute '{name}' must contain one value")
    return _decode_text(value.reshape(-1)[0])


def _attribute_float_tuple(
    attrs: h5py.AttributeManager, name: str
) -> tuple[float, ...] | None:
    if name not in attrs:
        return None
    values = np.asarray(attrs[name], dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Attribute '{name}' must contain only finite values")
    return tuple(float(value) for value in values)


def _attribute_text_tuple(
    attrs: h5py.AttributeManager, name: str
) -> tuple[str, ...] | None:
    if name not in attrs:
        return None
    values = np.asarray(attrs[name]).reshape(-1)
    return tuple(_decode_text(value) for value in values)


def _component_audit(name: str, node: h5py.Dataset | h5py.Group) -> ComponentAudit:
    if isinstance(node, h5py.Dataset):
        shape: tuple[int, ...] | None = tuple(int(value) for value in node.shape)
        dtype: str | None = str(node.dtype)
        constant = False
    else:
        shape = None
        dtype = None
        constant = "value" in node.attrs
        if constant:
            value = np.asarray(node.attrs["value"])
            dtype = str(value.dtype)
            if "shape" in node.attrs:
                shape = tuple(
                    int(item) for item in np.asarray(node.attrs["shape"]).reshape(-1)
                )

    return ComponentAudit(
        name=name,
        hdf5_path=node.name,
        shape=shape,
        dtype=dtype,
        unit_si=_attribute_scalar(node.attrs, "unitSI"),
        position=_attribute_float_tuple(node.attrs, "position"),
        constant=constant,
    )


def _record_components(node: h5py.Dataset | h5py.Group) -> dict[str, ComponentAudit]:
    if isinstance(node, h5py.Dataset):
        return {"SCALAR": _component_audit("SCALAR", node)}

    components: dict[str, ComponentAudit] = {}
    for name in sorted(node.keys()):
        child = node[name]
        if isinstance(child, (h5py.Dataset, h5py.Group)):
            components[name] = _component_audit(name, child)

    if not components and "value" in node.attrs:
        components["SCALAR"] = _component_audit("SCALAR", node)
    return components


def _record_audit(
    name: str,
    node: h5py.Dataset | h5py.Group,
    *,
    record_type: str,
) -> RecordAudit:
    return RecordAudit(
        name=name,
        hdf5_path=node.name,
        record_type=record_type,
        components=_record_components(node),
        axis_labels=_attribute_text_tuple(node.attrs, "axisLabels"),
        grid_spacing=_attribute_float_tuple(node.attrs, "gridSpacing"),
        grid_global_offset=_attribute_float_tuple(node.attrs, "gridGlobalOffset"),
        grid_unit_si=_attribute_scalar(node.attrs, "gridUnitSI"),
        geometry=_attribute_text(node.attrs, "geometry"),
        geometry_parameters=_attribute_text(node.attrs, "geometryParameters"),
        data_order=_attribute_text(node.attrs, "dataOrder"),
    )


def _iteration_time_s(iteration: h5py.Group) -> float | None:
    time = _attribute_scalar(iteration.attrs, "time")
    if time is None:
        return None
    unit = _attribute_scalar(iteration.attrs, "timeUnitSI")
    return time * (1.0 if unit is None else unit)


def _iteration_keys(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise KeyError(f"Missing openPMD 'data' group in {h5.filename}")
    data = h5["data"]
    keys = list(data.keys())
    if not keys:
        raise KeyError(f"openPMD 'data' group is empty in {h5.filename}")
    try:
        return sorted(keys, key=int)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric openPMD iteration key in {h5.filename}: {keys}"
        ) from exc


def read_openpmd_file_audits(path: str | Path) -> tuple[IterationAudit, ...]:
    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {source}")

    audits: list[IterationAudit] = []
    with h5py.File(source, "r") as h5:
        for iteration_key in _iteration_keys(h5):
            iteration = h5["data"][iteration_key]

            field_records: dict[str, RecordAudit] = {}
            fields = iteration.get("fields")
            if isinstance(fields, h5py.Group):
                for name in sorted(fields.keys()):
                    node = fields[name]
                    if isinstance(node, (h5py.Dataset, h5py.Group)):
                        field_records[name] = _record_audit(
                            name, node, record_type="mesh"
                        )

            particle_records: dict[str, dict[str, RecordAudit]] = {}
            particles = iteration.get("particles")
            if isinstance(particles, h5py.Group):
                for species_name in sorted(particles.keys()):
                    species = particles[species_name]
                    if not isinstance(species, h5py.Group):
                        continue
                    records: dict[str, RecordAudit] = {}
                    for record_name in sorted(species.keys()):
                        node = species[record_name]
                        if isinstance(node, (h5py.Dataset, h5py.Group)):
                            records[record_name] = _record_audit(
                                record_name,
                                node,
                                record_type="particle_record",
                            )
                    particle_records[species_name] = records

            audits.append(
                IterationAudit(
                    step=int(iteration_key),
                    iteration_key=iteration_key,
                    source_file=str(source),
                    series_directory=str(source.parent),
                    time_s=_iteration_time_s(iteration),
                    field_records=field_records,
                    particle_records=particle_records,
                )
            )
    return tuple(audits)


def classify_series_kind(iterations: tuple[IterationAudit, ...]) -> str:
    has_fields = any(item.has_fields for item in iterations)
    has_particles = any(item.has_particles for item in iterations)
    if has_fields and has_particles:
        return "mixed"
    if has_fields:
        return "fields"
    if has_particles:
        return "particles"
    return "unknown"


def _infer_file_pattern(files: tuple[Path, ...]) -> str | None:
    if not files:
        return None
    matches = [re.match(r"^(.*?)(\d+)(\.h5)$", path.name) for path in files]
    if any(match is None for match in matches):
        return None
    prefixes = {match.group(1) for match in matches if match is not None}
    suffixes = {match.group(3) for match in matches if match is not None}
    widths = {len(match.group(2)) for match in matches if match is not None}
    if len(prefixes) != 1 or len(suffixes) != 1 or len(widths) != 1:
        return None
    return f"{next(iter(prefixes))}%0{next(iter(widths))}T{next(iter(suffixes))}"


def find_openpmd_h5_series(case_dir: str | Path) -> tuple[SeriesAudit, ...]:
    case = Path(case_dir).resolve(strict=False)
    if not case.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case}")

    h5_files = sorted(path for path in case.rglob("*.h5") if path.is_file())
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found under case directory: {case}")

    by_directory: dict[Path, list[Path]] = {}
    for path in h5_files:
        by_directory.setdefault(path.parent, []).append(path)

    series: list[SeriesAudit] = []
    for directory in sorted(by_directory, key=str):
        files = tuple(sorted(by_directory[directory]))
        iterations = tuple(
            audit for path in files for audit in read_openpmd_file_audits(path)
        )
        steps = tuple(item.step for item in iterations)
        if len(set(steps)) != len(steps):
            raise ValueError(
                f"Duplicate openPMD iteration steps in series directory {directory}: {steps}"
            )
        ordered = tuple(sorted(iterations, key=lambda item: item.step))
        series.append(
            SeriesAudit(
                series_kind=classify_series_kind(ordered),
                directory=str(directory),
                file_pattern=_infer_file_pattern(files),
                files=tuple(str(path) for path in files),
                steps=tuple(item.step for item in ordered),
                iterations=ordered,
            )
        )
    return tuple(series)


def _candidate_paths(case: Path, pattern: str) -> tuple[str, ...]:
    return tuple(
        str(path.resolve(strict=False)) for path in sorted(case.glob(pattern))
    )


def find_publication_context_files(case_dir: str | Path) -> PublicationContextFiles:
    case = Path(case_dir).resolve(strict=False)
    if not case.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case}")

    return PublicationContextFiles(
        input_py_candidates=_candidate_paths(case, "**/input.py"),
        resolved_parameters_candidates=_candidate_paths(
            case, "**/resolved_parameters.json"
        ),
        final_bunch_ids_candidates=_candidate_paths(
            case,
            "post/publication_tracking_*/selection/final_bunch_ids.npy",
        ),
        bunch_trajectories_candidates=_candidate_paths(
            case,
            "post/publication_tracking_*/backtracking/bunch_trajectories.npz",
        ),
    )


def audit_publication_case(case_dir: str | Path) -> PublicationCaseAudit:
    case = Path(case_dir).resolve(strict=False)
    return PublicationCaseAudit(
        schema=PUBLICATION_CASE_AUDIT_SCHEMA,
        case_dir=str(case),
        context_files=find_publication_context_files(case),
        series=find_openpmd_h5_series(case),
    )


def write_publication_case_audit(
    audit: PublicationCaseAudit,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return output
