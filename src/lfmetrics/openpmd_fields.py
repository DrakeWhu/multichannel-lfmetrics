from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class MeshComponentData:
    source_file: str
    step: int
    time_s: float | None
    record_name: str
    component_name: str
    values_si: np.ndarray
    axis_labels: tuple[str, ...]
    coordinates_m: tuple[np.ndarray, ...]
    grid_spacing_m: tuple[float, ...]
    grid_global_offset_m: tuple[float, ...]
    position: tuple[float, ...]
    unit_si: float
    grid_unit_si: float
    geometry: str
    data_order: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.values_si.shape)

    def coordinate(self, axis_label: str) -> np.ndarray:
        try:
            index = self.axis_labels.index(axis_label)
        except ValueError as exc:
            raise KeyError(
                f"Axis '{axis_label}' not found; available axes: {self.axis_labels}"
            ) from exc
        return self.coordinates_m[index]


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _scalar_attribute(
    attrs: h5py.AttributeManager,
    name: str,
    *,
    default: float | None = None,
) -> float:
    if name not in attrs:
        if default is None:
            raise KeyError(f"Missing required scalar attribute '{name}'")
        return float(default)
    values = np.asarray(attrs[name], dtype=float)
    if values.size != 1:
        raise ValueError(f"Attribute '{name}' must be scalar, got shape {values.shape}")
    result = float(values.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"Attribute '{name}' must be finite")
    return result


def _float_tuple_attribute(
    attrs: h5py.AttributeManager,
    name: str,
) -> tuple[float, ...]:
    if name not in attrs:
        raise KeyError(f"Missing required attribute '{name}'")
    values = np.asarray(attrs[name], dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Attribute '{name}' must contain only finite values")
    return tuple(float(value) for value in values)


def _text_attribute(attrs: h5py.AttributeManager, name: str) -> str:
    if name not in attrs:
        raise KeyError(f"Missing required attribute '{name}'")
    values = np.asarray(attrs[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Attribute '{name}' must contain one value")
    return _decode_text(values[0])


def _text_tuple_attribute(
    attrs: h5py.AttributeManager,
    name: str,
) -> tuple[str, ...]:
    if name not in attrs:
        raise KeyError(f"Missing required attribute '{name}'")
    values = np.asarray(attrs[name]).reshape(-1)
    return tuple(_decode_text(value) for value in values)


def _iteration_keys(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise KeyError(f"Missing openPMD 'data' group in {h5.filename}")
    keys = list(h5["data"].keys())
    if not keys:
        raise KeyError(f"openPMD 'data' group is empty in {h5.filename}")
    try:
        return sorted(keys, key=int)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric openPMD iteration key in {h5.filename}: {keys}"
        ) from exc


def _iteration_time_s(iteration: h5py.Group) -> float | None:
    if "time" not in iteration.attrs:
        return None
    time = _scalar_attribute(iteration.attrs, "time")
    time_unit_si = _scalar_attribute(
        iteration.attrs,
        "timeUnitSI",
        default=1.0,
    )
    return time * time_unit_si


def _component_node(
    record: h5py.Dataset | h5py.Group,
    component_name: str,
) -> h5py.Dataset:
    if isinstance(record, h5py.Dataset):
        if component_name != "SCALAR":
            raise KeyError(
                f"Scalar record {record.name} has no component '{component_name}'"
            )
        return record

    if component_name not in record:
        raise KeyError(
            f"Component '{component_name}' not found in record {record.name}; "
            f"available components: {sorted(record.keys())}"
        )
    component = record[component_name]
    if not isinstance(component, h5py.Dataset):
        raise TypeError(
            f"Field component {component.name} is not a dataset; "
            "constant mesh components are not supported"
        )
    return component


def read_mesh_component(
    path: str | Path,
    record_name: str,
    component_name: str = "SCALAR",
    *,
    iteration: str | int | None = None,
) -> MeshComponentData:
    """Read one Cartesian openPMD mesh component in SI units.

    Array and coordinate order follows the mesh ``axisLabels`` metadata exactly.
    WarpX 3D file-based HDF5 diagnostics normally use C order with
    ``axisLabels = ("z", "y", "x")``.
    """
    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {source}")

    with h5py.File(source, "r") as h5:
        iteration_key = (
            str(iteration)
            if iteration is not None
            else _iteration_keys(h5)[-1]
        )
        if iteration_key not in h5["data"]:
            raise KeyError(
                f"Iteration '{iteration_key}' not found in {source}; "
                f"available iterations: {_iteration_keys(h5)}"
            )

        iteration_group = h5["data"][iteration_key]
        fields = iteration_group.get("fields")
        if not isinstance(fields, h5py.Group):
            raise KeyError(
                f"No fields group found in {source} at iteration {iteration_key}"
            )
        if record_name not in fields:
            raise KeyError(
                f"Field record '{record_name}' not found in {source}; "
                f"available records: {sorted(fields.keys())}"
            )

        record = fields[record_name]
        if not isinstance(record, (h5py.Dataset, h5py.Group)):
            raise TypeError(f"Unsupported HDF5 node for record {record_name}")

        axis_labels = _text_tuple_attribute(record.attrs, "axisLabels")
        grid_spacing = _float_tuple_attribute(record.attrs, "gridSpacing")
        grid_global_offset = _float_tuple_attribute(
            record.attrs,
            "gridGlobalOffset",
        )
        grid_unit_si = _scalar_attribute(
            record.attrs,
            "gridUnitSI",
            default=1.0,
        )
        geometry = _text_attribute(record.attrs, "geometry")
        data_order = _text_attribute(record.attrs, "dataOrder")

        if geometry != "cartesian":
            raise ValueError(
                f"Unsupported mesh geometry '{geometry}' in {record.name}; "
                "only cartesian meshes are supported"
            )
        if data_order != "C":
            raise ValueError(
                f"Unsupported mesh dataOrder '{data_order}' in {record.name}; "
                "only C order is supported"
            )

        component = _component_node(record, component_name)
        position = _float_tuple_attribute(component.attrs, "position")
        unit_si = _scalar_attribute(
            component.attrs,
            "unitSI",
            default=1.0,
        )
        values_si = np.asarray(component, dtype=float) * unit_si

        ndim = values_si.ndim
        metadata_lengths = {
            "axisLabels": len(axis_labels),
            "gridSpacing": len(grid_spacing),
            "gridGlobalOffset": len(grid_global_offset),
            "position": len(position),
        }
        mismatched = {
            name: length
            for name, length in metadata_lengths.items()
            if length != ndim
        }
        if mismatched:
            raise ValueError(
                f"Mesh metadata dimensionality does not match data ndim={ndim}: "
                f"{mismatched}"
            )

        grid_spacing_m = tuple(value * grid_unit_si for value in grid_spacing)
        grid_global_offset_m = tuple(
            value * grid_unit_si for value in grid_global_offset
        )
        coordinates_m = tuple(
            offset + (np.arange(size, dtype=float) + relative_position) * spacing
            for size, spacing, offset, relative_position in zip(
                values_si.shape,
                grid_spacing_m,
                grid_global_offset_m,
                position,
            )
        )

        return MeshComponentData(
            source_file=str(source),
            step=int(iteration_key),
            time_s=_iteration_time_s(iteration_group),
            record_name=record_name,
            component_name=component_name,
            values_si=np.asarray(values_si, dtype=float),
            axis_labels=axis_labels,
            coordinates_m=coordinates_m,
            grid_spacing_m=grid_spacing_m,
            grid_global_offset_m=grid_global_offset_m,
            position=position,
            unit_si=unit_si,
            grid_unit_si=grid_unit_si,
            geometry=geometry,
            data_order=data_order,
        )
