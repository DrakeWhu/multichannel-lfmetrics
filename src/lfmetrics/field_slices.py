from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .openpmd_fields import (
    _component_node,
    _float_tuple_attribute,
    _iteration_keys,
    _iteration_time_s,
    _scalar_attribute,
    _text_attribute,
    _text_tuple_attribute,
)


_VALID_PLANES = {"xy", "xz", "yz"}
_VALID_METHODS = {"nearest", "linear"}


@dataclass(frozen=True)
class MeshPlaneData:
    source_file: str
    step: int
    time_s: float | None
    record_name: str
    component_name: str
    plane: str
    normal_axis: str
    requested_coordinate_m: float
    actual_coordinate_m: float
    method: str
    source_indices: tuple[int, ...]
    source_coordinates_m: tuple[float, ...]
    source_weights: tuple[float, ...]
    values_si: np.ndarray
    axis_labels: tuple[str, str]
    coordinates_m: tuple[np.ndarray, np.ndarray]
    unit_si: float
    geometry: str
    data_order: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.values_si.shape)

    def coordinate(self, axis_label: str) -> np.ndarray:
        try:
            index = self.axis_labels.index(axis_label)
        except ValueError as exc:
            raise KeyError(
                f"Axis '{axis_label}' not found; available axes: {self.axis_labels}"
            ) from exc
        return self.coordinates_m[index]


def _normalize_plane(plane: str) -> str:
    normalized = plane.lower().strip()
    if normalized not in _VALID_PLANES:
        raise ValueError(
            f"Unsupported plane '{plane}'; choose one of {sorted(_VALID_PLANES)}"
        )
    return normalized


def _normalize_method(method: str) -> str:
    normalized = method.lower().strip()
    if normalized not in _VALID_METHODS:
        raise ValueError(
            f"Unsupported slice method '{method}'; "
            f"choose one of {sorted(_VALID_METHODS)}"
        )
    return normalized


def _coordinate_selection(
    coordinates_m: np.ndarray,
    requested_coordinate_m: float,
    method: str,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], float]:
    coordinates = np.asarray(coordinates_m, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Normal-axis coordinates must be a non-empty 1D array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Normal-axis coordinates must be finite")
    if coordinates.size > 1 and not np.all(np.diff(coordinates) > 0.0):
        raise ValueError("Normal-axis coordinates must be strictly increasing")

    requested = float(requested_coordinate_m)
    if not np.isfinite(requested):
        raise ValueError("Requested slice coordinate must be finite")

    lower_bound = float(coordinates[0])
    upper_bound = float(coordinates[-1])
    tolerance = (
        np.finfo(float).eps
        * max(abs(lower_bound), abs(upper_bound), 1.0)
        * 8.0
    )
    if requested < lower_bound - tolerance or requested > upper_bound + tolerance:
        raise ValueError(
            f"Requested coordinate {requested} m lies outside cell-center range "
            f"[{lower_bound}, {upper_bound}] m; extrapolation is not allowed"
        )

    if method == "nearest":
        index = int(np.argmin(np.abs(coordinates - requested)))
        actual = float(coordinates[index])
        return (index,), (actual,), (1.0,), actual

    nearest = int(np.argmin(np.abs(coordinates - requested)))
    if np.isclose(
        coordinates[nearest],
        requested,
        rtol=0.0,
        atol=tolerance,
    ):
        actual = float(coordinates[nearest])
        return (nearest,), (actual,), (1.0,), actual

    upper = int(np.searchsorted(coordinates, requested, side="right"))
    lower = upper - 1
    if lower < 0 or upper >= coordinates.size:
        raise ValueError(
            f"Could not bracket requested coordinate {requested} m without extrapolation"
        )

    lower_coordinate = float(coordinates[lower])
    upper_coordinate = float(coordinates[upper])
    upper_weight = (requested - lower_coordinate) / (
        upper_coordinate - lower_coordinate
    )
    lower_weight = 1.0 - upper_weight

    return (
        (lower, upper),
        (lower_coordinate, upper_coordinate),
        (float(lower_weight), float(upper_weight)),
        requested,
    )


def read_mesh_plane(
    path: str | Path,
    record_name: str,
    component_name: str = "SCALAR",
    *,
    plane: str,
    coordinate_m: float,
    method: str = "nearest",
    iteration: str | int | None = None,
) -> MeshPlaneData:
    """Read one physical 2D plane directly from a Cartesian openPMD mesh.

    Only the selected HDF5 plane, or the two planes needed for explicit linear
    interpolation along the normal axis, are materialized. No extrapolation or
    smoothing is performed.
    """
    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {source}")

    normalized_plane = _normalize_plane(plane)
    normalized_method = _normalize_method(method)

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

        if component.ndim != 3:
            raise ValueError(
                f"Mesh plane extraction requires a 3D component, "
                f"got shape {component.shape}"
            )

        metadata_lengths = {
            "axisLabels": len(axis_labels),
            "gridSpacing": len(grid_spacing),
            "gridGlobalOffset": len(grid_global_offset),
            "position": len(position),
        }
        mismatched = {
            name: length
            for name, length in metadata_lengths.items()
            if length != component.ndim
        }
        if mismatched:
            raise ValueError(
                f"Mesh metadata dimensionality does not match data ndim="
                f"{component.ndim}: {mismatched}"
            )

        if set(axis_labels) != {"x", "y", "z"}:
            raise ValueError(
                f"3D Cartesian plane extraction requires x, y, z axes; "
                f"got {axis_labels}"
            )

        plane_axes = set(normalized_plane)
        normal_axes = set(axis_labels) - plane_axes
        if len(normal_axes) != 1:
            raise ValueError(
                f"Plane '{normalized_plane}' is incompatible with axes {axis_labels}"
            )
        normal_axis = next(iter(normal_axes))
        normal_dimension = axis_labels.index(normal_axis)

        grid_spacing_m = tuple(
            float(value) * grid_unit_si for value in grid_spacing
        )
        grid_global_offset_m = tuple(
            float(value) * grid_unit_si for value in grid_global_offset
        )
        if any(value <= 0.0 for value in grid_spacing_m):
            raise ValueError(
                f"Grid spacing must be positive, got {grid_spacing_m}"
            )

        coordinates_m = tuple(
            offset + (np.arange(size, dtype=float) + relative_position) * spacing
            for size, spacing, offset, relative_position in zip(
                component.shape,
                grid_spacing_m,
                grid_global_offset_m,
                position,
            )
        )

        (
            source_indices,
            source_coordinates,
            source_weights,
            actual_coordinate,
        ) = _coordinate_selection(
            coordinates_m[normal_dimension],
            float(coordinate_m),
            normalized_method,
        )

        selector = [slice(None)] * component.ndim
        selector[normal_dimension] = source_indices[0]
        values_si = np.asarray(component[tuple(selector)], dtype=float)
        values_si *= unit_si * source_weights[0]

        for index, weight in zip(source_indices[1:], source_weights[1:]):
            selector[normal_dimension] = index
            values_si += (
                np.asarray(component[tuple(selector)], dtype=float)
                * unit_si
                * weight
            )

        retained_dimensions = tuple(
            index
            for index in range(component.ndim)
            if index != normal_dimension
        )
        retained_axis_labels = tuple(
            axis_labels[index] for index in retained_dimensions
        )
        retained_coordinates = tuple(
            coordinates_m[index] for index in retained_dimensions
        )

        if len(retained_axis_labels) != 2:
            raise AssertionError("Internal error: plane extraction is not 2D")
        expected_shape = tuple(
            component.shape[index] for index in retained_dimensions
        )
        if values_si.shape != expected_shape:
            raise ValueError(
                f"Extracted plane shape {values_si.shape} does not match "
                f"expected shape {expected_shape}"
            )

        return MeshPlaneData(
            source_file=str(source),
            step=int(iteration_key),
            time_s=_iteration_time_s(iteration_group),
            record_name=record_name,
            component_name=component_name,
            plane=normalized_plane,
            normal_axis=normal_axis,
            requested_coordinate_m=float(coordinate_m),
            actual_coordinate_m=float(actual_coordinate),
            method=normalized_method,
            source_indices=source_indices,
            source_coordinates_m=source_coordinates,
            source_weights=source_weights,
            values_si=np.asarray(values_si, dtype=float),
            axis_labels=retained_axis_labels,
            coordinates_m=retained_coordinates,
            unit_si=unit_si,
            geometry=geometry,
            data_order=data_order,
        )
