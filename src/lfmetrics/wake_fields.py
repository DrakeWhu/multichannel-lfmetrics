from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import C_LIGHT
from .field_slices import MeshPlaneData, read_mesh_plane


@dataclass(frozen=True)
class TransverseWakePlaneData:
    source_file: str
    step: int
    time_s: float | None
    plane: str
    normal_axis: str
    requested_coordinate_m: float
    actual_coordinate_m: float
    method: str
    beta: float
    propagation_sign: int
    wx_v_m: np.ndarray
    wy_v_m: np.ndarray
    axis_labels: tuple[str, str]
    coordinates_m: tuple[np.ndarray, np.ndarray]
    source_indices: tuple[int, ...]
    source_coordinates_m: tuple[float, ...]
    source_weights: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.wx_v_m.shape)

    def coordinate(self, axis_label: str) -> np.ndarray:
        try:
            index = self.axis_labels.index(axis_label)
        except ValueError as exc:
            raise KeyError(
                f"Axis '{axis_label}' not found; available axes: {self.axis_labels}"
            ) from exc
        return self.coordinates_m[index]


def _validated_beta(beta: float) -> float:
    value = float(beta)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"beta must be finite and satisfy 0 <= beta <= 1, got {beta}")
    return value


def _validated_propagation_sign(propagation_sign: int) -> int:
    value = int(propagation_sign)
    if value not in (-1, 1) or float(propagation_sign) != float(value):
        raise ValueError(
            "propagation_sign must be +1 for +z or -1 for -z, "
            f"got {propagation_sign}"
        )
    return value


def _assert_close_tuple(
    name: str,
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    atol: float = 1.0e-18,
) -> None:
    if len(reference) != len(candidate) or not np.allclose(
        reference,
        candidate,
        rtol=1.0e-12,
        atol=atol,
    ):
        raise ValueError(
            f"Wake-field components are not colocated: {name} differs; "
            f"reference={reference}, candidate={candidate}"
        )


def _assert_plane_compatible(
    reference: MeshPlaneData,
    candidate: MeshPlaneData,
    *,
    candidate_label: str,
) -> None:
    exact_fields = (
        "source_file",
        "step",
        "plane",
        "normal_axis",
        "method",
        "axis_labels",
        "shape",
        "source_indices",
        "geometry",
        "data_order",
    )
    for name in exact_fields:
        reference_value = getattr(reference, name)
        candidate_value = getattr(candidate, name)
        if reference_value != candidate_value:
            raise ValueError(
                "Wake-field components are not colocated: "
                f"{candidate_label}.{name}={candidate_value!r} "
                f"!= reference {reference_value!r}"
            )

    if reference.time_s is None or candidate.time_s is None:
        if reference.time_s is not candidate.time_s:
            raise ValueError(
                "Wake-field components are not colocated: iteration times differ"
            )
    elif not np.isclose(reference.time_s, candidate.time_s, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "Wake-field components are not colocated: iteration times differ"
        )

    for name in ("requested_coordinate_m", "actual_coordinate_m"):
        if not np.isclose(
            float(getattr(reference, name)),
            float(getattr(candidate, name)),
            rtol=1.0e-12,
            atol=1.0e-18,
        ):
            raise ValueError(
                "Wake-field components are not colocated: "
                f"{candidate_label}.{name} differs"
            )

    _assert_close_tuple(
        f"{candidate_label}.source_coordinates_m",
        reference.source_coordinates_m,
        candidate.source_coordinates_m,
    )
    _assert_close_tuple(
        f"{candidate_label}.source_weights",
        reference.source_weights,
        candidate.source_weights,
        atol=1.0e-14,
    )

    for axis_label in reference.axis_labels:
        reference_coordinate = reference.coordinate(axis_label)
        candidate_coordinate = candidate.coordinate(axis_label)
        if reference_coordinate.shape != candidate_coordinate.shape or not np.allclose(
            reference_coordinate,
            candidate_coordinate,
            rtol=1.0e-12,
            atol=1.0e-18,
        ):
            raise ValueError(
                "Wake-field components are not colocated: "
                f"coordinate axis '{axis_label}' differs for {candidate_label}"
            )


def read_transverse_wake_plane(
    path: str | Path,
    *,
    plane: str,
    coordinate_m: float,
    beta: float = 1.0,
    propagation_sign: int = 1,
    method: str = "linear",
    iteration: str | int | None = None,
) -> TransverseWakePlaneData:
    """Read the positive-test-charge transverse wake on one physical plane.

    For a test charge moving along ``s zhat`` with speed ``beta*c``:

    ``W_x = E_x - s*beta*c*B_y`` and
    ``W_y = E_y + s*beta*c*B_x``.

    The returned wake has units of V/m. Electron force is ``-e W``. Field
    components must already be colocated on the same openPMD mesh; otherwise
    this function raises instead of interpolating between distinct grids.
    """
    beta_value = _validated_beta(beta)
    sign = _validated_propagation_sign(propagation_sign)

    ex = read_mesh_plane(
        path,
        "E",
        "x",
        plane=plane,
        coordinate_m=coordinate_m,
        method=method,
        iteration=iteration,
    )
    by = read_mesh_plane(
        path,
        "B",
        "y",
        plane=plane,
        coordinate_m=coordinate_m,
        method=method,
        iteration=iteration,
    )
    _assert_plane_compatible(ex, by, candidate_label="B/y")

    ey = read_mesh_plane(
        path,
        "E",
        "y",
        plane=plane,
        coordinate_m=coordinate_m,
        method=method,
        iteration=iteration,
    )
    _assert_plane_compatible(ex, ey, candidate_label="E/y")

    bx = read_mesh_plane(
        path,
        "B",
        "x",
        plane=plane,
        coordinate_m=coordinate_m,
        method=method,
        iteration=iteration,
    )
    _assert_plane_compatible(ex, bx, candidate_label="B/x")

    magnetic_scale = float(sign) * beta_value * C_LIGHT
    wx_v_m = np.asarray(ex.values_si, dtype=float).copy()
    wx_v_m -= magnetic_scale * np.asarray(by.values_si, dtype=float)
    wy_v_m = np.asarray(ey.values_si, dtype=float).copy()
    wy_v_m += magnetic_scale * np.asarray(bx.values_si, dtype=float)

    return TransverseWakePlaneData(
        source_file=ex.source_file,
        step=ex.step,
        time_s=ex.time_s,
        plane=ex.plane,
        normal_axis=ex.normal_axis,
        requested_coordinate_m=ex.requested_coordinate_m,
        actual_coordinate_m=ex.actual_coordinate_m,
        method=ex.method,
        beta=beta_value,
        propagation_sign=sign,
        wx_v_m=np.asarray(wx_v_m, dtype=float),
        wy_v_m=np.asarray(wy_v_m, dtype=float),
        axis_labels=ex.axis_labels,
        coordinates_m=ex.coordinates_m,
        source_indices=ex.source_indices,
        source_coordinates_m=ex.source_coordinates_m,
        source_weights=ex.source_weights,
    )
