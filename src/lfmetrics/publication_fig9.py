from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import publication_fig9_v3 as _v3


PUBLICATION_FIG9_SCHEMA = _v3.PUBLICATION_FIG9_SCHEMA
PublicationFig9SnapshotResult = _v3.PublicationFig9SnapshotResult
deterministic_background_indices = _v3.deterministic_background_indices

_MAGNETIC_RANGE_LOCK = threading.Lock()


def _validated_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def write_publication_fig9_snapshot(**kwargs) -> PublicationFig9SnapshotResult:
    """Write the v3 snapshot with fixed-scale multiresolution magnetic arrows.

    The exterior keeps the established coarse quiver. Inside the central box,
    the coarse arrows are removed and replaced by one arrow at every native
    transverse field cell. Both layers share the same fixed physical
    normalization so all publication frames remain directly comparable.
    """

    magnetic_color_limit_kT = _validated_positive_finite(
        "magnetic_color_limit_kT",
        kwargs.pop("magnetic_color_limit_kT", 125.0),
    )
    magnetic_visible_floor_kT = _validated_positive_finite(
        "magnetic_visible_floor_kT",
        kwargs.pop("magnetic_visible_floor_kT", 5.0),
    )
    quiver_central_half_width_m = _validated_positive_finite(
        "quiver_central_half_width_m",
        kwargs.pop("quiver_central_half_width_m", 2.5e-6),
    )
    if magnetic_visible_floor_kT >= magnetic_color_limit_kT:
        raise ValueError(
            "magnetic_visible_floor_kT must be smaller than magnetic_color_limit_kT"
        )

    magnetic_floor_t = magnetic_visible_floor_kT * 1.0e3
    magnetic_limit_t = magnetic_color_limit_kT * 1.0e3
    central_half_width_um = quiver_central_half_width_m * 1.0e6
    magnetic_gamma = float(kwargs.get("magnetic_length_gamma", 1.8))

    original_range = _v3._positive_percentile_range
    original_reader = _v3.read_mesh_plane
    original_quiver = Axes.quiver
    original_savefig = Figure.savefig

    captured_planes: dict[str, object] = {}
    central_stats = {
        "candidate_native_points": 0,
        "visible_native_arrows": 0,
    }

    def fixed_range(
        values: np.ndarray,
        lower_percentile: float,
        upper_percentile: float,
    ) -> tuple[float, float]:
        del values, lower_percentile, upper_percentile
        return float(magnetic_floor_t), float(magnetic_limit_t)

    def capturing_reader(*args, **reader_kwargs):
        plane = original_reader(*args, **reader_kwargs)
        if len(args) >= 3:
            record_name = str(args[1])
            component_name = str(args[2])
        else:
            record_name = str(reader_kwargs.get("record_name", ""))
            component_name = str(reader_kwargs.get("component_name", ""))
        plane_name = str(reader_kwargs.get("plane", ""))
        if record_name == "B" and component_name in {"x", "y"} and plane_name == "xy":
            captured_planes[component_name] = plane
        return plane

    def coarse_without_center(self, X, Y, U, V, *args, **quiver_kwargs):
        x_values = np.asarray(X, dtype=float)
        y_values = np.asarray(Y, dtype=float)
        central = (
            (np.abs(x_values) <= central_half_width_um)
            & (np.abs(y_values) <= central_half_width_um)
        )
        if np.any(central):
            U = np.ma.masked_where(central, np.ma.asarray(U))
            V = np.ma.masked_where(central, np.ma.asarray(V))
            if args:
                color_values = np.ma.masked_where(central, np.ma.asarray(args[0]))
                args = (color_values, *args[1:])
        return original_quiver(self, X, Y, U, V, *args, **quiver_kwargs)

    def savefig_with_dense_center(self, *save_args, **save_kwargs):
        if {"x", "y"}.issubset(captured_planes):
            bx_plane = captured_planes["x"]
            by_plane = captured_planes["y"]
            x_grid, y_grid, bx_matrix = _v3._plane_for_display(
                bx_plane,
                horizontal_axis="x",
                vertical_axis="y",
            )
            x_by, y_by, by_matrix = _v3._plane_for_display(
                by_plane,
                horizontal_axis="x",
                vertical_axis="y",
            )
            if not np.allclose(x_grid, x_by, rtol=1.0e-12, atol=1.0e-18):
                raise ValueError("B/x and B/y x coordinates do not match")
            if not np.allclose(y_grid, y_by, rtol=1.0e-12, atol=1.0e-18):
                raise ValueError("B/x and B/y y coordinates do not match")

            x_indices = np.flatnonzero(np.abs(x_grid) <= quiver_central_half_width_m)
            y_indices = np.flatnonzero(np.abs(y_grid) <= quiver_central_half_width_m)
            if x_indices.size and y_indices.size:
                X_m, Y_m = np.meshgrid(x_grid[x_indices], y_grid[y_indices])
                Bx = bx_matrix[np.ix_(y_indices, x_indices)]
                By = by_matrix[np.ix_(y_indices, x_indices)]
                Bmag = np.hypot(Bx, By)
                denominator = magnetic_limit_t - magnetic_floor_t
                normalized = np.clip(
                    (Bmag - magnetic_floor_t) / denominator,
                    0.0,
                    1.0,
                )
                amplitude = np.power(normalized, magnetic_gamma)
                visible = normalized > 0.0
                safe = np.where(Bmag > 0.0, Bmag, 1.0)
                U = np.ma.masked_where(~visible, (Bx / safe) * amplitude)
                V = np.ma.masked_where(~visible, (By / safe) * amplitude)
                C = np.ma.masked_where(~visible, Bmag / 1.0e3)

                main_axes = [
                    axis
                    for axis in self.axes
                    if axis.get_xlabel() == r"$x$ [$\mu$m]"
                    and axis.get_ylabel() == r"$y$ [$\mu$m]"
                ]
                if len(main_axes) != 1:
                    raise RuntimeError(
                        "Could not identify unique transverse publication axis"
                    )
                dense_quiver = original_quiver(
                    main_axes[0],
                    X_m * 1.0e6,
                    Y_m * 1.0e6,
                    U,
                    V,
                    C,
                    cmap="inferno",
                    angles="xy",
                    scale_units="width",
                    scale=7.5,
                    width=0.0042,
                    headwidth=4.4,
                    headlength=5.4,
                    headaxislength=4.8,
                    pivot="mid",
                    alpha=0.96,
                    zorder=5.2,
                )
                dense_quiver.set_clim(
                    magnetic_visible_floor_kT,
                    magnetic_color_limit_kT,
                )
                central_stats["candidate_native_points"] = int(Bmag.size)
                central_stats["visible_native_arrows"] = int(
                    np.count_nonzero(visible)
                )
        return original_savefig(self, *save_args, **save_kwargs)

    with _MAGNETIC_RANGE_LOCK:
        _v3._positive_percentile_range = fixed_range
        _v3.read_mesh_plane = capturing_reader
        Axes.quiver = coarse_without_center
        Figure.savefig = savefig_with_dense_center
        try:
            result = _v3.write_publication_fig9_snapshot(**kwargs)
        finally:
            _v3._positive_percentile_range = original_range
            _v3.read_mesh_plane = original_reader
            Axes.quiver = original_quiver
            Figure.savefig = original_savefig

    compatible = replace(
        result,
        xy_vector_overlay="B/x,B/y quiver",
        magnetic_color_min_kT=magnetic_visible_floor_kT,
        magnetic_color_max_kT=magnetic_color_limit_kT,
        quiver_representation=(
            "fixed 5-125 kT inferno normalization; coarse exterior arrows and "
            "every native field cell inside the central transverse box"
        ),
    )

    manifest_path = Path(result.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"] = asdict(compatible)
    manifest["xy_vector_overlay"] = "B/x,B/y quiver"
    manifest["magnetic_color_limit_kT"] = magnetic_color_limit_kT
    manifest["magnetic_visible_floor_kT"] = magnetic_visible_floor_kT
    manifest["magnetic_color_scale_mode"] = "fixed floor and ceiling across frames"
    manifest["magnetic_quiver_grid_mode"] = "coarse exterior plus native central grid"
    manifest["quiver_central_half_width_m"] = quiver_central_half_width_m
    manifest["central_quiver_candidate_native_points"] = central_stats[
        "candidate_native_points"
    ]
    manifest["central_quiver_visible_native_arrows"] = central_stats[
        "visible_native_arrows"
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return compatible


__all__ = [
    "PUBLICATION_FIG9_SCHEMA",
    "PublicationFig9SnapshotResult",
    "deterministic_background_indices",
    "write_publication_fig9_snapshot",
]
