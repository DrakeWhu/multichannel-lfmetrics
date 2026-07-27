from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import E_CHARGE_C
from .field_slices import MeshPlaneData, read_mesh_plane
from .trajectory_analysis import TrajectoryData, load_trajectory_npz


PUBLICATION_RHO_SMOKE_SCHEMA = "multichannel_publication_rho_smoke_v1"


@dataclass(frozen=True)
class RhoBunchSmokeFrameResult:
    step: int
    time_s: float
    output_png: str
    source_field_h5: str
    source_trajectory_npz: str
    plane: str
    normal_axis: str
    coordinate_m: float
    slab_half_width_m: float
    n_present: int
    charge_present_pC: float
    n_in_slab: int
    charge_in_slab_pC: float
    n_plotted: int
    charge_plotted_pC: float
    plotted_particle_ids: tuple[int, ...]
    rho_min_C_m3: float
    rho_max_C_m3: float
    color_percentile: float
    color_limit_C_m3: float
    z_front_grid_m: float
    xi_definition: str
    no_particle_coarsening: bool


def _validated_longitudinal_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in ("xz", "yz"):
        raise ValueError("Longitudinal rho smoke plane must be 'xz' or 'yz'")
    return normalized


def _validated_nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _cell_edges(centers_m: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers_m, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Cell-center coordinates must be a 1D array with at least 2 points")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Cell-center coordinates must be finite")
    spacing = np.diff(centers)
    if not np.all(spacing > 0.0):
        raise ValueError("Cell-center coordinates must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=1.0e-18):
        raise ValueError("Publication smoke requires a uniform Cartesian coordinate axis")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]
    return edges


def _robust_symmetric_limit(values: np.ndarray, percentile: float) -> float:
    requested = float(percentile)
    if not np.isfinite(requested) or requested <= 0.0 or requested > 100.0:
        raise ValueError("color_percentile must satisfy 0 < value <= 100")

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    nonzero = finite & (array != 0.0)
    selected = np.abs(array[nonzero])
    if selected.size == 0:
        return 1.0

    limit = float(np.percentile(selected, requested))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Could not derive a positive finite rho color limit")
    return limit


def _step_index(data: TrajectoryData, step: int) -> int:
    matches = np.flatnonzero(data.steps == int(step))
    if matches.size != 1:
        raise KeyError(
            f"Trajectory step {int(step)} must appear exactly once; found {matches.size}"
        )
    return int(matches[0])


def _charge_pC(weights: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(weights, dtype=float)[np.asarray(mask, dtype=bool)]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Selected particle weighting must be finite")
    return float(np.sum(selected) * E_CHARGE_C * 1.0e12)


def _scatter_style(n: int) -> tuple[float, float]:
    if n <= 200:
        return 15.0, 0.90
    if n <= 2_000:
        return 7.0, 0.70
    if n <= 20_000:
        return 3.0, 0.55
    return 1.5, 0.40


def _validate_field_trajectory_alignment(
    field: MeshPlaneData,
    trajectory: TrajectoryData,
    frame_index: int,
) -> None:
    if field.step != int(trajectory.steps[frame_index]):
        raise ValueError("Field and trajectory steps do not match")
    if field.time_s is None:
        raise ValueError("Field diagnostic time is required for publication smoke")
    trajectory_time = float(trajectory.times_s[frame_index])
    if not np.isfinite(trajectory_time) or not np.isclose(
        field.time_s,
        trajectory_time,
        rtol=1.0e-12,
        atol=1.0e-18,
    ):
        raise ValueError(
            "Field and particle diagnostic times do not match: "
            f"field={field.time_s}, particles={trajectory_time}"
        )


def _render_rho_bunch_smoke_frame(
    *,
    field_h5: Path,
    trajectory_npz: Path,
    trajectory: TrajectoryData,
    output_png: Path,
    step: int,
    plane: str,
    coordinate_m: float,
    slab_half_width_m: float,
    method: str,
    color_percentile: float,
    case_label: str,
    dpi: int,
) -> RhoBunchSmokeFrameResult:
    field = read_mesh_plane(
        field_h5,
        "rho",
        "SCALAR",
        plane=plane,
        coordinate_m=coordinate_m,
        method=method,
        iteration=step,
    )
    frame_index = _step_index(trajectory, step)
    _validate_field_trajectory_alignment(field, trajectory, frame_index)

    transverse_axis = "x" if plane == "xz" else "y"
    expected_axes = ("z", transverse_axis)
    if field.axis_labels != expected_axes:
        raise ValueError(
            f"Unexpected axes for {plane} smoke: {field.axis_labels}, expected {expected_axes}"
        )

    z_centers = field.coordinate("z")
    transverse_centers = field.coordinate(transverse_axis)
    z_edges = _cell_edges(z_centers)
    transverse_edges = _cell_edges(transverse_centers)
    z_front_grid = float(z_edges[-1])
    xi_centers = z_centers - z_front_grid
    xi_edges = z_edges - z_front_grid

    present = np.asarray(trajectory.present[frame_index], dtype=bool)
    x = np.asarray(trajectory.x_m[frame_index], dtype=float)
    y = np.asarray(trajectory.y_m[frame_index], dtype=float)
    z = np.asarray(trajectory.z_m[frame_index], dtype=float)
    weights = np.asarray(trajectory.weighting[frame_index], dtype=float)

    normal_values = y if field.normal_axis == "y" else x
    transverse_values = x if transverse_axis == "x" else y
    finite_particle = (
        np.isfinite(normal_values)
        & np.isfinite(transverse_values)
        & np.isfinite(z)
        & np.isfinite(weights)
    )
    tolerance = max(1.0e-18, 8.0 * np.finfo(float).eps * max(1.0, abs(coordinate_m)))
    in_slab = (
        present
        & finite_particle
        & (np.abs(normal_values - coordinate_m) <= slab_half_width_m + tolerance)
    )
    in_view = (
        in_slab
        & (z >= z_edges[0])
        & (z <= z_edges[-1])
        & (transverse_values >= transverse_edges[0])
        & (transverse_values <= transverse_edges[-1])
    )

    rho = np.asarray(field.values_si, dtype=float)
    if rho.shape != (z_centers.size, transverse_centers.size):
        raise ValueError(
            f"rho shape {rho.shape} does not match coordinate axes "
            f"{(z_centers.size, transverse_centers.size)}"
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho plane contains NaN or infinite values")
    color_limit = _robust_symmetric_limit(rho, color_percentile)

    point_size, point_alpha = _scatter_style(int(np.count_nonzero(in_view)))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    image = ax.pcolormesh(
        xi_edges * 1.0e6,
        transverse_edges * 1.0e6,
        rho.T,
        shading="flat",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\rho$ [C m$^{-3}$]")

    if np.any(in_view):
        ax.scatter(
            (z[in_view] - z_front_grid) * 1.0e6,
            transverse_values[in_view] * 1.0e6,
            s=point_size,
            facecolors="white",
            edgecolors="black",
            linewidths=0.25,
            alpha=point_alpha,
            label="tracked final-bunch IDs in slab",
            zorder=3,
        )
        ax.legend(loc="lower left", fontsize=7, framealpha=0.80)

    n_present = int(np.count_nonzero(present))
    n_in_slab = int(np.count_nonzero(in_slab))
    n_plotted = int(np.count_nonzero(in_view))
    charge_present = _charge_pC(weights, present)
    charge_in_slab = _charge_pC(weights, in_slab)
    charge_plotted = _charge_pC(weights, in_view)

    annotation = (
        f"step {step}; t = {float(field.time_s) * 1.0e15:.3f} fs\n"
        f"present: N={n_present}, Q={charge_present:.3f} pC\n"
        f"slab |{field.normal_axis}-{coordinate_m * 1.0e6:.3f}| "
        f"<= {slab_half_width_m * 1.0e6:.3f} um: "
        f"N={n_in_slab}, Q={charge_in_slab:.3f} pC\n"
        f"rho color limit: +/-{color_limit:.3e} C m^-3 "
        f"(p{float(color_percentile):g} of nonzero |rho|)"
    )
    ax.text(
        0.99,
        0.98,
        annotation,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=4,
    )

    ax.set_xlabel(r"$\xi_{\mathrm{grid}} = z-z_{\mathrm{front,grid}}$ [$\mu$m]")
    ax.set_ylabel(f"{transverse_axis} [$\mu$m]")
    ax.set_title(
        f"{case_label}: rho {plane} at {field.normal_axis}="
        f"{coordinate_m * 1.0e6:g} um"
    )
    ax.set_xlim(float(xi_edges[0] * 1.0e6), float(xi_edges[-1] * 1.0e6))
    ax.set_ylim(
        float(transverse_edges[0] * 1.0e6),
        float(transverse_edges[-1] * 1.0e6),
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=int(dpi))
    plt.close(fig)

    plotted_ids = tuple(int(value) for value in trajectory.ids[in_view])
    return RhoBunchSmokeFrameResult(
        step=int(step),
        time_s=float(field.time_s),
        output_png=str(output_png.resolve(strict=False)),
        source_field_h5=str(field_h5.resolve(strict=False)),
        source_trajectory_npz=str(trajectory_npz.resolve(strict=False)),
        plane=plane,
        normal_axis=field.normal_axis,
        coordinate_m=float(coordinate_m),
        slab_half_width_m=float(slab_half_width_m),
        n_present=n_present,
        charge_present_pC=charge_present,
        n_in_slab=n_in_slab,
        charge_in_slab_pC=charge_in_slab,
        n_plotted=n_plotted,
        charge_plotted_pC=charge_plotted,
        plotted_particle_ids=plotted_ids,
        rho_min_C_m3=float(np.min(rho)),
        rho_max_C_m3=float(np.max(rho)),
        color_percentile=float(color_percentile),
        color_limit_C_m3=float(color_limit),
        z_front_grid_m=z_front_grid,
        xi_definition="z_particle_or_cell_center - z_front_grid_from_field_edges",
        no_particle_coarsening=True,
    )


def write_rho_bunch_smoke_series(
    *,
    case_dir: str | Path,
    trajectory_npz: str | Path,
    output_dir: str | Path,
    steps: Sequence[int],
    case_label: str,
    plane: str = "xz",
    coordinate_m: float = 0.0,
    slab_half_width_m: float,
    method: str = "linear",
    field_diagnostics_dir: str = "fields3D",
    color_percentile: float = 99.5,
    dpi: int = 200,
) -> list[RhoBunchSmokeFrameResult]:
    """Write versioned rho/trajectory smoke frames without particle coarsening."""

    case = Path(case_dir).resolve(strict=False)
    trajectory_path = Path(trajectory_npz).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    normalized_plane = _validated_longitudinal_plane(plane)
    coordinate = float(coordinate_m)
    if not np.isfinite(coordinate):
        raise ValueError("coordinate_m must be finite")
    slab_half_width = _validated_nonnegative_finite(
        "slab_half_width_m",
        slab_half_width_m,
    )
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    _robust_symmetric_limit(np.asarray([1.0]), color_percentile)

    requested_steps = tuple(int(step) for step in steps)
    if not requested_steps:
        raise ValueError("steps must contain at least one diagnostic step")
    if len(set(requested_steps)) != len(requested_steps):
        raise ValueError("steps must not contain duplicates")
    if not case.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory NPZ does not exist: {trajectory_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output}")

    trajectory = load_trajectory_npz(trajectory_path)
    trajectory_steps = set(int(value) for value in trajectory.steps)
    missing_trajectory_steps = [
        step for step in requested_steps if step not in trajectory_steps
    ]
    if missing_trajectory_steps:
        raise KeyError(
            f"Requested steps missing from trajectory NPZ: {missing_trajectory_steps}"
        )

    field_files = {
        step: case / field_diagnostics_dir / f"openpmd_{step:06d}.h5"
        for step in requested_steps
    }
    missing_fields = [str(path) for path in field_files.values() if not path.is_file()]
    if missing_fields:
        raise FileNotFoundError(
            "Missing field diagnostics for publication smoke: " + ", ".join(missing_fields)
        )

    output.mkdir(parents=True, exist_ok=False)
    results: list[RhoBunchSmokeFrameResult] = []
    for step in requested_steps:
        output_png = output / f"rho_{normalized_plane}_step{step:06d}.png"
        results.append(
            _render_rho_bunch_smoke_frame(
                field_h5=field_files[step],
                trajectory_npz=trajectory_path,
                trajectory=trajectory,
                output_png=output_png,
                step=step,
                plane=normalized_plane,
                coordinate_m=coordinate,
                slab_half_width_m=slab_half_width,
                method=method,
                color_percentile=float(color_percentile),
                case_label=str(case_label),
                dpi=dpi,
            )
        )

    manifest = {
        "schema": PUBLICATION_RHO_SMOKE_SCHEMA,
        "case_dir": str(case),
        "case_label": str(case_label),
        "source_trajectory_npz": str(trajectory_path),
        "field_diagnostics_dir": str(field_diagnostics_dir),
        "plane": normalized_plane,
        "coordinate_m": coordinate,
        "slab_half_width_m": slab_half_width,
        "method": str(method),
        "color_percentile": float(color_percentile),
        "dpi": int(dpi),
        "particle_selection": "fixed final-bunch persistent IDs present in each frame",
        "particle_coarsening": False,
        "frames": [asdict(result) for result in results],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
