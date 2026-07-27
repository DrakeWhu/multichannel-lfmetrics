from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .beam_metrics import weighted_percentile
from .constants import E_CHARGE_C
from .field_slices import MeshPlaneData, read_mesh_plane
from .publication_particles import (
    PublicationParticleFrame,
    read_publication_particle_frame,
    read_tracked_particle_ids,
    select_publication_particle_plane,
)
from .wake_fields import _assert_plane_compatible


PUBLICATION_FIG9_SCHEMA = "multichannel_publication_fig9_snapshot_v1"


@dataclass(frozen=True)
class PublicationFig9SnapshotResult:
    step: int
    time_s: float
    output_png: str
    manifest_json: str
    particle_h5: str
    field_h5: str
    tracked_ids_npy: str
    n_electrons_total: int
    n_tracked_requested: int
    n_tracked_present: int
    n_tracked_missing: int
    tracked_charge_present_pC: float
    center_x_m: float
    center_y_m: float
    center_z_m: float
    z_front_grid_m: float
    center_xi_grid_m: float
    xi_limits_m: tuple[float, float]
    x_limits_m: tuple[float, float]
    y_limits_m: tuple[float, float]
    background_slab_half_width_m: float
    max_background_points_per_panel: int
    n_background_xz_available: int
    n_background_xz_plotted: int
    n_background_yz_available: int
    n_background_yz_plotted: int
    n_background_xy_available: int
    n_background_xy_plotted: int
    n_tracked_xz_plotted: int
    n_tracked_yz_plotted: int
    n_tracked_xy_plotted: int
    ez_color_limit_TV_m: float
    bperp_color_limit_kT: float
    energy_color_min_MeV: float
    energy_color_max_MeV: float
    field_slice_method: str
    background_particle_mode: str
    tracked_particle_mode: str
    background_sampling: str
    quiver_representation: str


def _validated_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_percentile(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result > 100.0:
        raise ValueError(f"{name} must satisfy 0 < value <= 100")
    return result


def _cell_edges(centers_m: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers_m, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Grid centers must be a 1D array with at least two entries")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Grid centers must be finite")
    spacing = np.diff(centers)
    if not np.all(spacing > 0.0):
        raise ValueError("Grid centers must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=1.0e-18):
        raise ValueError("Publication snapshot requires uniform Cartesian axes")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing[0]
    edges[-1] = centers[-1] + 0.5 * spacing[-1]
    return edges


def _plane_for_display(
    plane: MeshPlaneData,
    *,
    horizontal_axis: str,
    vertical_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = plane.axis_labels
    if set(labels) != {horizontal_axis, vertical_axis}:
        raise ValueError(
            f"Plane axes {labels} do not match requested display axes "
            f"({horizontal_axis}, {vertical_axis})"
        )

    horizontal = np.asarray(plane.coordinate(horizontal_axis), dtype=float)
    vertical = np.asarray(plane.coordinate(vertical_axis), dtype=float)
    values = np.asarray(plane.values_si, dtype=float)

    if labels == (vertical_axis, horizontal_axis):
        matrix = values
    elif labels == (horizontal_axis, vertical_axis):
        matrix = values.T
    else:
        raise AssertionError("Unexpected 2D axis ordering")

    expected = (vertical.size, horizontal.size)
    if matrix.shape != expected:
        raise ValueError(
            f"Display matrix shape {matrix.shape} does not match coordinates {expected}"
        )
    return horizontal, vertical, matrix


def _finite_nonzero_percentile(values: np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    selected = np.abs(array[np.isfinite(array) & (array != 0.0)])
    if selected.size == 0:
        return 1.0
    result = float(np.percentile(selected, percentile))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("Could not determine a positive finite field color limit")
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    result = weighted_percentile(values, 50.0, weights)
    if result is None or not np.isfinite(result):
        raise ValueError("Could not determine a finite charge-weighted median")
    return float(result)


def _tracked_frame_mask(
    frame: PublicationParticleFrame,
    tracked_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    present = np.isin(frame.ids, tracked_ids, assume_unique=True)
    missing = tracked_ids[~np.isin(tracked_ids, frame.ids, assume_unique=True)]
    return np.asarray(present, dtype=bool), np.asarray(missing, dtype=np.uint64)


def _charge_pC(weights: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(weights, dtype=float)[np.asarray(mask, dtype=bool)]
    if not np.all(np.isfinite(selected)):
        raise ValueError("Selected particle weights must be finite")
    return float(np.sum(selected) * E_CHARGE_C * 1.0e12)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = source + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        mixed = mixed ^ (mixed >> np.uint64(31))
    return mixed


def deterministic_background_indices(
    mask: np.ndarray,
    ids: np.ndarray,
    *,
    max_points: int,
) -> np.ndarray:
    """Return an ID-hash sample without altering the physical selection mask."""

    if not isinstance(max_points, int) or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    selection = np.asarray(mask, dtype=bool)
    particle_ids = np.asarray(ids, dtype=np.uint64)
    if selection.shape != particle_ids.shape:
        raise ValueError("mask and ids must have the same shape")

    indices = np.flatnonzero(selection)
    if indices.size <= max_points:
        return indices

    hashes = _splitmix64(particle_ids[indices])
    chosen_local = np.argpartition(hashes, max_points - 1)[:max_points]
    chosen = indices[chosen_local]
    order = np.argsort(hashes[chosen_local], kind="stable")
    return np.asarray(chosen[order], dtype=np.int64)


def _view_mask(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    horizontal_limits: tuple[float, float],
    vertical_limits: tuple[float, float],
) -> np.ndarray:
    h = np.asarray(horizontal, dtype=float)
    v = np.asarray(vertical, dtype=float)
    return (
        np.isfinite(h)
        & np.isfinite(v)
        & (h >= horizontal_limits[0])
        & (h <= horizontal_limits[1])
        & (v >= vertical_limits[0])
        & (v <= vertical_limits[1])
    )


def _cropped_matrix_values(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    matrix: np.ndarray,
    horizontal_limits: tuple[float, float],
    vertical_limits: tuple[float, float],
) -> np.ndarray:
    hmask = (horizontal >= horizontal_limits[0]) & (horizontal <= horizontal_limits[1])
    vmask = (vertical >= vertical_limits[0]) & (vertical <= vertical_limits[1])
    if not np.any(hmask) or not np.any(vmask):
        raise ValueError("Requested publication view does not intersect the field grid")
    return np.asarray(matrix[np.ix_(vmask, hmask)], dtype=float)


def _background_scatter_style(n_points: int) -> tuple[float, float]:
    if n_points <= 5_000:
        return 1.8, 0.34
    if n_points <= 20_000:
        return 0.9, 0.24
    return 0.45, 0.18


def _tracked_scatter_style(n_points: int) -> tuple[float, float]:
    if n_points <= 500:
        return 12.0, 0.95
    if n_points <= 2_000:
        return 7.0, 0.88
    return 4.2, 0.82


def _top_colorbar(fig, ax, mappable, label: str):
    cax = ax.inset_axes([0.13, 1.025, 0.74, 0.045])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label(label, fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, pad=1)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    return colorbar


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=8,
    )


def _validate_time_alignment(frame: PublicationParticleFrame, *planes: MeshPlaneData) -> None:
    for plane in planes:
        if plane.step != frame.step:
            raise ValueError("Particle and field diagnostic steps do not match")
        if plane.time_s is None or not np.isclose(
            plane.time_s,
            frame.time_s,
            rtol=1.0e-12,
            atol=1.0e-18,
        ):
            raise ValueError("Particle and field diagnostic times do not match")


def write_publication_fig9_snapshot(
    *,
    particle_h5: str | Path,
    field_h5: str | Path,
    tracked_ids_npy: str | Path,
    output_dir: str | Path,
    step: int,
    case_label: str,
    background_slab_half_width_m: float = 0.75e-6,
    transverse_half_width_m: float = 6.0e-6,
    xi_padding_back_m: float = 4.0e-6,
    xi_padding_front_m: float = 6.0e-6,
    max_background_points_per_panel: int = 40_000,
    field_percentile: float = 99.5,
    magnetic_percentile: float = 99.5,
    energy_percentile: float = 99.0,
    field_slice_method: str = "linear",
    quiver_max_arrows_per_axis: int = 23,
    dpi: int = 220,
) -> PublicationFig9SnapshotResult:
    """Write one three-panel field/particle snapshot inspired by paper Fig. 9.

    Background electrons are selected in a physical slab and may be thinned only
    for rendering. All present tracked final-bunch IDs are projected into each
    panel without thinning and are colored by kinetic energy.
    """

    particle_path = Path(particle_h5).resolve(strict=False)
    field_path = Path(field_h5).resolve(strict=False)
    tracked_path = Path(tracked_ids_npy).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)

    for source in (particle_path, field_path, tracked_path):
        if not source.is_file():
            raise FileNotFoundError(f"Required publication source does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite publication output: {output}")

    slab_half_width = _validated_positive_finite(
        "background_slab_half_width_m",
        background_slab_half_width_m,
    )
    transverse_half_width = _validated_positive_finite(
        "transverse_half_width_m",
        transverse_half_width_m,
    )
    xi_padding_back = _validated_positive_finite(
        "xi_padding_back_m",
        xi_padding_back_m,
    )
    xi_padding_front = _validated_positive_finite(
        "xi_padding_front_m",
        xi_padding_front_m,
    )
    field_pct = _validated_percentile("field_percentile", field_percentile)
    magnetic_pct = _validated_percentile(
        "magnetic_percentile",
        magnetic_percentile,
    )
    energy_pct = _validated_percentile("energy_percentile", energy_percentile)
    if not isinstance(max_background_points_per_panel, int) or max_background_points_per_panel <= 0:
        raise ValueError("max_background_points_per_panel must be a positive integer")
    if not isinstance(quiver_max_arrows_per_axis, int) or quiver_max_arrows_per_axis < 2:
        raise ValueError("quiver_max_arrows_per_axis must be an integer >= 2")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    frame = read_publication_particle_frame(
        particle_path,
        species="electrons",
        iteration=int(step),
    )
    tracked_ids = read_tracked_particle_ids(tracked_path)
    tracked_present, missing_tracked = _tracked_frame_mask(frame, tracked_ids)
    if not np.any(tracked_present):
        raise ValueError("None of the tracked final-bunch IDs are present in this frame")
    if frame.particles.weighting is None:
        raise ValueError("Particle weighting is required for publication snapshots")

    weights = np.asarray(frame.particles.weighting, dtype=float)
    x_m = np.asarray(frame.particles.x_m, dtype=float)
    y_m = np.asarray(frame.particles.y_m, dtype=float)
    z_m = np.asarray(frame.particles.z_m, dtype=float)
    energy_MeV = np.asarray(frame.energy_MeV, dtype=float)

    tracked_weights = weights[tracked_present]
    center_x = _weighted_median(x_m[tracked_present], tracked_weights)
    center_y = _weighted_median(y_m[tracked_present], tracked_weights)
    center_z = _weighted_median(z_m[tracked_present], tracked_weights)

    ez_xz = read_mesh_plane(
        field_path,
        "E",
        "z",
        plane="xz",
        coordinate_m=center_y,
        method=field_slice_method,
        iteration=int(step),
    )
    ez_yz = read_mesh_plane(
        field_path,
        "E",
        "z",
        plane="yz",
        coordinate_m=center_x,
        method=field_slice_method,
        iteration=int(step),
    )
    bx_xy = read_mesh_plane(
        field_path,
        "B",
        "x",
        plane="xy",
        coordinate_m=center_z,
        method=field_slice_method,
        iteration=int(step),
    )
    by_xy = read_mesh_plane(
        field_path,
        "B",
        "y",
        plane="xy",
        coordinate_m=center_z,
        method=field_slice_method,
        iteration=int(step),
    )
    _assert_plane_compatible(bx_xy, by_xy, candidate_label="B/y")
    _validate_time_alignment(frame, ez_xz, ez_yz, bx_xy, by_xy)

    z_xz, x_grid, ez_xz_matrix = _plane_for_display(
        ez_xz,
        horizontal_axis="z",
        vertical_axis="x",
    )
    z_yz, y_grid, ez_yz_matrix = _plane_for_display(
        ez_yz,
        horizontal_axis="z",
        vertical_axis="y",
    )
    x_xy, y_xy, bx_matrix = _plane_for_display(
        bx_xy,
        horizontal_axis="x",
        vertical_axis="y",
    )
    x_xy_by, y_xy_by, by_matrix = _plane_for_display(
        by_xy,
        horizontal_axis="x",
        vertical_axis="y",
    )
    if not np.allclose(x_xy, x_xy_by, rtol=1.0e-12, atol=1.0e-18) or not np.allclose(
        y_xy,
        y_xy_by,
        rtol=1.0e-12,
        atol=1.0e-18,
    ):
        raise ValueError("B/x and B/y transverse coordinates do not match")
    if not np.allclose(z_xz, z_yz, rtol=1.0e-12, atol=1.0e-18):
        raise ValueError("Longitudinal xz and yz z coordinates do not match")

    z_edges = _cell_edges(z_xz)
    z_front_grid = float(z_edges[-1])
    xi_grid = z_xz - z_front_grid
    xi_edges = z_edges - z_front_grid
    tracked_xi = z_m[tracked_present] - z_front_grid
    center_xi = center_z - z_front_grid

    xi_p01 = weighted_percentile(tracked_xi, 1.0, tracked_weights)
    xi_p99 = weighted_percentile(tracked_xi, 99.0, tracked_weights)
    if xi_p01 is None or xi_p99 is None:
        raise ValueError("Could not determine tracked longitudinal extent")
    xi_limits = (
        max(float(xi_edges[0]), float(xi_p01) - xi_padding_back),
        min(float(xi_edges[-1]), float(xi_p99) + xi_padding_front),
    )
    if xi_limits[0] >= xi_limits[1]:
        raise ValueError("Derived xi publication limits are empty")

    x_limits = (center_x - transverse_half_width, center_x + transverse_half_width)
    y_limits = (center_y - transverse_half_width, center_y + transverse_half_width)

    ez_xz_crop = _cropped_matrix_values(
        xi_grid,
        x_grid,
        ez_xz_matrix,
        xi_limits,
        x_limits,
    )
    ez_yz_crop = _cropped_matrix_values(
        xi_grid,
        y_grid,
        ez_yz_matrix,
        xi_limits,
        y_limits,
    )
    ez_limit_v_m = _finite_nonzero_percentile(
        np.concatenate((ez_xz_crop.ravel(), ez_yz_crop.ravel())),
        field_pct,
    )

    bperp_matrix = np.hypot(bx_matrix, by_matrix)
    bperp_crop = _cropped_matrix_values(
        x_xy,
        y_xy,
        bperp_matrix,
        x_limits,
        y_limits,
    )
    bperp_limit_t = _finite_nonzero_percentile(bperp_crop, magnetic_pct)

    tracked_energy = energy_MeV[tracked_present]
    energy_vmax = weighted_percentile(tracked_energy, energy_pct, tracked_weights)
    if energy_vmax is None or not np.isfinite(energy_vmax) or energy_vmax <= 0.0:
        energy_vmax = float(np.max(tracked_energy))
    if not np.isfinite(energy_vmax) or energy_vmax <= 0.0:
        energy_vmax = 1.0e-6
    energy_vmin = 0.0

    selection_xy = select_publication_particle_plane(
        frame,
        tracked_ids,
        plane="xy",
        coordinate_m=center_z,
        slab_half_width_m=slab_half_width,
    )
    selection_xz = select_publication_particle_plane(
        frame,
        tracked_ids,
        plane="xz",
        coordinate_m=center_y,
        slab_half_width_m=slab_half_width,
    )
    selection_yz = select_publication_particle_plane(
        frame,
        tracked_ids,
        plane="yz",
        coordinate_m=center_x,
        slab_half_width_m=slab_half_width,
    )

    xi_particles = z_m - z_front_grid
    view_xy = _view_mask(x_m, y_m, x_limits, y_limits)
    view_xz = _view_mask(xi_particles, x_m, xi_limits, x_limits)
    view_yz = _view_mask(xi_particles, y_m, xi_limits, y_limits)

    background_xy_mask = selection_xy.background_mask & view_xy
    background_xz_mask = selection_xz.background_mask & view_xz
    background_yz_mask = selection_yz.background_mask & view_yz
    tracked_xy_mask = tracked_present & view_xy
    tracked_xz_mask = tracked_present & view_xz
    tracked_yz_mask = tracked_present & view_yz

    background_xy_indices = deterministic_background_indices(
        background_xy_mask,
        frame.ids,
        max_points=max_background_points_per_panel,
    )
    background_xz_indices = deterministic_background_indices(
        background_xz_mask,
        frame.ids,
        max_points=max_background_points_per_panel,
    )
    background_yz_indices = deterministic_background_indices(
        background_yz_mask,
        frame.ids,
        max_points=max_background_points_per_panel,
    )
    tracked_xy_indices = np.flatnonzero(tracked_xy_mask)
    tracked_xz_indices = np.flatnonzero(tracked_xz_mask)
    tracked_yz_indices = np.flatnonzero(tracked_yz_mask)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.25))
    ax_xy, ax_xz, ax_yz = axes
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.80, wspace=0.18)

    x_edges = _cell_edges(x_xy)
    y_edges = _cell_edges(y_xy)
    b_image = ax_xy.pcolormesh(
        x_edges * 1.0e6,
        y_edges * 1.0e6,
        bperp_matrix / 1.0e3,
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=bperp_limit_t / 1.0e3,
        rasterized=True,
    )

    qx_indices = np.unique(
        np.linspace(0, x_xy.size - 1, min(quiver_max_arrows_per_axis, x_xy.size)).astype(int)
    )
    qy_indices = np.unique(
        np.linspace(0, y_xy.size - 1, min(quiver_max_arrows_per_axis, y_xy.size)).astype(int)
    )
    qx_grid, qy_grid = np.meshgrid(x_xy[qx_indices], y_xy[qy_indices])
    qbx = bx_matrix[np.ix_(qy_indices, qx_indices)]
    qby = by_matrix[np.ix_(qy_indices, qx_indices)]
    qmag = np.hypot(qbx, qby)
    quiver_scale_t = _finite_nonzero_percentile(qmag, 95.0)
    clip_factor = np.ones_like(qmag)
    nonzero_qmag = qmag > 0.0
    clip_factor[nonzero_qmag] = np.minimum(
        1.0,
        quiver_scale_t / qmag[nonzero_qmag],
    )
    qu = qbx / quiver_scale_t * clip_factor
    qv = qby / quiver_scale_t * clip_factor
    ax_xy.quiver(
        qx_grid * 1.0e6,
        qy_grid * 1.0e6,
        qu,
        qv,
        color="white",
        alpha=0.82,
        angles="xy",
        scale_units="width",
        scale=18.0,
        width=0.0032,
        headwidth=3.2,
        headlength=4.2,
        zorder=3,
    )

    z_edges_yz = _cell_edges(z_yz) - z_front_grid
    ez_xz_image = ax_xz.pcolormesh(
        xi_edges * 1.0e6,
        _cell_edges(x_grid) * 1.0e6,
        ez_xz_matrix / 1.0e12,
        shading="flat",
        cmap="RdBu_r",
        vmin=-ez_limit_v_m / 1.0e12,
        vmax=ez_limit_v_m / 1.0e12,
        rasterized=True,
    )
    ez_yz_image = ax_yz.pcolormesh(
        z_edges_yz * 1.0e6,
        _cell_edges(y_grid) * 1.0e6,
        ez_yz_matrix / 1.0e12,
        shading="flat",
        cmap="RdBu_r",
        vmin=-ez_limit_v_m / 1.0e12,
        vmax=ez_limit_v_m / 1.0e12,
        rasterized=True,
    )

    for ax, background_indices, horizontal_values, vertical_values in (
        (ax_xy, background_xy_indices, x_m, y_m),
        (ax_xz, background_xz_indices, xi_particles, x_m),
        (ax_yz, background_yz_indices, xi_particles, y_m),
    ):
        size, alpha = _background_scatter_style(int(background_indices.size))
        ax.scatter(
            horizontal_values[background_indices] * 1.0e6,
            vertical_values[background_indices] * 1.0e6,
            s=size,
            c="0.38",
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            zorder=4,
        )

    tracked_scatter = None
    for ax, tracked_indices, horizontal_values, vertical_values in (
        (ax_xy, tracked_xy_indices, x_m, y_m),
        (ax_xz, tracked_xz_indices, xi_particles, x_m),
        (ax_yz, tracked_yz_indices, xi_particles, y_m),
    ):
        size, alpha = _tracked_scatter_style(int(tracked_indices.size))
        tracked_scatter = ax.scatter(
            horizontal_values[tracked_indices] * 1.0e6,
            vertical_values[tracked_indices] * 1.0e6,
            s=size,
            c=energy_MeV[tracked_indices],
            cmap="turbo",
            vmin=energy_vmin,
            vmax=energy_vmax,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.12,
            rasterized=True,
            zorder=6,
        )

    if tracked_scatter is None:
        raise AssertionError("Internal error: no tracked scatter was created")

    _top_colorbar(fig, ax_xy, b_image, r"$|B_\perp|$ [kT]")
    _top_colorbar(fig, ax_xz, ez_xz_image, r"$E_z$ [TV/m]")
    _top_colorbar(fig, ax_yz, tracked_scatter, r"tracked bunch $E_{\mathrm{kin}}$ [MeV]")

    ax_xy.set_xlabel(r"$x$ [$\mu$m]")
    ax_xy.set_ylabel(r"$y$ [$\mu$m]")
    ax_xz.set_xlabel(r"$\xi_{\mathrm{grid}}$ [$\mu$m]")
    ax_xz.set_ylabel(r"$x$ [$\mu$m]")
    ax_yz.set_xlabel(r"$\xi_{\mathrm{grid}}$ [$\mu$m]")
    ax_yz.set_ylabel(r"$y$ [$\mu$m]")

    ax_xy.set_xlim(*(value * 1.0e6 for value in x_limits))
    ax_xy.set_ylim(*(value * 1.0e6 for value in y_limits))
    ax_xz.set_xlim(*(value * 1.0e6 for value in xi_limits))
    ax_xz.set_ylim(*(value * 1.0e6 for value in x_limits))
    ax_yz.set_xlim(*(value * 1.0e6 for value in xi_limits))
    ax_yz.set_ylim(*(value * 1.0e6 for value in y_limits))
    ax_xy.set_aspect("equal", adjustable="box")

    _panel_label(ax_xy, "(a)")
    _panel_label(ax_xz, "(b)")
    _panel_label(ax_yz, "(c)")

    fig.suptitle(
        f"{case_label} — step {frame.step}, t = {frame.time_s * 1.0e15:.2f} fs",
        fontsize=12,
        y=0.985,
    )

    output.mkdir(parents=True, exist_ok=False)
    output_png = output / f"fig9_like_step{frame.step:06d}.png"
    fig.savefig(output_png, dpi=dpi, facecolor="white")
    plt.close(fig)

    result = PublicationFig9SnapshotResult(
        step=frame.step,
        time_s=frame.time_s,
        output_png=str(output_png),
        manifest_json=str(output / "manifest.json"),
        particle_h5=str(particle_path),
        field_h5=str(field_path),
        tracked_ids_npy=str(tracked_path),
        n_electrons_total=frame.size,
        n_tracked_requested=int(tracked_ids.size),
        n_tracked_present=int(np.count_nonzero(tracked_present)),
        n_tracked_missing=int(missing_tracked.size),
        tracked_charge_present_pC=_charge_pC(weights, tracked_present),
        center_x_m=center_x,
        center_y_m=center_y,
        center_z_m=center_z,
        z_front_grid_m=z_front_grid,
        center_xi_grid_m=center_xi,
        xi_limits_m=(float(xi_limits[0]), float(xi_limits[1])),
        x_limits_m=(float(x_limits[0]), float(x_limits[1])),
        y_limits_m=(float(y_limits[0]), float(y_limits[1])),
        background_slab_half_width_m=slab_half_width,
        max_background_points_per_panel=max_background_points_per_panel,
        n_background_xz_available=int(np.count_nonzero(background_xz_mask)),
        n_background_xz_plotted=int(background_xz_indices.size),
        n_background_yz_available=int(np.count_nonzero(background_yz_mask)),
        n_background_yz_plotted=int(background_yz_indices.size),
        n_background_xy_available=int(np.count_nonzero(background_xy_mask)),
        n_background_xy_plotted=int(background_xy_indices.size),
        n_tracked_xz_plotted=int(tracked_xz_indices.size),
        n_tracked_yz_plotted=int(tracked_yz_indices.size),
        n_tracked_xy_plotted=int(tracked_xy_indices.size),
        ez_color_limit_TV_m=ez_limit_v_m / 1.0e12,
        bperp_color_limit_kT=bperp_limit_t / 1.0e3,
        energy_color_min_MeV=energy_vmin,
        energy_color_max_MeV=float(energy_vmax),
        field_slice_method=str(field_slice_method),
        background_particle_mode=(
            "physical slab around each field cut, then deterministic visual thinning"
        ),
        tracked_particle_mode=(
            "all present fixed final-bunch IDs projected into each panel; no thinning"
        ),
        background_sampling="smallest splitmix64 hashes of persistent particle IDs",
        quiver_representation=(
            "B_perp direction with magnitude normalized to p95 and clipped at unity; "
            "field magnitude remains encoded by the background color"
        ),
    )

    manifest = {
        "schema": PUBLICATION_FIG9_SCHEMA,
        "result": asdict(result),
        "field_percentile": field_pct,
        "magnetic_percentile": magnetic_pct,
        "energy_percentile": energy_pct,
        "xi_definition": "z_particle_or_cell_center - z_front_grid_from_field_edges",
        "background_particle_ids_plotted": {
            "xy": [int(value) for value in frame.ids[background_xy_indices]],
            "xz": [int(value) for value in frame.ids[background_xz_indices]],
            "yz": [int(value) for value in frame.ids[background_yz_indices]],
        },
        "tracked_particle_ids_plotted": {
            "xy": [int(value) for value in frame.ids[tracked_xy_indices]],
            "xz": [int(value) for value in frame.ids[tracked_xz_indices]],
            "yz": [int(value) for value in frame.ids[tracked_yz_indices]],
        },
        "missing_tracked_particle_ids": [int(value) for value in missing_tracked],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result
