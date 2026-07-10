from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .beam_metrics import BeamMetrics, kinetic_energy_MeV, pz_MeV_c
from .constants import E_CHARGE_C
from .particles import ParticleData


PLOT_FILENAMES = (
    "energy_spectrum.png",
    "longitudinal_phase_space_z_pz.png",
    "longitudinal_energy_space_z_energy.png",
    "transverse_phase_space_x_thetax.png",
    "transverse_phase_space_y_thetay.png",
    "transverse_real_space_xy.png",
    "transverse_divergence_thetax_thetay.png",
)


def particle_plot_paths(output_dir: str | Path) -> list[Path]:
    """Return the fixed particle-plot contract paths in write order."""
    root = Path(output_dir)
    return [root / filename for filename in PLOT_FILENAMES]


def write_particle_plots(
    particles: ParticleData,
    metrics: BeamMetrics,
    *,
    output_dir: str | Path,
    energy_threshold_MeV: float,
    spectrum_min_energy_MeV: float | None = None,
    max_points: int = 200_000,
    spectrum_bins: int = 200,
) -> list[Path]:
    """Write the fixed visual diagnostics for one particle species.

    The plot family follows the established guiding-analysis convention: a
    weighted energy spectrum, longitudinal phase/energy spaces, and four
    transverse views.  The CSV and the plots use the same hot-energy threshold.
    A valid PNG is written even when the threshold selects no plottable
    particles; that is a valid physical outcome, not an analysis crash.
    """
    particles.validate()

    threshold = float(energy_threshold_MeV)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("energy_threshold_MeV must be a finite non-negative value")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if spectrum_bins <= 0:
        raise ValueError("spectrum_bins must be positive")

    spectrum_min = _resolve_spectrum_min_energy(
        spectrum_min_energy_MeV,
        energy_threshold_MeV=threshold,
    )

    output_paths = particle_plot_paths(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    x_m = np.asarray(particles.x_m, dtype=float)
    y_m = np.asarray(particles.y_m, dtype=float)
    z_m = np.asarray(particles.z_m, dtype=float)
    px_si = np.asarray(particles.px_si, dtype=float)
    py_si = np.asarray(particles.py_si, dtype=float)
    pz_si = np.asarray(particles.pz_si, dtype=float)
    energy_mev = kinetic_energy_MeV(px_si, py_si, pz_si)

    valid = np.isfinite(energy_mev)
    for values in (x_m, y_m, z_m, px_si, py_si, pz_si):
        valid &= np.isfinite(values)

    weights: np.ndarray | None = None
    if particles.weighting is not None:
        weights = np.asarray(particles.weighting, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0.0)

    hot = valid & (energy_mev >= threshold)

    _save_energy_spectrum(
        path=output_paths[0],
        particles=particles,
        energy_mev=energy_mev,
        valid=valid,
        weights=weights,
        energy_threshold_MeV=threshold,
        spectrum_min_energy_MeV=spectrum_min,
        spectrum_bins=spectrum_bins,
    )

    z_um = z_m * 1.0e6
    pz_mev_c = pz_MeV_c(pz_si)
    hot_indices = _sample_indices(hot, max_points=max_points)
    _save_scatter_or_no_data(
        path=output_paths[1],
        x=z_um,
        y=pz_mev_c,
        color=energy_mev,
        indices=hot_indices,
        xlabel="z [um]",
        ylabel="p_z [MeV/c]",
        title="Longitudinal phase space z-p_z",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot",
    )
    _save_scatter_or_no_data(
        path=output_paths[2],
        x=z_um,
        y=energy_mev,
        color=None,
        indices=hot_indices,
        xlabel="z [um]",
        ylabel="Ekin [MeV]",
        title="Longitudinal energy space z-Ekin",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot",
    )

    forward_hot = hot & (pz_si > 0.0)
    transverse_indices = _sample_indices(forward_hot, max_points=max_points)
    x_um = x_m * 1.0e6
    y_um = y_m * 1.0e6
    theta_x_mrad = np.arctan2(px_si, pz_si) * 1.0e3
    theta_y_mrad = np.arctan2(py_si, pz_si) * 1.0e3

    _save_scatter_or_no_data(
        path=output_paths[3],
        x=x_um,
        y=theta_x_mrad,
        color=energy_mev,
        indices=transverse_indices,
        xlabel="x [um]",
        ylabel="theta_x [mrad]",
        title="Transverse phase space x-theta_x",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot, forward",
    )
    _save_scatter_or_no_data(
        path=output_paths[4],
        x=y_um,
        y=theta_y_mrad,
        color=energy_mev,
        indices=transverse_indices,
        xlabel="y [um]",
        ylabel="theta_y [mrad]",
        title="Transverse phase space y-theta_y",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot, forward",
    )
    _save_scatter_or_no_data(
        path=output_paths[5],
        x=x_um,
        y=y_um,
        color=energy_mev,
        indices=transverse_indices,
        xlabel="x [um]",
        ylabel="y [um]",
        title="Transverse real space x-y",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot, forward",
    )
    _save_scatter_or_no_data(
        path=output_paths[6],
        x=theta_x_mrad,
        y=theta_y_mrad,
        color=energy_mev,
        indices=transverse_indices,
        xlabel="theta_x [mrad]",
        ylabel="theta_y [mrad]",
        title="Transverse divergence theta_x-theta_y",
        particles=particles,
        metrics=metrics,
        energy_threshold_MeV=threshold,
        selection_label="hot, forward",
    )

    return output_paths


def _save_energy_spectrum(
    *,
    path: Path,
    particles: ParticleData,
    energy_mev: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray | None,
    energy_threshold_MeV: float,
    spectrum_min_energy_MeV: float,
    spectrum_bins: int,
) -> Path:
    if not np.any(valid):
        return _save_no_data_plot(
            path=path,
            title=f"{particles.species} energy spectrum, step {particles.step}",
            message="No finite, positive-weight particles",
            xlabel="electron kinetic energy [MeV]",
            ylabel="weighted counts [a.u.]",
        )

    spectrum_mask = valid & (energy_mev >= spectrum_min_energy_MeV)
    if not np.any(spectrum_mask):
        return _save_no_data_plot(
            path=path,
            title=f"{particles.species} energy spectrum, step {particles.step}",
            message=f"No valid particles with Ekin >= {spectrum_min_energy_MeV:g} MeV",
            xlabel="electron kinetic energy [MeV]",
            ylabel="weighted counts [a.u.]",
        )

    energy = energy_mev[spectrum_mask]
    selected_weights = None if weights is None else weights[spectrum_mask]
    energy_max = float(np.max(energy))
    upper = max(
        energy_max,
        energy_threshold_MeV * 1.2,
        spectrum_min_energy_MeV + max(1.0, 0.1 * spectrum_min_energy_MeV),
    )
    counts, edges = np.histogram(
        energy,
        bins=int(spectrum_bins),
        range=(spectrum_min_energy_MeV, upper),
        weights=selected_weights,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.stairs(counts, edges, linewidth=1.2)

    if spectrum_min_energy_MeV <= energy_threshold_MeV <= upper:
        ax.axvline(
            energy_threshold_MeV,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"hot threshold = {energy_threshold_MeV:g} MeV",
        )

    ax.set_yscale("log")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")

    if selected_weights is None:
        quantity = f"N = {energy.size}"
        ylabel = "macroparticle counts"
    else:
        charge_pc = float(np.sum(selected_weights) * E_CHARGE_C * 1.0e12)
        quantity = f"N = {energy.size}\nQ = {charge_pc:.3g} pC"
        ylabel = "weighted counts [a.u.]"

    ax.text(
        0.02,
        0.95,
        f"{quantity}\nEmax = {energy_max:.3g} MeV",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_xlabel("electron kinetic energy [MeV]")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{particles.species} energy spectrum, step {particles.step}\n"
        f"plotted electrons: Ekin >= {spectrum_min_energy_MeV:g} MeV"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _resolve_spectrum_min_energy(
    spectrum_min_energy_MeV: float | None,
    *,
    energy_threshold_MeV: float,
) -> float:
    if spectrum_min_energy_MeV is None:
        return energy_threshold_MeV

    cutoff = float(spectrum_min_energy_MeV)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("spectrum_min_energy_MeV must be a finite non-negative value")
    return cutoff


def _save_scatter_or_no_data(
    *,
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | None,
    indices: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    particles: ParticleData,
    metrics: BeamMetrics,
    energy_threshold_MeV: float,
    selection_label: str,
) -> Path:
    if indices.size == 0:
        return _save_no_data_plot(
            path=path,
            title=f"{title}, {particles.species}, step {particles.step}",
            message=(
                f"No {selection_label} particles\nEkin >= {energy_threshold_MeV:g} MeV"
            ),
            xlabel=xlabel,
            ylabel=ylabel,
        )

    point_size, alpha = _scatter_style_for_npoints(int(indices.size))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if color is None:
        ax.scatter(
            x[indices],
            y[indices],
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )
    else:
        scatter = ax.scatter(
            x[indices],
            y[indices],
            s=point_size,
            c=color[indices],
            alpha=alpha,
            edgecolors="none",
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Ekin [MeV]")

    ax.text(
        0.02,
        0.98,
        _plot_annotation(metrics, n_plotted=int(indices.size)),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}, {particles.species}, step {particles.step}\n"
        f"Ekin >= {energy_threshold_MeV:g} MeV; {selection_label} selection"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _sample_indices(mask: np.ndarray, *, max_points: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= max_points:
        return indices
    keep = np.linspace(0, indices.size - 1, int(max_points)).astype(int)
    return indices[keep]


def _scatter_style_for_npoints(n: int) -> tuple[float, float]:
    if n <= 200:
        return 18.0, 0.85
    if n <= 2_000:
        return 8.0, 0.65
    if n <= 20_000:
        return 2.5, 0.45
    return 1.0, 0.30


def _plot_annotation(metrics: BeamMetrics, *, n_plotted: int) -> str:
    charge = (
        "nan"
        if metrics.charge_selected_pC is None
        else f"{metrics.charge_selected_pC:.3g} pC"
    )
    energy_p95 = (
        "nan" if metrics.energy_p95_MeV is None else f"{metrics.energy_p95_MeV:.3g} MeV"
    )
    forward = (
        "nan" if metrics.forward_fraction is None else f"{metrics.forward_fraction:.3g}"
    )
    return (
        f"N hot = {metrics.n_macroparticles_selected}, plotted = {n_plotted}\n"
        f"Q hot = {charge}; E p95 = {energy_p95}\n"
        f"forward fraction = {forward}"
    )


def _save_no_data_plot(
    *,
    path: Path,
    title: str,
    message: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
