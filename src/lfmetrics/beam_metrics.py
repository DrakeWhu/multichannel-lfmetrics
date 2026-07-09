from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import C_LIGHT, E_CHARGE_C, J_TO_MEV, M_E_KG
from .particles import ParticleData


@dataclass(frozen=True)
class BeamMetrics:
    species: str
    step: int
    n_macroparticles_total: int
    n_macroparticles_selected: int
    weight_total: Optional[float]
    weight_selected: Optional[float]
    charge_selected_pC: Optional[float]
    energy_threshold_MeV: float
    energy_mean_MeV: Optional[float]
    energy_std_MeV: Optional[float]
    energy_median_MeV: Optional[float]
    energy_p90_MeV: Optional[float]
    energy_p95_MeV: Optional[float]
    energy_max_MeV: Optional[float]
    pz_mean_MeV_c: Optional[float]
    pz_p95_MeV_c: Optional[float]
    forward_fraction: Optional[float]
    theta_r_rms_mrad: Optional[float]
    theta_r_p95_mrad: Optional[float]
    emitn_x_um_rad: Optional[float]
    emitn_y_um_rad: Optional[float]
    emitn_xy_um_rad: Optional[float]


def kinetic_energy_MeV(
    px_si: np.ndarray, py_si: np.ndarray, pz_si: np.ndarray
) -> np.ndarray:
    """Relativistic kinetic energy in MeV from SI momentum components."""
    p2 = px_si * px_si + py_si * py_si + pz_si * pz_si
    gamma = np.sqrt(1.0 + p2 / (M_E_KG * C_LIGHT) ** 2)
    return (gamma - 1.0) * M_E_KG * C_LIGHT**2 * J_TO_MEV


def pz_MeV_c(pz_si: np.ndarray) -> np.ndarray:
    """Longitudinal momentum in MeV/c."""
    return pz_si * C_LIGHT * J_TO_MEV


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    if values.size == 0:
        return None
    if weights is None:
        return float(np.mean(values))
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return None
    return float(np.sum(values * weights) / wsum)


def weighted_std(values: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    mean = weighted_mean(values, weights)
    if mean is None:
        return None
    if weights is None:
        return float(np.std(values))
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return None
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / wsum))


def percentile(values: np.ndarray, q: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def normalized_emittance_um_rad(
    x_m: np.ndarray, px_si: np.ndarray, py_si: np.ndarray, pz_si: np.ndarray
) -> Optional[float]:
    """Paraxial normalized emittance using x' ~= px/pz and <gamma> eps_geom."""
    if x_m.size < 2:
        return None

    eps = 1e-30 * M_E_KG * C_LIGHT
    pz_safe = np.where(np.abs(pz_si) < eps, eps, pz_si)
    xp = px_si / pz_safe

    x_c = x_m - np.mean(x_m)
    xp_c = xp - np.mean(xp)

    sig_x2 = np.mean(x_c * x_c)
    sig_xp2 = np.mean(xp_c * xp_c)
    sig_xxp = np.mean(x_c * xp_c)

    eps_geom2 = sig_x2 * sig_xp2 - sig_xxp * sig_xxp
    if eps_geom2 < 0.0 and abs(eps_geom2) < 1e-30:
        eps_geom2 = 0.0
    if eps_geom2 < 0.0 or not np.isfinite(eps_geom2):
        return None

    p = np.sqrt(px_si * px_si + py_si * py_si + pz_si * pz_si)
    gamma_bar = float(np.mean(np.sqrt(1.0 + (p / (M_E_KG * C_LIGHT)) ** 2)))
    return float(gamma_bar * np.sqrt(eps_geom2) * 1e6)


def compute_beam_metrics(
    particles: ParticleData, energy_threshold_MeV: float = 5.0
) -> BeamMetrics:
    particles.validate()

    px = np.asarray(particles.px_si, dtype=float)
    py = np.asarray(particles.py_si, dtype=float)
    pz = np.asarray(particles.pz_si, dtype=float)
    weights = (
        None
        if particles.weighting is None
        else np.asarray(particles.weighting, dtype=float)
    )

    energy = kinetic_energy_MeV(px, py, pz)
    selected = energy >= energy_threshold_MeV

    energy_sel = energy[selected]
    px_sel = px[selected]
    py_sel = py[selected]
    pz_sel = pz[selected]
    weights_sel = None if weights is None else weights[selected]

    pz_mevc = pz_MeV_c(pz_sel)
    p_abs = np.sqrt(px_sel * px_sel + py_sel * py_sel + pz_sel * pz_sel)
    theta_r = np.sqrt(px_sel * px_sel + py_sel * py_sel) / np.maximum(
        np.abs(pz_sel), 1e-300
    )

    weight_total = None if weights is None else float(np.sum(weights))
    weight_selected = None if weights_sel is None else float(np.sum(weights_sel))
    charge_selected_pC = (
        None if weight_selected is None else weight_selected * E_CHARGE_C * 1e12
    )

    emitn_x = normalized_emittance_um_rad(
        np.asarray(particles.x_m, dtype=float)[selected], px_sel, py_sel, pz_sel
    )
    emitn_y = normalized_emittance_um_rad(
        np.asarray(particles.y_m, dtype=float)[selected], py_sel, px_sel, pz_sel
    )
    emitn_xy = None
    if emitn_x is not None and emitn_y is not None:
        emitn_xy = float(np.sqrt(max(emitn_x, 0.0) * max(emitn_y, 0.0)))

    return BeamMetrics(
        species=particles.species,
        step=particles.step,
        n_macroparticles_total=particles.size,
        n_macroparticles_selected=int(np.sum(selected)),
        weight_total=weight_total,
        weight_selected=weight_selected,
        charge_selected_pC=charge_selected_pC,
        energy_threshold_MeV=energy_threshold_MeV,
        energy_mean_MeV=weighted_mean(energy_sel, weights_sel),
        energy_std_MeV=weighted_std(energy_sel, weights_sel),
        energy_median_MeV=percentile(energy_sel, 50),
        energy_p90_MeV=percentile(energy_sel, 90),
        energy_p95_MeV=percentile(energy_sel, 95),
        energy_max_MeV=None if energy_sel.size == 0 else float(np.max(energy_sel)),
        pz_mean_MeV_c=weighted_mean(pz_mevc, weights_sel),
        pz_p95_MeV_c=percentile(pz_mevc, 95),
        forward_fraction=None if energy_sel.size == 0 else float(np.mean(pz_sel > 0.0)),
        theta_r_rms_mrad=None
        if theta_r.size == 0
        else float(np.sqrt(np.mean(theta_r * theta_r)) * 1e3),
        theta_r_p95_mrad=percentile(theta_r * 1e3, 95),
        emitn_x_um_rad=emitn_x,
        emitn_y_um_rad=emitn_y,
        emitn_xy_um_rad=emitn_xy,
    )
