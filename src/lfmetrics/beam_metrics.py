from __future__ import annotations

from dataclasses import asdict, dataclass
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

    beam_quality_score: Optional[float]
    beam_quality_status: str
    beam_quality_flags: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def kinetic_energy_MeV(
    px_si: np.ndarray, py_si: np.ndarray, pz_si: np.ndarray
) -> np.ndarray:
    """Relativistic kinetic energy in MeV from SI momentum components."""
    px = np.asarray(px_si, dtype=float)
    py = np.asarray(py_si, dtype=float)
    pz = np.asarray(pz_si, dtype=float)

    p2 = px * px + py * py + pz * pz
    gamma = np.sqrt(1.0 + p2 / (M_E_KG * C_LIGHT) ** 2)
    return (gamma - 1.0) * M_E_KG * C_LIGHT**2 * J_TO_MEV


def pz_MeV_c(pz_si: np.ndarray) -> np.ndarray:
    """Longitudinal momentum in MeV/c."""
    return np.asarray(pz_si, dtype=float) * C_LIGHT * J_TO_MEV


def _clean_weights(weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return weights
    if np.any(weights < 0.0):
        raise ValueError("Particle weights must be non-negative")
    return weights


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = _clean_weights(weights)

    if values.size == 0:
        return None
    if weights is None:
        return float(np.mean(values))

    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return None
    return float(np.sum(values * weights) / wsum)


def weighted_std(values: np.ndarray, weights: Optional[np.ndarray]) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = _clean_weights(weights)

    mean = weighted_mean(values, weights)
    if mean is None:
        return None
    if weights is None:
        return float(np.std(values))

    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        return None
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / wsum))


def weighted_percentile(
    values: np.ndarray,
    percentile: float,
    weights: Optional[np.ndarray] = None,
) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = _clean_weights(weights)

    if values.size == 0:
        return None
    if weights is None:
        return float(np.percentile(values, percentile))

    if weights.shape[0] != values.shape[0]:
        raise ValueError("weights and values must have the same length")

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return None

    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)

    cutoff = percentile / 100.0 * total_weight
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, sorted_values.size - 1)
    return float(sorted_values[index])


def normalized_emittance_um_rad(
    transverse_position_m: np.ndarray,
    transverse_momentum_si: np.ndarray,
    other_transverse_momentum_si: np.ndarray,
    pz_si: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Paraxial normalized emittance using x' ~= px/pz and <gamma> eps_geom."""
    x = np.asarray(transverse_position_m, dtype=float)
    px = np.asarray(transverse_momentum_si, dtype=float)
    py = np.asarray(other_transverse_momentum_si, dtype=float)
    pz = np.asarray(pz_si, dtype=float)
    weights = _clean_weights(weights)

    if x.size < 2:
        return None

    eps = 1e-30 * M_E_KG * C_LIGHT
    pz_safe = np.where(np.abs(pz) < eps, eps, pz)
    xp = px / pz_safe

    if weights is None:
        x_mean = float(np.mean(x))
        xp_mean = float(np.mean(xp))
        x_c = x - x_mean
        xp_c = xp - xp_mean

        sig_x2 = float(np.mean(x_c * x_c))
        sig_xp2 = float(np.mean(xp_c * xp_c))
        sig_xxp = float(np.mean(x_c * xp_c))
    else:
        wsum = float(np.sum(weights))
        if wsum <= 0.0:
            return None

        x_mean = float(np.sum(weights * x) / wsum)
        xp_mean = float(np.sum(weights * xp) / wsum)
        x_c = x - x_mean
        xp_c = xp - xp_mean

        sig_x2 = float(np.sum(weights * x_c * x_c) / wsum)
        sig_xp2 = float(np.sum(weights * xp_c * xp_c) / wsum)
        sig_xxp = float(np.sum(weights * x_c * xp_c) / wsum)

    eps_geom2 = sig_x2 * sig_xp2 - sig_xxp * sig_xxp
    if eps_geom2 < 0.0 and abs(eps_geom2) < 1e-30:
        eps_geom2 = 0.0
    if eps_geom2 < 0.0 or not np.isfinite(eps_geom2):
        return None

    p_abs = np.sqrt(px * px + py * py + pz * pz)
    gamma = np.sqrt(1.0 + (p_abs / (M_E_KG * C_LIGHT)) ** 2)
    gamma_bar = weighted_mean(gamma, weights)
    if gamma_bar is None:
        return None

    return float(gamma_bar * np.sqrt(eps_geom2) * 1e6)


def _quality_score(
    energy_p95_MeV: Optional[float],
    charge_selected_pC: Optional[float],
    theta_r_p95_mrad: Optional[float],
    emitn_xy_um_rad: Optional[float],
) -> tuple[Optional[float], str, str]:
    """Simple transparent first-pass beam quality score.

    This is intentionally conservative and visible. It is not the final MORBO
    objective, only a first diagnostic score for ranking pilot runs.
    """
    flags: list[str] = []

    if energy_p95_MeV is None:
        flags.append("missing_energy_p95")
    if charge_selected_pC is None:
        flags.append("missing_charge_weighting")
    if theta_r_p95_mrad is None:
        flags.append("missing_theta_p95")
    if emitn_xy_um_rad is None:
        flags.append("missing_emitn_xy")

    if flags:
        return None, "incomplete", ";".join(flags)

    energy_component = min(max(energy_p95_MeV / 50.0, 0.0), 1.0)
    charge_component = min(max(charge_selected_pC / 10.0, 0.0), 1.0)
    divergence_component = 1.0 / (1.0 + max(theta_r_p95_mrad, 0.0) / 50.0)
    emit_component = 1.0 / (1.0 + max(emitn_xy_um_rad, 0.0) / 10.0)

    score = energy_component * charge_component * divergence_component * emit_component
    return float(score), "ok", ""


def compute_beam_metrics(
    particles: ParticleData,
    energy_threshold_MeV: float = 5.0,
) -> BeamMetrics:
    particles.validate()

    px = np.asarray(particles.px_si, dtype=float)
    py = np.asarray(particles.py_si, dtype=float)
    pz = np.asarray(particles.pz_si, dtype=float)
    x = np.asarray(particles.x_m, dtype=float)
    y = np.asarray(particles.y_m, dtype=float)
    weights = (
        None
        if particles.weighting is None
        else np.asarray(particles.weighting, dtype=float)
    )

    energy = kinetic_energy_MeV(px, py, pz)
    selected = energy >= energy_threshold_MeV

    px_sel = px[selected]
    py_sel = py[selected]
    pz_sel = pz[selected]
    x_sel = x[selected]
    y_sel = y[selected]
    energy_sel = energy[selected]
    weights_sel = None if weights is None else weights[selected]

    pz_mevc = pz_MeV_c(pz_sel)
    theta_r = np.sqrt(px_sel * px_sel + py_sel * py_sel) / np.maximum(
        np.abs(pz_sel), 1e-300
    )

    weight_total = None if weights is None else float(np.sum(weights))
    weight_selected = None if weights_sel is None else float(np.sum(weights_sel))
    charge_selected_pC = None
    if weight_selected is not None:
        charge_selected_pC = weight_selected * E_CHARGE_C * 1e12

    emitn_x = normalized_emittance_um_rad(x_sel, px_sel, py_sel, pz_sel, weights_sel)
    emitn_y = normalized_emittance_um_rad(y_sel, py_sel, px_sel, pz_sel, weights_sel)

    emitn_xy = None
    if emitn_x is not None and emitn_y is not None:
        emitn_xy = float(np.sqrt(max(emitn_x, 0.0) * max(emitn_y, 0.0)))

    energy_p95 = weighted_percentile(energy_sel, 95, weights_sel)
    theta_p95 = weighted_percentile(theta_r * 1e3, 95, weights_sel)

    quality_score, quality_status, quality_flags = _quality_score(
        energy_p95_MeV=energy_p95,
        charge_selected_pC=charge_selected_pC,
        theta_r_p95_mrad=theta_p95,
        emitn_xy_um_rad=emitn_xy,
    )

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
        energy_median_MeV=weighted_percentile(energy_sel, 50, weights_sel),
        energy_p90_MeV=weighted_percentile(energy_sel, 90, weights_sel),
        energy_p95_MeV=energy_p95,
        energy_max_MeV=None if energy_sel.size == 0 else float(np.max(energy_sel)),
        pz_mean_MeV_c=weighted_mean(pz_mevc, weights_sel),
        pz_p95_MeV_c=weighted_percentile(pz_mevc, 95, weights_sel),
        forward_fraction=None if energy_sel.size == 0 else float(np.mean(pz_sel > 0.0)),
        theta_r_rms_mrad=None
        if theta_r.size == 0
        else float(np.sqrt(np.mean(theta_r * theta_r)) * 1e3),
        theta_r_p95_mrad=theta_p95,
        emitn_x_um_rad=emitn_x,
        emitn_y_um_rad=emitn_y,
        emitn_xy_um_rad=emitn_xy,
        beam_quality_score=quality_score,
        beam_quality_status=quality_status,
        beam_quality_flags=quality_flags,
    )
