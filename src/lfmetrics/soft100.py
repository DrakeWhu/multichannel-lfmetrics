from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .beam_metrics import (
    kinetic_energy_MeV,
    weighted_mean,
    weighted_percentile,
    weighted_std,
)
from .constants import C_LIGHT, E_CHARGE_C, M_E_KG
from .particles import ParticleData


SOFT100_SCHEMA_VERSION = "soft100_v1"


@dataclass(frozen=True)
class Soft100Config:
    """Particle-level smooth acceptance for the 100 MeV campaign target."""

    energy_low_mev: float = 5.0
    energy_target_mev: float = 100.0
    reliability_floor: float = 0.05
    effective_count_reference: float = 100.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.energy_low_mev,
                self.energy_target_mev,
                self.reliability_floor,
                self.effective_count_reference,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Soft100Config values must be finite")
        if self.energy_low_mev < 0.0:
            raise ValueError("energy_low_mev must be non-negative")
        if self.energy_target_mev <= self.energy_low_mev:
            raise ValueError("energy_target_mev must be greater than energy_low_mev")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be within [0, 1]")
        if self.effective_count_reference <= 0.0:
            raise ValueError("effective_count_reference must be positive")


@dataclass(frozen=True)
class Soft100Metrics:
    soft100_schema_version: str
    soft100_energy_low_MeV: float
    soft100_energy_target_MeV: float
    soft100_reliability_floor: float
    soft100_effective_count_reference: float
    soft100_forward_only: bool
    soft100_status: str

    n_macroparticles_soft100: int
    weight_soft100: Optional[float]
    charge_soft100_pC: Optional[float]
    n_effective_soft100: Optional[float]
    reliability_soft100: Optional[float]

    energy_mean_soft100_MeV: Optional[float]
    energy_spread_rms_soft100_MeV: Optional[float]
    energy_relative_spread_rms_soft100: Optional[float]
    energy_p10_soft100_MeV: Optional[float]
    energy_p50_soft100_MeV: Optional[float]
    energy_p90_soft100_MeV: Optional[float]
    energy_p95_soft100_MeV: Optional[float]
    energy_max_soft100_MeV: Optional[float]

    theta_r_p90_soft100_mrad: Optional[float]
    theta_r_p95_soft100_mrad: Optional[float]
    emitn_x_soft100_um_rad: Optional[float]
    emitn_y_soft100_um_rad: Optional[float]
    emitn_xy_soft100_um_rad: Optional[float]

    charge_Ege100MeV_pC: Optional[float]
    n_macroparticles_Ege100MeV: int
    charge_Ege5MeV_forward_pC: Optional[float]
    halo_fraction_soft100: Optional[float]

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def smooth_energy_acceptance(
    energy_mev: np.ndarray,
    *,
    energy_low_mev: float,
    energy_target_mev: float,
) -> np.ndarray:
    """Cubic smoothstep acceptance: zero below low and one at/above target."""

    low = float(energy_low_mev)
    target = float(energy_target_mev)
    if not np.isfinite(low) or not np.isfinite(target) or target <= low:
        raise ValueError("energy acceptance requires finite target > low")

    energy = np.asarray(energy_mev, dtype=float)
    u = np.clip((energy - low) / (target - low), 0.0, 1.0)
    acceptance = 3.0 * u * u - 2.0 * u * u * u
    return np.where(np.isfinite(energy), acceptance, 0.0)


def soft_particle_weights(
    energy_mev: np.ndarray,
    physical_weights: np.ndarray,
    *,
    energy_low_mev: float,
    energy_target_mev: float,
) -> np.ndarray:
    """Return W_i = w_i a_i, ignoring invalid or non-positive physical weights."""

    energy = np.asarray(energy_mev, dtype=float)
    weights = np.asarray(physical_weights, dtype=float)
    if energy.shape != weights.shape:
        raise ValueError("energy and physical_weights must have the same shape")

    acceptance = smooth_energy_acceptance(
        energy,
        energy_low_mev=energy_low_mev,
        energy_target_mev=energy_target_mev,
    )
    valid = np.isfinite(weights) & (weights > 0.0)
    return np.where(valid, weights * acceptance, 0.0)


def soft_charge_pC(soft_weights: np.ndarray) -> float:
    """Physical charge represented by positive finite soft particle weights."""

    weights = np.asarray(soft_weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0.0)
    return float(np.sum(weights[valid]) * E_CHARGE_C * 1.0e12)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective count from the supplied weights."""

    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return 0.0

    selected = weights[valid]
    denominator = float(np.sum(selected * selected))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(selected) ** 2 / denominator)


def soft_reliability(
    n_effective: float,
    *,
    reliability_floor: float,
    effective_count_reference: float,
) -> float:
    """Bounded confidence diagnostic; this is not a performance objective."""

    n_eff = max(float(n_effective), 0.0)
    floor = float(reliability_floor)
    reference = float(effective_count_reference)
    if not np.isfinite(n_eff) or not np.isfinite(floor) or not np.isfinite(reference):
        raise ValueError("reliability inputs must be finite")
    if not 0.0 <= floor <= 1.0 or reference <= 0.0:
        raise ValueError("invalid reliability floor/reference")
    return float(floor + (1.0 - floor) * (1.0 - np.exp(-n_eff / reference)))


def weighted_relative_spread_rms(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return weighted RMS spread divided by the weighted mean, without caps."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")

    selected_values = values[valid]
    selected_weights = weights[valid]
    mean = float(np.average(selected_values, weights=selected_weights))
    if not np.isfinite(mean) or mean <= 0.0:
        return float("nan")

    variance = float(
        np.average((selected_values - mean) ** 2, weights=selected_weights)
    )
    return float(np.sqrt(max(variance, 0.0)) / mean)


def _weighted_covariance(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    first_mean = float(np.average(first, weights=weights))
    second_mean = float(np.average(second, weights=weights))
    return float(
        np.average(
            (first - first_mean) * (second - second_mean),
            weights=weights,
        )
    )


def _normalized_emittance_um_rad(
    position_m: np.ndarray,
    normalized_momentum: np.ndarray,
    weights: np.ndarray,
) -> float:
    position_variance = _weighted_covariance(position_m, position_m, weights)
    momentum_variance = _weighted_covariance(
        normalized_momentum,
        normalized_momentum,
        weights,
    )
    covariance = _weighted_covariance(position_m, normalized_momentum, weights)
    determinant = position_variance * momentum_variance - covariance * covariance
    return float(np.sqrt(max(determinant, 0.0)) * 1.0e6)


def _metrics_without_weighting(
    *,
    config: Soft100Config,
    n_soft: int,
    n_hard: int,
) -> Soft100Metrics:
    return Soft100Metrics(
        soft100_schema_version=SOFT100_SCHEMA_VERSION,
        soft100_energy_low_MeV=float(config.energy_low_mev),
        soft100_energy_target_MeV=float(config.energy_target_mev),
        soft100_reliability_floor=float(config.reliability_floor),
        soft100_effective_count_reference=float(
            config.effective_count_reference
        ),
        soft100_forward_only=True,
        soft100_status="missing_weighting",
        n_macroparticles_soft100=n_soft,
        weight_soft100=None,
        charge_soft100_pC=None,
        n_effective_soft100=None,
        reliability_soft100=None,
        energy_mean_soft100_MeV=None,
        energy_spread_rms_soft100_MeV=None,
        energy_relative_spread_rms_soft100=None,
        energy_p10_soft100_MeV=None,
        energy_p50_soft100_MeV=None,
        energy_p90_soft100_MeV=None,
        energy_p95_soft100_MeV=None,
        energy_max_soft100_MeV=None,
        theta_r_p90_soft100_mrad=None,
        theta_r_p95_soft100_mrad=None,
        emitn_x_soft100_um_rad=None,
        emitn_y_soft100_um_rad=None,
        emitn_xy_soft100_um_rad=None,
        charge_Ege100MeV_pC=None,
        n_macroparticles_Ege100MeV=n_hard,
        charge_Ege5MeV_forward_pC=None,
        halo_fraction_soft100=None,
    )


def compute_soft100_metrics(
    particles: ParticleData,
    config: Soft100Config | None = None,
) -> Soft100Metrics:
    """Compute raw, forward-only soft100 metrics from one particle species."""

    particles.validate()
    cfg = config or Soft100Config()

    px = np.asarray(particles.px_si, dtype=float)
    py = np.asarray(particles.py_si, dtype=float)
    pz = np.asarray(particles.pz_si, dtype=float)
    x = np.asarray(particles.x_m, dtype=float)
    y = np.asarray(particles.y_m, dtype=float)
    energy = kinetic_energy_MeV(px, py, pz)
    acceptance = smooth_energy_acceptance(
        energy,
        energy_low_mev=cfg.energy_low_mev,
        energy_target_mev=cfg.energy_target_mev,
    )

    forward = np.isfinite(energy) & np.isfinite(pz) & (pz > 0.0)
    accepted_without_weights = forward & (acceptance > 0.0)
    hard_without_weights = forward & (energy >= cfg.energy_target_mev)

    if particles.weighting is None:
        return _metrics_without_weighting(
            config=cfg,
            n_soft=int(np.count_nonzero(accepted_without_weights)),
            n_hard=int(np.count_nonzero(hard_without_weights)),
        )

    physical_weights = np.asarray(particles.weighting, dtype=float)
    valid_weights = np.isfinite(physical_weights) & (physical_weights > 0.0)
    base = forward & valid_weights
    soft_weights = np.where(base, physical_weights * acceptance, 0.0)
    soft = soft_weights > 0.0
    hard = base & (energy >= cfg.energy_target_mev)
    hot_forward = base & (energy >= cfg.energy_low_mev)

    weight_soft = float(np.sum(soft_weights))
    weight_hard = float(np.sum(physical_weights[hard])) if np.any(hard) else 0.0
    weight_hot_forward = (
        float(np.sum(physical_weights[hot_forward]))
        if np.any(hot_forward)
        else 0.0
    )
    n_effective = effective_sample_size(soft_weights)
    reliability = soft_reliability(
        n_effective,
        reliability_floor=cfg.reliability_floor,
        effective_count_reference=cfg.effective_count_reference,
    )
    halo_fraction = (
        1.0 - weight_soft / weight_hot_forward
        if weight_hot_forward > 0.0
        else None
    )

    common = {
        "soft100_schema_version": SOFT100_SCHEMA_VERSION,
        "soft100_energy_low_MeV": float(cfg.energy_low_mev),
        "soft100_energy_target_MeV": float(cfg.energy_target_mev),
        "soft100_reliability_floor": float(cfg.reliability_floor),
        "soft100_effective_count_reference": float(
            cfg.effective_count_reference
        ),
        "soft100_forward_only": True,
        "n_macroparticles_soft100": int(np.count_nonzero(soft)),
        "weight_soft100": weight_soft,
        "charge_soft100_pC": soft_charge_pC(soft_weights),
        "n_effective_soft100": n_effective,
        "reliability_soft100": reliability,
        "charge_Ege100MeV_pC": weight_hard * E_CHARGE_C * 1.0e12,
        "n_macroparticles_Ege100MeV": int(np.count_nonzero(hard)),
        "charge_Ege5MeV_forward_pC": (
            weight_hot_forward * E_CHARGE_C * 1.0e12
        ),
        "halo_fraction_soft100": halo_fraction,
    }

    if not np.any(soft):
        return Soft100Metrics(
            **common,
            soft100_status="no_accepted_particles",
            energy_mean_soft100_MeV=None,
            energy_spread_rms_soft100_MeV=None,
            energy_relative_spread_rms_soft100=None,
            energy_p10_soft100_MeV=None,
            energy_p50_soft100_MeV=None,
            energy_p90_soft100_MeV=None,
            energy_p95_soft100_MeV=None,
            energy_max_soft100_MeV=None,
            theta_r_p90_soft100_mrad=None,
            theta_r_p95_soft100_mrad=None,
            emitn_x_soft100_um_rad=None,
            emitn_y_soft100_um_rad=None,
            emitn_xy_soft100_um_rad=None,
        )

    selected_energy = energy[soft]
    selected_weights = soft_weights[soft]
    energy_spread = weighted_std(selected_energy, selected_weights)

    energy_metrics = {
        "energy_mean_soft100_MeV": weighted_mean(
            selected_energy,
            selected_weights,
        ),
        "energy_spread_rms_soft100_MeV": energy_spread,
        "energy_relative_spread_rms_soft100": weighted_relative_spread_rms(
            selected_energy,
            selected_weights,
        ),
        "energy_p10_soft100_MeV": weighted_percentile(
            selected_energy,
            10.0,
            selected_weights,
        ),
        "energy_p50_soft100_MeV": weighted_percentile(
            selected_energy,
            50.0,
            selected_weights,
        ),
        "energy_p90_soft100_MeV": weighted_percentile(
            selected_energy,
            90.0,
            selected_weights,
        ),
        "energy_p95_soft100_MeV": weighted_percentile(
            selected_energy,
            95.0,
            selected_weights,
        ),
        "energy_max_soft100_MeV": float(np.max(selected_energy)),
    }

    transverse = (
        soft
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(pz)
    )
    if not np.any(transverse):
        return Soft100Metrics(
            **common,
            **energy_metrics,
            soft100_status="no_transverse_particles",
            theta_r_p90_soft100_mrad=None,
            theta_r_p95_soft100_mrad=None,
            emitn_x_soft100_um_rad=None,
            emitn_y_soft100_um_rad=None,
            emitn_xy_soft100_um_rad=None,
        )

    transverse_weights = soft_weights[transverse]
    px_transverse = px[transverse]
    py_transverse = py[transverse]
    pz_transverse = pz[transverse]
    theta_r_mrad = np.sqrt(
        np.arctan2(px_transverse, pz_transverse) ** 2
        + np.arctan2(py_transverse, pz_transverse) ** 2
    ) * 1.0e3
    momentum_scale = M_E_KG * C_LIGHT
    emitn_x = _normalized_emittance_um_rad(
        x[transverse],
        px_transverse / momentum_scale,
        transverse_weights,
    )
    emitn_y = _normalized_emittance_um_rad(
        y[transverse],
        py_transverse / momentum_scale,
        transverse_weights,
    )

    return Soft100Metrics(
        **common,
        **energy_metrics,
        soft100_status="ok",
        theta_r_p90_soft100_mrad=weighted_percentile(
            theta_r_mrad,
            90.0,
            transverse_weights,
        ),
        theta_r_p95_soft100_mrad=weighted_percentile(
            theta_r_mrad,
            95.0,
            transverse_weights,
        ),
        emitn_x_soft100_um_rad=emitn_x,
        emitn_y_soft100_um_rad=emitn_y,
        emitn_xy_soft100_um_rad=float(
            np.sqrt(max(emitn_x, 0.0) * max(emitn_y, 0.0))
        ),
    )
