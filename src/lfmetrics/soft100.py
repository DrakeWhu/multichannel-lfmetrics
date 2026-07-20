from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import E_CHARGE_C


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
