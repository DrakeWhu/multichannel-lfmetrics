# src/lfmetrics/__init__.py
"""LFMetrics: particle and beam metrics for WarpX multichannel simulations."""

from .beam_metrics import BeamMetrics, compute_beam_metrics, kinetic_energy_MeV
from .particles import ParticleData

__all__ = [
    "BeamMetrics",
    "ParticleData",
    "compute_beam_metrics",
    "kinetic_energy_MeV",
]

__version__ = "0.1.0"
