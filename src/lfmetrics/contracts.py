from __future__ import annotations

from dataclasses import fields

from .beam_metrics import BeamMetrics
from .soft100 import Soft100Metrics


PARTICLE_SUMMARY_COLUMNS = [
    *[field.name for field in fields(BeamMetrics)],
    *[field.name for field in fields(Soft100Metrics)],
]


def validate_particle_summary_row(row: dict[str, object]) -> None:
    missing = [name for name in PARTICLE_SUMMARY_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"Missing particle summary columns: {missing}")
