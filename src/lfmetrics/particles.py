from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ParticleData:
    """Particle arrays for one species at one WarpX/openPMD iteration."""

    species: str
    step: int
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    px_si: np.ndarray
    py_si: np.ndarray
    pz_si: np.ndarray
    weighting: Optional[np.ndarray] = None

    def validate(self) -> None:
        arrays = [
            self.x_m,
            self.y_m,
            self.z_m,
            self.px_si,
            self.py_si,
            self.pz_si,
        ]
        lengths = {np.asarray(array).shape[0] for array in arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"Particle arrays have inconsistent lengths: {sorted(lengths)}"
            )

        if self.weighting is not None:
            weight_len = np.asarray(self.weighting).shape[0]
            particle_len = next(iter(lengths))
            if weight_len != particle_len:
                raise ValueError(
                    f"weighting length {weight_len} does not match particle length {particle_len}"
                )

    @property
    def size(self) -> int:
        return int(np.asarray(self.x_m).shape[0])
