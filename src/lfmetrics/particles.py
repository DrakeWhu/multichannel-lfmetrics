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
        arrays = [self.x_m, self.y_m, self.z_m, self.px_si, self.py_si, self.pz_si]
        lengths = {np.asarray(a).shape[0] for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"Particle arrays have inconsistent lengths: {sorted(lengths)}"
            )
        if self.weighting is not None and np.asarray(self.weighting).shape[0] != next(
            iter(lengths)
        ):
            raise ValueError("weighting has a different length from particle arrays")

    @property
    def size(self) -> int:
        return int(np.asarray(self.x_m).shape[0])
