from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import publication_fig9_v3 as _v3


PUBLICATION_FIG9_SCHEMA = _v3.PUBLICATION_FIG9_SCHEMA
PublicationFig9SnapshotResult = _v3.PublicationFig9SnapshotResult
deterministic_background_indices = _v3.deterministic_background_indices

_MAGNETIC_RANGE_LOCK = threading.Lock()


def _validated_magnetic_color_limit_kT(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("magnetic_color_limit_kT must be finite and positive")
    return result


def write_publication_fig9_snapshot(**kwargs) -> PublicationFig9SnapshotResult:
    """Write the v3 snapshot with a fixed magnetic color-scale ceiling.

    The public default is 50 kT for coherent comparisons across frames. The
    lower magnetic visibility threshold remains charge-frame dependent through
    the configured percentile, while the color and nonlinear arrow-length
    normalization use the fixed ceiling.
    """

    magnetic_color_limit_kT = _validated_magnetic_color_limit_kT(
        kwargs.pop("magnetic_color_limit_kT", 50.0)
    )
    magnetic_color_limit_t = magnetic_color_limit_kT * 1.0e3

    original_range = _v3._positive_percentile_range

    def fixed_ceiling_range(
        values: np.ndarray,
        lower_percentile: float,
        upper_percentile: float,
    ) -> tuple[float, float]:
        lower, _ = original_range(
            values,
            lower_percentile,
            upper_percentile,
        )
        if not np.isfinite(lower) or lower < 0.0 or lower >= magnetic_color_limit_t:
            lower = 0.0
        return float(lower), float(magnetic_color_limit_t)

    with _MAGNETIC_RANGE_LOCK:
        _v3._positive_percentile_range = fixed_ceiling_range
        try:
            result = _v3.write_publication_fig9_snapshot(**kwargs)
        finally:
            _v3._positive_percentile_range = original_range

    compatible = replace(
        result,
        xy_vector_overlay="B/x,B/y quiver",
        magnetic_color_max_kT=magnetic_color_limit_kT,
    )

    manifest_path = Path(result.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"] = asdict(compatible)
    manifest["xy_vector_overlay"] = "B/x,B/y quiver"
    manifest["magnetic_color_limit_kT"] = magnetic_color_limit_kT
    manifest["magnetic_color_scale_mode"] = (
        "fixed upper limit; lower limit from magnetic visibility percentile"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return compatible


__all__ = [
    "PUBLICATION_FIG9_SCHEMA",
    "PublicationFig9SnapshotResult",
    "deterministic_background_indices",
    "write_publication_fig9_snapshot",
]
