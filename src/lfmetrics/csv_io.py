from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import PARTICLE_SUMMARY_COLUMNS


def write_particle_summary_csv(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Write particle summary rows using the LFMetrics contract columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PARTICLE_SUMMARY_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    return path
