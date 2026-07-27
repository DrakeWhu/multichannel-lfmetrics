from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from .publication_fig9_v3 import (
    PUBLICATION_FIG9_SCHEMA,
    PublicationFig9SnapshotResult,
    deterministic_background_indices,
)
from .publication_fig9_v3 import (
    write_publication_fig9_snapshot as _write_publication_fig9_snapshot_v3,
)


def write_publication_fig9_snapshot(**kwargs) -> PublicationFig9SnapshotResult:
    """Write the v3 snapshot while preserving the established overlay label."""

    result = _write_publication_fig9_snapshot_v3(**kwargs)
    compatible = replace(result, xy_vector_overlay="B/x,B/y quiver")

    manifest_path = Path(result.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"] = asdict(compatible)
    manifest["xy_vector_overlay"] = "B/x,B/y quiver"
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
