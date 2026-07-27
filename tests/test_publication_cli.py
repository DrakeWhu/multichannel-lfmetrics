import json
import tempfile
import unittest
from pathlib import Path

import h5py

from lfmetrics.cli import main


def write_particle_frame(path: Path, step: int = 5000) -> None:
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        iteration = data.create_group(str(step))
        particles = iteration.create_group("particles")
        electrons = particles.create_group("electrons")
        electrons.create_dataset("id", data=[1, 2])


class PublicationAuditCLITests(unittest.TestCase):
    def test_cli_writes_relative_output_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case_publication"
            diagnostics = case_dir / "particle_diagnostic"
            diagnostics.mkdir(parents=True)
            write_particle_frame(diagnostics / "openpmd_005000.h5")

            args = [
                "audit-publication-case",
                str(case_dir),
                "--output",
                "post/publication_case_audit.json",
            ]
            self.assertEqual(main(args), 0)

            output = case_dir / "post" / "publication_case_audit.json"
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "multichannel_publication_case_audit_v1")
            self.assertEqual(len(payload["series"]), 1)
            self.assertEqual(payload["series"][0]["series_kind"], "particles")
            self.assertEqual(payload["series"][0]["steps"], [5000])

            self.assertEqual(main(args), 2)

    def test_cli_missing_case_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            rc = main(
                [
                    "audit-publication-case",
                    str(missing),
                    "--output",
                    "audit.json",
                ]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
