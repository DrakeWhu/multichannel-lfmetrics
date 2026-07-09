import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.cli import main
from lfmetrics.constants import C_LIGHT, M_E_KG


def write_species(group, name, gamma_beta_values, with_weighting=True):
    species = group.create_group(name)

    count = len(gamma_beta_values)

    position = species.create_group("position")
    position.create_dataset("x", data=np.linspace(0.0, 1e-6, count))
    position.create_dataset("y", data=np.linspace(0.0, 2e-6, count))
    position.create_dataset("z", data=np.linspace(0.0, 3e-6, count))

    momentum = species.create_group("momentum")
    momentum.create_dataset("x", data=np.zeros(count))
    momentum.create_dataset("y", data=np.zeros(count))
    momentum.create_dataset("z", data=np.asarray(gamma_beta_values) * M_E_KG * C_LIGHT)

    if with_weighting:
        species.create_dataset("weighting", data=np.arange(1, count + 1, dtype=float))


def write_synthetic_openpmd_file(path: Path, step: int = 5000):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        iteration = data.create_group(str(step))
        particles = iteration.create_group("particles")
        write_species(particles, "beam", gamma_beta_values=[2.0, 20.0, 50.0])
        write_species(particles, "electrons", gamma_beta_values=[1.0, 5.0, 10.0])


class CLITests(unittest.TestCase):
    def test_analyze_h5_writes_particle_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            h5_path = tmp_path / "openpmd_005000.h5"
            output = tmp_path / "particle_summary.csv"
            write_synthetic_openpmd_file(h5_path)

            rc = main(
                [
                    "analyze-h5",
                    str(h5_path),
                    "--species",
                    "beam",
                    "--energy-threshold-MeV",
                    "5",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["species"], "beam")
            self.assertEqual(rows[0]["step"], "5000")
            self.assertGreater(float(rows[0]["energy_p95_MeV"]), 0.0)

    def test_analyze_case_writes_default_post_output_for_multiple_species(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case_001"
            data_dir = case_dir / "3D"
            data_dir.mkdir(parents=True)
            write_synthetic_openpmd_file(data_dir / "openpmd_005000.h5")

            rc = main(
                [
                    "analyze-case",
                    str(case_dir),
                    "--species",
                    "beam,electrons",
                    "--energy-threshold-MeV",
                    "0",
                ]
            )

            output = case_dir / "post" / "particle_summary.csv"

            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["species"] for row in rows], ["beam", "electrons"])

    def test_missing_case_returns_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_case"

            rc = main(["analyze-case", str(missing)])

            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
