import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.openpmd_h5 import (
    choose_species,
    last_openpmd_h5_file,
    list_species_in_h5,
    parse_openpmd_step,
    read_particles_from_case,
    read_particles_from_h5,
)


def write_species(group, name, offset_z=0.0, with_weighting=True):
    species = group.create_group(name)

    position = species.create_group("position")
    position.create_dataset("x", data=np.array([1.0, 2.0]))
    position.create_dataset("y", data=np.array([3.0, 4.0]))
    position.create_dataset("z", data=np.array([5.0, 6.0]))

    position_offset = species.create_group("positionOffset")
    position_offset.create_dataset("x", data=np.array([0.0, 0.0]))
    position_offset.create_dataset("y", data=np.array([0.0, 0.0]))
    position_offset.create_dataset("z", data=np.array([offset_z, offset_z]))

    momentum = species.create_group("momentum")
    momentum.create_dataset("x", data=np.array([0.1, 0.2]))
    momentum.create_dataset("y", data=np.array([0.3, 0.4]))
    momentum.create_dataset("z", data=np.array([0.5, 0.6]))

    if with_weighting:
        species.create_dataset("weighting", data=np.array([10.0, 20.0]))


def write_synthetic_openpmd_file(path: Path, step: int = 5000):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        iteration = data.create_group(str(step))
        particles = iteration.create_group("particles")

        write_species(particles, "electrons", offset_z=10.0, with_weighting=True)
        write_species(particles, "beam", offset_z=20.0, with_weighting=True)


class OpenPMDH5SyntheticTests(unittest.TestCase):
    def test_parse_openpmd_step(self):
        self.assertEqual(parse_openpmd_step("openpmd_005000.h5"), 5000)
        self.assertEqual(parse_openpmd_step("3D/openpmd_000100.h5"), 100)
        self.assertIsNone(parse_openpmd_step("data.h5"))

    def test_choose_species_prefers_electrons(self):
        self.assertEqual(choose_species(["electrons", "beam"]), "electrons")

    def test_choose_species_respects_explicit_species(self):
        self.assertEqual(
            choose_species(["electrons", "beam"], requested_species="electrons"),
            "electrons",
        )

    def test_list_species_in_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_005000.h5"
            write_synthetic_openpmd_file(path)

            self.assertEqual(list_species_in_h5(path), ["beam", "electrons"])

    def test_read_particles_from_h5_uses_electron_priority_and_position_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_005000.h5"
            write_synthetic_openpmd_file(path)

            particles = read_particles_from_h5(path)

            self.assertEqual(particles.species, "electrons")
            self.assertEqual(particles.step, 5000)
            np.testing.assert_allclose(particles.z_m, np.array([15.0, 16.0]))
            np.testing.assert_allclose(particles.weighting, np.array([10.0, 20.0]))

    def test_read_particles_from_h5_can_select_electrons(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_005000.h5"
            write_synthetic_openpmd_file(path)

            particles = read_particles_from_h5(path, species="electrons")

            self.assertEqual(particles.species, "electrons")
            np.testing.assert_allclose(particles.z_m, np.array([15.0, 16.0]))

    def test_read_particles_from_case_uses_last_h5_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            data_dir = case_dir / "3D"
            data_dir.mkdir()

            write_synthetic_openpmd_file(data_dir / "openpmd_000100.h5", step=100)
            write_synthetic_openpmd_file(data_dir / "openpmd_005000.h5", step=5000)

            self.assertEqual(last_openpmd_h5_file(case_dir).name, "openpmd_005000.h5")

            particles = read_particles_from_case(case_dir, species="beam")
            self.assertEqual(particles.step, 5000)


if __name__ == "__main__":
    unittest.main()
