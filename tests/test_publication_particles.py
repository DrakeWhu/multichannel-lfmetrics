import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.beam_metrics import kinetic_energy_MeV
from lfmetrics.publication_particles import (
    read_publication_particle_frame,
    read_tracked_particle_ids,
    select_publication_particle_plane,
)


def write_particle_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group("300")
        iteration.attrs["time"] = 12.5
        iteration.attrs["timeUnitSI"] = 1.0e-15

        electrons = iteration.create_group("particles").create_group("electrons")
        electrons.create_dataset(
            "id",
            data=np.asarray([10, 20, 30, 40], dtype=np.uint64),
        )
        weighting = electrons.create_dataset(
            "weighting",
            data=np.asarray([1.0, 2.0, 3.0, 4.0]),
        )
        weighting.attrs["unitSI"] = 1.0

        position = electrons.create_group("position")
        values = {
            "x": np.asarray([-2.0, -1.0, 1.0, 2.0]),
            "y": np.asarray([0.0, 0.1, 2.0, 0.0]),
            "z": np.asarray([10.0, 11.0, 12.0, 13.0]),
        }
        for axis, axis_values in values.items():
            component = position.create_dataset(axis, data=axis_values)
            component.attrs["unitSI"] = 1.0e-6

        offsets = electrons.create_group("positionOffset")
        for axis in ("x", "y", "z"):
            component = offsets.create_group(axis)
            component.attrs["value"] = 0.5
            component.attrs["unitSI"] = 1.0e-6

        momentum = electrons.create_group("momentum")
        momentum_values = {
            "x": np.asarray([0.0, 1.0e-22, 0.0, -1.0e-22]),
            "y": np.asarray([0.0, 0.0, 1.0e-22, 0.0]),
            "z": np.asarray([1.0e-22, 2.0e-22, 3.0e-22, 4.0e-22]),
        }
        for axis, axis_values in momentum_values.items():
            component = momentum.create_dataset(axis, data=axis_values)
            component.attrs["unitSI"] = 1.0


class PublicationParticlesTests(unittest.TestCase):
    def test_reads_frame_with_ids_time_offsets_momenta_and_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_particle_frame(path)

            frame = read_publication_particle_frame(
                path,
                species="electrons",
                iteration=300,
            )

        self.assertEqual(frame.step, 300)
        self.assertEqual(frame.species, "electrons")
        self.assertEqual(frame.size, 4)
        self.assertAlmostEqual(frame.time_s, 12.5e-15)
        np.testing.assert_array_equal(
            frame.ids,
            np.asarray([10, 20, 30, 40], dtype=np.uint64),
        )
        np.testing.assert_allclose(
            frame.particles.x_m,
            np.asarray([-1.5, -0.5, 1.5, 2.5]) * 1.0e-6,
        )
        np.testing.assert_allclose(
            frame.particles.y_m,
            np.asarray([0.5, 0.6, 2.5, 0.5]) * 1.0e-6,
        )
        np.testing.assert_allclose(
            frame.particles.z_m,
            np.asarray([10.5, 11.5, 12.5, 13.5]) * 1.0e-6,
        )
        np.testing.assert_allclose(
            frame.energy_MeV,
            kinetic_energy_MeV(
                frame.particles.px_si,
                frame.particles.py_si,
                frame.particles.pz_si,
            ),
        )

    def test_xz_selection_partitions_background_and_tracked_bunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_particle_frame(path)
            frame = read_publication_particle_frame(path, iteration=300)

        selection = select_publication_particle_plane(
            frame,
            np.asarray([20, 40, 50], dtype=np.uint64),
            plane="xz",
            coordinate_m=0.5e-6,
            slab_half_width_m=0.2e-6,
        )

        self.assertEqual(selection.horizontal_axis, "z")
        self.assertEqual(selection.vertical_axis, "x")
        self.assertEqual(selection.normal_axis, "y")
        np.testing.assert_array_equal(
            selection.in_slab_mask,
            np.asarray([True, True, False, True]),
        )
        np.testing.assert_array_equal(
            selection.background_mask,
            np.asarray([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            selection.tracked_mask,
            np.asarray([False, True, False, True]),
        )
        np.testing.assert_array_equal(
            selection.missing_tracked_ids,
            np.asarray([50], dtype=np.uint64),
        )
        self.assertEqual(selection.n_background, 1)
        self.assertEqual(selection.n_tracked, 2)

    def test_plane_axis_conventions_match_publication_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_particle_frame(path)
            frame = read_publication_particle_frame(path, iteration=300)

        tracked = np.asarray([10], dtype=np.uint64)
        expected = {
            "xy": ("x", "y", "z"),
            "xz": ("z", "x", "y"),
            "yz": ("z", "y", "x"),
        }
        for plane, axes in expected.items():
            with self.subTest(plane=plane):
                selection = select_publication_particle_plane(
                    frame,
                    tracked,
                    plane=plane,
                    coordinate_m=0.0,
                    slab_half_width_m=100.0e-6,
                )
                self.assertEqual(
                    (
                        selection.horizontal_axis,
                        selection.vertical_axis,
                        selection.normal_axis,
                    ),
                    axes,
                )

    def test_loads_fixed_tracked_ids_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_bunch_ids.npy"
            np.save(
                path,
                np.asarray([40, 10, 20], dtype=np.uint64),
                allow_pickle=False,
            )
            ids = read_tracked_particle_ids(path)

        np.testing.assert_array_equal(
            ids,
            np.asarray([40, 10, 20], dtype=np.uint64),
        )

    def test_duplicate_tracked_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_particle_frame(path)
            frame = read_publication_particle_frame(path, iteration=300)

        with self.assertRaisesRegex(ValueError, "tracked_ids must be unique"):
            select_publication_particle_plane(
                frame,
                np.asarray([10, 10], dtype=np.uint64),
                plane="xy",
                coordinate_m=0.0,
                slab_half_width_m=1.0e-6,
            )

    def test_negative_slab_half_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_particle_frame(path)
            frame = read_publication_particle_frame(path, iteration=300)

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            select_publication_particle_plane(
                frame,
                np.asarray([10], dtype=np.uint64),
                plane="yz",
                coordinate_m=0.0,
                slab_half_width_m=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
