import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.publication_fig9 import (
    PUBLICATION_FIG9_SCHEMA,
    deterministic_background_indices,
    write_publication_fig9_snapshot,
)


def write_field_file(path: Path, *, step: int = 1400, time_s: float = 2.0e-13) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 10, 12)  # z, y, x
    z_index, y_index, x_index = np.indices(shape)

    ez_values = (
        np.sin(0.5 * z_index)
        * np.exp(-((x_index - 5.5) ** 2 + (y_index - 4.5) ** 2) / 20.0)
        * 2.0e12
    )
    bx_values = -(y_index - 4.5) * 2.0e3
    by_values = (x_index - 5.5) * 2.0e3

    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = time_s
        iteration.attrs["timeUnitSI"] = 1.0
        fields = iteration.create_group("fields")

        for record_name, components in (
            ("E", {"z": ez_values}),
            ("B", {"x": bx_values, "y": by_values}),
        ):
            record = fields.create_group(record_name)
            record.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
            record.attrs["gridSpacing"] = np.asarray([1.0, 1.0, 1.0])
            record.attrs["gridGlobalOffset"] = np.asarray([8.0, -5.0, -6.0])
            record.attrs["gridUnitSI"] = 1.0e-6
            record.attrs["geometry"] = "cartesian"
            record.attrs["dataOrder"] = "C"

            for component_name, values in components.items():
                component = record.create_dataset(component_name, data=values)
                component.attrs["position"] = np.asarray([0.5, 0.5, 0.5])
                component.attrs["unitSI"] = 1.0


def write_particle_file(
    path: Path,
    *,
    step: int = 1400,
    time_s: float = 2.0e-13,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.arange(1000, 1060, dtype=np.uint64)
    n = ids.size

    x_um = np.linspace(-4.5, 4.5, n)
    y_um = 2.5 * np.sin(np.linspace(0.0, 3.0 * np.pi, n))
    z_um = np.linspace(10.0, 14.5, n)
    pz = np.linspace(0.2e-21, 3.0e-21, n)

    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = time_s
        iteration.attrs["timeUnitSI"] = 1.0
        electrons = iteration.create_group("particles").create_group("electrons")
        electrons.create_dataset("id", data=ids)

        weighting = electrons.create_dataset("weighting", data=np.linspace(1.0, 2.0, n))
        weighting.attrs["unitSI"] = 1.0

        position = electrons.create_group("position")
        for axis, values_um in (
            ("x", x_um),
            ("y", y_um),
            ("z", z_um),
        ):
            component = position.create_dataset(axis, data=values_um)
            component.attrs["unitSI"] = 1.0e-6

        offsets = electrons.create_group("positionOffset")
        for axis in ("x", "y", "z"):
            component = offsets.create_group(axis)
            component.attrs["value"] = 0.0
            component.attrs["unitSI"] = 1.0

        momentum = electrons.create_group("momentum")
        for axis, values in (
            ("x", np.zeros(n)),
            ("y", np.zeros(n)),
            ("z", pz),
        ):
            component = momentum.create_dataset(axis, data=values)
            component.attrs["unitSI"] = 1.0

    return ids


class PublicationFig9Tests(unittest.TestCase):
    def test_deterministic_background_sample_is_repeatable_and_bounded(self):
        ids = np.arange(1, 101, dtype=np.uint64)
        mask = np.ones(ids.size, dtype=bool)
        first = deterministic_background_indices(mask, ids, max_points=17)
        second = deterministic_background_indices(mask, ids, max_points=17)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 17)
        self.assertEqual(np.unique(first).size, first.size)
        self.assertTrue(np.all(mask[first]))

    def test_writes_three_panel_snapshot_and_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_h5 = root / "3D" / "openpmd_001400.h5"
            field_h5 = root / "fields3D" / "openpmd_001400.h5"
            tracked_npy = root / "selection" / "final_bunch_ids.npy"
            output = root / "post" / "fig9_smoke"

            ids = write_particle_file(particle_h5)
            write_field_file(field_h5)
            tracked_npy.parent.mkdir(parents=True, exist_ok=True)
            tracked_ids = ids[[5, 9, 15, 22, 31, 40, 48, 55]]
            np.save(tracked_npy, tracked_ids, allow_pickle=False)

            result = write_publication_fig9_snapshot(
                particle_h5=particle_h5,
                field_h5=field_h5,
                tracked_ids_npy=tracked_npy,
                output_dir=output,
                step=1400,
                case_label="synthetic Fig. 9 smoke",
                background_slab_half_width_m=2.0e-6,
                transverse_half_width_m=6.0e-6,
                xi_padding_back_m=2.0e-6,
                xi_padding_front_m=2.0e-6,
                max_background_points_per_panel=11,
                quiver_max_arrows_per_axis=5,
                dpi=80,
            )

            png = output / "fig9_like_step001400.png"
            manifest_path = output / "manifest.json"
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 0)
            self.assertTrue(manifest_path.is_file())

            self.assertEqual(result.step, 1400)
            self.assertEqual(result.n_electrons_total, 60)
            self.assertEqual(result.n_tracked_requested, tracked_ids.size)
            self.assertEqual(result.n_tracked_present, tracked_ids.size)
            self.assertEqual(result.n_tracked_missing, 0)
            self.assertLessEqual(result.n_background_xy_plotted, 11)
            self.assertLessEqual(result.n_background_xz_plotted, 11)
            self.assertLessEqual(result.n_background_yz_plotted, 11)
            self.assertGreater(result.n_tracked_xy_plotted, 0)
            self.assertGreater(result.n_tracked_xz_plotted, 0)
            self.assertGreater(result.n_tracked_yz_plotted, 0)
            self.assertGreater(result.ez_color_limit_TV_m, 0.0)
            self.assertGreater(result.bperp_color_limit_kT, 0.0)
            self.assertGreater(result.energy_color_max_MeV, 0.0)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], PUBLICATION_FIG9_SCHEMA)
            self.assertEqual(
                manifest["result"]["tracked_particle_mode"],
                "all present fixed final-bunch IDs projected into each panel; no thinning",
            )
            self.assertEqual(
                len(manifest["tracked_particle_ids_plotted"]["xy"]),
                result.n_tracked_xy_plotted,
            )
            self.assertEqual(
                len(manifest["tracked_particle_ids_plotted"]["xz"]),
                result.n_tracked_xz_plotted,
            )
            self.assertEqual(
                len(manifest["tracked_particle_ids_plotted"]["yz"]),
                result.n_tracked_yz_plotted,
            )
            self.assertLessEqual(
                len(manifest["background_particle_ids_plotted"]["xy"]),
                11,
            )

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_h5 = root / "particle.h5"
            field_h5 = root / "field.h5"
            tracked_npy = root / "ids.npy"
            output = root / "existing"

            ids = write_particle_file(particle_h5)
            write_field_file(field_h5)
            np.save(tracked_npy, ids[:2], allow_pickle=False)
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_publication_fig9_snapshot(
                    particle_h5=particle_h5,
                    field_h5=field_h5,
                    tracked_ids_npy=tracked_npy,
                    output_dir=output,
                    step=1400,
                    case_label="synthetic",
                )

    def test_rejects_particle_field_time_mismatch_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_h5 = root / "particle.h5"
            field_h5 = root / "field.h5"
            tracked_npy = root / "ids.npy"
            output = root / "output"

            ids = write_particle_file(particle_h5, time_s=2.0e-13)
            write_field_file(field_h5, time_s=2.1e-13)
            np.save(tracked_npy, ids[:3], allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "times do not match"):
                write_publication_fig9_snapshot(
                    particle_h5=particle_h5,
                    field_h5=field_h5,
                    tracked_ids_npy=tracked_npy,
                    output_dir=output,
                    step=1400,
                    case_label="synthetic",
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
