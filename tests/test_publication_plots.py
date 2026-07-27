import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.publication_plots import write_rho_bunch_smoke_series


def write_field_frame(path: Path, *, time: float = 2.5) -> None:
    values = np.zeros((4, 4, 4), dtype=float)
    values[:, 1, :] = np.arange(16, dtype=float).reshape(4, 4) - 7.5
    values[:, 2, :] = values[:, 1, :] * 0.5

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group("300")
        iteration.attrs["time"] = time
        iteration.attrs["timeUnitSI"] = 1.0e-15
        fields = iteration.create_group("fields")
        rho = fields.create_dataset("rho", data=values)
        rho.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
        rho.attrs["gridSpacing"] = np.asarray([1.0, 2.0, 4.0])
        rho.attrs["gridGlobalOffset"] = np.asarray([10.0, -4.0, -8.0])
        rho.attrs["gridUnitSI"] = 1.0e-6
        rho.attrs["geometry"] = "cartesian"
        rho.attrs["dataOrder"] = "C"
        rho.attrs["unitSI"] = 3.0
        rho.attrs["position"] = np.asarray([0.5, 0.5, 0.5])


def write_trajectory(path: Path, *, time_s: float = 2.5e-15) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = np.ones((1, 3), dtype=bool)
    x_m = np.asarray([[0.0, 0.0, 20.0e-6]])
    y_m = np.asarray([[0.5e-6, 2.0e-6, 0.0]])
    z_m = np.asarray([[11.0e-6, 12.0e-6, 12.0e-6]])
    zeros = np.zeros((1, 3), dtype=float)
    np.savez_compressed(
        path,
        id=np.asarray([10, 20, 30], dtype=np.uint64),
        step=np.asarray([300], dtype=np.int64),
        time_s=np.asarray([time_s], dtype=float),
        present=present,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        px_si=zeros,
        py_si=zeros,
        pz_si=zeros,
        weighting=np.asarray([[1.0, 2.0, 3.0]]),
        energy_MeV=np.asarray([[1.0, 2.0, 3.0]]),
    )


class PublicationPlotsTests(unittest.TestCase):
    def test_writes_rho_smoke_with_exact_slab_ids_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            trajectory = case / "post" / "bunch_trajectories.npz"
            output = case / "post" / "publication_fields_deadbeef_smoke"
            write_field_frame(case / "fields3D" / "openpmd_000300.h5")
            write_trajectory(trajectory)

            results = write_rho_bunch_smoke_series(
                case_dir=case,
                trajectory_npz=trajectory,
                output_dir=output,
                steps=(300,),
                case_label="synthetic smoke",
                plane="xz",
                coordinate_m=0.0,
                slab_half_width_m=1.0e-6,
                color_percentile=90.0,
                dpi=80,
            )

            self.assertEqual(len(results), 1)
            frame = results[0]
            self.assertEqual(frame.n_present, 3)
            self.assertEqual(frame.n_in_slab, 2)
            self.assertEqual(frame.n_plotted, 1)
            self.assertEqual(frame.plotted_particle_ids, (10,))
            self.assertTrue(frame.no_particle_coarsening)
            self.assertEqual(frame.normal_axis, "y")
            self.assertEqual(frame.plane, "xz")
            self.assertGreater(frame.color_limit_C_m3, 0.0)

            png = output / "rho_xz_step000300.png"
            manifest_path = output / "manifest.json"
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["schema"],
                "multichannel_publication_rho_smoke_v1",
            )
            self.assertFalse(manifest["particle_coarsening"])
            self.assertEqual(
                manifest["frames"][0]["plotted_particle_ids"],
                [10],
            )

    def test_refuses_to_overwrite_complete_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            trajectory = case / "trajectory.npz"
            output = case / "existing"
            write_field_frame(case / "fields3D" / "openpmd_000300.h5")
            write_trajectory(trajectory)
            output.mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_rho_bunch_smoke_series(
                    case_dir=case,
                    trajectory_npz=trajectory,
                    output_dir=output,
                    steps=(300,),
                    case_label="synthetic smoke",
                    slab_half_width_m=1.0e-6,
                )

    def test_rejects_field_particle_time_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            trajectory = case / "trajectory.npz"
            output = case / "smoke"
            write_field_frame(case / "fields3D" / "openpmd_000300.h5")
            write_trajectory(trajectory, time_s=3.0e-15)

            with self.assertRaisesRegex(ValueError, "times do not match"):
                write_rho_bunch_smoke_series(
                    case_dir=case,
                    trajectory_npz=trajectory,
                    output_dir=output,
                    steps=(300,),
                    case_label="synthetic smoke",
                    slab_half_width_m=1.0e-6,
                )

    def test_rejects_non_longitudinal_plane_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            trajectory = case / "trajectory.npz"
            output = case / "smoke"
            write_field_frame(case / "fields3D" / "openpmd_000300.h5")
            write_trajectory(trajectory)

            with self.assertRaisesRegex(ValueError, "must be 'xz' or 'yz'"):
                write_rho_bunch_smoke_series(
                    case_dir=case,
                    trajectory_npz=trajectory,
                    output_dir=output,
                    steps=(300,),
                    case_label="synthetic smoke",
                    plane="xy",
                    slab_half_width_m=1.0e-6,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
