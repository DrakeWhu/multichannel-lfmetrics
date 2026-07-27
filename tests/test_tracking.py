import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.constants import C_LIGHT, M_E_KG
from lfmetrics.tracking import (
    backtrack_particle_ids,
    select_final_bunch,
    write_final_bunch_selection,
    write_tracking_outputs,
)


def write_frame(path: Path, step: int, ids, gamma_beta_z, z_um):
    ids = np.asarray(ids, dtype=np.uint64)
    gamma_beta_z = np.asarray(gamma_beta_z, dtype=float)
    count = ids.size

    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        iteration = data.create_group(str(step))
        iteration.attrs["time"] = float(step)
        iteration.attrs["timeUnitSI"] = 1.0e-15
        particles = iteration.create_group("particles")
        electrons = particles.create_group("electrons")
        electrons.create_dataset("id", data=ids)

        position = electrons.create_group("position")
        for name, values in {
            "x": np.linspace(0.0, 1.0, count),
            "y": np.linspace(1.0, 2.0, count),
            "z": np.asarray(z_um, dtype=float),
        }.items():
            dataset = position.create_dataset(name, data=values)
            dataset.attrs["unitSI"] = 1.0e-6

        momentum = electrons.create_group("momentum")
        for name, values in {
            "x": np.zeros(count),
            "y": np.zeros(count),
            "z": gamma_beta_z * M_E_KG * C_LIGHT,
        }.items():
            dataset = momentum.create_dataset(name, data=values)
            dataset.attrs["unitSI"] = 1.0

        weighting = electrons.create_dataset(
            "weighting",
            data=np.arange(1, count + 1, dtype=float),
        )
        weighting.attrs["unitSI"] = 1.0


class TrackingTests(unittest.TestCase):
    def test_select_final_bunch_and_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "openpmd_000200.h5"
            write_frame(
                h5_path,
                200,
                ids=[30, 10, 20],
                gamma_beta_z=[100.0, 10.0, -100.0],
                z_um=[30.0, 10.0, 20.0],
            )

            selection = select_final_bunch(
                h5_path,
                energy_threshold_MeV=20.0,
                forward_only=True,
            )

            self.assertEqual(selection.ids.tolist(), [30])
            self.assertGreater(float(selection.energy_MeV[0]), 20.0)
            self.assertGreater(float(selection.pz_si[0]), 0.0)

            output = root / "selection"
            summary = write_final_bunch_selection(selection, output)
            self.assertEqual(summary["n_macroparticles"], 1)
            self.assertTrue((output / "final_bunch_ids.npy").is_file())
            self.assertTrue((output / "final_bunch_state.npz").is_file())

    def test_backtrack_fixed_ids_through_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostics = root / "case" / "3D"
            diagnostics.mkdir(parents=True)

            write_frame(
                diagnostics / "openpmd_000100.h5",
                100,
                ids=[10, 20],
                gamma_beta_z=[2.0, 3.0],
                z_um=[1.0, 2.0],
            )
            write_frame(
                diagnostics / "openpmd_000200.h5",
                200,
                ids=[10, 20, 30],
                gamma_beta_z=[4.0, 5.0, 100.0],
                z_um=[3.0, 4.0, 5.0],
            )

            result = backtrack_particle_ids(
                root / "case",
                np.asarray([30], dtype=np.uint64),
            )

            self.assertEqual(result.steps.tolist(), [100, 200])
            self.assertEqual(result.present[:, 0].tolist(), [False, True])
            self.assertTrue(np.isnan(result.energy_MeV[0, 0]))
            self.assertGreater(float(result.energy_MeV[1, 0]), 20.0)

            output = root / "tracking"
            summary = write_tracking_outputs(result, output)
            self.assertEqual(summary["first_seen_histogram"], {"200": 1})

            with (output / "frame_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["n_present"] for row in rows], ["0", "1"])

            with (output / "tracking_summary.json").open() as handle:
                stored = json.load(handle)
            self.assertTrue(stored["first_seen_is_not_capture_time"])


if __name__ == "__main__":
    unittest.main()
